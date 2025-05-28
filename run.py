import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from datasets import load_dataset
import torch
import argparse
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from random import sample
import re
from dotenv import load_dotenv


from vllm import LLM as VLLM
from vllm import SamplingParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_prompt(tokenizer, instruction, content, hack=False):
    """
    Create a prompt for the model using the tokenizer's chat template.
    
    Args:
        tokenizer: The tokenizer to use for creating the prompt
        instruction: The system instruction
        content: The content/question to include in the prompt
        hack: Whether this is a hacked prompt (with prefix)
        
    Returns:
        The formatted prompt string
    """
    messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content}
            ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt

def load_dataset_data(dataset_name, question_field="problem", answer_field="answer", split="train", test_mode=False, max_test_examples=5):
    """
    Load data from a specified dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        question_field: Field name containing the question/problem
        answer_field: Field name containing the answer
        split: Dataset split to load
        test_mode: Whether to run in test mode (limit examples)
        max_test_examples: Maximum number of examples to process in test mode
        
    Returns:
        List of data items with standardized format
    """
    print(f"Loading dataset: {dataset_name}, using fields: {question_field}/{answer_field}")
    
    # Handle different dataset sources
    if dataset_name.startswith("juletxara/mgsm"):
        # For MGSM dataset, we need to use the language as a config and 'test' as the split
        # Extract language from split parameter (which should be lowercase)
        lang_config = split.lower()
        try:
            # Remove any config suffix if present
            base_dataset = dataset_name.split('@')[0]
            # Load the dataset with the language as the config and 'test' as the split
            # This is the correct way to load the MGSM dataset as provided by the user
            dataset = load_dataset(base_dataset, lang_config, split='test')
            # Convert to list to ensure we're working with dictionaries
            raw_data = list(dataset)
        except Exception as e:
            print(f"Error loading MGSM dataset with config {lang_config}: {e}")
            raise
    elif dataset_name.startswith("gpqa"):
        lang_split = split.lower()
        try:
            # For GPQA dataset, we should use the language code as the split name
            # similar to how AIME dataset is handled
            print(f"Loading GPQA dataset with language split: {lang_split}")
            dataset = load_dataset("gpqa_diamond_mc_multilingual", split=lang_split)
            raw_data = list(dataset)
        except Exception as e:
            print(f"Error loading GPQA dataset with split {lang_split}: {e}")
            raise
    elif dataset_name.startswith("aime_combined"):
        lang_split = split.lower()
        try:
            aime24 = load_dataset("aime_2024_multilingual", split=lang_split)
            aime25 = load_dataset("aime_2025_multilingual", split=lang_split)
            raw_data = list(aime24) + list(aime25)
        except Exception as e:
            print(f"Error loading AIME dataset with split {lang_split}: {e}")
            raise
    else:
        # Load a single dataset
        try:
            dataset = load_dataset(dataset_name, split='train')
            # Convert to list to ensure we're working with dictionaries
            raw_data = list(dataset)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    # Limit examples if in test mode
    if test_mode:
        print(f"TEST MODE: Limiting to {max_test_examples} examples")
        raw_data = raw_data[:max_test_examples]
    
    # Standardize the data format
    standardized_data = []
    for idx, item in enumerate(raw_data):
        # Create a standardized item with common metadata
        std_item = {
            "index": idx,
            "original_data": item,  # Keep the original data for reference
        }
        
        # Check if item is a dictionary (it should be)
        if not isinstance(item, dict):
            raise TypeError(f"Dataset item at index {idx} is not a dictionary. Got {type(item)} instead.")
        
        # Extract the question and answer using the specified field names
        if question_field in item:
            std_item["problem"] = item[question_field]
        else:
            raise ValueError(f"Question field '{question_field}' not found in dataset. Available fields: {list(item.keys())}")
            
        if answer_field in item:
            std_item["answer"] = item[answer_field]
        else:
            # For datasets without answers (test sets), use None
            std_item["answer"] = None
            
        # Add any additional metadata that might be useful
        for key in ["url", "year"]:
            if key in item:
                std_item[key] = item[key]
                
        standardized_data.append(std_item)
    
    print(f"Loaded {len(standardized_data)} items from {dataset_name}")
    return standardized_data


def run(args):
    """
    Main async function for running inference.
    
    Args:
        args: Command-line arguments
    """
    cache_dir = args.cache_dir if args.cache_dir else os.getenv("TMPDIR") 
    lang = args.lang
    save_dir = f'outputs_{args.seed}/'
    
    # Print test mode status
    if args.test_mode:
        print(f"Running in TEST MODE - will only process up to {args.max_test_examples} examples")
        # In test mode, use smaller max_tokens to speed up inference
        args.max_tokens = max(args.max_tokens, 1024)
        print(f"TEST MODE: Using reduced max_tokens={args.max_tokens}")

    # Load the instructions specifying the thinking and answering languages
    with open("instructions.json", encoding='utf-8') as f:
        instructions = json.load(f)
    f.close()

    # Load the prefix for <think> hackers
    with open("hack_prefix.json", encoding='utf-8') as f:
        hack_prefix = json.load(f)
    f.close()

    # Load dataset
    data = load_dataset_data(
        args.dataset, 
        question_field=args.question_field, 
        answer_field=args.answer_field,
        split=args.split,
        test_mode=args.test_mode,
        max_test_examples=args.max_test_examples
    )

    # Determine which model type we're using
    api_class = None # None when using vllm
    use_vllm = api_class is None
    
    # Initialize tokenizer and model based on the model type
    print(f"Using vllm for model: {args.mname}")
    tokenizer = AutoTokenizer.from_pretrained(args.mname, cache_dir=cache_dir, trust_remote_code=True)
    extra_kw = {"download_dir": cache_dir}
    print(f"Running on {torch.cuda.device_count()} GPUs")
    vmodel = VLLM(
            model=args.mname,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.60,
            dtype=torch.bfloat16,
            distributed_executor_backend="mp",
            trust_remote_code=True,
            max_num_seqs=100,
            max_model_len=args.max_tokens,
            seed=args.seed,
            disable_custom_all_reduce=True,
            **extra_kw
            )
    
    save_data = []
    
    # Prepare all prompts
    all_prompts = []
    all_prompt_types = []  # To track whether each prompt is normal or hack
    
    for index, ins in enumerate(data):
        # store common info
        meta_info = {"index": index, "answer": ins.get("answer")}
        
        # Add additional metadata if available
        for key in ["url", "year"]:
            if key in ins:
                meta_info[key] = ins[key]
                
        save_data.append(meta_info)

        # thinking language
        if args.lang_think == 'default':
            lang_think = args.lang
        else:
            lang_think = args.lang_think

        # make the two prompts, general and hacked
        content = instructions[lang_think][1].format(ins['problem'])

        # For API-based models, we use the raw content instead of tokenizer-processed prompts
        instruction = instructions[lang_think][0]
        if use_vllm:
            prompt = make_prompt(tokenizer, instruction, content)
        else:
            prompt = instruction + content

        # Manually add generation template with <think> tag
        if 'deepseek' in args.mname.lower() or 'skywork' in args.mname.lower() or 'ds-limo' in args.mname.lower():
            prompt += "<｜Assistant｜><think>\n"
        elif 'qwen3' in args.mname.lower():
            prompt += "<|im_start|>assistant\n<think>\n"
        elif 's1.1-limo' in args.mname.lower() or 'qwen2.5-7b' in args.mname.lower():
            # Overwrite the prompt
            prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"\
                    + content + "<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise ValueError(f"Unsupported model for vllm when manually adding <think> tag: {args.mname}")
        prompt_hack = prompt + f"{hack_prefix[lang_think]}"

        
        # Add prompts to the list
        all_prompts.append(prompt)
        all_prompt_types.append("normal")
        all_prompts.append(prompt_hack)
        all_prompt_types.append("hack")
    
    # Process prompts
    if use_vllm:
        # Generate responses with vllm
        if 'ds-limo' in args.mname.lower():
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=args.max_tokens,
            )
        elif 's1.1-limo' in args.mname.lower() or 'qwen2.5-7b' in args.mname.lower():
            stop_token_ids = tokenizer("<|im_end|>")["input_ids"]
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=args.max_tokens,
                stop_token_ids=stop_token_ids,
            )
        else:
            # Greedy decoding for Deepseek and Skywork models
            sampling_params = SamplingParams(
                temperature=1.0,
                top_k=1,
                max_tokens=args.max_tokens,
            )

        if args.seed == 0:
            # Forcing greedy decoding when seed is 0
            sampling_params = SamplingParams(
                temperature=1.0,
                top_k=1,
                max_tokens=args.max_tokens,
            )
        responses = vmodel.generate(all_prompts, sampling_params, use_tqdm=True)
    
    # Process responses
    for i, response in enumerate(responses):
        data_index = i // 2  # Each data item has 2 prompts (normal and hack)
        prompt_type = all_prompt_types[i]
        
        if prompt_type == "normal":
            field_prompt = 'prompt'
            field_response = 'response'
        else:  # hack
            field_prompt = 'prompt_hack'
            field_response = 'response_hack'
        
        save_data[data_index][field_prompt] = response.prompt
        save_data[data_index][field_response] = response.outputs[0].text
    
    # Save results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a more descriptive filename that includes the dataset
    dataset_name = args.dataset.split('/')[-1] if '/' in args.dataset else args.dataset
    test_suffix = "_test" if args.test_mode else ""
    output_filename = f"{save_dir}{args.mname.split('/')[-1]}_{dataset_name}_{args.lang}_think_{args.lang_think}{test_suffix}.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=True, sort_keys=False) # no indent for saving storage
        #json.dump(save_data, f, ensure_ascii=True, indent=4, sort_keys=False)
    
    print(f"Results saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--mname", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="LLM name")
    parser.add_argument("--lang", type=str, default="EN", help="Language")
    parser.add_argument("--lang_think", type=str, default="default", help="Language for thinking")
    parser.add_argument("--seed", type=int, default="2025", help="Seeds for generation")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to the cache directory")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="aime_combined", 
                        help="Dataset to use (aime_combined, aiw_hard_multilingual, juletxara/mgsm, etc.)")
    parser.add_argument("--question_field", type=str, default="problem", 
                        help="Field name containing the question/problem")
    parser.add_argument("--answer_field", type=str, default="answer", 
                        help="Field name containing the answer")
    parser.add_argument("--split", type=str, default="train", 
                        help="Dataset split to use")
    
    # Test mode parameters
    parser.add_argument("--test_mode", action="store_true",
                        help="Run in test mode with limited examples")
    parser.add_argument("--max_test_examples", type=int, default=5,
                        help="Maximum number of examples to process in test mode")

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
