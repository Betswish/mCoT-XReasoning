import argparse
import logging
import re
from typing import Any, List

import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

from math_verify import LatexExtractionConfig, parse, verify
from langdetect import detect_langs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "Pensez toujours en français.\n\n"
    "Résolvez le problème de mathématiques suivant étape par étape."
)
    # "Décrivez chaque étape de raisonnement sous la forme <think>processus de raisonnement ici</think>.\n"
    # "Enfin, fournissez votre réponse finale en l'entourant avec \\boxed{}.\n\n"
    # "Problème : {text}\n\n"
    # "La réponse doit être au format « La réponse est : \\boxed{réponse_finale} » avec le raisonnement dans les balises <think></think> et la réponse dans \\boxed{}."


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments with vllm_server_host.
    """
    parser = argparse.ArgumentParser(
        description="GRPO Training Script with vLLM server host option."
    )
    parser.add_argument(
        "--vllm_server_host",
        type=str,
        default="0.0.0.0",
        help="Hostname or IP address of the vLLM server (default: 0.0.0.0)."
    )
    # Add more arguments as needed.
    return parser.parse_args()


def make_conversation(example: dict) -> dict:
    """Format a training example into a conversation prompt.

    Args:
        example (dict): Example with a 'question' field.

    Returns:
        dict: Dictionary with a 'prompt' key for the conversation.
    """
    return {
        "prompt": [
            # {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SYSTEM_PROMPT + example["question"]},
        ],
    }


def make_conversation_test(example: dict) -> dict:
    """Format a test example into a conversation prompt.

    Args:
        example (dict): Example with a 'problem' field.

    Returns:
        dict: Dictionary with a 'prompt' key for the conversation.
    """
    return {
        "prompt": [
            # {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SYSTEM_PROMPT + example["problem"]},
        ],
    }


def format_reward(completions: List[Any], **kwargs) -> List[float]:
    """Reward function that checks if the completion matches the required French math prompt format.

    The expected format is:
        - At least one <think>...</think> block (for reasoning)
        - At least one '\\boxed{...}' (for the answer)

    Args:
        completions (List[Any]): List of completions, each a list of dicts with a "content" key.

    Returns:
        List[float]: 1.0 if the completion matches the required format, else 0.0.

    Example:
        >>> completions = [[{"content": "<think>raisonnement</think>\nLa réponse est : \\boxed{42}"}]]
        >>> format_reward(completions)
        [1.0]
    """
    think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    # Regex for '\\boxed{...}'
    answer_pattern = re.compile(r"\\boxed\{([^}]+?)\}",)

    rewards: List[float] = []
    for completion in completions:
        if (
            not completion
            or not isinstance(completion[0], dict)
            or "content" not in completion[0]
        ):
            rewards.append(0.0)
            continue
        content = completion[0]["content"]
        # has_think = bool(think_pattern.search(content))
        has_think = "</think>" in content 
        has_boxed = "\\boxed{" in content
        rewards.append(0.2 if has_think and has_boxed else 0.0)
    return rewards

def detect_language(text):
    try:
        text = text.strip()
        res_detect = dict()
        detections = detect_langs(text)
        for i in detections:
            current_lang = i.lang
            if current_lang == 'zh-cn': current_lang = 'zh'
            res_detect[current_lang] = i.prob
        top1 = detections[0].lang
        if top1 == 'zh-cn': top1 = 'zh'
        return res_detect, top1
    except Exception as e:
        return None, None
    
def language_reward(completions: List[Any], **kwargs) -> List[float]:
    """Reward function that checks if the completion is in French.

    Args:
        completions (List[Any]): List of completions, each a list of dicts with a "content" key.

    Returns:
        List[float]: 1.0 if the completion is in French, else 0.0.
    """
    rewards: List[float] = []
    for completion in completions:
        if (
            not completion
            or not isinstance(completion[0], dict)
            or "content" not in completion[0]
        ):
            rewards.append(0.0)
            continue
        content = completion[0]["content"]
        try:
            lang,_ = detect_language(content)
            rewards.append(lang["fr"]*0.2 if "fr" in lang else 0.0)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            logger.warning(f"lang detect failed:{lang} | Content: {content}")
            rewards.append(0.0)
    return rewards

def accuracy_reward(completions: List[Any], **kwargs) -> List[float]:
    """Reward function that checks if the completion matches the ground truth solution.

    Args:
        completions (List[Any]): List of completions, each a list of dicts with a "content" key.
        **kwargs: Must contain 'solution' (List[str]).

    Returns:
        List[float]: 1.0 if the answer matches, else 0.0.
    """
    solutions = kwargs.get("solution", [])
    completion_contents = [
        completion[0]["content"] if completion and isinstance(completion[0], dict) and "content" in completion[0] else ""
        for completion in completions
    ]
    rewards: List[float] = []
    for content, solution in zip(completion_contents, solutions):
        try:
            gold_parsed = [str(solution)]
        except Exception as e:
            logger.warning(f"conversion failed: {e}")
            logger.warning(f"solution: {solution} | solution type: {type(solution)}")
            gold_parsed = [solution]
        try:
            answer_parsed = parse(
                content,
                extraction_mode="any_match",
                extraction_config=[LatexExtractionConfig()],
            )
            # get answer_pattern = re.compile(r"\\boxed\{([^}]+?)\}",)
            answer_pattern = re.compile(r"\\boxed\{([^}]+?)\}",)
            answer_parsed.extend(answer_pattern.findall(content))
            rewards.append(float(verify(answer_parsed, gold_parsed)))
            # if rewards[-1] == 0.0:
            #     logger.warning(f"Verification failed: {answer_parsed} | {gold_parsed} | {content[:-23]}")
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            rewards.append(0.0)
    return rewards


def main() -> None:
    """Main function to run GRPO training."""
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model


    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # model_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # IGNORE
    # model_id = "shanchen/ds-limo-ja-500"
    model_id = "shanchen/ds-limo-fr-250"
    # model_id = "shanchen/ds-limo-ja-full"  # You may want to use a French-compatible model
    id = model_id.split("/")[1]  # IGNORE
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    ###
        #     quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
    ###

    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Load datasets - using French MATH dataset
    train_dataset_name: str = "bezir/MATH-500-multilingual"
    
    # Load French subset of MATH-500-multilingual
    train_dataset: Dataset = load_dataset("bezir/MATH-500-multilingual", "French", split="test")
    logger.info(f"using French MATH-500 dataset as train dataset")
    train_dataset = train_dataset.map(make_conversation_test)
    train_dataset = train_dataset.remove_columns("solution")
    train_dataset = train_dataset.add_column("solution", train_dataset["answer"])
    
    # For test dataset, you might want to use a different French math dataset
    # or create a validation split from the same dataset
    test_dataset: Dataset = load_dataset(
        "shanchen/aime_2025_multilingual", "default", split="fr"
    )
    test_dataset = test_dataset.map(make_conversation_test)
    
    # Retain only the 'prompt' and 'solution' columns
    train_dataset = train_dataset.select_columns(["prompt", "solution"])
    test_dataset = test_dataset.select_columns(["prompt", "solution"])

    logger.info(f"First training prompt: {train_dataset[0]['prompt']}")
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # train_data_size*num_generations/gradient_accumulation_steps/per_device_train_batch_size/device_num
    # Training configuration
    training_args = GRPOConfig(
        output_dir=f"/temp_work/ch225816/trl_grpo/kl_{id}_fr_dapo_format_lang_0.28_0.2f_0.2lang_64_{train_dataset_name.replace('/', '_')}_fr_8lora_7e-6",
        learning_rate=7e-6,
        remove_unused_columns=False,  # to access the solution column in accuracy_reward
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        num_train_epochs=30,
        bf16=True,
        epsilon_high=0.28,
        loss_type="dr_grpo",
        max_completion_length=5555,
        num_generations=24,
        max_prompt_length=512,
        use_vllm=True,
        vllm_mode="server",
        report_to=["wandb"],
        logging_steps=5,
        save_strategy="steps",
        save_steps=95,
        eval_strategy="steps",
        eval_steps=10,
        vllm_server_timeout=360,
        # beta=0
    )

    # Set vLLM server host from command-line argument
    setattr(training_args, "vllm_server_host", args.vllm_server_host)
    logger.info(f"Using vLLM server host: {training_args.vllm_server_host}")

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model, #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        reward_funcs=[format_reward, accuracy_reward, language_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Start training
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save the model
    logger.info("Saving trained model...")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()