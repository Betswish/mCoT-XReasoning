import re
import json
from tqdm import tqdm
import collections
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def eval_regex(response):
    # Comprehensive pattern list, now capturing any content inside the box
    patterns = [
    # \box{content}
    # r"\\box\{([^}]+?)\}",

    # \box content \box   — require at least one non‐whitespace char
    # r"\\box\s+(\S.+?\S)\s+\\box",

    # \boxed{content}
    r"\\boxed\{([^}]+?)\}",

    # Answer: content  (anchored at line start)
    # r"(?m)^Answer:\s*(\S.+?)\s*$",

    # “xxx is correct”  — non‐greedy and require at least one word char
    # r"(\w.+?)\s+is correct",

    # Final Answer: \box content
    # r"(?m)^Final Answer:\s*\\box\s+(\S.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.findall(pattern, response)
        if match:
            if 'A' in match and 'B' in match and 'C' in match and 'D' in match:
                for i in ['A', 'B', 'C', 'D']:
                    match.remove(i)
            return match
    return None

def eval(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name):
    # Load reward model if scoring is enabled
    rm = None
    rm_tokenizer = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Use CUDA if available
    if add_scores:
        print(f"Loading reward model: {reward_model_name} on device: {device}")
        rm = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16, # Use bfloat16 only on CUDA
            device_map=device, # device_map only works well on CUDA
            attn_implementation="flash_attention_2", # flash attention only on CUDA
            num_labels=1,
        )
        # If not using device_map (i.e., on CPU), explicitly move model to device
        if device != "cuda:0":
            rm.to(device)
        rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        print("Reward model loaded.")

    accuracy = 0
    accuracy_hack = 0
    fpath = f"{output_dir}/{mname.split('/')[1]}_{dataset}_{lang}_think_{lang_think}.json"
    with open(fpath, 'r') as f:
        instances = json.load(f)
    f.close()

    # Track whether we need to update the file
    file_updated = False

    for ins in tqdm(instances):
        response = ins['response']
        response_hack = ins['response_hack']
        gold_answer = str(ins['answer']) if 'gpqa' not in dataset else ins['answer'][-2:-1]

        # print(ins['index'], '/', len(instances)-1, gold_answer, ':')
        ans = eval_regex(response)
        ans_hack = eval_regex(response_hack)
        # ans = eval_regex(response)
        # ans_hack = eval_regex(response_hack)
        # print(ans, ans_hack)

        # Add boolean flags indicating if an answer was extracted
        has_extracted_answer = bool(ans) # True if ans is not None and not empty
        has_extracted_answer_hack = bool(ans_hack) # True if ans_hack is not None and not empty
        ins['has_extracted_answer'] = has_extracted_answer
        ins['has_extracted_answer_hack'] = has_extracted_answer_hack
        # Ensure file is marked for update if we add these fields
        if not file_updated: # Only set if not already set by score/store flags
             file_updated = True # Mark file for update as we added boolean flags

        # Store extracted answers in the JSON if requested
        if store_answers:
            ins['extracted_answer'] = ans
            # Removed duplicate line here
            ins['extracted_answer_hack'] = ans_hack
            file_updated = True

        # Calculate scores if enabled
        if add_scores and rm and rm_tokenizer:
            try:
                # Format for reward model
                prompt = ins.get('prompt', '') # Use get with default to avoid KeyError if field missing
                response = ins.get('response', '')
                prompt_hack = ins.get('prompt_hack', '')
                response_hack = ins.get('response_hack', '')

                conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
                conv_hack = [{"role": "user", "content": prompt_hack}, {"role": "assistant", "content": response_hack}]

                # Tokenize
                conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
                conv_hack_tokenized = rm_tokenizer.apply_chat_template(conv_hack, tokenize=True, return_tensors="pt").to(device)

                # Get scores
                with torch.no_grad():
                    score = rm(conv_tokenized).logits[0][0].item()
                    score_hack = rm(conv_hack_tokenized).logits[0][0].item()

                ins['score'] = score
                ins['score_hack'] = score_hack
                file_updated = True
            except Exception as e:
                print(f"Error scoring instance {ins.get('id', 'N/A')}: {e}")
                ins['score'] = None # Indicate scoring error
                ins['score_hack'] = None

        if ans:
            accuracy += (gold_answer in ans)
        if ans_hack:
            accuracy_hack += (gold_answer in ans_hack)

    # *** Temporarily disabled the following function ***
    file_updated = False
    # Write back the updated JSON if any modifications were made (answers stored, scores added, or boolean flags added)
    if file_updated:
        with open(fpath, 'w') as f:
            json.dump(instances, f, indent=2)
        update_message = []
        if store_answers: # This flag implies extracted answers were stored
            update_message.append("extracted answers")
        if add_scores: # This flag implies scores were added
            update_message.append("reward scores")
        # Always mention boolean flags were added if file was updated,
        # as they are added regardless of store/score flags now.
        update_message.append("extraction status flags")
        # Remove potential duplicates and join
        print(f"Updated {fpath} with {', '.join(sorted(list(set(update_message))))}")   
    # *** Temporarily disabled the above function ***

    with open('eval.csv', 'a') as f:
        # f.write(f"{mname}\t{dataset}\t{lang}\t{lang_think}\t{accuracy}/{len(instances)}={round(100*accuracy/len(instances),2)}%, {accuracy_hack}/{len(instances)}={round(100*accuracy_hack/len(instances),2)}%\n")
        f.write(f"{mname}\t{dataset}\t{lang}\t{lang_think}\t{round(100*accuracy/len(instances),1)}%\t{round(100*accuracy_hack/len(instances),1)}%\n")
    f.close()
    # print(f"{mname} {dataset} {lang}: {accuracy}/{len(instances)}={round(100*accuracy/len(instances),2)}%, {accuracy_hack}/{len(instances)}={round(100*accuracy_hack/len(instances),2)}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--output_dir", type=str, default="outputs_0", help="Loading from which directory")
    parser.add_argument("--store", action="store_true", help="Store extracted answers in JSON files")
    parser.add_argument("--add_scores", action="store_true", help="Add reward model scores to JSON files")
    parser.add_argument("--reward_model_name", type=str, default="Skywork/Skywork-Reward-Gemma-2-27B-v0.2", help="Reward model name")

    args = parser.parse_args()

    output_dir        = args.output_dir
    store_answers     = args.store
    add_scores        = args.add_scores
    reward_model_name = args.reward_model_name

    mnames = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        'Skywork/Skywork-OR1-7B',
        'Skywork/Skywork-OR1-32B',
        ]

    langs = ['EN', 'FR', 'DE', 'ZH', 'JA', 'RU', 'ES', 'BN', 'TH', 'SW', 'TE']
    langs_think = ['EN', 'FR', 'DE', 'ZH', 'JA', 'RU', 'ES', 'BN', 'TH', 'SW', 'TE']


    datasets = [
        'aime_combined', 
        'gpqa_diamond_mc_multilingual', 
        'mgsm'
        ]

    for dataset in datasets:
        for mname in mnames:
            for lang_think in langs_think:
                for lang in langs:
                    try:
                        eval(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name)
                    except Exception as e:
                        print(f"Error processing {mname} {dataset} {lang} {lang_think}: {e}")
                        continue
