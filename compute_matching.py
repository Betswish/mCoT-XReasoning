import re
import json
from tqdm import tqdm
import collections
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect_langs

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

langs = ["EN", "FR", "DE", "ZH", "JA", "RU", "ES", "SW", "BN", "TE", "TH"]
langs_think = ["EN", "FR", "DE", "ZH", "JA", "RU", "ES", "SW", "BN", "TE", "TH"]


datasets = [
    'aime_combined', 
    'gpqa_diamond_mc_multilingual', 
    'mgsm'
    ]


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

def compute_matching(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name):
    # Load reward model if scoring is enabled
    rm = None
    rm_tokenizer = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Use CUDA if available

    matching_rate_norm = 0
    matching_rate_hack = 0
    fpath = f"{output_dir}/{mname.split('/')[1]}_{dataset}_{lang}_think_{lang_think}.json"
    with open(fpath, 'r') as f:
        instances = json.load(f)
    f.close()

    for ins in tqdm(instances):
        response = ins['response']
        response_hack = ins['response_hack']
        gold_answer = str(ins['answer']) if 'gpqa' not in dataset else ins['answer'][-2:-1]

        lang_norm_list, lang_norm = detect_language(response.split('</think>')[0])
        lang_hack_list, lang_hack = detect_language(response_hack.split('</think>')[0])

        # Calculate scores if enabled
        matching_rate_norm += (lang_norm == lang_think.lower())
        matching_rate_hack += (lang_hack == lang_think.lower())

    with open('matching.csv', 'a') as f:
        f.write(f"{mname}\t{dataset}\t{lang}\t{lang_think}\t{round(100*matching_rate_norm/len(instances),2)}%\t{round(100*matching_rate_hack/len(instances),2)}%\n")
    f.close()

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


    for dataset in datasets:
        for mname in mnames:
            for lang in langs:
                for lang_think in langs_think:
                    try:
                        compute_matching(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name)
                    except Exception as e:
                        print(f"Error processing {mname} {dataset} {lang} {lang_think}: {e}")
                        continue
