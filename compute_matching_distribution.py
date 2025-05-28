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

full_langs = ['EN', 'FR', 'DE', 'ZH', 'JA', 'RU', 'ES', 'BN', 'TH', 'SW', 'TE']

langs = ['EN', 'FR', 'DE', 'ZH', 'JA', 'RU', 'ES', 'BN', 'TH', 'SW', 'TE']
langs_think = ['EN', 'FR', 'DE', 'ZH', 'JA', 'RU', 'ES', 'BN', 'TH', 'SW', 'TE']

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

def compute_matching_distribution(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name):
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

    matching_distribution_norm = {lang: 0 for lang in full_langs}
    matching_distribution_hack = {lang: 0 for lang in full_langs}

    matching_distribution_norm['others'] = 0
    matching_distribution_hack['others'] = 0

    for ins in tqdm(instances):
        response = ins['response']
        response_hack = ins['response_hack']
        gold_answer = str(ins['answer']) if 'gpqa' not in dataset else ins['answer'][-2:-1]

        lang_norm_list, lang_norm = detect_language(response.split('</think>')[0])
        lang_hack_list, lang_hack = detect_language(response_hack.split('</think>')[0])

        for k, v in lang_norm_list.items():
            if k.upper() not in matching_distribution_norm:
                matching_distribution_norm['others'] += v
            else:
                matching_distribution_norm[k.upper()] += v
        for k, v in lang_hack_list.items():
            if k.upper() not in matching_distribution_hack:
                matching_distribution_hack['others'] += v
            else:
                matching_distribution_hack[k.upper()] += v
        
        # Calculate scores if enabled
        matching_rate_norm += (lang_norm == lang_think.lower())
        matching_rate_hack += (lang_hack == lang_think.lower())

    # Normalize the distributions
    matching_distribution_norm = {k: round(100*v/len(instances),2) for k, v in matching_distribution_norm.items()}
    matching_distribution_hack = {k: round(100*v/len(instances),2) for k, v in matching_distribution_hack.items()}

    # Remove zero values
    matching_distribution_norm = {k: v for k, v in matching_distribution_norm.items() if v != 0}
    matching_distribution_hack = {k: v for k, v in matching_distribution_hack.items() if v != 0}

    # Sort the distributions
    matching_distribution_norm = dict(sorted(matching_distribution_norm.items(), key=lambda item: item[1], reverse=True))
    matching_distribution_hack = dict(sorted(matching_distribution_hack.items(), key=lambda item: item[1], reverse=True))

    with open('matching_distribution_norm.csv', 'a') as f:
        f.write(f"{lang} & {lang_think} & {matching_distribution_norm} \\\\ \n")

    with open('matching_distribution_hack.csv', 'a') as f:
        f.write(f"{lang} & {lang_think} & {matching_distribution_hack} \\\\ \n")
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--output_dir", type=str, default="outputs", help="Loading from which directory")
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
            for lang_think in langs_think:
                for lang in langs:
                    compute_matching_distribution(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name)