# Training: Budget Alignment for Multilingual Reasoning

This directory contains training scripts for **Budget Alignment: Making Models Reason in the User's Language** — a practical approach to align LLMs to reason in the user's query language (Japanese, French, Spanish, etc.) rather than defaulting to English/Chinese.

📖 **Blog Post**: https://huggingface.co/blog/shanchen/mcot-rl  
🤗 **Models & Data**: https://huggingface.co/collections/shanchen/xreasoning
🤔 **Eval code**: https://github.com/Betswish/mCoT-pass-K
---

## Overview

Large language models often **answer** in the requested language but **reason** in English/Chinese internally. This creates issues with instruction-following, human oversight, and multilingual evaluation trustworthiness.

Our two-step approach achieves strong language consistency without sacrificing accuracy:

1. **Small SFT** (~817 multilingual reasoning chains) → Reprogram the model's "inner monologue"
2. **Math-focused GRPO** → Recover and boost accuracy while preserving language consistency

### Key Results

- **Language consistency**: 99-100% for FR/ES, 85-95% for JA (vs. ~20-40% baseline)
- **Accuracy**: Pareto improvement on 9/12 language-dataset pairs
- **Efficiency**: Only 817 SFT examples + math-only GRPO

---

## Repository Structure

```
training/
├── accelerate_configs/        # Distributed training configs (DeepSpeed, FSDP)
│   ├── deepspeed_zero1.yaml
│   ├── deepspeed_zero2.yaml
│   ├── deepspeed_zero3.yaml
│   ├── fsdp1.yaml
│   ├── fsdp2.yaml
│   ├── multi_gpu.yaml
│   └── single_gpu.yaml
├── train.sh                   # Main training script (SLURM example)
├── trl_grpo_en.py            # GRPO training for English
├── trl_grpo_fr.py            # GRPO training for French
├── trl_grpo_jp.py            # GRPO training for Japanese
└── trl_hf_es.py              # SFT/training for Spanish
```

---

## Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n mcot-training python=3.10
conda activate mcot-training

# Install dependencies
pip install torch transformers datasets trl vllm accelerate deepspeed
pip install wandb  # Optional: for experiment tracking
```

### Step 1: Small SFT (Language Consistency)

Fine-tune the base model on ~817 multilingual reasoning chains:

```bash
# Example: Spanish SFT
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
    trl_hf_es.py
```

This step teaches the model to **reason in the target language** with minimal data.

### Step 2: GRPO (Accuracy Recovery)

Run GRPO on math data to boost accuracy while maintaining language consistency:

```bash
# Start vLLM inference server for the SFT model
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model shanchen/ds-limo-ja-500 \
    --tensor_parallel_size 1 &

# Run GRPO training
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    trl_grpo_jp.py
```

**Key GRPO Settings:**
- **No KL penalty**: Allows policy flexibility
- **High clip ratio**: 0.28 / -0.2 (DAPO-style) to prevent reverting to English
- **Rollout**: 24 samples per prompt
- **LoRA**: r=8, lr=1e-5
- **Rewards**: 1.0 (accuracy) + 0.2 (language consistency) + 0.2 (format)

### Full Pipeline (Batch Script)

See `train.sh` for a complete SLURM example:

```bash
sbatch train.sh
```

This script:
1. Sets up Ray cluster for distributed inference
2. Launches vLLM server on 1 GPU
3. Runs GRPO training on 3 GPUs with DeepSpeed ZeRO-3

---

## Configuration

### Distributed Training

Choose an appropriate config from `accelerate_configs/`:

- **Single GPU**: `single_gpu.yaml` (development)
- **Multi-GPU**: `multi_gpu.yaml` (small clusters)
- **DeepSpeed ZeRO-1/2/3**: `deepspeed_zero*.yaml` (large models, memory-efficient)
- **FSDP**: `fsdp*.yaml` (PyTorch-native distributed training)

### Language-Specific Scripts

Each script (`trl_grpo_{lang}.py`) contains:
- **Language-specific system prompts** with in-language reasoning instructions
- **Reward functions** that verify language consistency + correctness
- **Format checking** for `<think>...</think>` and `\\boxed{}` patterns

**Example (Japanese):**
```python
SYSTEM_PROMPT: str = (
    "常に日本語で考えてください。\n\n"
    "次の数学問題を段階的に解いてください。\n"
    "各ステップで、<think>推論過程をここに書く</think>の形式で推論を書いてください。\n"
    "最後に、\\boxed{}で囲んで最終回答を提供してください。\n\n"
    "問題: {text}\n\n"
    "回答は次の形式である必要があります: '答えは: \\boxed{最終回答}'、"
    "推論は<think></think>タグ内に、答えは\\boxed{}内に書いてください。"
)
```

---

## Practical Recommendations

Based on our experiments across JA/FR/ES on MMLU-Math, AIME, GPQA, and MMLU-Medicine:

### 🎯 If You Can Only Afford One Step

**Do SFT only** (few hundred high-quality chains):
- Near-perfect language consistency (FR/ES: 99-100%, JA: 85-95%)
- Often improves in-domain accuracy
- Minimal compute cost

### 🚀 If You Can Afford Two Steps

**Do SFT → GRPO-from-SFT**:
- SFT secures language consistency
- GRPO recovers/boosts accuracy on hard sets (AIME, GPQA)
- Use **high clip / no KL** to avoid reverting to English

### 🔀 Model Merging (Bonus)

Merge SFT models for different languages using [mergekit](https://github.com/arcee-ai/mergekit):

```bash
# Equal linear merge of base + SFT-JA + SFT-FR + SFT-ES
mergekit --config merge_config.yaml
```

**Benefits:**
- Shrinks worst-case losses across languages
- More robust on out-of-domain tasks (e.g., medicine)
- No additional training required

### ⚠️ Known Challenges & Fixes

| Issue | Languages | Fix |
|-------|-----------|-----|
| **Tokenization tax** | Japanese | Normalize digits (half-width), explicit templates |
| **Cue misalignment** | Spanish (AIME) | Prefer English math terms (`gcd`, `mod`), adjust rewards |
| **Domain mismatch** | All (Medicine) | Add small domain SFT or medical reward head |

---

## Evaluation

We evaluated on:
- **MMLU College Math** (in-domain, easy)
- **AIME 2025** (in-domain, hard)
- **GPQA** (out-of-domain, hard science)
- **MMLU Pro Medicine** (out-of-domain, hard medical)

**Metrics:**
- `pass@k (k=1,5,10)` with `n=32` rollouts for accuracy
- Language consistency % (both reasoning traces **and** final answers must be in the requested language)

### Key Findings

✅ **GRPO-from-SFT Pareto-dominates Base** in 9/12 language–dataset pairs  
✅ **SFT alone** solves language consistency with minimal cost  
✅ **Model merging** is a practical one-stop solution for robustness  

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{qi2025modelsreasonlanguagecontrolling,
    title={When Models Reason in Your Language: Controlling Thinking Language Comes at the Cost of Accuracy}, 
    author={Jirui Qi and Shan Chen and Zidi Xiong and Raquel Fernández and Danielle S. Bitterman and Arianna Bisazza},
    year={2025},
    eprint={2505.22888},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.22888}, 
}

@misc{chen2025budgetalignment,
    title={Budget Alignment: Making Models Reason in the User's Language}, 
    author={Shan Chen and Jirui Qi and Zidi Xiong and Arianna Bisazza and Raquel Fernández and Danielle S. Bitterman},
    year={2025},
    url={https://github.com/Betswish/mCoT-XReasoning/tree/main/training}, 
}
```

---

## Authors

[Shan Chen*](https://shanchen.dev/) | [Jirui Qi*](https://betswish.github.io/) | [Zidi Xiong](https://polaris-73.github.io/) | [Arianna Bisazza](https://inclow-lm.github.io/) | [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/) | [Danielle Bitterman](https://www.bittermanlab.org/)

**Institutions**: Mass General Brigham | Harvard University | University of Groningen | University of Amsterdam

---

## License

Please refer to the main repository for license information.

## Acknowledgments

- Built with [TRL](https://github.com/huggingface/trl) and [vLLM](https://github.com/vllm-project/vllm)
- Base model: [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- SFT data: [LiMO](https://arxiv.org/abs/2502.03387) (817 multilingual reasoning chains)

