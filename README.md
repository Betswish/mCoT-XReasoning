# hard-multilingual-cot

## Environments

For a quick start, you may load our environment easily with Conda:
```
conda env create -f mCoT.yaml
```

Python: `3.12.8`


## Supported Models

The script now supports the following types of models:

1. **VLLM-based models** (default):

- DeepSeek-R1-Distill Series
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    - deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    - deepseek-ai/DeepSeek-R1-Distill-Llama-70B

- Skywork-OR1 Series
    - Skywork/Skywork-OR1-7B
    - Skywork/Skywork-OR1-32B

## Quick-start

You may simply run `bash run_multilingual.sh` for getting all results in our paper!

## XReasoning Benchmark

For easier usage, we have uploaded our datasets on our Huggingface. But we still provide a copy under `XReasoning_data` in this repository.

## Additional Parameters

If you wanna run cross-lingual thinking with a single setup on your own, try `run.py` with specified parameters:

- `--temperature`: Sampling temperature (default: 0.6)
- `--top_p`: Top-p sampling parameter (default: 0.7)
- `--max_tokens`: Maximum number of tokens to generate (default: 4096)

Example:

```bash
python run.py --mname "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" --lang EN --temperature 0.7 --top_p 0.95 --max_tokens 16384
```