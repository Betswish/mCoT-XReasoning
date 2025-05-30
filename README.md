# When Models Reason in Your Language: Controlling Thinking Trace Language Comes at the Cost of Accuracy
[![Spaces](https://img.shields.io/badge/🤗-Open%20Data%20in%20HF-blue)](https://huggingface.co/collections/shanchen/xreasoning-681e7625c7a9ec4111a634b6)
[![Spaces](https://img.shields.io/badge/🤗-Open%20Trained%20Models%20in%20HF-orange)](https://huggingface.co/collections/shanchen/xreasoning-models-68377e15a2e86143dc4b0383)
## Environments

For a quick start, you may load our environment easily with Conda:
```
conda env create -f mCoT.yaml
```

Python: `3.12.8`


## Supported Models

The script now supports the following types of models:

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

## How to run
### Quick-start

As a quick start, you may simply run `bash run_multilingual.sh` for getting all results in our paper!

### Parameter Details

To execute cross-lingual reasoning tasks with a customized setup, utilize the `run.py` script with the following command-line arguments:

* `--mname`: Specifies the model name or path. For example, `"deepseek-ai/DeepSeek-R1-Distill-Llama-70B"` selects the 70B-parameter DeepSeek-R1-Distill-Llama model.

* `--lang`: Sets the language code for the input data. Supported options include `EN` (English), `ZH` (Chinese), `ES` (Spanish), `FR` (French), `DE`(German),  `JA`(Japanese), `RU`(Russian), `BN`(Bengali), `TH`(Thai), `SW`(Swahili), and `TE` (Telugu).

* `--lang_think`: Sets the language code for the language of thinking. Supported the same 11 languages.

* `--seed`: Sets seeds for generation with sampling decoding. When set to 0, the LRM will be forced to do greedy decoding.

* `--dataset`: Selects the dataset to use (`aime_combined`, `shanchen/gpqa_diamond_mc_multilingual:problem:solution`, `juletxara/mgsm`, etc.)

* `--temperature`: Controls the randomness of the model's output. Higher values (e.g., `0.9`) yield more diverse outputs, while lower values (e.g., `0.2`) produce more deterministic results.

* `--cache_dir`: Specifies the path to the cache directory, default `$TMPDIR`.

* `--top_p`: Applies nucleus sampling by considering the smallest set of tokens with a cumulative probability above this threshold. A value of `0.95` means the model will sample from the top 95% probability mass.

* `--max_tokens`: Defines the maximum number of tokens to generate in the output. Adjust this based on your model's capacity and the complexity of the task.

### Example Usage

To run the reasoning task in English using the DeepSeek-R1-Distill-Llama-70B model with specific sampling parameters:

```bash
python run.py \
  --mname "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
  --lang EN \
  --dataset aime_combined\
  --lang_think DE \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_tokens 16384
```

For a comprehensive list of available options and their descriptions, refer to the help command:

```bash
python run.py --help
```

## Evaluation (Language Matching \& Answer Accuracy)

For computing the language matching rate, run

```bash
python compute_matching.py --output_dir {YOUR output folder}
```

For showcasing the actual language distribution of the thinking traces, detect with LangDetect, run

```bash
python compute_matching_distribution.py --output_dir {YOUR output folder}
```

For calculating answer accuracy, run

```bash
python eval.py --output_dir {YOUR output folder}
```
* `output_dir`: output folder path, defaul `outputs_0`.


## XReasoning Benchmark

For easier usage, we have uploaded our datasets on our Huggingface. But we still provide a copy under `XReasoning_data` in this repository.
