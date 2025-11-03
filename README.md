# When Models Reason in Your Language: Controlling Thinking Language Comes at the Cost of Accuracy
<div align="center">
    
[![Spaces](https://img.shields.io/badge/🤗-Open%20Data%20in%20HF-blue)](https://huggingface.co/collections/shanchen/xreasoning-681e7625c7a9ec4111a634b6)
[![Spaces](https://img.shields.io/badge/🤗-Open%20Trained%20Models%20in%20HF-orange)](https://huggingface.co/collections/shanchen/xreasoning-models-68377e15a2e86143dc4b0383)
_<sup>†</sup>Co-first authors, <sup>‡</sup>Co-senior authors_



[Jirui Qi<sup>1†</sup>](https://betswish.github.io/) • [Shan Chen<sup>2,3,4†</sup>](https://shanchen.dev/) • [Zidi Xiong<sup>2</sup>](https://polaris-73.github.io/)

[Raquel Fernández<sup>5</sup>](https://staff.fnwi.uva.nl/r.fernandezrovira/) • [Danielle S. Bitterman<sup>2,3,4‡</sup>](https://www.bittermanlab.org/people/DanielleBitterman) • [Arianna Bisazza<sup>1‡</sup>](https://www.cs.rug.nl/~bisazza/)  

<sup>1</sup>University of Groningen, <sup>2</sup>Harvard University, <sup>3</sup>Mass General Brigham,

<sup>4</sup>Boston Children’s Hospital, <sup>5</sup>University of Amsterdam


</div>

> [!NOTE] 
> To see our exploration of GRPO for solving this issue. See Readme.md in `training/`.


---

**Latest update:** Our [paper](https://arxiv.org/abs/2505.22888) has been accepted by the Findings of [EMNLP 2025](https://2025.emnlp.org/)! 🎉 

**Overall Assessment Score (from four reviewers):** 3, 3, 3.5, 4 out of 5.

---

If you find the paper helpful and use the content, we kindly suggest you cite through:
```bibtex
@article{qi2025models,
  title={When Models Reason in Your Language: Controlling Thinking Trace Language Comes at the Cost of Accuracy},
  author={Qi, Jirui and Chen, Shan and Xiong, Zidi and Fern{\'a}ndez, Raquel and Bitterman, Danielle S and Bisazza, Arianna},
  journal={arXiv preprint arXiv:2505.22888},
  year={2025}
}
```

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

As a quick start, you may simply run `bash run_multilingual.sh` to get all generations of the LRMs tested in our paper! By default, the responses will be stored in `outputs_0/`.

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
* `output_dir`: output folder path, default `outputs_0/`.
* Two scores will be stored in each row of the .csv file: the first is the accuracy/matching rate of LRM with standard prompts; while the second is about the hacked prompts.

## XReasoning Benchmark

For easier usage, we have uploaded our datasets to our [🤗Huggingface](https://huggingface.co/collections/shanchen/xreasoning). But we still provide a copy under `XReasoning_data` in this repository.
