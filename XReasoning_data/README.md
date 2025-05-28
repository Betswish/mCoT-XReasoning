# XReasoning Benchmark

This folder contains the XReasoning benchmark we established, where we translate the AIME2024, AIME2025 and GPQA\_Diamond dataset into 10 non-English languages to be aligned with languages covered by MGSM.

You could load the datasets via the following command:

``
from datasets import load_dataset

name_data = DATASET_NAME
load_dataset(name_data)
``
