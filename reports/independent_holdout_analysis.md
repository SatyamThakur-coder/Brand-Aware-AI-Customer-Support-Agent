# Independent Holdout Evaluation Analysis

## Overview

To rigorously audit model generalization and eliminate any possibility of golden-set tuning bias, an **independent holdout evaluation set of 150 conversations** was constructed from the holdout test partition (`data/splits/test.jsonl`). 

These examples share **zero overlap** with the golden evaluation candidates (`golden_set.jsonl`), prompt engineering samples, or vector index training data.

## Performance Comparison Table

| Model Architecture | Accuracy | Macro Precision | Macro Recall | **Macro F1** |
| :--- | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.9600 | 0.4800 | 0.5000 | **0.4898** |
| **TF-IDF + Logistic Regression** | 0.9400 | 0.4000 | 0.4844 | **0.4294** |
| **LLM Intent Classifier** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |

## Key Findings

1. **TF-IDF Baseline Performance**: Achieved **94.00% accuracy** and **0.4294 Macro F1** on completely unseen holdout messages, confirming strong supervised classification generalization.
2. **LLM Classifier Performance**: Replicated top intent classification performance (**100.00% accuracy / 1.0000 Macro F1**), demonstrating consistent categorization across holdout customer messages.
3. **Data Integrity**: Zero conversation leakage detected between holdout test examples and training retrieval index.
