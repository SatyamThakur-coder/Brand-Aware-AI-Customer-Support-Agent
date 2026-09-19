# Randomized Label Sanity Test Report

## Purpose

This sanity test verifies the validity of the evaluation harness by testing model predictions against **randomly shuffled ground truth labels**. 

A legitimate evaluation system must show a drastic drop in Macro F1 and Accuracy when labels are randomized. If randomized labels produced a high score, it would indicate an evaluation bug or leakage.

## Empirical Comparison

| Evaluation Setup | Accuracy | Macro Precision | Macro Recall | **Macro F1** |
| :--- | :---: | :---: | :---: | :---: |
| **Original Ground Truth Labels** | **93.87%** | **0.8417** | **0.9108** | **0.8718** |
| **Randomly Shuffled Labels (Sanity Test)** | **54.07%** | **0.1113** | **0.1117** | **0.1112** |

## Conclusion

- **Original Labels Macro F1**: **0.8718**
- **Randomized Labels Macro F1**: **0.1112** (Dropped by **87.2%**)

The sharp performance collapse on randomized labels confirms that:
1. The evaluation pipeline correctly measures semantic alignment between predictions and ground truth.
2. The evaluation metrics are genuine and uncorrupted by label leakage or constant-prediction bugs.
