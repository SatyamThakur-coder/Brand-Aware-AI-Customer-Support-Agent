# Comprehensive Duplicate and Near-Duplicate Analysis Report

## Overview

This report documents the structural duplicate audit across dataset splits (`train.jsonl`, `validation.jsonl`, `test.jsonl`), the **Golden Evaluation Set** (`golden_set.jsonl`), and the **FAISS Vector Index**.

---

## 1. Conversation ID & Tweet ID Overlap Analysis

| Partition Pair | Shared Conversation IDs | Leakage Status |
| :--- | :---: | :---: |
| **Train vs Validation** | **0** | **Clean (Zero Leakage)** |
| **Train vs Test** | **0** | **Clean (Zero Leakage)** |
| **Val vs Test** | **0** | **Clean (Zero Leakage)** |
| **Golden Set vs Train** | **0** | **Clean (Zero Leakage)** |
| **Golden Set vs Val** | **0** | **Clean (Zero Leakage)** |
| **Golden Set vs Test** | **165** | **Sampled strictly from holdout Test split** |

---

## 2. Text Content & Near-Duplicate Analysis

| Audit Metric | Exact Count | Percentage of Golden Set (176 items) |
| :--- | :---: | :---: |
| **Exact Text Match in Training Set** | **4** | **2.27%** |
| **Exact Text Match in Validation Set** | **1** | **0.57%** |
| **Exact Text Match in Test Split** | **176** | **100.00%** |
| **Near-Duplicate Text in Train (>0.85 Sim)** | **5** | **2.84%** |
| **Clean Holdout Examples (Unique)** | **171** | **97.16%** |

---

## 3. Findings & Mitigation

1. **Conversation Grouping**: Grouping messages by `conversation_id` before splitting ensured **0 shared conversation IDs** between training and test sets.
2. **Short Generic Text Matches**: The 4 exact text matches between Golden Set and Training Set represent high-frequency generic tweets (e.g. `"Hello @AmazonHelp"`).
3. **Retrieval Self-Exclusion**: To prevent evaluation queries from matching identical training text, `src/retrieval.py` enforces explicit `exclude_conv_id` and `exclude_text` filters during search.
