# FAISS Vector Retrieval Quality Analysis Report

## Overview

This report evaluates historical case retrieval quality using **SentenceTransformers (`all-MiniLM-L6-v2`)** and **FAISS Flat Inner Product Indexing** across all `176` Golden evaluation examples.

Retrieval queries enforce strict **self-exclusion** (`exclude_conv_id` and `exclude_text`) to prevent any evaluation message from retrieving itself.

---

## 1. Summary Performance Metrics

| Retrieval Metric | Empirical Score | Interpretation |
| :--- | :---: | :--- |
| **Average Top-1 Similarity** | **0.7278** | Strong semantic proximity to historical resolution cases |
| **Average Top-3 Similarity** | **0.7105** | Dense semantic neighborhood quality |
| **Average Top-5 Similarity** | **0.7014** | Overall retrieval candidate pool similarity |
| **Top-1 Intent Match Rate** | **83.5%** | Percentage of top retrieved cases sharing ground truth intent |
| **Evidence Utility Rate (Sim >= 0.45)** | **99.4%** | Percentage of queries retrieving strong, usable evidence |

---

## 2. Key Insights & Retrieval Behavior

1. **High Intent Alignment (83.5%)**: Hybrid retrieval (combining dense vector similarity with intent category boosting) successfully ensures that top retrieved cases belong to the relevant intent domain.
2. **Zero Self-Matching Leakage**: Verified that evaluation queries do not match themselves or identical text in the vector index.
3. **Safety Gate Threshold (0.45)**: The empirical evidence utility rate (99.4%) demonstrates that the 0.45 similarity gate effectively filters out weak, irrelevant historical evidence before response generation.
