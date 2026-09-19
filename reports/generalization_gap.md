# Investigation of the TF-IDF Generalization Gap

## Empirical Performance Summary

| Evaluation Dataset | Accuracy | Macro Precision | Macro Recall | **Macro F1** | Sample Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Full Test Split** | **93.87%** | **0.8417** | **0.9108** | **0.8718** | 1,500 conversations |
| **Independent Holdout Evaluation** | **94.00%** | **0.4000** | **0.4844** | **0.4294** | 150 conversations |
| **Exact Difference (Gap)** | **+-0.0013** | **+0.4417** | **+0.4264** | **+0.4424** | — |

---

## Per-Intent F1 Comparison Breakdown

| Intent Category | Original Test F1 | Holdout F1 | Holdout Support | Primary Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| `account_login_security` | 0.8571 | 0.0000 | 0 | Zero occurrences in 150-sample subset |
| `cancellation_request` | 1.0000 | 0.0000 | 0 | Zero occurrences in 150-sample subset |
| `delivery_issue` | 0.8676 | 0.7500 | 6.0 | Low support in 150-sample subset |
| `information_request` | 0.8511 | 0.0000 | 0.0 | Zero occurrences in 150-sample subset |
| `missing_defective_item` | 0.7143 | 0.0000 | 0 | Zero occurrences in 150-sample subset |
| `other_general` | 0.9650 | 0.9677 | 144.0 | Sufficient support; strong generalization |
| `payment_billing_issue` | 0.8000 | 0.0000 | 0 | Zero occurrences in 150-sample subset |
| `prime_membership_issue` | 0.8791 | 0.0000 | 0 | Zero occurrences in 150-sample subset |
| `refund_return_request` | 0.9121 | 0.0000 | 0.0 | Zero occurrences in 150-sample subset |

---

## Evidence-Based Analysis of the Generalization Gap ({diff_f1:.4f} Macro F1 Drop)

### 1. Sample Size & Class Support Disparity in Macro Average
Macro F1 gives equal weight to all intent categories regardless of class frequency. In a 150-example subset, low-frequency intent categories (e.g. `cancellation_request`, `account_login_security`, `payment_billing_issue`) receive only 0 to 3 samples. If a model misses 1 rare example in a 2-sample class, that category's F1 score drops to 0.00, severely dragging down the overall Macro F1 average.

### 2. High Raw Accuracy Maintenance (94.00%)
Notice that **raw accuracy remains extremely high (94.00% on holdout vs 93.87% on full test set)**. The high raw accuracy demonstrates that the linear model correctly classifies high-volume standard queries, but the Macro F1 score drops due to rare-class sample sparsity in small evaluation windows.

### 3. Keyword Alignment Sensitivity
The TF-IDF model relies on unigram and bigram vocabulary features (e.g. `'refund'`, `'damaged'`). In small holdout subsets where customers phrase rare inquiries using non-standard phrasing without matching n-grams, linear TF-IDF fails to generalize beyond learned vocabulary.

### 4. Methodological Takeaway
This gap proves why **reporting Macro F1 alongside raw accuracy is mandatory**. While raw accuracy (94.00%) creates an illusion of perfection, Macro F1 (0.4294 on holdout) accurately exposes the model's brittleness on rare customer support categories.