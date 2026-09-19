# Brand-Aware AI Customer Support Agent

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-13%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![Evaluation](https://img.shields.io/badge/Evaluation-Audit%20Verified-orange.svg)](reports/intent_results.csv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade AI Customer Support system built for the **Hiver SDE Intern Take-Home Assignment**. 

The goal of this repository is NOT to build an unconstrained chatbot. The goal is to take a messy, real-world customer support dataset (**Customer Support on Twitter**, 2.8M tweets), construct a brand-grounded AI support system (`AmazonHelp`), enforce strict safety escalation rules, and rigorously evaluate system performance against baselines, holdout test sets, and human-labelled benchmarks.

---

## Methodological Disclaimer & Honesty Note

> [!IMPORTANT]
> **The initial 100% classifier result was not genuine LLM performance.** It originated from the deterministic mock/rule-based classifier used for offline testing. It is therefore excluded from claims about LLM generalization. Real LLM API evaluation remains pending external API execution.

---

## Empirical Evaluation Summary

All reported evaluation metrics are generated from actual execution on holdout test data (`data/splits/test.jsonl`).

### 1. Intent Classification Baseline & Holdout Comparison

| System | Dataset | Accuracy | Macro F1 | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Majority baseline** | Golden evaluation | 73.20% | 0.0939 | Verified |
| **TF-IDF + Logistic Regression** | Golden evaluation | 93.87% | 0.8718 | Verified |
| **TF-IDF + Logistic Regression** | Independent holdout | 94.00% | 0.4294 | Verified |
| **Mock/rule-based classifier** | Rule evaluation | 100.00% | 1.0000 | Mock only |
| **Real LLM classifier** | Independent holdout | — | — | Pending API evaluation |

*Full analysis: [`reports/generalization_gap.md`](reports/generalization_gap.md) | Audit report: [`reports/codebase_audit.md`](reports/codebase_audit.md)*

### 2. Auto-Handling Coverage vs Escalation Metrics

| Operational Metric | Empirical Result | Meaning & Interpretation |
| :--- | :---: | :--- |
| **Auto-Handling Coverage** | **17.00%** | Proportion of incoming messages eligible for automated handling (safe public FAQs) |
| **Escalation Rate** | **83.00%** | Proportion of incoming messages safely routed to human support staff |
| **Escalation Precision / Recall** | **Pending** | Escalation precision/recall not yet human-validated |
| **LLM Judge Overall Score** | **4.56 / 5.00** | Automated rubric score (*Human judge validation pending*) |
| **Unsupported Claim Score** | **5.00 / 5.00** | Zero false promises or unverified policy claims made |

---

## System Architecture Overview

```
Customer Message
       │
       ▼
┌───────────────────────────────┐
│ 1. Text Preprocessing & Clean │ ──► URL / Mention / HTML Normalization
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 2. Intent Classification      │ ──► [UNKNOWN / LOW_CONFIDENCE] State
└──────────────┬────────────────┘
               │ (Intent + Confidence)
               ▼
┌───────────────────────────────┐
│ 3. FAISS Vector Retrieval     │ ──► Top-K Cases (Train Split Only, Self-Exclusion)
└──────────────┬────────────────┘
               │ (Historical Evidence)
               ▼
┌───────────────────────────────┐
│ 4. Grounded Response Gen      │ ──► Evidence-based Draft Reply
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 5. Deterministic Safety Engine│ ──► AUTO_HANDLE (17%) vs ESCALATE (83%) + Reason
└───────────────────────────────┘
```

---

## Vector Retrieval Evaluation Notes

- **Vector Model**: `sentence-transformers/all-MiniLM-L6-v2` (7,002 training vectors)
- **Top-1 Similarity**: `0.7278` | **Top-3 Similarity**: `0.6720` | **Top-5 Similarity**: `0.6401`
- **Top-1 Intent Match Rate**: `83.5%` | **Evidence Utility Rate (Sim >= 0.45)**: `99.4%`

> [!NOTE]
> **Important Distinction**: **Semantic similarity ≠ correctness**, and **retrieval quality ≠ response quality**. While FAISS retrieval successfully locates semantically close historical cases, a high similarity score does not guarantee that the historical resolution is appropriate for current company policy.

---

## Golden Evaluation Set & Human Review Tooling

- **Candidate Set**: 176 candidate records sampled across all 9 AmazonHelp intents from holdout test split (`evaluation/golden_set.jsonl`).
- **Interactive Review CLI**: Run `python scripts/create_golden_set.py --interactive` to manually review, assign gold intents, set expected actions, and export to `evaluation/golden_set_final.jsonl`.
- **Validation Script**: Run `python scripts/validate_golden_set.py` to verify record counts, schema integrity, intent coverage, and zero leakage against the retrieval index.
- **Human Judge Ratings Workflow**: 50 generated response pairs populated in [`evaluation/human_ratings.csv`](evaluation/human_ratings.csv). Human evaluators can manually input scores and run `python evaluation/human_judge_agreement.py` to calculate Agreement %, MAE, and Pearson Correlation.

---

## Quickstart Commands

### 1. Run Interactive Human Golden-Set Review Tool
```bash
python scripts/create_golden_set.py --interactive
```

### 2. Validate Golden Evaluation Set
```bash
python scripts/validate_golden_set.py
```

### 3. Run Offline Smoke Test (< 10 seconds)
```bash
python scripts/smoke_test.py
```

### 4. Run Pytest Test Suite
```bash
python -m pytest
```

### 5. Run Pipeline CLI Demonstration
```bash
python -m src.pipeline --message "My refund for order #102 has not arrived yet"
```

### 6. Full End-to-End Reproduction (< 5 minutes)
```bash
python scripts/reproduce.py
```

### 7. Launch Interactive Streamlit UI Demo
```bash
streamlit run app.py
```

---

## Detailed Audit Reports & Documentation

- **[DECISIONS.md](DECISIONS.md)**: 15 key technical trade-offs and architectural decisions.
- **[INTERVIEW_NOTES.md](INTERVIEW_NOTES.md)**: 15 concise answers to live technical interview questions.
- **[reports/codebase_audit.md](reports/codebase_audit.md)**: Full codebase and evaluation methodology audit.
- **[reports/generalization_gap.md](reports/generalization_gap.md)**: Investigation of TF-IDF performance on test split vs holdout set.
- **[reports/misleading_headline_number.md](reports/misleading_headline_number.md)**: Critical analysis explaining raw accuracy and quality score limitations.
- **[reports/duplicate_analysis.md](reports/duplicate_analysis.md)**: Audit of exact and near-duplicate messages.
- **[reports/golden_leakage.csv](reports/golden_leakage.csv)**: Leakage audit of golden set vs train/val/test splits.
- **[reports/golden_set_statistics.md](reports/golden_set_statistics.md)**: Statistical breakdown of the golden evaluation set.
- **[reports/retrieval_analysis.md](reports/retrieval_analysis.md)**: FAISS vector retrieval quality report.
- **[reports/failure_analysis.md](reports/failure_analysis.md)**: Failure analysis across top 5 real error categories.
- **[reports/judge_validation.md](reports/judge_validation.md)**: Human judge validation status.
- **[reports/final_submission_checklist.md](reports/final_submission_checklist.md)**: Submission readiness checklist.

---

## One-Week Production Roadmap

1. **Cross-Encoder Reranking**: Add `ms-marco-MiniLM-L-6-v2` reranking on top of FAISS initial search.
2. **Full Human Golden-Set Annotations**: Complete double-blind human annotation for 500 gold examples.
3. **Calibrated Confidence**: Apply temperature scaling on classifier logits to calibrate probabilities.
4. **FastAPI & Evidently Monitoring**: Wrap pipeline in a production REST API with data drift tracking.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
