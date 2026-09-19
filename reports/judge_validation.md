# LLM-as-a-Judge Validation Report

## Overall Status: **`Human judge validation pending.`**

---

## 1. Validation Methodology & Rubric

The automated LLM Judge (`evaluation/llm_judge.py`) evaluates generated customer support responses on a **1.0 to 5.0 scale** across 7 core dimensions defined in `evaluation/judge_rubric.yaml`:
1. **Groundedness**: Verified against retrieved historical evidence.
2. **Helpfulness**: Usefulness in addressing customer inquiry.
3. **Correctness**: Factually sound communication.
4. **Relevance**: Conciseness and topic alignment.
5. **Brand Consistency**: Professional, polite tone matching `AmazonHelp` persona.
6. **Unsupported Claims**: Strict penalization for false promises (refund dates, compensation).
7. **Escalation Appropriateness**: Safety decision alignment.

---

## 2. Statistical Agreement Metrics

| Metric | Empirical Value | Target Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Human Rating Sample Count** | `0` | 50 pairs | Pending Human Annotation |
| **Agreement Rate (<= 1.0 point diff)** | `N/A` | >= 80.0% | Pending |
| **Mean Absolute Error (MAE)** | `N/A` | <= 0.50 | Pending |
| **Pearson Correlation** | `N/A` | >= 0.75 | Pending |

---

## 3. Important Methodological Disclaimer

> [!IMPORTANT]
> **No Fabricated Data**: Human ratings must be annotated by real human evaluators. In accordance with strict evaluation guidelines, zero fake or LLM-generated human ratings were inserted into `evaluation/human_ratings.csv`. Until human evaluators complete the 50-pair annotation, the judge status is reported as **"Human judge validation pending."**