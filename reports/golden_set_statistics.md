# Golden Evaluation Set Statistical Summary Report

- **Total Golden Evaluation Examples**: `176`
- **Human Review Status**: `0 FINALIZED` | `176 PENDING_HUMAN_REVIEW`

---

## 1. Intent Category Distribution

| Intent Category | Count | Percentage | Mandatory Escalation Rule |
| :--- | :---: | :---: | :---: |
| `delivery_issue` | **22** | 12.5% | No (AUTO_HANDLE Candidate) |
| `refund_return_request` | **22** | 12.5% | Yes (ESCALATE) |
| `prime_membership_issue` | **22** | 12.5% | No (AUTO_HANDLE Candidate) |
| `information_request` | **22** | 12.5% | No (AUTO_HANDLE Candidate) |
| `other_general` | **22** | 12.5% | No (AUTO_HANDLE Candidate) |
| `payment_billing_issue` | **21** | 11.9% | Yes (ESCALATE) |
| `account_login_security` | **17** | 9.7% | Yes (ESCALATE) |
| `cancellation_request` | **16** | 9.1% | Yes (ESCALATE) |
| `missing_defective_item` | **12** | 6.8% | Yes (ESCALATE) |

---

## 2. Expected Action Distribution

- **ESCALATE**: `88` (50.0%)
- **AUTO_HANDLE**: `88` (50.0%)

---

## 3. Sampling Methodology

1. Candidates were sampled using stratified sampling across all 9 AmazonHelp intent categories from holdout test split (`data/splits/test.jsonl`).
2. Zero conversation leakage verified against training split and FAISS retrieval index.
3. Interactive developer labelling CLI (`scripts/create_golden_set.py --interactive`) allows human developers to inspect, modify, and confirm gold labels.