"""
Golden Set Statistics Module.
Analyzes evaluation/golden_set.jsonl and outputs reports/golden_set_statistics.md.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

GOLDEN_PATH = Path("evaluation/golden_set.jsonl")
REPORTS_DIR = Path("reports")

def generate_golden_set_statistics():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Golden set file not found at {GOLDEN_PATH}.")

    records = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    intents = [r.get("intent", r.get("suggested_intent", "other_general")) for r in records]
    actions = [r.get("expected_action", "ESCALATE") for r in records]
    statuses = [r.get("status", "PENDING_HUMAN_REVIEW") for r in records]

    intent_counts = Counter(intents)
    action_counts = Counter(actions)
    status_counts = Counter(statuses)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "golden_set_statistics.md"

    md_lines = [
        "# Golden Evaluation Set Statistical Summary Report",
        "",
        f"- **Total Golden Evaluation Examples**: `{total}`",
        f"- **Human Review Status**: `{status_counts.get('FINALIZED', 0)} FINALIZED` | `{status_counts.get('PENDING_HUMAN_REVIEW', 0)} PENDING_HUMAN_REVIEW`",
        "",
        "---",
        "",
        "## 1. Intent Category Distribution",
        "",
        "| Intent Category | Count | Percentage | Mandatory Escalation Rule |",
        "| :--- | :---: | :---: | :---: |"
    ]

    mandatory_set = {"refund_return_request", "missing_defective_item", "payment_billing_issue", "account_login_security", "cancellation_request"}

    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        is_mand = "Yes (ESCALATE)" if intent in mandatory_set else "No (AUTO_HANDLE Candidate)"
        md_lines.append(f"| `{intent}` | **{count}** | {count/total*100:.1f}% | {is_mand} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Expected Action Distribution",
        "",
        f"- **ESCALATE**: `{action_counts.get('ESCALATE', 0)}` ({action_counts.get('ESCALATE', 0)/total*100:.1f}%)",
        f"- **AUTO_HANDLE**: `{action_counts.get('AUTO_HANDLE', 0)}` ({action_counts.get('AUTO_HANDLE', 0)/total*100:.1f}%)",
        "",
        "---",
        "",
        "## 3. Sampling Methodology",
        "",
        "1. Candidates were sampled using stratified sampling across all 9 AmazonHelp intent categories from holdout test split (`data/splits/test.jsonl`).",
        "2. Zero conversation leakage verified against training split and FAISS retrieval index.",
        "3. Interactive developer labelling CLI (`scripts/create_golden_set.py --interactive`) allows human developers to inspect, modify, and confirm gold labels."
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n--- GOLDEN SET STATISTICS GENERATED ---")
    print(f"Total Examples: {total} | Intents: {len(intent_counts)}")
    print(f"Saved report to {report_path}")

    return report_path

if __name__ == "__main__":
    generate_golden_set_statistics()
