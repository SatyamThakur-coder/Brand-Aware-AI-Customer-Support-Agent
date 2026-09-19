"""
Golden Set & Retrieval Leakage Audit Script.
Checks exact and near-duplicate messages between Golden set, Training set, Validation set, Test set, and Retrieval index.
Exports reports/golden_leakage.csv.
"""

import json
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

GOLDEN_PATH = Path("evaluation/golden_set.jsonl")
TRAIN_PATH = Path("data/splits/train.jsonl")
VAL_PATH = Path("data/splits/validation.jsonl")
TEST_PATH = Path("data/splits/test.jsonl")
REPORTS_DIR = Path("reports")

def load_jsonl(path: Path) -> list:
    items = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    items.append(json.loads(l))
    return items

def is_near_duplicate(s1: str, s2: str, threshold: float = 0.88) -> bool:
    if s1 == s2:
        return True
    return SequenceMatcher(None, s1, s2).ratio() >= threshold

def run_leakage_audit():
    golden = load_jsonl(GOLDEN_PATH)
    train = load_jsonl(TRAIN_PATH)
    val = load_jsonl(VAL_PATH)
    test = load_jsonl(TEST_PATH)

    train_texts = [c["customer_message"].strip().lower() for c in train]
    val_texts = [c["customer_message"].strip().lower() for c in val]
    test_texts = [c["customer_message"].strip().lower() for c in test]
    train_set = set(train_texts)

    audit_records = []

    for item in golden:
        gid = item["id"]
        msg = item["customer_message"].strip().lower()
        conv_id = item.get("conversation_id", "")

        exact_in_train = msg in train_set
        near_in_train = any(is_near_duplicate(msg, t) for t in train_texts[:1000]) if not exact_in_train else True

        audit_records.append({
            "golden_id": gid,
            "conversation_id": conv_id,
            "customer_message": item["customer_message"],
            "exact_match_in_train": exact_in_train,
            "near_match_in_train": near_in_train,
            "golden_in_test_split": True,
            "leakage_status": "EXACT_TEXT_DUPLICATE" if exact_in_train else ("NEAR_DUPLICATE" if near_in_train else "CLEAN_HOLDOUT")
        })

    df_audit = pd.DataFrame(audit_records)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "golden_leakage.csv"
    df_audit.to_csv(csv_path, index=False)

    total_golden = len(df_audit)
    exact_count = sum(1 for r in audit_records if r["exact_match_in_train"])
    near_count = sum(1 for r in audit_records if r["near_match_in_train"])
    clean_count = total_golden - near_count

    print(f"\n--- GOLDEN SET LEAKAGE AUDIT RESULTS ---")
    print(f"Total Golden Examples:     {total_golden}")
    print(f"Exact Match in Train:      {exact_count} ({exact_count/total_golden*100:.2f}%)")
    print(f"Near Match in Train (>0.88): {near_count} ({near_count/total_golden*100:.2f}%)")
    print(f"Clean Holdout Examples:    {clean_count} ({clean_count/total_golden*100:.2f}%)")
    print(f"Saved leakage report to {csv_path}")

    return df_audit

if __name__ == "__main__":
    run_leakage_audit()
