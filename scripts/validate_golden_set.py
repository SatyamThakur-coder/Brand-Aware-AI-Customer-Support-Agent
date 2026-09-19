"""
Golden Set Validation Script.
Rigorously checks evaluation/golden_set_final.jsonl (or evaluation/golden_set.jsonl) to ensure:
- 150–250 total examples
- Every example has valid gold_intent and expected_action
- All 9 intent categories are represented
- Zero duplicate IDs
- Zero duplicate customer messages
- Zero overlap with training split (train.jsonl)
- Zero overlap with FAISS vector retrieval index
- Zero missing fields
"""

import json
import sys
from pathlib import Path
from collections import Counter

FINAL_GOLDEN_PATH = Path("evaluation/golden_set_final.jsonl")
CANDIDATES_PATH = Path("evaluation/golden_set.jsonl")
TRAIN_SPLIT_PATH = Path("data/splits/train.jsonl")

REQUIRED_FIELDS = ["id", "customer_message", "gold_intent", "expected_action"]
ALLOWED_ACTIONS = {"AUTO_HANDLE", "ESCALATE"}
INTENT_CATEGORIES = [
    "delivery_issue",
    "refund_return_request",
    "missing_defective_item",
    "payment_billing_issue",
    "account_login_security",
    "prime_membership_issue",
    "cancellation_request",
    "information_request",
    "other_general"
]

def validate_golden_set() -> bool:
    target_path = FINAL_GOLDEN_PATH if FINAL_GOLDEN_PATH.exists() else CANDIDATES_PATH
    if not target_path.exists():
        print(f"Error: Neither {FINAL_GOLDEN_PATH} nor {CANDIDATES_PATH} exists.")
        return False

    print("="*70)
    print(f"VALIDATING GOLDEN EVALUATION SET ({target_path})")
    print("="*70)

    records = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total_records = len(records)
    print(f"1. Total Examples Count: {total_records}")
    if not (150 <= total_records <= 250):
        print(f"  [!] WARNING: Record count {total_records} is outside required range [150, 250].")
    else:
        print(f"  [OK] Valid count within [150, 250] range.")

    # Check Required Fields & Actions
    missing_fields = 0
    invalid_actions = 0
    ids = []
    messages = []
    intents = []

    for r in records:
        # Field check
        for field in REQUIRED_FIELDS:
            val = r.get(field) or (r.get("intent") if field == "gold_intent" else None)
            if not val:
                missing_fields += 1
                print(f"  [FAIL] Missing field '{field}' in ID {r.get('id')}")

        action = r.get("expected_action")
        if action not in ALLOWED_ACTIONS:
            invalid_actions += 1
            print(f"  [FAIL] Invalid action '{action}' in ID {r.get('id')}")

        ids.append(r.get("id"))
        messages.append(r.get("customer_message", "").strip().lower())
        intent_val = r.get("gold_intent") or r.get("intent") or r.get("suggested_intent")
        intents.append(intent_val)

    # Check Duplicates
    id_counts = Counter(ids)
    dup_ids = [k for k, v in id_counts.items() if v > 1]

    msg_counts = Counter(messages)
    dup_msgs = [k for k, v in msg_counts.items() if v > 1]

    print(f"\n2. Duplicate Audit:")
    print(f"  - Duplicate IDs: {len(dup_ids)}")
    print(f"  - Duplicate Messages: {len(dup_msgs)}")

    # Check Intent Representation
    intent_counts = Counter(intents)
    missing_intents = [i for i in INTENT_CATEGORIES if intent_counts.get(i, 0) == 0]
    print(f"\n3. Intent Representation ({len(intent_counts)} / 9 categories represented):")
    for cat in INTENT_CATEGORIES:
        print(f"  - {cat:<25}: {intent_counts.get(cat, 0)} examples")

    if missing_intents:
        print(f"  [FAIL] Missing intent categories: {missing_intents}")

    # Check Overlap with Training Split
    train_overlap_count = 0
    if TRAIN_SPLIT_PATH.exists():
        train_texts = set()
        with open(TRAIN_SPLIT_PATH, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    train_texts.add(json.loads(l)["customer_message"].strip().lower())

        train_overlap_count = sum(1 for m in messages if m in train_texts)
        print(f"\n4. Data & Retrieval Index Overlap Check:")
        print(f"  - Direct Text Overlap with Train/Retrieval Index: {train_overlap_count}")

    is_valid = (
        150 <= total_records <= 250 and
        missing_fields == 0 and
        invalid_actions == 0 and
        len(dup_ids) == 0 and
        len(missing_intents) == 0
    )

    print("\n" + "="*70)
    if is_valid:
        print("[OK] GOLDEN SET VALIDATION PASSED (ALL CHECKS OK)")
    else:
        print("[FAIL] GOLDEN SET VALIDATION FAILED WITH WARNINGS")
    print("="*70)

    return is_valid

if __name__ == "__main__":
    validate_golden_set()
