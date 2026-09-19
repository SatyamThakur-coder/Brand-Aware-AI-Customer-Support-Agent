"""
Dataset split module enforcing strict conversation-level partitioning.
Splits data into Train (70%), Validation (15%), and Test (15%) with zero conversation leakage.
Assigns silver intent labels for baseline training using keyword rules.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

INTENT_KEYWORDS = {
    "refund_return_request": [r"\brefund\b", r"\breturn\b", r"\bmoney back\b", r"\breimburse\b"],
    "delivery_issue": [r"\bdelivery\b", r"\bpackage\b", r"\btracking\b", r"\bshipped\b", r"\barrive\b", r"\bcourier\b", r"\bdelayed\b"],
    "missing_defective_item": [r"\bmissing\b", r"\bbroken\b", r"\bdamaged\b", r"\bshattered\b", r"\bwrong item\b", r"\bdefective\b"],
    "payment_billing_issue": [r"\bcharge\b", r"\bbilling\b", r"\bdouble charged\b", r"\bgift card\b", r"\bpayment\b", r"\bcredit card\b"],
    "account_login_security": [r"\bpassword\b", r"\blocked out\b", r"\blogin\b", r"\bsign in\b", r"\bhacked\b", r"\bsecurity\b", r"\b2fa\b"],
    "prime_membership_issue": [r"\bprime\b", r"\bmembership\b", r"\bsubscription\b", r"\bfree trial\b", r"\bprime video\b"],
    "cancellation_request": [r"\bcancel\b", r"\bcancellation\b", r"\bstop order\b"],
    "information_request": [r"\bhow to\b", r"\bwhere can i\b", r"\bpolicy\b", r"\bwarranty\b", r"\bhours\b", r"\bwhat is\b"]
}

def assign_silver_intent(text: str) -> str:
    """
    Rule-based keyword matcher to assign silver intent labels for baseline training.
    """
    lower_text = text.lower()
    for intent, patterns in INTENT_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, lower_text):
                return intent
    return "other_general"

def split_conversations(
    processed_path: str = "data/processed/conversations_AmazonHelp.jsonl",
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Path, Path, Path]:
    """
    Performs strict conversation-level splitting to prevent data leakage across splits.
    Groups by unique conversation_id first so all messages in a thread stay together.
    """
    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found at {path}. Run preprocessing first.")

    print(f"Loading conversations from {path} for splitting...")
    conv_groups = defaultdict(list)
    total_records = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["silver_intent"] = assign_silver_intent(item["customer_message"])
                conv_groups[item["conversation_id"]].append(item)
                total_records += 1

    unique_conv_ids = list(conv_groups.keys())
    random.seed(seed)
    random.shuffle(unique_conv_ids)

    n_total_convs = len(unique_conv_ids)
    n_train_convs = int(n_total_convs * train_ratio)
    n_val_convs = int(n_total_convs * val_ratio)

    train_ids = set(unique_conv_ids[:n_train_convs])
    val_ids = set(unique_conv_ids[n_train_convs:n_train_convs + n_val_convs])
    test_ids = set(unique_conv_ids[n_train_convs + n_val_convs:])

    # Strict Zero-Leakage Checks
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected between Train and Val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage detected between Val and Test!"

    train_data = [item for cid in train_ids for item in conv_groups[cid]]
    val_data = [item for cid in val_ids for item in conv_groups[cid]]
    test_data = [item for cid in test_ids for item in conv_groups[cid]]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    test_path = out_dir / "test.jsonl"

    for p, dataset in [(train_path, train_data), (val_path, val_data), (test_path, test_data)]:
        with open(p, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

    print(f"Split complete (Strict conversation-level grouping, ZERO leakage):")
    print(f"  Unique Conversations Total: {n_total_convs:,}")
    print(f"  Train: {len(train_ids):,} convs / {len(train_data):,} records ({len(train_data)/total_records*100:.1f}%) -> {train_path}")
    print(f"  Val:   {len(val_ids):,} convs / {len(val_data):,} records ({len(val_data)/total_records*100:.1f}%) -> {val_path}")
    print(f"  Test:  {len(test_ids):,} convs / {len(test_data):,} records ({len(test_data)/total_records*100:.1f}%) -> {test_path}")

    return train_path, val_path, test_path

if __name__ == "__main__":
    split_conversations()
