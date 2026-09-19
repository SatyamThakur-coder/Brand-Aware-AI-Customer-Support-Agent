"""
Interactive Golden-Set Human Labelling CLI.
Allows human developers to review candidate customer support messages,
assign/confirm gold_intent, set expected_action (AUTO_HANDLE vs ESCALATE), add notes,
and output human-confirmed evaluation records to evaluation/golden_set_final.jsonl.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

INPUT_SPLIT = Path("data/splits/test.jsonl")
CANDIDATES_PATH = Path("evaluation/golden_set.jsonl")
FINAL_GOLDEN_PATH = Path("evaluation/golden_set_final.jsonl")

INTENT_OPTIONS = [
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

def generate_candidate_golden_set(num_samples: int = 200) -> List[Dict[str, Any]]:
    """
    Samples candidates from test set stratified by silver intent.
    Initializes candidate file evaluation/golden_set.jsonl.
    """
    if not INPUT_SPLIT.exists():
        raise FileNotFoundError(f"Test split not found at {INPUT_SPLIT}. Run src.dataset_split first.")

    print(f"Sampling {num_samples} candidate examples from {INPUT_SPLIT}...")
    by_intent = {}
    with open(INPUT_SPLIT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                silver = item.get("silver_intent", "other_general")
                if silver not in by_intent:
                    by_intent[silver] = []
                by_intent[silver].append(item)

    candidates = []
    per_intent_count = num_samples // len(INTENT_OPTIONS)

    idx = 1
    for intent in INTENT_OPTIONS:
        items = by_intent.get(intent, [])
        sampled = items[:per_intent_count]
        for item in sampled:
            suggested_action = "ESCALATE" if intent in ["refund_return_request", "missing_defective_item", "payment_billing_issue", "account_login_security", "cancellation_request"] else "AUTO_HANDLE"
            
            candidates.append({
                "id": f"gold_{idx:03d}",
                "conversation_id": item["conversation_id"],
                "customer_message": item["customer_message"],
                "brand_response_reference": item["brand_response"],
                "suggested_intent": intent,
                "gold_intent": intent,
                "expected_action": suggested_action,
                "notes": "Candidate sampled for human review."
            })
            idx += 1

    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    print(f"Saved {len(candidates)} candidates to {CANDIDATES_PATH}.")
    return candidates

def interactive_labelling():
    """
    Interactive CLI tool for human developer to manually review and confirm gold labels.
    Saves final human-confirmed results to evaluation/golden_set_final.jsonl.
    """
    if not CANDIDATES_PATH.exists():
        generate_candidate_golden_set()

    candidates = []
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    # Check existing progress in final file
    finalized_dict = {}
    if FINAL_GOLDEN_PATH.exists():
        with open(FINAL_GOLDEN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    finalized_dict[item["id"]] = item

    print("\n" + "="*70)
    print("HUMAN GOLDEN-SET LABELLING CLI TOOL")
    print("="*70)
    print(f"Total Candidates: {len(candidates)} | Already Human-Confirmed: {len(finalized_dict)}")
    print("Instructions:")
    print("  - Press Enter to accept suggested default value.")
    print("  - Type 'q' at any prompt to save progress and exit.")
    print("  - Type 'b' to jump back to the previous record.")
    print("="*70)

    final_records = list(finalized_dict.values())
    current_idx = 0

    # Skip already finalized if starting fresh
    while current_idx < len(candidates) and candidates[current_idx]["id"] in finalized_dict:
        current_idx += 1

    while current_idx < len(candidates):
        cand = candidates[current_idx]
        cid = cand["id"]

        existing = finalized_dict.get(cid, {})
        current_intent = existing.get("gold_intent", cand.get("gold_intent", cand["suggested_intent"]))
        current_action = existing.get("expected_action", cand.get("expected_action", "ESCALATE"))
        current_notes = existing.get("notes", "Human reviewed.")

        print("\n" + "-"*70)
        print(f"RECORD [{current_idx + 1}/{len(candidates)}] — ID: {cid}")
        print(f"Customer Inquiry: \"{cand['customer_message']}\"")
        print(f"Brand Response Ref: \"{cand['brand_response_reference']}\"")
        print("-" * 70)

        # 1. Gold Intent Confirmation
        print("Intent Options:")
        for i, opt in enumerate(INTENT_OPTIONS, 1):
            marker = " (SUGGESTED)" if opt == current_intent else ""
            print(f"  [{i}] {opt:<25}{marker}")

        choice = input(f"\nSelect Gold Intent [1-9] (Default: {current_intent}): ").strip().lower()
        if choice == 'q':
            print("Saving progress and exiting...")
            break
        elif choice == 'b' and current_idx > 0:
            current_idx -= 1
            continue

        selected_intent = current_intent
        if choice.isdigit() and 1 <= int(choice) <= len(INTENT_OPTIONS):
            selected_intent = INTENT_OPTIONS[int(choice) - 1]

        # 2. Expected Action Confirmation
        print(f"\nExpected Action Options:")
        print(f"  [1] AUTO_HANDLE")
        print(f"  [2] ESCALATE")
        act_input = input(f"Select Action [1/2] (Default: {current_action}): ").strip().lower()
        if act_input == 'q':
            break

        selected_action = current_action
        if act_input == '1':
            selected_action = "AUTO_HANDLE"
        elif act_input == '2':
            selected_action = "ESCALATE"

        # 3. Optional Annotation Notes
        note_input = input(f"Optional Annotation Note (Default: '{current_notes}'): ").strip()
        if note_input == 'q':
            break
        selected_notes = note_input if note_input else current_notes

        # Save record
        record = {
            "id": cid,
            "conversation_id": cand["conversation_id"],
            "customer_message": cand["customer_message"],
            "gold_intent": selected_intent,
            "expected_action": selected_action,
            "notes": selected_notes
        }
        finalized_dict[cid] = record
        print(f"✓ Confirmed {cid}: intent='{selected_intent}', action='{selected_action}'")

        # Save to file after every entry
        FINAL_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FINAL_GOLDEN_PATH, "w", encoding="utf-8") as f:
            for r in finalized_dict.values():
                f.write(json.dumps(r) + "\n")

        current_idx += 1

    print("\n" + "="*70)
    print(f"Human Labelling Session Ended.")
    print(f"Confirmed Human Records: {len(finalized_dict)} / {len(candidates)} -> {FINAL_GOLDEN_PATH}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden Set Human Labelling CLI")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive human review CLI mode")
    parser.add_argument("--sample-size", type=int, default=176, help="Number of candidate samples")
    args = parser.parse_args()

    if args.interactive:
        interactive_labelling()
    else:
        generate_candidate_golden_set(args.sample_size)
