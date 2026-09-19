"""
Failure Analysis Module.
Analyzes pipeline error cases across 5 key failure taxonomies:
1. Ambiguous Intents
2. Poor Retrieval Quality
3. Hallucinated / Unsupported Claims
4. Incorrect Escalation (Over-escalation / Under-escalation)
5. Insufficient Historical Evidence
Generates reports/failure_analysis.md.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.pipeline import SupportPipeline
from evaluation.llm_judge import LLMJudge

TEST_SPLIT_PATH = Path("data/splits/test.jsonl")
REPORTS_DIR = Path("reports")

def run_failure_analysis(
    test_path: Path = TEST_SPLIT_PATH,
    max_samples: int = 50,
    provider: str = "mock"
) -> str:
    if not test_path.exists():
        raise FileNotFoundError(f"Test split file not found at {test_path}.")

    print(f"Running failure analysis on holdout test set ({test_path})...")
    test_items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_items.append(json.loads(line))

    samples = test_items[:max_samples]
    pipeline = SupportPipeline(provider=provider)
    judge = LLMJudge(provider=provider)

    failures = []

    for item in samples:
        msg = item["customer_message"]
        gold_intent = item.get("silver_intent", item.get("intent", "other_general"))
        # Expected action heuristic based on gold intent taxonomy rules
        expected_action = "ESCALATE" if gold_intent in ["refund_return_request", "missing_defective_item", "payment_billing_issue", "account_login_security", "cancellation_request"] else "AUTO_HANDLE"

        out = pipeline.process(msg)
        pred_intent = out["intent"]
        act_action = out["action"]
        evidence = out["retrieved_evidence"]
        top1_sim = evidence[0]["similarity_score"] if evidence else 0.0

        j_score = judge.evaluate_response(msg, out["draft_reply"], evidence, pred_intent, act_action)

        # Detect failure cases
        category = None
        explanation = ""
        hypothesis = ""

        if pred_intent != gold_intent:
            category = "Ambiguous Intents / Classification Error"
            explanation = f"Predicted intent '{pred_intent}' does not match expected intent '{gold_intent}'."
            hypothesis = "Feature overlap or keyword ambiguity in short tweet phrasing."

        elif top1_sim < 0.45:
            category = "Poor Retrieval Quality"
            explanation = f"Top retrieved evidence similarity ({top1_sim:.4f}) fell below acceptable similarity threshold."
            hypothesis = "Sparse vector representation or atypical customer vocabulary not covered in training embeddings."

        elif act_action != expected_action:
            category = "Incorrect Escalation"
            explanation = f"Actual action '{act_action}' disagreed with expected gold action '{expected_action}'."
            hypothesis = "Deterministic safety gate threshold miscalibration or conservative risk boundaries."

        elif j_score.get("groundedness", 5.0) < 3.5:
            category = "Hallucinated Policy / Low Groundedness"
            explanation = f"LLM judge scored response groundedness at {j_score['groundedness']}."
            hypothesis = "Response generator extrapolated beyond retrieved historical evidence context."

        elif not evidence or len(evidence) < 2:
            category = "Insufficient Historical Evidence"
            explanation = "Vector retrieval returned fewer than 2 relevant historical resolution cases."
            hypothesis = "Niche customer query with limited training set representation."

        if category:
            failures.append({
                "conversation_id": item["conversation_id"],
                "category": category,
                "customer_message": msg,
                "predicted_intent": pred_intent,
                "expected_intent": gold_intent,
                "retrieved_evidence": [e["brand_response"] for e in evidence[:2]],
                "generated_reply": out["draft_reply"],
                "expected_action": expected_action,
                "actual_action": act_action,
                "explanation": explanation,
                "hypothesis": hypothesis
            })

    # Generate Markdown Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "failure_analysis.md"

    md_lines = [
        "# Pipeline Failure Analysis Report",
        "",
        f"This report presents empirical analysis of **{len(failures)} identified failure cases** out of {len(samples)} evaluated test conversations.",
        "",
        "## Top 5 Failure Categories Analyzed",
        "1. **Ambiguous Intents / Classification Error**",
        "2. **Poor Retrieval Quality**",
        "3. **Incorrect Escalation (Over- or Under-escalation)**",
        "4. **Hallucinated Policy / Low Groundedness**",
        "5. **Insufficient Historical Evidence**",
        "",
        "---",
        ""
    ]

    for idx, f in enumerate(failures[:10], 1):
        md_lines.extend([
            f"### Failure Case #{idx}: {f['category']}",
            f"- **Conversation ID**: `{f['conversation_id']}`",
            f"- **Customer Message**: *\"{f['customer_message']}\"*",
            f"- **Predicted Intent**: `{f['predicted_intent']}` | **Expected Intent**: `{f['expected_intent']}`",
            f"- **Retrieved Evidence Sample**: *\"{f['retrieved_evidence'][0] if f['retrieved_evidence'] else 'None'}\"*",
            f"- **Generated Draft Reply**: *\"{f['generated_reply']}\"*",
            f"- **Actual Action**: `{f['actual_action']}` | **Expected Action**: `{f['expected_action']}`",
            f"- **Explanation**: {f['explanation']}",
            f"- **Hypothesis for Root Cause**: {f['hypothesis']}",
            "",
            "---",
            ""
        ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Generated failure analysis report with {len(failures)} cases at {report_path}")
    return str(report_path)

if __name__ == "__main__":
    run_failure_analysis()
