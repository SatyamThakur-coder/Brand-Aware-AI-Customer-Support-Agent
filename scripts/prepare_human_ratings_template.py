"""
Human Ratings Template Generator.
Populates evaluation/human_ratings.csv with 50 sample generated response pairs from holdout test items.
Leaves human rating columns blank for manual human evaluation.
"""

import json
import pandas as pd
from pathlib import Path

from src.pipeline import SupportPipeline
from evaluation.llm_judge import LLMJudge

TEST_SPLIT_PATH = Path("data/splits/test.jsonl")
HUMAN_RATINGS_PATH = Path("evaluation/human_ratings.csv")

def prepare_human_ratings_template(num_samples: int = 50):
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Test split not found at {TEST_SPLIT_PATH}.")

    print(f"Generating 50 evaluation response pairs for {HUMAN_RATINGS_PATH}...")
    pipeline = SupportPipeline(provider="mock")
    judge = LLMJudge(provider="mock")

    test_items = [json.loads(l) for l in open(TEST_SPLIT_PATH, "r", encoding="utf-8") if l.strip()]
    samples = test_items[:num_samples]

    rows = []
    for i, item in enumerate(samples, 1):
        msg = item["customer_message"]
        out = pipeline.process(msg)
        j_score = judge.evaluate_response(
            customer_message=msg,
            generated_reply=out["draft_reply"],
            evidence_cases=out["retrieved_evidence"],
            predicted_intent=out["intent"],
            action=out["action"]
        )

        rows.append({
            "id": f"human_eval_{i:03d}",
            "conversation_id": item["conversation_id"],
            "customer_message": msg,
            "predicted_intent": out["intent"],
            "action": out["action"],
            "generated_reply": out["draft_reply"],
            # Leave human ratings empty for real human entry
            "human_groundedness": None,
            "human_helpfulness": None,
            "human_correctness": None,
            "human_relevance": None,
            "human_brand_consistency": None,
            "human_unsupported_claims": None,
            "human_escalation_appropriateness": None,
            "human_overall": None,
            # Automated LLM Judge ratings for comparison
            "llm_groundedness": j_score["groundedness"],
            "llm_helpfulness": j_score["helpfulness"],
            "llm_relevance": j_score["relevance"],
            "llm_overall": j_score["overall"],
            "status": "PENDING_HUMAN_RATING"
        })

    df = pd.DataFrame(rows)
    HUMAN_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HUMAN_RATINGS_PATH, index=False)
    print(f"Successfully generated {len(df)} response pairs in {HUMAN_RATINGS_PATH}.")
    print("INSTRUCTIONS FOR HUMAN EVALUATORS:")
    print("  1. Open evaluation/human_ratings.csv in Excel/CSV editor.")
    print("  2. Enter scores (1.0 to 5.0) for human_* columns.")
    print("  3. Set status='FINALIZED' for completed rows.")
    print("  4. Run `python evaluation/human_judge_agreement.py` to calculate Agreement %, MAE, and Pearson Correlation.")

if __name__ == "__main__":
    prepare_human_ratings_template()
