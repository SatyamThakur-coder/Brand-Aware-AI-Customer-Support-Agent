"""
Human vs LLM Judge Agreement Module.
Measures statistical agreement (Agreement %, MAE, Pearson Correlation) between human ratings and LLM Judge.
If human_ratings.csv is missing or pending, explicitly reports "Human judge validation pending."
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

HUMAN_RATINGS_PATH = Path("evaluation/human_ratings.csv")
REPORTS_DIR = Path("reports")

def create_human_ratings_template(sample_pipeline_results: list = None) -> Path:
    """
    Creates evaluation/human_ratings.csv template for human annotators.
    """
    HUMAN_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HUMAN_RATINGS_PATH.exists():
        columns = [
            "id", "customer_message", "generated_reply",
            "human_groundedness", "human_helpfulness", "human_relevance", "human_overall",
            "llm_groundedness", "llm_helpfulness", "llm_relevance", "llm_overall", "status"
        ]
        df_empty = pd.DataFrame(columns=columns)
        df_empty.to_csv(HUMAN_RATINGS_PATH, index=False)
        print(f"Created human ratings template at {HUMAN_RATINGS_PATH}")
    return HUMAN_RATINGS_PATH

def evaluate_human_judge_agreement() -> Dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "judge_agreement.csv"

    create_human_ratings_template()

    if not HUMAN_RATINGS_PATH.exists() or os.path.getsize(HUMAN_RATINGS_PATH) == 0:
        msg = "Human judge validation pending."
        print(f"\n--- HUMAN VS LLM JUDGE AGREEMENT ---")
        print(msg)
        df_res = pd.DataFrame([{"status": msg, "agreement_rate": None, "mae": None, "correlation": None}])
        df_res.to_csv(out_csv, index=False)
        return {"status": msg}

    df = pd.read_csv(HUMAN_RATINGS_PATH)

    # Check if human ratings are populated
    valid_rows = df.dropna(subset=["human_overall", "llm_overall"])
    if len(valid_rows) < 5:
        msg = "Human judge validation pending."
        print(f"\n--- HUMAN VS LLM JUDGE AGREEMENT ---")
        print(f"Only {len(valid_rows)} human ratings found. {msg}")
        df_res = pd.DataFrame([{"status": msg, "agreement_rate": None, "mae": None, "correlation": None}])
        df_res.to_csv(out_csv, index=False)
        return {"status": msg}

    # Compute metrics
    human_scores = valid_rows["human_overall"].astype(float).values
    llm_scores = valid_rows["llm_overall"].astype(float).values

    abs_diffs = np.abs(human_scores - llm_scores)
    mae = float(np.mean(abs_diffs))
    # Agreement rate: percentage of pairs within 1.0 point difference
    agreement_rate = float(np.mean(abs_diffs <= 1.0) * 100.0)

    # Pearson correlation
    corr = float(np.corrcoef(human_scores, llm_scores)[0, 1]) if len(human_scores) > 1 else 1.0

    print(f"\n--- HUMAN VS LLM JUDGE AGREEMENT METRICS ---")
    print(f"Sample Size:         {len(valid_rows)} pairs")
    print(f"Agreement Rate (<=1pt): {agreement_rate:.1f}%")
    print(f"Mean Absolute Error:    {mae:.4f}")
    print(f"Pearson Correlation:    {corr:.4f}")

    df_res = pd.DataFrame([{
        "status": "VALIDATED",
        "sample_size": len(valid_rows),
        "agreement_rate_pct": round(agreement_rate, 2),
        "mean_absolute_error": round(mae, 4),
        "pearson_correlation": round(corr, 4)
    }])
    df_res.to_csv(out_csv, index=False)
    print(f"Saved judge agreement metrics to {out_csv}")

    return {
        "status": "VALIDATED",
        "agreement_rate_pct": agreement_rate,
        "mae": mae,
        "correlation": corr
    }

if __name__ == "__main__":
    evaluate_human_judge_agreement()
