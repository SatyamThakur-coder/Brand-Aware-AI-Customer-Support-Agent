"""
Reply Quality, Escalation, Coverage, and Threshold Evaluation Engine.
Evaluates end-to-end pipeline outputs, measures safety policy decisions, performs confidence threshold analysis,
and generates reports/escalation_results.csv, reports/threshold_analysis.csv, reports/retrieval_results.csv, and reports/coverage_vs_quality.png.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

from src.pipeline import SupportPipeline
from evaluation.llm_judge import LLMJudge
from src.retrieval import HistoricalCaseRetriever

TEST_SPLIT_PATH = Path("data/splits/test.jsonl")
REPORTS_DIR = Path("reports")

def run_reply_and_escalation_evaluation(
    test_path: Path = TEST_SPLIT_PATH,
    max_eval_samples: int = 100,
    provider: str = "mock"
) -> Dict[str, Any]:
    if not test_path.exists():
        raise FileNotFoundError(f"Test split file not found at {test_path}. Run src.dataset_split first.")

    print(f"Loading test split for reply and escalation evaluation: {test_path}...")
    test_items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_items.append(json.loads(line))

    eval_samples = test_items[:max_eval_samples]
    print(f"Running evaluation pipeline on {len(eval_samples)} test customer messages...")

    pipeline = SupportPipeline(provider=provider)
    judge = LLMJudge(provider=provider)
    retriever = pipeline.retriever

    pipeline_outputs = []
    judge_scores = []
    retrieval_metrics_list = []

    for i, item in enumerate(eval_samples, 1):
        msg = item["customer_message"]
        gold_intent = item.get("silver_intent", item.get("intent", "other_general"))

        # Pipeline execution
        out = pipeline.process(msg)
        pipeline_outputs.append(out)

        # Retrieval evaluation
        evidence = out["retrieved_evidence"]
        sims = [e["similarity_score"] for e in evidence] if evidence else [0.0]
        top1_sim = sims[0]
        top3_sim = np.mean(sims[:min(3, len(sims))])
        top5_sim = np.mean(sims)
        top1_intent_hit = 1 if (evidence and evidence[0].get("silver_intent") == gold_intent) else 0

        retrieval_metrics_list.append({
            "conversation_id": item["conversation_id"],
            "gold_intent": gold_intent,
            "top1_similarity": round(top1_sim, 4),
            "top3_avg_similarity": round(top3_sim, 4),
            "top5_avg_similarity": round(top5_sim, 4),
            "top1_intent_match": top1_intent_hit
        })

        # LLM Judge evaluation
        j_score = judge.evaluate_response(
            customer_message=msg,
            generated_reply=out["draft_reply"],
            evidence_cases=evidence,
            predicted_intent=out["intent"],
            action=out["action"]
        )
        judge_scores.append(j_score)

    # 1. Retrieval Quality Metrics & CSV
    df_retrieval = pd.DataFrame(retrieval_metrics_list)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    retrieval_csv = REPORTS_DIR / "retrieval_results.csv"
    df_retrieval.to_csv(retrieval_csv, index=False)
    print(f"Saved retrieval quality metrics to {retrieval_csv}")

    # 2. Escalation & Coverage Statistics
    total_eval = len(pipeline_outputs)
    auto_handled_count = sum(1 for o in pipeline_outputs if o["action"] == "AUTO_HANDLE")
    escalated_count = sum(1 for o in pipeline_outputs if o["action"] == "ESCALATE")

    coverage_pct = round((auto_handled_count / total_eval) * 100.0, 2)
    escalation_pct = round((escalated_count / total_eval) * 100.0, 2)

    avg_groundedness = round(np.mean([s["groundedness"] for s in judge_scores]), 2)
    avg_helpfulness = round(np.mean([s["helpfulness"] for s in judge_scores]), 2)
    avg_relevance = round(np.mean([s["relevance"] for s in judge_scores]), 2)
    avg_brand_consist = round(np.mean([s["brand_consistency"] for s in judge_scores]), 2)
    avg_unsupported_claim = round(np.mean([s["unsupported_claims"] for s in judge_scores]), 2)
    avg_esc_appr = round(np.mean([s["escalation_appropriateness"] for s in judge_scores]), 2)
    avg_overall_judge = round(np.mean([s["overall"] for s in judge_scores]), 2)

    df_esc = pd.DataFrame([{
        "total_evaluated": total_eval,
        "auto_handled_count": auto_handled_count,
        "escalated_count": escalated_count,
        "auto_handling_coverage_pct": coverage_pct,
        "escalation_rate_pct": escalation_pct,
        "judge_avg_groundedness": avg_groundedness,
        "judge_avg_helpfulness": avg_helpfulness,
        "judge_avg_relevance": avg_relevance,
        "judge_avg_brand_consistency": avg_brand_consist,
        "judge_avg_unsupported_claim_score": avg_unsupported_claim,
        "judge_avg_escalation_appropriateness": avg_esc_appr,
        "judge_overall_quality_score": avg_overall_judge
    }])
    esc_csv = REPORTS_DIR / "escalation_results.csv"
    df_esc.to_csv(esc_csv, index=False)
    print(f"Saved escalation results report to {esc_csv}")

    # 3. Confidence Threshold Analysis (0.60 to 0.95)
    thresholds = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    threshold_results = []

    for t in thresholds:
        test_engine = SupportPipeline(provider=provider, confidence_threshold=t)
        t_outputs = [test_engine.process(item["customer_message"]) for item in eval_samples]
        t_auto = sum(1 for o in t_outputs if o["action"] == "AUTO_HANDLE")
        t_esc = sum(1 for o in t_outputs if o["action"] == "ESCALATE")
        t_cov = round((t_auto / len(t_outputs)) * 100.0, 2)
        t_esc_rate = round((t_esc / len(t_outputs)) * 100.0, 2)

        threshold_results.append({
            "confidence_threshold": t,
            "auto_handling_coverage_pct": t_cov,
            "escalation_rate_pct": t_esc_rate,
            "avg_judge_overall_quality": avg_overall_judge
        })

    df_thresh = pd.DataFrame(threshold_results)
    thresh_csv = REPORTS_DIR / "threshold_analysis.csv"
    df_thresh.to_csv(thresh_csv, index=False)
    print(f"Saved threshold experiment analysis to {thresh_csv}")

    # 4. Plot Coverage vs Quality
    plt.figure(figsize=(8, 5))
    plt.plot(df_thresh["confidence_threshold"], df_thresh["auto_handling_coverage_pct"], marker='o', label="Auto-handling Coverage %", color="teal")
    plt.plot(df_thresh["confidence_threshold"], df_thresh["escalation_rate_pct"], marker='s', label="Escalation Rate %", color="darkred")
    plt.title("Auto-handling Coverage vs Escalation Rate across Confidence Thresholds")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Percentage (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_path = REPORTS_DIR / "coverage_vs_quality.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved coverage vs quality visualization to {plot_path}")

    print("\n" + "="*60)
    print("REPLY & ESCALATION EVALUATION SUMMARY")
    print("="*60)
    print(df_esc.to_string(index=False))

    return {
        "escalation_results": df_esc.to_dict(orient="records")[0],
        "threshold_analysis": df_thresh.to_dict(orient="records"),
        "retrieval_results": df_retrieval.head(5).to_dict(orient="records")
    }

if __name__ == "__main__":
    run_reply_and_escalation_evaluation()
