"""
Master Evaluation Harness.
Runs all evaluation modules and generates complete evaluation reports:
- reports/intent_results.csv
- reports/confusion_matrix.png
- reports/retrieval_results.csv
- reports/escalation_results.csv
- reports/threshold_analysis.csv
- reports/coverage_vs_quality.png
- reports/judge_agreement.csv
- reports/failure_analysis.md
"""

import sys
import argparse
from pathlib import Path

from evaluation.evaluate_intent import run_intent_evaluation
from evaluation.evaluate_replies import run_reply_and_escalation_evaluation
from evaluation.human_judge_agreement import evaluate_human_judge_agreement
from analysis.failure_analysis import run_failure_analysis

def run_all_evaluations(provider: str = "mock"):
    print("\n" + "="*70)
    print("STARTING COMPLETE EVALUATION SUITE FOR AI SUPPORT AGENT")
    print("="*70)

    # 1. Intent Classification Evaluation
    print("\n[STEP 1/4] Running Intent Classification Evaluation...")
    intent_df = run_intent_evaluation()

    # 2. Reply Quality, Escalation, and Threshold Evaluation
    print("\n[STEP 2/4] Running Reply Quality, Escalation, & Threshold Evaluation...")
    reply_res = run_reply_and_escalation_evaluation(provider=provider, max_eval_samples=100)

    # 3. Human vs LLM Judge Agreement Evaluation
    print("\n[STEP 3/4] Running Human vs LLM Judge Agreement Check...")
    judge_res = evaluate_human_judge_agreement()

    # 4. Failure Analysis
    print("\n[STEP 4/4] Running Failure Analysis...")
    failure_path = run_failure_analysis(provider=provider, max_samples=50)

    print("\n" + "="*70)
    print("EVALUATION SUITE COMPLETE — GENERATED REPORTS")
    print("="*70)
    print("  1. Intent Metrics Report:      reports/intent_results.csv")
    print("  2. Confusion Matrix Plot:      reports/confusion_matrix.png")
    print("  3. Retrieval Metrics Report:   reports/retrieval_results.csv")
    print("  4. Escalation Results Report:  reports/escalation_results.csv")
    print("  5. Threshold Analysis CSV:     reports/threshold_analysis.csv")
    print("  6. Coverage vs Quality Plot:   reports/coverage_vs_quality.png")
    print("  7. Judge Agreement Report:     reports/judge_agreement.csv")
    print("  8. Failure Analysis Report:    reports/failure_analysis.md")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete evaluation suite.")
    parser.add_argument("--provider", type=str, default="mock", help="LLM provider: openai, gemini, or mock")
    args = parser.parse_args()

    run_all_evaluations(provider=args.provider)
