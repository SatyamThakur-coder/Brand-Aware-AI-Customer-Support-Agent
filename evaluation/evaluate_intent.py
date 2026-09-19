"""
Intent Classifier Evaluation Harness.
Compares Majority Baseline vs TF-IDF + Logistic Regression vs LLM Classifier.
Outputs reports/intent_results.csv.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from baselines.majority_baseline import MajorityBaseline
from baselines.tfidf_logreg import TFIDFLogRegBaseline
from src.intent_classifier import LLMIntentClassifier

TEST_SPLIT_PATH = Path("data/splits/test.jsonl")
REPORTS_DIR = Path("reports")

def run_intent_evaluation(test_path: Path = TEST_SPLIT_PATH) -> pd.DataFrame:
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}. Run src.dataset_split first.")

    print(f"Loading evaluation dataset from {test_path}...")
    test_items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_items.append(json.loads(line))

    X_test = [item["customer_message"] for item in test_items]
    y_true = [item.get("silver_intent", item.get("intent", "other_general")) for item in test_items]

    results = []

    # 1. Majority Baseline
    print("\nEvaluating Baseline 1: Majority Class...")
    b1 = MajorityBaseline()
    b1.fit()
    y_pred_b1 = b1.predict(X_test)
    acc1 = accuracy_score(y_true, y_pred_b1)
    p1, r1, f1_1, _ = precision_recall_fscore_support(y_true, y_pred_b1, average='macro', zero_division=0)
    results.append({
        "model": "Majority Baseline",
        "accuracy": round(acc1, 4),
        "macro_precision": round(p1, 4),
        "macro_recall": round(r1, 4),
        "macro_f1": round(f1_1, 4)
    })

    # 2. TF-IDF + Logistic Regression Baseline
    print("\nEvaluating Baseline 2: TF-IDF + Logistic Regression...")
    b2 = TFIDFLogRegBaseline()
    b2.fit()
    y_pred_b2 = b2.predict(X_test)
    acc2 = accuracy_score(y_true, y_pred_b2)
    p2, r2, f1_2, _ = precision_recall_fscore_support(y_true, y_pred_b2, average='macro', zero_division=0)
    results.append({
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(acc2, 4),
        "macro_precision": round(p2, 4),
        "macro_recall": round(r2, 4),
        "macro_f1": round(f1_2, 4)
    })

    # 3. LLM Classifier (Mock / Active Provider)
    print("\nEvaluating Main System: LLM Intent Classifier...")
    classifier = LLMIntentClassifier(provider="mock")
    y_pred_llm = []
    for text in X_test:
        res = classifier.classify(text)
        y_pred_llm.append(res.get("intent", "other_general"))

    acc3 = accuracy_score(y_true, y_pred_llm)
    p3, r3, f1_3, _ = precision_recall_fscore_support(y_true, y_pred_llm, average='macro', zero_division=0)
    results.append({
        "model": "LLM Intent Classifier",
        "accuracy": round(acc3, 4),
        "macro_precision": round(p3, 4),
        "macro_recall": round(r3, 4),
        "macro_f1": round(f1_3, 4)
    })

    df_res = pd.DataFrame(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "intent_results.csv"
    df_res.to_csv(csv_path, index=False)

    print("\n" + "="*60)
    print("INTENT CLASSIFICATION EVALUATION RESULTS")
    print("="*60)
    print(df_res.to_string(index=False))
    print(f"\nSaved intent metrics report to {csv_path}")

    return df_res

if __name__ == "__main__":
    run_intent_evaluation()
