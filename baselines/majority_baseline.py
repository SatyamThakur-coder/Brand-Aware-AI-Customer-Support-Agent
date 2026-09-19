"""
Baseline 1: Majority Class Classifier.
Always predicts the most common intent present in the training set.
"""

import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

class MajorityBaseline:
    def __init__(self):
        self.majority_intent = None

    def fit(self, train_path: str = "data/splits/train.jsonl"):
        intents = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    intents.append(item.get("silver_intent", "delivery_issue"))
                    
        counts = Counter(intents)
        self.majority_intent = counts.most_common(1)[0][0]
        print(f"MajorityBaseline trained. Most common class: '{self.majority_intent}' ({counts[self.majority_intent]} occurrences)")

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_intent] * len(texts)

    def evaluate(self, test_path: str = "data/splits/test.jsonl") -> Dict[str, float]:
        test_items = []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_items.append(json.loads(line))

        y_true = [item.get("silver_intent", item.get("intent", "other_general")) for item in test_items]
        y_pred = self.predict([item["customer_message"] for item in test_items])

        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

        metrics = {
            "accuracy": float(acc),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro)
        }

        print("\n--- MAJORITY BASELINE METRICS ---")
        print(f"Accuracy:        {acc:.4f}")
        print(f"Macro Precision: {p_macro:.4f}")
        print(f"Macro Recall:    {r_macro:.4f}")
        print(f"Macro F1:        {f1_macro:.4f}")

        return metrics

if __name__ == "__main__":
    baseline = MajorityBaseline()
    baseline.fit()
    baseline.evaluate()
