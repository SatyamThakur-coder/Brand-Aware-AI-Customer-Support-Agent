"""
Baseline 2: TF-IDF + Logistic Regression Classifier.
Supervised text classification baseline for intent detection.
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

REPORTS_DIR = Path("reports")

class TFIDFLogRegBaseline:
    def __init__(self, max_features: int = 5000, C: float = 1.0):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('clf', LogisticRegression(C=C, max_iter=1000, class_weight='balanced'))
        ])

    def fit(self, train_path: str = "data/splits/train.jsonl"):
        train_items = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_items.append(json.loads(line))

        X_train = [item["customer_message"] for item in train_items]
        y_train = [item.get("silver_intent", "other_general") for item in train_items]

        print(f"Training TF-IDF + Logistic Regression baseline on {len(X_train):,} examples...")
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def predict(self, texts: List[str]) -> List[str]:
        return list(self.model.predict(texts))

    def evaluate(self, test_path: str = "data/splits/test.jsonl", save_confusion_matrix: bool = True) -> Dict[str, Any]:
        test_items = []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_items.append(json.loads(line))

        X_test = [item["customer_message"] for item in test_items]
        y_true = [item.get("silver_intent", item.get("intent", "other_general")) for item in test_items]
        y_pred = self.predict(X_test)

        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        print("\n--- TF-IDF + LOGISTIC REGRESSION BASELINE METRICS ---")
        print(f"Accuracy:        {acc:.4f}")
        print(f"Macro Precision: {p_macro:.4f}")
        print(f"Macro Recall:    {r_macro:.4f}")
        print(f"Macro F1:        {f1_macro:.4f}")

        if save_confusion_matrix:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            labels = sorted(list(set(y_true).union(set(y_pred))))
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title("TF-IDF + Logistic Regression Confusion Matrix")
            plt.xlabel("Predicted Intent")
            plt.ylabel("True Intent")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            cm_path = REPORTS_DIR / "confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            print(f"Saved confusion matrix plot to {cm_path}")

        return {
            "accuracy": float(acc),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "classification_report": report
        }

if __name__ == "__main__":
    baseline = TFIDFLogRegBaseline()
    baseline.fit()
    baseline.evaluate()
