"""
Unified Customer Support AI Pipeline.
Orchestrates Preprocessing -> Intent Classification -> Vector Retrieval -> Grounded Generation -> Escalation Engine.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.preprocessing import clean_tweet_text
from src.intent_classifier import LLMIntentClassifier
from src.retrieval import HistoricalCaseRetriever
from src.response_generator import GroundedResponseGenerator
from src.escalation import EscalationEngine

class SupportPipeline:
    def __init__(
        self,
        brand_name: str = "AmazonHelp",
        provider: str = None,
        top_k: int = 5,
        confidence_threshold: float = 0.80,
        min_retrieval_similarity: float = 0.45
    ):
        self.brand_name = brand_name
        self.classifier = LLMIntentClassifier(provider=provider)
        self.retriever = HistoricalCaseRetriever(top_k=top_k)
        self.generator = GroundedResponseGenerator(provider=provider)
        self.escalation_engine = EscalationEngine(
            confidence_threshold=confidence_threshold,
            min_retrieval_similarity=min_retrieval_similarity
        )
        # Ensure index is ready
        self.retriever.build_index()

    def process(self, raw_message: str) -> Dict[str, Any]:
        cleaned_msg = clean_tweet_text(raw_message, brand_name=self.brand_name)
        if not cleaned_msg:
            cleaned_msg = raw_message

        # Step 1: Intent Classification
        cls_res = self.classifier.classify(cleaned_msg)
        intent = cls_res.get("intent", "UNKNOWN")
        confidence = float(cls_res.get("confidence", 0.0))

        # Step 2: Historical Case Retrieval
        evidence_cases = self.retriever.retrieve(cleaned_msg, predicted_intent=intent)

        # Step 3: Grounded Response Generation
        gen_res = self.generator.generate(cleaned_msg, intent, confidence, evidence_cases)

        # Step 4: Deterministic Escalation Safety Rules
        esc_res = self.escalation_engine.evaluate_escalation(
            customer_message=cleaned_msg,
            predicted_intent=intent,
            confidence=confidence,
            evidence_cases=evidence_cases,
            generation_result=gen_res
        )

        return {
            "customer_message": raw_message,
            "cleaned_message": cleaned_msg,
            "intent": intent,
            "confidence": confidence,
            "classifier_reasoning": cls_res.get("reasoning", ""),
            "retrieved_evidence": evidence_cases,
            "draft_reply": gen_res.get("draft_reply", ""),
            "generation_reasoning": gen_res.get("reasoning", ""),
            "action": esc_res["action"],
            "escalation_reason": esc_res["reason"],
            "evidence_ids": [c["conversation_id"] for c in evidence_cases]
        }

    def format_cli_output(self, result: Dict[str, Any]) -> str:
        evidence_list = "\n".join([f"  - {c['conversation_id']} (similarity: {c['similarity_score']:.4f})" for c in result["retrieved_evidence"][:3]])
        
        output = f"""
============================================================
AI CUSTOMER SUPPORT AGENT PIPELINE RESULT
============================================================

## CUSTOMER MESSAGE
"{result['customer_message']}"

## INTENT CLASSIFICATION
Intent:     {result['intent']}
Confidence: {result['confidence']:.2f}
Reasoning:  {result['classifier_reasoning']}

## HISTORICAL RETRIEVAL EVIDENCE
{evidence_list if evidence_list else '  None'}

## DRAFT RESPONSE
{result['draft_reply']}

## DECISION & ACTION
Action: {result['action']}
Reason: {result['escalation_reason']}
============================================================
"""
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Customer Support Pipeline on customer message.")
    parser.add_argument("--message", type=str, default="My refund for order #102 has not arrived yet", help="Customer message string")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: openai, gemini, or mock")
    args = parser.parse_args()

    pipeline = SupportPipeline(provider=args.provider)
    result = pipeline.process(args.message)
    print(pipeline.format_cli_output(result))
