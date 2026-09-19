"""
Smoke Test Verification Script.
Runs end-to-end software architecture verification offline using the Mock provider.
Ensures preprocessing, intent classification, vector retrieval, response generation, escalation safety, and unified pipeline execute cleanly without errors or API keys.
"""

import sys
import os
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force mock mode
os.environ["LLM_PROVIDER"] = "mock"
os.environ["MOCK_MODE"] = "true"

def run_smoke_test():
    print("="*60)
    print("RUNNING OFFLINE SMOKE TEST (MOCK PROVIDER MODE)")
    print("="*60)

    # 1. Preprocessing Verification
    print("\n[1/6] Testing Text Cleaning & Preprocessing...")
    from src.preprocessing import clean_tweet_text
    raw = "Hey @AmazonHelp my order https://t.co/abc arrived broken &amp; shattered!"
    cleaned = clean_tweet_text(raw, brand_name="AmazonHelp")
    assert "[BRAND]" in cleaned
    assert "[URL]" in cleaned
    assert "&amp;" not in cleaned
    print("  [OK] Preprocessing test passed.")

    # 2. Intent Classifier Interface
    print("\n[2/6] Testing Intent Classifier Interface...")
    from src.intent_classifier import LLMIntentClassifier
    classifier = LLMIntentClassifier(provider="mock")
    cls_res = classifier.classify("Where is my refund for order #102?")
    assert "intent" in cls_res
    assert "confidence" in cls_res
    assert cls_res["intent"] == "refund_return_request"
    print(f"  [OK] Classifier test passed (Intent: {cls_res['intent']}, Confidence: {cls_res['confidence']}).")

    # 3. Vector Retrieval Verification
    print("\n[3/6] Testing Historical Vector Retrieval...")
    from src.retrieval import HistoricalCaseRetriever
    retriever = HistoricalCaseRetriever(top_k=3)
    retriever.build_index()
    evidence = retriever.retrieve("Where is my refund?", predicted_intent="refund_return_request")
    assert len(evidence) > 0
    assert "similarity_score" in evidence[0]
    print(f"  [OK] Retrieval test passed (Retrieved {len(evidence)} cases, Top-1 sim: {evidence[0]['similarity_score']:.4f}).")

    # 4. Response Generator Interface
    print("\n[4/6] Testing Grounded Response Generator...")
    from src.response_generator import GroundedResponseGenerator
    generator = GroundedResponseGenerator(provider="mock")
    gen_res = generator.generate("Where is my refund?", "refund_return_request", 0.90, evidence)
    assert "draft_reply" in gen_res
    assert "supported_by_evidence" in gen_res
    print("  [OK] Response Generator test passed.")

    # 5. Deterministic Escalation Safety Engine
    print("\n[5/6] Testing Escalation Safety Policy Engine...")
    from src.escalation import EscalationEngine
    engine = EscalationEngine()
    esc_res = engine.evaluate_escalation(
        customer_message="Where is my refund?",
        predicted_intent="refund_return_request",
        confidence=0.95,
        evidence_cases=evidence,
        generation_result=gen_res
    )
    assert "action" in esc_res
    assert esc_res["action"] == "ESCALATE" # Mandatory escalation intent
    print(f"  [OK] Escalation Engine test passed (Decision: {esc_res['action']}).")

    # 6. Unified Pipeline Verification
    print("\n[6/6] Testing Unified Support Pipeline...")
    from src.pipeline import SupportPipeline
    pipeline = SupportPipeline(provider="mock")
    out = pipeline.process("Can I change my delivery address for order #405?")
    assert out["action"] in ["AUTO_HANDLE", "ESCALATE"]
    assert len(out["retrieved_evidence"]) > 0
    print("  [OK] Unified Pipeline test passed.")

    print("\n" + "="*60)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY! (SYSTEM IS HEALTHY)")
    print("="*60)

if __name__ == "__main__":
    run_smoke_test()
