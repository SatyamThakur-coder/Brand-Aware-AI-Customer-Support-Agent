"""
Comprehensive Pytest Test Suite for Brand-Aware AI Support Agent.
Tests preprocessing, intent classification, vector retrieval, grounded response generation,
escalation safety rules, fallback handling, and unified pipeline execution.
"""

import pytest
import os
import json
import pandas as pd
from pathlib import Path

from src.preprocessing import clean_tweet_text, reconstruct_conversations
from src.intent_taxonomy import load_intent_taxonomy
from src.intent_classifier import LLMIntentClassifier
from src.retrieval import HistoricalCaseRetriever
from src.response_generator import GroundedResponseGenerator
from src.escalation import EscalationEngine
from src.pipeline import SupportPipeline

# Force Mock Mode for fast offline unit tests
os.environ["LLM_PROVIDER"] = "mock"
os.environ["MOCK_MODE"] = "true"

def test_data_cleaning():
    raw_text = "Hey @AmazonHelp my refund for order https://t.co/xyz123 hasn't arrived &amp; items were shattered! @user2"
    cleaned = clean_tweet_text(raw_text, brand_name="AmazonHelp")
    assert "[BRAND]" in cleaned
    assert "[URL]" in cleaned
    assert "[USER]" in cleaned
    assert "&amp;" not in cleaned
    assert "https://" not in cleaned

def test_intent_taxonomy_loading():
    taxonomy = load_intent_taxonomy()
    assert "intents" in taxonomy
    assert "refund_return_request" in taxonomy["intents"]
    assert "delivery_issue" in taxonomy["intents"]

def test_intent_classifier_output_schema():
    classifier = LLMIntentClassifier(provider="mock")
    res = classifier.classify("Where is my package?")
    assert "intent" in res
    assert "confidence" in res
    assert "reasoning" in res
    assert 0.0 <= res["confidence"] <= 1.0

def test_intent_classifier_empty_and_low_confidence():
    classifier = LLMIntentClassifier(provider="mock")
    res_empty = classifier.classify("")
    assert res_empty["intent"] == "UNKNOWN"
    assert res_empty["confidence"] == 0.0

def test_retrieval_index_and_search():
    retriever = HistoricalCaseRetriever(top_k=3)
    retriever.build_index()
    results = retriever.retrieve("Where is my refund for order #102?")
    assert len(results) > 0
    assert "conversation_id" in results[0]
    assert "similarity_score" in results[0]

def test_response_generator_mock():
    gen = GroundedResponseGenerator(provider="mock")
    evidence = [{
        "conversation_id": "conv_101",
        "customer_message": "Where is my refund?",
        "brand_response": "Please check your account refund tracker.",
        "similarity_score": 0.85
    }]
    res = gen.generate("Where is my refund?", "refund_return_request", 0.90, evidence)
    assert "draft_reply" in res
    assert "supported_by_evidence" in res
    assert res["supported_by_evidence"] is True

def test_escalation_mandatory_intent():
    engine = EscalationEngine()
    esc = engine.evaluate_escalation(
        customer_message="Where is my refund?",
        predicted_intent="refund_return_request",
        confidence=0.95,
        evidence_cases=[{"similarity_score": 0.88}],
        generation_result={"supported_by_evidence": True}
    )
    assert esc["action"] == "ESCALATE"
    assert "refund_return_request" in esc["reason"]

def test_escalation_low_confidence():
    engine = EscalationEngine(confidence_threshold=0.80)
    esc = engine.evaluate_escalation(
        customer_message="What is the weather?",
        predicted_intent="information_request",
        confidence=0.60, # Below threshold
        evidence_cases=[{"similarity_score": 0.70}],
        generation_result={"supported_by_evidence": True}
    )
    assert esc["action"] == "ESCALATE"
    assert "below safe threshold" in esc["reason"]

def test_escalation_missing_evidence():
    engine = EscalationEngine()
    esc = engine.evaluate_escalation(
        customer_message="Some rare inquiry",
        predicted_intent="information_request",
        confidence=0.90,
        evidence_cases=[], # Missing evidence
        generation_result={"supported_by_evidence": False}
    )
    assert esc["action"] == "ESCALATE"
    assert "No historical resolution evidence" in esc["reason"]

def test_full_pipeline_execution():
    pipeline = SupportPipeline(provider="mock")
    out = pipeline.process("Can I check store warranty policy?")
    assert "customer_message" in out
    assert "intent" in out
    assert "action" in out
    assert "draft_reply" in out
    assert out["action"] in ["AUTO_HANDLE", "ESCALATE"]
