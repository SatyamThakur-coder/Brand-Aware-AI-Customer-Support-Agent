"""
Grounded Response Generation Module.
Generates draft responses strictly supported by retrieved historical evidence.
Guards against inventing refund timelines, discounts, guarantees, or private transaction claims.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from src.cache import ResponseCache

load_dotenv()

class GroundedResponseGenerator:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.model = model or os.getenv("OPENAI_MODEL", os.getenv("GEMINI_MODEL", "gpt-4o-mini"))
        self.cache = ResponseCache()

    def _build_prompt(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        evidence_cases: List[Dict[str, Any]]
    ) -> str:
        evidence_str = ""
        for i, case in enumerate(evidence_cases, 1):
            evidence_str += f"""
--- HISTORICAL EVIDENCE ITEM #{i} (ID: {case.get('conversation_id', 'unknown')}) ---
Customer Inquiry: "{case.get('customer_message', '')}"
Historical Brand Reply: "{case.get('brand_response', '')}"
Similarity Score: {case.get('similarity_score', 0.0):.4f}
"""

        prompt = f"""You are a brand-aware customer support response generator for AmazonHelp.
Generate a polite, helpful draft support reply grounded ONLY in the retrieved historical evidence cases below.

CRITICAL CONSTRAINTS:
1. Treat historical replies ONLY as evidence of how similar questions were communicated, NOT as active company policy or guarantees.
2. DO NOT invent or promise:
   - Exact refund delivery dates or processing timelines (e.g., "3-5 business days").
   - Financial compensation, promo codes, or discounts.
   - Private account, order status, or transaction confirmations that require real-time system lookup.
3. If the historical evidence does not provide a safe public procedure, draft a minimal helpful response stating that account verification is required.

CUSTOMER MESSAGE:
"{customer_message}"

PREDICTED INTENT: {predicted_intent} (Confidence: {confidence:.2f})

RETRIEVED EVIDENCE:
{evidence_str}

Respond strictly in valid JSON format:
{{
  "draft_reply": "<your_grounded_response_text>",
  "supported_by_evidence": <true_or_false>,
  "evidence_ids_used": ["<conv_id_1>", "<conv_id_2>"],
  "reasoning": "<brief explanation of evidence grounding>"
}}
"""
        return prompt

    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        evidence_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not customer_message.strip() or predicted_intent == "UNKNOWN":
            return {
                "draft_reply": "Thank you for reaching out. To assist you properly, could you please provide more details regarding your request?",
                "supported_by_evidence": False,
                "evidence_ids_used": [],
                "reasoning": "Message unclear or unknown intent."
            }

        prompt = self._build_prompt(customer_message, predicted_intent, confidence, evidence_cases)

        # Check Cache
        cached = self.cache.get(prompt, self.model)
        if cached:
            return cached

        # Deterministic Mock Provider for offline testing & smoke test
        if self.provider == "mock" or os.getenv("MOCK_MODE", "false").lower() == "true":
            # Extract top evidence brand response if available
            top_reply = evidence_cases[0]["brand_response"] if evidence_cases else "Please reach out to our support team with your order details so we can assist."
            evidence_ids = [c["conversation_id"] for c in evidence_cases[:2]] if evidence_cases else []
            
            result = {
                "draft_reply": f"Hello! {top_reply}",
                "supported_by_evidence": len(evidence_cases) > 0,
                "evidence_ids_used": evidence_ids,
                "reasoning": "Generated grounded draft based on top historical resolution pattern."
            }
            self.cache.set(prompt, "mock", result)
            return result

        # OpenAI Provider
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return self._mock_fallback(evidence_cases)
            try:
                import requests
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
                res.raise_for_status()
                result = json.loads(res.json()["choices"][0]["message"]["content"])
                self.cache.set(prompt, self.model, result)
                return result
            except Exception as e:
                print(f"OpenAI API failed ({e}). Using mock response generator.")
                return self._mock_fallback(evidence_cases)

        # Gemini Provider
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return self._mock_fallback(evidence_cases)
            try:
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"}
                }
                res = requests.post(url, json=payload, timeout=15)
                res.raise_for_status()
                result = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
                self.cache.set(prompt, self.model, result)
                return result
            except Exception as e:
                print(f"Gemini API failed ({e}). Using mock response generator.")
                return self._mock_fallback(evidence_cases)

        return self._mock_fallback(evidence_cases)

    def _mock_fallback(self, evidence_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        top_reply = evidence_cases[0]["brand_response"] if evidence_cases else "Please contact Amazon customer support with your order details."
        evidence_ids = [c["conversation_id"] for c in evidence_cases[:2]] if evidence_cases else []
        return {
            "draft_reply": f"Hello! {top_reply}",
            "supported_by_evidence": len(evidence_cases) > 0,
            "evidence_ids_used": evidence_ids,
            "reasoning": "Fallback grounded draft using top retrieved evidence item."
        }

if __name__ == "__main__":
    gen = GroundedResponseGenerator(provider="mock")
    sample_evidence = [{
        "conversation_id": "conv_101",
        "customer_message": "Where is my refund?",
        "brand_response": "Please check your online account refund tracker or contact us with your order number.",
        "similarity_score": 0.82
    }]
    res = gen.generate("Where is my refund?", "refund_return_request", 0.90, sample_evidence)
    print("Generated Response Output:")
    print(json.dumps(res, indent=2))
