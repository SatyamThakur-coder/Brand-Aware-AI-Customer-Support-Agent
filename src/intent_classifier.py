"""
Intent Classifier Module with Provider Abstraction (OpenAI, Gemini, Mock).
Classifies customer messages into brand-specific intents with confidence scoring and unknown state support.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

from src.intent_taxonomy import load_intent_taxonomy
from src.cache import ResponseCache

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()

class MockLLMClassifier:
    """
    Deterministic rule-based mock classifier for offline testing, CI, and smoke tests.
    """
    def __init__(self, taxonomy: Dict[str, Any]):
        self.taxonomy = taxonomy["intents"]
        from src.dataset_split import INTENT_KEYWORDS
        self.keywords = INTENT_KEYWORDS

    def classify(self, message: str) -> Dict[str, Any]:
        lower = message.lower()
        if not message.strip() or len(message.strip()) < 5:
            return {
                "intent": "UNKNOWN",
                "confidence": 0.20,
                "reasoning": "Message too short or empty."
            }

        matched_intents = []
        for intent, patterns in self.keywords.items():
            for pat in patterns:
                if re.search(pat, lower):
                    matched_intents.append(intent)
                    break

        if not matched_intents:
            return {
                "intent": "other_general",
                "confidence": 0.65,
                "reasoning": "No specific category keywords matched; classified as general."
            }

        if len(matched_intents) == 1:
            return {
                "intent": matched_intents[0],
                "confidence": 0.92,
                "reasoning": f"Matched strong keyword pattern for {matched_intents[0]}."
            }

        # Multiple matches -> return top match with slightly lower confidence
        return {
            "intent": matched_intents[0],
            "confidence": 0.78,
            "reasoning": f"Multiple category signals ({', '.join(matched_intents)}); selected primary."
        }

class LLMIntentClassifier:
    def __init__(self, provider: str = None, model: str = None):
        self.taxonomy = load_intent_taxonomy()
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.model = model or os.getenv("OPENAI_MODEL", os.getenv("GEMINI_MODEL", "gpt-4o-mini"))
        self.cache = ResponseCache()
        self.mock_classifier = MockLLMClassifier(self.taxonomy)

    def _build_prompt(self, customer_message: str) -> str:
        intent_descriptions = []
        for name, info in self.taxonomy["intents"].items():
            intent_descriptions.append(f"- {name}: {info['description']}")
        
        prompt = f"""You are an expert customer support intent classifier for AmazonHelp.
Classify the following customer message into EXACTLY ONE of the supported intents below, or UNKNOWN if ambiguous/unclear.

SUPPORTED INTENTS:
{chr(10).join(intent_descriptions)}
- UNKNOWN: Use if the message is too ambiguous, unreadable, or missing necessary context.

CUSTOMER MESSAGE:
"{customer_message}"

Respond strictly in valid JSON format with no markdown wrappers:
{{
  "intent": "<intent_name_or_UNKNOWN>",
  "confidence": <float between 0.00 and 1.00>,
  "reasoning": "<brief explanation>"
}}
"""
        return prompt

    def classify(self, customer_message: str) -> Dict[str, Any]:
        if not customer_message.strip():
            return {
                "intent": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "Empty customer message."
            }

        prompt = self._build_prompt(customer_message)

        # Check Cache
        cached = self.cache.get(prompt, self.model)
        if cached:
            return cached

        # Mock Provider Fallback
        if self.provider == "mock" or os.getenv("MOCK_MODE", "false").lower() == "true":
            result = self.mock_classifier.classify(customer_message)
            self.cache.set(prompt, "mock", result)
            return result

        # OpenAI Provider
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY missing. Falling back to deterministic Mock provider.")
                return self.mock_classifier.classify(customer_message)

            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)
                self.cache.set(prompt, self.model, result)
                return result
            except Exception as e:
                print(f"OpenAI API call failed ({e}). Using Mock provider fallback.")
                return self.mock_classifier.classify(customer_message)

        # Google Gemini Provider
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Warning: GEMINI_API_KEY missing. Falling back to deterministic Mock provider.")
                return self.mock_classifier.classify(customer_message)

            try:
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"}
                }
                res = requests.post(url, json=payload, timeout=15)
                res.raise_for_status()
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text)
                self.cache.set(prompt, self.model, result)
                return result
            except Exception as e:
                print(f"Gemini API call failed ({e}). Using Mock provider fallback.")
                return self.mock_classifier.classify(customer_message)

        return self.mock_classifier.classify(customer_message)

if __name__ == "__main__":
    classifier = LLMIntentClassifier(provider="mock")
    sample_msg = "My refund for order #102 has not arrived yet, where is my money?"
    res = classifier.classify(sample_msg)
    print("Test Intent Classification Output:")
    print(json.dumps(res, indent=2))
