"""
Deterministic Escalation Policy Engine.
Enforces multi-gate safety rules so that the LLM does not have unrestricted authority to auto-handle customer messages.
"""

from typing import Dict, Any, List, Tuple

MANDATORY_ESCALATE_INTENTS = [
    "refund_return_request",
    "missing_defective_item",
    "payment_billing_issue",
    "account_login_security",
    "cancellation_request"
]

class EscalationEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.80,
        min_retrieval_similarity: float = 0.45,
        mandatory_escalate_intents: List[str] = MANDATORY_ESCALATE_INTENTS
    ):
        self.confidence_threshold = confidence_threshold
        self.min_retrieval_similarity = min_retrieval_similarity
        self.mandatory_escalate_intents = set(mandatory_escalate_intents)

    def evaluate_escalation(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        evidence_cases: List[Dict[str, Any]],
        generation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates deterministic safety gates to decide AUTO_HANDLE vs ESCALATE.
        """
        reasons = []

        # Gate 1: Check Unknown or Low Confidence State
        if predicted_intent == "UNKNOWN":
            return {
                "action": "ESCALATE",
                "reason": "Classifier determined customer message is ambiguous or unreadable (UNKNOWN state).",
                "gate_passed": False
            }

        if confidence < self.confidence_threshold:
            return {
                "action": "ESCALATE",
                "reason": f"Classifier confidence ({confidence:.2f}) is below safe threshold ({self.confidence_threshold:.2f}).",
                "gate_passed": False
            }

        # Gate 2: Mandatory Escalation Intents (Private Account / High-Risk Financial Issues)
        if predicted_intent in self.mandatory_escalate_intents:
            return {
                "action": "ESCALATE",
                "reason": f"Intent category '{predicted_intent}' requires private customer account lookup or sensitive transaction authority.",
                "gate_passed": False
            }

        # Gate 3: Retrieval Evidence Quality Check
        if not evidence_cases:
            return {
                "action": "ESCALATE",
                "reason": "No historical resolution evidence cases were found in the vector index.",
                "gate_passed": False
            }

        top1_sim = evidence_cases[0].get("similarity_score", 0.0)
        if top1_sim < self.min_retrieval_similarity:
            return {
                "action": "ESCALATE",
                "reason": f"Top retrieved historical evidence similarity ({top1_sim:.4f}) is below minimum threshold ({self.min_retrieval_similarity:.2f}).",
                "gate_passed": False
            }

        # Gate 4: Groundedness Check from Generation
        if not generation_result.get("supported_by_evidence", False):
            return {
                "action": "ESCALATE",
                "reason": "Draft reply generation is not fully supported by retrieved historical evidence.",
                "gate_passed": False
            }

        # All Gates Passed -> Safe for Auto-Handling
        return {
            "action": "AUTO_HANDLE",
            "reason": f"Message categorized as safe intent '{predicted_intent}' (confidence: {confidence:.2f}) supported by strong historical evidence (similarity: {top1_sim:.4f}).",
            "gate_passed": True
        }

if __name__ == "__main__":
    engine = EscalationEngine()
    res1 = engine.evaluate_escalation(
        customer_message="Where is my refund?",
        predicted_intent="refund_return_request",
        confidence=0.95,
        evidence_cases=[{"similarity_score": 0.85}],
        generation_result={"supported_by_evidence": True}
    )
    print("Test Escalation output (refund request):")
    print(res1)
