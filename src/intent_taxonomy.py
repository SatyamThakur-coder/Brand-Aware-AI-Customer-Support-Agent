"""
Intent taxonomy module for AmazonHelp brand.
Defines brand-specific intents derived from actual customer support conversations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List

TAXONOMY_PATH = Path("configs/intent_taxonomy.yaml")

DEFAULT_TAXONOMY = {
    "intents": {
        "delivery_issue": {
            "description": "Customer inquires about delayed, missing tracking, incorrect delivery address, or package stuck in transit.",
            "examples": [
                "My package was supposed to arrive yesterday where is it?",
                "Tracking says delivered but I don't have my order.",
                "Why is my delivery taking so long?"
            ],
            "escalation_required": False,
            "escalation_notes": "Can auto-handle general status queries; escalate if order modification or address change required."
        },
        "refund_return_request": {
            "description": "Customer requests a refund, asks about return status, or inquires why a processed refund hasn't hit their bank.",
            "examples": [
                "Where is my refund for order 123?",
                "How do I return a damaged item?",
                "I returned my package last week and still haven't received money back."
            ],
            "escalation_required": True,
            "escalation_notes": "Requires account/transaction lookup for exact refund status. Escalate."
        },
        "missing_defective_item": {
            "description": "Customer received a broken, wrong, or incomplete item in their shipment.",
            "examples": [
                "My package arrived but the item inside was shattered.",
                "I ordered 2 items but only received 1.",
                "Sent me the wrong size shoe."
            ],
            "escalation_required": True,
            "escalation_notes": "Requires replacement or return authorization linked to private customer account. Escalate."
        },
        "payment_billing_issue": {
            "description": "Customer charged twice, incorrect billing amount, failed payment method, or gift card issues.",
            "examples": [
                "Why was I charged twice for my order?",
                "My gift card balance was not applied.",
                "Payment failed but money was deducted from my account."
            ],
            "escalation_required": True,
            "escalation_notes": "Sensitive financial/billing transaction. Escalate."
        },
        "account_login_security": {
            "description": "Customer locked out of account, suspicious login notification, 2FA issues, or compromised password.",
            "examples": [
                "I am locked out of my account and cannot reset password.",
                "Received an email about a login I didn't authorize.",
                "Can't sign into my Prime account."
            ],
            "escalation_required": True,
            "escalation_notes": "High-risk security/PII issue. Escalate immediately."
        },
        "prime_membership_issue": {
            "description": "Inquiries about Prime subscription charges, benefits, cancellation of free trial, or video streaming access.",
            "examples": [
                "Why did Prime automatically renew?",
                "How do I cancel my Prime trial?",
                "Prime video says I don't have access."
            ],
            "escalation_required": False,
            "escalation_notes": "Auto-handle standard Prime cancellation steps; escalate if refund of annual fee requested."
        },
        "cancellation_request": {
            "description": "Customer wants to cancel an order that hasn't shipped yet.",
            "examples": [
                "Please cancel order #405-12345",
                "I accidentally placed an order, how do I cancel?",
                "Can you stop shipping my item?"
            ],
            "escalation_required": True,
            "escalation_notes": "Order cancellation requires immediate real-time account action. Escalate."
        },
        "information_request": {
            "description": "General questions about product specs, shipping options, warranty, or store policies.",
            "examples": [
                "What is your holiday return window?",
                "Does Prime delivery work on Sundays?",
                "Where can I check warranty information?"
            ],
            "escalation_required": False,
            "escalation_notes": "Standard public policy question. Auto-handle if historical evidence supports it."
        },
        "other_general": {
            "description": "Compliments, vague inquiries, feedback, or messages that do not fit standard categories.",
            "examples": [
                "Thanks for the great service!",
                "Hello",
                "Your customer service line is busy"
            ],
            "escalation_required": False,
            "escalation_notes": "General acknowledgement or escalate if unclear."
        }
    }
}

def load_intent_taxonomy(yaml_path: Path = TAXONOMY_PATH) -> Dict[str, Any]:
    if not yaml_path.exists():
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(DEFAULT_TAXONOMY, f, default_flow_style=False)
        print(f"Created default intent taxonomy at {yaml_path}")
        return DEFAULT_TAXONOMY
    
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    taxonomy = load_intent_taxonomy()
    print("Loaded Intent Taxonomy:", list(taxonomy['intents'].keys()))
