# Golden Evaluation Set Guidelines

## Overview

The Golden Evaluation Set (`evaluation/golden_set.jsonl`) contains 198 human-reviewed, high-quality benchmark examples used to rigorously evaluate the AI Customer Support Agent.

## Sampling & Stratification Methodology

1. **Stratified Sampling**: Candidate messages were sampled across all 9 AmazonHelp intent taxonomy categories from the holdout test set (`data/splits/test.jsonl`).
2. **Zero Conversation Leakage**: All golden evaluation records belong strictly to the holdout test split. No golden evaluation example is present in the training set (`train.jsonl`), validation set (`validation.jsonl`), or the retrieval FAISS vector index.

## Annotation Fields

Each record in `evaluation/golden_set.jsonl` contains:

```json
{
  "id": "gold_001",
  "conversation_id": "conv_10293",
  "customer_message": "Where is my refund for order #102-39281?",
  "brand_response_reference": "Please send us a DM with your order number so we can look into your refund.",
  "intent": "refund_return_request",
  "expected_action": "ESCALATE",
  "notes": "Requires private account/order lookup to check refund status.",
  "status": "FINALIZED"
}
```

## Intent Classification Rules

- **`delivery_issue`**: Delayed packages, lost tracking, courier inquiries.
- **`refund_return_request`**: Requests for money back, return status inquiries.
- **`missing_defective_item`**: Shattered, missing, or damaged items received in parcel.
- **`payment_billing_issue`**: Double charges, payment gateway errors, gift card balance issues.
- **`account_login_security`**: Account lockout, password reset, suspicious logins.
- **`prime_membership_issue`**: Prime subscription auto-renewal, Prime video streaming issues.
- **`cancellation_request`**: Order cancellation requests before shipment.
- **`information_request`**: Store policy, warranty details, general FAQs.
- **`other_general`**: Pleasantries, generic feedback, ambiguous text.

## Escalation Policy Rules for Gold Labels

- **AUTO_HANDLE**: Public policies, standard FAQs, general Prime trial cancellation instructions.
- **ESCALATE**: Any request requiring account lookup, private transaction verification, financial billing adjustments, order cancellations, or security credential resets.

## Status Flags

- **`PENDING_HUMAN_REVIEW`**: Candidate generated automatically by sampling script. Not yet confirmed.
- **`FINALIZED`**: Manually reviewed and verified by a human annotator.
