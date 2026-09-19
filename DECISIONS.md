# Technical & Architectural Decision Log

This document records 15 critical, non-obvious engineering decisions and trade-offs made during the development of the **Brand-Aware AI Customer Support Agent** (`hiver-sde-ai-support-agent`).

---

### Decision 1: Brand Selection — Why `AmazonHelp` Was Selected
- **Choice**: Selected `AmazonHelp` over other brand handles (`AppleSupport`, `Uber_Support`, `SpotifyCares`, `Delta`).
- **Rationale**: Based on empirical dataset exploration (`scripts/explore_dataset.py`), `AmazonHelp` contains the single largest volume of brand replies (**169,840 tweets**) and multi-turn resolution pairs (**42,709 sampled threads**). It covers a diverse, representative range of e-commerce support categories (delivery delays, returns/refunds, prime subscriptions, missing items, billing disputes) rather than narrow tech-support queries.

---

### Decision 2: Intent Taxonomy — Derived from Brand Data (9 Categories)
- **Choice**: Designed a custom 9-intent taxonomy specifically tailored to `AmazonHelp` data rather than blindly importing generic benchmark taxonomies (e.g., Banking77).
- **Rationale**: Banking77 contains 77 banking-specific intents (e.g., card pin reset, ATM withdrawal limits) irrelevant to e-commerce customer support on Twitter. Deriving 9 categories directly from empirical tweet frequency ensures high semantic coverage and actionable escalation boundaries.

---

### Decision 3: Evaluation Metrics — Why Macro F1 is the Primary Metric
- **Choice**: Evaluated all classifiers using **Macro F1** alongside raw Accuracy and Per-Intent metrics.
- **Rationale**: Real customer support datasets suffer from heavy class imbalance (e.g., general inquiries represent ~70% of messages). A trivial Majority Class Baseline achieves a misleadingly high **73.20% accuracy** while yielding a useless **0.0939 Macro F1**. Macro F1 weights all intent classes equally, exposing failures on low-frequency, high-stakes intents.

---

### Decision 4: Conversation-Level Dataset Splitting (Zero Leakage)
- **Choice**: Grouped multi-turn messages by unique `conversation_id` before partitioning into 70% Train, 15% Validation, and 15% Test splits.
- **Rationale**: Randomly splitting individual tweets causes severe data leakage where customer turns appear in training while brand response turns appear in test. Grouping at the conversation level guarantees zero conversation leakage across training data, holdout test data, and the FAISS vector index.

---

### Decision 5: TF-IDF + Logistic Regression as the Supervised Baseline
- **Choice**: Implemented TF-IDF vectorization paired with class-weighted Logistic Regression as Baseline 2.
- **Rationale**: TF-IDF + Logistic Regression is computationally fast, highly interpretable, and establishes a strong linear baseline (**93.87% accuracy / 0.8718 Macro F1**). It serves as a benchmark to prove whether complex LLM architectures deliver measurable improvements over classical ML.

---

### Decision 6: Local FAISS Indexing with SentenceTransformers (`all-MiniLM-L6-v2`)
- **Choice**: Built a local FAISS vector search index using normalized `all-MiniLM-L6-v2` dense embeddings.
- **Rationale**: `all-MiniLM-L6-v2` offers an optimal balance between semantic embedding quality and fast inference speed (~380 sentences/sec on CPU). Local FAISS flat inner-product indexing avoids third-party cloud vector DB dependencies (e.g., Pinecone/Weaviate), guaranteeing 100% offline reproducibility.

---

### Decision 7: Top-K = 5 Historical Case Retrieval
- **Choice**: Configured default retrieval depth to `TOP_K = 5`.
- **Rationale**: Empirical retrieval testing showed that Top 5 retrieved cases provide sufficient context without exceeding prompt context windows or introducing irrelevant noise. Hybrid reranking boosts candidate cases matching the predicted intent.

---

### Decision 8: Deterministic Escalation Safety Engine (Multi-Gate Policy)
- **Choice**: Restricted the LLM from making unconstrained `AUTO_HANDLE` decisions. Implemented a deterministic 6-gate policy layer in `src/escalation.py`.
- **Rationale**: LLMs are prone to over-confidence and hallucinated policy promises. Forcing deterministic gates (confidence >= 0.80, similarity >= 0.45, mandatory escalation intent filter) ensures high-risk queries are safely routed to human agents.

---

### Decision 9: Manual Golden Evaluation Set Annotation
- **Choice**: Required human developer verification for the 198 golden evaluation set records via `scripts/create_golden_set.py`.
- **Rationale**: Synthetic LLM-generated golden labels introduce self-evaluating circular bias. Forcing manual human verification guarantees an uncorrupted ground truth benchmark.

---

### Decision 10: Human vs LLM Judge Agreement Infrastructure
- **Choice**: Built formal statistical comparison infrastructure (`evaluation/human_judge_agreement.py`) measuring Agreement %, MAE, and Pearson correlation between human ratings and LLM Judge scores.
- **Rationale**: An LLM judge cannot be trusted blindly without empirical validation against human judgment. If human ratings are missing, the system explicitly reports **"Human judge validation pending."**

---

### Decision 11: Historical Replies as Evidence, NOT Current Policy
- **Choice**: Instructed response generator prompts to treat retrieved historical brand replies strictly as evidence of past communication, not immutable company policy.
- **Rationale**: Company policies, refund windows, and promotional terms change over time. Treating historical tweets as absolute policy risks making false promises to customers.

---

### Decision 12: Mandatory Escalation for Private Account & Financial Queries
- **Choice**: Automatically escalated all inquiries categorized as `refund_return_request`, `payment_billing_issue`, `account_login_security`, `missing_defective_item`, or `cancellation_request`.
- **Rationale**: Resolving these inquiries requires accessing private PII, order databases, or executing real-time financial transactions unavailable to a public social support agent.

---

### Decision 13: Capped Development Dataset Processing (10,000 Conversations)
- **Choice**: Capped processed development dataset to 10,000 conversations while providing full download scripts for the 2.8M row Kaggle dataset.
- **Rationale**: Processing 2.8 million tweets on every development iteration creates unnecessary compute bottleneck. 10,000 conversations provide ample statistical depth while allowing full pipeline execution in under 15 minutes.

---

### Decision 14: Read-Only Draft Generation (No Automated Account Actions)
- **Choice**: Designed the AI agent as a read-only draft response and triage system without direct API access to cancel orders or issue refunds.
- **Rationale**: Executing automated financial or account mutations without human-in-the-loop review introduces unacceptable business risk.

---

### Decision 15: Explicit Failure Analysis & Misleading Metric Reporting
- **Choice**: Created dedicated failure analysis tools (`analysis/failure_analysis.py`) and a mandatory critical report (`reports/misleading_headline_number.md`).
- **Rationale**: Engineering excellence requires transparency about where a system fails. Highlighting failure modes and explaining headline metric limitations proves rigorous, honest evaluation.
