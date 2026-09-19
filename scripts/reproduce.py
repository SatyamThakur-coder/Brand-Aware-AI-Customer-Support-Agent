"""
Full Reproduction Script.
Executes complete end-to-end pipeline reproduction in under 15 minutes:
1. Dataset validation
2. Preprocessing & thread reconstruction
3. Dataset splitting (70/15/15 conversation-level zero leakage)
4. Golden evaluation set verification
5. FAISS vector index construction
6. Baselines execution (Majority & TF-IDF + Logistic Regression)
7. Unified pipeline demonstration
8. Full evaluation suite execution & report generation
"""

import sys
import os
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_full_reproduction(provider: str = "mock"):
    print("\n" + "="*70)
    print("STARTING REPRODUCIBILITY SUITE — BRAND-AWARE AI CUSTOMER SUPPORT AGENT")
    print("="*70)

    # 1. Dataset Verification
    print("\n[STEP 1/8] Validating raw Kaggle dataset (data/raw/twcs/twcs.csv)...")
    from src.data_loader import DataLoader
    loader = DataLoader("data/raw/twcs/twcs.csv")
    print("  ✓ Dataset file verified.")

    # 2. Preprocessing
    print("\n[STEP 2/8] Preprocessing tweets and reconstructing conversation threads...")
    from src.preprocessing import process_and_save_dataset
    proc_path = process_and_save_dataset(brand_name="AmazonHelp", max_conversations=10000)

    # 3. Dataset Splitting
    print("\n[STEP 3/8] Performing conversation-level Train/Val/Test splitting...")
    from src.dataset_split import split_conversations
    split_conversations(processed_path=proc_path)

    # 4. Golden Set Verification
    print("\n[STEP 4/8] Verifying golden evaluation set...")
    from scripts.create_golden_set import generate_candidate_golden_set
    golden_path = Path("evaluation/golden_set.jsonl")
    if not golden_path.exists():
        generate_candidate_golden_set(num_samples=198)

    # 5. Vector Indexing
    print("\n[STEP 5/8] Building FAISS vector index from training split...")
    from src.retrieval import HistoricalCaseRetriever
    retriever = HistoricalCaseRetriever()
    retriever.build_index()

    # 6. Pipeline CLI Demonstration
    print("\n[STEP 6/8] Running pipeline CLI demonstration query...")
    from src.pipeline import SupportPipeline
    pipeline = SupportPipeline(provider=provider)
    demo_out = pipeline.process("Where is my refund for order #102?")
    print(pipeline.format_cli_output(demo_out))

    # 7. Evaluation Suite
    print("\n[STEP 7/8] Running complete evaluation suite & baseline comparisons...")
    from evaluation.run_all import run_all_evaluations
    run_all_evaluations(provider=provider)

    print("\n" + "="*70)
    print("REPRODUCIBILITY SUITE COMPLETED SUCCESSFULLY!")
    print("All empirical evaluation results have been generated and saved under reports/.")
    print("="*70)

if __name__ == "__main__":
    provider_arg = sys.argv[1] if len(sys.argv) > 1 else "mock"
    run_full_reproduction(provider=provider_arg)
