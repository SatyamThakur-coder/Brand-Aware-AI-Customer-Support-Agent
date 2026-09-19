"""
Historical Case Retrieval Module using SentenceTransformers and FAISS.
Builds vector index ONLY from training split (train.jsonl) to strictly avoid data leakage.
Supports hybrid semantic search + intent filtering and retrieval self-exclusion mechanism.
"""

import json
import pickle
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index")
TRAIN_SPLIT_PATH = Path("data/splits/train.jsonl")

class HistoricalCaseRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_dir: Path = INDEX_DIR,
        top_k: int = 5
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self.encoder = None
        self.index = None
        self.cases: List[Dict[str, Any]] = []
        
        self.index_file = self.index_dir / "faiss_index.bin"
        self.metadata_file = self.index_dir / "cases.pkl"

    def _load_encoder(self):
        if self.encoder is None:
            print(f"Loading SentenceTransformer model '{self.model_name}'...")
            self.encoder = SentenceTransformer(self.model_name)

    def build_index(self, train_path: Path = TRAIN_SPLIT_PATH, force_rebuild: bool = False):
        """
        Builds FAISS vector index strictly from training set data.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index_file.exists() and self.metadata_file.exists() and not force_rebuild:
            print(f"Loading pre-built vector index from {self.index_dir}...")
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, "rb") as f:
                self.cases = pickle.load(f)
            print(f"Loaded index containing {len(self.cases):,} historical training cases.")
            return

        if not train_path.exists():
            raise FileNotFoundError(f"Training split file not found at {train_path}. Run src.dataset_split first.")

        print(f"Building FAISS vector index from training split: {train_path}...")
        self._load_encoder()

        cases = []
        texts = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cases.append(item)
                    texts.append(item["customer_message"])

        print(f"Encoding {len(texts):,} historical training customer messages...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # Cosine similarity via normalized vectors
        self.index.add(embeddings)
        self.cases = cases

        # Save index and metadata
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.cases, f)

        print(f"Successfully built and persisted FAISS index with {len(cases):,} vectors at {self.index_dir}.")

    def retrieve(
        self,
        query_text: str,
        predicted_intent: Optional[str] = None,
        top_k: Optional[int] = None,
        exclude_conv_id: Optional[str] = None,
        exclude_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves Top K historically similar cases.
        Enforces self-exclusion: drops candidates matching exclude_conv_id or exact exclude_text.
        """
        if self.index is None or not self.cases:
            self.build_index()

        self._load_encoder()
        k = top_k or self.top_k
        query_vec = self.encoder.encode([query_text], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        # Retrieve top 50 candidates for hybrid re-ranking
        search_k = min(50, len(self.cases))
        scores, indices = self.index.search(query_vec, search_k)

        norm_query_text = (exclude_text or query_text).strip().lower()

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.cases):
                continue
            case = dict(self.cases[idx])

            # Self-Exclusion checks
            if exclude_conv_id and case.get("conversation_id") == exclude_conv_id:
                continue
            if norm_query_text and case.get("customer_message", "").strip().lower() == norm_query_text:
                continue

            sim_score = float(score)

            # Hybrid Intent Boost: add +0.10 similarity bonus if intent matches predicted intent
            if predicted_intent and case.get("silver_intent") == predicted_intent:
                sim_score += 0.10

            case["similarity_score"] = float(score) # Raw score
            case["rerank_score"] = float(sim_score)
            candidates.append(case)

        # Sort by hybrid rerank_score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:k]

if __name__ == "__main__":
    retriever = HistoricalCaseRetriever()
    retriever.build_index()
    sample_q = "Where is my refund for order #102?"
    results = retriever.retrieve(sample_q, predicted_intent="refund_return_request", top_k=3)

    print("\nTop 3 Retrieved Historical Evidence Cases:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Conv ID: {r['conversation_id']} | Sim: {r['similarity_score']:.4f} | Rerank: {r['rerank_score']:.4f}")
        print(f"    Customer: {r['customer_message']}")
        print(f"    Historical Brand Reply: {r['brand_response']}")
