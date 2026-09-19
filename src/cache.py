"""
File-backed LLM response caching module.
Stores prompt-hash to JSON response mappings to eliminate redundant LLM API calls and ensure fast reproducible runs.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

CACHE_DIR = Path(".cache/llm_cache")

class ResponseCache:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, prompt: str, model: str) -> str:
        content = f"{model}::{prompt}".encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        key = self._hash_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(self, prompt: str, model: str, response_data: Dict[str, Any]):
        key = self._hash_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write cache: {e}")
