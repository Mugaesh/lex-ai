"""
Robust embedding helper for LexAI.
Attempts to load SentenceTransformer; falls back to an offline deterministic
multilingual text vectorizer if HuggingFace Hub is unreachable.
"""
import hashlib
import os
import re
from typing import List
import numpy as np

from src.config import EMBEDDING_MODEL

_embedding_instance = None


class FallbackEmbedder:
    """Fast, offline 384-dimensional dense bag-of-words / hash vectorizer."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_token(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            # Tokenize words and character n-grams to support English, Hindi, Tamil
            tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
            for token in tokens:
                idx = self._hash_token(token)
                vec[idx] += 1.0
            for i in range(max(0, len(text) - 3)):
                ngram = text[i : i + 3].lower()
                idx = self._hash_token(ngram)
                vec[idx] += 0.5

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors)


def get_embedder():
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    # Check if local files only works without any network delay
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_instance = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        print(f"Loaded locally cached SentenceTransformer: {EMBEDDING_MODEL}")
        return _embedding_instance
    except Exception:
        pass

    # Use fast, deterministic multilingual dense vectorizer
    print(f"Using high-speed offline multilingual vectorizer (384-dim).")
    _embedding_instance = FallbackEmbedder(dim=384)
    return _embedding_instance
