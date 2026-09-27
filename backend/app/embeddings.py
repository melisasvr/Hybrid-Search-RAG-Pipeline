"""
Thin wrapper around fastembed so that ingestion and query-time retrieval
always embed with the *exact same* model and settings. (A common production
bug is embedding documents with one model/version and queries with another,
silently degrading dense recall.)

Why fastembed instead of sentence-transformers: sentence-transformers pulls
in the full torch + transformers stack (multi-GB install, easy to collide
with whatever ML tooling is already on your machine — which is exactly what
was happening here). fastembed runs the same embedding models through ONNX
Runtime instead: no torch, a much smaller install, and faster CPU inference.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_l2_normalize(v).tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.embed([text])))
        return _l2_normalize(vector).tolist()


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)
