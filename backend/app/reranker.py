"""
Cross-encoder reranking: the final precision pass.

Dense + BM25 fusion is a cheap, high-recall way to shortlist ~10-20
plausible candidates from a large corpus. But bi-encoder similarity and
term overlap are both weak proxies for "does this chunk actually answer the
question." A cross-encoder scores the (query, chunk) pair jointly, which is
far more accurate for ranking a small candidate set — and too slow to run
over the whole corpus, which is why it comes *after* fusion, not instead of it.

Implemented via fastembed's TextCrossEncoder (ONNX Runtime) rather than
sentence-transformers' CrossEncoder, for the same reason as embeddings.py:
no torch/transformers dependency, much smaller install, no version fights
with other ML tooling on the machine.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from .models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        documents = [c.chunk.text for c in candidates]
        scores = list(self._model.rerank(query, documents))
        for c, score in zip(candidates, scores):
            c.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]


@lru_cache
def get_reranker(model_name: str) -> Reranker:
    return Reranker(model_name)
