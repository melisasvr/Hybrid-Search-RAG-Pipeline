"""
Hybrid retrieval pipeline: dense vectors + BM25 -> reciprocal rank fusion
-> cross-encoder rerank.

Why reciprocal rank fusion (RRF) instead of a weighted sum of raw scores?
Cosine similarity and BM25 scores live on completely different, corpus-
dependent scales, so a weighted sum needs constant re-tuning as the corpus
changes. RRF only looks at *rank position* in each list:

    RRF(chunk) = sum over retrievers of  1 / (k + rank)

which is scale-free, has one boring constant (k, typically 60) instead of a
brittle alpha, and is the same technique used in Elasticsearch's and
Weaviate's built-in hybrid search. We then take the fused top-N and rerank
those with a cross-encoder for the final precision pass.
"""
from __future__ import annotations

from .bm25_index import BM25Index
from .chunk_store import ChunkStore
from .config import Settings
from .embeddings import Embedder
from .models import RetrievedChunk
from .reranker import Reranker
from .vector_store import VectorStore


def reciprocal_rank_fusion(
    dense_ranked: list[tuple[str, float]],
    bm25_ranked: list[tuple[str, float]],
    k: int,
) -> dict[str, dict]:
    """Returns {chunk_id: {fused_score, dense_score, dense_rank, bm25_score, bm25_rank}}."""
    fused: dict[str, dict] = {}

    for rank, (chunk_id, score) in enumerate(dense_ranked, start=1):
        fused.setdefault(chunk_id, {})
        fused[chunk_id]["dense_score"] = score
        fused[chunk_id]["dense_rank"] = rank
        fused[chunk_id]["fused_score"] = fused[chunk_id].get("fused_score", 0.0) + 1.0 / (k + rank)

    for rank, (chunk_id, score) in enumerate(bm25_ranked, start=1):
        fused.setdefault(chunk_id, {})
        fused[chunk_id]["bm25_score"] = score
        fused[chunk_id]["bm25_rank"] = rank
        fused[chunk_id]["fused_score"] = fused[chunk_id].get("fused_score", 0.0) + 1.0 / (k + rank)

    return fused


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        chunk_store: ChunkStore,
        reranker: Reranker,
    ):
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.chunk_store = chunk_store
        self.reranker = reranker

    def retrieve(
        self, query: str, top_k_final: int | None = None, doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        s = self.settings
        top_k_final = top_k_final or s.top_k_final

        # Stage 1a: dense retrieval
        query_embedding = self.embedder.embed_query(query)
        dense_ranked = self.vector_store.query(query_embedding, s.top_k_dense, doc_ids=doc_ids)

        # Stage 1b: sparse (BM25) retrieval
        bm25_ranked_all = self.bm25_index.search(query, s.top_k_bm25 * 3 if doc_ids else s.top_k_bm25)
        if doc_ids:
            allowed = {c.chunk_id for c in self.chunk_store.all_chunks() if c.doc_id in doc_ids}
            bm25_ranked_all = [(cid, sc) for cid, sc in bm25_ranked_all if cid in allowed]
        bm25_ranked = bm25_ranked_all[: s.top_k_bm25]

        # Stage 2: reciprocal rank fusion merges the two ranked lists
        fused = reciprocal_rank_fusion(dense_ranked, bm25_ranked, k=s.rrf_k)
        fused_sorted = sorted(fused.items(), key=lambda kv: kv[1]["fused_score"], reverse=True)
        candidate_ids = [cid for cid, _ in fused_sorted[: s.top_k_rerank_candidates]]

        candidates: list[RetrievedChunk] = []
        for cid in candidate_ids:
            chunk = self.chunk_store.get(cid)
            if chunk is None:
                continue  # stale index entry; chunk was deleted from the store
            meta = fused[cid]
            candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=meta.get("dense_score"),
                    dense_rank=meta.get("dense_rank"),
                    bm25_score=meta.get("bm25_score"),
                    bm25_rank=meta.get("bm25_rank"),
                    fused_score=meta.get("fused_score"),
                )
            )

        # Stage 3: cross-encoder rerank of the fused shortlist
        return self.reranker.rerank(query, candidates, top_k=top_k_final)
