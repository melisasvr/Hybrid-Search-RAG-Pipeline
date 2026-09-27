"""
Dense retrieval via ChromaDB.

We bring our own embeddings (computed through `Embedder`) rather than letting
Chroma manage an embedding function internally, so ingestion and query-time
embedding are guaranteed to go through the identical code path.
"""
from __future__ import annotations

import chromadb

from .models import Chunk

COLLECTION_NAME = "company_docs"


class VectorStore:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "page": c.page if c.page is not None else -1,
                }
                for c in chunks
            ],
        )

    def query(
        self, query_embedding: list[float], top_k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Returns [(chunk_id, similarity_score), ...] ranked best-first.

        Chroma returns cosine *distance*; we convert to a similarity in
        [0, 1] (similarity = 1 - distance/2 for normalized vectors, but we
        keep it simple with 1 - distance which is monotonic and adequate
        for fusion/reranking purposes).
        """
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        ids = result["ids"][0] if result["ids"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._collection.count()
