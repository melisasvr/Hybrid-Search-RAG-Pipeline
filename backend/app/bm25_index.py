"""
BM25 sparse retrieval, the classic term-frequency counterpart to dense
vectors. It's what catches the exact product code, error string, or acronym
that an embedding model tends to blur past.
"""
from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Wraps rank_bm25 with chunk_id tracking and disk persistence.

    rank_bm25 has no incremental-update API, so we keep the full tokenized
    corpus in memory and rebuild the BM25Okapi object whenever chunks are
    added or removed. This is fine up to tens of thousands of chunks; for
    larger corpora swap in Elasticsearch/OpenSearch's native BM25.
    """

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            self._chunk_ids = data["chunk_ids"]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._rebuild()

    def _persist(self) -> None:
        with open(self._path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self._chunk_ids, "tokenized_corpus": self._tokenized_corpus},
                f,
            )

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def add_chunks(self, chunk_ids: list[str], texts: list[str]) -> None:
        with self._lock:
            self._chunk_ids.extend(chunk_ids)
            self._tokenized_corpus.extend(tokenize(t) for t in texts)
            self._rebuild()
            self._persist()

    def delete_document(self, doc_id_predicate) -> None:
        """doc_id_predicate(chunk_id) -> bool, True if the chunk should be removed."""
        with self._lock:
            keep = [
                (cid, toks)
                for cid, toks in zip(self._chunk_ids, self._tokenized_corpus)
                if not doc_id_predicate(cid)
            ]
            self._chunk_ids = [cid for cid, _ in keep]
            self._tokenized_corpus = [toks for _, toks in keep]
            self._rebuild()
            self._persist()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, score) for cid, score in ranked[:top_k] if score > 0]

    def count(self) -> int:
        return len(self._chunk_ids)
