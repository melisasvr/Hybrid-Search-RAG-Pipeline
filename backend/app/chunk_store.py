"""
ChunkStore: the single source of truth for chunk text + metadata.

ChromaDB and the BM25 index both need to resolve a chunk_id back to its
text, source filename, and page number. Rather than trusting each store's
own copy of metadata (and risking drift between them), everything reads
that information from here. It's a small JSON file for portability in this
project; swap for SQLite/Postgres if you need concurrent writers.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .models import Chunk


class ChunkStore:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._chunks: dict[str, Chunk] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._chunks = {cid: Chunk(**data) for cid, data in raw.items()}

    def _persist(self) -> None:
        serializable = {cid: c.model_dump() for cid, c in self._chunks.items()}
        self._path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock:
            for c in chunks:
                self._chunks[c.chunk_id] = c
            self._persist()

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            to_delete = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
            for cid in to_delete:
                del self._chunks[cid]
            self._persist()
            return len(to_delete)

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self._chunks.values():
            if c.doc_id not in docs:
                docs[c.doc_id] = {"doc_id": c.doc_id, "source": c.source, "num_chunks": 0}
            docs[c.doc_id]["num_chunks"] += 1
        return list(docs.values())
