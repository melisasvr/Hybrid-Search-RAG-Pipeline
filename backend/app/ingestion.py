"""Ingestion pipeline: file on disk -> chunks -> embeddings -> dense + sparse indexes."""
from __future__ import annotations

import uuid
from pathlib import Path

from .bm25_index import BM25Index
from .chunk_store import ChunkStore
from .chunking import load_and_chunk_document
from .config import Settings
from .embeddings import Embedder
from .models import IngestResponse
from .vector_store import VectorStore


def ingest_file(
    path: Path,
    original_filename: str,
    settings: Settings,
    embedder: Embedder,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    chunk_store: ChunkStore,
) -> IngestResponse:
    doc_id = uuid.uuid4().hex[:12]

    chunks = load_and_chunk_document(
        path=path,
        doc_id=doc_id,
        source=original_filename,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise ValueError(f"No text could be extracted from '{original_filename}'.")

    embeddings = embedder.embed_documents([c.text for c in chunks])

    # Order matters: persist metadata first so a crash mid-index never
    # leaves the vector/BM25 stores pointing at chunk_ids with no metadata.
    chunk_store.add_chunks(chunks)
    vector_store.add_chunks(chunks, embeddings)
    bm25_index.add_chunks([c.chunk_id for c in chunks], [c.text for c in chunks])

    return IngestResponse(doc_id=doc_id, source=original_filename, num_chunks=len(chunks))
