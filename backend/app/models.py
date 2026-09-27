"""Pydantic schemas shared across the API and pipeline internals."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable unit of text plus its provenance."""

    chunk_id: str
    doc_id: str
    source: str  # original filename
    chunk_index: int  # position of this chunk within the document
    page: Optional[int] = None
    text: str


class RetrievedChunk(BaseModel):
    """A chunk annotated with scores from each stage of hybrid retrieval."""

    chunk: Chunk
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    num_chunks: int


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    num_chunks: int
    message: str = "Document ingested successfully."


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = None
    doc_ids: Optional[list[str]] = None  # optional filter to specific documents


VerificationStatus = Literal["verified", "weak", "unverified", "invalid_citation"]


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    index: int
    doc_id: str
    source: str
    page: Optional[int] = None
    chunk_id: str
    snippet: str
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    verification_status: VerificationStatus
    verification_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]  # full retrieval debug trail
    latency_ms: float
    warnings: list[str] = []
