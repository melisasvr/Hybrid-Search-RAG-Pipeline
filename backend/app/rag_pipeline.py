"""Ties hybrid retrieval + Groq generation + citation verification into one call."""
from __future__ import annotations

import time

from .citation_verifier import verify_citations
from .config import Settings
from .hybrid_retriever import HybridRetriever
from .llm import GroqLLM
from .models import QueryResponse


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: HybridRetriever, llm: GroqLLM):
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    def answer(
        self, question: str, top_k: int | None = None, doc_ids: list[str] | None = None
    ) -> QueryResponse:
        start = time.perf_counter()
        warnings: list[str] = []

        retrieved = self.retriever.retrieve(question, top_k_final=top_k, doc_ids=doc_ids)

        if not retrieved:
            latency_ms = (time.perf_counter() - start) * 1000
            return QueryResponse(
                question=question,
                answer=(
                    "No relevant documents have been ingested yet, or none matched this "
                    "question. Upload documents first, or try rephrasing."
                ),
                citations=[],
                retrieved_chunks=[],
                latency_ms=round(latency_ms, 1),
                warnings=["No chunks retrieved."],
            )

        raw_answer = self.llm.generate_answer(question, retrieved)

        # Normalize full-width/unicode brackets to standard ASCII square brackets
        raw_answer = raw_answer.replace("【", "[").replace("】", "]")

        citations = verify_citations(
            answer=raw_answer,
            retrieved_chunks=retrieved,
            verified_threshold=self.settings.citation_verified_threshold,
            weak_threshold=self.settings.citation_weak_threshold,
        )

        if any(c.verification_status == "invalid_citation" for c in citations):
            warnings.append(
                "The model referenced a citation number that does not correspond to any "
                "retrieved passage. Treat that portion of the answer with caution."
            )
        if any(c.verification_status == "unverified" for c in citations):
            warnings.append(
                "One or more citations have low lexical overlap with their cited passage "
                "and may not be well grounded."
            )
        if not any(f"[{i}]" in raw_answer for i in range(1, len(retrieved) + 1)):
            warnings.append("The answer contains no citations at all.")

        latency_ms = (time.perf_counter() - start) * 1000
        return QueryResponse(
            question=question,
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=retrieved,
            latency_ms=round(latency_ms, 1),
            warnings=warnings,
        )