"""
Groq-backed answer generation.

The system prompt is the main lever against hallucination at generation
time: it forces numbered citations that map 1:1 to the chunks we actually
retrieved, and gives the model an explicit, non-punished way to say "I don't
know" when the context doesn't support an answer. Citation *verification*
(citation_verifier.py) is the second, independent line of defense that
catches cases where the model cites a real chunk but still drifts from it.
"""
from __future__ import annotations

from groq import Groq

from .models import RetrievedChunk

SYSTEM_PROMPT = """You are a precise assistant answering questions using ONLY the numbered \
context passages provided below. Rules:

1. Answer using only information present in the context. Do not use outside knowledge.
2. Every factual sentence in your answer must end with a citation marker like [1] or [1][3] \
referencing the passage number(s) it is drawn from.
3. If the context does not contain enough information to answer, say so explicitly \
("The provided documents do not contain enough information to answer this.") instead of guessing.
4. Do not invent citation numbers that are not in the provided context.
5. Be concise and direct. Do not repeat the question back."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        page_info = f", page {rc.chunk.page}" if rc.chunk.page else ""
        lines.append(f"[{i}] (source: {rc.chunk.source}{page_info})\n{rc.chunk.text}")
    return "\n\n".join(lines)


class GroqLLM:
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_answer(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        context_block = build_context_block(context_chunks)
        user_prompt = (
            f"Context passages:\n\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer using the rules above, citing passage numbers in brackets."
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
