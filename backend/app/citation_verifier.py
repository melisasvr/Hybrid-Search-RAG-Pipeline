"""
Citation verification: the second line of defense against hallucination.

The LLM is instructed to cite [n] markers, but instructions are not
guarantees. This module independently checks each citation two ways:

1. Structural validity: does [n] refer to a passage number we actually gave
   the model? (Catches invented citations outright.)
2. Groundedness: for the sentence containing the citation, how much lexical
   overlap does it have with the text of the chunk(s) it cites, via TF-IDF
   cosine similarity? A sentence that cites [2] but shares almost no
   vocabulary with chunk 2 is flagged as "weak" (drifted from the source)
   rather than silently trusted.

This is a lightweight, dependency-light heuristic (no extra LLM call, no
extra network round trip) that is good enough to visibly flag drift in a
demo/portfolio setting. For production you'd likely add an LLM-as-judge
pass or a trained NLI entailment model for a stronger signal.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Citation, RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")

# A citation marker like "[1]" or "[1][3]" is written *after* the sentence-
# ending punctuation it belongs to (e.g. "...monthly. [1] Sick leave..."),
# so a sentence boundary is "punctuation, plus any citation markers that
# immediately follow it" — not just the punctuation alone. Matching on that
# combined boundary (rather than re.split on whitespace) keeps each citation
# attached to the sentence it actually cites instead of the next one.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?:\s*\[\d+\])*")


def _split_sentences(text: str) -> list[str]:
    sentences = []
    start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        sentences.append(text[start : m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _lexical_similarity(a: str, b: str) -> float:
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([a, b])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except ValueError:
        # Happens if both strings are pure stopwords/empty after vectorizing.
        return 0.0


def _has_contradictory_tokens(sentence: str, chunk_text: str) -> bool:
    """
    Checks for hallucinated numbers or core negation/polarity mismatches.
    """
    sentence_lower = sentence.lower()
    chunk_lower = chunk_text.lower()

    # 1. Number check
    sentence_nums = set(re.findall(r"\b\d+\b", sentence))
    chunk_nums = set(re.findall(r"\b\d+\b", chunk_text))
    if sentence_nums and not sentence_nums.issubset(chunk_nums):
        return True

    # 2. Polarity / Negation mismatch check
    # Check if one text contains a negative form (not/n't/never/no) while the other does not
    neg_patterns = ["not", "n't", "never", "no ", "do not", "does not", "did not"]
    
    sentence_has_neg = any(p in sentence_lower for p in neg_patterns)
    chunk_has_neg = any(p in chunk_lower for p in neg_patterns)
    
    if sentence_has_neg != chunk_has_neg:
        return True

    # 3. Direct Reversal check (e.g., "always" vs "does not")
    if ("always" in sentence_lower and "never" in chunk_lower) or ("never" in sentence_lower and "always" in chunk_lower):
        return True
    if ("always" in sentence_lower and "not" in chunk_lower) or ("not" in sentence_lower and "always" in chunk_lower):
        return True

    return False

    # 3. Polarity / Reversal check (e.g., "always" vs "does not")
    reversal_pairs = [("always", "never"), ("does", "does not"), ("will", "will not")]
    for word_a, word_b in reversal_pairs:
        if (word_a in sentence_lower and word_b in chunk_lower) or (word_b in sentence_lower and word_a in chunk_lower):
            return True

    return False


def verify_citations(
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
    verified_threshold: float,
    weak_threshold: float,
) -> list[Citation]:
    sentences = _split_sentences(answer)
    # index -> best groundedness score seen across all sentences citing it
    best_score_for_index: dict[int, float] = {}
    cited_indices: set[int] = set()

    for sentence in sentences:
        markers = [int(m) for m in _CITATION_RE.findall(sentence)]
        if not markers:
            continue
        clean_sentence = _CITATION_RE.sub("", sentence).strip()
        for idx in markers:
            cited_indices.add(idx)
            if 1 <= idx <= len(retrieved_chunks):
                chunk_text = retrieved_chunks[idx - 1].chunk.text
                score = _lexical_similarity(clean_sentence, chunk_text)
                best_score_for_index[idx] = max(best_score_for_index.get(idx, 0.0), score)

    citations: list[Citation] = []
    for idx in sorted(cited_indices):
        marker = f"[{idx}]"
        if not (1 <= idx <= len(retrieved_chunks)):
            citations.append(
                Citation(
                    marker=marker,
                    index=idx,
                    doc_id="unknown",
                    source="unknown",
                    page=None,
                    chunk_id="unknown",
                    snippet="(citation number does not match any retrieved passage)",
                    verification_status="invalid_citation",
                    verification_score=0.0,
                )
            )
            continue

        rc = retrieved_chunks[idx - 1]
        score = best_score_for_index.get(idx, 0.0)
        
        # Check if the sentence introduces numbers or negations missing from the source chunk[cite: 2]
        has_contradiction = _has_contradictory_tokens(clean_sentence, rc.chunk.text)

        if has_contradiction:
            status = "unverified"
            score = 0.0
        elif score >= verified_threshold:
            status = "verified"
        elif score >= weak_threshold:
            status = "weak"
        else:
            status = "unverified"

        snippet = rc.chunk.text[:300] + ("..." if len(rc.chunk.text) > 300 else "")
        citations.append(
            Citation(
                marker=marker,
                index=idx,
                doc_id=rc.chunk.doc_id,
                source=rc.chunk.source,
                page=rc.chunk.page,
                chunk_id=rc.chunk.chunk_id,
                snippet=snippet,
                retrieval_score=rc.fused_score,
                rerank_score=rc.rerank_score,
                verification_status=status,
                verification_score=round(score, 3),
            )
        )
    return citations