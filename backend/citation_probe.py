"""Probe for Hybrid-Search-RAG-Pipeline citation verifier.
Run from the repo's backend/ directory:
    pip install scikit-learn pydantic numpy
    python citation_probe.py
Shows which kinds of unsupported sentences still score "verified" under a TF-IDF
overlap check (wrong number, negation, reversed rule) and which are caught.
"""
import sys
sys.path.insert(0, ".")
from app.models import Chunk, RetrievedChunk
from app.citation_verifier import verify_citations

chunk_text = ("Full-time employees accrue 15 days of paid vacation per year. "
              "Unused vacation does not roll over after March 31.")
rc = [RetrievedChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", source="handbook.md",
                                 chunk_index=0, text=chunk_text))]
cases = {
    "faithful":          "Full-time employees accrue 15 days of paid vacation per year. [1]",
    "wrong number":      "Full-time employees accrue 30 days of paid vacation per year. [1]",
    "negated":           "Full-time employees do not accrue paid vacation per year. [1]",
    "reversed rollover": "Unused vacation always rolls over after March 31. [1]",
    "unrelated":         "The cafeteria serves lunch at noon on weekdays. [1]",
    "invalid marker":    "Employees accrue vacation. [4]",
}
for name, answer in cases.items():
    c = verify_citations(answer, rc, 0.35, 0.15)[0]
    print(f"{name:18s} -> {c.verification_status:17s} score={c.verification_score}")