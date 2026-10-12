# Hybrid Search RAG Pipeline

Production-style RAG over a company's internal documents: dense vector search
(ChromaDB) + BM25 keyword search + cross-encoder reranking, grounded answer
generation via Groq, and independent citation verification so answers don't
just *look* cited; they're checked against what was actually retrieved.

Embeddings and reranking run through **fastembed** (ONNX Runtime) rather
than sentence-transformers: same model quality, but no PyTorch/`transformers`
dependency chain, a much smaller install, and no version fights with other
ML tooling that might already be on your machine.

## Architecture

```
                     ┌─────────────────────┐
   PDF/DOCX/TXT/MD   │   Ingestion          │
   ───────────────▶  │  load → chunk →      │
                     │  embed (per chunk)   │
                     └──────────┬───────────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼                             ▼
        ┌────────────────┐           ┌──────────────────┐
        │  ChromaDB       │           │  BM25 index       │
        │  (dense vectors)│           │  (sparse/keyword) │
        └────────┬────────┘           └─────────┬─────────┘
                 │                                │
                 └───────────────┬────────────────┘
                                 ▼
                   Reciprocal Rank Fusion (RRF)
                                 │
                                 ▼
                 Cross-encoder reranker (top ~10 → top 5)
                                 │
                                 ▼
                    Groq LLM (grounded, cited answer)
                                 │
                                 ▼
                  Citation verifier (structural + lexical
                    groundedness check against source chunks)
                                 │
                                 ▼
                      FastAPI response ──▶ Streamlit UI
```

## Project structure

```
rag-hybrid-search/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app: upload / query / list / delete endpoints
│   │   ├── config.py            All tunables, loaded from .env
│   │   ├── models.py            Pydantic schemas (Chunk, Citation, QueryResponse, ...)
│   │   ├── chunking.py          Document loaders (PDF/DOCX/TXT/MD) + recursive token-aware chunker
│   │   ├── chunk_store.py       JSON-backed metadata store (single source of truth for citations)
│   │   ├── embeddings.py        fastembed (ONNX) wrapper (shared by ingest + query) — no torch
│   │   ├── vector_store.py      ChromaDB wrapper (dense retrieval)
│   │   ├── bm25_index.py        rank_bm25 wrapper (sparse retrieval) with disk persistence
│   │   ├── hybrid_retriever.py  Reciprocal Rank Fusion of dense + BM25, then reranking
│   │   ├── reranker.py          fastembed cross-encoder reranker (final precision pass) — no torch
│   │   ├── llm.py               Groq client + grounded-answer system prompt
│   │   ├── citation_verifier.py Parses [n] citations, checks validity + lexical groundedness
│   │   ├── ingestion.py         Orchestrates load → chunk → embed → index
│   │   ├── citation_probe.py    Standalone script to audit citation verifier edge cases
│   │   └── rag_pipeline.py      Orchestrates retrieve → generate → verify
│   ├── tests/
│   │   └── test_smoke_e2e.py    Full-request-path smoke test (fakes the ML models, see below)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── streamlit_app.py         Upload UI + chat UI + citation/debug panels
│   └── requirements.txt
├── data/sample_docs/            A sample employee handbook to test with immediately
└── .gitignore
```

## Setup (no Docker plain Python/venv)

Requires Python 3.11+ and a free [Groq API key](https://console.groq.com/keys).

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...

uvicorn app.main:app --reload --port 8000
```

The first startup will download two small ONNX models via fastembed the
embedder (`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) and the reranker
(`Xenova/ms-marco-MiniLM-L-6-v2`, ~80MB) and cache them locally. No PyTorch
or `transformers` involved, so this is a quick, lightweight download, not a
multi-GB one.

Check it's up: `curl http://localhost:8000/api/health`

### 2. Frontend

In a second terminal:

```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run streamlit_app.py
```

This opens at `http://localhost:8501`. If your backend isn't on
`localhost:8000`, set `RAG_API_BASE` before launching, e.g.:

```bash
RAG_API_BASE=http://localhost:9000 streamlit run streamlit_app.py
```

### 3. Try it

Upload `data/sample_docs/northwind_employee_handbook.md` from the sidebar,
then ask something like *"How many vacation days do employees get and can
they roll them over?"* you should see a cited answer plus a sources panel
showing each citation's verification status.

## Running the smoke test
- There's a full-request-path test that fakes out the embedding model, the
reranker, and the Groq client (deterministic stand-ins, no GPU/network/API
key needed) to verify that upload → hybrid retrieval → fusion → reranking →
generation → citation verification → deletion are all wired together
correctly:

```bash
cd backend
pip install -r requirements.txt fastapi httpx  # httpx needed for TestClient
python -m tests.test_smoke_e2e
```

- This is a wiring/regression check, not a substitute for trying the real
app with real models before trusting its answers on your own documents.

## Design decisions (the "why", not just the "what")

**Why fastembed instead of sentence-transformers/torch:** the embedding and
reranking models used here are small (~80-90MB), but `sentence-transformers`
pulls in the full PyTorch + `transformers` stack to run them a multi-GB
install that's easy to collide with other ML tooling already on your machine
(a different pinned `torch`, a different `transformers`, etc.), and it's the
single biggest source of "why won't this install/import" pain in practice.
`fastembed` runs the exact same ONNX-exported models through ONNX Runtime
directly: no torch, a much smaller footprint, and it's isolated from whatever
else you have installed. Same retrieval quality, far fewer moving parts.

**Chunking**: token-aware (`via tiktoken`) and recursive it tries to split
on paragraph breaks first, then lines, then sentences, only falling back to
a hard token cut for something like a giant unbroken table. Chunks overlap
by a configurable number of tokens so facts sitting near a boundary aren't
stranded. Every chunk keeps its source filename, page number, and position,
because a citation is only as good as its metadata.

**Why hybrid retrieval, not just dense:** embeddings are great at semantic
match ("how much time off do new parents get" ↔ "parental leave") but blur
right past exact strings a product SKU, an error code, a specific policy
name. BM25 catches those. Running both and combining them beats either
alone, especially on internal docs full of domain-specific terms embeddings
were never trained on.

**Why Reciprocal Rank Fusion, not a weighted score blend:** cosine
similarity and BM25 scores live on totally different, corpus-dependent
scales, so a weighted sum (`0.5 * dense + 0.5 * bm25`) needs constant
re-tuning as your corpus changes. RRF only looks at rank position in each
list `1 / (k + rank)` which is scale-free and needs one boring, stable
constant instead of a brittle alpha. It's the same idea Elasticsearch and
Weaviate use for their built-in hybrid search.

**Why rerank after fusion, not instead of it:** a cross-encoder scores the
(query, passage) pair jointly and is far more accurate than bi-encoder
similarity or term overlap but it's too slow to run over an entire corpus.
Fusion cheaply shortlists ~10 plausible candidates from everything; the
cross-encoder then does the expensive, accurate pass over just those 10.

**Why citation verification is a separate step from generation:**
instructing the LLM to cite sources reduces hallucination but doesn't
eliminate it the model can cite a real passage number while still
drifting from what that passage says. The verifier independently (a) checks
that every [`n`] actually maps to a passage that was retrieved (catches
invented citation numbers), and (b) computes TF-IDF lexical overlap between
each cited sentence and its source passage, supplemented with light structural heuristics for number consistency and negation mismatches (catching "real citation, wrong/unsupported claim").


## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Index stats + whether `GROQ_API_KEY` is configured |
| `/api/documents/upload` | POST (multipart `file`) | Ingest a PDF/DOCX/TXT/MD file |
| `/api/documents` | GET | List ingested documents |
| `/api/documents/{doc_id}` | DELETE | Remove a document from all indexes |
| `/api/query` | POST `{question, top_k?, doc_ids?}` | Ask a question, get a cited, verified answer |

## Known limitations / what you'd add for a real production system

- **Citation groundedness is heuristic (TF-IDF + basic token checks), not deep semantic NLI.**
- It catches hallucinated numbers and direct reversals or missing details, but can still miss subtle semantic re-interpretations that reuse the source vocabulary. 
- A stronger version would add a trained NLI entailment model or a cheap LLM-as-judge pass.
- **BM25 rebuilds its index in memory on every write** (`rank_bm25` has no
  incremental update API). Fine for thousands of chunks; for a large corpus
  swap in OpenSearch/Elasticsearch's native BM25.
- **No auth, rate limiting, or multi-tenancy** add an API gateway/auth
  middleware before exposing this beyond localhost.
- **No eval harness.** For an interview follow-up, consider adding a small
  labeled Q&A set and reporting retrieval recall@k and citation-accuracy
  as a metric, not just a demo.
