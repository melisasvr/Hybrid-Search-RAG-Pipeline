"""
Full-stack smoke test for the FastAPI app, with the heavy ML models
(sentence-transformers embedder, cross-encoder reranker, tiktoken, Groq)
replaced by deterministic fakes.

Why fakes instead of real models: this test is meant to run fast, offline,
and in CI without a GPU, a Hugging Face download, or a real GROQ_API_KEY. It
validates that every module (chunking -> embedding -> ChromaDB -> BM25 ->
fusion -> reranking -> Groq call -> citation verification -> API responses)
is wired together correctly end-to-end at the HTTP layer. It is a wiring/
regression test, not a substitute for manually trying the real app with
real models and a real API key before you rely on its answers.

Run with:  python -m tests.test_smoke_e2e   (from the backend/ directory)
"""
import sys
import types
import hashlib
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------
# Fake tiktoken: deterministic whitespace-based "tokens" so chunking still
# behaves like real chunking algorithmically, just without the BPE download.
# ---------------------------------------------------------------------
tiktoken_stub = types.ModuleType("tiktoken")


class _FakeEncoding:
    def encode(self, text):
        return text.split(" ")

    def decode(self, tokens):
        return " ".join(tokens)


tiktoken_stub.get_encoding = lambda name: _FakeEncoding()
sys.modules["tiktoken"] = tiktoken_stub

# ---------------------------------------------------------------------
# Fake fastembed: hash-based fake embeddings (deterministic, same text ->
# same vector) so dense retrieval still returns consistent nearest-neighbor
# style results without downloading real ONNX model weights.
# ---------------------------------------------------------------------
import numpy as np

fastembed_stub = types.ModuleType("fastembed")
fastembed_rerank_stub = types.ModuleType("fastembed.rerank")
fastembed_ce_stub = types.ModuleType("fastembed.rerank.cross_encoder")


def _fake_vector(text: str, dim: int = 16):
    h = hashlib.sha256(text.encode()).digest()
    return np.array([b / 255.0 for b in h[:dim]])


class FakeTextEmbedding:
    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, texts):
        return (_fake_vector(t) for t in texts)


class FakeTextCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def rerank(self, query, documents):
        # Fake relevance: token overlap between query and passage.
        q = set(query.lower().split())
        for doc in documents:
            p = set(doc.lower().split())
            yield len(q & p) / (len(q) + 1e-6)


fastembed_stub.TextEmbedding = FakeTextEmbedding
fastembed_ce_stub.TextCrossEncoder = FakeTextCrossEncoder
sys.modules["fastembed"] = fastembed_stub
sys.modules["fastembed.rerank"] = fastembed_rerank_stub
sys.modules["fastembed.rerank.cross_encoder"] = fastembed_ce_stub

# ---------------------------------------------------------------------
# Fake Groq client: returns a canned, properly-cited answer so we can
# exercise the citation verifier on a realistic response shape.
# ---------------------------------------------------------------------
groq_stub = types.ModuleType("groq")


class _FakeChoice:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeGroq:
    def __init__(self, api_key):
        self.api_key = api_key
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, model, temperature, max_tokens, messages):
        user_msg = messages[-1]["content"]
        # naive: cite passage [1] and echo a snippet of it back
        content = (
            "Based on the provided context, the vacation policy grants 15 days of "
            "paid time off per year, accrued monthly. [1]"
        )
        return _FakeCompletion(content)


groq_stub.Groq = FakeGroq
sys.modules["groq"] = groq_stub

# ---------------------------------------------------------------------
# Now import the real app with the fakes in place
# ---------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

tmp_dir = tempfile.mkdtemp()
import os

os.environ["CHROMA_PERSIST_DIR"] = f"{tmp_dir}/chroma"
os.environ["CHUNK_STORE_PATH"] = f"{tmp_dir}/chunks.json"
os.environ["BM25_INDEX_PATH"] = f"{tmp_dir}/bm25.pkl"
os.environ["UPLOAD_DIR"] = f"{tmp_dir}/uploads"
os.environ["GROQ_API_KEY"] = "fake-key-for-smoke-test"
os.environ["CHUNK_SIZE_TOKENS"] = "60"
os.environ["CHUNK_OVERLAP_TOKENS"] = "10"

from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    print("== /api/health ==")
    r = client.get("/api/health")
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["groq_configured"] is True

    print("\n== upload a document ==")
    doc_text = (
        "Vacation Policy\n\n"
        "Employees are entitled to 15 days of paid vacation per calendar year, "
        "accrued monthly at a rate of 1.25 days per month. Unused vacation days "
        "may be carried over to the following year up to a maximum of 5 days.\n\n"
        "Sick Leave Policy\n\n"
        "Employees receive 10 days of paid sick leave annually. A doctor's note "
        "is required for absences longer than 3 consecutive days.\n\n"
    ) * 3
    r = client.post(
        "/api/documents/upload",
        files={"file": ("hr_handbook.txt", doc_text.encode(), "text/plain")},
    )
    print(r.status_code, r.json())
    assert r.status_code == 200
    doc_id = r.json()["doc_id"]
    assert r.json()["num_chunks"] > 1

    print("\n== list documents ==")
    r = client.get("/api/documents")
    print(r.status_code, r.json())
    assert len(r.json()) == 1

    print("\n== query ==")
    r = client.post("/api/query", json={"question": "How many vacation days do employees get?"})
    print(r.status_code)
    body = r.json()
    print("Answer:", body["answer"])
    print("Citations:", body["citations"])
    print("Warnings:", body["warnings"])
    print("Latency ms:", body["latency_ms"])
    assert r.status_code == 200
    assert len(body["retrieved_chunks"]) > 0
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["verification_status"] in ("verified", "weak", "unverified")

    print("\n== delete document ==")
    r = client.delete(f"/api/documents/{doc_id}")
    print(r.status_code, r.json())
    assert r.status_code == 200

    r = client.get("/api/documents")
    assert r.json() == []

print("\nALL SMOKE TESTS PASSED — full request path (upload -> hybrid retrieve -> rerank -> generate -> verify -> delete) is wired correctly.")
