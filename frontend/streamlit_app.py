"""
Streamlit UI for the Hybrid Search RAG pipeline.

Talks to the FastAPI backend over HTTP only no direct imports of backend
internals — so the two can be deployed/scaled independently.
"""
import os

import requests
import streamlit as st

API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Hybrid Search RAG", page_icon="🔎", layout="wide")

STATUS_BADGE = {
    "verified": "✅ Verified",
    "weak": "⚠️ Weak match",
    "unverified": "❌ Unverified",
    "invalid_citation": "🚫 Invalid citation",
}


def api_get(path: str):
    return requests.get(f"{API_BASE}{path}", timeout=30)


def api_post(path: str, **kwargs):
    return requests.post(f"{API_BASE}{path}", timeout=120, **kwargs)


def api_delete(path: str):
    return requests.delete(f"{API_BASE}{path}", timeout=30)


def error_detail(resp: requests.Response) -> str:
    """Best-effort extraction of an error message, regardless of whether the
    backend returned JSON (the normal case) or plain text (e.g. a proxy
    timeout, or an error that occurred before FastAPI could format it)."""
    try:
        return resp.json().get("detail", resp.text)
    except (ValueError, requests.exceptions.JSONDecodeError):
        return resp.text.strip() or f"HTTP {resp.status_code} with no response body."


# ---------------------------------------------------------------------------
# Sidebar: health, ingestion, document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Hybrid Search RAG")
    st.caption("Dense (Chroma) + BM25 + cross-encoder rerank, Groq generation, verified citations.")

    try:
        health = api_get("/api/health").json()
        st.success(
            f"Backend online · {health['documents_indexed']} docs · "
            f"{health['chunks_indexed']} chunks · model: {health['groq_model']}"
        )
        if not health["groq_configured"]:
            st.warning("GROQ_API_KEY is not set on the backend — queries will fail until it is.")
    except requests.RequestException:
        st.error(f"Cannot reach backend at {API_BASE}. Is it running?")

    st.divider()
    st.subheader("Upload documents")
    uploaded = st.file_uploader(
        "PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest uploaded files", type="primary", use_container_width=True):
        for f in uploaded:
            with st.spinner(f"Chunking, embedding, and indexing '{f.name}'..."):
                resp = api_post(
                    "/api/documents/upload",
                    files={"file": (f.name, f.getvalue())},
                )
            if resp.ok:
                data = resp.json()
                st.success(f"{f.name}: {data['num_chunks']} chunks indexed.")
            else:
                st.error(f"{f.name}: {error_detail(resp)}")
        st.rerun()

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs = api_get("/api/documents").json()
    except requests.RequestException:
        docs = []

    if not docs:
        st.caption("No documents indexed yet.")
    for d in docs:
        col1, col2 = st.columns([4, 1])
        col1.write(f"📄 **{d['source']}** ({d['num_chunks']} chunks)")
        if col2.button("🗑️", key=f"del_{d['doc_id']}", help="Delete this document"):
            api_delete(f"/api/documents/{d['doc_id']}")
            st.rerun()

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Chunks used to answer (top_k)", min_value=1, max_value=10, value=5)
    filter_to_docs = st.multiselect(
        "Restrict search to specific documents (optional)",
        options=[d["source"] for d in docs],
    )
    show_debug = st.checkbox("Show retrieval debug panel", value=False)


# ---------------------------------------------------------------------------
# Main: chat-style Q&A
# ---------------------------------------------------------------------------
st.header("Ask your internal docs")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, QueryResponse dict)

for question, result in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(result["answer"])

        for w in result.get("warnings", []):
            st.warning(w)

        if result["citations"]:
            st.markdown("**Sources**")
            for c in result["citations"]:
                badge = STATUS_BADGE.get(c["verification_status"], c["verification_status"])
                page_str = f", p.{c['page']}" if c.get("page") else ""
                with st.expander(f"{c['marker']} {c['source']}{page_str} — {badge}"):
                    st.write(c["snippet"])
                    st.caption(
                        f"Groundedness score: {c['verification_score']:.2f} · "
                        f"Fused retrieval score: {c.get('retrieval_score') or 0:.4f} · "
                        f"Rerank score: {c.get('rerank_score') or 0:.4f}"
                    )

        if show_debug and result.get("retrieved_chunks"):
            with st.expander("🔧 Retrieval debug (dense vs BM25 vs fused vs rerank)"):
                rows = []
                for rc in result["retrieved_chunks"]:
                    rows.append(
                        {
                            "source": rc["chunk"]["source"],
                            "dense_rank": rc.get("dense_rank"),
                            "dense_score": round(rc["dense_score"], 4) if rc.get("dense_score") is not None else None,
                            "bm25_rank": rc.get("bm25_rank"),
                            "bm25_score": round(rc["bm25_score"], 4) if rc.get("bm25_score") is not None else None,
                            "fused_score": round(rc["fused_score"], 5) if rc.get("fused_score") is not None else None,
                            "rerank_score": round(rc["rerank_score"], 4) if rc.get("rerank_score") is not None else None,
                        }
                    )
                st.dataframe(rows, use_container_width=True)
                st.caption(f"Total latency: {result['latency_ms']} ms")

question = st.chat_input("Ask a question about the indexed documents...")
if question:
    st.session_state.history.append((question, None))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving (dense + BM25) → reranking → generating grounded answer..."):
            doc_ids = None
            if filter_to_docs:
                doc_ids = [d["doc_id"] for d in docs if d["source"] in filter_to_docs]
            resp = api_post(
                "/api/query",
                json={"question": question, "top_k": top_k, "doc_ids": doc_ids},
            )
        if resp.ok:
            result = resp.json()
            st.session_state.history[-1] = (question, result)
            st.write(result["answer"])
            for w in result.get("warnings", []):
                st.warning(w)
            if result["citations"]:
                st.markdown("**Sources**")
                for c in result["citations"]:
                    badge = STATUS_BADGE.get(c["verification_status"], c["verification_status"])
                    page_str = f", p.{c['page']}" if c.get("page") else ""
                    with st.expander(f"{c['marker']} {c['source']}{page_str} — {badge}"):
                        st.write(c["snippet"])
        else:
            st.session_state.history.pop()
            st.error(error_detail(resp))