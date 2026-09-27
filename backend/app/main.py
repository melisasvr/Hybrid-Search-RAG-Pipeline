"""
FastAPI entrypoint. Wires together the vector store, BM25 index, chunk
store, embedder, reranker, Groq client, and RAG pipeline as app-level
singletons created once at startup, then exposes them through a handful of
endpoints for ingestion and querying.
"""
from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .bm25_index import BM25Index
from .chunk_store import ChunkStore
from .chunking import LOADERS
from .config import get_settings
from .embeddings import get_embedder
from .hybrid_retriever import HybridRetriever
from .ingestion import ingest_file
from .llm import GroqLLM
from .models import DocumentInfo, IngestResponse, QueryRequest, QueryResponse
from .rag_pipeline import RAGPipeline
from .reranker import get_reranker
from .vector_store import VectorStore

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    embedder = get_embedder(settings.embedding_model)
    vector_store = VectorStore(settings.chroma_persist_dir)
    bm25_index = BM25Index(settings.bm25_index_path)
    chunk_store = ChunkStore(settings.chunk_store_path)
    reranker = get_reranker(settings.reranker_model)

    retriever = HybridRetriever(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        chunk_store=chunk_store,
        reranker=reranker,
    )

    llm = None
    if settings.groq_api_key:
        llm = GroqLLM(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
        )

    state.update(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        chunk_store=chunk_store,
        reranker=reranker,
        retriever=retriever,
        llm=llm,
        pipeline=RAGPipeline(settings, retriever, llm) if llm else None,
    )
    yield
    state.clear()


app = FastAPI(
    title="Hybrid Search RAG API",
    description="Production-style RAG over internal docs: dense + BM25 + reranking, with citation verification.",
    version="1.0.0",
    lifespan=lifespan,
)

_settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _settings_for_cors.cors_origins == "*" else _settings_for_cors.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Without this, an unhandled exception anywhere in a route (a bad Groq API
    key, a rate limit, a bug) falls through to Starlette's default handler,
    which returns plain text, not JSON. The frontend expects every error
    response to be JSON with a `detail` field, so a plain-text 500 crashes
    the *frontend* trying to parse it instead of showing the real error.
    This ensures every failure is at least reported as readable JSON.
    """
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


def _require_llm() -> GroqLLM:
    if state.get("llm") is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server. Set it in backend/.env and restart.",
        )
    return state["llm"]


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents_indexed": len(state["chunk_store"].list_documents()),
        "chunks_indexed": state["vector_store"].count(),
        "groq_configured": state.get("llm") is not None,
        "groq_model": state["settings"].groq_model,
    }


@app.post("/api/documents/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    settings = state["settings"]
    ext = Path(file.filename or "").suffix.lower()
    if ext not in LOADERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(LOADERS.keys())}",
        )

    tmp_path = Path(settings.upload_dir) / f"{uuid.uuid4().hex}{ext}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_file(
            path=tmp_path,
            original_filename=file.filename or tmp_path.name,
            settings=settings,
            embedder=state["embedder"],
            vector_store=state["vector_store"],
            bm25_index=state["bm25_index"],
            chunk_store=state["chunk_store"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in state["chunk_store"].list_documents()]


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    removed = state["chunk_store"].delete_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document with id '{doc_id}' found.")
    state["vector_store"].delete_document(doc_id)
    state["bm25_index"].delete_document(lambda cid: cid.startswith(doc_id))
    return {"doc_id": doc_id, "chunks_removed": removed}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    _require_llm()
    pipeline: RAGPipeline = state["pipeline"]
    return pipeline.answer(request.question, top_k=request.top_k, doc_ids=request.doc_ids)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)