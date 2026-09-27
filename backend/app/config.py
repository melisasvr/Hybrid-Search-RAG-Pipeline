"""
Centralized configuration for the RAG pipeline.

Everything that a hiring manager would expect to be tunable (chunk size,
retrieval fan-out, fusion constant, verification thresholds, model choice)
lives here as a single source of truth, loaded from environment variables
or a .env file. Nothing below should be hardcoded elsewhere in the app.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Groq LLM ----
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024

    # ---- Storage paths ----
    chroma_persist_dir: str = "./storage/chroma_db"
    chunk_store_path: str = "./storage/chunk_store.json"
    bm25_index_path: str = "./storage/bm25_index.pkl"
    upload_dir: str = "./storage/uploads"

    # ---- Models (served via fastembed / ONNX Runtime, no torch) ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # ---- Chunking ----
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120

    # ---- Hybrid retrieval ----
    top_k_dense: int = 15
    top_k_bm25: int = 15
    rrf_k: int = 60  # reciprocal rank fusion damping constant
    top_k_rerank_candidates: int = 10
    top_k_final: int = 5

    # ---- Citation verification ----
    citation_verified_threshold: float = 0.35
    citation_weak_threshold: float = 0.15

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
