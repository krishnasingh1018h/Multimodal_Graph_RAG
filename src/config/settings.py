"""Central settings. All secrets from env; models/embeddings chosen here."""
from __future__ import annotations
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM providers (generator + distinct verifier) ---
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    NVIDIA_NIM_API_KEY: str = ""
    NVIDIA_NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

    # Local LM Studio (OpenAI-compatible).
    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LMSTUDIO_API_KEY: str = "lm-studio"

    GENERATOR_PROVIDER: str = "openrouter"   # openrouter | google | openai | nim | lmstudio
    GENERATOR_MODEL: str = "nvidia/nemotron-3.5-lightning:free"
    VERIFIER_PROVIDER: str = "openrouter"
    VERIFIER_MODEL: str = "cohere/north-mini-code:free"
    NIM_MODEL: str = "meta/llama-3.1-70b-instruct"

    # --- Reranker (cross-encoder, CPU-safe) ---
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_DEVICE: str = "cpu"           # force cpu: GPU VRAM held by LM Studio
    RERANKER_TOPN: int = 12                # candidates scored, top TOP_K returned
    USE_BM25: bool = True

    # --- Embeddings / vector store ---
    # Nomic-Embed-Text v1.5 local (GPU if VRAM allows, else CPU). Chroma stores
    # whatever dim the embedder returns — wipe data/chroma_db when switching.
    EMBEDDING_PROVIDER: str = "nomic"  # nomic (local) | huggingface | lmstudio
    EMBEDDING_MODEL: str = "text-embedding-nomic-embed-text-v1.5"
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    NOMIC_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    NOMIC_MIN_GPU_MB: int = 1500
    VECTOR_BACKEND: str = "chroma"       # chroma (installed) | qdrant (optional)
    CHROMA_DIR: str = "data/chroma_db"
    COLLECTION: str = "learning_companion"
    QDRANT_URL: str = "http://localhost:6333"

    # --- Retrieval / grounding ---
    TOP_K: int = 6
    GRADE_THRESHOLD: float = 0.35
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 135

    # --- Persistence ---
    CHECKPOINT_DB: str = "data/checkpoints/graph.db"
    BKT_DB: str = "data/checkpoints/bkt.json"
    QUESTION_BANK: str = "data/checkpoints/questions.jsonl"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # map legacy NVIDIA_NIM key name in existing .env
    if not s.NVIDIA_NIM_API_KEY:
        s.NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM", "")
    return s
