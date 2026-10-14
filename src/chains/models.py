"""LangChain-only model/embedding factories. No raw SDK calls elsewhere."""
from __future__ import annotations
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import get_settings


def _lmstudio_chat(model: str, temperature: float):
    """LM Studio via OpenAI-compatible endpoint (ChatOpenAI is a LangChain class).

    Reasoning models (ornith) spend most tokens on hidden reasoning: set a high
    max_tokens so the visible answer isn't cut to '' (finish_reason=length with
    empty content was observed at the 256-token default).
    """
    s = get_settings()
    return ChatOpenAI(model=model, temperature=temperature,
                      base_url=s.LMSTUDIO_BASE_URL, api_key=s.LMSTUDIO_API_KEY,
                      max_tokens=2048, max_retries=1, request_timeout=300)


def _openrouter_chat(model: str, temperature: float):
    s = get_settings()
    if not s.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required. Add your OpenRouter key to .env."
        )
    return ChatOpenAI(model=model, temperature=temperature,
                      base_url=s.OPENROUTER_BASE_URL,
                      api_key=s.OPENROUTER_API_KEY)


def get_generator():
    """Primary answering / generation chat model (LangChain)."""
    s = get_settings()
    if s.GENERATOR_PROVIDER == "openrouter":
        return _openrouter_chat(s.GENERATOR_MODEL, 0.2)
    if s.GENERATOR_PROVIDER == "openai":
        return ChatOpenAI(model=s.GENERATOR_MODEL, api_key=s.OPENAI_API_KEY,
                          temperature=0.2)
    if s.GENERATOR_PROVIDER == "nim":
        return ChatOpenAI(model=s.NIM_MODEL, base_url=s.NVIDIA_NIM_BASE_URL,
                          api_key=s.NVIDIA_NIM_API_KEY, temperature=0.2)
    if s.GENERATOR_PROVIDER == "lmstudio":
        return _lmstudio_chat(s.GENERATOR_MODEL, 0.2)
    return ChatGoogleGenerativeAI(model=s.GENERATOR_MODEL, temperature=0.2)


def get_verifier():
    """MUST be a different vendor/model from generator (cross-model validation)."""
    s = get_settings()
    if s.VERIFIER_PROVIDER == "openrouter":
        return _openrouter_chat(s.VERIFIER_MODEL, 0.0)
    if s.VERIFIER_PROVIDER == "openai":
        return ChatOpenAI(model=s.VERIFIER_MODEL, api_key=s.OPENAI_API_KEY,
                          temperature=0.0)
    if s.VERIFIER_PROVIDER == "nim":
        return ChatOpenAI(model=s.NIM_MODEL, base_url=s.NVIDIA_NIM_BASE_URL,
                          api_key=s.NVIDIA_NIM_API_KEY, temperature=0.0)
    if s.VERIFIER_PROVIDER == "lmstudio":
        return _lmstudio_chat(s.VERIFIER_MODEL, 0.0)
    return ChatGoogleGenerativeAI(model=s.VERIFIER_MODEL, temperature=0.0)


@lru_cache
def get_embeddings():
    """Nomic-Embed-Text v1.5 via sentence-transformers, GPU if VRAM allows else CPU.

    Priority: (1) local Nomic on GPU when >= ~1.5GB free, else CPU;
    (2) HF MiniLM fallback; (3) LM Studio server embeddings (needs the nomic
    model loaded there). OpenAI is last and requires a real API key.
    """
    import torch
    s = get_settings()
    try:
        from sentence_transformers import SentenceTransformer
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: reimport
        free_mb = 0
        if torch.cuda.is_available():
            try:
                free_mb = torch.cuda.mem_get_info()[0] / 1024**2
            except Exception:
                free_mb = 0
        device = f"cuda:{torch.cuda.current_device()}" if free_mb > 1500 else "cpu"
        print(f"[embed] Nomic-Embed-Text v1.5 on {device} (GPU free: {free_mb:.0f} MB)")
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
                                    device=device, trust_remote_code=True)
        return HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",
                                     model_kwargs={"device": device},
                                     encode_kwargs={"normalize_embeddings": True})
    except Exception as e:
        print(f"[embed] Nomic failed ({e}); trying MiniLM fallback")
    try:
        return HuggingFaceEmbeddings(model_name=s.HF_EMBEDDING_MODEL)
    except Exception as e:
        print(f"[embed] MiniLM failed ({e}); trying LM Studio server")
    if s.EMBEDDING_PROVIDER in ("lmstudio", "auto"):
        try:
            return OpenAIEmbeddings(model=s.EMBEDDING_MODEL, base_url=s.LMSTUDIO_BASE_URL,
                                    api_key=s.LMSTUDIO_API_KEY,
                                    check_embedding_ctx_length=False)
        except Exception as e:
            print(f"[embed] LM Studio embeddings failed ({e})")
    raise RuntimeError(
        "No embedding backend available. Set OPENAI_API_KEY for cloud embeddings, "
        "or load text-embedding-nomic-embed-text-v1.5 in LM Studio.")
