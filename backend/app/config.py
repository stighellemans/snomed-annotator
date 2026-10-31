"""Runtime configuration, loaded from environment / .env.

All index paths default to the repo-local ``backend/data/indexes/`` tree that
``python -m indexing.build`` writes, so the app is self-contained — no sibling
repos. Override any path via .env when deploying.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Backend root = parent of the app package (.../backend). .env lives here.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_ROOT / ".env"
load_dotenv(_ENV_PATH)

_INDEXES = _BACKEND_ROOT / "data" / "indexes"
_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower().replace("-", "_")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    llm_provider: str = _LLM_PROVIDER
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "").strip()
    llm_api: str = os.environ.get("LLM_API", "").strip().lower()
    llm_json_mode: str = os.environ.get("LLM_JSON_MODE", "json_schema").strip().lower()
    llm_model: str = os.environ.get("LLM_MODEL", "gpt-5.4-mini" if _LLM_PROVIDER == "openai" else "")
    llm_timeout_seconds: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
    llm_max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
    # Reasoning effort for the gpt-5 series: minimal | low | medium | high.
    # Selection/extraction are simple structured tasks, so "low" is plenty and much faster.
    llm_reasoning_effort: str = os.environ.get("LLM_REASONING_EFFORT", "low" if _LLM_PROVIDER == "openai" else "")
    # Concurrency for the per-span selection LLM calls (they are independent).
    select_workers: int = int(os.environ.get("SELECT_WORKERS", "8"))
    embedder_model: str = os.environ.get("EMBEDDER_MODEL", "FremyCompany/BioLORD-2023-M")
    # cpu | cuda | mps. Default cpu so an Azure CPU VM works out of the box; set
    # cuda on a GPU VM, mps on Apple Silicon.
    embedder_device: str = os.environ.get("EMBEDDER_DEVICE", "cpu")
    # BioLORD index dir: faiss.index + metadata.parquet (built by indexing.build).
    index_dir: str = os.environ.get("INDEX_DIR", str(_INDEXES / "biolord"))
    # Default embedder for requests that don't specify one.
    default_embedder: str = os.environ.get("DEFAULT_EMBEDDER", "biolord")
    # OpenAI embedding backend: 1536-dim FAISS index + sqlite meta (built by
    # indexing.build). Leave OPENAI_INDEX_PATH empty to hide the option.
    openai_embed_model: str = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    openai_index_path: str = os.environ.get("OPENAI_INDEX_PATH", str(_INDEXES / "openai" / "faiss.index"))
    openai_meta_path: str = os.environ.get("OPENAI_META_PATH", str(_INDEXES / "openai" / "store.meta.sqlite"))
    top_k: int = int(os.environ.get("TOP_K", "10"))
    max_spans: int = int(os.environ.get("MAX_SPANS", "40"))
    min_confidence: float = float(os.environ.get("MIN_CONFIDENCE", "0.4"))


settings = Settings()
