"""FastAPI app exposing the SNOMED annotation pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .cache import JsonCache, make_key
from .config import settings
from .pipeline import annotate, describe_prompts, peek_result
from .vector_store import embedder_available, get_store

_EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "examples.json"
_search_cache = JsonCache("search")

# Display labels for the frontend embedder selector.
_EMBEDDERS = [
    {"id": "biolord", "label": "BioLORD-2023-M (local, multilingual)"},
    {"id": "openai", "label": "OpenAI text-embedding-3-small"},
]

app = FastAPI(title="SNOMED CT Annotator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_up() -> None:
    """Pay the embedder first-forward + FAISS page-in cost at boot, not on the
    first user request (~6-7s on macOS/MPS)."""
    try:
        get_store().search(["warmup"], top_k=1)
    except Exception:  # never block startup on warmup
        pass


class AnnotateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    use_llm_selection: bool = Field(
        True,
        description="If false, skip the LLM arbitrator and take the top-1 vector candidate per span.",
    )
    embedder: str = Field("biolord", description="Embedding backend: 'biolord' or 'openai'.")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "snomed-annotator",
        "llm_model": settings.llm_model,
        "embedder_model": settings.embedder_model,
        "index_dir": settings.index_dir,
        "default_embedder": settings.default_embedder,
    }


@app.get("/api/embedders")
def embedders_endpoint():
    """Embedder options + whether each one's index is available on disk."""
    return {
        "default": settings.default_embedder,
        "embedders": [{**e, "available": embedder_available(e["id"])} for e in _EMBEDDERS],
    }


@app.post("/api/annotate")
def annotate_endpoint(req: AnnotateRequest):
    try:
        return annotate(req.text, use_llm_selection=req.use_llm_selection, embedder=req.embedder)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # surface pipeline/LLM errors to the client
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _load_examples() -> list:
    if not _EXAMPLES_PATH.exists():
        return []
    return json.loads(_EXAMPLES_PATH.read_text("utf-8"))


@app.get("/api/examples")
def examples_endpoint():
    """The CSV `Description` texts (+ source IcdCode) for the Examples table."""
    return {"examples": _load_examples()}


@app.get("/api/examples/results")
def example_results_endpoint(embedder: str = "biolord", use_llm_selection: bool = True):
    """Persisted annotate results for each example, read straight from the disk
    cache without recomputing. `results` is aligned with the examples order;
    entries are null where that example has not been annotated yet for this
    (mode, embedder). Lets the Examples table rehydrate on load instead of
    starting blank after every reload."""
    results = [
        peek_result(ex["description"], use_llm_selection=use_llm_selection, embedder=embedder)
        for ex in _load_examples()
    ]
    return {"results": results}


@app.get("/api/prompts")
def prompts_endpoint():
    """System/user prompt templates + structured-output schemas for the two LLM
    stages (NER span extraction and the concept-selection arbitrator)."""
    return {"prompts": describe_prompts()}


@app.get("/api/search")
def search_endpoint(q: str, top_k: int = 10, embedder: str = "biolord"):
    """Direct SNOMED vector lookup (Search tab + span inspector). Cached per
    (embedder, top_k, query) so repeats — and OpenAI API calls — are free."""
    if not q.strip():
        return {"query": q, "candidates": []}
    cache_key = make_key("search", embedder, top_k, q.strip())
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return {"query": q, "candidates": cached, "embedder": embedder, "cached": True}
    try:
        candidates = get_store(embedder).search([q], top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _search_cache.set(cache_key, candidates)
    return {"query": q, "candidates": candidates, "embedder": embedder}


# Serve the production build from the same process for a one-command demo.
# Keep this mount last so /api routes take precedence.
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if (_FRONTEND_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
