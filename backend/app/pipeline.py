"""Annotation pipeline: LLM span extraction -> vector retrieval -> LLM selection.

Multilingual variant: because the index uses multilingual BioLORD, we embed the
Dutch span directly. The LLM also proposes a normalized clinical search term
(English preferred, since SNOMED descriptions are English) which we use as a
second query to improve recall. Candidates from both queries are merged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .cache import JsonCache, make_key
from .config import settings
from .llm import cache_identity, call_structured
from .vector_store import get_store

CATEGORIES = "[disorder, finding, procedure, organism, body_structure, substance, drug, symptom, test]"

# Bump when the extraction prompt changes so cached NER is invalidated.
EXTRACTION_PROMPT_VERSION = "2-negation"
# Bump when selection prompt / result shape changes so cached results are invalidated.
RESULT_VERSION = "1"

_ner_cache = JsonCache("ner")
_result_cache = JsonCache("results")

# --- Prompt templates (also surfaced verbatim by the Prompts tab) -------------

NER_SYSTEM = (
    "You are a clinical NLP assistant. Extract clinically relevant spans from Dutch clinical text. "
    "Only include conditions, findings, procedures, diagnostics, symptoms, organisms, anatomy, and medications. "
    "Return spans as exact substrings copied verbatim from the source text. "
    "Do NOT extract negated or absent findings: when a finding is negated or ruled out by cues such as "
    "'geen', 'niet', 'zonder', 'noch', 'afwezig', 'uitgesloten', 'geen tekenen van', or 'geen aanwijzingen voor', "
    "skip it entirely. Only extract findings asserted to be present (currently or in the past)."
)


def ner_user(text: str, max_spans: int) -> str:
    return (
        f"Extract up to {max_spans} spans. For each span, provide:\n"
        "- text: exact substring from the note (verbatim, preserving casing)\n"
        "- search_term: a concise normalized clinical term for SNOMED CT search (English preferred)\n"
        f"- category: one of {CATEGORIES}\n\n"
        "Exclude anything that is negated or stated to be absent.\n\n"
        "Text:\n" + text
    )


NER_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "search_term": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["text", "search_term", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["spans"],
    "additionalProperties": False,
}

ARBITRATOR_SYSTEM = (
    "You are a clinical terminologist. Choose the single best SNOMED CT concept for the span "
    "from the candidate list, using the surrounding Dutch context. If none truly match, return status=none."
)


def arbitrator_user(context: str, span_text: str, search_term: str, candidates_block: str) -> str:
    return (
        "Context (Dutch):\n"
        f"{context}\n\n"
        "Span text:\n"
        f"{span_text}\n\n"
        "Normalized search term:\n"
        f"{search_term}\n\n"
        "Candidates:\n" + candidates_block + "\n\n"
        "Pick the single best concept_id, or status=none if nothing fits."
    )


ARBITRATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["linked", "none"]},
        "concept_id": {"type": "string"},
        "term": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["status", "concept_id", "term", "confidence"],
    "additionalProperties": False,
}


def describe_prompts() -> List[Dict[str, Any]]:
    """Static description of the two LLM prompts, for the Prompts tab. Per-request
    content (note text, span, candidates) is shown as ‹placeholders›; configured
    values (max_spans, top_k) are rendered as their current settings."""
    candidates_block = (
        "‹one line per retrieved candidate, e.g.›\n"
        "- 49436004 | Atrial fibrillation (disorder) | tag=disorder | score=0.8421"
    )
    return [
        {
            "id": "ner",
            "title": "Span extraction (NER)",
            "purpose": (
                "Pulls clinically relevant spans out of the Dutch note. Runs once per note, "
                "then cached and reused by both arbitrator modes."
            ),
            "model": settings.llm_model,
            "version": EXTRACTION_PROMPT_VERSION,
            "structured_output": "span_extraction",
            "system": NER_SYSTEM,
            "user": ner_user("‹the Dutch clinical note›", settings.max_spans),
            "schema": NER_SCHEMA,
        },
        {
            "id": "arbitrator",
            "title": "Concept selection (LLM arbitrator)",
            "purpose": (
                f"Picks the single best SNOMED concept for a span from its top-{settings.top_k} vector "
                "candidates. One call per span; skipped entirely in top-1 vector mode."
            ),
            "model": settings.llm_model,
            "version": RESULT_VERSION,
            "structured_output": "concept_selection",
            "system": ARBITRATOR_SYSTEM,
            "user": arbitrator_user(
                "‹surrounding Dutch context (the full note)›",
                "‹span text›",
                "‹normalized search term›",
                candidates_block,
            ),
            "schema": ARBITRATOR_SCHEMA,
        },
    ]


def extract_spans(text: str, max_spans: int) -> List[Dict[str, Any]]:
    # NER is identical regardless of arbitrator mode, so cache + reuse it.
    key = make_key("ner", EXTRACTION_PROMPT_VERSION, cache_identity(), max_spans, text)
    cached = _ner_cache.get(key)
    if cached is not None:
        return cached
    spans = call_structured(
        NER_SYSTEM, ner_user(text, max_spans), "span_extraction", NER_SCHEMA
    ).get("spans", [])
    _ner_cache.set(key, spans)
    return spans


def choose_best_concept(
    context: str,
    span_text: str,
    search_term: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not candidates:
        return {"status": "none", "concept_id": "", "term": "", "confidence": 0.0}
    candidate_lines = [
        f"- {c['concept_id']} | {c['fsn']} | tag={c['semantic_tag']} | score={c['vector_score']:.4f}"
        for c in candidates
    ]
    user = arbitrator_user(context, span_text, search_term, "\n".join(candidate_lines))
    return call_structured(ARBITRATOR_SYSTEM, user, "concept_selection", ARBITRATOR_SCHEMA)


def _locate(text: str, span_text: str, cursor: int) -> Dict[str, int]:
    """Find span_text at or after cursor, falling back to a global / case-insensitive search."""
    start = text.find(span_text, cursor)
    if start == -1:
        start = text.find(span_text)
    if start == -1:
        low = text.lower().find(span_text.lower(), cursor)
        if low == -1:
            low = text.lower().find(span_text.lower())
        start = low
    if start == -1:
        return {"start": -1, "end": -1}
    return {"start": start, "end": start + len(span_text)}


def _top1(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Vector-only selection: take the highest-scoring candidate, no LLM arbitration."""
    if not candidates:
        return {"status": "none", "concept_id": "", "term": "", "confidence": 0.0}
    top = candidates[0]
    return {
        "status": "linked",
        "concept_id": top["concept_id"],
        "term": top["fsn"],
        "confidence": float(top["vector_score"]),
    }


def _result_key(text: str, use_llm_selection: bool, embedder: str) -> str:
    """Key for the full-result cache: everything that affects the output (including
    the embedder, since candidates differ per index), so the same example never
    recomputes. NER is cached separately and reused across embedders AND modes."""
    return make_key(
        "result",
        RESULT_VERSION,
        EXTRACTION_PROMPT_VERSION,
        use_llm_selection,
        embedder,
        cache_identity(),
        settings.max_spans,
        settings.top_k,
        settings.min_confidence,
        text,
    )


def peek_result(text: str, use_llm_selection: bool = True, embedder: str = None) -> Dict[str, Any] | None:
    """Return the persisted result for this input if present, WITHOUT computing it.
    Used to rehydrate the Examples table from disk on load."""
    embedder = embedder or settings.default_embedder
    return _result_cache.get(_result_key(text, use_llm_selection, embedder))


def annotate(text: str, use_llm_selection: bool = True, embedder: str = None) -> Dict[str, Any]:
    embedder = embedder or settings.default_embedder
    result_key = _result_key(text, use_llm_selection, embedder)
    cached = _result_cache.get(result_key)
    if cached is not None:
        return cached

    store = get_store(embedder)
    spans = extract_spans(text, settings.max_spans)

    # Phase 1 (sequential): locate spans + retrieve candidates. Locating keeps a
    # moving cursor to disambiguate repeated spans, so it must stay ordered.
    # Retrieval is fast (~0.1s/span once warm).
    pending: List[Dict[str, Any]] = []
    cursor = 0
    for span in spans:
        span_text = (span.get("text") or "").strip()
        if not span_text:
            continue
        loc = _locate(text, span_text, cursor)
        if loc["start"] == -1:
            continue
        cursor = loc["end"]
        search_term = (span.get("search_term") or "").strip()
        pending.append(
            {
                "span_text": span_text,
                "start": loc["start"],
                "end": loc["end"],
                "category": span.get("category", ""),
                "search_term": search_term,
                "candidates": store.search([span_text, search_term], top_k=settings.top_k),
            }
        )

    # Phase 2: pick a concept per span. With the LLM arbitrator on, the per-span
    # selection calls are independent, so fan them out (latency collapses from
    # sum-of-calls to slowest single call). With it off, take the top-1 vector
    # candidate directly — no LLM, effectively instant.
    if not pending:
        selections = []
    elif use_llm_selection:
        def _select(item: Dict[str, Any]) -> Dict[str, Any]:
            return choose_best_concept(
                text, item["span_text"], item["search_term"], item["candidates"]
            )

        workers = min(settings.select_workers, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            selections = list(pool.map(_select, pending))
    else:
        selections = [_top1(item["candidates"]) for item in pending]

    results: List[Dict[str, Any]] = []
    for item, selection in zip(pending, selections):
        confidence = float(selection.get("confidence") or 0.0)
        status = selection.get("status", "none")
        chosen_id = str(selection.get("concept_id") or "")
        candidate_ids = {str(candidate["concept_id"]) for candidate in item["candidates"]}
        if status != "linked" or confidence < settings.min_confidence or chosen_id not in candidate_ids:
            chosen_id = ""
            status = "none"
        results.append(
            {
                "text": item["span_text"],
                "start": item["start"],
                "end": item["end"],
                "category": item["category"],
                "search_term": item["search_term"],
                "concept_id": chosen_id,
                "snomed_term": store.concept_label(chosen_id) if chosen_id else "",
                "confidence": confidence,
                "status": status,
                "candidates": item["candidates"],
            }
        )
    output = {"text": text, "spans": results}
    _result_cache.set(result_key, output)
    return output
