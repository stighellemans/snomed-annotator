import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import faiss  # type: ignore
import numpy as np
from openai import OpenAI


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def vector_search(
    db_path: str,
    client: OpenAI,
    embedding_model: str,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    query_embedding = embed_texts(client, embedding_model, [query])[0]
    query_norm = sum(x * x for x in query_embedding) ** 0.5
    if query_norm == 0:
        return []

    best: List[Tuple[float, Dict[str, Any]]] = []
    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            embedding = row.get("embedding")
            norm = row.get("norm")
            if not embedding or not norm:
                continue
            score = sum(q * v for q, v in zip(query_embedding, embedding)) / (query_norm * norm)
            if len(best) < top_k:
                best.append((score, row))
                best.sort(key=lambda x: x[0], reverse=True)
            elif score > best[-1][0]:
                best[-1] = (score, row)
                best.sort(key=lambda x: x[0], reverse=True)

    return [_candidate_from_row(row, score) for score, row in best]


def load_faiss_meta(meta_path: str) -> List[Dict[str, Any]]:
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                meta.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return meta


def faiss_vector_search(
    index: Any,
    meta: List[Dict[str, Any]],
    client: OpenAI,
    embedding_model: str,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    query_embedding = embed_texts(client, embedding_model, [query])[0]
    query_vector = np.asarray(query_embedding, dtype="float32")
    norm = float(np.linalg.norm(query_vector))
    if norm == 0:
        return []
    query_vector = query_vector / norm
    scores, ids = index.search(query_vector.reshape(1, -1), top_k)
    candidates = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue
        candidates.append(_candidate_from_row(meta[idx], float(score)))
    return candidates


def _candidate_from_row(row: Dict[str, Any], score: float) -> Dict[str, Any]:
    term = row.get("term", "")
    return {
        "concept_id": row.get("concept_id"),
        "pt": term,
        "fsn": term if row.get("type_id") == "900000000000003001" else "",
        "matched_term": term,
        "vector_score": float(score),
    }


def find_span(text: str, span_text: str) -> Dict[str, int]:
    start = text.find(span_text)
    if start == -1:
        start = text.lower().find(span_text.lower())
        if start == -1:
            return {"start": -1, "end": -1}
    return {"start": start, "end": start + len(span_text)}


def call_openai_structured(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema_name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI response was not valid JSON") from exc


def extract_spans(client: OpenAI, model: str, text: str, max_spans: int) -> List[Dict[str, Any]]:
    system = (
        "You are a clinical NLP assistant. Extract clinically relevant spans from Dutch clinical text. "
        "Only include conditions, findings, procedures, diagnostics, symptoms, organisms, anatomy, and medications. "
        "Return spans as exact substrings copied from the source text."
    )
    user = (
        f"Extract up to {max_spans} spans. For each span, provide:\n"
        "- text: exact substring from the note\n"
        "- english_term: best English search term for SNOMED CT\n"
        "- category: one of [disorder, finding, procedure, organism, body_structure, substance, drug, symptom, test]\n\n"
        "Text:\n" + text
    )
    schema = {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "english_term": {"type": "string"},
                        "category": {"type": "string"},
                    },
                    "required": ["text", "english_term", "category"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["spans"],
        "additionalProperties": False,
    }
    return call_openai_structured(client, model, system, user, "span_extraction", schema).get("spans", [])


def suggest_search_terms(
    client: OpenAI,
    model: str,
    context: str,
    span_text: str,
    english_term: str,
    max_terms: int,
) -> List[str]:
    if max_terms <= 0:
        return []
    system = (
        "You are a clinical terminologist. Suggest concise English SNOMED CT search terms. "
        "Return short noun phrases only, no punctuation."
    )
    user = (
        "Context (Dutch):\n"
        f"{context}\n\n"
        "Span text:\n"
        f"{span_text}\n\n"
        "Current English search term:\n"
        f"{english_term}\n\n"
        f"Provide up to {max_terms} alternative search terms."
    )
    schema = {
        "type": "object",
        "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
        "required": ["terms"],
        "additionalProperties": False,
    }
    result = call_openai_structured(client, model, system, user, "term_suggestions", schema)
    terms = []
    for term in result.get("terms", []):
        cleaned = term.strip()
        if cleaned and cleaned.lower() != english_term.lower() and cleaned not in terms:
            terms.append(cleaned)
        if len(terms) >= max_terms:
            break
    return terms


def choose_best_concept(
    client: OpenAI,
    model: str,
    context: str,
    span_text: str,
    english_term: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not candidates:
        return {"status": "none", "snomed_id": "", "term": "", "confidence": 0}

    system = (
        "You are a clinical terminologist. Choose the best SNOMED CT concept from the candidate list. "
        "If none match, return status=none."
    )
    candidate_lines = [
        (
            f"- {c.get('concept_id')} | {c.get('pt')} | {c.get('fsn')} | "
            f"matched={c.get('matched_term')} | score={c.get('vector_score'):.4f}"
        )
        for c in candidates
    ]
    user = (
        "Context (Dutch):\n"
        f"{context}\n\n"
        "Span text:\n"
        f"{span_text}\n\n"
        "English term for search:\n"
        f"{english_term}\n\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\n"
        "Pick the single best concept or NONE if nothing fits."
    )
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["linked", "none"]},
            "snomed_id": {"type": "string"},
            "term": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["status", "snomed_id", "term", "confidence"],
        "additionalProperties": False,
    }
    return call_openai_structured(client, model, system, user, "concept_selection", schema)


def collect_candidates(
    search_terms: List[str],
    client: OpenAI,
    embedding_model: str,
    vector_db: str,
    faiss_index: Any,
    faiss_meta: List[Dict[str, Any]],
    top_k: int,
    min_score: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_concept: Dict[str, Dict[str, Any]] = {}
    attempts = []
    for term in search_terms:
        if faiss_index is not None:
            candidates = faiss_vector_search(faiss_index, faiss_meta, client, embedding_model, term, top_k)
            source = "faiss"
        else:
            candidates = vector_search(vector_db, client, embedding_model, term, top_k)
            source = "vector"
        candidates = [c for c in candidates if c.get("vector_score", 0) >= min_score]
        attempts.append({"term": term, "candidates": len(candidates), "source": source})
        for candidate in candidates:
            concept_id = candidate.get("concept_id")
            if not concept_id:
                continue
            existing = by_concept.get(concept_id)
            if existing is None or candidate.get("vector_score", 0) > existing.get("vector_score", 0):
                by_concept[concept_id] = candidate

    candidates = sorted(by_concept.values(), key=lambda row: row.get("vector_score", 0), reverse=True)
    return candidates[:top_k], attempts


def main() -> int:
    parser = argparse.ArgumentParser(description="Vector-only SNOMED linker using OpenAI")
    parser.add_argument("--input", default="data/example_documents.json")
    parser.add_argument("--output", default="output/linked_spans_semantic.json")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--max-spans", type=int, default=50)
    parser.add_argument("--max-search-tries", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--embedding-model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--vector-db", default="cache/snomed_vector_db.jsonl")
    parser.add_argument("--faiss-index", default="")
    parser.add_argument("--faiss-meta", default="")
    parser.add_argument("--vector-top-k", type=int, default=8)
    parser.add_argument("--vector-min-score", type=float, default=0.45)
    parser.add_argument("--log-file", default="output/run.log")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    if bool(args.faiss_index) != bool(args.faiss_meta):
        print("Provide both --faiss-index and --faiss-meta, or neither.", file=sys.stderr)
        return 2
    if args.faiss_index and (not os.path.exists(args.faiss_index) or not os.path.exists(args.faiss_meta)):
        print("FAISS index or metadata file not found.", file=sys.stderr)
        return 2
    if not args.faiss_index and not os.path.exists(args.vector_db):
        print(f"Vector DB not found: {args.vector_db}", file=sys.stderr)
        return 2

    ensure_parent_dir(args.log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Starting vector-only SNOMED linking")
    logging.info("Input=%s Output=%s Model=%s", args.input, args.output, args.model)

    client = OpenAI()
    docs = read_json(args.input)
    faiss_index = None
    faiss_meta: List[Dict[str, Any]] = []
    if args.faiss_index:
        faiss_index = faiss.read_index(args.faiss_index)
        faiss_meta = load_faiss_meta(args.faiss_meta)
        logging.info("Loaded FAISS index=%s meta=%s", args.faiss_index, args.faiss_meta)

    output = []
    for doc in docs:
        content = doc.get("content", "")
        logging.info("Doc id=%s title=%s date=%s", doc.get("id"), doc.get("title"), doc.get("date"))
        spans = extract_spans(client, args.model, content, args.max_spans)
        logging.info("Extracted spans=%s", len(spans))
        linked_spans = []

        for span in spans:
            span_text = span.get("text", "").strip()
            if not span_text:
                continue
            idx = find_span(content, span_text)
            if idx["start"] == -1:
                continue

            english_term = span.get("english_term", "").strip() or span_text
            search_terms = [english_term]
            search_terms.extend(
                suggest_search_terms(
                    client,
                    args.model,
                    context=content,
                    span_text=span_text,
                    english_term=english_term,
                    max_terms=max(0, args.max_search_tries - 1),
                )
            )
            candidates, search_attempts = collect_candidates(
                search_terms=search_terms[: max(1, args.max_search_tries)],
                client=client,
                embedding_model=args.embedding_model,
                vector_db=args.vector_db,
                faiss_index=faiss_index,
                faiss_meta=faiss_meta,
                top_k=args.vector_top_k,
                min_score=args.vector_min_score,
            )
            selection = choose_best_concept(
                client,
                args.model,
                context=content,
                span_text=span_text,
                english_term=english_term,
                candidates=candidates,
            )
            if selection.get("confidence", 0) < args.min_confidence:
                selection = {"status": "none", "snomed_id": "", "term": "", "confidence": selection.get("confidence", 0)}

            linked_spans.append(
                {
                    "text": span_text,
                    "start": idx["start"],
                    "end": idx["end"],
                    "english_term": english_term,
                    "category": span.get("category"),
                    "snomed_id": selection.get("snomed_id"),
                    "snomed_term": selection.get("term"),
                    "confidence": selection.get("confidence"),
                    "status": selection.get("status"),
                    "search_attempts": search_attempts,
                }
            )

        output.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "date": doc.get("date"),
                "spans": linked_spans,
            }
        )

    write_json(args.output, output)
    logging.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
