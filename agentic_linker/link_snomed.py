import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import requests
from openai import OpenAI
import logging


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def snowstorm_search(base_url: str, term: str, limit: int, language: str) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/browser/MAIN/descriptions"
    params = {
        "term": term,
        "active": "true",
        "conceptActive": "true",
        "groupByConcept": "true",
        "limit": str(limit),
    }
    if language:
        params["language"] = language
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", [])
    results = []
    for item in items:
        concept = item.get("concept") or {}
        results.append(
            {
                "concept_id": concept.get("conceptId"),
                "pt": concept.get("pt"),
                "fsn": concept.get("fsn"),
                "matched_term": item.get("term"),
            }
        )
    return results[:limit]


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
    results = []
    for score, row in best:
        term = row.get("term", "")
        results.append(
            {
                "concept_id": row.get("concept_id"),
                "pt": term,
                "fsn": "",
                "matched_term": term,
                "vector_score": score,
            }
        )
    return results


def find_span(text: str, span_text: str) -> Dict[str, int]:
    start = text.find(span_text)
    if start == -1:
        lower_text = text.lower()
        lower_span = span_text.lower()
        start = lower_text.find(lower_span)
        if start == -1:
            return {"start": -1, "end": -1}
    end = start + len(span_text)
    return {"start": start, "end": end}


def call_openai_structured(client: OpenAI, model: str, system: str, user: str, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
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
    except json.JSONDecodeError:
        raise RuntimeError("OpenAI response was not valid JSON")


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
    result = call_openai_structured(client, model, system, user, "span_extraction", schema)
    return result.get("spans", [])


def choose_best_concept(
    client: OpenAI,
    model: str,
    context: str,
    span_text: str,
    english_term: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    system = (
        "You are a clinical terminologist. Choose the best SNOMED CT concept from the candidate list. "
        "If none match, return status=none."
    )
    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"- {c.get('concept_id')} | {c.get('pt')} | {c.get('fsn')} | matched={c.get('matched_term')}"
        )
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
    result = call_openai_structured(client, model, system, user, "concept_selection", schema)
    return result


def suggest_search_terms(
    client: OpenAI,
    model: str,
    context: str,
    span_text: str,
    english_term: str,
    max_terms: int,
) -> List[str]:
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
        "properties": {
            "terms": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["terms"],
        "additionalProperties": False,
    }
    result = call_openai_structured(client, model, system, user, "term_suggestions", schema)
    terms = []
    for term in result.get("terms", []):
        cleaned = term.strip()
        if cleaned and cleaned not in terms and cleaned.lower() != english_term.lower():
            terms.append(cleaned)
        if len(terms) >= max_terms:
            break
    return terms


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic SNOMED linker using Snowstorm + OpenAI")
    parser.add_argument("--input", default="documents.json")
    parser.add_argument("--output", default="linked_spans.json")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-spans", type=int, default=50)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--max-search-tries", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--vector-db", default="agentic_linker/snomed_vector_db.jsonl")
    parser.add_argument("--vector-top-k", type=int, default=8)
    parser.add_argument("--vector-min-score", type=float, default=0.45)
    parser.add_argument("--log-file", default="agentic_linker/run.log")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Starting run")
    logging.info("Input=%s Output=%s Model=%s BaseURL=%s", args.input, args.output, args.model, args.base_url)
    if args.vector_db and not os.path.exists(args.vector_db):
        logging.warning("Vector DB not found at %s (vector fallback disabled).", args.vector_db)

    client = OpenAI()
    docs = read_json(args.input)

    output = []
    for doc in docs:
        content = doc.get("content", "")
        logging.info("Doc id=%s title=%s date=%s", doc.get("id"), doc.get("title"), doc.get("date"))
        logging.info("Extracting spans (max=%s)...", args.max_spans)
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
            search_attempts = []
            terms_to_try = [english_term]
            selection = {"status": "none", "snomed_id": "", "term": "", "confidence": 0}
            had_candidates = False
            while len(terms_to_try) < max(1, args.max_search_tries):
                new_terms = suggest_search_terms(
                    client,
                    args.model,
                    context=content,
                    span_text=span_text,
                    english_term=terms_to_try[-1],
                    max_terms=args.max_search_tries - len(terms_to_try),
                )
                if not new_terms:
                    break
                terms_to_try.extend(new_terms)
            logging.info("Span='%s' terms_to_try=%s", span_text, terms_to_try[: max(1, args.max_search_tries)])
            for term in terms_to_try[: max(1, args.max_search_tries)]:
                candidates = snowstorm_search(args.base_url, term, args.candidate_limit, args.language)
                if not candidates:
                    search_attempts.append({"term": term, "candidates": 0, "status": "no_results"})
                    logging.info("Search term '%s' -> 0 candidates", term)
                    continue
                had_candidates = True
                selection = choose_best_concept(
                    client,
                    args.model,
                    context=content,
                    span_text=span_text,
                    english_term=term,
                    candidates=candidates,
                )
                search_attempts.append(
                    {
                        "term": term,
                        "candidates": len(candidates),
                        "status": selection.get("status"),
                        "confidence": selection.get("confidence"),
                    }
                )
                logging.info(
                    "Search term '%s' -> candidates=%s status=%s conf=%s snomed_id=%s",
                    term,
                    len(candidates),
                    selection.get("status"),
                    selection.get("confidence"),
                    selection.get("snomed_id"),
                )
                if selection.get("status") == "linked" and selection.get("confidence", 0) >= args.min_confidence:
                    break
            if not had_candidates and args.vector_db and os.path.exists(args.vector_db):
                vector_candidates = vector_search(
                    args.vector_db,
                    client,
                    args.embedding_model,
                    english_term,
                    args.vector_top_k,
                )
                if vector_candidates:
                    filtered = [
                        c
                        for c in vector_candidates
                        if c.get("vector_score", 0) >= args.vector_min_score
                    ]
                    if filtered:
                        selection = choose_best_concept(
                            client,
                            args.model,
                            context=content,
                            span_text=span_text,
                            english_term=english_term,
                            candidates=filtered,
                        )
                        search_attempts.append(
                            {
                                "term": english_term,
                                "candidates": len(filtered),
                                "status": selection.get("status"),
                                "confidence": selection.get("confidence"),
                                "source": "vector",
                            }
                        )
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
    logging.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
