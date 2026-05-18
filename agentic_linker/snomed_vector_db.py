import argparse
import csv
import io
import json
import logging
import os
import zipfile
from typing import Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI
import faiss  # type: ignore
import numpy as np


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _find_description_file(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        candidates = [
            name
            for name in z.namelist()
            if "Snapshot/Terminology/" in name
            and "Description_Snapshot-en_INT" in name
            and name.endswith(".txt")
        ]
        if not candidates:
            raise FileNotFoundError("No Description_Snapshot-en_INT file found in zip")
        return sorted(candidates)[-1]


def _find_concept_file(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        candidates = [
            name
            for name in z.namelist()
            if "Snapshot/Terminology/" in name
            and "Concept_Snapshot_INT" in name
            and name.endswith(".txt")
        ]
        if not candidates:
            raise FileNotFoundError("No Concept_Snapshot_INT file found in zip")
        return sorted(candidates)[-1]


def _find_language_refset_file(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        candidates = [
            name
            for name in z.namelist()
            if "Snapshot/Refset/Language/" in name
            and "LanguageSnapshot-en_INT" in name
            and name.endswith(".txt")
        ]
        if not candidates:
            raise FileNotFoundError("No LanguageSnapshot-en_INT file found in zip")
        return sorted(candidates)[-1]


def _open_description_handle(
    zip_path: Optional[str] = None,
    description_path: Optional[str] = None,
) -> Tuple[io.TextIOBase, Optional[zipfile.ZipFile]]:
    if zip_path:
        desc_file = _find_description_file(zip_path)
        zf = zipfile.ZipFile(zip_path, "r")
        raw = zf.open(desc_file, "r")
        return io.TextIOWrapper(raw, encoding="utf-8"), zf
    if description_path:
        return open(description_path, "r", encoding="utf-8"), None
    raise ValueError("Either zip_path or description_path is required")


def _count_rows(
    zip_path: Optional[str] = None,
    description_path: Optional[str] = None,
) -> int:
    handle, zf = _open_description_handle(zip_path=zip_path, description_path=description_path)
    try:
        count = -1
        for _ in handle:
            count += 1
        return max(count, 0)
    finally:
        handle.close()
        if zf:
            zf.close()


def iter_descriptions(
    zip_path: Optional[str] = None,
    description_path: Optional[str] = None,
) -> Iterable[Dict[str, str]]:
    if zip_path:
        handle, zf = _open_description_handle(zip_path=zip_path)
        try:
            yield from _iter_description_rows(handle)
        finally:
            handle.close()
            if zf:
                zf.close()
        return
    if description_path:
        handle, zf = _open_description_handle(description_path=description_path)
        try:
            yield from _iter_description_rows(handle)
        finally:
            handle.close()
            if zf:
                zf.close()
        return
    raise ValueError("Either zip_path or description_path is required")


def _iter_description_rows(handle) -> Iterable[Dict[str, str]]:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        yield row


def load_active_concepts(zip_path: Optional[str]) -> Set[str]:
    if not zip_path:
        return set()
    concept_file = _find_concept_file(zip_path)
    zf = zipfile.ZipFile(zip_path, "r")
    try:
        with zf.open(concept_file, "r") as raw:
            handle = io.TextIOWrapper(raw, encoding="utf-8")
            reader = csv.DictReader(handle, delimiter="\t")
            active: Set[str] = set()
            for row in reader:
                if row.get("active") == "1":
                    cid = row.get("id")
                    if cid:
                        active.add(cid)
            return active
    finally:
        zf.close()


def load_preferred_description_ids(zip_path: Optional[str]) -> Set[str]:
    if not zip_path:
        return set()
    preferred_acceptability = "900000000000548007"
    lang_file = _find_language_refset_file(zip_path)
    zf = zipfile.ZipFile(zip_path, "r")
    try:
        with zf.open(lang_file, "r") as raw:
            handle = io.TextIOWrapper(raw, encoding="utf-8")
            reader = csv.DictReader(handle, delimiter="\t")
            preferred: Set[str] = set()
            for row in reader:
                if row.get("active") != "1":
                    continue
                if row.get("acceptabilityId") != preferred_acceptability:
                    continue
                desc_id = row.get("referencedComponentId")
                if desc_id:
                    preferred.add(desc_id)
            return preferred
    finally:
        zf.close()


def build_vector_db(
    output_path: str,
    client: OpenAI,
    embedding_model: str,
    zip_path: Optional[str] = None,
    description_path: Optional[str] = None,
    batch_size: int = 256,
    progress_every: int = 10000,
    count_total: bool = True,
    limit: int = 0,
    start_row: int = 0,
    resume: bool = False,
    state_path: Optional[str] = None,
    concepts_only: bool = True,
    preferred_only: bool = True,
    fsn_fallback: bool = False,
) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    processed = 0
    buffer_terms: List[str] = []
    buffer_rows: List[Dict[str, str]] = []
    seen_concepts: Set[str] = set()
    total_rows = _count_rows(zip_path=zip_path, description_path=description_path) if count_total else 0
    if total_rows:
        logging.info("Description rows=%s (includes inactive/non-en).", total_rows)
    if resume and start_row:
        logging.info("Resuming from row=%s", start_row)
    if concepts_only and not zip_path:
        logging.warning("concepts_only requires --zip; disabling concepts_only.")
        concepts_only = False
    if preferred_only and not zip_path:
        logging.warning("preferred_only requires --zip; disabling preferred_only.")
        preferred_only = False
    active_concepts = load_active_concepts(zip_path) if concepts_only else set()
    if concepts_only:
        logging.info("Active concepts=%s", len(active_concepts))
    preferred_descriptions = load_preferred_description_ids(zip_path) if preferred_only else set()
    if preferred_only:
        logging.info("Preferred descriptions=%s", len(preferred_descriptions))

    def flush() -> int:
        nonlocal buffer_terms, buffer_rows, written
        if not buffer_terms:
            return 0
        embeddings = embed_texts(client, embedding_model, buffer_terms)
        with open(output_path, "a", encoding="utf-8") as out:
            for row, embedding in zip(buffer_rows, embeddings):
                norm = sum(x * x for x in embedding) ** 0.5
                out.write(
                    json.dumps(
                        {
                            "concept_id": row.get("conceptId"),
                            "term": row.get("term"),
                            "type_id": row.get("typeId"),
                            "embedding": embedding,
                            "norm": norm,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
        buffer_terms = []
        buffer_rows = []
        return written

    def process_rows(only_fsn: bool = False) -> None:
        nonlocal processed, written
        for row in iter_descriptions(zip_path=zip_path, description_path=description_path):
            processed += 1
            if resume and processed <= start_row:
                continue
            if progress_every and processed % progress_every == 0:
                if total_rows:
                    pct = processed / total_rows * 100
                    logging.info("Processed=%s (%.1f%%) Written=%s", processed, pct, written)
                else:
                    logging.info("Processed=%s Written=%s", processed, written)
            if row.get("active") != "1":
                continue
            if row.get("languageCode") != "en":
                continue
            if concepts_only and row.get("conceptId") not in active_concepts:
                continue
            if preferred_only and row.get("id") not in preferred_descriptions:
                continue
            if only_fsn and row.get("typeId") != "900000000000003001":
                continue
            concept_id = row.get("conceptId")
            if concept_id in seen_concepts:
                continue
            term = (row.get("term") or "").strip()
            if not term:
                continue
            buffer_terms.append(term)
            buffer_rows.append(row)
            seen_concepts.add(concept_id)
            if len(buffer_terms) >= batch_size:
                flush()
            if limit and written >= limit:
                break
            if state_path and progress_every and processed % progress_every == 0:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump({"processed": processed, "written": written}, f)

    process_rows(only_fsn=False)
    if fsn_fallback and concepts_only:
        logging.info("FSN fallback pass for missing concepts")
        process_rows(only_fsn=True)

    flush()
    return written


def search_vector_db(db_path: str, client: OpenAI, embedding_model: str, query: str, top_k: int) -> List[Dict[str, str]]:
    if not os.path.exists(db_path):
        return []
    embedding = embed_texts(client, embedding_model, [query])[0]
    query_norm = sum(x * x for x in embedding) ** 0.5
    if query_norm == 0:
        return []
    best: List[Dict[str, str]] = []

    def score(row: Dict[str, str]) -> float:
        vec = row.get("embedding") or []
        norm = row.get("norm") or 0
        if not vec or not norm:
            return -1.0
        return sum(q * v for q, v in zip(embedding, vec)) / (query_norm * norm)

    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_score = score(row)
            row["score"] = row_score
            if len(best) < top_k:
                best.append(row)
                best.sort(key=lambda x: x["score"], reverse=True)
            elif row_score > best[-1]["score"]:
                best[-1] = row
                best.sort(key=lambda x: x["score"], reverse=True)
    return best


def build_faiss_index(
    input_path: str,
    index_path: str,
    meta_path: str,
    dims: int,
    batch_size: int = 5000,
) -> int:
    if faiss is None or np is None:
        raise RuntimeError("faiss/numpy not available. Install faiss-cpu and numpy.")
    index = faiss.IndexFlatIP(dims)
    count = 0
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    vectors = []
    with open(meta_path, "a", encoding="utf-8") as meta_out:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                emb = row.get("embedding")
                if not emb:
                    continue
                vec = np.asarray(emb, dtype="float32")
                if vec.shape[0] != dims:
                    continue
                norm = float(np.linalg.norm(vec))
                if norm == 0:
                    continue
                vec = vec / norm
                vectors.append(vec)
                meta_out.write(
                    json.dumps(
                        {
                            "concept_id": row.get("concept_id"),
                            "term": row.get("term"),
                            "type_id": row.get("type_id"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if len(vectors) >= batch_size:
                    index.add(np.vstack(vectors))
                    count += len(vectors)
                    vectors = []
        if vectors:
            index.add(np.vstack(vectors))
            count += len(vectors)
    faiss.write_index(index, index_path)
    return count


def load_faiss_meta(meta_path: str) -> List[Dict[str, str]]:
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


def search_faiss(
    index_path: str,
    meta_path: str,
    client: OpenAI,
    embedding_model: str,
    query: str,
    top_k: int,
) -> List[Dict[str, str]]:
    if faiss is None or np is None:
        raise RuntimeError("faiss/numpy not available. Install faiss-cpu and numpy.")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return []
    index = faiss.read_index(index_path)
    meta = load_faiss_meta(meta_path)
    q = embed_texts(client, embedding_model, [query])[0]
    qv = np.asarray(q, dtype="float32")
    norm = float(np.linalg.norm(qv))
    if norm == 0:
        return []
    qv = qv / norm
    scores, ids = index.search(qv.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue
        row = meta[idx].copy()
        row["vector_score"] = float(score)
        results.append(row)
    return results


def filter_vector_db(
    input_path: str,
    output_path: str,
    prefer_fsn: bool = True,
    active_concepts: Optional[Set[str]] = None,
) -> int:
    seen: Set[str] = set()
    buffered: Dict[str, Dict[str, str]] = {}
    written = 0

    def flush_row(row: Dict[str, str]) -> None:
        nonlocal written
        with open(output_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1

    def pick_existing(concept_id: str, row: Dict[str, str]) -> None:
        existing = buffered.get(concept_id)
        if not existing:
            buffered[concept_id] = row
            return
        if prefer_fsn:
            existing_is_fsn = existing.get("type_id") == "900000000000003001"
            row_is_fsn = row.get("type_id") == "900000000000003001"
            if row_is_fsn and not existing_is_fsn:
                buffered[concept_id] = row

    if os.path.exists(output_path):
        os.remove(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            concept_id = row.get("concept_id")
            if not concept_id:
                continue
            if active_concepts is not None and concept_id not in active_concepts:
                continue
            if not prefer_fsn and concept_id in seen:
                continue
            pick_existing(concept_id, row)
            if not prefer_fsn:
                seen.add(concept_id)

    for row in buffered.values():
        flush_row(row)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/search a simple SNOMED CT vector DB")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build a JSONL vector DB from SNOMED CT descriptions")
    build.add_argument("--zip", dest="zip_path")
    build.add_argument("--description", dest="description_path")
    build.add_argument("--output", default="agentic_linker/snomed_vector_db.jsonl")
    build.add_argument("--embedding-model", default="text-embedding-3-small")
    build.add_argument("--batch-size", type=int, default=256)
    build.add_argument("--progress-every", type=int, default=10000)
    build.add_argument("--no-count-total", action="store_true")
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--start-row", type=int, default=0)
    build.add_argument("--state-path", default="agentic_linker/snomed_vector_db.state.json")
    build.add_argument("--all-descriptions", action="store_true")
    build.add_argument("--no-preferred-only", action="store_true")
    build.add_argument("--fsn-fallback", action="store_true")

    search = sub.add_parser("search", help="Search the vector DB")
    search.add_argument("--db", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--embedding-model", default="text-embedding-3-small")

    index = sub.add_parser("index", help="Build a FAISS index from an existing JSONL DB")
    index.add_argument("--input", required=True)
    index.add_argument("--index", default="agentic_linker/snomed_vector_db.index")
    index.add_argument("--meta", default="agentic_linker/snomed_vector_db.meta.jsonl")
    index.add_argument("--dims", type=int, default=1536)
    index.add_argument("--batch-size", type=int, default=5000)

    filt = sub.add_parser("filter", help="Filter an existing JSONL vector DB without re-embedding")
    filt.add_argument("--input", required=True)
    filt.add_argument("--output", default="agentic_linker/snomed_vector_db_concepts.jsonl")
    filt.add_argument("--prefer-fsn", action="store_true", default=True)
    filt.add_argument("--no-prefer-fsn", action="store_true")
    filt.add_argument("--zip", dest="zip_path")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    client = OpenAI()
    csv.field_size_limit(10 * 1024 * 1024)

    if args.cmd == "build":
        if not args.zip_path and not args.description_path:
            raise SystemExit("Provide --zip or --description")
        if os.path.exists(args.output) and not args.resume:
            os.remove(args.output)
        logging.info("Starting vector DB build")
        logging.info("Output=%s Model=%s Batch=%s", args.output, args.embedding_model, args.batch_size)
        if args.resume and os.path.exists(args.state_path) and args.start_row == 0:
            try:
                with open(args.state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                args.start_row = int(state.get("processed", 0))
                logging.info("Loaded resume state from %s (row=%s)", args.state_path, args.start_row)
            except (OSError, ValueError, json.JSONDecodeError):
                logging.warning("Failed to load resume state from %s", args.state_path)
        count = build_vector_db(
            output_path=args.output,
            client=client,
            embedding_model=args.embedding_model,
            zip_path=args.zip_path,
            description_path=args.description_path,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            count_total=not args.no_count_total,
            limit=args.limit,
            start_row=args.start_row,
            resume=args.resume,
            state_path=args.state_path if args.resume else None,
            concepts_only=not args.all_descriptions,
            preferred_only=not args.no_preferred_only,
            fsn_fallback=args.fsn_fallback,
        )
        logging.info("Wrote %s embeddings to %s", count, args.output)
        return 0

    if args.cmd == "search":
        results = search_vector_db(
            db_path=args.db,
            client=client,
            embedding_model=args.embedding_model,
            query=args.query,
            top_k=args.top_k,
        )
        for row in results:
            print(f"{row.get('score'):.4f}\t{row.get('concept_id')}\t{row.get('term')}")
        return 0

    if args.cmd == "index":
        count = build_faiss_index(
            input_path=args.input,
            index_path=args.index,
            meta_path=args.meta,
            dims=args.dims,
            batch_size=args.batch_size,
        )
        logging.info("Wrote FAISS index (%s vectors) to %s and meta to %s", count, args.index, args.meta)
        return 0

    if args.cmd == "filter":
        if args.no_prefer_fsn:
            args.prefer_fsn = False
        active = load_active_concepts(args.zip_path) if args.zip_path else None
        if active is not None:
            logging.info("Active concepts=%s", len(active))
        count = filter_vector_db(
            input_path=args.input,
            output_path=args.output,
            prefer_fsn=args.prefer_fsn,
            active_concepts=active,
        )
        logging.info("Wrote %s rows to %s", count, args.output)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
