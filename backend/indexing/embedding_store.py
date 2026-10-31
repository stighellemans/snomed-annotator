import argparse
import logging
import os
import sqlite3
from dataclasses import dataclass
from heapq import heappush, heapreplace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _init_metadata_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            idx INTEGER PRIMARY KEY,
            concept_id TEXT,
            term TEXT,
            type_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS store_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _set_store_info(conn: sqlite3.Connection, info: Dict[str, str]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO store_info(key, value) VALUES (?, ?)",
        list(info.items()),
    )


def _get_store_info(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute("SELECT key, value FROM store_info").fetchall()
    return {k: v for k, v in rows}


class EmbeddingStoreWriter:
    def __init__(
        self,
        *,
        embeddings_path: str,
        metadata_path: str,
        source_path: Optional[str] = None,
        overwrite: bool = True,
        dtype: str = "float16",
    ) -> None:
        _ensure_parent(embeddings_path)
        _ensure_parent(metadata_path)

        if overwrite:
            if os.path.exists(embeddings_path):
                os.remove(embeddings_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.source_path = source_path
        self.dtype = np.dtype(dtype)
        self.count = 0
        self.dims: Optional[int] = None
        self._closed = False
        self._conn = sqlite3.connect(metadata_path)
        _init_metadata_db(self._conn)
        self._emb_out = open(embeddings_path, "wb")

    def append(
        self,
        *,
        concept_id: Optional[str],
        term: Optional[str],
        type_id: Optional[str],
        embedding: Sequence[float],
    ) -> bool:
        vec_f32 = np.asarray(embedding, dtype=np.float32)
        if vec_f32.ndim != 1:
            return False
        if self.dims is None:
            self.dims = int(vec_f32.shape[0])
        if int(vec_f32.shape[0]) != self.dims:
            return False

        vec_store = vec_f32.astype(self.dtype, copy=False)
        vec_store.tofile(self._emb_out)
        self._conn.execute(
            "INSERT INTO metadata(idx, concept_id, term, type_id) VALUES (?, ?, ?, ?)",
            (self.count, concept_id, term, type_id),
        )
        self.count += 1
        return True

    def append_batch(
        self,
        rows: Sequence[Dict[str, object]],
        embeddings: Sequence[Sequence[float]],
        *,
        concept_key: str = "concept_id",
        term_key: str = "term",
        type_key: str = "type_id",
    ) -> int:
        written = 0
        for row, embedding in zip(rows, embeddings):
            concept_id = row.get(concept_key)
            term = row.get(term_key)
            type_id = row.get(type_key)
            if not self.append(
                concept_id=str(concept_id) if concept_id is not None else None,
                term=str(term) if term is not None else None,
                type_id=str(type_id) if type_id is not None else None,
                embedding=embedding,
            ):
                continue
            written += 1
        return written

    def commit(self) -> None:
        self._conn.commit()
        self._emb_out.flush()

    def finalize(self) -> Dict[str, int]:
        if self._closed:
            raise RuntimeError("EmbeddingStoreWriter already closed.")
        if self.dims is None or self.count == 0:
            self.close(discard=True)
            raise RuntimeError("No valid embeddings written.")

        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_concept_id ON metadata(concept_id)")
        source_path = self.source_path or ""
        _set_store_info(
            self._conn,
            {
                "count": str(self.count),
                "dims": str(self.dims),
                "dtype": str(self.dtype),
                "row_bytes": str(self.dtype.itemsize * self.dims),
                "source_path": source_path,
            },
        )
        self._conn.commit()
        self.close(discard=False)
        return {
            "count": self.count,
            "dims": self.dims,
            "embedding_bytes": os.path.getsize(self.embeddings_path),
            "metadata_bytes": os.path.getsize(self.metadata_path),
        }

    def close(self, discard: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._emb_out.close()
        finally:
            self._conn.close()
        if discard:
            if os.path.exists(self.embeddings_path):
                os.remove(self.embeddings_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)

@dataclass
class EmbeddingStore:
    embeddings_path: str
    metadata_path: str
    count: int
    dims: int
    dtype: np.dtype
    embeddings: np.memmap
    conn: sqlite3.Connection

    def close(self) -> None:
        self.conn.close()


def open_store(embeddings_path: str, metadata_path: str) -> EmbeddingStore:
    conn = sqlite3.connect(metadata_path)
    info = _get_store_info(conn)
    if not info:
        conn.close()
        raise RuntimeError(f"Missing store_info in metadata file: {metadata_path}")

    count = int(info["count"])
    dims = int(info["dims"])
    dtype = np.dtype(info.get("dtype", "float16"))

    expected_bytes = count * dims * dtype.itemsize
    actual_bytes = os.path.getsize(embeddings_path)
    if expected_bytes != actual_bytes:
        conn.close()
        raise RuntimeError(
            f"Embedding file size mismatch: expected={expected_bytes} actual={actual_bytes}"
        )

    embeddings = np.memmap(
        embeddings_path,
        dtype=dtype,
        mode="r",
        shape=(count, dims),
    )
    return EmbeddingStore(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        count=count,
        dims=dims,
        dtype=dtype,
        embeddings=embeddings,
        conn=conn,
    )


def estimate_memory(
    *,
    count: int,
    dims: int,
    stored_dtype: np.dtype,
    chunk_size: int,
) -> Dict[str, int]:
    stored_bytes = count * dims * stored_dtype.itemsize
    float32_bytes = count * dims * np.dtype("float32").itemsize
    query_bytes = dims * np.dtype("float32").itemsize
    chunk_matrix_bytes = chunk_size * dims * np.dtype("float32").itemsize
    chunk_scores_bytes = chunk_size * np.dtype("float32").itemsize

    return {
        "stored_embedding_bytes": stored_bytes,
        "float32_embedding_bytes": float32_bytes,
        "query_bytes": query_bytes,
        "chunk_matrix_bytes": chunk_matrix_bytes,
        "chunk_scores_bytes": chunk_scores_bytes,
        "search_working_set_bytes": query_bytes + chunk_matrix_bytes + chunk_scores_bytes,
    }


def top_k_cosine(
    store: EmbeddingStore,
    query_vector: Sequence[float],
    *,
    top_k: int = 8,
    chunk_size: int = 8192,
) -> List[Dict[str, object]]:
    if top_k <= 0:
        return []

    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if q.shape[0] != store.dims:
        raise ValueError(f"Expected query dims={store.dims}, got={q.shape[0]}")

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []

    heap: List[Tuple[float, int]] = []
    for start in range(0, store.count, chunk_size):
        end = min(start + chunk_size, store.count)
        block = np.asarray(store.embeddings[start:end], dtype=np.float32)
        dots = block @ q
        block_norms = np.linalg.norm(block, axis=1)
        denom = block_norms * q_norm
        scores = np.divide(
            dots,
            denom,
            out=np.full(dots.shape, -np.inf, dtype=np.float32),
            where=denom > 0,
        )

        k_local = min(top_k, scores.shape[0])
        local_ids = np.argpartition(scores, -k_local)[-k_local:]
        for local_idx in local_ids:
            score = float(scores[int(local_idx)])
            global_idx = start + int(local_idx)
            if len(heap) < top_k:
                heappush(heap, (score, global_idx))
            elif score > heap[0][0]:
                heapreplace(heap, (score, global_idx))

    ranked = sorted(heap, key=lambda item: item[0], reverse=True)
    results: List[Dict[str, object]] = []
    for score, idx in ranked:
        meta = store.conn.execute(
            "SELECT concept_id, term, type_id FROM metadata WHERE idx = ?",
            (idx,),
        ).fetchone()
        if not meta:
            continue
        concept_id, term, type_id = meta
        results.append(
            {
                "idx": idx,
                "concept_id": concept_id,
                "term": term,
                "type_id": type_id,
                "score": score,
            }
        )
    return results


def filter_store_by_concept(
    *,
    source_embeddings_path: str,
    source_metadata_path: str,
    output_embeddings_path: str,
    output_metadata_path: str,
    prefer_fsn: bool = True,
) -> Dict[str, int]:
    source = open_store(source_embeddings_path, source_metadata_path)
    writer = EmbeddingStoreWriter(
        embeddings_path=output_embeddings_path,
        metadata_path=output_metadata_path,
        source_path=source_embeddings_path,
        overwrite=True,
        dtype="float16",
    )
    try:
        picked: Dict[str, Tuple[int, Optional[str], Optional[str]]] = {}
        fsn_type = "900000000000003001"
        for idx, concept_id, term, type_id in source.conn.execute(
            "SELECT idx, concept_id, term, type_id FROM metadata ORDER BY idx"
        ):
            if not concept_id:
                continue
            existing = picked.get(concept_id)
            if not existing:
                picked[concept_id] = (int(idx), term, type_id)
                continue
            if not prefer_fsn:
                continue
            _, _, existing_type = existing
            if type_id == fsn_type and existing_type != fsn_type:
                picked[concept_id] = (int(idx), term, type_id)

        for concept_id, (idx, term, type_id) in picked.items():
            vec = np.asarray(source.embeddings[idx], dtype=np.float32)
            writer.append(
                concept_id=concept_id,
                term=term,
                type_id=type_id,
                embedding=vec,
            )
        summary = writer.finalize()
        summary["source_rows"] = source.count
        summary["filtered_concepts"] = len(picked)
        return summary
    except Exception:
        writer.close(discard=True)
        raise
    finally:
        source.close()


def _print_report(
    *,
    source_path: Optional[str],
    embeddings_path: str,
    metadata_path: str,
    count: int,
    dims: int,
    dtype: np.dtype,
    chunk_size: int,
) -> None:
    embedding_bytes = os.path.getsize(embeddings_path)
    metadata_bytes = os.path.getsize(metadata_path)
    total_bytes = embedding_bytes + metadata_bytes
    mem = estimate_memory(count=count, dims=dims, stored_dtype=dtype, chunk_size=chunk_size)

    print(f"rows={count} dims={dims} dtype={dtype}")
    print(f"embeddings_file={embeddings_path} ({human_bytes(embedding_bytes)})")
    print(f"metadata_file={metadata_path} ({human_bytes(metadata_bytes)})")
    print(f"total_file_size={human_bytes(total_bytes)}")
    print(
        "equivalent_float32_embeddings="
        f"{human_bytes(mem['float32_embedding_bytes'])}"
        f" (saves {human_bytes(mem['float32_embedding_bytes'] - mem['stored_embedding_bytes'])})"
    )
    print(
        "estimated_search_working_set="
        f"{human_bytes(mem['search_working_set_bytes'])}"
        f" (chunk_size={chunk_size})"
    )

    if source_path and os.path.exists(source_path):
        source_bytes = os.path.getsize(source_path)
        saved = source_bytes - total_bytes
        ratio = total_bytes / source_bytes if source_bytes else 0.0
        print(f"source_input={source_path} ({human_bytes(source_bytes)})")
        print(f"saved_vs_source={human_bytes(saved)} ({(1.0 - ratio) * 100:.2f}% smaller)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and transform SNOMED embedding stores")
    sub = parser.add_subparsers(dest="cmd", required=True)

    report = sub.add_parser("report", help="Print size/memory report for an existing store")
    report.add_argument("--embeddings", required=True)
    report.add_argument("--metadata", required=True)
    report.add_argument("--source", help="Optional source file for size comparison")
    report.add_argument("--chunk-size", type=int, default=8192)

    filt = sub.add_parser(
        "filter-concepts",
        help="Build a concept-only store (one row per concept) from an existing store",
    )
    filt.add_argument("--source-embeddings", required=True)
    filt.add_argument("--source-metadata", required=True)
    filt.add_argument("--embeddings", required=True)
    filt.add_argument("--metadata", required=True)
    filt.add_argument("--no-prefer-fsn", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.cmd == "report":
        store = open_store(args.embeddings, args.metadata)
        try:
            _print_report(
                source_path=args.source,
                embeddings_path=args.embeddings,
                metadata_path=args.metadata,
                count=store.count,
                dims=store.dims,
                dtype=store.dtype,
                chunk_size=args.chunk_size,
            )
        finally:
            store.close()
        return 0

    if args.cmd == "filter-concepts":
        summary = filter_store_by_concept(
            source_embeddings_path=args.source_embeddings,
            source_metadata_path=args.source_metadata,
            output_embeddings_path=args.embeddings,
            output_metadata_path=args.metadata,
            prefer_fsn=not args.no_prefer_fsn,
        )
        print(
            "source_rows="
            f"{summary['source_rows']} filtered_rows={summary['count']} filtered_concepts={summary['filtered_concepts']}"
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
