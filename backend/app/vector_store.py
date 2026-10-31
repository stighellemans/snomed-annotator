"""FAISS-backed SNOMED vector search, supporting multiple embedding backends.

- biolord: local BioLORD-2023-M, FAISS IndexFlat + parquet metadata (768-dim).
- openai:  text-embedding-3-small, FAISS IndexIDMap2 + sqlite metadata (1536-dim).

Both indices are built locally by ``python -m indexing.build`` into
``backend/data/indexes/{biolord,openai}/`` — no external repos.

Both return the same candidate shape. The OpenAI sqlite lacks FSN / semantic_tag,
so those display fields are enriched from the BioLORD parquet (same SNOMED), keyed
by concept_id.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import settings
from .embedder import get_embedder


# ---- shared concept display info (FSN + semantic tag per concept) -------------
_concept_info: Tuple[Dict[str, str], Dict[str, str]] | None = None


def get_concept_info() -> Tuple[Dict[str, str], Dict[str, str]]:
    """(concept_id -> FSN label, concept_id -> semantic_tag), loaded once from the
    BioLORD parquet. Used to enrich both backends' results."""
    global _concept_info
    if _concept_info is None:
        meta_path = Path(settings.index_dir) / "metadata.parquet"
        if not meta_path.exists():
            # BioLORD parquet not built (e.g. OpenAI backend used alone). Fall
            # back to empty maps; OpenAIStore then displays the description term.
            _concept_info = ({}, {})
            return _concept_info
        df = pd.read_parquet(meta_path, columns=["concept_id", "term", "semantic_tag", "is_fsn"])
        label: Dict[str, str] = {}
        tag: Dict[str, str] = {}
        fsn = df[df["is_fsn"] == True]  # noqa: E712
        for cid, term in zip(fsn["concept_id"].astype(str), fsn["term"].astype(str)):
            label.setdefault(cid, term)
        for cid, t in zip(df["concept_id"].astype(str), df["semantic_tag"].astype(str)):
            tag.setdefault(cid, t)
        _concept_info = (label, tag)
    return _concept_info


class BaseStore:
    embedder_name = "biolord"

    def __init__(self) -> None:
        self.index = None  # set by subclass
        self._label, self._tag = get_concept_info()

    def concept_label(self, concept_id: str) -> str:
        return self._label.get(str(concept_id), "")

    def _lookup_batch(self, idxs: Sequence[int]) -> Dict[int, Dict[str, str]]:
        """idx -> {concept_id, term, term_no_tag}. Implemented per backend."""
        raise NotImplementedError

    def search(self, queries: Sequence[str], top_k: int = 10, overfetch: int = 6) -> List[Dict[str, object]]:
        queries = [q for q in {q.strip() for q in queries} if q]
        if not queries:
            return []
        vectors = get_embedder(self.embedder_name).encode(queries).astype("float32")
        fetch_k = max(top_k * overfetch, top_k)
        scores, ids = self.index.search(vectors, fetch_k)

        all_idx = {int(i) for row in ids for i in row if i >= 0}
        meta_by_idx = self._lookup_batch(sorted(all_idx))

        by_concept: Dict[str, Dict[str, object]] = {}
        for row_scores, row_ids, query in zip(scores, ids, queries):
            for score, idx in zip(row_scores, row_ids):
                if idx < 0:
                    continue
                meta = meta_by_idx.get(int(idx))
                if not meta:
                    continue
                concept_id = meta["concept_id"]
                score = float(score)
                existing = by_concept.get(concept_id)
                if existing is not None and score <= float(existing["vector_score"]):
                    continue
                by_concept[concept_id] = {
                    "concept_id": concept_id,
                    "term": meta["term_no_tag"],
                    "fsn": self._label.get(concept_id) or meta["term"],
                    "semantic_tag": self._tag.get(concept_id, ""),
                    "matched_term": meta["term"],
                    "vector_score": score,
                    "matched_query": query,
                }
        ranked = sorted(by_concept.values(), key=lambda r: r["vector_score"], reverse=True)
        return ranked[:top_k]


class BiolordStore(BaseStore):
    embedder_name = "biolord"

    def __init__(self, index_dir: str):
        import faiss

        super().__init__()
        root = Path(index_dir)
        index_path = root / "faiss.index"
        meta_path = root / "metadata.parquet"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"BioLORD index not found in {root}. Expected faiss.index + metadata.parquet."
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata: pd.DataFrame = pd.read_parquet(meta_path)

    def _lookup_batch(self, idxs: Sequence[int]) -> Dict[int, Dict[str, str]]:
        if not idxs:
            return {}
        rows = self.metadata.iloc[list(idxs)]
        out: Dict[int, Dict[str, str]] = {}
        for idx, term, term_no_tag, cid in zip(
            idxs,
            rows["term"].astype(str),
            rows.get("term_no_tag", rows["term"]).astype(str),
            rows["concept_id"].astype(str),
        ):
            out[int(idx)] = {"concept_id": cid, "term": term, "term_no_tag": term_no_tag}
        return out


class OpenAIStore(BaseStore):
    embedder_name = "openai"

    def __init__(self, index_path: str, meta_path: str):
        import faiss

        super().__init__()
        if not Path(index_path).exists() or not Path(meta_path).exists():
            raise FileNotFoundError(
                f"OpenAI index not found ({index_path} / {meta_path}). Set OPENAI_INDEX_PATH."
            )
        self.index = faiss.read_index(str(index_path))
        # Read-only, multi-thread safe connection (search may run from worker threads).
        self._conn = sqlite3.connect(f"file:{meta_path}?mode=ro", uri=True, check_same_thread=False)

    def _lookup_batch(self, idxs: Sequence[int]) -> Dict[int, Dict[str, str]]:
        if not idxs:
            return {}
        out: Dict[int, Dict[str, str]] = {}
        CHUNK = 900  # stay under SQLite's variable limit
        for start in range(0, len(idxs), CHUNK):
            chunk = idxs[start : start + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT idx, concept_id, term FROM metadata WHERE idx IN ({placeholders})",
                tuple(int(i) for i in chunk),
            )
            for idx, cid, term in cur:
                term = str(term)
                out[int(idx)] = {"concept_id": str(cid), "term": term, "term_no_tag": term}
        return out


# ---- name-keyed store registry (lazy) ----------------------------------------
_stores: Dict[str, BaseStore] = {}


def embedder_available(name: str) -> bool:
    if name == "openai":
        return bool(settings.openai_index_path) and Path(settings.openai_index_path).exists() \
            and Path(settings.openai_meta_path).exists()
    return (Path(settings.index_dir) / "faiss.index").exists()


def get_store(embedder: str = None) -> BaseStore:
    name = embedder or settings.default_embedder
    if name not in _stores:
        if name == "openai":
            _stores[name] = OpenAIStore(settings.openai_index_path, settings.openai_meta_path)
        else:
            name = "biolord"
            _stores[name] = BiolordStore(settings.index_dir)
    return _stores[name]
