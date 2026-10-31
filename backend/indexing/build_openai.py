"""Build the OpenAI-embedding FAISS index from prepared SNOMED terms.

Self-contained port of the snomed-search-module ``vector_db`` builder. Consumes
the same ``terms.parquet`` that ``prepare_rf2`` produces (one row per active
English description = "with synonyms"), embeds each term with the OpenAI
embeddings API, writes a float16 memmap store + sqlite metadata, then builds a
FAISS ``IndexIDMap2`` (cosine over L2-normalized vectors).

Outputs in ``out_dir``:
- ``faiss.index``  — what the webapp's OpenAIStore reads
- ``store.meta.sqlite`` — idx -> (concept_id, term, type_id)
- ``store.emb``    — float16 embeddings (intermediate; kept for rebuilds)

No GPU involved — embedding happens in OpenAI's API, so this path is identical
on CPU or CUDA hosts. Cost scales with the number of descriptions (~1.4M for the
full International edition); use ``limit`` to smoke-test cheaply.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .embedding_store import EmbeddingStoreWriter, open_store
from .io_utils import ensure_dir

INDEX_NAME = "faiss.index"
EMB_NAME = "store.emb"
META_NAME = "store.meta.sqlite"


def _client(api_key: Optional[str] = None):
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set; required to build the OpenAI index.")
    return OpenAI(api_key=key)


def _embed_batch(client, model: str, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _build_store(
    terms_path: str | Path,
    emb_path: str | Path,
    meta_path: str | Path,
    client,
    model: str,
    batch_size: int = 256,
    limit: int = 0,
) -> int:
    """Embed every term and stream it into a float16 store + sqlite metadata."""
    terms = pd.read_parquet(terms_path, columns=["concept_id", "term", "type_id"])
    if limit:
        terms = terms.head(limit)

    writer = EmbeddingStoreWriter(
        embeddings_path=str(emb_path),
        metadata_path=str(meta_path),
        source_path=str(terms_path),
        overwrite=True,
        dtype="float16",
    )
    written = 0
    try:
        records = terms.to_dict("records")
        for start in tqdm(range(0, len(records), batch_size), desc="embedding"):
            chunk = records[start : start + batch_size]
            texts = [str(r["term"]) for r in chunk]
            embeddings = _embed_batch(client, model, texts)
            rows = [
                {
                    "concept_id": str(r["concept_id"]),
                    "term": str(r["term"]),
                    "type_id": str(r["type_id"]),
                }
                for r in chunk
            ]
            written += writer.append_batch(rows, embeddings)
            writer.commit()
        summary = writer.finalize()
        return int(summary["count"])
    except Exception:
        writer.close(discard=True)
        raise


def _build_faiss_index(emb_path: str | Path, meta_path: str | Path, index_path: str | Path,
                       batch_size: int = 8192) -> int:
    """FAISS IndexIDMap2 over L2-normalized vectors (cosine via inner product)."""
    import faiss

    store = open_store(str(emb_path), str(meta_path))
    try:
        base = faiss.IndexFlatIP(store.dims)
        index = faiss.IndexIDMap2(base)
        if os.path.exists(index_path):
            os.remove(index_path)
        total = 0
        for start in range(0, store.count, batch_size):
            end = min(start + batch_size, store.count)
            block = np.asarray(store.embeddings[start:end], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            valid = norms[:, 0] > 0
            if not np.any(valid):
                continue
            normalized = block[valid] / norms[valid]
            ids = np.arange(start, end, dtype=np.int64)[valid]
            index.add_with_ids(normalized, ids)
            total += int(normalized.shape[0])
        faiss.write_index(index, str(index_path))
        return total
    finally:
        store.close()


def build_openai_index(
    terms_path: str | Path,
    out_dir: str | Path,
    model: str = "text-embedding-3-small",
    batch_size: int = 256,
    limit: int = 0,
    api_key: Optional[str] = None,
) -> dict:
    """Build ``faiss.index`` + ``store.meta.sqlite`` (+ ``store.emb``) in ``out_dir``."""
    out = ensure_dir(out_dir)
    emb_path = out / EMB_NAME
    meta_path = out / META_NAME
    index_path = out / INDEX_NAME

    client = _client(api_key)
    rows = _build_store(terms_path, emb_path, meta_path, client, model, batch_size, limit)
    indexed = _build_faiss_index(emb_path, meta_path, index_path)

    summary = {"rows": rows, "indexed": indexed, "index_dir": str(out)}
    print("[openai] " + json.dumps(summary))
    return summary
