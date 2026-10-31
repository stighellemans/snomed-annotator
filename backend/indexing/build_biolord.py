"""Build the local BioLORD FAISS index from prepared SNOMED terms.

Ported verbatim (pooling + normalization) from the original snomed-embedder
``build-index`` command so query vectors produced by ``backend/app/embedder.py``
land in exactly the same space. Mean-pool token embeddings over the attention
mask, L2-normalize, FAISS ``IndexFlatIP`` (cosine).

Device handling: ``cpu`` / ``cuda`` / ``mps`` are all accepted; an unavailable
accelerator falls back to ``cpu`` with a warning, so the same command works on a
Mac (mps), an Azure CPU VM (cpu), or an Azure GPU VM (cuda).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io_utils import ensure_dir


def _resolve_device(device: str):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("[biolord] cuda requested but unavailable -> falling back to cpu")
        device = "cpu"
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        print("[biolord] mps requested but unavailable -> falling back to cpu")
        device = "cpu"
    return torch.device(device)


def build_biolord_index(
    terms_path: str | Path,
    out_dir: str | Path,
    model_name: str = "FremyCompany/BioLORD-2023-M",
    device: str = "cpu",
    batch_size: int = 128,
    max_length: int = 128,
    dtype: str = "float16",
    limit: int = 0,
) -> dict:
    """Encode every term and write ``faiss.index`` + ``metadata.parquet``
    (+ ``embeddings.npy``) into ``out_dir``. Returns a small summary dict."""
    import faiss
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    out = ensure_dir(out_dir)
    terms = pd.read_parquet(terms_path)
    if limit:
        terms = terms.head(limit)
    texts = [str(value) for value in terms["term"].tolist()]

    print(f"[biolord] loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    torch_device = _resolve_device(device)
    print(f"[biolord] encoding {len(texts)} terms on {torch_device}")
    model.to(torch_device)
    model.eval()

    encoded_batches: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encoding"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(torch_device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        token_embeddings = output[0]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        encoded_batches.append(pooled.cpu().numpy().astype("float32"))

    embeddings = np.vstack(encoded_batches).astype("float32")
    if dtype == "float16":
        np.save(out / "embeddings.npy", embeddings.astype("float16"))
    else:
        np.save(out / "embeddings.npy", embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(out / "faiss.index"))

    metadata = terms.reset_index(drop=True).copy()
    metadata["idx"] = metadata.index
    metadata.to_parquet(out / "metadata.parquet", index=False)

    summary = {"rows": len(metadata), "dims": int(embeddings.shape[1]), "index_dir": str(out)}
    print("[biolord] " + json.dumps(summary))
    return summary
