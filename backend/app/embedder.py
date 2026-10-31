"""Multilingual BioLORD query encoder.

The FAISS index is built by ``indexing/build_biolord.py`` with transformers
``AutoModel``, mean-pooling token embeddings over the attention mask and
L2-normalizing. We replicate that EXACTLY here so query vectors live in the same
space as the indexed term vectors. Do not swap in a different pooling on one side
without rebuilding the index.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .config import settings


class BioLordEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Fall back to cpu if the requested device is unavailable.
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        torch = self._torch
        items = [str(t) for t in texts]
        if not items:
            return np.zeros((0, self.model.config.hidden_size), dtype="float32")
        encoded = self.tokenizer(
            items,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output[0]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class OpenAIEmbedder:
    """Query encoder for the OpenAI text-embedding index. Returns L2-normalized
    vectors so inner-product search matches the (normalized) indexed vectors."""

    def __init__(self, model: str):
        self.model = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from .llm import get_embedding_client

        items = [str(t) for t in texts]
        if not items:
            return np.zeros((0, 1536), dtype="float32")
        resp = get_embedding_client().embeddings.create(model=self.model, input=items)
        vecs = np.asarray([d.embedding for d in resp.data], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_embedders: dict[str, object] = {}


def get_embedder(name: str = "biolord"):
    """Return a cached encoder for the given embedder name ('biolord' | 'openai')."""
    if name not in _embedders:
        if name == "openai":
            _embedders[name] = OpenAIEmbedder(settings.openai_embed_model)
        else:
            _embedders[name] = BioLordEmbedder(settings.embedder_model, settings.embedder_device)
    return _embedders[name]
