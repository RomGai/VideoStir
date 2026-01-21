"""Lightweight helpers for Qwen3 embedding-based similarity."""

from __future__ import annotations

from typing import Iterable, List

import torch
from sentence_transformers import SentenceTransformer

_EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_embed_model = SentenceTransformer(_EMBED_MODEL_NAME)


def _encode_queries(queries: Iterable[str]) -> torch.Tensor:
    return _embed_model.encode(
        list(queries),
        prompt_name="query",
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def _encode_documents(documents: Iterable[str]) -> torch.Tensor:
    return _embed_model.encode(
        list(documents), convert_to_tensor=True, normalize_embeddings=True
    )


def compute_similarities(query: str, documents: List[str]) -> List[float]:
    """Return cosine similarities between a query and documents."""

    if not documents:
        return []

    query_embedding = _encode_queries([query])
    document_embeddings = _encode_documents(documents)
    similarities = query_embedding @ document_embeddings.T
    return similarities.squeeze(0).tolist()

