"""Cosine-similarity retrieval over the embedding index, using torch."""

from __future__ import annotations

import numpy as np
import torch

from gigi.indexing.loader import Chunk


def similarity_scores(query_emb: torch.Tensor, embeddings: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Dot product over L2-normalized vectors == cosine similarity, shape (n,)."""
    matrix = torch.as_tensor(embeddings).float()
    return torch.matmul(matrix, query_emb.reshape(-1).float())


def top_k(
    chunks: list[Chunk],
    embeddings: torch.Tensor | np.ndarray,
    query_emb: torch.Tensor,
    k: int = 6,
) -> list[tuple[Chunk, float]]:
    scores = similarity_scores(query_emb, embeddings)
    k = min(k, len(chunks))
    indices = torch.topk(scores, k).indices.tolist()
    return [(chunks[i], float(scores[i])) for i in indices]