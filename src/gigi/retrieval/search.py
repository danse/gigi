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


def mmr_top_k(
    chunks: list[Chunk],
    embeddings: torch.Tensor | np.ndarray,
    query_emb: torch.Tensor,
    k: int = 16,
    pool_k: int | None = None,
    lambda_: float = 0.7,
) -> list[tuple[Chunk, float]]:
    """Cosine pool, then MMR so the kept chunks are on-topic and not near-duplicates."""
    n = len(chunks)
    if n == 0:
        return []
    scores = similarity_scores(query_emb, embeddings)
    pool_k = min(n, pool_k or max(k * 5, 50))
    pool_idx = torch.topk(scores, pool_k).indices.tolist()
    k = min(k, pool_k)
    if k <= 1:
        return [(chunks[i], float(scores[i])) for i in pool_idx[:k]]

    matrix = torch.as_tensor(embeddings).float()
    selected = [pool_idx[0]]
    remaining = pool_idx[1:]
    while len(selected) < k and remaining:
        rem = torch.tensor(remaining, dtype=torch.long)
        rel = scores[rem]
        red = torch.matmul(matrix[rem], matrix[selected].T).max(dim=1).values
        mmr = lambda_ * rel - (1.0 - lambda_) * red
        pick = int(rem[int(torch.argmax(mmr))])
        selected.append(pick)
        remaining.remove(pick)
    return [(chunks[i], float(scores[i])) for i in selected]
