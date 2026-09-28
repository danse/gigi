"""Spherical k-means over L2-normalized embeddings (index-time topic map)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gigi.indexing.loader import Chunk


@dataclass
class ClusterRecord:
    id: int
    centroid_idx: int
    size: int
    heading: str
    source: str
    # Chunk indices in this cluster, nearest-centroid first (centroid_idx first).
    # Absent in indexes built before this field existed.
    members: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {
            "id": self.id,
            "centroid_idx": self.centroid_idx,
            "size": self.size,
            "heading": self.heading,
            "source": self.source,
        }
        if self.members:
            d["members"] = self.members
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ClusterRecord:
        return cls(
            id=int(d["id"]),
            centroid_idx=int(d["centroid_idx"]),
            size=int(d["size"]),
            heading=str(d.get("heading", "")),
            source=str(d.get("source", "")),
            members=[int(i) for i in d.get("members", [])],
        )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _weighted_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    """Sample an index with probability proportional to *weights*.

    ``Generator.choice(..., p=...)`` rejects float32-derived distributions that
    do not sum to 1 exactly; a CDF draw does not care.
    """
    w = np.nan_to_num(np.asarray(weights, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(w, 0, out=w)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        return int(rng.integers(w.size))
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return int(np.searchsorted(cdf, rng.random(), side="right"))


def _kmeans_pp_cosine(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = matrix.shape[0]
    centers = np.empty((k, matrix.shape[1]), dtype=np.float32)
    centers[0] = matrix[int(rng.integers(n))]
    min_sim = matrix @ centers[0]
    for i in range(1, k):
        dist = np.clip(1.0 - min_sim, 1e-12, None)
        idx = _weighted_index(rng, dist)
        centers[i] = matrix[idx]
        min_sim = np.maximum(min_sim, matrix @ centers[i])
    return _l2_normalize(centers)


def spherical_kmeans(
    embeddings: np.ndarray,
    k: int,
    n_iter: int = 15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (n,) labels and (k, dim) L2-normalized centers."""
    matrix = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    n = matrix.shape[0]
    k = min(max(k, 1), n)
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_cosine(matrix, k, rng)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        labels = (matrix @ centers.T).argmax(axis=1).astype(np.int32)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            members = matrix[labels == j]
            if members.shape[0] == 0:
                new_centers[j] = matrix[int(rng.integers(n))]
            else:
                new_centers[j] = members.mean(axis=0)
        centers = _l2_normalize(new_centers)
    labels = (matrix @ centers.T).argmax(axis=1).astype(np.int32)
    return labels, centers


def build_clusters(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    n_clusters: int = 16,
    seed: int = 0,
) -> list[ClusterRecord]:
    """Cluster embeddings and pick the chunk nearest each center as representative."""
    n = len(chunks)
    if n == 0:
        return []
    k = min(n_clusters, n)
    labels, centers = spherical_kmeans(embeddings, k, seed=seed)
    matrix = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    records: list[ClusterRecord] = []
    for j in range(len(centers)):
        mask = np.flatnonzero(labels == j)
        if mask.size == 0:
            continue
        scores = matrix[mask] @ centers[j]
        order = mask[np.argsort(-scores)]  # nearest to the centroid first
        nearest = int(order[0])
        members = [int(i) for i in order]
        members_list = [chunks[int(i)] for i in mask]
        heading = Counter(c.heading or Path(c.source).name for c in members_list).most_common(1)[0][0]
        records.append(
            ClusterRecord(
                id=len(records),
                centroid_idx=nearest,
                size=int(mask.size),
                heading=heading,
                source=chunks[nearest].source,
                members=members,
            )
        )
    records.sort(key=lambda r: r.size, reverse=True)
    for i, rec in enumerate(records):
        rec.id = i
    return records
