"""Pure retrieval-quality metrics (offline — no LLM involved)."""

from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(hits: Iterable[str], expected: Iterable[str], k: int | None = None) -> float:
    """Fraction of expected sources present in the top-*k* hits."""
    expected = set(expected)
    if not expected:
        return 0.0
    if k is not None:
        hits = list(hits)[:k]
    return len(set(hits) & expected) / len(expected)


def precision_at_k(hits: Iterable[str], expected: Iterable[str], k: int | None = None) -> float:
    """Fraction of top-*k* hits that are expected sources."""
    hits = list(hits)
    if k is not None:
        hits = hits[:k]
    if not hits:
        return 0.0
    return len(set(hits) & set(expected)) / len(hits)


def mrr(hits: Iterable[str], expected: Iterable[str]) -> float:
    """Reciprocal rank of the first expected source (0 if none is hit)."""
    expected = set(expected)
    for rank, source in enumerate(hits, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def overview_coverage(rep_sources: Iterable[str], expected: Iterable[str]) -> float:
    """Fraction of expected documents that surface among cluster representatives."""
    expected = set(expected)
    if not expected:
        return 0.0
    return len(set(rep_sources) & expected) / len(expected)