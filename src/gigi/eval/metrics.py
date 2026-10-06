"""Pure retrieval-quality metrics (offline — no LLM involved)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


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


def bootstrap_ci(
    values: Iterable[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    plan: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile-bootstrap ``(1 - alpha)`` confidence interval for the mean.

    Resamples the per-case values with replacement ``n_boot`` times, recomputes
    the mean of each resample, and returns the ``alpha/2`` and
    ``1 - alpha/2`` percentiles of that distribution. On a golden set of ~20
    cases the interval is wide and coarse — that is the honest signal. Fixed
    *seed* makes the output reproducible.

    Pass a *plan* (from :func:`make_resample_plan`) to reuse an exact resample
    of case indices instead of drawing a private one; that is what turns two
    marginal intervals into a paired comparison.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    n = arr.size
    if n == 0:
        return (0.0, 0.0)
    idx = plan if plan is not None else make_resample_plan(n, n_boot, seed)
    resampled = arr[idx].mean(axis=1)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    lo, hi = np.percentile(resampled, q)
    return (float(lo), float(hi))


def make_resample_plan(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Deterministic ``(n_boot, n)`` matrix of resampled case indices.

    One plan per per-metric case set, shared by *every* config, makes the
    bootstrap paired: configs are compared on the same resampled cases rather
    than on independent streams. ``(n, n_boot, seed)`` fully determines it.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_diff_ci(
    values_a: Iterable[float],
    values_b: Iterable[float],
    plan: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired percentile-bootstrap CI for *mean(a) - mean(b)*.

    Both value vectors are resampled through the *same* case-index *plan*
    (case i is jointly drawn or jointly excluded for a and b), so the interval
    describes the difference distribution directly. Overlapping marginal
    intervals are not a verdict; this interval excluding 0 is.

    Returns ``(lo, hi, p_beat)`` where ``p_beat`` is the fraction of resamples
    in which ``mean(a) > mean(b)`` (≈ 0.5 when the configs are equivalent,
    1.0 when a dominates).
    """
    a = np.asarray(list(values_a), dtype=np.float64)
    b = np.asarray(list(values_b), dtype=np.float64)
    diff = a[plan].mean(axis=1) - b[plan].mean(axis=1)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    lo, hi = np.percentile(diff, q)
    return (float(lo), float(hi), float((diff > 0).mean()))


def overview_coverage(rep_sources: Iterable[str], expected: Iterable[str]) -> float:
    """Fraction of expected documents that surface among cluster representatives."""
    expected = set(expected)
    if not expected:
        return 0.0
    return len(set(rep_sources) & expected) / len(expected)