"""Pure retrieval-quality metric functions (no model downloads)."""

from __future__ import annotations

from gigi.eval import metrics as m


def test_recall_at_k_basic():
    assert m.recall_at_k(["a.md", "b.md", "c.md"], {"a.md", "c.md"}) == 1.0
    assert m.recall_at_k(["a.md", "b.md"], {"c.md"}) == 0.0
    assert m.recall_at_k(["a.md", "b.md", "c.md"], {"a.md", "d.md"}, k=1) == 0.5


def test_recall_at_k_truncation_and_duplicates():
    # duplicates in hits collapse via the result set
    assert m.recall_at_k(["a.md", "a.md", "a.md"], {"a.md"}) == 1.0
    assert m.recall_at_k(["x.md", "a.md"], {"a.md"}, k=0) == 0.0
    # k larger than the hit list is fine
    assert m.recall_at_k(["a.md"], {"a.md"}, k=100) == 1.0


def test_recall_at_k_empty_expected():
    assert m.recall_at_k(["a.md"], set()) == 0.0
    assert m.recall_at_k([], set()) == 0.0


def test_mrr():
    assert m.mrr(["a.md", "b.md", "c.md"], {"c.md"}) == 1 / 3
    assert m.mrr(["x.md"], {"y.md"}) == 0.0
    assert m.mrr(["a.md", "b.md"], {"b.md"}) == 0.5
    assert m.mrr([], {"a.md"}) == 0.0
    # first occurrence wins, later duplicates irrelevant
    assert m.mrr(["a.md", "b.md", "a.md"], {"a.md"}) == 1.0
    assert m.mrr(["b.md", "b.md"], {"b.md"}) == 1.0
    assert m.mrr(["x.md", "b.md", "b.md"], {"b.md"}) == 0.5


def test_precision_at_k():
    assert m.precision_at_k(["a.md", "b.md"], {"a.md"}) == 0.5
    assert m.precision_at_k(["a.md", "b.md"], {"c.md"}) == 0.0
    assert m.precision_at_k([], {"a.md"}) == 0.0
    assert m.precision_at_k(["a.md", "b.md", "c.md"], {"a.md"}, k=1) == 1.0


def test_mean():
    assert m.mean([0.0, 1.0, 0.5]) == 0.5
    assert m.mean([]) == 0.0
    assert m.mean([0.4]) == 0.4


def test_overview_coverage():
    assert m.overview_coverage(["a.md", "b.md"], {"a.md", "b.md", "c.md"}) == 2 / 3
    assert m.overview_coverage([], {"a.md"}) == 0.0
    assert m.overview_coverage(["a.md"], set()) == 0.0
    assert m.overview_coverage(["a.md", "b.md"], {"a.md"}) == 1.0