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


def test_bootstrap_ci_brackets_the_sample_mean():
    values = list(range(1, 101))  # mean 50.5
    lo, hi = m.bootstrap_ci(values, n_boot=2000, seed=0)
    assert lo < 50.5 < hi
    assert lo <= hi


def test_bootstrap_ci_is_deterministic_for_a_seed():
    values = [1.0, 2.0, 3.0, 100.0]
    a = m.bootstrap_ci(values, n_boot=500, seed=7)
    b = m.bootstrap_ci(values, n_boot=500, seed=7)
    assert a == b


def test_bootstrap_ci_constant_values_collapse_to_the_point():
    assert m.bootstrap_ci([0.5, 0.5, 0.5], n_boot=100, seed=0) == (0.5, 0.5)


def test_bootstrap_ci_empty_values():
    assert m.bootstrap_ci([], n_boot=100, seed=0) == (0.0, 0.0)


def test_make_resample_plan_is_deterministic_and_shared():
    a = m.make_resample_plan(16, 100, seed=0)
    b = m.make_resample_plan(16, 100, seed=0)
    assert a.shape == (100, 16)
    assert (a == b).all()
    assert a.min() >= 0 and a.max() < 16


def test_bootstrap_ci_accepts_a_shared_plan():
    values = [1.0, 2.0, 3.0, 100.0]
    plan = m.make_resample_plan(4, 500, seed=7)
    # Passing the explicit plan is equivalent to drawing it internally.
    assert m.bootstrap_ci(values, n_boot=500, seed=7) == m.bootstrap_ci(
        values, n_boot=500, seed=7, plan=plan
    )


def test_bootstrap_diff_ci_separated_values_exclude_zero():
    a = [1.0] * 20
    b = [0.0] * 20
    plan = m.make_resample_plan(20, 2000, seed=0)
    lo, hi, p = m.bootstrap_diff_ci(a, b, plan)
    assert lo > 0 and hi > 0, "a strictly better config must exclude 0"
    assert p == 1.0


def test_bootstrap_diff_ci_identical_values_are_a_perfect_tie():
    a = [1.0, 2.0, 3.0] * 7
    plan = m.make_resample_plan(21, 500, seed=0)
    lo, hi, p = m.bootstrap_diff_ci(a, a, plan)
    assert (lo, hi) == (0.0, 0.0), "identical vectors give a zero difference"
    assert p == 0.0


def test_bootstrap_diff_ci_shares_the_case_draw_per_iteration():
    # The pairing matters: values are jointly resampled, so a config whose per-
    # case values mirror the reference but with a constant shift is detected
    # exactly, where independent resamples could oscillate around zero.
    a = [round(0.1 * i, 6) for i in range(20)]
    b = [round(0.1 * i + 0.5, 6) for i in range(20)]
    plan = m.make_resample_plan(20, 2000, seed=0)
    lo, hi, p = m.bootstrap_diff_ci(a, a, plan)
    assert (lo, hi, p) == (0.0, 0.0, 0.0)
    lo2, hi2, p2 = m.bootstrap_diff_ci(b, a, plan)
    assert lo2 > 0 and hi2 > 0 and p2 == 1.0