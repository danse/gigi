"""Runner aggregation logic (offline — no models, fake retrieval stage)."""

from __future__ import annotations

from gigi.config import Settings
from gigi.eval.golden import GoldenCase
from gigi.eval.runner import _add_pairwise, evaluate_config
from gigi.indexing.cluster import ClusterRecord
from gigi.indexing.loader import Chunk


def _retrieved_dict(source: str, score: float) -> dict:
    return {"source": source, "text": "body", "heading": "", "score": score}


class FakeStage:
    """A stand-in for RetrievalStage that returns fixed sources per question."""

    def __init__(self, mapping: dict[str, list[dict]]):
        self._mapping = mapping

    def retrieve(self, question: str, settings: Settings) -> list[dict]:
        return list(self._mapping.get(question, []))


def _fixture() -> tuple[Settings, list[Chunk], list[ClusterRecord]]:
    chunks = [
        Chunk(source="a.md", text="a", heading="", idx=0),
        Chunk(source="b.md", text="b", heading="", idx=1),
    ]
    clusters = [
        ClusterRecord(id=0, centroid_idx=0, size=1, heading="", source="a.md"),
        ClusterRecord(id=1, centroid_idx=1, size=1, heading="", source="b.md"),
    ]
    settings = Settings(grade_threshold=0.3, rerank_enabled=False, top_k=16)
    return settings, chunks, clusters


def test_evaluate_config_metrics_and_score():
    settings, chunks, clusters = _fixture()
    cases = (
        GoldenCase("q1", ("a.md",), "en", "specific"),  # hit at rank 1
        GoldenCase("q2", ("a.md",), "en", "specific"),  # wrong but non-empty
        GoldenCase("q3", ("a.md",), "en", "specific"),  # nothing survives grading
        GoldenCase("overview", ("a.md", "b.md"), "en", "overview"),
    )
    stage = FakeStage(
        {
            "q1": [_retrieved_dict("a.md", 0.9), _retrieved_dict("b.md", 0.5)],
            "q2": [_retrieved_dict("b.md", 0.9)],
            "q3": [],
        }
    )

    report = evaluate_config(
        "test", settings, stage, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters,
    )

    q1, q2, q3, _overview = report.cases
    assert (q1.recall8, q1.mrr, q1.answer_basis, q1.bail) == (1.0, 1.0, 1.0, False)
    assert (q2.recall8, q2.mrr, q2.answer_basis, q2.bail) == (0.0, 0.0, 0.0, False)
    assert (q3.recall8, q3.mrr, q3.answer_basis, q3.bail) == (0.0, 0.0, 0.0, True)

    assert report.recall8 == 1 / 3
    assert report.mrr == 1 / 3
    assert report.answer_basis == 1 / 3
    assert report.bail_rate == 1 / 3
    assert report.coverage == 1.0
    # per-case scores: q1=1.0, q2=0.0*4+... -> 0.1, q3=0.0, overview=1.0
    assert report.score == (1.0 + 0.1 + 0.0 + 1.0) / 4


def test_evaluate_config_grade_threshold_none_keeps_everything():
    # resolve_grade_threshold() returns None when rerank is enabled, so hits
    # below an explicit threshold are kept (matching the production grade node).
    settings, chunks, clusters = _fixture()
    settings = Settings(grade_threshold=None, rerank_enabled=True, top_k=16)
    cases = (GoldenCase("q", ("a.md",), "en", "specific"),)
    stage = FakeStage({"q": [_retrieved_dict("a.md", -5.0)]})

    report = evaluate_config(
        "test", settings, stage, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters,
    )
    q = report.cases[0]
    assert q.bail is False
    assert q.answer_basis == 1.0
    assert report.bail_rate == 0.0


def test_evaluate_config_reports_bootstrap_ci_per_metric():
    settings, chunks, clusters = _fixture()
    cases = (
        GoldenCase("q1", ("a.md",), "en", "specific"),
        GoldenCase("q2", ("a.md",), "en", "specific"),
        GoldenCase("q3", ("a.md",), "en", "specific"),
        GoldenCase("overview", ("a.md", "b.md"), "en", "overview"),
    )
    stage = FakeStage(
        {
            "q1": [_retrieved_dict("a.md", 0.9)],
            "q2": [_retrieved_dict("b.md", 0.9)],
            "q3": [],
        }
    )

    report = evaluate_config(
        "test", settings, stage, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters, n_boot=500, seed=3,
    )

    assert set(report.ci) == {
        "score", "recall8", "mrr", "answer_basis", "bail_rate", "coverage",
    }
    for lo, hi in report.ci.values():
        assert lo <= hi
    assert report.n_boot == 500
    assert report.seed == 3
    # same inputs + same seed -> identical intervals
    again = evaluate_config(
        "test", settings, stage, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters, n_boot=500, seed=3,
    )
    assert again.ci == report.ci


def test_evaluate_config_disables_ci_with_zero_resamples():
    settings, chunks, clusters = _fixture()
    cases = (GoldenCase("q", ("a.md",), "en", "specific"),)
    stage = FakeStage({"q": [_retrieved_dict("a.md", 0.9)]})

    report = evaluate_config(
        "test", settings, stage, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters, n_boot=0,
    )
    assert report.ci is None


def test_add_pairwise_attaches_paired_cis_against_reference():
    settings, chunks, clusters = _fixture()
    cases = (
        GoldenCase("q1", ("a.md",), "en", "specific"),
        GoldenCase("q2", ("a.md",), "en", "specific"),
        GoldenCase("q3", ("a.md",), "en", "specific"),
        GoldenCase("overview", ("a.md", "b.md"), "en", "overview"),
    )
    good = FakeStage(
        {
            "q1": [_retrieved_dict("a.md", 0.9)],
            "q2": [_retrieved_dict("a.md", 0.9)],
            "q3": [_retrieved_dict("a.md", 0.9)],
        }
    )
    bad = FakeStage(
        {
            "q1": [_retrieved_dict("b.md", 0.9)],
            "q2": [_retrieved_dict("b.md", 0.9)],
            "q3": [_retrieved_dict("b.md", 0.9)],
        }
    )
    ref = evaluate_config(
        "ref", settings, good, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters, n_boot=0,
    )
    other = evaluate_config(
        "other", settings, bad, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters, n_boot=0,
    )

    _add_pairwise(ref, [other], n_boot=500, seed=0)

    assert other.pairwise is not None
    assert other.pairwise.ref == "ref"
    assert set(other.pairwise.ci) == {
        "score", "recall8", "mrr", "answer_basis", "bail_rate", "coverage",
    }
    assert set(other.pairwise.p_beat) == set(other.pairwise.ci)
    for metric in ("score", "recall8", "mrr", "answer_basis"):
        lo, hi = other.pairwise.ci[metric]
        assert lo <= hi
        assert lo > 0, f"reference must strictly beat {metric}"
    # overview coverage comes from the same clusters for both configs: a tie.
    assert other.pairwise.ci["coverage"] == (0.0, 0.0)
    assert other.pairwise.p_beat["coverage"] == 0.0
    assert other.pairwise.p_beat["score"] > 0.5

    # the reference itself keeps no pairwise record, and n_boot == 0 disables it
    assert ref.pairwise is None
    before = other.pairwise.ci["score"]
    _add_pairwise(ref, [other], n_boot=0, seed=0)
    assert other.pairwise.ci["score"] == before, "n_boot == 0 must leave existing records untouched"
    untouched = evaluate_config(
        "untouched", settings, bad, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters, n_boot=0,
    )
    _add_pairwise(ref, [untouched], n_boot=0, seed=0)
    assert untouched.pairwise is None


def test_add_pairwise_identical_config_is_a_perfect_tie():
    settings, chunks, clusters = _fixture()
    cases = (GoldenCase("q", ("a.md",), "en", "specific"),)
    stage = FakeStage({"q": [_retrieved_dict("a.md", 0.9)]})
    ref = evaluate_config(
        "same", settings, stage, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters, n_boot=0,
    )
    clone = evaluate_config(
        "same", settings, stage, store=None, chunks=chunks,
        embeddings=None, cases=cases, clusters=clusters, n_boot=0,
    )

    _add_pairwise(ref, [clone], n_boot=500, seed=0)

    assert clone.pairwise is not None
    assert clone.pairwise.ci["score"] == (0.0, 0.0)
    assert clone.pairwise.p_beat["score"] == 0.0
    assert clone.pairwise.p_beat["recall8"] == 0.0