"""Runner aggregation logic (offline — no models, fake retrieval stage)."""

from __future__ import annotations

from gigi.config import Settings
from gigi.eval.golden import GoldenCase
from gigi.eval.runner import evaluate_config
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