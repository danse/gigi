"""Settings defaults and grade-threshold resolution (no model downloads)."""

from __future__ import annotations

from gigi.config import Settings


def test_defaults_are_the_grid_tuned_values():
    settings = Settings()
    # tuned with `gigi eval --grid`: reranking hurt mrr/basis, so it is off
    assert settings.rerank_enabled is False
    assert settings.top_k == 16
    assert settings.rerank_top_k == 8
    assert settings.mmr_lambda == 0.7
    assert settings.n_clusters == 16
    assert settings.overview_per_cluster == 2
    assert settings.grade_threshold is None


def test_grade_threshold_defaults_to_0_3_without_reranker():
    # no reranker -> cosine candidates are graded at 0.3 unless overridden
    assert Settings().resolve_grade_threshold() == 0.3
    assert Settings(rerank_enabled=False).resolve_grade_threshold() == 0.3


def test_grade_threshold_off_with_reranker():
    # the reranker already truncated the candidates, so nothing is cut
    assert Settings(rerank_enabled=True).resolve_grade_threshold() is None


def test_explicit_grade_threshold_wins():
    assert Settings(grade_threshold=0.5, rerank_enabled=True).resolve_grade_threshold() == 0.5
    assert Settings(grade_threshold=0.5, rerank_enabled=False).resolve_grade_threshold() == 0.5


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("GIGI_RERANK", "1")
    monkeypatch.setenv("GIGI_TOP_K", "24")
    settings = Settings.from_env()
    assert settings.rerank_enabled is True
    assert settings.top_k == 24