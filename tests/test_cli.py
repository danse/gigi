"""CLI-level tests for `gigi ask` conversation continuity (fully offline)."""

from __future__ import annotations

import numpy as np
import torch
from typer.testing import CliRunner

from gigi.agent.llm import StubLLM
from gigi.agent.nodes import Services
from gigi.cli import _SUMMARY_QUESTION, _load_thread_id, app
from gigi.config import Settings
from gigi.indexing.cluster import ClusterRecord
from gigi.indexing.loader import Chunk
from gigi.indexing.store import IndexManifest

CHUNKS = [
    Chunk(source="deployment.md", text="Deploying to production uses the deploy.sh script.", heading="Deployment"),
    Chunk(source="onboarding.md", text="New teammates run make setup to install the toolchain.", heading="Setup"),
]

EMBEDDINGS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

MISSING_MANIFEST = IndexManifest("test", 2, "2026-01-01T00:00:00Z", len(CHUNKS))


class _Store:
    def load(self):
        return CHUNKS, EMBEDDINGS, MISSING_MANIFEST

    def check_embed_model(self, expected: str) -> None:
        return None

    def load_clusters(self):
        return [
            ClusterRecord(id=0, centroid_idx=0, size=10, heading="Deployment", source="deployment.md"),
            ClusterRecord(id=1, centroid_idx=1, size=4, heading="Setup", source="onboarding.md"),
        ]


class _Embedder:
    def encode_one(self, text: str) -> torch.Tensor:
        return torch.tensor([1.0, 0.0], dtype=torch.float32)


class _RecordingLLM(StubLLM):
    def __init__(self):
        super().__init__()
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return super().complete(messages)


def _make_services(llm) -> Services:
    return Services(
        settings=Settings(rerank_enabled=False, max_attempts=2),
        store=_Store(),
        embedder=_Embedder(),
        llm=llm,
        reranker=None,
    )


def test_ask_continues_conversation_across_invocations(monkeypatch, tmp_path):
    llm = _RecordingLLM()
    monkeypatch.setenv("GIGI_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr("gigi.cli._services", lambda settings: _make_services(llm))

    runner = CliRunner()
    first = runner.invoke(app, ["ask", "How do I deploy to production?"])
    assert first.exit_code == 0, first.output

    thread_id = _load_thread_id(tmp_path)
    assert thread_id, "ask must persist the thread id between invocations"

    second = runner.invoke(app, ["ask", "And how do I roll it back?"])
    assert second.exit_code == 0, second.output
    assert _load_thread_id(tmp_path) == thread_id, "a follow-up reuses the same thread"

    roles = [m["role"] for m in llm.calls[-1]]
    assert roles == ["system", "user", "assistant", "user"], "follow-up must see previous turn"


def test_ask_reset_starts_a_new_thread(monkeypatch, tmp_path):
    llm = _RecordingLLM()
    monkeypatch.setenv("GIGI_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr("gigi.cli._services", lambda settings: _make_services(llm))

    runner = CliRunner()
    assert runner.invoke(app, ["ask", "How do I deploy?"]).exit_code == 0
    original = _load_thread_id(tmp_path)
    assert original

    assert runner.invoke(app, ["ask", "What is the onboarding process?", "--reset"]).exit_code == 0
    assert _load_thread_id(tmp_path) != original, "--reset must mint a new thread"
    assert [m["role"] for m in llm.calls[-1]] == ["system", "user"]


def test_summarise_forces_overview_branch_without_reranker(monkeypatch, tmp_path):
    monkeypatch.setenv("GIGI_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr("gigi.cli._require_matching_index", lambda store, settings: None)
    captured: dict = {}

    def fake_run_agent(graph, question, thread_id=None, *, overview=False):
        captured["question"] = question
        captured["overview"] = overview
        return {
            "answer": "Topics include deployment and onboarding.",
            "relevant": [
                {"source": "deployment.md", "heading": "Deployment", "score": 0.5},
                {"source": "onboarding.md", "heading": "Setup", "score": 0.25},
            ],
        }

    monkeypatch.setattr("gigi.cli.run_agent", fake_run_agent)
    monkeypatch.setattr("gigi.cli.build_graph", lambda services: object())  # graph never invoked

    runner = CliRunner()
    result = runner.invoke(app, ["summarise"])
    assert result.exit_code == 0, result.output
    assert captured["overview"] is True, "summarise must take the cluster-representatives branch"
    assert captured["question"] == _SUMMARY_QUESTION
    assert "deployment.md" in result.output
    assert "onboarding.md" in result.output
    assert "Topics include" in result.output