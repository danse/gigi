"""Offline end-to-end tests of the LangGraph agent using fake services."""

from __future__ import annotations

import numpy as np
import torch

from gigi.agent.graph import build_graph, run_agent
from gigi.agent.llm import StubLLM
from gigi.agent.nodes import Services
from gigi.config import Settings
from gigi.indexing.loader import Chunk
from gigi.indexing.store import IndexManifest

CHUNKS = [
    Chunk(source="deployment.md", text="Deploying to production uses the deploy.sh script.", heading="Deployment"),
    Chunk(source="onboarding.md", text="New teammates run make setup to install the toolchain.", heading="Setup"),
]

# Normalized 2-d embeddings: chunk 0 on the x-axis, chunk 1 on the y-axis.
EMBEDDINGS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

MISSING_MANIFEST = IndexManifest("test", 2, "2026-01-01T00:00:00Z", len(CHUNKS))


class FakeStore:
    def load(self):
        return CHUNKS, EMBEDDINGS, MISSING_MANIFEST


class FakeEmbedder:
    def __init__(self, query: list[float]):
        self._query = query

    def encode_one(self, text: str) -> torch.Tensor:
        return torch.tensor(self._query, dtype=torch.float32)


class FlakyLLM(StubLLM):
    """Returns an ungrounded answer once, then a grounded answer."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            return "I don't know."
        return "The deployment uses the deploy.sh script."


def make_services(query: list[float], llm=None) -> Services:
    settings = Settings(rerank_enabled=False, max_attempts=2)
    return Services(
        settings=settings,
        store=FakeStore(),
        embedder=FakeEmbedder(query),
        llm=llm or StubLLM(),
        reranker=None,
    )


def test_graph_answers_grounded_question():
    services = make_services([1.0, 0.0])
    graph = build_graph(services)
    result = run_agent(graph, "How do I deploy to production?")
    assert result["relevant"], "expected matching chunk to be graded relevant"
    assert result["relevant"][0]["source"] == "deployment.md"
    assert result["grounded"] is True
    assert result["answer"]


def test_graph_bails_out_when_nothing_relevant():
    services = make_services([0.0, 0.0])  # orthogonal to both chunks -> below threshold
    graph = build_graph(services)
    result = run_agent(graph, "What color is the sky on mars?")
    assert result["relevant"] == []
    assert "No relevant context" in result["answer"]
    assert result["grounded"] is False


def test_graph_retries_when_answer_not_grounded():
    llm = FlakyLLM()
    services = make_services([1.0, 0.0], llm=llm)
    graph = build_graph(services)
    result = run_agent(graph, "How do I deploy?")
    assert llm.calls == 2, "expected exactly one self-correction retry"
    assert result["attempt"] == 2
    assert result["grounded"] is True
    assert "deploy.sh" in result["answer"]