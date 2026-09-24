"""Offline end-to-end tests of the LangGraph agent using fake services."""

from __future__ import annotations

import sqlite3

import numpy as np
import torch
from langgraph.checkpoint.sqlite import SqliteSaver

from gigi.agent.graph import (
    GENERATE_OVERVIEW,
    NO_ANSWER,
    build_graph,
    route_after_retrieve,
    run_agent,
)
from gigi.agent.llm import StubLLM
from gigi.agent.nodes import Services
from gigi.config import Settings
from gigi.indexing.cluster import ClusterRecord
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

    def load_clusters(self):
        return [
            ClusterRecord(id=0, centroid_idx=0, size=10, heading="Deployment", source="deployment.md"),
            ClusterRecord(id=1, centroid_idx=1, size=4, heading="Setup", source="onboarding.md"),
        ]


class FakeReranker:
    def rerank(self, query, chunks, top_k=None):
        ranked = [(c, 0.2) for c in chunks]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


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


class RecordingLLM(StubLLM):
    """StubLLM that records every messages list it is sent."""

    def __init__(self):
        super().__init__()
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return super().complete(messages)


def make_services(query: list[float], llm=None) -> Services:
    settings = Settings(rerank_enabled=False, max_attempts=2)
    return Services(
        settings=settings,
        store=FakeStore(),
        embedder=FakeEmbedder(query),
        llm=llm or StubLLM(),
        reranker=None,
    )


def test_route_after_retrieve_branches_on_overview():
    assert route_after_retrieve({"overview": True, "retrieved": [{"text": "x"}]}) == GENERATE_OVERVIEW
    assert route_after_retrieve({"overview": True, "retrieved": []}) == NO_ANSWER
    assert route_after_retrieve({"overview": False, "retrieved": [{"text": "x"}]}) == "grade"


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


def test_graph_overview_uses_cluster_representatives():
    services = make_services([0.0, 0.0])
    graph = build_graph(services)
    result = run_agent(graph, "what are these documents about?")
    assert result["overview"] is True
    assert {c["source"] for c in result["relevant"]} == {"deployment.md", "onboarding.md"}
    assert result["answer"]


def test_rerank_keeps_hits_below_old_logit_threshold():
    services = Services(
        settings=Settings(rerank_enabled=True, max_attempts=1),
        store=FakeStore(),
        embedder=FakeEmbedder([1.0, 0.0]),
        llm=StubLLM(),
        reranker=FakeReranker(),
    )
    result = run_agent(build_graph(services), "How do I deploy to production?")
    assert result["relevant"], "rerank logits of 0.2 used to be dropped by grade threshold 1.0"
    assert result["overview"] is False


def _saver(db_path) -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(str(db_path), check_same_thread=False))


def _roles(messages: list[dict]) -> list[str]:
    return [m["role"] for m in messages]


def test_conversation_memory_carries_across_asks(tmp_path):
    llm = RecordingLLM()
    services = make_services([1.0, 0.0], llm=llm)
    db = tmp_path / "ckpt.sqlite"

    # First "process": ask turn 1.
    graph1 = build_graph(services, checkpointer=_saver(db))
    first = run_agent(graph1, "How do I deploy to production?", thread_id="t1")
    assert first["attempt"] == 1
    assert first["grounded"] is True
    assert len(first["history"]) == 2

    # Second "process": a fresh graph + fresh connection on the same checkpoint
    # file must still see turn 1's history.
    graph2 = build_graph(services, checkpointer=_saver(db))
    second = run_agent(graph2, "And how do I roll it back?", thread_id="t1")
    assert second["attempt"] == 1, "per-turn fields like attempt must reset between asks"
    assert second["grounded"] is True
    assert len(second["history"]) == 4

    last_call = llm.calls[-1]
    assert _roles(last_call) == ["system", "user", "assistant", "user"]
    assert last_call[1]["content"] == "How do I deploy to production?"


def test_new_thread_starts_with_empty_history(tmp_path):
    llm = RecordingLLM()
    graph = build_graph(
        make_services([1.0, 0.0], llm=llm),
        checkpointer=_saver(tmp_path / "ckpt.sqlite"),
    )
    run_agent(graph, "How do I deploy to production?", thread_id="a")
    second = run_agent(graph, "What is the onboarding process?", thread_id="b")
    assert second["history"] == [
        {"role": "user", "content": "What is the onboarding process?"},
        {"role": "assistant", "content": second["answer"]},
    ]
    assert _roles(llm.calls[-1]) == ["system", "user"]


def test_ungrounded_retry_is_not_stored_in_history(tmp_path):
    llm = FlakyLLM()
    graph = build_graph(
        make_services([1.0, 0.0], llm=llm),
        checkpointer=_saver(tmp_path / "ckpt.sqlite"),
    )
    result = run_agent(graph, "How do I deploy?", thread_id="t")
    assert llm.calls == 2
    assert result["attempt"] == 2
    assert len(result["history"]) == 2, "only the grounded turn should be remembered"