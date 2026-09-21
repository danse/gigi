"""LangGraph node implementations (built as closures over injected services)."""

from __future__ import annotations

from gigi.agent.llm import LLM
from gigi.agent.prompts import build_messages
from gigi.config import Settings
from gigi.indexing.embedder import Embedder
from gigi.indexing.store import IndexStore
from gigi.retrieval.rerank import Reranker
from gigi.retrieval.search import top_k


class Services:
    """Everything the graph nodes need, injected so tests can run fully offline."""

    def __init__(
        self,
        settings: Settings,
        store: IndexStore,
        embedder: Embedder,
        llm: LLM,
        reranker: Reranker | None = None,
    ):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker


def _chunk_dict(source: str, text: str, heading: str, score: float) -> dict:
    return {"source": source, "text": text, "heading": heading, "score": score}


def make_retrieve_node(services: Services):
    def retrieve(state: dict) -> dict:
        chunks, embeddings, _ = services.store.load()
        query_emb = services.embedder.encode_one(state["question"])
        candidates = top_k(chunks, embeddings, query_emb, k=services.settings.top_k)
        if services.reranker is not None:
            reranked = services.reranker.rerank(
                state["question"],
                [c for c, _ in candidates],
                top_k=services.settings.rerank_top_k,
            )
            candidates = reranked
        retrieved = [
            _chunk_dict(c.source, c.text, c.heading, score) for c, score in candidates
        ]
        return {"retrieved": retrieved}

    return retrieve


def make_grade_node(settings: Settings):
    def grade(state: dict) -> dict:
        threshold = settings.resolve_grade_threshold()
        relevant = [r for r in state["retrieved"] if r["score"] >= threshold]
        return {"relevant": relevant}

    return grade


def make_no_answer_node():
    def no_answer(state: dict) -> dict:
        return {
            "answer": "No relevant context found in the indexed documents.",
            "grounded": False,
        }

    return no_answer


def _is_grounded(answer: str) -> bool:
    lowered = (answer or "").lower()
    return bool(answer.strip()) and "i don't know" not in lowered


def make_generate_node(services: Services):
    def generate(state: dict) -> dict:
        refine = state.get("attempt", 0) > 0
        messages = build_messages(state["question"], state["relevant"], refine=refine)
        answer = services.llm.complete(messages)
        return {
            "answer": answer,
            "grounded": _is_grounded(answer),
            "attempt": state.get("attempt", 0) + 1,
        }

    return generate