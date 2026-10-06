"""LangGraph node implementations (built as closures over injected services)."""

from __future__ import annotations

from gigi.agent.llm import LLM

NO_ANSWER_MESSAGE = "No reliable answer found in the indexed documents."
from gigi.agent.prompts import build_messages
from gigi.config import Settings
from gigi.indexing.cluster import ClusterRecord, build_clusters
from gigi.indexing.embedder import Embedder
from gigi.indexing.store import IndexStore
from gigi.retrieval.overview import cluster_representatives
from gigi.retrieval.rerank import Reranker
from gigi.retrieval.search import mmr_top_k


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


def _overview_chunks(services: Services, chunks, embeddings) -> list[dict]:
    loader = getattr(services.store, "load_clusters", None)
    clusters: list[ClusterRecord] = loader() if callable(loader) else []
    if not clusters:
        clusters = build_clusters(chunks, embeddings, n_clusters=services.settings.n_clusters)
    return cluster_representatives(
        chunks, clusters, per_cluster=services.settings.overview_per_cluster
    )


def make_retrieve_node(services: Services):
    def retrieve(state: dict) -> dict:
        chunks, embeddings, _ = services.store.load()
        # Fresh turn: forget per-question fields left by the previous turn.
        base = {"answer": ""}
        if state.get("overview", False):
            representatives = _overview_chunks(services, chunks, embeddings)
            return {
                **base,
                "retrieved": representatives,
                "relevant": representatives,
                "overview": True,
            }

        query_emb = services.embedder.encode_one(state["question"])
        candidates = mmr_top_k(
            chunks,
            embeddings,
            query_emb,
            k=services.settings.top_k,
            lambda_=services.settings.mmr_lambda,
        )
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
        return {**base, "retrieved": retrieved, "overview": False}

    return retrieve


def make_grade_node(settings: Settings):
    def grade(state: dict) -> dict:
        retrieved = state.get("retrieved") or []
        threshold = settings.resolve_grade_threshold()
        if threshold is None:
            return {"relevant": retrieved}
        relevant = [r for r in retrieved if r["score"] >= threshold]
        return {"relevant": relevant}

    return grade


def make_no_answer_node():
    def no_answer(state: dict) -> dict:
        return {"answer": NO_ANSWER_MESSAGE}

    return no_answer


def _is_degenerate(answer: str) -> bool:
    """Empty output, or the canonical refusal the prompt instructs (verbatim).

    This is deliberately *not* a grounding/verification layer: any other output
    is accepted as-is, and the grade threshold is the only relevance gate
    (retrieval-side). A refusal cannot be reliably recognised past this exact
    string, so we do not try — the sources shown under each answer belong to
    the human to verify.
    """
    text = (answer or "").strip()
    if not text:
        return True
    return text.rstrip(".").lower() == "i don't know"


def make_generate_node(services: Services, *, overview: bool = False):
    def generate(state: dict) -> dict:
        messages = build_messages(
            state["question"],
            state["relevant"],
            overview=overview,
            history=state.get("history") or [],
        )
        answer = services.llm.complete(messages)
        result: dict = {"answer": answer, "degenerate": _is_degenerate(answer)}
        if not result["degenerate"]:
            # Every real turn joins the conversation history; only output that
            # routes to the no-answer node (empty/refused) is not remembered,
            # so a decline is never replayed as trusted context.
            result["history"] = (state.get("history") or []) + [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ]
        return result

    return generate
