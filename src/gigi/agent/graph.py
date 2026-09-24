"""LangGraph workflow: retrieve, then either summarize clusters or grade → generate."""

from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from gigi.agent.nodes import (
    Services,
    make_generate_node,
    make_grade_node,
    make_no_answer_node,
    make_retrieve_node,
)
from gigi.agent.state import GraphState

NO_ANSWER = "no_answer"
GENERATE = "generate"
GENERATE_OVERVIEW = "generate_overview"


def _retry_or_end(node: str, max_attempts: int):
    def route(state: dict) -> str:
        if not state["grounded"] and state.get("attempt", 0) < max_attempts:
            return node
        return END

    return route


def route_after_retrieve(state: dict) -> str:
    if state.get("overview"):
        return GENERATE_OVERVIEW if state.get("retrieved") else NO_ANSWER
    return "grade"


def build_graph(services: Services):
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", make_retrieve_node(services))
    builder.add_node("grade", make_grade_node(services.settings))
    builder.add_node(GENERATE, make_generate_node(services, overview=False))
    builder.add_node(GENERATE_OVERVIEW, make_generate_node(services, overview=True))
    builder.add_node(NO_ANSWER, make_no_answer_node())

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {GENERATE_OVERVIEW: GENERATE_OVERVIEW, "grade": "grade", NO_ANSWER: NO_ANSWER},
    )
    builder.add_conditional_edges(
        "grade",
        lambda state: GENERATE if state["relevant"] else NO_ANSWER,
        {GENERATE: GENERATE, NO_ANSWER: NO_ANSWER},
    )

    max_attempts = services.settings.max_attempts
    builder.add_conditional_edges(
        GENERATE,
        _retry_or_end(GENERATE, max_attempts),
        {GENERATE: GENERATE, END: END},
    )
    builder.add_conditional_edges(
        GENERATE_OVERVIEW,
        _retry_or_end(GENERATE_OVERVIEW, max_attempts),
        {GENERATE_OVERVIEW: GENERATE_OVERVIEW, END: END},
    )
    builder.add_edge(NO_ANSWER, END)

    return builder.compile(checkpointer=MemorySaver())


def run_agent(graph, question: str, thread_id: str | None = None) -> dict:
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    return graph.invoke({"question": question}, config)
