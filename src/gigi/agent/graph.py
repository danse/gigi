"""LangGraph workflow: retrieve -> grade -> generate (with self-correction)."""

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


def build_graph(services: Services):
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", make_retrieve_node(services))
    builder.add_node("grade", make_grade_node(services.settings))
    builder.add_node("generate", make_generate_node(services))
    builder.add_node(NO_ANSWER, make_no_answer_node())

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")

    builder.add_conditional_edges(
        "grade",
        lambda state: "generate" if state["relevant"] else NO_ANSWER,
        {"generate": "generate", NO_ANSWER: NO_ANSWER},
    )

    builder.add_conditional_edges(
        "generate",
        lambda state: (
            "generate"
            if not state["grounded"] and state.get("attempt", 0) < services.settings.max_attempts
            else END
        ),
        {"generate": "generate", END: END},
    )

    builder.add_edge(NO_ANSWER, END)

    return builder.compile(checkpointer=MemorySaver())


def run_agent(graph, question: str, thread_id: str | None = None) -> dict:
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    return graph.invoke({"question": question}, config)