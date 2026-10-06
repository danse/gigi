"""Shared state schema for the LangGraph agent."""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    question: str
    retrieved: list[dict]
    relevant: list[dict]
    answer: str
    degenerate: bool
    overview: bool
    history: list[dict]