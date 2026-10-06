"""Prompt templates for answer generation."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a precise assistant that answers questions strictly from the provided context.

Rules:
- Base your answer ONLY on the context below. Do not use outside knowledge.
- You may summarize and synthesize across multiple context passages.
- If the question asks what the documents are about, or for an overview, summarize the topics covered in the context.
- If the context is unrelated to the question, reply exactly: "I don't know."
- End your answer with the source file(s) you used, formatted as: [source: path/to/file]
- Keep the answer concise (under 250 words)."""


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        heading = f" ({chunk['heading']})" if chunk.get("heading") else ""
        blocks.append(f"[{i}] source: {chunk['source']}{heading}\n{chunk['text']}")
    return "\n\n".join(blocks)


def format_overview_context(chunks: list[dict]) -> str:
    """Group representative passages by topic so the summary is per-topic.

    Chunks carry a ``cluster`` key from ``cluster_representatives``; passages
    of the same cluster are the same topic, and the LLM is told so, instead of
    being handed one flat soup of representatives.
    """
    groups: dict[int, list[dict]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.get("cluster"), []).append(chunk)
    blocks = []
    for i, (_, members) in enumerate(groups.items(), 1):
        lines = [f"Topic {i}:"]
        for j, chunk in enumerate(members, 1):
            heading = f" ({chunk['heading']})" if chunk.get("heading") else ""
            lines.append(f"[{j}] source: {chunk['source']}{heading}\n{chunk['text']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_history(history: list[dict], max_messages: int = 8) -> list[dict]:
    """Last `max_messages` turns as chat messages (cap keeps prompts short)."""
    return [m for m in history[-max_messages:] if m.get("role") in {"user", "assistant"}]


def build_messages(
    question: str,
    chunks: list[dict],
    overview: bool = False,
    history: list[dict] | None = None,
) -> list[dict]:
    if overview:
        user = (
            "The passages below are grouped by topic; each group is one topic "
            "found in the index.\n"
            "For every topic, write a one-line description of what it is about.\n"
            "Then write a final short paragraph describing the overall content "
            "of the index.\n\n"
            + format_overview_context(chunks)
        )
    else:
        user = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(format_history(history or []))
    messages.append({"role": "user", "content": user})
    return messages
