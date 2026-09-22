"""Prompt templates for grounded answer generation."""

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


def build_messages(
    question: str,
    chunks: list[dict],
    refine: bool = False,
    overview: bool = False,
) -> list[dict]:
    context = format_context(chunks)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    if overview:
        user = (
            "The context is representative passages covering the main topics in the index. "
            "Summarize what the documents are about, using those passages.\n\n" + user
        )
    if refine:
        user = (
            "Your previous answer was not grounded in the context. "
            "Answer again using only the context.\n\n" + user
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
