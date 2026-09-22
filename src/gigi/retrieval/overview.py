"""Detect corpus-overview questions that should use topic clusters, not query NN."""

from __future__ import annotations

import re

_DOC_WORDS = r"(documents?|docs|files?|index|corpus|folder|embeddings?)"

_ABOUT_RE = re.compile(
    rf"\bwhat (are these {_DOC_WORDS} about|is this {_DOC_WORDS} about)\b",
    re.IGNORECASE,
)
_SUMMARIZE_RE = re.compile(r"\b(summariz[e]|summary|overview|topics?)\b", re.IGNORECASE)
_DOC_RE = re.compile(rf"\b{_DOC_WORDS}\b", re.IGNORECASE)
_BARE_RE = re.compile(
    r"^(summariz[e]|summary|overview|what is this about|what are they about)$",
    re.IGNORECASE,
)


def is_overview_query(question: str) -> bool:
    q = re.sub(r"\s+", " ", question).strip(" ?!.")
    if not q:
        return False
    if _BARE_RE.match(q) or _ABOUT_RE.search(q):
        return True
    return bool(_SUMMARIZE_RE.search(q) and _DOC_RE.search(q))
