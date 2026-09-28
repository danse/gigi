"""Detect corpus-overview questions that should use topic clusters, not query NN.

The index is multilingual (Italian/Spanish/Catalan/English), so overview
detection understands all four languages. A question counts when it either
asks what the documents are *about*, or combines a summarize-word with a
document-word.
"""

from __future__ import annotations

import re

from gigi.indexing.cluster import ClusterRecord
from gigi.indexing.loader import Chunk

# (about_regex, summary_word_regex, document_word_regex) per language.
_LANGUAGES = [
    # English
    (
        (
            r"\bwhat (are these (?:documents?|docs|files?|index|corpus|folder|embeddings?) about|"
            r"is this (?:documents?|docs|files?|index|corpus|folder|embeddings?) about)\b"
        ),
        r"\bsummariz(?:e|es)?|summary|overview|topics?\b",
        r"\b(documents?|docs|files?|index|corpus|folder|embeddings?)\b",
    ),
    # Italian
    (
        r"\bdi cosa (?:parlano|trattano) (?:questi|tutti) (?:documenti|file)\b",
        r"\briassumi|riepiloga|sommari[o]?|panoramica|argomenti?\b",
        r"\b(documenti?|file|indice|corpus|cartella)\b",
    ),
    # Spanish
    (
        r"\bde qué (?:tratan|hablan) (?:estos|todos los) (?:documentos|archivos)\b",
        r"\bresumen?|resum(?:e|en)|panorama general|temas?\b",
        r"\b(documentos?|archivos?|índice|corpus|carpeta)\b",
    ),
    # Catalan
    (
        r"\bde què (?:tracten|parlen) (?:aquests|tots els) (?:documents|arxius)\b",
        r"\bresumeix|resum|visió general|temes?\b",
        r"\b(documents?|arxius?|índex|corpus|carpeta)\b",
    ),
]

_COMPILED = [
    (re.compile(a, re.IGNORECASE), re.compile(s, re.IGNORECASE), re.compile(d, re.IGNORECASE))
    for a, s, d in _LANGUAGES
]

_BARE_RE = re.compile(
    r"^("
    r"summarize|summary|overview|what is this about|what are they about|"
    r"riassumi i documenti|sommario|panoramica|di cosa parlano questi documenti|"
    r"resume los documentos|resumen|panorama general|de qué tratan estos documentos|"
    r"resumeix els documents|resum|visió general|de què tracten aquests documents"
    r")$",
    re.IGNORECASE,
)


def is_overview_query(question: str) -> bool:
    q = re.sub(r"\s+", " ", question).strip(" ?!.")
    if not q:
        return False
    if _BARE_RE.match(q):
        return True
    for about, summary, doc in _COMPILED:
        if about.search(q):
            return True
        if summary.search(q) and doc.search(q):
            return True
    return False


def cluster_representatives(chunks: list[Chunk], clusters: list[ClusterRecord]) -> list[dict]:
    """One representative chunk per topic cluster, with a coverage score (0..1).

    Shared by the overview node (production) and the eval runner (offline), so
    `gigi eval` measures exactly what `gigi ask` shows for overview questions.
    """
    n = max(len(chunks), 1)
    retrieved: list[dict] = []
    for cluster in clusters:
        if cluster.centroid_idx < 0 or cluster.centroid_idx >= len(chunks):
            continue
        chunk = chunks[cluster.centroid_idx]
        retrieved.append(
            {
                "source": chunk.source,
                "text": chunk.text,
                "heading": cluster.heading or "",
                "score": cluster.size / n,
            }
        )
    return retrieved