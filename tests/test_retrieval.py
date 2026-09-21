"""Unit tests for chunking and torch-based retrieval (no model downloads)."""

from __future__ import annotations

import numpy as np
import torch

from gigi.indexing.loader import Chunk, chunk_text
from gigi.retrieval.search import similarity_scores, top_k


def test_chunk_text_tracks_headings():
    text = "# Intro\n\nSome introductory text.\n\n## Details\n\nMore details here.\n"
    chunks = chunk_text(text, "doc.md", size=100, overlap=10)
    headings = [c.heading for c in chunks]
    assert headings[0] == "Intro"
    assert "Details" in headings


def test_chunk_text_keeps_all_content():
    text = "".join(f"paragraph {i} " + ("words " * 60) + "\n\n" for i in range(5))
    chunks = chunk_text(text, "doc.md", size=200, overlap=50)
    assert len(chunks) >= 5
    joined = " ".join(c.text for c in chunks)
    assert "paragraph 0" in joined and "paragraph 4" in joined


def test_similarity_scores_matches_closest_embedding():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    query = torch.tensor([1.0, 0.0])
    scores = similarity_scores(query, embeddings)
    assert int(torch.argmax(scores)) == 0


def test_top_k_ordering():
    chunks = [Chunk(source="a", text="a"), Chunk(source="b", text="b"), Chunk(source="c", text="c")]
    embeddings = np.array([[0.1, 0.9], [1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    query = torch.tensor([1.0, 0.0])
    results = top_k(chunks, embeddings, query, k=2)
    assert [c.source for c, _ in results] == ["b", "c"]
    assert results[0][1] > results[1][1]