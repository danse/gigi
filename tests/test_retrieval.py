"""Unit tests for chunking and torch-based retrieval (no model downloads)."""

from __future__ import annotations

import numpy as np
import torch

from gigi.indexing.loader import Chunk, chunk_text
from gigi.retrieval.search import mmr_top_k, similarity_scores, top_k


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


def test_chunk_text_skips_heading_only_stubs():
    text = (
        "# Architecture\n\n"
        "## Overview\n\n"
        "The platform is a three-tier app.\n\n"
        "## Backend\n\n"
        "The API is FastAPI.\n"
    )
    chunks = chunk_text(text, "doc.md", size=800, overlap=50)
    assert chunks
    assert all("tier" in c.text or "FastAPI" in c.text for c in chunks)
    assert any("three-tier" in c.text for c in chunks)
    assert any("FastAPI" in c.text for c in chunks)


def test_mmr_top_k_keeps_best_hit_first():
    chunks = [
        Chunk(source="a", text="near duplicate 1"),
        Chunk(source="a", text="near duplicate 2"),
        Chunk(source="b", text="other topic"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    query = torch.tensor([1.0, 0.0])
    results = mmr_top_k(chunks, embeddings, query, k=2, pool_k=3, lambda_=0.3)
    assert results[0][0].source == "a"
    assert {c.source for c, _ in results} == {"a", "b"}