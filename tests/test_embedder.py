"""Unit tests for embedding model-family prefixes (no model downloads)."""

from __future__ import annotations

import torch

from gigi.indexing.embedder import Embedder


def test_e5_uses_query_and_passage_prefixes():
    embedder = Embedder("intfloat/multilingual-e5-small")
    assert embedder.format_query("cos'è il deploy?") == "query: cos'è il deploy?"
    assert embedder.format_passage("il deploy usa uno script.") == "passage: il deploy usa uno script."


def test_e5_prefixes_are_idempotent():
    embedder = Embedder("intfloat/multilingual-e5-small")
    assert embedder.format_query("query: già prefissata") == "query: già prefissata"
    assert embedder.format_passage("passage: già prefissata") == "passage: già prefissata"


def test_bge_uses_query_instruction_only():
    embedder = Embedder("BAAI/bge-m3")
    assert embedder.format_query("hello").startswith(
        "Represent this sentence for searching relevant passages:"
    )
    assert embedder.format_passage("hello") == "hello"


def test_unknown_model_has_no_prefixes():
    embedder = Embedder("some/other-model")
    assert embedder.format_query("hello") == "hello"
    assert embedder.format_passage("hello") == "hello"


class _FakeSentenceModel:
    def __init__(self):
        self.last: list[str] | None = None

    def encode(self, texts, convert_to_tensor=True, normalize_embeddings=True):
        self.last = list(texts)
        return torch.zeros(len(texts), 4)


def test_encode_chunks_applies_passage_prefix():
    embedder = Embedder("intfloat/multilingual-e5-small")
    fake = _FakeSentenceModel()
    embedder._load = lambda: fake
    embedder.encode_chunks(["una frase", "un'altra"])
    assert fake.last == ["passage: una frase", "passage: un'altra"]


def test_encode_one_applies_query_prefix():
    embedder = Embedder("intfloat/multilingual-e5-small")
    fake = _FakeSentenceModel()
    embedder._load = lambda: fake
    embedder.encode_one("che cos'è questo?")
    assert fake.last == ["query: che cos'è questo?"]