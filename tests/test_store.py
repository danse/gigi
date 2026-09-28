"""Round-trip tests for the on-disk index (no model downloads)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gigi.indexing.loader import Chunk
from gigi.indexing.store import IndexStore


def _embeddings(n: int, dim: int = 2) -> np.ndarray:
    return np.zeros((n, dim), dtype=np.float32)


def test_store_roundtrip(tmp_path):
    store = IndexStore(tmp_path)
    chunks = [
        Chunk(source="a.md", text="alpha", heading="A", idx=0),
        Chunk(source="b.md", text="beta", heading="B", idx=1),
    ]
    embeddings_in = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.save(chunks, embeddings_in, "test-model")
    loaded, embeddings, manifest = store.load()
    assert [c.text for c in loaded] == ["alpha", "beta"]
    assert embeddings.shape == (2, 2)
    assert manifest.n_chunks == 2
    assert manifest.model_name == "test-model"
    clusters = store.load_clusters()
    assert len(clusters) == 2
    assert sum(c.size for c in clusters) == 2


def test_store_roundtrip_unicode_line_separators(tmp_path):
    """Document text may contain separators that splitlines() treats as newlines."""
    store = IndexStore(tmp_path)
    text = "hello\u2028world\u2029para\u0085next"
    chunks = [Chunk(source="doc.md", text=text, heading="H", idx=0)]
    store.save(chunks, _embeddings(1), "test-model")

    raw = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8")
    assert "\u2028" in raw
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw.splitlines()[0])

    loaded, _, _ = store.load()
    assert loaded[0].text == text


def test_load_accepts_json_array(tmp_path):
    store = IndexStore(tmp_path)
    store.save([Chunk(source="a.md", text="x", idx=0)], _embeddings(1), "test-model")
    (tmp_path / "chunks.jsonl").write_text(
        json.dumps([{"source": "a.md", "text": "x", "heading": "", "idx": 0}]),
        encoding="utf-8",
    )
    loaded, _, _ = store.load()
    assert loaded[0].text == "x"


def test_recluster_rewrites_clusters_only(tmp_path):
    store = IndexStore(tmp_path)
    chunks = [
        Chunk(source="a.md", text="alpha", heading="A", idx=0),
        Chunk(source="b.md", text="beta", heading="B", idx=1),
    ]
    embeddings_in = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.save(chunks, embeddings_in, "test-model", n_clusters=2)
    emb_before = store.embeddings_file.read_bytes()
    chunks_before = store.chunks_file.read_text()
    store.clusters_file.write_text("{}", encoding="utf-8")
    clusters = store.recluster(n_clusters=2)
    assert len(clusters) == 2
    assert sum(c.size for c in clusters) == 2
    assert store.embeddings_file.read_bytes() == emb_before
    assert store.chunks_file.read_text() == chunks_before


def test_index_recluster_cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from gigi.cli import app

    index_dir = tmp_path / ".index"
    monkeypatch.setenv("GIGI_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("GIGI_EMBED_MODEL", "test-model")
    store = IndexStore(index_dir)
    store.save(
        [
            Chunk(source="a.md", text="alpha", heading="A", idx=0),
            Chunk(source="b.md", text="beta", heading="B", idx=1),
        ],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "test-model",
        n_clusters=2,
    )
    emb_before = store.embeddings_file.read_bytes()
    result = CliRunner().invoke(app, ["index", "--recluster"])
    assert result.exit_code == 0, result.output
    assert store.embeddings_file.read_bytes() == emb_before
    assert len(store.load_clusters()) == 2


def test_index_recluster_cli_requires_index(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from gigi.cli import app

    monkeypatch.setenv("GIGI_INDEX_DIR", str(tmp_path / "missing"))
    result = CliRunner().invoke(app, ["index", "--recluster"])
    assert result.exit_code == 1
    assert "No index found" in result.output


def test_load_corrupt_index_has_rebuild_hint(tmp_path):
    store = IndexStore(tmp_path)
    store.save([Chunk(source="a.md", text="x", idx=0)], _embeddings(1), "test-model")
    (tmp_path / "chunks.jsonl").write_text('{"source": "a.md", "text": "unterminated\n', encoding="utf-8")
    with pytest.raises(ValueError, match="re-run `gigi index"):
        store.load()


def test_check_embed_model_rejects_mismatch(tmp_path):
    store = IndexStore(tmp_path)
    store.save([Chunk(source="a.md", text="x", idx=0)], _embeddings(1), "old-model")
    with pytest.raises(ValueError, match="old-model"):
        store.check_embed_model("intfloat/multilingual-e5-small")


def test_check_embed_model_accepts_match(tmp_path):
    store = IndexStore(tmp_path)
    store.save([Chunk(source="a.md", text="x", idx=0)], _embeddings(1), "intfloat/multilingual-e5-small")
    store.check_embed_model("intfloat/multilingual-e5-small")  # must not raise


def test_ask_refuses_index_from_other_model(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from gigi.cli import app

    index_dir = tmp_path / ".index"
    monkeypatch.setenv("GIGI_INDEX_DIR", str(index_dir))
    store = IndexStore(index_dir)
    store.save(
        [Chunk(source="a.md", text="deploy tramite script", idx=0)],
        _embeddings(1),
        "other-model",
    )
    result = CliRunner().invoke(app, ["ask", "di cosa parlano questi documenti?"])
    assert result.exit_code == 1
    assert "other-model" in result.output
    assert "gigi index" in result.output
