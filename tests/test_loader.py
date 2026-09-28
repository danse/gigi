"""Document discovery and chunking (no model downloads)."""

from __future__ import annotations

from gigi.indexing.loader import chunk_text, load_documents


def test_load_documents_excludes_directories(tmp_path):
    root = tmp_path
    (root / "data.md").write_text("a real document.", encoding="utf-8")
    idx = root / ".index"
    idx.mkdir()
    (idx / "thread_id").write_text("abc123", encoding="utf-8")
    (idx / "notes.md").write_text("junk that must not be indexed.", encoding="utf-8")

    chunks = load_documents(root, exclude=[idx])
    assert any(c.source.endswith("data.md") for c in chunks)
    assert not any(".index" in c.source for c in chunks), "excluded dirs must be skipped"

    # Without the exclusion the same corpus does pick the stale index file up.
    all_chunks = load_documents(root)
    assert any(".index" in c.source for c in all_chunks)


def test_chunk_text_tracks_heading_and_skips_heading_only_fragments():
    chunks = chunk_text(
        "# Title\n\ndeploy via make release.\n\n## Rollback\n\nuse the script rollback.sh.",
        "ops.md",
        size=800,
    )
    assert [c.heading for c in chunks] == ["Title", "Rollback"]
    assert any("rollback.sh" in c.text for c in chunks)
    assert all(c.source == "ops.md" for c in chunks)
    assert [c.idx for c in chunks] == [0, 1]


def test_load_documents_missing_root_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")