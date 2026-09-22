"""Document discovery, text extraction and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".mdx", ".txt", ".rst", ".pdf", ""}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Chunk:
    source: str
    text: str
    heading: str = ""
    idx: int = 0

    def as_dict(self) -> dict:
        return {"source": self.source, "text": self.text, "heading": self.heading, "idx": self.idx}

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(
            source=str(d["source"]),
            text=str(d["text"]),
            heading=str(d.get("heading", "")),
            idx=int(d.get("idx", 0)),
        )


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _heading_only(text: str) -> bool:
    """True when *text* is markdown headings and whitespace, with no body."""
    saw_heading = False
    for line in text.splitlines():
        if not line.strip():
            continue
        if not _HEADING_RE.match(line):
            return False
        saw_heading = True
    return saw_heading


def chunk_text(
    text: str,
    source: str,
    size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """Split *text* into overlapping character windows, tracking markdown headings."""
    lines = re.split(r"(?<=\n)", text)
    chunks: list[Chunk] = []
    current = ""
    heading = ""

    def flush() -> None:
        nonlocal current
        if current.strip() and not _heading_only(current):
            chunks.append(Chunk(source=source, text=current, heading=heading, idx=len(chunks)))
        current = ""

    for line in lines:
        m = _HEADING_RE.match(line)
        if m and current.strip() and not _heading_only(current):
            flush()
        if m:
            heading = m.group(2).strip()
        if len(current) + len(line) <= size:
            current += line
        else:
            flush()
            current = current[-overlap:] + line if overlap else line
            while len(current) > size:
                if not _heading_only(current[:size]):
                    chunks.append(
                        Chunk(source=source, text=current[:size], heading=heading, idx=len(chunks))
                    )
                current = current[size - overlap :] if overlap else current[size:]
    flush()
    return chunks


def load_documents(root: Path, chunk_size: int = 800, chunk_overlap: int = 100) -> list[Chunk]:
    """Discover all supported files under *root* and return their chunks."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"document root does not exist: {root}")
    chunks: list[Chunk] = []
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for path in files:
        text = read_text(path)
        if not text.strip():
            continue
        chunks.extend(chunk_text(text, str(path), size=chunk_size, overlap=chunk_overlap))
    return chunks
