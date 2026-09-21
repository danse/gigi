"""gigi command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from gigi import __version__
from gigi.agent.graph import build_graph, run_agent
from gigi.agent.llm import get_llm
from gigi.agent.nodes import Services
from gigi.config import Settings
from gigi.indexing.embedder import Embedder
from gigi.indexing.loader import load_documents
from gigi.indexing.store import IndexStore
from gigi.retrieval.rerank import Reranker

app = typer.Typer(
    name="gigi",
    help="Local document Q&A assistant built on PyTorch + LangGraph.",
    no_args_is_help=True,
)


def _services(settings: Settings) -> Services:
    store = IndexStore(settings.index_dir)
    embedder = Embedder(settings.embed_model)
    reranker = Reranker(settings.rerank_model) if settings.rerank_enabled else None
    llm = get_llm(settings)
    return Services(settings=settings, store=store, embedder=embedder, llm=llm, reranker=reranker)


@app.command()
def index(root: Annotated[Path, typer.Argument(help="Directory of documents to index.")]) -> None:
    """Build (or rebuild) the embedding index from a document folder."""
    settings = Settings.from_env()
    store = IndexStore(settings.index_dir)
    chunks = load_documents(root, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if not chunks:
        typer.secho(f"no supported documents found under {root}", fg=typer.colors.RED)
        raise typer.Exit(1)
    embedder = Embedder(settings.embed_model)
    typer.echo(f"Chunking {len(chunks)} chunks...")
    embeddings = embedder.encode([c.text for c in chunks]).cpu().numpy().astype(np.float32)
    store.clear()
    manifest = store.save(chunks, embeddings, settings.embed_model)
    typer.secho(
        f"Indexed {manifest.n_chunks} chunks ({manifest.dimension}-dim embeddings) "
        f"with {manifest.model_name} into {store.index_dir}",
        fg=typer.colors.GREEN,
    )


@app.command()
def ask(question: Annotated[str, typer.Argument(help="The question to answer.")]) -> None:
    """Ask a question over the indexed documents."""
    settings = Settings.from_env()
    services = _services(settings)
    graph = build_graph(services)
    result = run_agent(graph, question)

    if not result.get("relevant"):
        typer.secho(result["answer"], fg=typer.colors.YELLOW)
        raise typer.Exit(1)

    typer.secho(result["answer"], fg=typer.colors.WHITE)
    typer.echo("")
    typer.secho("Sources:", bold=True)
    seen = set()
    for chunk in result["relevant"]:
        key = chunk["source"]
        if key in seen:
            continue
        seen.add(key)
        heading = f" — {chunk['heading']}" if chunk.get("heading") else ""
        typer.echo(f"  • {key}{heading}  (score {chunk['score']:.2f})")


@app.command()
def status() -> None:
    """Show the current index state."""
    settings = Settings.from_env()
    store = IndexStore(settings.index_dir)
    if not store.exists():
        typer.secho("No index found. Run `gigi index <dir>` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    chunks, _, manifest = store.load()
    sources = sorted({c.source for c in chunks})
    typer.echo(f"Index dir : {store.index_dir}")
    typer.echo(f"Chunks    : {manifest.n_chunks}")
    typer.echo(f"Embed dim : {manifest.dimension}")
    typer.echo(f"Model     : {manifest.model_name}")
    typer.echo(f"Created   : {manifest.created_at}")
    typer.echo(f"Files     : {len(sources)}")


@app.command()
def version() -> None:
    """Print the gigi version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()