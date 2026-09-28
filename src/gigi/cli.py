"""gigi command-line interface."""

from __future__ import annotations

import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Annotated

from gigi.cpucompat import configure_cpu, configure_torch

configure_cpu()

import numpy as np
import torch

configure_torch(torch)
import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from gigi import __version__
from gigi.agent.graph import build_graph, run_agent
from gigi.agent.llm import LLMError, get_llm
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

_THREAD_FILE = "thread_id"


def _load_thread_id(index_dir: Path) -> str | None:
    path = index_dir / _THREAD_FILE
    return path.read_text().strip() if path.exists() else None


def _save_thread_id(index_dir: Path, thread_id: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / _THREAD_FILE).write_text(thread_id)


def _checkpointer(index_dir: Path) -> AbstractContextManager[SqliteSaver]:
    """Durable conversation memory, so consecutive `ask` calls remember earlier turns.

    The returned value is a context manager: the sqlite connection it wraps must
    stay open for the whole graph run.
    """
    db_path = index_dir / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(db_path))


def _services(settings: Settings) -> Services:
    store = IndexStore(settings.index_dir)
    embedder = Embedder(settings.embed_model)
    reranker = Reranker(settings.rerank_model) if settings.rerank_enabled else None
    llm = get_llm(settings)
    return Services(settings=settings, store=store, embedder=embedder, llm=llm, reranker=reranker)


def _require_matching_index(store: IndexStore, settings: Settings) -> None:
    """Refuse to run against an index built with a different embedding model."""
    try:
        store.check_embed_model(settings.embed_model)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc


@app.command()
def index(
    root: Annotated[
        Path | None,
        typer.Argument(help="Directory of documents to index."),
    ] = None,
    recluster: Annotated[
        bool,
        typer.Option(
            "--recluster",
            help="Rebuild topic clusters from the existing embeddings, without re-embedding.",
        ),
    ] = False,
) -> None:
    """Build (or rebuild) the embedding index from a document folder."""
    settings = Settings.from_env()
    store = IndexStore(settings.index_dir)
    if recluster:
        _require_matching_index(store, settings)
        typer.echo(f"Reclustering embeddings in {store.index_dir}...")
        clusters = store.recluster(n_clusters=settings.n_clusters)
        typer.secho(
            f"Wrote {len(clusters)} topics from {sum(c.size for c in clusters)} chunks "
            f"into {store.clusters_file}",
            fg=typer.colors.GREEN,
        )
        return
    if root is None:
        typer.secho("document directory required (or pass --recluster)", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(f"Scanning {root} for documents...")
    chunks = load_documents(root, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if not chunks:
        typer.secho(f"no supported documents found under {root}", fg=typer.colors.RED)
        raise typer.Exit(1)
    embedder = Embedder(settings.embed_model)
    typer.echo(f"Found {len(chunks)} chunks; embedding with {settings.embed_model}...")
    batch_size = settings.embed_batch_size
    embedded = []
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task("Embedding", total=len(chunks))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embedded.append(embedder.encode_chunks([c.text for c in batch]))
            progress.update(
                task,
                advance=len(batch),
                description=f"Embedding {batch[-1].source}",
            )
    embeddings = torch.cat(embedded).cpu().numpy().astype(np.float32)
    store.clear()
    manifest = store.save(chunks, embeddings, settings.embed_model, n_clusters=settings.n_clusters)
    n_topics = len(store.load_clusters())
    typer.secho(
        f"Indexed {manifest.n_chunks} chunks ({manifest.dimension}-dim embeddings, "
        f"{n_topics} topics) with {manifest.model_name} into {store.index_dir}",
        fg=typer.colors.GREEN,
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Start a new conversation, forgetting previous turns."),
    ] = False,
) -> None:
    """Ask a question over the indexed documents. Consecutive asks continue the conversation."""
    settings = Settings.from_env()
    services = _services(settings)
    _require_matching_index(services.store, settings)
    index_dir = settings.index_dir

    thread_id = None if reset else _load_thread_id(index_dir)
    if thread_id is None:
        thread_id = uuid.uuid4().hex
    try:
        with _checkpointer(index_dir) as checkpointer:
            graph = build_graph(services, checkpointer=checkpointer)
            result = run_agent(graph, question, thread_id=thread_id)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    _save_thread_id(index_dir, thread_id)

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
    try:
        store.check_embed_model(settings.embed_model)
        chunks, _, manifest = store.load()
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    clusters = store.load_clusters()
    sources = sorted({c.source for c in chunks})
    typer.echo(f"Index dir : {store.index_dir}")
    typer.echo(f"Chunks    : {manifest.n_chunks}")
    typer.echo(f"Topics    : {len(clusters)}")
    typer.echo(f"Embed dim : {manifest.dimension}")
    typer.echo(f"Model     : {manifest.model_name}")
    typer.echo(f"Created   : {manifest.created_at}")
    typer.echo(f"Files     : {len(sources)}")
    if clusters:
        typer.echo("Top topics:")
        for cluster in clusters[:8]:
            typer.echo(f"  • {cluster.heading or cluster.source}  ({cluster.size} chunks)")


@app.command()
def version() -> None:
    """Print the gigi version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()