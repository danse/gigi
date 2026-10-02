"""Offline retrieval-quality evaluation over the committed fixture corpus.

`gigi eval` exercises the exact production pipeline (`gigi index` + `gigi ask`)
but never calls the LLM: the fixture corpus is embedded once into a temporary
index, and every golden question is scored against the retrieval result.

Metrics (all offline):
  * specific cases — recall@8 (pre-grade candidates), MRR and answer-basis rate
    (post-grade), bail rate (nothing relevant survives grading)
  * overview cases — cluster-representative coverage of the fixture documents

`--grid` sweeps the retrieval knobs (top_k, rerank, thresholds, MMR lambda,
n_clusters) against the same embeddings. Cross-encoder scores are precomputed
once per question and reused across configs, so the sweep stays fast.
"""

from __future__ import annotations

import itertools
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from gigi.config import Settings
from gigi.eval.golden import CORPUS_DIR, GOLDEN, GoldenCase
from gigi.eval.metrics import bootstrap_ci, mean, mrr, overview_coverage, recall_at_k
from gigi.indexing.build import build_index
from gigi.indexing.cluster import ClusterRecord, build_clusters
from gigi.indexing.embedder import Embedder
from gigi.indexing.loader import Chunk
from gigi.indexing.store import IndexStore
from gigi.retrieval.overview import cluster_representatives
from gigi.retrieval.rerank import Reranker
from gigi.retrieval.search import mmr_top_k

KNOBS = ("top_k", "rerank_top_k", "rerank_enabled", "grade_threshold", "mmr_lambda", "n_clusters")

# Knob sweep for `--grid`. chunk_size/chunk_overlap are deliberately excluded:
# they change the chunking itself, so sweeping them would be mixing signals.
GRID = {
    "top_k": [8, 16, 32],
    "rerank_top_k": [4, 8],
    "rerank_enabled": [True, False],
    "grade_threshold": [None, 0.2, 0.3, 0.5],
    "mmr_lambda": [0.3, 0.5, 0.7, 0.9],
    "n_clusters": [8, 16, 24],
}


@dataclass
class CaseResult:
    question: str
    language: str
    kind: str
    recall8: float
    mrr: float
    answer_basis: float
    bail: bool
    coverage: float
    expected: str

    def score(self) -> float:
        if self.kind == "overview":
            return self.coverage
        return (
            0.4 * self.recall8
            + 0.3 * self.mrr
            + 0.2 * self.answer_basis
            + 0.1 * (0.0 if self.bail else 1.0)
        )


@dataclass
class ConfigReport:
    label: str
    settings: dict
    cases: list[CaseResult]
    score: float
    recall8: float
    mrr: float
    answer_basis: float
    bail_rate: float
    coverage: float
    # Percentile-bootstrap 95% CIs per metric (None when `n_boot == 0`).
    ci: dict[str, tuple[float, float]] | None = None
    n_boot: int = 0
    seed: int = 0


@dataclass
class EvalReport:
    baseline: ConfigReport
    grid: list[ConfigReport]


def _chunk_dict(source: str, text: str, heading: str, score: float) -> dict:
    return {"source": source, "text": text, "heading": heading, "score": score}


class RetrievalStage:
    """Mirror of the graph's retrieve+grade layers, without the LangGraph shell.

    When *precomputed* scores are supplied the reranker is not called again per
    config: the cross-encoder score of every (question, chunk) pair is a fixed
    number, so selection by it is equivalent to re-running the model.
    """

    def __init__(
        self,
        embedder: Embedder,
        reranker: Reranker | None,
        chunks: list[Chunk],
        embeddings,
        precomputed: dict[str, dict[int, float]] | None = None,
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.chunks = chunks
        self.embeddings = embeddings
        self.precomputed = precomputed or {}
        self._query_cache: dict[str, object] = {}

    def retrieve(self, question: str, settings: Settings) -> list[dict]:
        if question not in self._query_cache:
            self._query_cache[question] = self.embedder.encode_one(question)
        query_emb = self._query_cache[question]
        candidates = mmr_top_k(
            self.chunks,
            self.embeddings,
            query_emb,
            k=settings.top_k,
            lambda_=settings.mmr_lambda,
        )
        if settings.rerank_enabled and self.reranker is not None:
            scores = self.precomputed.get(question)
            if scores is not None:
                ranked = [(c, scores[id(c)]) for c, _ in candidates]
                ranked.sort(key=lambda x: x[1], reverse=True)
                candidates = ranked[: settings.rerank_top_k]
            else:
                candidates = self.reranker.rerank(
                    question,
                    [c for c, _ in candidates],
                    top_k=settings.rerank_top_k,
                )
        return [
            _chunk_dict(c.source, c.text, c.heading, score) for c, score in candidates
        ]


def _knob_dict(settings: Settings) -> dict:
    return {k: getattr(settings, k) for k in KNOBS}


def _precompute_rerank(
    reranker: Reranker, chunks: list[Chunk], cases: tuple[GoldenCase, ...]
) -> dict[str, dict[int, float]]:
    """Cross-encoder score of every needle chunk for every specific question,
    so the grid sweep never touches the model again."""
    out: dict[str, dict[int, float]] = {}
    for case in cases:
        if case.kind != "specific":
            continue
        ranked = reranker.rerank(case.question, chunks, top_k=None)
        out[case.question] = {id(c): score for c, score in ranked}
    return out


def evaluate_config(
    label: str,
    settings: Settings,
    stage: RetrievalStage,
    store: IndexStore,
    chunks: list[Chunk],
    embeddings,
    cases: tuple[GoldenCase, ...] = GOLDEN,
    clusters: list[ClusterRecord] | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> ConfigReport:
    if clusters is None:
        clusters = store.load_clusters()
    results: list[CaseResult] = []
    threshold = settings.resolve_grade_threshold()

    for case in cases:
        if case.kind == "overview":
            rep_sources = {
                r["source"]
                for r in cluster_representatives(
                    chunks, clusters, per_cluster=settings.overview_per_cluster
                )
            }
            results.append(
                CaseResult(
                    question=case.question,
                    language=case.language,
                    kind="overview",
                    recall8=0.0,
                    mrr=0.0,
                    answer_basis=0.0,
                    bail=False,
                    coverage=overview_coverage(rep_sources, case.expected),
                    expected=", ".join(case.expected),
                )
            )
            continue

        retrieved = stage.retrieve(case.question, settings)
        relevant = (
            retrieved
            if threshold is None
            else [r for r in retrieved if r["score"] >= threshold]
        )
        pre_hits = [r["source"] for r in retrieved]
        rel_hits = [r["source"] for r in relevant]
        expected = set(case.expected)
        results.append(
            CaseResult(
                question=case.question,
                language=case.language,
                kind="specific",
                recall8=recall_at_k(pre_hits, expected, 8),
                mrr=mrr(rel_hits, expected),
                answer_basis=1.0 if rel_hits and rel_hits[0] in expected else 0.0,
                bail=not rel_hits,
                coverage=0.0,
                expected=", ".join(case.expected),
            )
        )

    return _aggregate(label, settings, results, n_boot=n_boot, seed=seed)


def _aggregate(
    label: str,
    settings: Settings,
    results: list[CaseResult],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> ConfigReport:
    specific = [r for r in results if r.kind == "specific"]
    overview = [r for r in results if r.kind == "overview"]
    ci = None
    if n_boot > 0:
        ci = {
            "score": bootstrap_ci([r.score() for r in results], n_boot=n_boot, seed=seed),
            "recall8": bootstrap_ci([r.recall8 for r in specific], n_boot=n_boot, seed=seed),
            "mrr": bootstrap_ci([r.mrr for r in specific], n_boot=n_boot, seed=seed),
            "answer_basis": bootstrap_ci(
                [r.answer_basis for r in specific], n_boot=n_boot, seed=seed
            ),
            "bail_rate": bootstrap_ci(
                [1.0 if r.bail else 0.0 for r in specific], n_boot=n_boot, seed=seed
            ),
            "coverage": bootstrap_ci([r.coverage for r in overview], n_boot=n_boot, seed=seed),
        }
    return ConfigReport(
        label=label,
        settings=_knob_dict(settings),
        cases=results,
        score=mean([r.score() for r in results]),
        recall8=mean([r.recall8 for r in specific]),
        mrr=mean([r.mrr for r in specific]),
        answer_basis=mean([r.answer_basis for r in specific]),
        bail_rate=mean([1.0 if r.bail else 0.0 for r in specific]),
        coverage=mean([r.coverage for r in overview]),
        ci=ci,
        n_boot=n_boot,
        seed=seed,
    )


def _grid_label(params: dict) -> str:
    grade = params["grade_threshold"]
    return (
        f"top={params['top_k']} rrk={params['rerank_top_k']} "
        f"rerank={'y' if params['rerank_enabled'] else 'n'} "
        f"grade={grade} λ={params['mmr_lambda']} cl={params['n_clusters']}"
    )


def run_eval(
    settings: Settings | None = None,
    *,
    grid: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> EvalReport:
    """Build a throwaway fixture index and score the golden set against it."""
    settings = settings or Settings.from_env()
    tmp_root = Path(tempfile.mkdtemp(prefix="gigi-eval-"))
    index_dir = tmp_root / ".index"
    try:
        return _run_eval_in(settings, index_dir, grid=grid, n_boot=n_boot, seed=seed)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _run_eval_in(
    settings: Settings,
    index_dir: Path,
    *,
    grid: bool,
    n_boot: int = 2000,
    seed: int = 0,
) -> EvalReport:
    store, _manifest = build_index(CORPUS_DIR, settings, index_dir=index_dir)
    chunks, embeddings, _ = store.load()
    # the store records full paths as sources; the golden set uses bare file
    # names, so normalize before scoring.
    for chunk in chunks:
        chunk.source = Path(chunk.source).name

    embedder = Embedder(settings.embed_model)
    reranker: Reranker | None = None
    if grid or settings.rerank_enabled:
        reranker = Reranker(settings.rerank_model)
    precomputed = (
        _precompute_rerank(reranker, chunks, GOLDEN) if reranker is not None else None
    )
    stage = RetrievalStage(embedder, reranker, chunks, embeddings, precomputed=precomputed)

    baseline = evaluate_config(
        "baseline", settings, stage, store, chunks, embeddings, n_boot=n_boot, seed=seed
    )
    if not grid:
        return EvalReport(baseline=baseline, grid=[])

    cluster_cache: dict[int, list[ClusterRecord]] = {}
    reports: list[ConfigReport] = []
    keys = tuple(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        cfg = replace(settings, **params)
        n_clusters = cfg.n_clusters
        if n_clusters not in cluster_cache:
            cluster_cache[n_clusters] = build_clusters(chunks, embeddings, n_clusters=n_clusters)
        reports.append(
            evaluate_config(
                _grid_label(params),
                cfg,
                stage,
                store,
                chunks,
                embeddings,
                clusters=cluster_cache[n_clusters],
                n_boot=n_boot,
                seed=seed,
            )
        )
    reports.sort(key=lambda r: r.score, reverse=True)
    return EvalReport(baseline=baseline, grid=reports)


def format_report(report: EvalReport) -> str:
    baseline = report.baseline
    n_docs = len(list(CORPUS_DIR.glob("*.md")))
    lines = [
        (
            f"gigi eval — {len(baseline.cases)} golden questions over {n_docs} fixture "
            "docs (en/it/es/ca) — offline, no LLM"
        ),
        "",
    ]
    header = "baseline (default settings):"
    if baseline.ci:
        header += (
            f"   [95% CI: {baseline.n_boot} bootstrap resamples, seed {baseline.seed}]"
        )
    lines.append(header)
    for name, value, key in (
        ("score", baseline.score, "score"),
        ("recall@8", baseline.recall8, "recall8"),
        ("mrr", baseline.mrr, "mrr"),
        ("answer-basis", baseline.answer_basis, "answer_basis"),
        ("bail rate", baseline.bail_rate, "bail_rate"),
        ("overview cov", baseline.coverage, "coverage"),
    ):
        ci = baseline.ci.get(key) if baseline.ci else None
        line = f"  {name:<13}{value:.3f}"
        if ci:
            line += f"  [{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(line)
    lines.append("")
    lines.append("cases (specific):")
    for case in baseline.cases:
        if case.kind != "specific":
            continue
        mark = "✓" if case.answer_basis else ("✗" if case.bail else "~")
        lines.append(
            f"  [{case.language:2}] rec8={case.recall8:.2f} mrr={case.mrr:.2f} "
            f"basis={mark}  {case.question}  ->  {case.expected}"
        )
    lines.append("")
    lines.append("cases (overview):")
    for case in baseline.cases:
        if case.kind != "overview":
            continue
        lines.append(f"  [{case.language:2}] cov={case.coverage:.2f}  {case.question}")
    if report.grid:
        lines.append("")
        lines.append(
            f"knob sweep — {len(report.grid)} configs "
            "(top_k × rerank_top_k × rerank × grade × λ × n_clusters), best first:"
        )
        lines.append("   #   score  ci95                  rec8   mrr    cov   config")
        for i, config in enumerate(report.grid[:12], start=1):
            ci = config.ci.get("score") if config.ci else None
            ci95 = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
            lines.append(
                f"  {i:>2}  {config.score:.3f}  {ci95:<18}{config.recall8:.3f} "
                f"{config.mrr:.3f} {config.coverage:.3f}  {config.label}"
            )
        bci = baseline.ci.get("score") if baseline.ci else None
        bci95 = f"[{bci[0]:.3f}, {bci[1]:.3f}]" if bci else "n/a"
        lines.append(
            f"  -   {baseline.score:.3f}  {bci95:<18}{baseline.recall8:.3f} "
            f"{baseline.mrr:.3f} {baseline.coverage:.3f}  baseline"
        )
    return "\n".join(lines)


def to_json(report: EvalReport) -> str:
    return json.dumps(asdict(report), indent=2)