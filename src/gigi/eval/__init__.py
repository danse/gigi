"""Offline retrieval-quality evaluation (`gigi eval`)."""

from gigi.eval.golden import CORPUS_DIR, GOLDEN
from gigi.eval.metrics import mrr, overview_coverage, recall_at_k

__all__ = ["CORPUS_DIR", "GOLDEN", "mrr", "overview_coverage", "recall_at_k"]