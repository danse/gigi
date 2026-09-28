"""Multilingual corpus-overview question detection (no model downloads)."""

from __future__ import annotations

from gigi.retrieval.overview import is_overview_query


def test_english_overview_questions():
    assert is_overview_query("what are these documents about?")
    assert is_overview_query("summarize the corpus")
    assert is_overview_query("summary")


def test_italian_overview_questions():
    assert is_overview_query("di cosa parlano questi documenti?")
    assert is_overview_query("riassumi i documenti")
    assert is_overview_query("sommario")


def test_spanish_overview_questions():
    assert is_overview_query("¿de qué tratan estos documentos?")
    assert is_overview_query("resumen de los documentos")
    assert is_overview_query("resume los documentos")


def test_catalan_overview_questions():
    assert is_overview_query("de què tracten aquests documents?")
    assert is_overview_query("resumeix els documents")
    assert is_overview_query("resum")


def test_non_overview_questions_arent_classified():
    assert not is_overview_query("che cos'è il deploy?")
    assert not is_overview_query("¿cómo despliego la app?")
    assert not is_overview_query("what is the deployment process?")