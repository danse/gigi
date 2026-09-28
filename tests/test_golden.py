"""Golden set integrity over the committed fixture corpus (no model downloads)."""

from __future__ import annotations

from gigi.config import Settings
from gigi.eval.golden import CORPUS_DIR, GOLDEN
from gigi.indexing.loader import chunk_text, read_text


def _fixture_files() -> set[str]:
    return {p.name for p in CORPUS_DIR.glob("*.md") if p.is_file()}


def test_corpus_is_committed_and_language_balanced():
    files = _fixture_files()
    assert len(files) >= 20
    langs: dict[str, int] = {}
    for name in files:
        lang = name.split(".")[1] if name.count(".") >= 2 else "?"
        langs[lang] = langs.get(lang, 0) + 1
    for lang in ("en", "it", "es", "ca"):
        assert langs.get(lang, 0) >= 4, f"corpus is too thin in {lang}"


def test_golden_questions_are_unique():
    questions = [c.question.strip().lower() for c in GOLDEN]
    assert len(questions) == len(set(questions))
    assert len(GOLDEN) >= 20


def test_golden_expected_sources_exist():
    files = _fixture_files()
    for case in GOLDEN:
        for source in case.expected:
            assert source in files, f"{case.question!r} expects missing {source}"


def test_golden_expected_sources_produce_chunks():
    size, overlap = Settings().chunk_size, Settings().chunk_overlap
    for case in GOLDEN:
        for source in case.expected:
            text = read_text(CORPUS_DIR / source)
            chunks = chunk_text(text, source, size=size, overlap=overlap)
            assert chunks, f"{source} (expected by {case.question!r}) has no body text"


def test_specific_cases_target_a_single_source():
    for case in GOLDEN:
        if case.kind == "specific":
            assert len(case.expected) == 1, (
                f"specific case {case.question!r} must target exactly one source"
            )


def test_overview_cases_cover_every_language():
    overviews = [c for c in GOLDEN if c.kind == "overview"]
    assert len(overviews) >= 1
    assert {c.language for c in overviews} == {"en", "it", "es", "ca"}


def test_overview_cases_expect_the_whole_corpus():
    all_docs = _fixture_files()
    for case in GOLDEN:
        if case.kind == "overview":
            assert set(case.expected) == all_docs, (
                f"overview case {case.question!r} must expect every fixture document"
            )


def test_question_language_matches_expected_source_language():
    for case in GOLDEN:
        if case.kind != "specific":
            continue
        for source in case.expected:
            expected_lang = source.split(".")[1] if source.count(".") >= 2 else None
            assert expected_lang == case.language, (
                f"{case.question!r} is [{case.language}] but expects {source} "
                f"which is [{expected_lang}]"
            )