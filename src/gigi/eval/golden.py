"""Golden question set over the committed multilingual fixture corpus.

`GOLDEN` is the ground truth for `gigi eval`. Each *specific* case expects the
chunk that directly answers the question; each *overview* case expects every
fixture document, since a corpus-overview query must surface topics across the
whole index (one representative per cluster).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CORPUS_DIR = Path(__file__).parent / "corpus"

_ALL_DOCS = (
    # English
    "deploy.en.md",
    "onboarding.en.md",
    "backups.en.md",
    "monitoring.en.md",
    "api.en.md",
    "ci.en.md",
    "costs.en.md",
    # Italian
    "deploy.it.md",
    "onboarding.it.md",
    "backups.it.md",
    "database.it.md",
    "sicurezza.it.md",
    "rete.it.md",
    # Spanish
    "despliegue.es.md",
    "seguridad.es.md",
    "frontend.es.md",
    "networking.es.md",
    "almacenamiento.es.md",
    "alertas.es.md",
    # Catalan
    "dades.ca.md",
    "sistemes.ca.md",
    "accessibilitat.ca.md",
    "publicacio.ca.md",
)


@dataclass(frozen=True)
class GoldenCase:
    question: str
    expected: tuple[str, ...]
    language: str
    kind: Literal["specific", "overview"]


GOLDEN = (
    # English — specific (5)
    GoldenCase(
        "How do I deploy the application to production?",
        ("deploy.en.md",),
        "en",
        "specific",
    ),
    GoldenCase(
        "What is the backup retention policy?",
        ("backups.en.md",),
        "en",
        "specific",
    ),
    GoldenCase(
        "How do I create a monitoring alert?",
        ("monitoring.en.md",),
        "en",
        "specific",
    ),
    GoldenCase(
        "How is API access authenticated?",
        ("api.en.md",),
        "en",
        "specific",
    ),
    GoldenCase(
        "How do I onboard a new teammate?",
        ("onboarding.en.md",),
        "en",
        "specific",
    ),
    # Italian — specific (5)
    GoldenCase(
        "come si rilascia una nuova versione in produzione?",
        ("deploy.it.md",),
        "it",
        "specific",
    ),
    GoldenCase(
        "quanto tempo conserviamo i backup?",
        ("backups.it.md",),
        "it",
        "specific",
    ),
    GoldenCase(
        "come si aggiunge un nuovo membro al team?",
        ("onboarding.it.md",),
        "it",
        "specific",
    ),
    GoldenCase(
        "come si esegue una migrazione del database?",
        ("database.it.md",),
        "it",
        "specific",
    ),
    GoldenCase(
        "quali misure di sicurezza adottiamo?",
        ("sicurezza.it.md",),
        "it",
        "specific",
    ),
    # Spanish — specific (4)
    GoldenCase(
        "¿cómo se publica una release?",
        ("despliegue.es.md",),
        "es",
        "specific",
    ),
    GoldenCase(
        "¿qué política de contraseñas usamos?",
        ("seguridad.es.md",),
        "es",
        "specific",
    ),
    GoldenCase(
        "¿cómo se compila el frontend?",
        ("frontend.es.md",),
        "es",
        "specific",
    ),
    GoldenCase(
        "¿cómo se configura la VPN?",
        ("networking.es.md",),
        "es",
        "specific",
    ),
    # Catalan — specific (2)
    GoldenCase(
        "què fem amb les dades personals?",
        ("dades.ca.md",),
        "ca",
        "specific",
    ),
    GoldenCase(
        "com es publica el lloc web?",
        ("publicacio.ca.md",),
        "ca",
        "specific",
    ),
    # Overview — one per language (4). Expected = every fixture document:
    # a corpus-overview answer should surface a topic from each document.
    GoldenCase("what are these documents about?", _ALL_DOCS, "en", "overview"),
    GoldenCase("di cosa parlano questi documenti?", _ALL_DOCS, "it", "overview"),
    GoldenCase("de qué tratan estos documentos?", _ALL_DOCS, "es", "overview"),
    GoldenCase("de què tracten aquests documents?", _ALL_DOCS, "ca", "overview"),
)