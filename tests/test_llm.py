"""LLM client error mapping (no network)."""

from __future__ import annotations

import httpx
import pytest

from gigi.agent.llm import LLMError, OllamaLLM


def test_ollama_timeout_becomes_llm_error(monkeypatch):
    llm = OllamaLLM(timeout=1)

    def boom(*_args, **_kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx, "request", boom)
    with pytest.raises(LLMError, match="timed out"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_ollama_connect_error_becomes_llm_error(monkeypatch):
    llm = OllamaLLM(timeout=1)

    def boom(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "request", boom)
    with pytest.raises(LLMError, match="could not reach"):
        llm.complete([{"role": "user", "content": "hi"}])
