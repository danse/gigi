"""Pluggable LLM backend used by the generate node."""

from __future__ import annotations

import httpx

from gigi.config import Settings


class LLMError(Exception):
    pass


class LLM:
    def complete(self, messages: list[dict]) -> str:
        raise NotImplementedError


class OllamaLLM(LLM):
    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3.2", timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        resp = httpx.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LLMError(f"ollama error {resp.status_code}: {resp.text}")
        return resp.json()["message"]["content"]


class OpenAICompatLLM(LLM):
    def __init__(self, url: str, api_key: str, model: str, timeout: float = 180.0):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        resp = httpx.post(
            f"{self.url}/v1/chat/completions",
            json={"model": self.model, "messages": messages},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LLMError(f"openai-compatible error {resp.status_code}: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]


class StubLLM(LLM):
    """Offline LLM for tests/demo: echoes the top context line so the graph is runnable without a server."""

    def complete(self, messages: list[dict]) -> str:
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        first_line = user.splitlines()[1] if len(user.splitlines()) > 1 else user
        snippet = first_line.strip().strip("[]").replace("source:", "source:")
        return f"[stub] Answer synthesized from context. Top source: {snippet[:200]}"


def get_llm(settings: Settings) -> LLM:
    kind = settings.llm_kind.lower()
    if kind == "ollama":
        return OllamaLLM(url=settings.ollama_url, model=settings.ollama_model)
    if kind == "openai":
        if not settings.openai_url or not settings.openai_api_key:
            raise LLMError("GIGI_LLM=openai requires GIGI_OPENAI_URL and GIGI_OPENAI_API_KEY")
        return OpenAICompatLLM(
            url=settings.openai_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model or "gpt-4o-mini",
        )
    if kind == "stub":
        return StubLLM()
    raise LLMError(f"unknown GIGI_LLM kind: {kind!r} (expected ollama | openai | stub)")