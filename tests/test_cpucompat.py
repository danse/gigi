"""CPU ISA pinning for machines without AVX (no model downloads)."""

from __future__ import annotations

import os

from gigi.cpucompat import configure_cpu, needs_sse_cap


def test_needs_sse_cap_on_celeron_like_flags():
    assert needs_sse_cap({"sse2", "sse4_1", "sse4_2", "ssse3"})
    assert not needs_sse_cap({"sse2", "sse4_2", "avx", "avx2"})
    assert not needs_sse_cap({"fp", "asimd", "aes"})  # aarch64
    assert not needs_sse_cap(set())


def test_configure_cpu_pins_without_avx(monkeypatch):
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    monkeypatch.delenv("MKL_ENABLE_INSTRUCTIONS", raising=False)
    monkeypatch.delenv("ONEDNN_MAX_CPU_ISA", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert configure_cpu({"sse2", "sse4_2"}) is True
    assert os.environ["ATEN_CPU_CAPABILITY"] == "default"
    assert os.environ["MKL_ENABLE_INSTRUCTIONS"] == "SSE4_2"
    assert os.environ["ONEDNN_MAX_CPU_ISA"] == "SSE41"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_configure_cpu_skips_avx_hosts(monkeypatch):
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    assert configure_cpu({"sse2", "avx2"}) is False
    assert "ATEN_CPU_CAPABILITY" not in os.environ


def test_configure_cpu_respects_existing_env(monkeypatch):
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "avx2")
    assert configure_cpu({"sse2", "sse4_2"}) is True
    assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"
