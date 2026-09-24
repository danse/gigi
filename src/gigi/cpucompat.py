"""Pin MKL/oneDNN/ATen to SSE on x86 CPUs that lack AVX.

This module must be imported before numpy or torch. Those libraries JIT
kernels per OpenMP thread; without an explicit cap, some threads can emit
AVX code on CPUs that only have SSE (intermittent SIGILL).
"""

from __future__ import annotations

import os
from pathlib import Path

_ISA_ENV = {
    "ATEN_CPU_CAPABILITY": "default",
    "MKL_ENABLE_INSTRUCTIONS": "SSE4_2",
    "DNNL_MAX_CPU_ISA": "SSE41",
    "ONEDNN_MAX_CPU_ISA": "SSE41",
}


def cpu_flags() -> set[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("flags", "Features")):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def needs_sse_cap(flags: set[str] | None = None) -> bool:
    flags = cpu_flags() if flags is None else flags
    if not flags:
        return False
    x86 = bool({"sse2", "sse4_1", "sse4_2"} & flags)
    if not x86:
        return False
    return "avx" not in flags and "avx2" not in flags


def configure_cpu(flags: set[str] | None = None) -> bool:
    """Set ISA env vars when AVX is unavailable. Return True if pinned."""
    if not needs_sse_cap(flags):
        return False
    for key, value in _ISA_ENV.items():
        os.environ.setdefault(key, value)
    return True


def configure_torch(torch_module: object) -> None:
    """Disable mkldnn after torch import when we pinned SSE."""
    if os.environ.get("ATEN_CPU_CAPABILITY") != "default":
        return
    backends = getattr(torch_module, "backends", None)
    mkldnn = getattr(backends, "mkldnn", None)
    if mkldnn is not None:
        mkldnn.enabled = False


configure_cpu()
