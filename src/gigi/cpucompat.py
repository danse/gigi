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
    # HuggingFace tokenizers (Rayon) and MKL OpenMP: one thread, no AVX JIT race.
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
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
    """Disable AVX-prone backends after torch import when we pinned SSE."""
    if os.environ.get("ATEN_CPU_CAPABILITY") != "default":
        return
    backends = getattr(torch_module, "backends", None)
    for name in ("mkldnn", "nnpack"):
        backend = getattr(backends, name, None)
        if backend is not None and hasattr(backend, "enabled"):
            backend.enabled = False
    try:
        torch_module.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass


configure_cpu()
