"""Round-trip TTNN attrs through the in-tree ttmlir-opt binary.

The compiler emits a textual TTNN snippet; this module parses each attr
back through `third_party/tt-mlir/build/bin/ttmlir-opt` so a divergence
between the emit syntax and the pinned tt-mlir dialect fails loudly.

The runner assumes a successful `scripts/build_ttmlir.sh` has produced
the in-tree binary; it never falls back to a prebuilt out-of-tree
checkout.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Tuple


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "..", "..", ".."))
_TTMLIR_OPT = os.path.join(
    _ROOT, "third_party", "tt-mlir", "build", "bin", "ttmlir-opt")

# Attrs the codegen currently emits.
_EMITTED_ATTRS: Tuple[str, ...] = (
    "program",
    "kernel_cb",
    "kernel_cb_format",
    "kernel_semaphore",
    "compute_kernel",
    "source_compute_kernel",
)


class TTNNLintError(RuntimeError):
    pass


@dataclass(frozen=True)
class LintResult:
    attrs_seen: Tuple[str, ...]
    attrs_rejected: Tuple[Tuple[str, str], ...]


def ensure_available() -> None:
    if not os.path.exists(_TTMLIR_OPT):
        raise TTNNLintError(
            f"ttmlir-opt not found at {_TTMLIR_OPT}. Run "
            f"scripts/build_ttmlir.sh first.")


def _balanced_slice(text: str, start: int) -> int:
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _run_attr(attr_text: str) -> subprocess.CompletedProcess:
    wrapped = f"module attributes {{ttnn.probe = {attr_text}}} {{}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(wrapped)
        path = f.name
    try:
        return subprocess.run([_TTMLIR_OPT, path], capture_output=True,
                              text=True, timeout=30)
    finally:
        os.unlink(path)


def lint(emitted: str) -> LintResult:
    """Round-trip each occurrence of every tracked attr."""
    ensure_available()
    seen: list = []
    rejected: list = []
    for mnemonic in _EMITTED_ATTRS:
        token = f"#ttnn.{mnemonic}<"
        if token not in emitted:
            continue
        seen.append(mnemonic)
        cursor = 0
        while True:
            start = emitted.find(token, cursor)
            if start < 0:
                break
            open_bracket = start + len(token) - 1
            close = _balanced_slice(emitted, open_bracket)
            if close < 0:
                rejected.append((mnemonic, "<unterminated attr>"))
                break
            attr = emitted[start:close + 1]
            r = _run_attr(attr)
            if r.returncode != 0 or "error" in (r.stderr or "").lower():
                rejected.append((mnemonic, (r.stderr or r.stdout).strip()))
            cursor = close + 1
    return LintResult(attrs_seen=tuple(seen),
                      attrs_rejected=tuple(rejected))
