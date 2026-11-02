"""Structural lint for emitted TTNN text against the submodule's TD.

A real round-trip would pipe the emitted module through tt-mlir's
`ttmlir-opt`, which needs the tt-mlir dialect libraries built with the
nix FHS toolchain. That build is a separate workstream (CP8+).

This module stands in: it extracts each attr's parameter list from
`third_party/tt-mlir/include/ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.td`
and verifies that the codegen's emitted text names every required key.
A TD change that renames or drops a key fails the lint; a new TTNN
syntax that round-trips through tt-mlir does not. Honest limitation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple


_TD_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "third_party", "tt-mlir",
    "include", "ttmlir", "Dialect", "TTNN", "IR", "TTNNOpsAttrs.td",
))

# Attrs the codegen currently emits. Each entry maps the attr's mnemonic
# (as it appears in `#ttnn.<mnemonic>`) to its TD symbol name.
_EMITTED_ATTRS: Dict[str, str] = {
    "program": "TTNN_ProgramAttr",
    "kernel_cb": "TTNN_KernelCBAttr",
    "kernel_cb_format": "TTNN_KernelCBFormatAttr",
    "kernel_semaphore": "TTNN_KernelSemaphoreAttr",
    "compute_kernel": "TTNN_ComputeKernelAttr",
}

# TD params the codegen is responsible for emitting. The lint also
# cross-checks against the TD definition so a param drop here is caught
# by extract_td_params below.
_REQUIRED_KEYS: Dict[str, FrozenSet[str]] = {
    "program": frozenset({"kernels", "cbs", "semaphores"}),
    "kernel_cb": frozenset({"total_size", "core_ranges", "formats"}),
    "kernel_cb_format": frozenset({"buffer_index", "dtype", "page_size"}),
    "kernel_semaphore": frozenset({
        "id", "core_type", "core_ranges", "initial_value",
    }),
    "compute_kernel": frozenset({"symbol_ref", "core_ranges"}),
}


@dataclass(frozen=True)
class LintResult:
    attrs_seen: Tuple[str, ...]
    attrs_missing_keys: Tuple[Tuple[str, str], ...]


class TTNNLintError(ValueError):
    pass


def extract_td_params(symbol: str) -> FrozenSet[str]:
    """Return the parameter names declared on a TTNN TD attr by symbol name.

    Looks for a line `def <symbol> ...` and then the parameters block
    `let parameters = (ins ... );`. Variable name tokens starting with `$`
    are the parameter names.
    """
    with open(_TD_PATH) as f:
        src = f.read()
    m = re.search(rf"^def {re.escape(symbol)}\b.*?let parameters = \(ins\s+(.*?)\);",
                  src, re.DOTALL | re.MULTILINE)
    if m is None:
        raise TTNNLintError(f"{symbol} not found in TTNNOpsAttrs.td")
    body = m.group(1)
    names = re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", body)
    return frozenset(names)


def _balanced_slice(text: str, start: int) -> int:
    """Return the index of the `>` that closes the `<` at `start`.

    Counts nested `<>` so an inner attr body does not end the slice early.
    """
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


def lint(emitted: str) -> LintResult:
    """Verify every attr the codegen uses carries its required TD keys."""
    seen = []
    missing: list = []
    for mnemonic, keys in _REQUIRED_KEYS.items():
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
                missing.append((mnemonic, "<unterminated>"))
                break
            body = emitted[open_bracket + 1:close]
            for k in keys:
                if f"{k} =" not in body and f"{k}=" not in body:
                    missing.append((mnemonic, k))
            cursor = close + 1
    return LintResult(attrs_seen=tuple(seen),
                      attrs_missing_keys=tuple(missing))


def validate_td_sync() -> None:
    """Raise if `_REQUIRED_KEYS` names a param not declared in the TD."""
    for mnemonic, symbol in _EMITTED_ATTRS.items():
        td_params = extract_td_params(symbol)
        declared = _REQUIRED_KEYS[mnemonic]
        unknown = declared - td_params
        if unknown:
            raise TTNNLintError(
                f"{mnemonic} ({symbol}): lint expects keys not in TD: "
                f"{sorted(unknown)}")
