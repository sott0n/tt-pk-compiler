"""Textual builder for the `tt_pk` dialect, verified by tt-pk-opt.

The builder keeps the Python surface narrow on purpose: a Program holds a
list of CoreSet / Kernel entries, `assemble` renders it as MLIR text, and
`verify` round-trips that text through tt-pk-opt. Replace with proper CAPI
bindings in a later checkpoint.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple


def _resolve_tt_pk_opt() -> str:
    env = os.environ.get("TT_PK_OPT")
    if env:
        return env
    found = shutil.which("tt-pk-opt")
    if found:
        return found
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default = os.path.join(repo_root, "mlir", "build", "dialect", "tools",
                           "tt-pk-opt", "tt-pk-opt")
    if os.path.exists(default):
        return default
    raise FileNotFoundError(
        "tt-pk-opt not on PATH; set TT_PK_OPT or build mlir/build/.../tt-pk-opt")


@dataclass(frozen=True)
class CoreSet:
    name: str
    cores: Tuple[Tuple[int, int], ...]

    def flat(self) -> List[int]:
        out: List[int] = []
        for x, y in self.cores:
            out.extend((x, y))
        return out


@dataclass(frozen=True)
class Kernel:
    kernel_name: str
    core_set: CoreSet


@dataclass
class Program:
    core_sets: List[CoreSet] = field(default_factory=list)
    kernels: List[Kernel] = field(default_factory=list)

    def add_core_set(self, name: str, cores: Iterable[Tuple[int, int]]) -> CoreSet:
        cs = CoreSet(name=name, cores=tuple(cores))
        self.core_sets.append(cs)
        return cs

    def launch(self, kernel_name: str, core_set: CoreSet) -> Kernel:
        k = Kernel(kernel_name=kernel_name, core_set=core_set)
        self.kernels.append(k)
        return k


def assemble(program: Program) -> str:
    lines = ["tt_pk.program {"]
    for cs in program.core_sets:
        flat = ", ".join(str(c) for c in cs.flat())
        lines.append(f"  %{cs.name} = tt_pk.core_set [{flat}] : !tt_pk.core_set")
    for k in program.kernels:
        lines.append(
            f'  tt_pk.kernel "{k.kernel_name}" on %{k.core_set.name} '
            f": !tt_pk.core_set")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


@dataclass
class VerifyResult:
    ok: bool
    stdout: str
    stderr: str


def verify(program: Program) -> VerifyResult:
    text = assemble(program)
    tool = _resolve_tt_pk_opt()
    proc = subprocess.run(
        [tool],
        input=text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return VerifyResult(ok=proc.returncode == 0,
                        stdout=proc.stdout,
                        stderr=proc.stderr)
