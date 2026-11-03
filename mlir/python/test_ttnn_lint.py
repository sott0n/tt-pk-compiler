"""CP7b gate: round-trip emitted TTNN attrs through in-tree ttmlir-opt.

Builds the Qwen3.6-27B schedule, emits its TTNN snippet, and parses each
`#ttnn.program / kernel_cb / kernel_cb_format / kernel_semaphore /
compute_kernel` attribute through ttmlir-opt. Requires the in-tree
submodule binary; run `scripts/build_ttmlir.sh` once before running.

A negative check mutates one required key and confirms ttmlir-opt rejects
the broken attribute.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import (TTNNLintError, emit_ttnn_generic, ensure_available,
                   greedy_search, lint, lower_to_resident)
from tt_pk.qwen36 import qwen36_27b_problem


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    try:
        ensure_available()
    except TTNNLintError as e:
        print(f"[SKIP] {e}")
        return

    problem = qwen36_27b_problem()
    schedule = lower_to_resident(problem, greedy_search(problem))
    text = emit_ttnn_generic(schedule)

    result = lint(text)
    for attr in ("program", "kernel_cb", "kernel_cb_format",
                 "source_compute_kernel"):
        _check(attr in result.attrs_seen,
               f"ttmlir-opt saw ttnn.{attr}")
    _check(result.attrs_rejected == (),
           f"ttmlir-opt accepted every emitted attr "
           f"(rejected: {result.attrs_rejected})")

    broken = text.replace("total_size =", "garbage =", 1)
    broken_result = lint(broken)
    _check(any("kernel_cb" == m[0] for m in broken_result.attrs_rejected),
           "ttmlir-opt rejects a kernel_cb with a dropped total_size")

    print("OK")


if __name__ == "__main__":
    main()
