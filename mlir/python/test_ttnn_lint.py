"""CP7b gate: TD-driven structural lint of the emitted TTNN text.

Real round-trip through tt-mlir's ttmlir-opt needs the tt-mlir dialect
libraries built with the nix FHS toolchain (see mlir/INTERFACE.md). The
lint stands in for now; it checks every attr the codegen emits against
the parameter names declared in the submodule's TTNNOpsAttrs.td.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import (TTNNLintError, emit_ttnn_generic, extract_td_params,
                   greedy_search, lint, lower_to_resident, validate_td_sync)
from tt_pk.qwen36 import qwen36_27b_problem


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    validate_td_sync()
    _check(True, "lint required-keys are a subset of TD parameters")

    expected = {
        "TTNN_ProgramAttr": {"kernels", "cbs", "semaphores"},
        "TTNN_KernelCBAttr": {"total_size", "core_ranges", "formats", "buffer"},
        "TTNN_KernelCBFormatAttr": {"buffer_index", "dtype", "page_size"},
        "TTNN_KernelSemaphoreAttr": {"id", "core_type", "core_ranges",
                                     "initial_value"},
    }
    for symbol, want in expected.items():
        got = extract_td_params(symbol)
        _check(want.issubset(got),
               f"TD {symbol} declares {sorted(want)}")

    problem = qwen36_27b_problem()
    schedule = lower_to_resident(problem, greedy_search(problem))
    text = emit_ttnn_generic(schedule)

    result = lint(text)
    _check("program" in result.attrs_seen, "lint saw ttnn.program attr")
    _check("kernel_cb" in result.attrs_seen, "lint saw ttnn.kernel_cb attr")
    _check("kernel_cb_format" in result.attrs_seen,
           "lint saw ttnn.kernel_cb_format attr")
    _check("kernel_semaphore" in result.attrs_seen,
           "lint saw ttnn.kernel_semaphore attr")
    _check("compute_kernel" in result.attrs_seen,
           "lint saw ttnn.compute_kernel attr")
    _check(result.attrs_missing_keys == (),
           f"no attr instance is missing a required key "
           f"(got {result.attrs_missing_keys})")

    # Negative check: dropping a required key must fail the lint.
    broken = text.replace("total_size =", "garbage =", 1)
    broken_result = lint(broken)
    _check(any(m[1] == "total_size" for m in broken_result.attrs_missing_keys),
           "lint catches a missing total_size key")

    print("OK")


if __name__ == "__main__":
    main()
