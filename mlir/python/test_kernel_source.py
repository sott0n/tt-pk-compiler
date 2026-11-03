"""CP8 gate: compute-kernel source import + source_compute_kernel emission.

Builds the Qwen3.6-27B problem, attaches the testdata kernel body to the
`qkv_proj` KernelSpec, emits the TTNN snippet, and round-trips it
through the in-tree `ttmlir-opt`. The op with a body must emit a
`#ttnn.source_compute_kernel`; the ops without one must stay on
`#ttnn.compute_kernel` with a private function symbol.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import (ensure_available, emit_ttnn_generic, escape_for_mlir,
                   greedy_search, lint, load_kernel_source, lower_to_resident,
                   TTNNLintError)
from tt_pk.problem import AnalysisProblem, DecoderLayerSpec, KernelSpec
from tt_pk.qwen36 import bh_p150_hardware, qwen36_27b_problem


_FIXTURE = os.path.join(HERE, "testdata", "kernels", "qwen36_qkv_proj.cpp")


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def _attach_source(problem: AnalysisProblem, name: str, body: str
                   ) -> AnalysisProblem:
    kernels = []
    for k in problem.layer.kernels:
        if k.name == name:
            kernels.append(KernelSpec(
                name=k.name, k=k.k, n=k.n, produces=k.produces,
                consumes=k.consumes, dtype=k.dtype, cost=k.cost,
                compute_config=k.compute_config,
                compute_kernel_source=body))
        else:
            kernels.append(k)
    return AnalysisProblem(
        model_name=problem.model_name,
        layer=DecoderLayerSpec(kernels=tuple(kernels)),
        hardware=problem.hardware)


def main() -> None:
    body = load_kernel_source(_FIXTURE)
    _check(body.startswith("// Placeholder"),
           "fixture loaded from testdata/kernels/qwen36_qkv_proj.cpp")
    _check(escape_for_mlir('say "hi"\n') == 'say \\"hi\\"\\n',
           "escape_for_mlir handles quotes and newline")

    problem = _attach_source(qwen36_27b_problem(), "qwen36.qkv_proj", body)
    schedule = lower_to_resident(problem, greedy_search(problem))
    text = emit_ttnn_generic(schedule)

    _check("#ttnn.source_compute_kernel<" in text,
           "qkv_proj kernel emitted as source_compute_kernel")
    _check("private @qwen36_qkv_proj_kernel" not in text,
           "no private declaration for source-backed kernel")
    _check("private @qwen36_o_proj_kernel()" in text,
           "ops without source keep private symbol declarations")
    _check("#ttnn.compute_kernel<symbol_ref = @qwen36_o_proj_kernel" in text,
           "ops without source still emit symbol_ref compute_kernel")

    try:
        ensure_available()
    except TTNNLintError as e:
        print(f"[SKIP] ttmlir-opt not built: {e}")
        return

    result = lint(text)
    _check("source_compute_kernel" in result.attrs_seen,
           "ttmlir-opt saw source_compute_kernel")
    _check("compute_kernel" in result.attrs_seen,
           "ttmlir-opt saw compute_kernel on no-source ops")
    _check(result.attrs_rejected == (),
           f"ttmlir-opt accepted every emitted attr "
           f"(rejected: {result.attrs_rejected})")

    print("OK")


if __name__ == "__main__":
    main()
