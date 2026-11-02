"""Codegen-side half of the tt-mlir interface contract (M2).

Emits a textual TTNN snippet with one `ttnn.generic` op per ResidentOp.
CB and semaphore attributes are left empty on purpose; CP7+ fills them.
The emitted text is a format pledge, not a round-trippable module.
"""

from __future__ import annotations

from .resident import ResidentSchedule


def _kernel_sym(kernel_name: str) -> str:
    safe = kernel_name.replace(".", "_")
    return f"@{safe}_kernel"


def emit_ttnn_generic(schedule: ResidentSchedule) -> str:
    """Render a ResidentSchedule as module-scope TTNN text.

    The result names a kernel function per op and a `decoder_layer` that
    calls them through `ttnn.generic`. SSA ids reuse the schedule's
    `produces` / `consumes` labels so the data dependencies read directly.
    """
    lines = ["module {"]
    for op in schedule.ops:
        lines.append(f"  func.func private {_kernel_sym(op.kernel_name)}()"
                     " { /* opaque */ }")
    lines.append("")
    lines.append("  func.func @decoder_layer() {")
    for op in schedule.ops:
        cores = ", ".join(f"[{x}, {y}]" for x, y in op.cores)
        operands = ", ".join(f"%{c}" for c in op.consumes) or "/* none */"
        lines.append(
            f"    %{op.produces} = ttnn.generic {{"
            f" program = #ttnn.generic_program<"
            f"kernels = [#ttnn.kernel_decl<{_kernel_sym(op.kernel_name)}, "
            f"cores = [{cores}]>], cbs = [], semaphores = []>"
            f" }} : ({operands}) -> tensor<_>"
        )
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
