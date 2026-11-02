"""M2 contract gate: importer whitelist + ttnn.generic codegen shape.

Validates the Qwen3.6-27B decoder layer against the TTIR import whitelist
and checks the emitted TTNN snippet carries one `ttnn.generic` per op
with cores from the resident schedule.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import (ImporterError, TTIR_WHITELIST, emit_ttnn_generic,
                   greedy_search, lower_to_resident, qwen36_provenance,
                   validate_provenance)
from tt_pk.qwen36 import qwen36_27b_problem


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    problem = qwen36_27b_problem()
    prov = qwen36_provenance()

    validate_provenance(problem.layer, prov)
    _check(True, "qwen36 kernels all map onto the TTIR whitelist")

    try:
        validate_provenance(problem.layer, {**prov, "qwen36.qkv_proj": ("ttir.unknown",)})
    except ImporterError:
        _check(True, "unknown TTIR op is rejected by the importer")
    else:
        _check(False, "importer silently accepted a non-whitelisted op")

    for op in TTIR_WHITELIST:
        _check(op.startswith("ttir."), f"whitelist entry {op} is a ttir op")

    plan = greedy_search(problem)
    schedule = lower_to_resident(problem, plan)
    text = emit_ttnn_generic(schedule)

    _check(text.count("= ttnn.generic ") == len(schedule.ops),
           "one ttnn.generic emitted per ResidentOp")
    for op in schedule.ops:
        safe = op.kernel_name.replace(".", "_") + "_kernel"
        _check(safe in text, f"codegen names kernel {safe}")
        _check(f"({op.cores[0][0]}, {op.cores[0][1]})" in text,
               f"codegen carries first core of {op.kernel_name}")

    _check("kernel_cb<" in text, "codegen emits ttnn.kernel_cb descriptors")
    _check("kernel_cb_format<" in text,
           "codegen emits ttnn.kernel_cb_format descriptors")
    _check("kernel_semaphore<" in text,
           "codegen emits ttnn.kernel_semaphore descriptors")

    print("OK")


if __name__ == "__main__":
    main()
