"""Resident decode-step shape check.

Lowers Qwen3.6-27B's one decoder layer end-to-end and asserts the schedule
has the correct kernel set, respects data dependencies, and places kernels
on disjoint cores. CP4 stops at structural correctness; CP5 wires the
schedule to real hardware and the baseline comparison.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import (assemble, greedy_search, lower_to_resident, materialize,
                   verify)
from tt_pk.qwen36 import qwen36_27b_problem


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


EXPECTED_KERNELS = {
    "qwen36.qkv_proj",
    "qwen36.attention",
    "qwen36.o_proj",
    "qwen36.mlp_gate",
    "qwen36.mlp_up",
    "qwen36.mlp_down",
}


def main() -> None:
    problem = qwen36_27b_problem()
    plan = greedy_search(problem)
    schedule = lower_to_resident(problem, plan)

    _check(schedule.model_name == "qwen36_27b", "schedule tags model")
    _check(schedule.hardware == "bh_p150", "schedule tags hardware")
    _check(len(schedule.ops) == 6, "schedule has 6 ops")
    _check({op.kernel_name for op in schedule.ops} == EXPECTED_KERNELS,
           "schedule carries the expected kernel set")

    produced = set()
    for op in schedule.ops:
        for c in op.consumes:
            # x_norm and x_norm2 are layer-level activations, produced by
            # norms outside the decoder-layer kernel set.
            if c in ("x_norm", "x_norm2"):
                continue
            _check(c in produced,
                   f"{op.kernel_name} consumes {c!r} already produced")
        if op.produces:
            produced.add(op.produces)
    _check({"qkv", "attn", "attn_proj", "gate", "up", "mlp_out"}.issubset(
                produced),
           "schedule produces every expected activation")

    seen = set()
    for op in schedule.ops:
        for c in op.cores:
            _check(c not in seen,
                   f"core {c} not shared across ops")
            seen.add(c)
    _check(len(seen) > 0, "schedule places on at least one core")

    prog = materialize(plan)
    text = assemble(prog)
    for name in EXPECTED_KERNELS:
        _check(name in text, f"materialize mentions {name}")
    result = verify(prog)
    if not result.ok:
        print("--- tt-pk-opt stderr ---")
        print(result.stderr)
    _check(result.ok, "lowered program passes CoreSetDisjoint")

    print("OK")


if __name__ == "__main__":
    main()
