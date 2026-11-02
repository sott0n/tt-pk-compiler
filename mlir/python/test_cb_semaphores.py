"""CP7a gate: CB sizes and semaphore graph on the Qwen3.6-27B schedule."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import (derive_cbs, derive_semaphores, greedy_search,
                   lower_to_resident)
from tt_pk.qwen36 import qwen36_27b_problem


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    problem = qwen36_27b_problem()
    plan = greedy_search(problem)
    schedule = lower_to_resident(problem, plan)
    ops_by_name = {op.kernel_name: op for op in schedule.ops}

    for op in schedule.ops:
        cbs = derive_cbs(op)
        _check(len(cbs) == len(op.consumes) + 2,
               f"{op.kernel_name}: cbs = inputs ({len(op.consumes)}) "
               f"+ weight + output")
        for i, cb in enumerate(cbs):
            _check(cb.buffer_index == i,
                   f"{op.kernel_name}: cb {i} has monotonic buffer_index")
            _check(cb.total_size() > 0,
                   f"{op.kernel_name}: cb {i} total_size > 0")

    down = ops_by_name["qwen36.mlp_down"]
    down_cbs = derive_cbs(down)
    _check(len(down_cbs) == 4,
           "mlp_down has 4 CBs (2 consumes: gate, up + weight + out)")

    sems = derive_semaphores(schedule)
    # In the Qwen3.6-27B layer, qkv_proj -> attention, attention -> o_proj,
    # mlp_gate -> mlp_down, mlp_up -> mlp_down = 4 intra-layer edges.
    _check(len(sems) == 4,
           f"4 producer->consumer edges inside the layer (got {len(sems)})")
    ids = {s.id for s in sems}
    _check(ids == {0, 1, 2, 3},
           "semaphore ids are a dense 0..n-1 range")
    edges = {(s.producer, s.consumer) for s in sems}
    expected_edges = {
        ("qwen36.qkv_proj", "qwen36.attention"),
        ("qwen36.attention", "qwen36.o_proj"),
        ("qwen36.mlp_gate", "qwen36.mlp_down"),
        ("qwen36.mlp_up", "qwen36.mlp_down"),
    }
    _check(edges == expected_edges,
           "intra-layer dep edges match the kernel graph")

    print("OK")


if __name__ == "__main__":
    main()
