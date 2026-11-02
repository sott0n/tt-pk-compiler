"""Dummy search: row-major greedy allocator over an AnalysisProblem.

CP3 placeholder. Weights every kernel by (k * n) and assigns contiguous
cores proportional to the sum of weights. Later checkpoints replace this
with a real cost-model-driven search.

The emitted CandidatePlan is in topological order: a kernel never appears
before one of its producers. Materialize relies on this for ordering.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .plan import CandidatePlan, CoreAssignment
from .problem import AnalysisProblem, KernelSpec


def _weights(kernels: Tuple[KernelSpec, ...]) -> List[int]:
    return [k.effective_cost() for k in kernels]


def _share_cores(total: int, weights: List[int]) -> List[int]:
    weight_sum = sum(weights)
    raw = [max(1, (w * total) // weight_sum) for w in weights]
    # Row-major greedy over-allocates when (w*total) rounds up for every
    # kernel. Shed cores from the heaviest bucket until the sum fits.
    while sum(raw) > total:
        i = raw.index(max(raw))
        raw[i] -= 1
    return raw


def _topo_order(kernels: Tuple[KernelSpec, ...]) -> List[KernelSpec]:
    producers: Dict[str, str] = {k.produces: k.name for k in kernels if k.produces}
    pending: Dict[str, KernelSpec] = {k.name: k for k in kernels}
    deps: Dict[str, Set[str]] = {
        k.name: {producers[c] for c in k.consumes if c in producers}
        for k in kernels
    }
    ordered: List[KernelSpec] = []
    emitted: Set[str] = set()
    while pending:
        ready = [name for name, k in pending.items()
                 if deps[name].issubset(emitted)]
        if not ready:
            raise RuntimeError(
                f"dependency cycle or missing producer among: "
                f"{sorted(pending.keys())}")
        # Preserve the input order among same-level kernels so the plan
        # stays stable across runs.
        ready_sorted = sorted(ready,
                              key=lambda n: list(pending.keys()).index(n))
        for name in ready_sorted:
            ordered.append(pending.pop(name))
            emitted.add(name)
    return ordered


def greedy_search(problem: AnalysisProblem) -> CandidatePlan:
    hw = problem.hardware
    reserved = set(hw.reserved_cores)
    cursor: List[Tuple[int, int]] = []
    for y in range(hw.grid_y):
        for x in range(hw.grid_x):
            if (x, y) in reserved:
                continue
            cursor.append((x, y))

    kernels = tuple(_topo_order(problem.layer.kernels))
    counts = _share_cores(len(cursor), _weights(kernels))

    assignments: List[CoreAssignment] = []
    offset = 0
    for kernel, count in zip(kernels, counts):
        chunk = tuple(cursor[offset:offset + count])
        if len(chunk) < count:
            raise RuntimeError(
                f"greedy_search ran out of cores for {kernel.name}: "
                f"asked {count}, got {len(chunk)}")
        assignments.append(CoreAssignment(kernel_name=kernel.name, cores=chunk))
        offset += count
    return CandidatePlan(assignments=tuple(assignments))
