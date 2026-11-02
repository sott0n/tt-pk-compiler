"""Dummy search: row-major greedy allocator over an AnalysisProblem.

CP3 placeholder. Weights every kernel by (k * n) and assigns contiguous
cores proportional to the sum of weights. Later checkpoints replace this
with a real cost-model-driven search.
"""

from __future__ import annotations

from typing import List, Tuple

from .plan import CandidatePlan, CoreAssignment
from .problem import AnalysisProblem, KernelSpec


def _weights(kernels: Tuple[KernelSpec, ...]) -> List[int]:
    return [max(1, k.k * k.n) for k in kernels]


def _share_cores(total: int, weights: List[int]) -> List[int]:
    weight_sum = sum(weights)
    raw = [max(1, (w * total) // weight_sum) for w in weights]
    # Row-major greedy over-allocates when (w*total) rounds up for every
    # kernel. Shed cores from the heaviest bucket until the sum fits.
    while sum(raw) > total:
        i = raw.index(max(raw))
        raw[i] -= 1
    return raw


def greedy_search(problem: AnalysisProblem) -> CandidatePlan:
    hw = problem.hardware
    reserved = set(hw.reserved_cores)
    cursor: List[Tuple[int, int]] = []
    for y in range(hw.grid_y):
        for x in range(hw.grid_x):
            if (x, y) in reserved:
                continue
            cursor.append((x, y))

    kernels = problem.layer.kernels
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
