"""Lower tt_pk CandidatePlan to a Resident schedule.

A ResidentSchedule names the kernels the Resident runtime will invoke in
order, each with its placement and tensor dependencies. CP4 keeps the
format host-only; the runtime wiring lands in CP5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .plan import CandidatePlan
from .problem import AnalysisProblem, KernelSpec


@dataclass(frozen=True)
class ResidentOp:
    kernel_name: str
    cores: Tuple[Tuple[int, int], ...]
    consumes: Tuple[str, ...]
    produces: str
    k: int
    n: int
    dtype: str


@dataclass(frozen=True)
class ResidentSchedule:
    model_name: str
    hardware: str
    ops: Tuple[ResidentOp, ...]


def lower_to_resident(problem: AnalysisProblem,
                      plan: CandidatePlan) -> ResidentSchedule:
    by_name: Dict[str, KernelSpec] = {k.name: k for k in problem.layer.kernels}
    ops = []
    for assignment in plan.assignments:
        spec = by_name.get(assignment.kernel_name)
        if spec is None:
            raise KeyError(
                f"plan references kernel {assignment.kernel_name!r} that is "
                f"not in the AnalysisProblem")
        ops.append(ResidentOp(
            kernel_name=spec.name,
            cores=assignment.cores,
            consumes=spec.consumes,
            produces=spec.produces,
            k=spec.k,
            n=spec.n,
            dtype=spec.dtype,
        ))
    return ResidentSchedule(
        model_name=problem.model_name,
        hardware=problem.hardware.name,
        ops=tuple(ops),
    )
