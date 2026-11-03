"""Lower tt_pk CandidatePlan to a Resident schedule.

A ResidentSchedule names the kernels the Resident runtime will invoke in
order, each with its placement and tensor dependencies. The schedule
also carries role-based placements (`workers_cores`) that the Resident
runtime consumes directly via its RESIDENT_PK_PLAN hook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .plan import CandidatePlan
from .problem import AnalysisProblem, ComputeKernelConfig, KernelSpec


@dataclass(frozen=True)
class ResidentOp:
    kernel_name: str
    cores: Tuple[Tuple[int, int], ...]
    consumes: Tuple[str, ...]
    produces: str
    k: int
    n: int
    dtype: str
    compute_config: ComputeKernelConfig
    compute_kernel_source: Optional[str] = None


@dataclass(frozen=True)
class ResidentSchedule:
    model_name: str
    hardware: str
    ops: Tuple[ResidentOp, ...]
    # Role-based placement fields consumed by the Resident runtime's
    # RESIDENT_PK_PLAN hook. Optional until the compiler plans them on its
    # own; for CP5b they are filled in from a probe of the live runtime.
    workers_cores: Tuple[Tuple[int, int], ...] = ()


def lower_to_resident(problem: AnalysisProblem,
                      plan: CandidatePlan,
                      workers_cores: Optional[Tuple[Tuple[int, int], ...]] = None,
                      ) -> ResidentSchedule:
    by_name: Dict[str, KernelSpec] = {k.name: k for k in problem.layer.kernels}
    hw_policy = problem.hardware.default_compute_config
    ops = []
    for assignment in plan.assignments:
        spec = by_name.get(assignment.kernel_name)
        if spec is None:
            raise KeyError(
                f"plan references kernel {assignment.kernel_name!r} that is "
                f"not in the AnalysisProblem")
        cfg = spec.compute_config or hw_policy
        if cfg is None:
            raise ValueError(
                f"kernel {spec.name!r} has no compute_config and hardware "
                f"{problem.hardware.name!r} has no default policy")
        ops.append(ResidentOp(
            kernel_name=spec.name,
            cores=assignment.cores,
            consumes=spec.consumes,
            produces=spec.produces,
            k=spec.k,
            n=spec.n,
            dtype=spec.dtype,
            compute_config=cfg,
            compute_kernel_source=spec.compute_kernel_source,
        ))
    return ResidentSchedule(
        model_name=problem.model_name,
        hardware=problem.hardware.name,
        ops=tuple(ops),
        workers_cores=tuple(workers_cores) if workers_cores else (),
    )


def to_json(schedule: ResidentSchedule) -> str:
    """Render the schedule as the JSON Resident's RESIDENT_PK_PLAN hook reads.

    Fields the hook does not know about are included for forward compat but
    ignored by the current Resident patch.
    """
    doc = {
        "model_name": schedule.model_name,
        "hardware": schedule.hardware,
        "workers_cores": [list(c) for c in schedule.workers_cores],
        "ops": [
            {
                "kernel_name": op.kernel_name,
                "cores": [list(c) for c in op.cores],
                "consumes": list(op.consumes),
                "produces": op.produces,
                "k": op.k,
                "n": op.n,
                "dtype": op.dtype,
            }
            for op in schedule.ops
        ],
    }
    return json.dumps(doc, indent=2)
