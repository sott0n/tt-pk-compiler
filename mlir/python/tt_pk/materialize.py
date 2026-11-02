"""Materialize: CandidatePlan -> tt_pk Program (MLIR assembly).

Stage 3 of the 3-stage IR. The emitted Program must pass CoreSetDisjoint.
Any plan that violates disjointness is a search bug, not a materializer
bug, so we trust the plan and let tt-pk-opt surface the error.
"""

from __future__ import annotations

import re

from .builder import Program
from .plan import CandidatePlan


_SSA_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _ssa_name(kernel_name: str) -> str:
    return "cs_" + _SSA_SAFE.sub("_", kernel_name)


def materialize(plan: CandidatePlan) -> Program:
    prog = Program()
    for assignment in plan.assignments:
        cs = prog.add_core_set(_ssa_name(assignment.kernel_name),
                               assignment.cores)
        prog.launch(assignment.kernel_name, cs)
    return prog
