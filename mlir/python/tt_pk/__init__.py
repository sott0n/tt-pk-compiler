"""Python entry point for the `tt_pk` MLIR dialect.

Textual builder plus a 3-stage pipeline (AnalysisProblem -> CandidatePlan
-> tt_pk Program). The driver stays subprocess-based until the dialect C
API lands.
"""

from .builder import CoreSet, Kernel, Program, assemble, verify
from .materialize import materialize
from .plan import CandidatePlan, CoreAssignment
from .problem import (AnalysisProblem, DecoderLayerSpec, HardwareSpec,
                      KernelSpec)
from .resident import ResidentOp, ResidentSchedule, lower_to_resident
from .search import greedy_search

__all__ = [
    "CoreSet", "Kernel", "Program", "assemble", "verify",
    "AnalysisProblem", "DecoderLayerSpec", "HardwareSpec", "KernelSpec",
    "CandidatePlan", "CoreAssignment",
    "greedy_search", "materialize",
    "ResidentOp", "ResidentSchedule", "lower_to_resident",
]
