"""CandidatePlan: one placement choice over an AnalysisProblem.

Stage 2 of the 3-stage IR. Frozen once produced; the search stage emits it
and the materialize stage reads it. No kernel appears twice; no core is
listed under two kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CoreAssignment:
    kernel_name: str
    cores: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class CandidatePlan:
    assignments: Tuple[CoreAssignment, ...]
