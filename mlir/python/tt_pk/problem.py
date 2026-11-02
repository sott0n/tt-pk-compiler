"""AnalysisProblem: what the compiler must place.

Stage 1 of the 3-stage IR. Pure data. Independent of hardware mapping and
of any particular placement choice. Downstream stages read it; they never
mutate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class KernelSpec:
    name: str
    k: int
    n: int
    dtype: str = "bf16"


@dataclass(frozen=True)
class DecoderLayerSpec:
    kernels: Tuple[KernelSpec, ...]


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    grid_x: int
    grid_y: int
    reserved_cores: Tuple[Tuple[int, int], ...] = ()

    def total_cores(self) -> int:
        return self.grid_x * self.grid_y - len(self.reserved_cores)


@dataclass(frozen=True)
class AnalysisProblem:
    model_name: str
    layer: DecoderLayerSpec
    hardware: HardwareSpec
