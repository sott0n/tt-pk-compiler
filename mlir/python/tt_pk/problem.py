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
    produces: str = ""
    consumes: Tuple[str, ...] = ()
    dtype: str = "bf16"
    # Explicit per-token cost for the search's weighted share. When 0, the
    # search falls back to k*n, which is fine for projections but blind to
    # the num_heads / context-length axes that make attention expensive
    # at decode. Set this to override.
    cost: int = 0

    def effective_cost(self) -> int:
        return self.cost if self.cost > 0 else max(1, self.k * self.n)


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
