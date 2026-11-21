"""Strategy parameter types.

Phase 0 scaffold. See mlir/INTERFACE.md section "Parametric inputs".
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class MeshParams:
    tp_axis: int = 1
    pp_axis: int = 1


@dataclass(frozen=True)
class RoleCount:
    streamer: int = 0
    worker: int = 0
    tail: int = 0
    leader: int = 0
    hub: int = 0
    head: int = 0


@dataclass(frozen=True)
class WeightRing:
    size_tiles: int = 0
    stages: int = 0


@dataclass(frozen=True)
class StateBuffer:
    kv_block: int = 0
    max_seq: int = 0


@dataclass(frozen=True)
class StrategyParams:
    workers: int = 0
    fanin: int = 0
    roles: RoleCount = field(default_factory=RoleCount)
    mesh: MeshParams = field(default_factory=MeshParams)
    weight_ring: WeightRing = field(default_factory=WeightRing)
    state: StateBuffer = field(default_factory=StateBuffer)
    step_loop: str = "fused"
    cb_sizes: Tuple[Tuple[str, int], ...] = ()
