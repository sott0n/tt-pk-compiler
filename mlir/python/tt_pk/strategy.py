"""Parametric strategy calculation.

Phase 2. Takes an AnalysisProblem + Topology + StrategyParams, returns a
Strategy (role-to-core assignment, CB layout, semaphore graph, weight
rings, state buffers, mesh). Also runs the emit-time L1 constraint
check.

No authority — a record of initial judgement. The Agent may deviate when
rewriting the compute kernels.

See mlir/INTERFACE.md § The two compiler artifacts / A and § The four
irreducible values the compiler keeps.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .params import StrategyParams
from .problem import (AnalysisProblem, HardwareSpec, MeshSpec,
                      StateBufferSpec, WeightRingSpec)
from .qwen36_topology import Topology


@dataclass(frozen=True)
class RoleAssignment:
    node: str
    role: str
    cores: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class CBLayout:
    name: str
    buffer_index: int
    dtype: str
    page_size: int
    tile_count: int
    cores: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class SemEdge:
    src: str
    dst: str
    stream: str
    sem_id: int
    # Tiles consumed per iter on the consumer (dst) and produced per iter
    # on the producer (src). Typically equal, but matmul-style
    # contraction makes them differ (qk consumes 4 Q tiles per 1 S tile
    # it produces).
    tile_rate_in: int = 1
    tile_rate_out: int = 1
    # Unique tiles the producer walks before wrapping the DRAM / local
    # address back to zero. 0 means linear addressing; a positive value
    # lets the producer reuse the same cycle_out tiles across iters, as
    # GQA's reader_k reuses one kv-head's eight K tiles.
    tile_cycle_out: int = 0


@dataclass(frozen=True)
class WeightRingLayout:
    spec: WeightRingSpec
    cores: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class StateBufferLayout:
    spec: StateBufferSpec
    cores: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class Strategy:
    roles: Tuple[RoleAssignment, ...] = ()
    cbs: Tuple[CBLayout, ...] = ()
    sem_graph: Tuple[SemEdge, ...] = ()
    weight_rings: Tuple[WeightRingLayout, ...] = ()
    state_buffers: Tuple[StateBufferLayout, ...] = ()
    mesh: MeshSpec = MeshSpec()
    params: StrategyParams = field(default_factory=StrategyParams)


class StrategyError(ValueError):
    pass


def _available_cores(hardware: HardwareSpec) -> List[Tuple[int, int]]:
    reserved = set(hardware.reserved_cores)
    pool: List[Tuple[int, int]] = []
    for x in range(hardware.grid_x):
        for y in range(hardware.grid_y):
            if (x, y) not in reserved:
                pool.append((x, y))
    return pool


def _role_count(params: StrategyParams, role: str) -> int:
    r = params.roles
    return {"streamer": r.streamer, "worker": r.worker, "tail": r.tail,
            "leader": r.leader, "hub": r.hub, "head": r.head}[role]


def _per_die_budget(params: StrategyParams, role: str, tp_axis: int) -> int:
    # Each die carries 1/tp_axis of the layer's role budget.
    total = _role_count(params, role)
    return max(1, total // max(1, tp_axis))


def _assign_roles(
    topology: Topology, hardware: HardwareSpec, params: StrategyParams,
    mesh: MeshSpec,
) -> Tuple[RoleAssignment, ...]:
    pool = _available_cores(hardware)
    cursor = 0
    by_role: Dict[str, List[str]] = {}
    for node in topology.nodes:
        by_role.setdefault(node.role, []).append(node.name)
    out: List[RoleAssignment] = []
    for node in topology.nodes:
        budget = _per_die_budget(params, node.role, mesh.tp_axis)
        share = max(1, budget // len(by_role[node.role]))
        if cursor + share > len(pool):
            raise StrategyError(
                f"pool exhausted assigning {node.name}: need {share}, "
                f"pool left {len(pool) - cursor}")
        cores = tuple(pool[cursor:cursor + share])
        cursor += share
        out.append(RoleAssignment(node.name, node.role, cores))
    return tuple(out)


def _derive_cbs(
    topology: Topology, roles: Tuple[RoleAssignment, ...],
    params: StrategyParams,
) -> Tuple[CBLayout, ...]:
    overrides = {k: v for k, v in params.cb_sizes}
    cores_by_node = {r.node: r.cores for r in roles}
    streams: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    for node in topology.nodes:
        for stream in node.produces:
            streams.setdefault(stream, cores_by_node[node.name])
    out: List[CBLayout] = []
    for idx, (name, cores) in enumerate(streams.items()):
        tile_count = overrides.get(name, 2)
        out.append(CBLayout(
            name=name, buffer_index=idx, dtype="bf16",
            page_size=2048, tile_count=tile_count, cores=cores))
    return tuple(out)


def _tile_rate(node, stream: str) -> int:
    for s, r in node.tile_rates:
        if s == stream:
            return r
    return 1


def _tile_cycle(node, stream: str) -> int:
    for s, c in node.output_cycle:
        if s == stream:
            return c
    return 0


def _derive_sem_graph(topology: Topology) -> Tuple[SemEdge, ...]:
    producers: Dict[str, str] = {}
    nodes_by_name = {n.name: n for n in topology.nodes}
    for node in topology.nodes:
        for stream in node.produces:
            producers[stream] = node.name
    out: List[SemEdge] = []
    sem_id = 0
    for node in topology.nodes:
        for stream in node.consumes:
            src_name = producers.get(stream)
            if src_name is None:
                continue
            src_node = nodes_by_name[src_name]
            out.append(SemEdge(
                src=src_name, dst=node.name, stream=stream,
                sem_id=sem_id,
                tile_rate_in=_tile_rate(node, stream),
                tile_rate_out=_tile_rate(src_node, stream),
                tile_cycle_out=_tile_cycle(src_node, stream)))
            sem_id += 1
    return tuple(out)


def _place_weight_rings(
    problem: AnalysisProblem, roles: Tuple[RoleAssignment, ...],
) -> Tuple[WeightRingLayout, ...]:
    # Weight rings sit on worker cores; they stream weights into the
    # per-worker compute step.
    worker_cores: Tuple[Tuple[int, int], ...] = ()
    for r in roles:
        if r.role == "worker":
            worker_cores = worker_cores + r.cores
    return tuple(WeightRingLayout(spec=w, cores=worker_cores)
                 for w in problem.weight_rings)


def _place_state_buffers(
    problem: AnalysisProblem, roles: Tuple[RoleAssignment, ...],
) -> Tuple[StateBufferLayout, ...]:
    # State buffers live where the attention workers do their KV reads.
    worker_cores: Tuple[Tuple[int, int], ...] = ()
    for r in roles:
        if r.role == "worker":
            worker_cores = worker_cores + r.cores
    return tuple(StateBufferLayout(spec=s, cores=worker_cores)
                 for s in problem.state_buffers)


# bf16 tile is 32x32 half-precision + metadata; TTNN reports 2112 bytes.
_BF16_TILE_BYTES = 2112


def _check_l1_budget(
    hardware: HardwareSpec, cbs: Tuple[CBLayout, ...],
    rings: Tuple[WeightRingLayout, ...],
    states: Tuple[StateBufferLayout, ...],
) -> None:
    per_core: Dict[Tuple[int, int], int] = {}
    for cb in cbs:
        bytes_each = cb.tile_count * _BF16_TILE_BYTES
        for c in cb.cores:
            per_core[c] = per_core.get(c, 0) + bytes_each
    for ring in rings:
        bytes_each = ring.spec.tile_count * _BF16_TILE_BYTES
        for c in ring.cores:
            per_core[c] = per_core.get(c, 0) + bytes_each
    for state in states:
        bytes_each = state.spec.tile_count * _BF16_TILE_BYTES
        for c in state.cores:
            per_core[c] = per_core.get(c, 0) + bytes_each
    budget = hardware.l1_bytes_per_core
    for core, used in per_core.items():
        if used > budget:
            raise StrategyError(
                f"L1 over budget at {core}: {used} > {budget} bytes")


def compute_strategy(
    problem: AnalysisProblem, topology: Topology, params: StrategyParams,
) -> Strategy:
    roles = _assign_roles(topology, problem.hardware, params, problem.mesh)
    cbs = _derive_cbs(topology, roles, params)
    sem_graph = _derive_sem_graph(topology)
    rings = _place_weight_rings(problem, roles)
    states = _place_state_buffers(problem, roles)
    _check_l1_budget(problem.hardware, cbs, rings, states)
    return Strategy(roles=roles, cbs=cbs, sem_graph=sem_graph,
                    weight_rings=rings, state_buffers=states,
                    mesh=problem.mesh, params=params)
