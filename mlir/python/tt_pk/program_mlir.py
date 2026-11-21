"""TTNN Program / MeshProgram MLIR emit (artifact A).

Phase 3. Emits a TTNN ProgramDescriptor-shaped MLIR as a record of the
compiler's initial Strategy. Informational — the Agent is free to read
it but not required to follow it.

See mlir/INTERFACE.md § The two compiler artifacts / A.
"""

from typing import Iterable

from .strategy import (CBLayout, RoleAssignment, SemEdge, StateBufferLayout,
                       Strategy, WeightRingLayout)


def _fmt_cores(cores: Iterable[tuple]) -> str:
    return "[" + ", ".join(f"[{x}, {y}]" for x, y in cores) + "]"


def _fmt_role(r: RoleAssignment) -> str:
    return (f'#tt_pk.role<node = "{r.node}", role = "{r.role}", '
            f'cores = {_fmt_cores(r.cores)}>')


def _fmt_cb(cb: CBLayout) -> str:
    return (f'#tt_pk.cb<name = "{cb.name}", buffer_index = {cb.buffer_index}, '
            f'dtype = "{cb.dtype}", page_size = {cb.page_size}, '
            f'tile_count = {cb.tile_count}, cores = {_fmt_cores(cb.cores)}>')


def _fmt_sem(e: SemEdge) -> str:
    return (f'#tt_pk.sem<id = {e.sem_id}, src = "{e.src}", dst = "{e.dst}", '
            f'stream = "{e.stream}">')


# Backpressure sem travels the reverse direction so the receiver can tell
# the producer that a ring slot is free; needs its own declaration so
# tt-metal allocates matching L1 offsets on both ends.
def _fmt_sem_back(e: SemEdge) -> str:
    return (f'#tt_pk.sem<id = {e.sem_back_id}, src = "{e.dst}", '
            f'dst = "{e.src}", stream = "{e.stream}_back">')


def _fmt_ring(w: WeightRingLayout) -> str:
    return (f'#tt_pk.weight_ring<name = "{w.spec.name}", '
            f'stages = {w.spec.stages}, tile_count = {w.spec.tile_count}, '
            f'dtype = "{w.spec.dtype}", cores = {_fmt_cores(w.cores)}>')


def _fmt_state(s: StateBufferLayout) -> str:
    return (f'#tt_pk.state<name = "{s.spec.name}", '
            f'tile_count = {s.spec.tile_count}, dtype = "{s.spec.dtype}", '
            f'cores = {_fmt_cores(s.cores)}>')


def _fmt_list(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return "[]"
    return "[\n      " + ",\n      ".join(items) + "\n    ]"


def _emit_body(name: str, strategy: Strategy, mesh_attr: str = "") -> str:
    attrs = []
    if mesh_attr:
        attrs.append(mesh_attr)
    attrs.append(
        f"tt_pk.mesh = <tp_axis = {strategy.mesh.tp_axis}, "
        f"pp_axis = {strategy.mesh.pp_axis}>")
    attrs.append(
        "tt_pk.roles = " + _fmt_list(_fmt_role(r) for r in strategy.roles))
    attrs.append(
        "tt_pk.cbs = " + _fmt_list(_fmt_cb(c) for c in strategy.cbs))
    sems = [_fmt_sem(s) for s in strategy.sem_graph]
    sems += [_fmt_sem_back(s) for s in strategy.sem_graph]
    attrs.append("tt_pk.sems = " + _fmt_list(sems))
    attrs.append(
        "tt_pk.weight_rings = " + _fmt_list(
            _fmt_ring(w) for w in strategy.weight_rings))
    attrs.append(
        "tt_pk.state_buffers = " + _fmt_list(
            _fmt_state(s) for s in strategy.state_buffers))
    body = ",\n    ".join(attrs)
    return (
        "module {\n"
        f"  func.func @{name}() attributes {{\n"
        f"    {body}\n"
        "  } {\n"
        "    return\n"
        "  }\n"
        "}\n"
    )


def emit_program_mlir(strategy: Strategy, name: str = "program") -> str:
    return _emit_body(name, strategy)


def emit_mesh_program_mlir(
    strategy: Strategy, name: str = "mesh_program"
) -> str:
    mesh_attr = (
        f"tt_pk.mesh_shape = [{strategy.mesh.tp_axis}, "
        f"{strategy.mesh.pp_axis}]")
    return _emit_body(name, strategy, mesh_attr=mesh_attr)
