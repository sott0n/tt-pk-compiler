"""Idiomatic C++ skeleton generator (artifact B).

One topology node maps to 1-3 Tensix kernel slots (BRISC / NCRISC /
COMPUTE). COMPUTE cannot touch the NOC, so data movement always lands
on BRISC/NCRISC; compute math always lands on COMPUTE. Push model for
cross-core forwarding (producer does `noc_async_write` to the consumer
CB slot + remote `noc_semaphore_inc`). External I/O goes through DRAM
interleaved banks (`get_noc_addr_from_bank_id`).

Role → kernel slots:
  streamer: BRISC (DRAM read + push to consumer).
  tail:     NCRISC (recv from upstream + DRAM write).
  worker / head / hub: BRISC (recv upstream, push local CB for COMPUTE)
    + COMPUTE (compute gap) + NCRISC (fwd to consumer).
  leader:   BRISC (coordinator).

See mlir/INTERFACE.md § The two compiler artifacts / B.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .strategy import SemEdge, Strategy


@dataclass(frozen=True)
class CppFile:
    role: str
    node: str
    slot: str
    path: str
    source: str


_ROLE_SLOTS: Dict[str, Tuple[str, ...]] = {
    "streamer": ("BRISC",),
    "worker": ("BRISC", "COMPUTE", "NCRISC"),
    "head": ("BRISC", "COMPUTE", "NCRISC"),
    "hub": ("BRISC", "COMPUTE", "NCRISC"),
    "leader": ("BRISC",),
    "tail": ("NCRISC",),
}

_INCL_DATAFLOW = '#include "api/dataflow/dataflow_api.h"\n'
_INCL_TENSOR = (
    '#include "api/dataflow/dataflow_api.h"\n'
    '#include "api/tensor/tensor_accessor_args.h"\n'
    '#include "api/tensor/tensor_accessor.h"\n')
_INCL_COMPUTE = (
    '#include "api/compute/compute_kernel_hw_startup.h"\n'
    '#include "api/compute/eltwise_unary/eltwise_unary.h"\n')


def _edges_by_node(
    sems: Tuple[SemEdge, ...]
) -> Tuple[Dict[str, List[SemEdge]], Dict[str, List[SemEdge]]]:
    in_edges: Dict[str, List[SemEdge]] = {}
    out_edges: Dict[str, List[SemEdge]] = {}
    for e in sems:
        in_edges.setdefault(e.dst, []).append(e)
        out_edges.setdefault(e.src, []).append(e)
    return in_edges, out_edges


def _header(node: str, role: str, slot: str) -> str:
    return f"// role: {role}  node: {node}  slot: {slot}\n"



# BRISC on a streamer: read each out-stream tile from a DRAM-interleaved
# tensor through a TensorAccessor (tt-metal handles the per-tile bank
# calculation), then push the tile to the consumer core's CB slot and
# bump the consumer sem. Single out-stream only; multi-stream readers
# will need per-accessor CT indices.
def _render_streamer_brisc(node: str, out_edges: List[SemEdge]) -> str:
    if len(out_edges) != 1:
        raise ValueError(
            f"streamer {node!r}: single out-edge only (got "
            f"{len(out_edges)})")
    e = out_edges[0]
    lines = [_header(node, "streamer", "BRISC"), _INCL_TENSOR, "",
             "void kernel_main() {",
             "    constexpr uint32_t iters = get_compile_time_arg_val(0);"]
    lines.append(
        f"    constexpr uint32_t cb_{e.stream} = "
        f"get_compile_time_arg_val(1);")
    lines.append(
        f"    constexpr auto {e.stream}_src_args = "
        f"TensorAccessorArgs<2>();")
    for name_idx, name in enumerate((
            f"{e.stream}_src_addr", f"{e.stream}_dest_x",
            f"{e.stream}_dest_y", f"{e.stream}_dest_cb_addr",
            f"{e.stream}_dest_sem_id",
            f"{e.stream}_src_tile_start")):
        lines.append(
            f"    const uint32_t {name} = "
            f"get_arg_val<uint32_t>({name_idx});")
    lines.append(
        f"    uint32_t {e.stream}_dest_sem_addr = "
        f"static_cast<uint32_t>("
        f"get_semaphore({e.stream}_dest_sem_id));")
    lines.append(
        f"    const auto {e.stream}_src_accessor = TensorAccessor(")
    lines.append(
        f"        {e.stream}_src_args, {e.stream}_src_addr, "
        f"get_tile_size(cb_{e.stream}));")
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    tile = f"get_tile_size(cb_{e.stream})"
    n = e.tile_rate_out
    c = e.tile_cycle_out
    lines.append(
        f"        for (uint32_t t = 0; t < {n}; t++) {{")
    lines.append(
        f"            uint32_t tile_idx = i * {n} + t;")
    if c > 0:
        lines.append(
            f"            uint32_t src_tile_idx = "
            f"{e.stream}_src_tile_start + (tile_idx % {c});")
    else:
        lines.append(
            f"            uint32_t src_tile_idx = "
            f"{e.stream}_src_tile_start + tile_idx;")
    lines.append(
        f"            cb_reserve_back(cb_{e.stream}, 1);")
    lines.append(
        f"            noc_async_read_tile(src_tile_idx, "
        f"{e.stream}_src_accessor,")
    lines.append(
        f"                get_write_ptr(cb_{e.stream}));")
    lines.append("            noc_async_read_barrier();")
    # Push before forwarding so get_read_ptr returns the just-read slot;
    # also balances cb_pop_front so pages_received/acked stay in sync.
    lines.append(
        f"            cb_push_back(cb_{e.stream}, 1);")
    lines.append(
        f"            uint64_t {e.stream}_dest = get_noc_addr(")
    lines.append(
        f"                {e.stream}_dest_x, {e.stream}_dest_y,")
    lines.append(
        f"                {e.stream}_dest_cb_addr + "
        f"tile_idx * {tile});")
    lines.append(
        f"            noc_async_write(get_read_ptr(cb_{e.stream}), "
        f"{e.stream}_dest, {tile});")
    lines.append("            noc_async_write_barrier();")
    lines.append(
        f"            uint64_t {e.stream}_sem = get_noc_addr(")
    lines.append(
        f"                {e.stream}_dest_x, {e.stream}_dest_y, "
        f"{e.stream}_dest_sem_addr);")
    lines.append(
        f"            noc_semaphore_inc({e.stream}_sem, 1);")
    lines.append(
        f"            cb_pop_front(cb_{e.stream}, 1);")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# BRISC on a compute node: upstream producer bumps the arrival sem once
# per tile it pushes; this BRISC waits for tile_rate_in tiles then bumps
# the local CB so COMPUTE's cb_wait_front sees them all.
def _render_compute_brisc(
    node: str, role: str, in_edges: List[SemEdge]
) -> str:
    lines = [_header(node, role, "BRISC"), _INCL_DATAFLOW, "",
             "void kernel_main() {",
             "    constexpr uint32_t iters = get_compile_time_arg_val(0);"]
    slot = 1
    for e in in_edges:
        lines.append(
            f"    constexpr uint32_t cb_{e.stream} = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
        lines.append(
            f"    constexpr uint32_t sem_{e.stream}_in_id = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
    for e in in_edges:
        lines.append(
            f"    volatile tt_l1_ptr uint32_t* sem_{e.stream}_in = ")
        lines.append(
            f"        reinterpret_cast<volatile tt_l1_ptr uint32_t*>("
            f"get_semaphore(sem_{e.stream}_in_id));")
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in in_edges:
        lines.append(
            f"        noc_semaphore_wait_min(sem_{e.stream}_in, "
            f"(i + 1) * {e.tile_rate_in});")
        lines.append(
            f"        cb_reserve_back(cb_{e.stream}, {e.tile_rate_in});")
        lines.append(
            f"        cb_push_back(cb_{e.stream}, {e.tile_rate_in});")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# COMPUTE: local CB wait, compute gap, local CB push. No NOC.
def _render_compute(
    node: str, role: str, in_edges: List[SemEdge],
    out_edges: List[SemEdge],
) -> str:
    lines = [_header(node, role, "COMPUTE"), _INCL_COMPUTE, "",
             "void kernel_main() {",
             "    constexpr uint32_t iters = get_compile_time_arg_val(0);"]
    slot = 1
    for e in in_edges:
        lines.append(
            f"    constexpr uint32_t cb_{e.stream} = "
            f"get_compile_time_arg_val({slot});  // in")
        slot += 1
    for e in out_edges:
        lines.append(
            f"    constexpr uint32_t cb_{e.stream} = "
            f"get_compile_time_arg_val({slot});  // out")
        slot += 1
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in in_edges:
        lines.append(
            f"        cb_wait_front(cb_{e.stream}, {e.tile_rate_in});")
    for e in out_edges:
        lines.append(
            f"        cb_reserve_back(cb_{e.stream}, {e.tile_rate_out});")
    lines.append("")
    lines.append("        // ===== Agent compute here =====")
    lines.append("        // ===== end =====")
    lines.append("")
    for e in in_edges:
        lines.append(
            f"        cb_pop_front(cb_{e.stream}, {e.tile_rate_in});")
    for e in out_edges:
        lines.append(
            f"        cb_push_back(cb_{e.stream}, {e.tile_rate_out});")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# NCRISC on a compute node: COMPUTE pushed rate_out tiles per iter into
# local out-CB; NCRISC forwards each tile to the consumer core and bumps
# the consumer sem per tile.
def _render_compute_ncrisc(
    node: str, role: str, out_edges: List[SemEdge]
) -> str:
    lines = [_header(node, role, "NCRISC"), _INCL_DATAFLOW, "",
             "void kernel_main() {",
             "    constexpr uint32_t iters = get_compile_time_arg_val(0);"]
    slot = 1
    for e in out_edges:
        lines.append(
            f"    constexpr uint32_t cb_{e.stream} = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
    rt = 0
    for e in out_edges:
        for name in (f"{e.stream}_dest_x", f"{e.stream}_dest_y",
                     f"{e.stream}_dest_cb_addr",
                     f"{e.stream}_dest_sem_id"):
            lines.append(
                f"    const uint32_t {name} = "
                f"get_arg_val<uint32_t>({rt});")
            rt += 1
    for e in out_edges:
        lines.append(
            f"    uint32_t {e.stream}_dest_sem_addr = "
            f"static_cast<uint32_t>("
            f"get_semaphore({e.stream}_dest_sem_id));")
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in out_edges:
        tile = f"get_tile_size(cb_{e.stream})"
        n = e.tile_rate_out
        lines.append(
            f"        cb_wait_front(cb_{e.stream}, {n});")
        lines.append(
            f"        for (uint32_t t = 0; t < {n}; t++) {{")
        lines.append(
            f"            uint32_t tile_idx = i * {n} + t;")
        lines.append(
            f"            uint64_t {e.stream}_dest = get_noc_addr(")
        lines.append(
            f"                {e.stream}_dest_x, {e.stream}_dest_y,")
        lines.append(
            f"                {e.stream}_dest_cb_addr + "
            f"tile_idx * {tile});")
        lines.append(
            f"            noc_async_write(")
        lines.append(
            f"                get_read_ptr(cb_{e.stream}) + t * {tile},")
        lines.append(
            f"                {e.stream}_dest, {tile});")
        lines.append("            noc_async_write_barrier();")
        lines.append(
            f"            uint64_t {e.stream}_sem = get_noc_addr(")
        lines.append(
            f"                {e.stream}_dest_x, {e.stream}_dest_y, "
            f"{e.stream}_dest_sem_addr);")
        lines.append(
            f"            noc_semaphore_inc({e.stream}_sem, 1);")
        lines.append("        }")
        lines.append(f"        cb_pop_front(cb_{e.stream}, {n});")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# NCRISC on a tail: wait the in-edge sem, read from the local CB, and
# write to the DRAM-interleaved output tensor via a TensorAccessor.
# Single in-edge only.
def _render_tail_ncrisc(node: str, in_edges: List[SemEdge]) -> str:
    if len(in_edges) != 1:
        raise ValueError(
            f"tail {node!r}: single in-edge only (got "
            f"{len(in_edges)})")
    e = in_edges[0]
    lines = [_header(node, "tail", "NCRISC"), _INCL_TENSOR, "",
             "void kernel_main() {",
             "    constexpr uint32_t iters = get_compile_time_arg_val(0);"]
    lines.append(
        f"    constexpr uint32_t cb_{e.stream} = "
        f"get_compile_time_arg_val(1);")
    lines.append(
        f"    constexpr uint32_t sem_{e.stream}_in_id = "
        f"get_compile_time_arg_val(2);")
    lines.append(
        f"    constexpr auto {e.stream}_dst_args = "
        f"TensorAccessorArgs<3>();")
    lines.append(
        f"    volatile tt_l1_ptr uint32_t* sem_{e.stream}_in = ")
    lines.append(
        f"        reinterpret_cast<volatile tt_l1_ptr uint32_t*>("
        f"get_semaphore(sem_{e.stream}_in_id));")
    lines.append(
        f"    const uint32_t {e.stream}_dst_addr = "
        f"get_arg_val<uint32_t>(0);")
    lines.append(
        f"    const uint32_t {e.stream}_dst_tile_start = "
        f"get_arg_val<uint32_t>(1);")
    lines.append(
        f"    const auto {e.stream}_dst_accessor = TensorAccessor(")
    lines.append(
        f"        {e.stream}_dst_args, {e.stream}_dst_addr, "
        f"get_tile_size(cb_{e.stream}));")
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    lines.append(
        f"        noc_semaphore_wait_min(sem_{e.stream}_in, i + 1);")
    lines.append(f"        cb_wait_front(cb_{e.stream}, 1);")
    lines.append(
        f"        noc_async_write_tile({e.stream}_dst_tile_start + i,")
    lines.append(
        f"            {e.stream}_dst_accessor, "
        f"get_read_ptr(cb_{e.stream}));")
    lines.append("        noc_async_write_barrier();")
    lines.append(f"        cb_pop_front(cb_{e.stream}, 1);")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# A leader is a single-BRISC coordinator; nothing else touches the NOC.
def _render_leader_brisc(
    node: str, in_edges: List[SemEdge], out_edges: List[SemEdge]
) -> str:
    # For now, a leader looks like a worker's BRISC body — gate the
    # downstream stages by relaying in-edges. Agent refines.
    return _render_compute_brisc(node, "leader", in_edges)


def _render_node(
    node: str, role: str, in_edges: List[SemEdge],
    out_edges: List[SemEdge],
) -> List[CppFile]:
    slots = _ROLE_SLOTS.get(role)
    if slots is None:
        raise ValueError(f"cpp_template: unknown role {role!r}")
    files: List[CppFile] = []
    for s in slots:
        if role == "streamer" and s == "BRISC":
            src = _render_streamer_brisc(node, out_edges)
        elif role == "tail" and s == "NCRISC":
            src = _render_tail_ncrisc(node, in_edges)
        elif role == "leader" and s == "BRISC":
            src = _render_leader_brisc(node, in_edges, out_edges)
        elif s == "BRISC":
            src = _render_compute_brisc(node, role, in_edges)
        elif s == "COMPUTE":
            src = _render_compute(node, role, in_edges, out_edges)
        elif s == "NCRISC":
            src = _render_compute_ncrisc(node, role, out_edges)
        else:
            raise ValueError(f"cpp_template: unhandled slot {s!r}")
        path = f"{node}_{s.lower()}.cpp"
        files.append(CppFile(role=role, node=node, slot=s, path=path,
                             source=src))
    return files


def emit_cpp_skeleton(strategy: Strategy) -> Tuple[CppFile, ...]:
    in_edges, out_edges = _edges_by_node(strategy.sem_graph)
    out: List[CppFile] = []
    for r in strategy.roles:
        out.extend(_render_node(
            r.node, r.role,
            in_edges.get(r.node, []),
            out_edges.get(r.node, [])))
    return tuple(out)
