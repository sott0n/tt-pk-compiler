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


# BRISC on a streamer: read each out-stream tile from DRAM interleaved,
# push to the consumer core's CB slot (remote L1) and bump the consumer
# sem. DRAM bank addressing via get_noc_addr_from_bank_id<true>.
def _render_streamer_brisc(node: str, out_edges: List[SemEdge]) -> str:
    lines = [_header(node, "streamer", "BRISC"), _INCL_DATAFLOW, "",
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
        for name in (f"{e.stream}_src_addr", f"{e.stream}_src_bank",
                     f"{e.stream}_dest_x", f"{e.stream}_dest_y",
                     f"{e.stream}_dest_cb_addr",
                     f"{e.stream}_dest_sem_addr"):
            lines.append(
                f"    const uint32_t {name} = "
                f"get_common_arg_val<uint32_t>({rt});")
            rt += 1
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in out_edges:
        tile = f"get_tile_size(cb_{e.stream})"
        lines.append(f"        cb_reserve_back(cb_{e.stream}, 1);")
        lines.append(
            f"        uint64_t {e.stream}_dram = "
            f"get_noc_addr_from_bank_id<true>(")
        lines.append(
            f"            {e.stream}_src_bank, "
            f"{e.stream}_src_addr + i * {tile});")
        lines.append(
            f"        noc_async_read({e.stream}_dram, "
            f"get_write_ptr(cb_{e.stream}), {tile});")
        lines.append("        noc_async_read_barrier();")
        lines.append(
            f"        uint64_t {e.stream}_dest = get_noc_addr(")
        lines.append(
            f"            {e.stream}_dest_x, {e.stream}_dest_y,")
        lines.append(
            f"            {e.stream}_dest_cb_addr + i * {tile});")
        lines.append(
            f"        noc_async_write(get_read_ptr(cb_{e.stream}), "
            f"{e.stream}_dest, {tile});")
        lines.append("        noc_async_write_barrier();")
        lines.append(
            f"        uint64_t {e.stream}_sem = get_noc_addr(")
        lines.append(
            f"            {e.stream}_dest_x, {e.stream}_dest_y, "
            f"{e.stream}_dest_sem_addr);")
        lines.append(
            f"        noc_semaphore_inc({e.stream}_sem, 1);")
        lines.append(f"        cb_pop_front(cb_{e.stream}, 1);")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# BRISC on a compute node: upstream data lands in local CB L1 (producer
# wrote it); BRISC waits the arrival sem and calls cb_push_back locally
# so the local COMPUTE's cb_wait_front can see it.
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
            f"    constexpr uint32_t sem_{e.stream}_in = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in in_edges:
        lines.append(
            f"        noc_semaphore_wait_min(sem_{e.stream}_in, i + 1);")
        lines.append(f"        cb_reserve_back(cb_{e.stream}, 1);")
        lines.append(f"        cb_push_back(cb_{e.stream}, 1);")
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
        lines.append(f"        cb_wait_front(cb_{e.stream}, 1);")
    for e in out_edges:
        lines.append(f"        cb_reserve_back(cb_{e.stream}, 1);")
    lines.append("")
    lines.append("        // ===== Agent compute here =====")
    lines.append("        // ===== end =====")
    lines.append("")
    for e in in_edges:
        lines.append(f"        cb_pop_front(cb_{e.stream}, 1);")
    for e in out_edges:
        lines.append(f"        cb_push_back(cb_{e.stream}, 1);")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# NCRISC on a compute node: COMPUTE pushed into local out-CB; forward it
# to each consumer core (push model), then bump the consumer sem.
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
                     f"{e.stream}_dest_sem_addr"):
            lines.append(
                f"    const uint32_t {name} = "
                f"get_common_arg_val<uint32_t>({rt});")
            rt += 1
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in out_edges:
        tile = f"get_tile_size(cb_{e.stream})"
        lines.append(f"        cb_wait_front(cb_{e.stream}, 1);")
        lines.append(
            f"        uint64_t {e.stream}_dest = get_noc_addr(")
        lines.append(
            f"            {e.stream}_dest_x, {e.stream}_dest_y,")
        lines.append(
            f"            {e.stream}_dest_cb_addr + i * {tile});")
        lines.append(
            f"        noc_async_write(get_read_ptr(cb_{e.stream}), "
            f"{e.stream}_dest, {tile});")
        lines.append("        noc_async_write_barrier();")
        lines.append(
            f"        uint64_t {e.stream}_sem = get_noc_addr(")
        lines.append(
            f"            {e.stream}_dest_x, {e.stream}_dest_y, "
            f"{e.stream}_dest_sem_addr);")
        lines.append(
            f"        noc_semaphore_inc({e.stream}_sem, 1);")
        lines.append(f"        cb_pop_front(cb_{e.stream}, 1);")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# NCRISC on a tail: wait each in-edge sem, read from local CB, write to
# DRAM interleaved bank.
def _render_tail_ncrisc(node: str, in_edges: List[SemEdge]) -> str:
    lines = [_header(node, "tail", "NCRISC"), _INCL_DATAFLOW, "",
             "void kernel_main() {",
             "    constexpr uint32_t iters = get_compile_time_arg_val(0);"]
    slot = 1
    for e in in_edges:
        lines.append(
            f"    constexpr uint32_t cb_{e.stream} = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
        lines.append(
            f"    constexpr uint32_t sem_{e.stream}_in = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
    rt = 0
    for e in in_edges:
        for name in (f"{e.stream}_dst_addr", f"{e.stream}_dst_bank"):
            lines.append(
                f"    const uint32_t {name} = "
                f"get_common_arg_val<uint32_t>({rt});")
            rt += 1
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in in_edges:
        tile = f"get_tile_size(cb_{e.stream})"
        lines.append(
            f"        noc_semaphore_wait_min(sem_{e.stream}_in, i + 1);")
        lines.append(f"        cb_wait_front(cb_{e.stream}, 1);")
        lines.append(
            f"        uint64_t {e.stream}_dram = "
            f"get_noc_addr_from_bank_id<false>(")
        lines.append(
            f"            {e.stream}_dst_bank, "
            f"{e.stream}_dst_addr + i * {tile});")
        lines.append(
            f"        noc_async_write(get_read_ptr(cb_{e.stream}), "
            f"{e.stream}_dram, {tile});")
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
