"""Idiomatic C++ skeleton generator (artifact B).

Hand-written Python templates for persistent kernel skeletons.

- streamer: external DRAM → CB reader (full NOC body, no compute gap).
- tail: CB → external DRAM writer (full NOC body, no compute gap).
- worker / head / hub / leader: step loop + sem/CB sync + a commented
  compute gap owned by the Agent.

Not EmitC. See mlir/INTERFACE.md § The two compiler artifacts / B.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .strategy import SemEdge, Strategy


@dataclass(frozen=True)
class CppFile:
    role: str
    node: str
    path: str
    source: str


_COMPUTE_ROLES = ("worker", "head")
_DATAFLOW_STREAMER = "streamer"
_DATAFLOW_TAIL = "tail"


def _edges_by_node(
    sems: Tuple[SemEdge, ...]
) -> Tuple[Dict[str, List[SemEdge]], Dict[str, List[SemEdge]]]:
    in_edges: Dict[str, List[SemEdge]] = {}
    out_edges: Dict[str, List[SemEdge]] = {}
    for e in sems:
        in_edges.setdefault(e.dst, []).append(e)
        out_edges.setdefault(e.src, []).append(e)
    return in_edges, out_edges


def _compute_include(role: str) -> str:
    if role in _COMPUTE_ROLES:
        return ('#include "api/compute/compute_kernel_hw_startup.h"\n'
                '#include "api/compute/eltwise_unary/eltwise_unary.h"\n')
    return '#include "api/dataflow/dataflow_api.h"\n'


def _header(node: str, role: str) -> str:
    return f"// role: {role}  node: {node}\n"


def _render_compute(
    node: str, role: str, in_edges: List[SemEdge],
    out_edges: List[SemEdge],
) -> str:
    lines = [_header(node, role), _compute_include(role), "",
             "void kernel_main() {",
             "    constexpr uint32_t iters = get_compile_time_arg_val(0);"]
    slot = 1
    for e in in_edges:
        lines.append(
            f"    constexpr uint32_t cb_{e.stream} = "
            f"get_compile_time_arg_val({slot});  // in")
        slot += 1
        lines.append(
            f"    constexpr uint32_t sem_{e.stream}_in = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
    for e in out_edges:
        lines.append(
            f"    constexpr uint32_t cb_{e.stream} = "
            f"get_compile_time_arg_val({slot});  // out")
        slot += 1
        lines.append(
            f"    constexpr uint32_t sem_{e.stream}_out = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in in_edges:
        lines.append(
            f"        noc_semaphore_wait_min(sem_{e.stream}_in, i + 1);")
        lines.append(f"        cb_wait_front(cb_{e.stream}, 1);")
    lines.append("")
    lines.append("        // ===== Agent compute here =====")
    lines.append("        // ===== end =====")
    lines.append("")
    for e in in_edges:
        lines.append(f"        cb_pop_front(cb_{e.stream}, 1);")
    for e in out_edges:
        lines.append(f"        cb_reserve_back(cb_{e.stream}, 1);")
        lines.append(f"        cb_push_back(cb_{e.stream}, 1);")
        lines.append(f"        noc_semaphore_inc(sem_{e.stream}_out, 1);")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _render_streamer(node: str, out_edges: List[SemEdge]) -> str:
    lines = [_header(node, _DATAFLOW_STREAMER),
             '#include "api/dataflow/dataflow_api.h"\n', "",
             "void kernel_main() {",
             "    constexpr uint32_t iters = get_compile_time_arg_val(0);"]
    slot = 1
    for e in out_edges:
        lines.append(
            f"    constexpr uint32_t cb_{e.stream} = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
        lines.append(
            f"    constexpr uint32_t sem_{e.stream}_out = "
            f"get_compile_time_arg_val({slot});")
        slot += 1
    for idx, e in enumerate(out_edges):
        lines.append(
            f"    const uint32_t {e.stream}_src_addr = "
            f"get_common_arg_val<uint32_t>({idx});")
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in out_edges:
        tile = f"get_tile_size(cb_{e.stream})"
        lines.append(f"        cb_reserve_back(cb_{e.stream}, 1);")
        lines.append(
            f"        uint64_t {e.stream}_noc = get_noc_addr(")
        lines.append(
            "            my_x[noc_index], my_y[noc_index],")
        lines.append(
            f"            {e.stream}_src_addr + i * {tile});")
        lines.append(
            f"        noc_async_read({e.stream}_noc, "
            f"get_write_ptr(cb_{e.stream}), {tile});")
    lines.append("        noc_async_read_barrier();")
    for e in out_edges:
        lines.append(f"        cb_push_back(cb_{e.stream}, 1);")
        lines.append(f"        noc_semaphore_inc(sem_{e.stream}_out, 1);")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _render_tail(node: str, in_edges: List[SemEdge]) -> str:
    lines = [_header(node, _DATAFLOW_TAIL),
             '#include "api/dataflow/dataflow_api.h"\n', "",
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
    for idx, e in enumerate(in_edges):
        lines.append(
            f"    const uint32_t {e.stream}_dst_addr = "
            f"get_common_arg_val<uint32_t>({idx});")
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    for e in in_edges:
        lines.append(
            f"        noc_semaphore_wait_min(sem_{e.stream}_in, i + 1);")
        lines.append(f"        cb_wait_front(cb_{e.stream}, 1);")
    for e in in_edges:
        tile = f"get_tile_size(cb_{e.stream})"
        lines.append(
            f"        uint64_t {e.stream}_noc = get_noc_addr(")
        lines.append(
            "            my_x[noc_index], my_y[noc_index],")
        lines.append(
            f"            {e.stream}_dst_addr + i * {tile});")
        lines.append(
            f"        noc_async_write(get_read_ptr(cb_{e.stream}), "
            f"{e.stream}_noc, {tile});")
    lines.append("        noc_async_write_barrier();")
    for e in in_edges:
        lines.append(f"        cb_pop_front(cb_{e.stream}, 1);")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _render(
    node: str, role: str, in_edges: List[SemEdge],
    out_edges: List[SemEdge],
) -> str:
    if role == _DATAFLOW_STREAMER:
        return _render_streamer(node, out_edges)
    if role == _DATAFLOW_TAIL:
        return _render_tail(node, in_edges)
    return _render_compute(node, role, in_edges, out_edges)


def emit_cpp_skeleton(strategy: Strategy) -> Tuple[CppFile, ...]:
    in_edges, out_edges = _edges_by_node(strategy.sem_graph)
    out: List[CppFile] = []
    for r in strategy.roles:
        src = _render(r.node, r.role,
                      in_edges.get(r.node, []),
                      out_edges.get(r.node, []))
        out.append(CppFile(role=r.role, node=r.node,
                           path=f"{r.node}.cpp", source=src))
    return tuple(out)
