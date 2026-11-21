"""Idiomatic C++ skeleton generator (artifact B).

Phase 4. Hand-written Python templates for persistent kernel skeletons:
step loop + semaphore wait/inc + CB reserve/push/wait/pop + a commented
compute gap owned by the Agent. One .cpp per strategy node, named by
role. Not EmitC.

See mlir/INTERFACE.md § The two compiler artifacts / B.
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


def _edges_by_node(
    sems: Tuple[SemEdge, ...]
) -> Tuple[Dict[str, List[SemEdge]], Dict[str, List[SemEdge]]]:
    in_edges: Dict[str, List[SemEdge]] = {}
    out_edges: Dict[str, List[SemEdge]] = {}
    for e in sems:
        in_edges.setdefault(e.dst, []).append(e)
        out_edges.setdefault(e.src, []).append(e)
    return in_edges, out_edges


def _include(role: str) -> str:
    if role in _COMPUTE_ROLES:
        return ('#include "api/compute/compute_kernel_hw_startup.h"\n'
                '#include "api/compute/eltwise_unary/eltwise_unary.h"\n')
    return '#include "api/dataflow/dataflow_api.h"\n'


def _ct_args(
    role: str, in_edges: List[SemEdge], out_edges: List[SemEdge]
) -> List[str]:
    lines = [
        "    constexpr uint32_t iters = get_compile_time_arg_val(0);"
    ]
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
    return lines


def _loop_body(
    in_edges: List[SemEdge], out_edges: List[SemEdge]
) -> List[str]:
    body: List[str] = []
    for e in in_edges:
        body.append(
            f"        noc_semaphore_wait_min(sem_{e.stream}_in, i + 1);")
        body.append(f"        cb_wait_front(cb_{e.stream}, 1);")
    body.append("")
    body.append("        // ===== Agent compute here =====")
    body.append("        // ===== end =====")
    body.append("")
    for e in in_edges:
        body.append(f"        cb_pop_front(cb_{e.stream}, 1);")
    for e in out_edges:
        body.append(f"        cb_reserve_back(cb_{e.stream}, 1);")
        body.append(f"        cb_push_back(cb_{e.stream}, 1);")
        body.append(f"        noc_semaphore_inc(sem_{e.stream}_out, 1);")
    return body


def _render(
    node: str, role: str, in_edges: List[SemEdge], out_edges: List[SemEdge]
) -> str:
    header = f"// role: {role}  node: {node}\n"
    lines = [header, _include(role), "", "void kernel_main() {"]
    lines.extend(_ct_args(role, in_edges, out_edges))
    lines.append("")
    lines.append("    for (uint32_t i = 0; i < iters; i++) {")
    lines.extend(_loop_body(in_edges, out_edges))
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


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
