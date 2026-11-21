"""End-to-end generator: strategy + artifact A + artifact B to disk.

Takes an AnalysisProblem + Topology + StrategyParams, runs the full
compiler pipeline, and writes the two artifacts to an output directory.
See mlir/INTERFACE.md § The two compiler artifacts.
"""

import os
import re
from dataclasses import dataclass
from typing import Tuple

from .cpp_template import emit_cpp_skeleton
from .params import StrategyParams
from .problem import AnalysisProblem
from .program_mlir import emit_mesh_program_mlir, emit_program_mlir
from .qwen36_topology import Topology
from .strategy import Strategy, compute_strategy


class VerifyError(ValueError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    out_dir: str
    strategy: Strategy
    mlir_path: str
    cpp_paths: Tuple[str, ...]


def generate(
    problem: AnalysisProblem, topology: Topology, params: StrategyParams,
    out_dir: str,
) -> GenerationResult:
    os.makedirs(out_dir, exist_ok=True)
    strategy = compute_strategy(problem, topology, params)

    if problem.mesh.tp_axis > 1 or problem.mesh.pp_axis > 1:
        mlir_text = emit_mesh_program_mlir(strategy, name=topology.name)
    else:
        mlir_text = emit_program_mlir(strategy, name=topology.name)
    mlir_path = os.path.join(out_dir, f"{topology.name}.mlir")
    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    cpp_files = emit_cpp_skeleton(strategy)
    cpp_paths = []
    for cpp in cpp_files:
        p = os.path.join(out_dir, cpp.path)
        with open(p, "w") as f:
            f.write(cpp.source)
        cpp_paths.append(p)

    return GenerationResult(
        out_dir=out_dir, strategy=strategy, mlir_path=mlir_path,
        cpp_paths=tuple(cpp_paths))


_CPP_COMMON = (
    "void kernel_main()",
    "get_compile_time_arg_val(0)",
    "for (uint32_t i = 0; i < iters; i++)",
)
_COMPUTE_GAP = ("// ===== Agent compute here =====", "// ===== end =====")

# Must match _ROLE_SLOTS in cpp_template.
_EXPECTED_SLOTS = {
    "streamer": ("BRISC",),
    "worker": ("BRISC", "COMPUTE", "NCRISC"),
    "head": ("BRISC", "COMPUTE", "NCRISC"),
    "hub": ("BRISC", "COMPUTE", "NCRISC"),
    "leader": ("BRISC",),
    "tail": ("NCRISC",),
}


def _check_slot(
    cpp_path: str, src: str, role: str, slot: str, node: str,
    in_edges, out_edges,
) -> None:
    for needle in _CPP_COMMON:
        if needle not in src:
            raise VerifyError(f"{cpp_path}: missing {needle!r}")
    if slot == "COMPUTE":
        for needle in _COMPUTE_GAP:
            if needle not in src:
                raise VerifyError(
                    f"{cpp_path}: missing compute gap {needle!r}")
        if "noc_async" in src or "noc_semaphore_inc" in src:
            raise VerifyError(
                f"{cpp_path}: COMPUTE slot must not call NOC")
        if in_edges and "cb_wait_front" not in src:
            raise VerifyError(
                f"{cpp_path}: COMPUTE has in-edges but no cb_wait_front")
        if out_edges and "cb_push_back" not in src:
            raise VerifyError(
                f"{cpp_path}: COMPUTE has out-edges but no cb_push_back")
        return
    if role == "streamer" and slot == "BRISC":
        for needle in ("noc_async_read", "noc_async_write",
                       "noc_semaphore_inc", "get_common_arg_val",
                       "get_noc_addr_from_bank_id"):
            if needle not in src:
                raise VerifyError(
                    f"{cpp_path}: streamer BRISC missing {needle!r}")
        return
    if role == "tail" and slot == "NCRISC":
        for needle in ("noc_semaphore_wait_min", "noc_async_write",
                       "get_common_arg_val",
                       "get_noc_addr_from_bank_id"):
            if needle not in src:
                raise VerifyError(
                    f"{cpp_path}: tail NCRISC missing {needle!r}")
        return
    # Compute-node BRISC/NCRISC data mover.
    if slot == "BRISC":
        if in_edges and "noc_semaphore_wait_min" not in src:
            raise VerifyError(
                f"{cpp_path}: BRISC in-edges but no sem wait")
        if in_edges and "cb_push_back" not in src:
            raise VerifyError(
                f"{cpp_path}: BRISC in-edges but no local cb_push_back")
        return
    if slot == "NCRISC":
        if out_edges:
            for needle in ("cb_wait_front", "noc_async_write",
                           "noc_semaphore_inc", "get_common_arg_val"):
                if needle not in src:
                    raise VerifyError(
                        f"{cpp_path}: NCRISC out-edges but missing "
                        f"{needle!r}")
        return
    raise VerifyError(
        f"{cpp_path}: unhandled (role={role!r}, slot={slot!r})")


def verify_structure(result: GenerationResult) -> None:
    role_by_node = {r.node: r.role for r in result.strategy.roles}
    expected_total = sum(len(_EXPECTED_SLOTS[r.role])
                         for r in result.strategy.roles)
    if len(result.cpp_paths) != expected_total:
        raise VerifyError(
            f"cpp count {len(result.cpp_paths)} != expected "
            f"{expected_total} (sum of slots per node)")
    per_node_slots: dict = {}
    for cpp_path in result.cpp_paths:
        with open(cpp_path) as f:
            src = f.read()
        hdr = re.search(
            r"// role: (\S+)\s+node: (\S+)\s+slot: (\S+)", src)
        if not hdr:
            raise VerifyError(
                f"{cpp_path}: missing role/node/slot header")
        role, node, slot = hdr.group(1), hdr.group(2), hdr.group(3)
        if role_by_node.get(node) != role:
            raise VerifyError(
                f"{cpp_path}: role {role!r} mismatch strategy")
        per_node_slots.setdefault(node, []).append(slot)
        in_edges = [e for e in result.strategy.sem_graph
                    if e.dst == node]
        out_edges = [e for e in result.strategy.sem_graph
                     if e.src == node]
        _check_slot(cpp_path, src, role, slot, node, in_edges, out_edges)
    for node, role in role_by_node.items():
        expected = set(_EXPECTED_SLOTS[role])
        got = set(per_node_slots.get(node, []))
        if got != expected:
            raise VerifyError(
                f"node {node!r}: slots {got} != expected {expected}")

    with open(result.mlir_path) as f:
        mlir = f.read()
    if mlir.count("<") != mlir.count(">"):
        raise VerifyError(
            f"{result.mlir_path}: angle brackets unbalanced "
            f"({mlir.count('<')} < vs {mlir.count('>')} >)")
    if mlir.count("[") != mlir.count("]"):
        raise VerifyError(
            f"{result.mlir_path}: square brackets unbalanced")
    if mlir.count("{") != mlir.count("}"):
        raise VerifyError(
            f"{result.mlir_path}: braces unbalanced")
    for attr in ("tt_pk.mesh", "tt_pk.roles", "tt_pk.cbs", "tt_pk.sems",
                 "tt_pk.weight_rings", "tt_pk.state_buffers"):
        if attr not in mlir:
            raise VerifyError(f"{result.mlir_path}: missing {attr}")
