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


_CPP_REQUIRED = (
    "void kernel_main()",
    "get_compile_time_arg_val(0)",
    "for (uint32_t i = 0; i < iters; i++)",
    "// ===== Agent compute here =====",
    "// ===== end =====",
)


def verify_structure(result: GenerationResult) -> None:
    n_nodes = len(result.strategy.roles)
    if len(result.cpp_paths) != n_nodes:
        raise VerifyError(
            f"cpp count {len(result.cpp_paths)} != node count {n_nodes}")

    for cpp_path in result.cpp_paths:
        with open(cpp_path) as f:
            src = f.read()
        for needle in _CPP_REQUIRED:
            if needle not in src:
                raise VerifyError(f"{cpp_path}: missing {needle!r}")
        role_match = re.search(r"// role: (\S+)\s+node: (\S+)", src)
        if not role_match:
            raise VerifyError(f"{cpp_path}: missing role/node header")
        role, node = role_match.group(1), role_match.group(2)
        node_edges_in = [e for e in result.strategy.sem_graph
                         if e.dst == node]
        node_edges_out = [e for e in result.strategy.sem_graph
                          if e.src == node]
        if node_edges_in and "noc_semaphore_wait_min" not in src:
            raise VerifyError(
                f"{cpp_path}: has in-edges but no sem wait")
        if node_edges_out and "noc_semaphore_inc" not in src:
            raise VerifyError(
                f"{cpp_path}: has out-edges but no sem inc")
        if node_edges_in and "cb_wait_front" not in src:
            raise VerifyError(f"{cpp_path}: has in-edges but no cb wait")
        if node_edges_out and "cb_push_back" not in src:
            raise VerifyError(f"{cpp_path}: has out-edges but no cb push")

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
