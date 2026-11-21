"""E2E structure verify for v2 pipeline.

Runs generate() on Qwen3.6 decoder_layer and attention topologies,
then verify_structure() on each result. Host-only, no device.
"""

import os
import sys
import tempfile
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk.generate import generate, verify_structure
from tt_pk.params import RoleCount, StrategyParams
from tt_pk.problem import MeshSpec, StateBufferSpec, WeightRingSpec
from tt_pk.qwen36 import qwen36_27b_problem
from tt_pk.qwen36_topology import attention_topology, decoder_layer_topology


def _run(name, topology, params, mesh=MeshSpec()):
    prob = qwen36_27b_problem()
    prob = replace(prob, mesh=mesh,
                   weight_rings=(WeightRingSpec("w", stages=3,
                                                tile_count=16),),
                   state_buffers=(StateBufferSpec("kv", tile_count=4),))
    out = tempfile.mkdtemp(prefix=f"gen_{name}_")
    res = generate(prob, topology, params, out)
    verify_structure(res)
    print(f"[PASS] {name}: {len(res.cpp_paths)} cpp, "
          f"out_dir={res.out_dir}")


def main():
    _run("decoder_tp1", decoder_layer_topology(),
         StrategyParams(roles=RoleCount(worker=36)))
    _run("attention_tp1", attention_topology(),
         StrategyParams(roles=RoleCount(worker=24, hub=4)))
    _run("attention_tp4", attention_topology(),
         StrategyParams(roles=RoleCount(worker=96, hub=4)),
         mesh=MeshSpec(tp_axis=4))
    print("OK")


if __name__ == "__main__":
    main()
