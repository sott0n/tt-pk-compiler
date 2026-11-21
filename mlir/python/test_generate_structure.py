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
from tt_pk.qwen36_topology import (attention_topology,
                                   decoder_layer_topology, gdn_topology)


def _run(name, topology, params, mesh=MeshSpec(),
         weight_rings=None, state_buffers=None):
    prob = qwen36_27b_problem()
    if weight_rings is None:
        weight_rings = (WeightRingSpec("w", stages=3, tile_count=16),)
    if state_buffers is None:
        state_buffers = (StateBufferSpec("kv", tile_count=4),)
    prob = replace(prob, mesh=mesh, weight_rings=weight_rings,
                   state_buffers=state_buffers)
    out = tempfile.mkdtemp(prefix=f"gen_{name}_")
    res = generate(prob, topology, params, out)
    verify_structure(res)
    print(f"[PASS] {name}: {len(res.cpp_paths)} cpp, "
          f"out_dir={res.out_dir}")


def _gdn_rings():
    # Weight rings for the full gated_deltanet_forward. The qkvzba
    # projection is fused on one worker node, so W_qkvzba is one large
    # ring. W_log is the A_log parameter the delta rule looks up to form
    # the decay gate; conv rings carry weight and bias together.
    return (
        WeightRingSpec("w_qkvzba", stages=3, tile_count=32,
                       nodes=("qkvzba_proj",)),
        WeightRingSpec("w_conv_q", stages=3, tile_count=4,
                       nodes=("q_conv",)),
        WeightRingSpec("w_conv_k", stages=3, tile_count=4,
                       nodes=("k_conv",)),
        WeightRingSpec("w_conv_v", stages=3, tile_count=4,
                       nodes=("v_conv",)),
        WeightRingSpec("w_log", stages=3, tile_count=2,
                       nodes=("delta_rule",)),
        WeightRingSpec("w_rms", stages=3, tile_count=2,
                       nodes=("rms_norm_gated",)),
        WeightRingSpec("w_o", stages=3, tile_count=16, nodes=("o_proj",)),
    )


def _gdn_states():
    return (
        StateBufferSpec("s_recurrent", tile_count=8,
                        nodes=("delta_rule",)),
        StateBufferSpec("conv_state_q", tile_count=2, nodes=("q_conv",)),
        StateBufferSpec("conv_state_k", tile_count=2, nodes=("k_conv",)),
        StateBufferSpec("conv_state_v", tile_count=2, nodes=("v_conv",)),
    )


def main():
    _run("decoder_tp1", decoder_layer_topology(),
         StrategyParams(roles=RoleCount(worker=36)))
    _run("attention_tp1", attention_topology(),
         StrategyParams(roles=RoleCount(streamer=4, worker=24, hub=4,
                                       tail=4)))
    _run("attention_tp4", attention_topology(),
         StrategyParams(roles=RoleCount(streamer=16, worker=96, hub=4,
                                       tail=16)),
         mesh=MeshSpec(tp_axis=4))
    _run("gdn_tp1", gdn_topology(),
         StrategyParams(roles=RoleCount(streamer=1, worker=36, tail=1)),
         weight_rings=_gdn_rings(), state_buffers=_gdn_states())
    print("OK")


if __name__ == "__main__":
    main()
