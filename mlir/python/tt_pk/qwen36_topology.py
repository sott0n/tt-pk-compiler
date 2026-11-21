"""Qwen3.6 topology, aligned with tt-qwen-resident.

Phase 1. Role-oriented topology (streamer / worker / tail / leader / hub
/ head) for the decoder layer and its attention cluster. Replaces the
old Transformer-shaped 6-kernel scaffold in qwen36.py. Numeric constants
come from Qwen3.6-27B's public config. See mlir/INTERFACE.md § Module
layout (v2) and § Parametric inputs.
"""

from dataclasses import dataclass, field
from typing import Tuple


HIDDEN = 4096
HEAD_DIM = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
GQA_RATIO = NUM_HEADS // NUM_KV_HEADS
INTERMEDIATE = 11008
QKV_OUT = HEAD_DIM * (NUM_HEADS + 2 * NUM_KV_HEADS)


@dataclass(frozen=True)
class TopologyNode:
    name: str
    role: str
    consumes: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Topology:
    name: str
    nodes: Tuple[TopologyNode, ...] = field(default_factory=tuple)


def attention_topology() -> Topology:
    return Topology(
        name="qwen36.attention",
        nodes=(
            TopologyNode("qk", "worker", ("q", "k"), ("s",)),
            TopologyNode("softmax", "worker", ("s",), ("p",)),
            TopologyNode("pv", "worker", ("p", "v"), ("o",)),
            TopologyNode("concat_heads", "hub", ("o",), ("attn",)),
        ),
    )


def decoder_layer_topology() -> Topology:
    return Topology(
        name="qwen36.decoder_layer",
        nodes=(
            TopologyNode("qkv_proj", "worker", ("x_norm",), ("qkv",)),
            TopologyNode("attention", "worker", ("qkv",), ("attn",)),
            TopologyNode("o_proj", "worker", ("attn",), ("attn_proj",)),
            TopologyNode("mlp_gate", "worker", ("x_norm2",), ("gate",)),
            TopologyNode("mlp_up", "worker", ("x_norm2",), ("up",)),
            TopologyNode("mlp_down", "worker", ("gate", "up"), ("mlp_out",)),
        ),
    )
