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
    # Tiles exchanged on each stream per node iter. Unlisted streams
    # default to 1. Lets a matmul node say "4 Q tiles and 4 K tiles per
    # S tile", or a streamer say "my iter count is the per-stream tile
    # total" (default 1 already matches that for 1-stream streamers).
    tile_rates: Tuple[Tuple[str, int], ...] = ()
    # Per-produced-stream cycle length: how many unique tiles the
    # producer walks before looping its DRAM / local address back to
    # zero. Unlisted streams default to no cycling (linear addressing).
    # Lets reader_k read eight kv-head tiles and reuse them four times
    # for GQA without changing the iter count.
    output_cycle: Tuple[Tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Topology:
    name: str
    nodes: Tuple[TopologyNode, ...] = field(default_factory=tuple)


def attention_topology() -> Topology:
    # hd: contraction width per S output tile along the head dim (Q @
    #   K^T, four Q and four K tiles per S tile).
    # l: S/P row width softmax reduces over, and the P contraction
    #   width per O tile (P @ V, two P and two V tiles per O tile).
    # One qk / softmax / pv iter is one head: it emits L S, L P or HD_N
    #   O tiles. The K and V reads per iter cover every output along the
    #   N axis — eight K tiles (L × HD) feed the L S outputs, eight V
    #   tiles feed the HD_N=HD O outputs.
    hd = 4
    l = 2
    return Topology(
        name="qwen36.attention",
        nodes=(
            TopologyNode("reader_q", "streamer", (), ("q",)),
            TopologyNode("reader_k", "streamer", (), ("k",),
                         output_cycle=(("k", 8),)),
            TopologyNode("reader_v", "streamer", (), ("v",),
                         output_cycle=(("v", 8),)),
            TopologyNode("qk", "worker", ("q", "k"), ("s",),
                         tile_rates=(("q", hd), ("k", hd * l),
                                     ("s", l))),
            TopologyNode("softmax", "worker", ("s",), ("p",),
                         tile_rates=(("s", l), ("p", l))),
            TopologyNode("pv", "worker", ("p", "v"), ("o",),
                         tile_rates=(("p", l), ("v", l * hd),
                                     ("o", hd))),
            TopologyNode("concat_heads", "hub", ("o",), ("attn",)),
            TopologyNode("writer", "tail", ("attn",), ()),
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


# GDN (Gated DeltaNet) decode-time topology, modeled after
# tt-qwen-resident gdn/decode.recurrent_forward. The recurrent delta rule
# reads and updates a persistent [B, H, K, V] state; conv state is per
# q/k/v. "qkvz_proj" is the fused projection emitting q, k, v, gate.
def gdn_topology() -> Topology:
    return Topology(
        name="qwen36.gdn",
        nodes=(
            TopologyNode("qkvz_proj", "worker", ("x_norm",),
                         ("q", "k", "v", "g")),
            TopologyNode("q_conv", "worker", ("q",), ("q_c",)),
            TopologyNode("k_conv", "worker", ("k",), ("k_c",)),
            TopologyNode("v_conv", "worker", ("v",), ("v_c",)),
            TopologyNode("l2norm_q", "worker", ("q_c",), ("q_n",)),
            TopologyNode("l2norm_k", "worker", ("k_c",), ("k_n",)),
            TopologyNode("silu_g", "worker", ("g",), ("g_a",)),
            TopologyNode("delta_rule", "worker",
                         ("q_n", "k_n", "v_c", "g_a"), ("o",)),
            TopologyNode("o_proj", "worker", ("o",), ("gdn_out",)),
        ),
    )
