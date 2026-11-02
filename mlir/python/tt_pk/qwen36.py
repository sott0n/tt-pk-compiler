"""Qwen3.6-27B decoder-layer snapshot as an AnalysisProblem.

Numbers are the model's public config. They feed the search's weighting
only; CP3 does not require them to drive an accurate placement.
"""

from __future__ import annotations

from .problem import (AnalysisProblem, DecoderLayerSpec, HardwareSpec,
                      KernelSpec)


_HIDDEN = 4096
_HEAD_DIM = 128
_NUM_HEADS = 32
_NUM_KV_HEADS = 8
_INTERMEDIATE = 11008
_QKV_OUT = _HEAD_DIM * (_NUM_HEADS + 2 * _NUM_KV_HEADS)
# Attention weight calibrated so attention gets ~10-15% of the worker
# grid at the Qwen3.6-27B shape, roughly matching the compute share
# observed in the M0 Resident baseline. The decode-side scaling factor
# is a tuning constant, not a true context length; a real roofline-ish
# cost model lands in a later checkpoint.
_ATTN_DECODE_SCALE = 64
_ATTN_COST = _HEAD_DIM * _HEAD_DIM * _NUM_HEADS * _ATTN_DECODE_SCALE


def qwen36_27b_layer() -> DecoderLayerSpec:
    return DecoderLayerSpec(kernels=(
        KernelSpec("qwen36.qkv_proj", k=_HIDDEN, n=_QKV_OUT,
                   produces="qkv", consumes=("x_norm",)),
        KernelSpec("qwen36.attention", k=_HEAD_DIM, n=_HEAD_DIM,
                   produces="attn", consumes=("qkv",),
                   cost=_ATTN_COST),
        KernelSpec("qwen36.o_proj", k=_HEAD_DIM * _NUM_HEADS, n=_HIDDEN,
                   produces="attn_proj", consumes=("attn",)),
        KernelSpec("qwen36.mlp_gate", k=_HIDDEN, n=_INTERMEDIATE,
                   produces="gate", consumes=("x_norm2",)),
        KernelSpec("qwen36.mlp_up", k=_HIDDEN, n=_INTERMEDIATE,
                   produces="up", consumes=("x_norm2",)),
        KernelSpec("qwen36.mlp_down", k=_INTERMEDIATE, n=_HIDDEN,
                   produces="mlp_out", consumes=("gate", "up")),
    ))


def bh_p150_hardware() -> HardwareSpec:
    # Blackhole P150 Tensix grid is 13 x 10 (130 workers before reserves).
    # Dispatcher takes column 0, ethernet reserves the top edge; those
    # tiles are blocked out so the search can only land on true workers.
    # Numbers are conservative placeholders until the tt-metal cluster
    # spec is parsed into the compiler.
    reserved = tuple(
        [(0, y) for y in range(10)] +
        [(x, 0) for x in range(1, 13)]
    )
    return HardwareSpec(name="bh_p150", grid_x=13, grid_y=10,
                        reserved_cores=reserved)


def qwen36_27b_problem() -> AnalysisProblem:
    return AnalysisProblem(
        model_name="qwen36_27b",
        layer=qwen36_27b_layer(),
        hardware=bh_p150_hardware(),
    )
