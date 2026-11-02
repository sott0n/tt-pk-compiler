"""Importer-side half of the tt-mlir interface contract (M2).

A real importer parses a TTIR module and emits an AnalysisProblem. CP6
only locks the contract: it validates that the KernelSpecs the frontend
hands over map one-to-one onto the allowed TTIR op set, and refuses
unknown mappings. Actual MLIR parsing lands in a later checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

from .problem import DecoderLayerSpec, KernelSpec


# TTIR ops the import contract recognizes. See mlir/INTERFACE.md.
TTIR_WHITELIST: FrozenSet[str] = frozenset({
    "ttir.rms_norm",
    "ttir.linear",
    "ttir.matmul",
    "ttir.softmax",
    "ttir.concatenate_heads",
    "ttir.concat",
    "ttir.reshape",
    "ttir.permute",
    "ttir.embedding",
    "ttir.all_reduce",
})


class ImporterError(ValueError):
    """Raised when a KernelSpec's TTIR provenance is not in the whitelist."""


@dataclass(frozen=True)
class TTIRProvenance:
    """Which TTIR ops a KernelSpec was fused from. Empty tuple = unknown."""
    kernel_name: str
    ttir_ops: Tuple[str, ...]


def validate_provenance(layer: DecoderLayerSpec,
                        provenance: Dict[str, Tuple[str, ...]]) -> None:
    """Raise ImporterError if any KernelSpec cites a TTIR op not in the whitelist.

    `provenance` maps kernel name -> tuple of TTIR op names the frontend
    fused into that kernel. Missing entries are allowed only for the
    identity "x_norm / x_norm2" roots, which are layer-level activations
    produced by the enclosing norm (handled upstream).
    """
    for k in layer.kernels:
        ops = provenance.get(k.name, ())
        if not ops:
            raise ImporterError(
                f"kernel {k.name!r} has no TTIR provenance; the contract "
                f"requires every imported kernel to declare the TTIR ops it "
                f"was fused from")
        unknown = [op for op in ops if op not in TTIR_WHITELIST]
        if unknown:
            raise ImporterError(
                f"kernel {k.name!r} cites TTIR ops not in the import "
                f"whitelist: {unknown}")


def qwen36_provenance() -> Dict[str, Tuple[str, ...]]:
    """TTIR provenance for the Qwen3.6-27B decoder layer kernels.

    Attention is the one cluster that fuses four TTIR ops into one
    KernelSpec; the others map one-to-one onto a single projection.
    """
    return {
        "qwen36.qkv_proj": ("ttir.rms_norm", "ttir.linear"),
        "qwen36.attention": (
            "ttir.matmul", "ttir.softmax", "ttir.matmul",
            "ttir.concatenate_heads",
        ),
        "qwen36.o_proj": ("ttir.linear", "ttir.all_reduce"),
        "qwen36.mlp_gate": ("ttir.rms_norm", "ttir.matmul"),
        "qwen36.mlp_up": ("ttir.matmul",),
        "qwen36.mlp_down": ("ttir.matmul", "ttir.all_reduce"),
    }
