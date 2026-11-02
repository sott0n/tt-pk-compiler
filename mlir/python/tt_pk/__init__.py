"""Python entry point for the `tt_pk` MLIR dialect.

CP2 scope: build a textual `tt_pk` program and verify it through tt-pk-opt.
Later checkpoints replace the subprocess driver with proper CAPI bindings.
"""

from .builder import CoreSet, Kernel, Program, verify, assemble

__all__ = ["CoreSet", "Kernel", "Program", "verify", "assemble"]
