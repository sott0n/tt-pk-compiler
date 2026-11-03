"""Python entry point for the `tt_pk` MLIR dialect.

Textual builder plus a 3-stage pipeline (AnalysisProblem -> CandidatePlan
-> tt_pk Program). The driver stays subprocess-based until the dialect C
API lands.
"""

from .builder import CoreSet, Kernel, Program, assemble, verify
from .materialize import materialize
from .plan import CandidatePlan, CoreAssignment
from .problem import (AnalysisProblem, DecoderLayerSpec, HardwareSpec,
                      KernelEntry, KernelSpec)
from .codegen import (CBSpec, SemSpec, derive_cbs, derive_semaphores,
                      emit_ttnn_generic)
from .kernel_source import KernelSourceError, escape_for_mlir, load as load_kernel_source
from .ttnn_lint import LintResult, TTNNLintError, ensure_available, lint
from .importer import (ImporterError, TTIR_WHITELIST, TTIRProvenance,
                       qwen36_provenance, validate_provenance)
from .resident import (ResidentOp, ResidentSchedule, lower_to_resident,
                       to_json as resident_to_json)
from .search import greedy_search

__all__ = [
    "CoreSet", "Kernel", "Program", "assemble", "verify",
    "AnalysisProblem", "DecoderLayerSpec", "HardwareSpec", "KernelEntry",
    "KernelSpec",
    "CandidatePlan", "CoreAssignment",
    "greedy_search", "materialize",
    "ResidentOp", "ResidentSchedule", "lower_to_resident",
    "resident_to_json",
    "ImporterError", "TTIR_WHITELIST", "TTIRProvenance",
    "qwen36_provenance", "validate_provenance",
    "emit_ttnn_generic", "CBSpec", "SemSpec",
    "derive_cbs", "derive_semaphores",
    "LintResult", "TTNNLintError", "ensure_available", "lint",
    "KernelSourceError", "escape_for_mlir", "load_kernel_source",
]
