"""Codegen-side half of the tt-mlir interface contract (M2).

Emits a textual TTNN snippet with one `ttnn.generic` op per ResidentOp.
CP7 fills in CB and semaphore attributes with real sizes derived from
KernelSpec shapes and the schedule's data-dependency edges. The exact
syntax is not yet verified against tt-mlir's parser; CP7b wires that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .kernel_source import escape_for_mlir
from .resident import ResidentOp, ResidentSchedule


# Per-dtype 32x32 tile byte count, mirroring tt-mlir's
# `ttcore::getTileSizeBytes` (lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp).
# Keys are the tt-mlir DataType enum spellings as the TTNN parser accepts
# them in `#ttnn.kernel_cb_format<dtype = ...>`.
_TILE_BYTES_BY_DTYPE: Dict[str, int] = {
    "f32": 4096,
    "f16": 2048,
    "bf16": 2048,
    "bfp_f8": 1088,
    "bfp_bf8": 1088,
    "bfp_f4": 576,
    "bfp_bf4": 576,
    "bfp_f2": 320,
    "bfp_bf2": 320,
    "u32": 4096,
    "u16": 2048,
    "u8": 1024,
    "si32": 4096,
    "i1": 1024,
}
_DOUBLE_BUFFER = 2


@dataclass(frozen=True)
class CBSpec:
    """One CB that lives on a kernel's cores."""
    buffer_index: int
    dtype: str
    page_size: int
    tile_count: int

    def total_size(self) -> int:
        return self.tile_count * self.page_size


@dataclass(frozen=True)
class SemSpec:
    """One semaphore between a producer kernel and a consumer kernel."""
    id: int
    producer: str
    consumer: str
    initial_value: int = 0


@dataclass(frozen=True)
class KernelCodegen:
    """Everything the codegen emits per ResidentOp."""
    op: ResidentOp
    cbs: Tuple[CBSpec, ...]
    waits_for: Tuple[int, ...]      # semaphore ids this op waits on
    signals: Tuple[int, ...] = ()   # semaphore ids this op signals on exit


def derive_cbs(op: ResidentOp) -> Tuple[CBSpec, ...]:
    """One input CB per consumed tensor, one weight CB, one output CB.

    Tile count = double_buffer * ceil(K / tile) for inputs/weight and
    double_buffer * ceil(N / tile) for the output. The dtype's tile byte
    count comes from `_TILE_BYTES_BY_DTYPE`; an unknown dtype raises so
    the compiler never silently emits a wrong page size.
    """
    page_size = _TILE_BYTES_BY_DTYPE.get(op.dtype)
    if page_size is None:
        raise NotImplementedError(
            f"codegen has no tile byte count for dtype {op.dtype!r}; "
            f"known: {sorted(_TILE_BYTES_BY_DTYPE)}")
    k_tiles = max(1, (op.k + 31) // 32)
    n_tiles = max(1, (op.n + 31) // 32)

    cbs: List[CBSpec] = []
    index = 0
    for _ in op.consumes:
        cbs.append(CBSpec(index, op.dtype, page_size,
                          _DOUBLE_BUFFER * k_tiles))
        index += 1
    cbs.append(CBSpec(index, op.dtype, page_size,
                      _DOUBLE_BUFFER * k_tiles))
    index += 1
    cbs.append(CBSpec(index, op.dtype, page_size,
                      _DOUBLE_BUFFER * n_tiles))
    return tuple(cbs)


def derive_semaphores(schedule: ResidentSchedule) -> Tuple[SemSpec, ...]:
    """One semaphore per producer->consumer edge in the schedule.

    Only edges where both ends are kernels in the schedule earn a
    semaphore. Layer-level activations (x_norm, x_norm2) are produced by
    norms outside the decoder-layer and need host-fed signalling, which
    is CP7b's job.
    """
    producers: Dict[str, str] = {op.produces: op.kernel_name
                                 for op in schedule.ops if op.produces}
    sems: List[SemSpec] = []
    next_id = 0
    for consumer in schedule.ops:
        for tensor in consumer.consumes:
            producer = producers.get(tensor)
            if producer is None:
                continue
            sems.append(SemSpec(id=next_id, producer=producer,
                                consumer=consumer.kernel_name))
            next_id += 1
    return tuple(sems)


def _kernel_sym(kernel_name: str) -> str:
    safe = kernel_name.replace(".", "_")
    return f"@{safe}_kernel"


def _emit_core_ranges(cores: Tuple[Tuple[int, int], ...]) -> str:
    # One 1x1 CoreRange per core. Coalescing contiguous rows into bigger
    # CoreRange attrs is CP8 scope; what matters here is a syntax
    # ttmlir-opt accepts.
    ranges = ", ".join(f"#ttnn.core_range<({x}, {y}), ({x}, {y})>"
                       for x, y in cores)
    return f"#ttnn.core_range_set<[{ranges}]>"


def _emit_cb(cb: CBSpec, cores_text: str) -> str:
    fmt = (f"#ttnn.kernel_cb_format<buffer_index = {cb.buffer_index}, "
           f"dtype = {cb.dtype}, page_size = {cb.page_size}>")
    return (f"#ttnn.kernel_cb<total_size = {cb.total_size()}, "
            f"core_ranges = {cores_text}, formats = [{fmt}]>")


def _emit_sem(sem: SemSpec, cores_text: str) -> str:
    return (f"#ttnn.kernel_semaphore<id = {sem.id}, core_type = worker, "
            f"core_ranges = {cores_text}, "
            f"initial_value = {sem.initial_value}>")


def _emit_data_kernel(mnemonic: str, body: str, cores_text: str) -> str:
    src_lit = f"\"{escape_for_mlir(body)}\""
    return (f"#ttnn.{mnemonic}<source = {src_lit}"
            f", core_ranges = {cores_text}"
            f", ct_args = []"
            f", common_rt_args = []"
            f", rt_args = []>")


def _emit_compute_kernel(op: ResidentOp, cores_text: str,
                         cb_count: int) -> str:
    # Each CB needs an UnpackToDestMode entry; the enum is non-optional
    # in the TD. "default" means "do not force fp32" and matches the
    # policy fields; a bf16-first-class entry can land when the IR
    # starts tracking per-CB precision overrides.
    unpack_modes = ", ".join(["default"] * cb_count)
    cfg = op.compute_config
    body_src = op.compute_kernel_source
    if body_src is not None:
        src_lit = f"\"{escape_for_mlir(body_src)}\""
        return (
            f"#ttnn.source_compute_kernel<source = {src_lit}"
            f", core_ranges = {cores_text}"
            f", math_fidelity = {cfg.math_fidelity}"
            f", fp32_dest_acc_en = {str(cfg.fp32_dest_acc_en).lower()}"
            f", dst_full_sync_en = {str(cfg.dst_full_sync_en).lower()}"
            f", unpack_to_dest_modes = [{unpack_modes}]"
            f", bfp8_pack_precise = {str(cfg.bfp8_pack_precise).lower()}"
            f", math_approx_mode = {str(cfg.math_approx_mode).lower()}"
            f", ct_args = []"
            f", common_rt_args = []"
            f", rt_args = []>"
        )
    return (
        f"#ttnn.compute_kernel<symbol_ref = {_kernel_sym(op.kernel_name)}"
        f", core_ranges = {cores_text}"
        f", math_fidelity = {cfg.math_fidelity}"
        f", fp32_dest_acc_en = {str(cfg.fp32_dest_acc_en).lower()}"
        f", dst_full_sync_en = {str(cfg.dst_full_sync_en).lower()}"
        f", unpack_to_dest_modes = [{unpack_modes}]"
        f", bfp8_pack_precise = {str(cfg.bfp8_pack_precise).lower()}"
        f", math_approx_mode = {str(cfg.math_approx_mode).lower()}"
        f", ct_args = []"
        f", common_rt_args = []"
        f", rt_args = []>"
    )


def _per_op_sem_waits(schedule: ResidentSchedule,
                      sems: Tuple[SemSpec, ...]
                      ) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    waits: Dict[str, List[int]] = {op.kernel_name: [] for op in schedule.ops}
    signals: Dict[str, List[int]] = {op.kernel_name: [] for op in schedule.ops}
    for s in sems:
        waits[s.consumer].append(s.id)
        signals[s.producer].append(s.id)
    return {name: (tuple(waits[name]), tuple(signals[name]))
            for name in waits}


def emit_ttnn_generic(schedule: ResidentSchedule) -> str:
    sems = derive_semaphores(schedule)
    sem_table = _per_op_sem_waits(schedule, sems)

    lines = ["module {"]
    for op in schedule.ops:
        if op.compute_kernel_source is None:
            lines.append(f"  func.func private {_kernel_sym(op.kernel_name)}()")
    lines.append("")
    lines.append("  func.func @decoder_layer() {")
    for op in schedule.ops:
        cores_text = _emit_core_ranges(op.cores)
        cbs = derive_cbs(op)
        cb_text = ", ".join(_emit_cb(cb, cores_text) for cb in cbs)
        op_sems = [s for s in sems
                   if s.producer == op.kernel_name
                   or s.consumer == op.kernel_name]
        sem_text = ", ".join(_emit_sem(s, cores_text) for s in op_sems)
        waits, signals = sem_table[op.kernel_name]
        operands = ", ".join(f"%{c}" for c in op.consumes) or "/* none */"
        kernels_text = [_emit_compute_kernel(op, cores_text, len(cbs))]
        if op.reader_kernel_source is not None:
            kernels_text.append(_emit_data_kernel(
                "source_read_kernel", op.reader_kernel_source, cores_text))
        if op.writer_kernel_source is not None:
            kernels_text.append(_emit_data_kernel(
                "source_write_kernel", op.writer_kernel_source, cores_text))
        kernel_decl = ", ".join(kernels_text)
        lines.append(
            f"    %{op.produces} = ttnn.generic {{"
            f" program = #ttnn.program<kernels = [{kernel_decl}], "
            f"cbs = [{cb_text}], semaphores = [{sem_text}]> "
            f"}} {{waits = {list(waits)}, signals = {list(signals)}}} : "
            f"({operands}) -> tensor<_>"
        )
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
