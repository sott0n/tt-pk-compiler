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
    # Nested-list form; the ttnn.generic verifier wants the array literal
    # directly on the `core_ranges` field, not a `#ttnn.core_range_set`.
    ranges = ", ".join(f"#ttnn.core_range<({x}, {y}), ({x}, {y})>"
                       for x, y in cores)
    return f"<[{ranges}]>"


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


# The flatbuffer target JIT-links the C++ body into a RISC-V elf whose
# `_start` references `kernel_main()`. An empty body fails that link,
# so default to the tt-metal "blank" kernel: a no-op `kernel_main`.
_COMPUTE_STUB = "void kernel_main() {}"
_READER_STUB = "void kernel_main() {}"
_WRITER_STUB = "void kernel_main() {}"


def _emit_compute_kernel(op: ResidentOp, cores_text: str,
                         cb_count: int) -> str:
    # Each CB needs an UnpackToDestMode entry; the enum is non-optional
    # in the TD. The flatbuffer target refuses the symbol_ref form unless
    # a matching ttkernel.func lives in the module, so default to a stub
    # inline source when the KernelSpec carries none. The agent swaps it.
    unpack_modes = ", ".join(["default"] * cb_count)
    cfg = op.compute_config
    body_src = op.compute_kernel_source or _COMPUTE_STUB
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


def _mlir_elt(dtype: str) -> str:
    # Bit-packed TTNN dtypes have no MLIR tensor element type; carry them
    # on CB formats and fall back to bf16 at the tensor level.
    return "bf16" if dtype.startswith("bfp") else dtype


def _tile_grid(k: int, n: int) -> Tuple[int, int]:
    return max(1, (k + 31) // 32), max(1, (n + 31) // 32)


def emit_ttnn_generic(schedule: ResidentSchedule) -> str:
    tensor_shape: Dict[str, Tuple[int, int]] = {}
    tensor_dtype: Dict[str, str] = {}
    arg_order: List[str] = []
    seen: set = set()
    for op in schedule.ops:
        for name in op.consumes:
            tensor_shape.setdefault(name, (op.k, op.n))
            tensor_dtype.setdefault(name, op.dtype)
            if name not in seen:
                arg_order.append(name); seen.add(name)
        tensor_shape[op.produces] = (op.k, op.n)
        tensor_dtype[op.produces] = op.dtype
        if op.produces not in seen:
            arg_order.append(op.produces); seen.add(op.produces)

    layouts: Dict[Tuple[int, int, str], str] = {}

    def layout_for(name: str) -> Tuple[str, str, int, int]:
        k, n = tensor_shape[name]
        elt = _mlir_elt(tensor_dtype[name])
        key = (k, n, elt)
        if key not in layouts:
            layouts[key] = f"#ttnn_layout{len(layouts)}"
        return layouts[key], elt, k, n

    arg_type: Dict[str, str] = {}
    for name in arg_order:
        alias, elt, k, n = layout_for(name)
        arg_type[name] = f"tensor<{k}x{n}x{elt}, {alias}>"

    lines: List[str] = ["#dram = #ttnn.buffer_type<dram>"]
    for (k, n, elt), alias in layouts.items():
        kt, nt = _tile_grid(k, n)
        lines.append(
            f"{alias} = #ttnn.ttnn_layout<"
            f"(d0, d1) -> (d0, d1), <1x1>, "
            f"memref<{kt}x{nt}x!ttcore.tile<32x32, {elt}>, #dram>, "
            f"<interleaved>>")
    lines.append("")
    lines.append("module {")

    args_text = ", ".join(f"%{n}: {arg_type[n]}" for n in arg_order)
    final = schedule.ops[-1].produces
    func_sym = f"{schedule.model_name}_graph".replace(".", "_")
    lines.append(
        f"  func.func @{func_sym}({args_text}) -> {arg_type[final]} "
        f"attributes {{tt.function_type = \"forward_device\"}} {{")

    for op in schedule.ops:
        cores_text = _emit_core_ranges(op.cores)
        cbs = derive_cbs(op)
        cb_text = ", ".join(_emit_cb(cb, cores_text) for cb in cbs)
        kernels_text = [_emit_compute_kernel(op, cores_text, len(cbs))]
        reader_src = op.reader_kernel_source or _READER_STUB
        writer_src = op.writer_kernel_source or _WRITER_STUB
        kernels_text.append(_emit_data_kernel(
            "source_read_kernel", reader_src, cores_text))
        kernels_text.append(_emit_data_kernel(
            "source_write_kernel", writer_src, cores_text))
        kernel_decl = ", ".join(kernels_text)

        operands = list(op.consumes) + [op.produces]
        operand_vals = ", ".join(f"%{t}" for t in operands)
        operand_types = ", ".join(arg_type[t] for t in operands)
        lines.append(
            f"    \"ttnn.generic\"({operand_vals}) <{{"
            f"operandSegmentSizes = array<i32: {len(operands)}, 0>, "
            f"program = #ttnn.program<kernels = [{kernel_decl}], "
            f"cbs = [{cb_text}], semaphores = []>}}> : "
            f"({operand_types}) -> ()")

    lines.append(f"    return %{final} : {arg_type[final]}")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
