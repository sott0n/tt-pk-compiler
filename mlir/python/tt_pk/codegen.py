"""Codegen-side half of the tt-mlir interface contract (M2).

Emits a textual TTNN snippet with one `ttnn.generic` op per ResidentOp.
CP7 fills in CB and semaphore attributes with real sizes derived from
KernelSpec shapes and the schedule's data-dependency edges. The exact
syntax is not yet verified against tt-mlir's parser; CP7b wires that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .codegen_types import CBSpec, SemSpec
from .kernel_source import escape_for_mlir
from .problem import KernelEntry
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


def _emit_one_arg(value, prefix: str, i: int) -> str:
    # An int is a literal ct/rt_arg value. Tuples tag typed KernelArg attrs:
    #   ("address_of_tensor", N)       -> #ttnn.kernel_arg_address_of_tensor<N>
    #   ("cb_buffer_index", N)         -> #ttnn.kernel_arg_cb_buffer_index<N>
    #   ("tensor_accessor_args", N)    -> #ttnn.kernel_arg_tensor_accessor_args<N>
    if isinstance(value, tuple) and len(value) == 2:
        kind, idx = value
        if kind == "address_of_tensor":
            return f"#ttnn.kernel_arg_address_of_tensor<{int(idx)}>"
        if kind == "cb_buffer_index":
            return f"#ttnn.kernel_arg_cb_buffer_index<{int(idx)}>"
        if kind == "tensor_accessor_args":
            return f"#ttnn.kernel_arg_tensor_accessor_args<{int(idx)}>"
        raise ValueError(f"unknown kernel_arg tag {kind!r}")
    return (f"#ttnn.kernel_named_arg<name = \"{prefix}{i}\", "
            f"value = {int(value)}>")


def _emit_named_args(values, prefix: str) -> str:
    if not values:
        return "[]"
    parts = ", ".join(_emit_one_arg(v, prefix, i) for i, v in enumerate(values))
    return f"[{parts}]"


def _emit_rt_args(rt_args) -> str:
    # Emit per-core rt_args: `[#ttnn.core_runtime_args<core_coord=<x,y>,
    # args=[...]>, ...]`. Each args list is a sequence of KernelArg values
    # (literal uint or typed tag), same shape as common_rt_args.
    if not rt_args:
        return "[]"
    parts = []
    for (x, y), args in rt_args:
        if args:
            args_body = ", ".join(_emit_one_arg(v, f'rt{x}_{y}_', i)
                                  for i, v in enumerate(args))
        else:
            args_body = ""
        parts.append(f"#ttnn.core_runtime_args<core_coord = "
                     f"#ttnn.core_coord<{x}, {y}>, args = [{args_body}]>")
    return f"[{', '.join(parts)}]"


def _emit_data_kernel(mnemonic: str, body: str, cores_text: str,
                      ct_args: Tuple[int, ...] = (),
                      common_rt_args: Tuple[int, ...] = (),
                      rt_args=()) -> str:
    src_lit = f"\"{escape_for_mlir(body)}\""
    return (f"#ttnn.{mnemonic}<source = {src_lit}"
            f", core_ranges = {cores_text}"
            f", ct_args = {_emit_named_args(ct_args, 'ct_arg_')}"
            f", common_rt_args = {_emit_named_args(common_rt_args, 'rt_arg_')}"
            f", rt_args = {_emit_rt_args(rt_args)}>")


# The flatbuffer target JIT-links the C++ body into a RISC-V elf whose
# `_start` references `kernel_main()`. An empty body fails that link,
# so default to the tt-metal "blank" kernel: a no-op `kernel_main`.
_COMPUTE_STUB = "void kernel_main() {}"
_READER_STUB = "void kernel_main() {}"
_WRITER_STUB = "void kernel_main() {}"


def _emit_compute_source(body: str, cores_text: str,
                         cfg, cb_count: int,
                         ct_args: Tuple[int, ...] = (),
                         common_rt_args: Tuple[int, ...] = (),
                         rt_args=()) -> str:
    # Each CB needs an UnpackToDestMode entry; the enum is non-optional
    # in the TD. The flatbuffer target refuses the symbol_ref form unless
    # a matching ttkernel.func lives in the module.
    unpack_modes = ", ".join(["default"] * cb_count)
    src_lit = f"\"{escape_for_mlir(body)}\""
    return (
        f"#ttnn.source_compute_kernel<source = {src_lit}"
        f", core_ranges = {cores_text}"
        f", math_fidelity = {cfg.math_fidelity}"
        f", fp32_dest_acc_en = {str(cfg.fp32_dest_acc_en).lower()}"
        f", dst_full_sync_en = {str(cfg.dst_full_sync_en).lower()}"
        f", unpack_to_dest_modes = [{unpack_modes}]"
        f", bfp8_pack_precise = {str(cfg.bfp8_pack_precise).lower()}"
        f", math_approx_mode = {str(cfg.math_approx_mode).lower()}"
        f", ct_args = {_emit_named_args(ct_args, 'ct_arg_')}"
        f", common_rt_args = {_emit_named_args(common_rt_args, 'rt_arg_')}"
        f", rt_args = {_emit_rt_args(rt_args)}>"
    )


def _emit_compute_kernel(op: ResidentOp, cores_text: str,
                         cb_count: int) -> str:
    body = op.compute_kernel_source or _COMPUTE_STUB
    return _emit_compute_source(body, cores_text, op.compute_config, cb_count)


def _emit_entry(entry: KernelEntry, cb_count: int,
                default_cfg) -> str:
    cores_text = _emit_core_ranges(entry.cores)
    if entry.role == "compute":
        cfg = entry.compute_config or default_cfg
        return _emit_compute_source(entry.source, cores_text, cfg, cb_count,
                                    entry.ct_args, entry.common_rt_args,
                                    entry.rt_args)
    if entry.role == "reader":
        return _emit_data_kernel("source_read_kernel", entry.source,
                                 cores_text, entry.ct_args,
                                 entry.common_rt_args, entry.rt_args)
    if entry.role == "writer":
        return _emit_data_kernel("source_write_kernel", entry.source,
                                 cores_text, entry.ct_args,
                                 entry.common_rt_args, entry.rt_args)
    raise ValueError(f"unknown KernelEntry.role {entry.role!r}; "
                     f"expected compute | reader | writer")


def _mlir_elt(dtype: str) -> str:
    # Bit-packed TTNN dtypes have no MLIR tensor element type; carry them
    # on CB formats and fall back to bf16 at the tensor level.
    return "bf16" if dtype.startswith("bfp") else dtype


def _tile_grid(k: int, n: int) -> Tuple[int, int]:
    return max(1, (k + 31) // 32), max(1, (n + 31) // 32)


def emit_ttnn_generic(schedule: ResidentSchedule) -> str:
    tensor_shape: Dict[str, Tuple[int, int]] = {}
    tensor_dtype: Dict[str, str] = {}
    # Inputs are tensors consumed by some op but produced by none; they
    # arrive as function arguments. Outputs are produced by an op and
    # allocated inside the function via ttnn.empty, matching the TTNN
    # standard-flow test examples (see
    # test/ttmlir/Dialect/TTNN/generic/generic_scalar_arg_flatbuffer.mlir).
    produced: set = set()
    for op in schedule.ops:
        produced.add(op.produces)
        for name in op.consumes:
            tensor_shape.setdefault(name, (op.k, op.n))
            tensor_dtype.setdefault(name, op.dtype)
        tensor_shape[op.produces] = (op.k, op.n)
        tensor_dtype[op.produces] = op.dtype

    input_order: List[str] = []
    seen: set = set()
    for op in schedule.ops:
        for name in op.consumes:
            if name in produced or name in seen:
                continue
            input_order.append(name); seen.add(name)

    layouts: Dict[Tuple[int, int, str], str] = {}

    def layout_for(name: str) -> Tuple[str, str, int, int]:
        k, n = tensor_shape[name]
        elt = _mlir_elt(tensor_dtype[name])
        key = (k, n, elt)
        if key not in layouts:
            layouts[key] = f"#ttnn_layout{len(layouts)}"
        return layouts[key], elt, k, n

    arg_type: Dict[str, str] = {}
    for name in list(input_order) + list(produced):
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

    args_text = ", ".join(f"%{n}: {arg_type[n]}" for n in input_order)
    final = schedule.ops[-1].produces
    func_sym = f"{schedule.model_name}_graph".replace(".", "_")
    lines.append(
        f"  func.func @{func_sym}({args_text}) -> {arg_type[final]} "
        f"attributes {{tt.function_type = \"forward_device\"}} {{")

    # Grab a device handle once; each ttnn.empty uses it to allocate.
    lines.append(
        "    %dev = \"ttnn.get_device\"() <{mesh_offset = "
        "#ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : "
        "() -> !ttnn.device")

    # Allocate every produced tensor via ttnn.empty up front; ops below
    # consume the SSA values directly.
    for name in produced:
        k, n = tensor_shape[name]
        elt = _mlir_elt(tensor_dtype[name])
        lines.append(
            f"    %{name} = \"ttnn.empty\"(%dev) <{{"
            f"layout = #ttnn.layout<tile>, "
            f"shape = #ttnn.shape<{k}x{n}>}}> : "
            f"(!ttnn.device) -> {arg_type[name]}")

    for op in schedule.ops:
        cores_text = _emit_core_ranges(op.cores)
        if op.cbs:
            # Per-CB core_ranges honoured so leader / worker / tail cores can
            # share an index without stepping on each other's L1.
            cbs = op.cbs
            cb_entries = [_emit_cb(cb,
                                   _emit_core_ranges(cb.cores or op.cores))
                          for cb in cbs]
        else:
            cbs = derive_cbs(op)
            cb_entries = [_emit_cb(cb, cores_text) for cb in cbs]
        cb_text = ", ".join(cb_entries)
        if op.kernels:
            # Multi-kernel program (e.g. attention 7 kernels across leader
            # and worker grids). Each entry carries its own cores + source.
            kernels_text = [_emit_entry(e, len(cbs), op.compute_config)
                            for e in op.kernels]
        else:
            kernels_text = [_emit_compute_kernel(op, cores_text, len(cbs))]
            reader_src = op.reader_kernel_source or _READER_STUB
            writer_src = op.writer_kernel_source or _WRITER_STUB
            kernels_text.append(_emit_data_kernel(
                "source_read_kernel", reader_src, cores_text))
            kernels_text.append(_emit_data_kernel(
                "source_write_kernel", writer_src, cores_text))
        kernel_decl = ", ".join(kernels_text)

        if op.semaphores:
            sem_entries = [_emit_sem(s,
                                     _emit_core_ranges(s.cores or op.cores))
                           for s in op.semaphores]
            sem_text = ", ".join(sem_entries)
        else:
            sem_text = ""

        operands = list(op.consumes) + [op.produces]
        operand_vals = ", ".join(f"%{t}" for t in operands)
        operand_types = ", ".join(arg_type[t] for t in operands)
        lines.append(
            f"    \"ttnn.generic\"({operand_vals}) <{{"
            f"operandSegmentSizes = array<i32: {len(operands)}, 0>, "
            f"program = #ttnn.program<kernels = [{kernel_decl}], "
            f"cbs = [{cb_text}], semaphores = [{sem_text}]>}}> : "
            f"({operand_types}) -> ()")

    lines.append(f"    return %{final} : {arg_type[final]}")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
