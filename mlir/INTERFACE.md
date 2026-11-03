# tt-pk ↔ tt-mlir interface contract (M2)

Defines what the Tensix Persistent Kernel compiler reads from tt-mlir and
what it lowers back into. Scope is the Qwen3.6-27B decoder layer. Any op
outside this contract is CP6+ scope.

## Pipeline position

```
TTIR module  --(import)-->  AnalysisProblem  --(search)-->  CandidatePlan
    ^                                                             |
    |                                                             v
tt-mlir frontend                                           ResidentSchedule
                                                                  |
                                                                  v
                                                  TTNN GenericOp subgraph
                                                       --(codegen)-->
                                                  tt-mlir ttnn module
```

The compiler never emits backend code directly. Codegen stops at
`ttnn.generic` subgraph; running the generic op is tt-mlir's job.

## Import whitelist (TTIR ops the compiler recognizes)

A decoder layer that uses only these ops is importable. Any other TTIR op
raises `ImporterError` and the layer must be lowered to this set first.

| TTIR op | Mapped to KernelSpec | Qwen3.6 role |
|---|---|---|
| `ttir.rms_norm` | norm folded into next projection | pre-attention / pre-mlp norm |
| `ttir.linear` | projection matmul with bias | qkv_proj, o_proj |
| `ttir.matmul` | plain matmul (no bias) | mlp_gate, mlp_up, mlp_down |
| `ttir.softmax` | fused into attention kernel | attention softmax |
| `ttir.concatenate_heads` | fused into attention kernel | attention head merge |
| `ttir.concat` | activation plumbing | qkv split, mlp cat |
| `ttir.reshape` | layout-only | head reshape |
| `ttir.permute` | layout-only | qk transpose |
| `ttir.embedding` | input embedding | embed tokens |
| `ttir.all_reduce` | TP sum | TP all-reduce after mlp_down / o_proj |

Attention is imported as the TTIR cluster
`matmul (q,k) → softmax → matmul (attn,v) → concatenate_heads`.
The importer collapses this cluster into one KernelSpec named
`<model>.attention`; it does not import the four TTIR ops as four
separate kernels.

## Codegen contract (what the compiler emits)

For each KernelOp in a materialized tt_pk Program, codegen emits one
`ttnn.generic` op. The emitted text is a module-scope snippet:

```mlir
module {
  func.func private @qwen36_qkv_proj_kernel() { /* kernel body */ }
  func.func @decoder_layer(...) -> ... {
    %0 = ttnn.generic {
      program = #ttnn.generic_program<
        kernels = [#ttnn.kernel_decl<@qwen36_qkv_proj_kernel, cores = [[3,0], ...]>],
        cbs = [...],
        semaphores = [...]
      >
    } : (...) -> ...
    ...
  }
}
```

Fields the compiler fills in from `ResidentSchedule`:

| Field | Source |
|---|---|
| `kernel_decl.symbol` | KernelOp name + `_kernel` suffix |
| `kernel_decl.cores` | KernelOp core_set, same (x,y) list |
| `generic.operands` | ResidentOp.consumes → SSA refs |
| `generic.results` | ResidentOp.produces → SSA ref |

Fields now filled in:

- `cbs` (CP7a): one `kernel_cb` per consumed tensor + one weight CB +
  one output CB, sized as `double_buffer * ceil(K or N / 32) * 2112
  bytes` for bf16. Other dtypes raise.
- `semaphores` (CP7a): one `kernel_semaphore` per intra-layer
  producer→consumer edge, dense ids starting at 0,
  `initial_value = 0`. Each ttnn.generic op carries `waits` and
  `signals` id lists.
- `compute_kernel` policy fields (CP7b): `math_fidelity`,
  `fp32_dest_acc_en`, `dst_full_sync_en`, `bfp8_pack_precise`,
  `math_approx_mode` come from `KernelSpec.compute_config`, falling
  back to `HardwareSpec.default_compute_config`. The codegen never
  invents these; a kernel without either errors in `lower_to_resident`.
- Compute-kernel C++ body (CP8): `KernelSpec.compute_kernel_source`
  (raw text, or a path read by `tt_pk.load_kernel_source`) is carried
  through `ResidentOp` to the codegen. When present, codegen emits
  `#ttnn.source_compute_kernel<source = "...">` with the body MLIR-
  escaped and drops the private function declaration. Without a body
  the kernel stays on `#ttnn.compute_kernel<symbol_ref = @...>`.
- Reader/writer kernel C++ bodies (CP9):
  `KernelSpec.reader_kernel_source` and
  `KernelSpec.writer_kernel_source` are carried through `ResidentOp`
  and emitted as `#ttnn.source_read_kernel` / `#ttnn.source_write_kernel`
  entries in the same `program.kernels` list as the compute kernel,
  on the same core ranges. Either side can be omitted independently.
- Multi-dtype CB sizing (CP10): `derive_cbs` looks the per-tile byte
  count up by `op.dtype` from a table mirroring tt-mlir's
  `ttcore::getTileSizeBytes`. Dtype spellings are the TTNN parser's
  own (`f32`, `bf16`, `bfp_bf8`, `bfp_bf4`, `u16`, `si32`, `i1`, ...);
  the CB's `dtype` field now echoes `op.dtype` directly. An unknown
  dtype raises `NotImplementedError`.

Fields still NOT filled in (CP11+):
- Layer-boundary semaphores for `x_norm` / `x_norm2` (host-fed).
- Coalescing contiguous cores into bigger `CoreRange` attrs; the
  codegen emits one 1x1 range per worker today.

## Round-trip status

- Textual tt_pk Program → tt-pk-opt → VERIFY: works (CP2).
- AnalysisProblem → CandidatePlan → tt_pk Program: works (CP3).
- CandidatePlan → ResidentSchedule → plan.json: works (CP4, CP5b).
- ResidentSchedule → emitted ttnn text snippet: works (CP6, CP7a).
- Emitted ttnn text → in-tree `ttmlir-opt` round-trip: works (CP7b, see
  `tt_pk.ttnn_lint`). Each emitted attribute is parsed back through
  the submodule's own `ttmlir-opt` binary.
- Compute-kernel C++ body import → `#ttnn.source_compute_kernel`
  round-trip: works (CP8, see `test_kernel_source.py`). The fixture
  at `python/testdata/kernels/qwen36_qkv_proj.cpp` is the in-tree
  reference body.
- Reader/writer kernel C++ body import →
  `#ttnn.source_read_kernel` / `#ttnn.source_write_kernel` round-trip:
  works (CP9, same test). Fixtures at
  `python/testdata/kernels/qwen36_qkv_proj_reader.cpp` and
  `..._writer.cpp`.
- Per-dtype CB page sizing → `#ttnn.kernel_cb_format` round-trip across
  all TTNN DataType spellings: works (CP10, see
  `test_multi_dtype_cb.py`). `bfp_bf8` and `f32` snippets parse back
  through the submodule's `ttmlir-opt`.

To rebuild the in-tree binary from a fresh checkout:

```
scripts/build_ttmlir.sh
```

This stages a project-local toolchain at `.ttmlir-toolchain/` and
builds `third_party/tt-mlir/build/bin/ttmlir-opt` with
`TTMLIR_ENABLE_RUNTIME=OFF`, so no tt-metal link is required.

## Not covered (CP11+)

- Compiler-generated kernel bodies (CP8/CP9 import external C++; they
  do not synthesize one).
- Semaphore graph lowered from async-live-range analysis (D9).
- Multi-chip / mesh codegen (needs tt-mlir mesh attrs).
- ttir → tt-pk importer that actually parses MLIR text (needs Python
  binding to tt-mlir dialect registry).
