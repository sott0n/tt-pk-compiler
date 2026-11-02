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

Fields NOT yet filled in (CP6+ scope):

- `cbs`: left as placeholder `[]`; needs memory-size pass.
- `semaphores`: left as placeholder `[]`; needs sync pass.
- Kernel body function content: left as `/* opaque */`; needs CP7 codegen
  that reads Resident runtime kernel sources.

## Round-trip status

- Textual tt_pk Program → tt-pk-opt → VERIFY: works (CP2).
- AnalysisProblem → CandidatePlan → tt_pk Program: works (CP3).
- CandidatePlan → ResidentSchedule → plan.json: works (CP4, CP5b).
- ResidentSchedule → emitted ttnn text snippet: works (this doc).
- Emitted ttnn text → tt-mlir ttmlir-opt verify: NOT WIRED yet (needs
  tt-mlir Python binding or a dedicated tt-pk-opt → ttmlir-opt bridge).

## Not covered (CP7+)

- Real kernel body generation (needs Resident kernel source import).
- CB sizing from `K, N, dtype`.
- Semaphore graph lowered from async-live-range analysis (D9).
- Multi-chip / mesh codegen (needs tt-mlir mesh attrs).
- ttir → tt-pk importer that actually parses MLIR text (needs Python
  binding to tt-mlir dialect registry).
