# tt-pk ↔ tt-mlir interface contract (v2)

Defines what the Tensix Persistent Kernel compiler reads from tt-mlir and
what it lowers back into. Scope is the Qwen3.6-27B decoder layer. Any op
outside this contract is CP6+ scope.

## Responsibility split: compiler vs Agent

Final goal (unchanged): Persistent Kernel 実装のモデルを E2E で出力して
性能を出す。The compiler is one half of the system. An Agent — a separate
Claude session — owns the compute kernel bodies and the perf loop. The
split is fixed:

| Concern | Compiler | Agent |
|---|---|---|
| Nature | parametric generator | performance engineer |
| Scope | multi-model generic | model-specific |
| Cadence | one shot per parameter set | measure → fix iteration |
| Output | initial scaffold (strategy + skeleton .cpp) | final perf landing |
| Authority | none (no verify) | full (free to rewrite .cpp) |
| Legal placement / core_range | yes | no (reads from compiler) |
| CB layout / count / size | yes (initial) | overrides at will |
| Semaphore graph / sync | yes (initial) | overrides at will |
| Multi-chip split axis (TP, PP) | yes | no |
| Step loop structure | yes (skeleton) | rewrites at will |
| Compute kernel body | no (empty region) | yes (full ownership) |
| L1 / BW / legality constraint check | yes (emit-time) | no |
| Hardware run, perf measurement, numerical verify | no | yes |

### Why the split changed from v1

v1 reserved verify authority for the compiler: the Agent picked a legal
body, the compiler re-checked CB shapes / rt_args / core_ranges. That
reservation did not survive first contact with the real workload. Two
things broke it:

- The Agent needs to rewrite step loops, sync order and CB plumbing to
  reach perf. A re-verify pass that demands bit-identical structural
  constraints blocks that iteration.
- The compiler cannot verify numerics without running on hardware, which
  is the Agent's job and the Agent's environment.

So the compiler keeps its four irreducible values and drops the rest.

### The four irreducible values the compiler keeps

1. Automatic placement + budget calculation (manual breaks down).
2. Parametric re-generation (flip one knob, consistent update).
3. Emit-time constraint check (L1 fit, BW, legality).
4. Multi-model coverage (initial scaffold for new models).

Perf-driven variant selection, autotuning kernel internals, reading real
device timings are all Agent scope.

## The two compiler artifacts

For each generation request the compiler emits two artifacts.

### A: strategy MLIR (machine-readable, informational)

- TTNN `ProgramDescriptor` / `MeshProgramDescriptor`-shaped MLIR.
- Contents: role assignment, CB layout, semaphore graph, state buffers,
  weight ring, mesh split axes.
- Status: a **record** of the compiler's initial judgement. The Agent
  may read it; the Agent is free to deviate.

### B: idiomatic C++ skeleton (.cpp files)

- Not EmitC. EmitC is SSA-shaped and not readable by a human writing
  compute math.
- Python-side hand-written templates (`cpp_template.py`).
- Each file contains: step loop + semaphore wait/inc + CB
  reserve/push/wait/pop + role branching. The compute region is a
  commented-out gap:

```cpp
void kernel_main() {
    constexpr uint32_t iters = get_compile_time_arg_val(0);
    constexpr uint32_t sem_q = get_compile_time_arg_val(2);

    for (uint32_t i = 0; i < iters; i++) {
        noc_semaphore_wait_min(q_sem, i + 1);
        cb_wait_front(cb_q, 1);

        // ===== Agent compute here =====
        // ===== end =====

        cb_push_back(cb_out, 1);
        noc_semaphore_inc(next_sem, 1);
    }
}
```

The Agent owns the gap, plus the right to edit the rest when perf
requires.

## Pipeline position

```
TTIR module  --(import)-->  AnalysisProblem  --(strategy)-->  Strategy
    ^                                                             |
    |                                                             v
tt-mlir frontend                                               Params
                                                                  |
                                               +------------------+------------------+
                                               v                                     v
                                     program_mlir.emit                      cpp_template.emit
                                   (artifact A: MLIR)                    (artifact B: .cpp)
```

The compiler never emits backend code directly. Artifact A is MLIR text,
artifact B is C++ text. Both are informational scaffolds. Running the
Program is tt-mlir + tt-metal's job, driven by the Agent after the Agent
has filled in the compute regions.

## Parametric inputs

Strategy knobs the compiler accepts (per topology):

- WORKERS count, FANIN, KV_BLOCK.
- Role assignment table (streamer n, worker m, tail 1, leader 1,
  hub 1, head nv*G, …).
- CB count / size table.
- Step loop structure (fused / split).
- TP axis, PP axis.
- Weight ring size.

The Agent runs these as part of perf engineering: flip a knob, ask the
compiler to re-emit, re-measure.

## Import whitelist (TTIR ops the compiler recognizes)

A decoder layer that uses only these ops is importable. Any other TTIR op
raises `ImporterError` and the layer must be lowered to this set first.

| TTIR op | Qwen3.6 role |
|---|---|
| `ttir.rms_norm` | pre-attention / pre-mlp norm |
| `ttir.linear` | qkv_proj, o_proj |
| `ttir.matmul` | mlp_gate, mlp_up, mlp_down |
| `ttir.softmax` | attention softmax |
| `ttir.concatenate_heads` | attention head merge |
| `ttir.concat` | qkv split, mlp cat |
| `ttir.reshape` / `ttir.permute` | layout only |
| `ttir.embedding` | embed tokens |
| `ttir.all_reduce` | TP sum after mlp_down / o_proj |

Attention imports as the TTIR cluster
`matmul (q,k) → softmax → matmul (attn,v) → concatenate_heads`, collapsed
to one topology node named `<model>.attention`.

## Module layout (v2)

New (Phase 0 scaffolding):

- `params.py` — strategy parameter types.
- `strategy.py` — parametric strategy calculation.
- `cpp_template.py` — idiomatic C++ skeleton generation.
- `program_mlir.py` — artifact A emit.
- `qwen36_topology.py` — tt-qwen-resident-aligned topology.

Rewritten (Phase 1+):

- `qwen36.py` — Transformer 6-kernel → streamer / leader / worker / tail
  / hub / head topology.
- `search.py` — row-major greedy → role-aware + CB budget.

Extended (Phase 1+):

- `problem.py` / `plan.py` / `resident.py` — state buffer, weight ring,
  mesh, role type additions.

Kept as-is:

- `codegen_types.py` (CBSpec / SemSpec).
- `builder.py` (tt_pk dialect textual builder).
- `importer.py` (TTIR whitelist check).

Kept as legacy (do not delete, do not advance):

- `codegen.py` — v1 `ttnn.generic` + `source=""` emitter.
- `kernel_source.py` — v1 external .cpp text loader.
- `ttnn_lint.py` — v1 emit verifier.
- 025c129 real-M1 verify tests
  (`test_attention_*_verify.py`, `test_qk_mm_verify.py`,
  `test_softmax_mtile_verify.py`) — reference computations for the
  Agent side.

## Phases

| Phase | Content |
|---|---|
| 0 | INTERFACE.md v2, scaffold new modules, keep old as legacy |
| 1 | AnalysisProblem extension, qwen36 topology rewrite, params type |
| 2 | Strategy calculation (role + CB + sem + weight ring + state) |
| 3 | Artifact A emit (TTNN Program MLIR) |
| 4 | Artifact B emit (idiomatic C++ skeleton) |
| 5 | Qwen3.6 Transformer-only E2E generate, emit structure verify |
| 6 | GDN topology added |

## Milestones

Old M1-M5 said "compiles and runs", which spans compiler and Agent.
v2 splits each into a compiler-side milestone (scaffolding emit) and an
agent-side milestone (compute + run + verify).

| Milestone | compiler-side | agent-side |
|---|---|---|
| M1 | 1 attention layer: strategy + .cpp skeleton emit (Phase 0-5) | compute + tt-metal build + launch + hw ±1e-3 |
| M2 | 4-chip TP mesh variant of M1 (Phase 5 mesh) | QB2 run |
| M3 | 1 GDN layer: strategy + .cpp skeleton (Phase 6) | true GDN math (gated_deltanet_forward end to end: beta, decay, output gate, rms_norm_gated, conv bias) on hw; ±1e-3 vs tt-qwen-resident, pcc > 0.999 |
| M4 | 48 GDN + 16 attn stacked decode strategy (Phase 6+) | full run + numerics |
| M5 | prefill + decode + handoff strategy (Phase 6+) | handoff impl + full run |

Phase 0-6 is compiler build order. M1-M5 is the joint (compiler +
Agent + hardware) outcome.

## Not covered

- Compiler-synthesized compute bodies. The compute region stays empty;
  the Agent fills it. Compiler-synthesis of compute math is not a goal.
- Compiler-side perf tuning (Agent scope).
- Numerical verify (Agent scope, on hardware).
- Real TTIR parse via tt-mlir Python binding (S2 pre-req, deferred).
- Mesh per-die program nesting in artifact A. Phase 3 emits only the
  `tt_pk.mesh_shape` attribute; the per-die program body is added as a
  Phase 5 expansion for M2.

## Status snapshot (2026-11-21)

- 025c129 landed real-M1 attention PASS at 1/4 scale
  (Qwen3.6 attention cluster math, NUM_HEADS=8, NUM_KV_HEADS=2,
  HEAD_DIM=128, L=2 on single BH p150). max_abs=2.441e-04,
  pcc=0.999987.
- Full 32-head × 8-kv fused kernel hangs at >8 total heads; deadlock
  root cause unresolved under v1 single-kernel design.
- v2 moves compute ownership to the Agent and keeps the compiler on
  strategy + skeleton emit. The 025c129 tests stay as agent-side
  reference.
