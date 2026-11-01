# tt-pk-compiler

Tensix Persistent Compiler.

Builds a single Plan — placement, layout, DRAM/L1 memory, execution order,
synchronization — from a model, workload and hardware spec, then lowers it
through a `tt_pk` MLIR dialect to existing Resident-style kernels and
`ttnn.generic_op` descriptors.

Reference implementation: [tt-qwen-resident](https://github.com/sott0n/tt-qwen-resident).

## Layout

| Path | Purpose |
|---|---|
| `baseline/` | Pin, env capture, matrix driver — see `baseline/README.md` |
| `mlir/` | `tt_pk` dialect and passes — see `mlir/README.md` |
| `third_party/tt-mlir` | Pinned tt-mlir submodule |

## Pins

See `baseline/pins.yaml`. Current:
- `tt-mlir` @ `f9dc1b3e5101639a8d582c449117bdae26c8129f`
- `tt-metal` @ `816841ddc93511b042267684c2fdcbb4974b3bec`
- Blackhole firmware bundle `19.2.0`

## First-time setup

```
git submodule update --init --recursive
TT_PK_LLVM_BUILD_DIR=$HOME/tt-mlir/env/build/llvm-project-prefix/src/llvm-project-build \
  cmake -S mlir -B mlir/build -G Ninja
ninja -C mlir/build
```

The `TT_PK_LLVM_BUILD_DIR` fallback reuses a sibling tt-mlir build. See
`mlir/README.md` for other resolution options.
