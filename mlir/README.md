# mlir/

Native MLIR components of the Tensix Persistent Compiler.

## Layout

| Path | Purpose |
|---|---|
| `CMakeLists.txt` | Finds LLVM/MLIR, pulls in dialect subdirs |
| `dialect/include/tt_pk/IR/` | TableGen (`.td`) + dialect C++ headers |
| `dialect/lib/IR/` | Dialect + op implementations (including verifiers) |
| `dialect/tools/tt-pk-opt/` | mlir-opt-style driver, registers `tt_pk` |
| `dialect/test/` | Hand-written `.mlir` fixtures for parse+verify |
| `python/tt_pk/` | Python surface: text builder + tt-pk-opt driver |

## LLVM / MLIR source

- Pin via the top-level `third_party/tt-mlir` submodule.
- Build is reused, not duplicated. Resolution order:
  1. `-DLLVM_DIR=.../lib/cmake/llvm` / `-DMLIR_DIR=.../lib/cmake/mlir`
  2. `$TT_PK_LLVM_BUILD_DIR/lib/cmake/{llvm,mlir}`
  3. `third_party/tt-mlir/env/build/llvm-project-prefix/src/llvm-project-build/...`

Example when reusing a sibling tt-mlir checkout's build:

```
TT_PK_LLVM_BUILD_DIR=$HOME/tt-mlir/env/build/llvm-project-prefix/src/llvm-project-build \
  cmake -S mlir -B mlir/build -G Ninja
```

## Build and test

```
TT_PK_LLVM_BUILD_DIR=$HOME/tt-mlir/env/build/llvm-project-prefix/src/llvm-project-build \
  cmake -S mlir -B mlir/build -G Ninja
ninja -C mlir/build tt-pk-opt
mlir/build/dialect/tools/tt-pk-opt/tt-pk-opt mlir/dialect/test/core_set_disjoint_ok.mlir
mlir/build/dialect/tools/tt-pk-opt/tt-pk-opt mlir/dialect/test/qwen36_decode.mlir
python3 mlir/python/test_builder.py
```

The `-fail` test prints a verifier error and exits non-zero on purpose — it
pins the `CoreSetDisjoint` message.

A C shim in `dialect/tools/tt-pk-opt/compat_arc4random.c` lets the pre-built
nix LLVM archives link on glibc 2.35; glibc 2.38+ hosts drop the shim at
preprocess time.
