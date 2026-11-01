# mlir/

Native MLIR components of the Tensix Persistent Compiler.

## Layout

| Path | Purpose |
|---|---|
| `CMakeLists.txt` | Finds LLVM/MLIR, pulls in dialect subdirs |
| `dialect/` | `tt_pk` dialect sources (ops, attributes, verifiers) |

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

## Configure

Configure alone is enough to confirm the LLVM/MLIR packages were discovered.
Build targets appear starting at CP2.
