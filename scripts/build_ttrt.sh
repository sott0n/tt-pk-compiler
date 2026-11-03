#!/bin/bash
# In-tree build of ttrt + runtime (links tt-metal). Separate build dir
# so the opt-only build stays intact.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TTMLIR_TOOLCHAIN_DIR="$ROOT/.ttmlir-toolchain"
export TTMLIR_VENV_DIR="$TTMLIR_TOOLCHAIN_DIR/venv"
export CC="${CC:-/usr/bin/clang-17}"
export CXX="${CXX:-/usr/bin/clang++-17}"
export TTMLIR_ENV_ACTIVATED=1
export PATH="$TTMLIR_TOOLCHAIN_DIR/bin:$TTMLIR_VENV_DIR/bin:$PATH"

SRC="$ROOT/third_party/tt-mlir"
BUILD="$SRC/build-runtime"

if [ ! -x "$TTMLIR_VENV_DIR/bin/python3" ]; then
  echo "[build_ttrt] toolchain venv missing; run scripts/build_ttmlir.sh first" >&2
  exit 1
fi

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DTTMLIR_ENABLE_RUNTIME=ON \
  -DTT_RUNTIME_ENABLE_TTNN=ON \
  -DTTMLIR_ENABLE_TTRT=ON \
  -DTTMLIR_ENABLE_TESTS=OFF \
  -DTTMLIR_ENABLE_OPMODEL_TESTS=OFF \
  -DTTMLIR_ENABLE_ALCHEMIST=OFF \
  -DTTMLIR_ENABLE_EXPLORER=OFF \
  -DTTMLIR_ENABLE_STABLEHLO=OFF \
  -DTTMETAL_DISABLE_PRECOMPILE_HEADERS=ON
cmake --build "$BUILD" --target ttrt

echo "[build_ttrt] done: $BUILD/tools/ttrt/ttrt"
