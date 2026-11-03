#!/bin/bash
# Build the in-tree tt-mlir submodule's ttmlir-opt.
#
# Writes toolchain and build artifacts under the project tree so no
# system-wide paths or out-of-tree checkouts are touched. Expects a real
# python3.12 on PATH or at $TTMLIR_PYTHON_VERSION.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TTMLIR_TOOLCHAIN_DIR="$ROOT/.ttmlir-toolchain"
export TTMLIR_VENV_DIR="$TTMLIR_TOOLCHAIN_DIR/venv"
export TTMLIR_PYTHON_VERSION="${TTMLIR_PYTHON_VERSION:-python3.12}"
export CC="${CC:-/usr/bin/clang-17}"
export CXX="${CXX:-/usr/bin/clang++-17}"
export TTMLIR_ENV_ACTIVATED=1
export PATH="$TTMLIR_TOOLCHAIN_DIR/bin:$TTMLIR_VENV_DIR/bin:$PATH"

SRC="$ROOT/third_party/tt-mlir"
TOOLCHAIN_BUILD="$SRC/env/build"
MAIN_BUILD="$SRC/build"

mkdir -p "$TTMLIR_TOOLCHAIN_DIR"

if [ ! -x "$TTMLIR_VENV_DIR/bin/python3" ]; then
  echo "[build_ttmlir] creating venv at $TTMLIR_VENV_DIR"
  "$TTMLIR_PYTHON_VERSION" -m venv "$TTMLIR_VENV_DIR"
  "$TTMLIR_VENV_DIR/bin/pip" install --upgrade pip
  "$TTMLIR_VENV_DIR/bin/pip" install --no-cache-dir \
    -r "$SRC/env/build-requirements.txt"
fi

# Stage 1: toolchain (LLVM, StableHLO, Shardy, flatbuffers).
echo "[build_ttmlir] stage 1: toolchain"
cmake -S "$SRC/env" -B "$TOOLCHAIN_BUILD" -G Ninja \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$TOOLCHAIN_BUILD"

# Toolchain binaries ship with RUNPATH=$ORIGIN/../lib only, so libz has
# to live under $TTMLIR_TOOLCHAIN_DIR/lib. Symlinking from the system
# location keeps the fix inside this project tree.
for soname in libz.so libz.so.1; do
  src="/usr/lib/x86_64-linux-gnu/$soname"
  dst="$TTMLIR_TOOLCHAIN_DIR/lib/$soname"
  if [ -e "$src" ] && [ ! -e "$dst" ]; then
    ln -s "$src" "$dst"
  fi
done

# Stage 2: ttmlir-opt + ttmlir-translate. Runtime stays off so no tt-metal link.
# ttmlir-translate is needed to lower emitted TTNN text to a .ttnn flatbuffer
# that ttrt (from a separate runtime-on build) can run on hardware.
echo "[build_ttmlir] stage 2: ttmlir-opt + ttmlir-translate"
cmake -S "$SRC" -B "$MAIN_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DTTMLIR_ENABLE_RUNTIME=OFF \
  -DTTMLIR_ENABLE_TTRT=OFF \
  -DTTMLIR_ENABLE_TESTS=OFF \
  -DTTMLIR_ENABLE_OPMODEL_TESTS=OFF \
  -DTTMLIR_ENABLE_ALCHEMIST=OFF \
  -DTTMLIR_ENABLE_EXPLORER=OFF \
  -DTTMLIR_ENABLE_STABLEHLO=OFF
cmake --build "$MAIN_BUILD" --target ttmlir-opt ttmlir-translate

echo "[build_ttmlir] done: $MAIN_BUILD/bin/{ttmlir-opt,ttmlir-translate}"
