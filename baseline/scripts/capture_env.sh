#!/usr/bin/env bash
# Snapshot host, driver, firmware, Python, and visible mesh device.
set -euo pipefail

OUT_DIR=${1:?usage: capture_env.sh <out_dir>}
mkdir -p "$OUT_DIR"

{
  echo "# host"
  uname -a
  echo
  echo "# os-release"
  cat /etc/os-release 2>/dev/null || true
  echo
  echo "# python"
  python3 --version 2>&1 || true
  which python3
  echo
  echo "# cpu"
  lscpu 2>/dev/null | head -20 || true
} > "$OUT_DIR/host.txt"

tt-smi -s "$OUT_DIR/tt-smi.json" 2>&1 | tee "$OUT_DIR/tt-smi.log" || true
tt-smi -ls 2>&1 > "$OUT_DIR/tt-smi-devices.txt" || true

{
  env | grep -E '^(TT_|MESH_|RESIDENT_|BENCH_|TRACY|TRACE|HF_)' | sort
} > "$OUT_DIR/env.txt" || true
