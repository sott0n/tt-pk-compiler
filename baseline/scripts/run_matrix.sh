#!/usr/bin/env bash
# Drive the baseline/matrix.yaml cells against tt-qwen-resident and bundle
# JSONL + env snapshot under results/<run_id>/.
#
# Dependencies: yq (for matrix.yaml), the Resident checkout, sourced env.sh.
#
# Env:
#   RESIDENT_DIR   Resident checkout (default: $HOME/tt-qwen-resident)
#   TT_MLIR_DIR    tt-mlir checkout  (default: $HOME/tt-mlir)
#   MESH_DEVICE    Mesh id           (default: from matrix.yaml)
#   RUN_ID         Run id            (default: date-based)
#   CELLS          Comma list to filter cells (optional)
#   TRACE          "on" | "off" (toggle TT_METAL_PROGRAM_TRACE)
set -euo pipefail

BASELINE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$BASELINE_ROOT
RESIDENT=${RESIDENT_DIR:-$HOME/tt-qwen-resident}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/results/$RUN_ID
mkdir -p "$OUT"

command -v yq >/dev/null || { echo "yq required (mikefarah/yq)"; exit 2; }
[[ -d $RESIDENT ]] || { echo "Resident checkout not at $RESIDENT"; exit 2; }

MESH=${MESH_DEVICE:-$(yq -r '.mesh_device' "$ROOT/matrix.yaml")}
export MESH_DEVICE=$MESH

"$ROOT/scripts/capture_pins.sh" "$OUT/pins.json"
"$ROOT/scripts/capture_env.sh"  "$OUT"

cp "$ROOT/pins.yaml"   "$OUT/pins.yaml"
cp "$ROOT/matrix.yaml" "$OUT/matrix.yaml"

# shellcheck disable=SC1091
source "$RESIDENT/scripts/env.sh"
# Resident's env.sh resets $ROOT to its own repo; restore ours for the yq
# lookups below.
ROOT=$BASELINE_ROOT

if [[ ${TRACE:-off} == on ]]; then export TT_METAL_PROGRAM_TRACE=1; else unset TT_METAL_PROGRAM_TRACE; fi

mapfile -t NAMES < <(yq -r '.cells[].name' "$ROOT/matrix.yaml")
if [[ -n ${CELLS:-} ]]; then
  IFS=',' read -r -a FILTER <<< "$CELLS"
  NAMES=( "${FILTER[@]}" )
fi

for name in "${NAMES[@]}"; do
  bench=$(yq -r --arg n "$name" '.cells[] | select(.name == $n) | .bench' "$ROOT/matrix.yaml")
  env_pairs=$(yq -r --arg n "$name" '.cells[] | select(.name == $n) | .env | to_entries | map("\(.key)=\(.value)") | .[]' "$ROOT/matrix.yaml")
  log="$OUT/${name}.log"
  jsonl="$OUT/${name}.jsonl"

  echo "::: $name :::"
  (
    cd "$RESIDENT"
    while IFS= read -r kv; do [[ -z $kv ]] && continue; export "$kv"; done <<< "$env_pairs"
    export BENCH_OUT=$jsonl
    pytest "$bench" 2>&1 | tee "$log"
  ) || echo "cell $name failed" >> "$OUT/failures.txt"
done

echo "run_id=$RUN_ID out=$OUT"
