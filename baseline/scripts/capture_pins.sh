#!/usr/bin/env bash
# Verify declared pins in baseline/pins.yaml match the HEADs of local checkouts.
# Writes a JSON snapshot to results/<run_id>/pins.json.
set -euo pipefail

RESIDENT=${RESIDENT_DIR:-$HOME/tt-qwen-resident}
TT_METAL=${TT_METAL_DIR:-$RESIDENT/tt-metal}
TT_MLIR=${TT_MLIR_DIR:-$HOME/tt-mlir}
OUT=${1:-/dev/stdout}

head_of() {
  local dir=$1
  [[ -e $dir/.git ]] || { echo "missing"; return; }
  local sha dirty
  sha=$(git -C "$dir" rev-parse HEAD 2>/dev/null) || { echo "missing"; return; }
  dirty=$(git -C "$dir" status --porcelain | wc -l)
  echo "${sha}$([[ $dirty -gt 0 ]] && echo -dirty)"
}

cat > "$OUT" <<EOF
{
  "resident":   {"path": "$RESIDENT", "head": "$(head_of "$RESIDENT")"},
  "tt_metal":   {"path": "$TT_METAL", "head": "$(head_of "$TT_METAL")"},
  "tt_mlir":    {"path": "$TT_MLIR",  "head": "$(head_of "$TT_MLIR")"}
}
EOF
