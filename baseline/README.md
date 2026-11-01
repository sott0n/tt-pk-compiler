# baseline/

Reproducible baseline for the Tensix Persistent Kernel compiler.

Covers plan M0:
- Pin Resident / tt-mlir / tt-metal / firmware / toolchain.
- Snapshot host, driver, tt-smi, env.
- Run existing Resident benches under a declared matrix.
- Store results per-run so later candidates compare apples to apples.

## Layout

| Path | Purpose |
|---|---|
| `pins.yaml` | Declared target commits / firmware floor |
| `matrix.yaml` | Benches × env knobs × trace mode |
| `scripts/capture_pins.sh` | Read live HEADs; mark `-dirty` if working tree is modified |
| `scripts/capture_env.sh` | tt-smi, host, python, filtered env vars |
| `scripts/run_matrix.sh` | Drive the matrix; store JSONL + logs under `results/<run_id>/` |
| `scripts/collect.py` | Flatten one run's JSONL into `summary.csv` |
| `results/` | One directory per run (gitignored) |

## Run

```
# repo locations default to $HOME/tt-qwen-resident and $HOME/tt-mlir
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ) \
  bash baseline/scripts/run_matrix.sh
python3 baseline/scripts/collect.py baseline/results/$RUN_ID
```

Filter cells:
```
CELLS=decode_b1_short,ttft_511 bash baseline/scripts/run_matrix.sh
```

Toggle program-level trace:
```
TRACE=on bash baseline/scripts/run_matrix.sh
```

## Baseline contract

- Only HEADs that match `pins.yaml` count as a reference baseline.
- Any `-dirty` entry in `pins.json` makes the run advisory, not a baseline.
- Separate README numbers (sott0n/tt-qwen-resident) from newly-measured ones.
  README is a target. Local re-measurement is the baseline.

## Not covered yet

- Reader / compute / writer timeline + ring-wait capture (needs Tracy wiring).
- host-fed vs device-fed sampling split (needs matrix dimension when flags land).
- Firmware bundle version detection (`blackhole_fw_bundle` is `unknown`).
