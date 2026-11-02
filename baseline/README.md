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

## Resident local patch (ttft)

- `patches/tt-qwen-resident-progress-logs.patch` adds 4 lines of
  per-layer `logger.info` so weight-staging progress reaches stderr.
- Without it, the TTFT bench stays silent long enough for the MCP
  broker's 300 s silence watchdog to kill the job.
- Apply before the ttft cell:
  `git -C $HOME/tt-qwen-resident apply $PWD/baseline/patches/tt-qwen-resident-progress-logs.patch`
- Any run with the patch applied will show `resident` as `-dirty`
  in `pins.json`. That is expected until the change lands upstream.

## Resident local patch (pk-plan hook)

- `patches/tt-qwen-resident-pk-plan-hook.patch` adds a 20-line
  `read_pk_plan()` helper and a `workers` override block.
- Lets `RESIDENT_PK_PROBE=1` log the live `workers_cores` / `hub_core`
  layout and `RESIDENT_PK_PLAN=<json>` replace the hardcoded worker
  rectangle with a tt-pk-compiler-emitted one.
- Apply once:
  `git -C $HOME/tt-qwen-resident apply $PWD/baseline/patches/tt-qwen-resident-pk-plan-hook.patch`
- The hook is behaviorally neutral when the plan matches Resident's
  own `worker_rect` output; see CP5b note at `~/.tt-buddy/notes/tt-pk-compiler.md`.

## Not covered yet

- Reader / compute / writer timeline + ring-wait capture (needs Tracy wiring).
- host-fed vs device-fed sampling split (needs matrix dimension when flags land).
- `accuracy_decode` cell requires a Qwen3.6 `.refpt` (set `RESIDENT_REFPT`).
