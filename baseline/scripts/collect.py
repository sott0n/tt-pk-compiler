#!/usr/bin/env python3
"""Collapse one run's JSONL benches into a flat summary.csv for comparison."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys


def flatten(record: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in record.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        print(f"not a dir: {args.run_dir}", file=sys.stderr)
        return 2

    rows = []
    for jsonl in sorted(args.run_dir.glob("*.jsonl")):
        cell = jsonl.stem
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({"cell": cell, **flatten(rec)})

    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    out = args.out or args.run_dir / "summary.csv"
    cols = sorted({k for r in rows for k in r})
    cols = ["cell", *[c for c in cols if c != "cell"]]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
