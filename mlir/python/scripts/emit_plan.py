"""Write a Resident plan JSON from an AnalysisProblem + CandidatePlan.

Takes workers_cores as a JSON file produced by a RESIDENT_PK_PROBE run
and bundles it into the plan the Resident patch consumes.

Usage:
  python3 emit_plan.py --probe probe.json --out plan.json
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from tt_pk import greedy_search, lower_to_resident, resident_to_json
from tt_pk.qwen36 import qwen36_27b_problem


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True,
                   help="JSON with workers_cores from RESIDENT_PK_PROBE")
    p.add_argument("--out", required=True, help="destination plan.json")
    args = p.parse_args()

    with open(args.probe) as f:
        probe = json.load(f)
    workers_cores = tuple((int(x), int(y)) for x, y in probe["workers_cores"])
    if len(workers_cores) != 32:
        raise SystemExit(
            f"probe workers_cores has {len(workers_cores)} entries; "
            f"Resident expects WORKERS=32")

    problem = qwen36_27b_problem()
    plan = greedy_search(problem)
    schedule = lower_to_resident(problem, plan, workers_cores=workers_cores)
    with open(args.out, "w") as f:
        f.write(resident_to_json(schedule))
    print(f"wrote {args.out} with {len(workers_cores)} workers_cores "
          f"and {len(schedule.ops)} ops")


if __name__ == "__main__":
    main()
