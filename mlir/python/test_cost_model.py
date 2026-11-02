"""Cost-model regression: attention gets a meaningful share.

Guards two properties the dummy k*n weighting failed on:
  - attention receives > 1 core at the Qwen3.6-27B shape on bh_p150
  - reserved cores are excluded from any kernel's placement
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import greedy_search
from tt_pk.qwen36 import qwen36_27b_problem


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    problem = qwen36_27b_problem()
    plan = greedy_search(problem)
    by_name = {a.kernel_name: a for a in plan.assignments}

    attn = by_name["qwen36.attention"]
    _check(len(attn.cores) >= 8,
           f"attention gets >= 8 cores (got {len(attn.cores)})")

    mlp_down = by_name["qwen36.mlp_down"]
    _check(len(attn.cores) <= len(mlp_down.cores),
           "attention does not out-allocate mlp_down (compute-dominant)")

    reserved = set(problem.hardware.reserved_cores)
    for a in plan.assignments:
        for c in a.cores:
            _check(c not in reserved,
                   f"{a.kernel_name} does not land on reserved core {c}")

    print("OK")


if __name__ == "__main__":
    main()
