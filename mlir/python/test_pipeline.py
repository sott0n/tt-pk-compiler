"""End-to-end: Qwen3.6-27B config -> AnalysisProblem -> CandidatePlan ->
tt_pk Program -> tt-pk-opt verify."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import assemble, greedy_search, materialize, verify
from tt_pk.qwen36 import qwen36_27b_problem


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    problem = qwen36_27b_problem()
    _check(len(problem.layer.kernels) == 6, "qwen36 layer has 6 kernels")

    plan = greedy_search(problem)
    _check(len(plan.assignments) == 6, "plan assigns every kernel")

    seen_cores: set = set()
    for a in plan.assignments:
        for c in a.cores:
            _check(c not in seen_cores,
                   f"core {c} not reused across kernels")
            seen_cores.add(c)

    kernel_names = {a.kernel_name for a in plan.assignments}
    _check(kernel_names == {k.name for k in problem.layer.kernels},
           "every kernel appears in the plan")

    prog = materialize(plan)
    text = assemble(prog)
    _check("tt_pk.program" in text, "materialize emits tt_pk.program")
    for a in plan.assignments:
        _check(a.kernel_name in text,
               f"materialize mentions {a.kernel_name}")

    result = verify(prog)
    if not result.ok:
        print("--- tt-pk-opt stderr ---")
        print(result.stderr)
    _check(result.ok, "greedy plan passes CoreSetDisjoint")

    print("OK")


if __name__ == "__main__":
    main()
