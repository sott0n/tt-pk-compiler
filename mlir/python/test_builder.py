"""Smoke test for the tt_pk textual builder."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import Program, assemble, verify


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    prog = Program()
    cs_a = prog.add_core_set("csa", [(0, 0), (1, 0)])
    cs_b = prog.add_core_set("csb", [(2, 0), (3, 0)])
    prog.launch("gate_up", cs_a)
    prog.launch("down", cs_b)

    text = assemble(prog)
    _check("tt_pk.program" in text, "assemble emits program op")
    _check("tt_pk.core_set" in text, "assemble emits core_set op")

    result = verify(prog)
    _check(result.ok, "disjoint program round-trips through tt-pk-opt")

    overlap = Program()
    cs_x = overlap.add_core_set("x", [(0, 0), (1, 0)])
    cs_y = overlap.add_core_set("y", [(1, 0), (2, 0)])
    overlap.launch("k", cs_x)
    overlap.launch("k", cs_y)
    bad = verify(overlap)
    _check(not bad.ok, "overlapping program is rejected")
    _check("(1,0) already assigned" in bad.stderr,
           "overlap error names the colliding core")

    print("OK")


if __name__ == "__main__":
    main()
