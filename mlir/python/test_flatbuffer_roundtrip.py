"""M1 plumbing gate: emitted TTNN text → .ttnn flatbuffer.

Pipes `emit_ttnn_generic` output through the in-tree
`ttmlir-opt --ttcore-register-device` and
`ttmlir-translate --ttnn-to-flatbuffer` to prove the module shape is
ingestible by the real tt-mlir backend, not only by the attr-level lint.

Requires `scripts/build_ttmlir.sh` to have produced ttmlir-opt and
ttmlir-translate under `third_party/tt-mlir/build/bin/`.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import emit_ttnn_generic, greedy_search, lower_to_resident
from tt_pk.qwen36 import qwen36_27b_problem


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr}")
    return r


def main() -> None:
    if not os.path.exists(_OPT) or not os.path.exists(_XLATE):
        print(f"[SKIP] ttmlir-opt / ttmlir-translate not built at "
              f"{os.path.dirname(_OPT)}")
        return

    schedule = lower_to_resident(qwen36_27b_problem(),
                                 greedy_search(qwen36_27b_problem()))
    text = emit_ttnn_generic(schedule)

    with tempfile.TemporaryDirectory() as tmp:
        emit_mlir = os.path.join(tmp, "emit.mlir")
        registered = os.path.join(tmp, "registered.mlir")
        flatbuf = os.path.join(tmp, "out.ttnn")
        with open(emit_mlir, "w") as f:
            f.write(text)

        _run([_OPT, "--ttcore-register-device", emit_mlir, "-o", registered])
        _check(os.path.getsize(registered) > 0,
               "ttmlir-opt --ttcore-register-device produces non-empty IR")

        _run([_XLATE, "--ttnn-to-flatbuffer", registered, "-o", flatbuf])
        size = os.path.getsize(flatbuf)
        _check(size > 0,
               f"ttmlir-translate emits a non-empty flatbuffer ({size} B)")

    print("OK")


if __name__ == "__main__":
    main()
