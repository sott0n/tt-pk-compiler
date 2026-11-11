"""M1e gate 0: drive our source_* codegen onto silicon with noop kernels.

Minimal 1-core ttnn.generic — noop reader/writer/compute. No sems, no
rt_args. If this hangs, the gap is in our source_* lowering path itself,
not in attention-specific rt_args.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk.codegen_types import CBSpec
from tt_pk.problem import ComputeKernelConfig, KernelEntry
from tt_pk.resident import ResidentOp, ResidentSchedule
from tt_pk.codegen import emit_ttnn_generic


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr}")


def main():
    if not os.path.exists(_OPT) or not os.path.exists(_XLATE):
        print("[SKIP] ttmlir-opt / ttmlir-translate missing"); return

    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)

    core = ((0, 0),)
    noop = "void kernel_main() {}"
    entries = (
        KernelEntry("reader", noop, core),
        KernelEntry("writer", noop, core),
        KernelEntry("compute", noop, core, compute_config=cfg),
    )
    cbs = (
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
    )
    op = ResidentOp(
        kernel_name="noop.single",
        cores=core,
        consumes=("x",), produces="y",
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="noop_m1e", hardware="bh_p150", ops=(op,)))

    out_dir = os.environ.get("NOOP_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        mlir = os.path.join(out_dir, "noop.mlir")
        reg = os.path.join(out_dir, "noop_reg.mlir")
        fb = os.path.join(out_dir, "noop.ttnn")
    else:
        tmp_ctx = tempfile.TemporaryDirectory()
        mlir = os.path.join(tmp_ctx.name, "noop.mlir")
        reg = os.path.join(tmp_ctx.name, "noop_reg.mlir")
        fb = os.path.join(tmp_ctx.name, "noop.ttnn")

    with open(mlir, "w") as f:
        f.write(text)

    sysdesc_arg = "--ttcore-register-device"
    sysdesc_env = os.environ.get("SYSTEM_DESC_PATH")
    if sysdesc_env:
        sysdesc_arg = f"--ttcore-register-device=system-desc-path={sysdesc_env}"
    _run([_OPT, sysdesc_arg, mlir, "-o", reg])
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    size = os.path.getsize(fb)
    print(f"[PASS] flatbuffer emitted ({size} B)")
    if out_dir:
        print(f"saved: {fb}")
    print("OK")


if __name__ == "__main__":
    main()
