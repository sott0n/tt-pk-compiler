"""M2 gate: 4-chip TP mesh attention compiles for QB2.

Reuses the M1 toy attention ops, sets mesh_shape=1x4 on get_device,
and compiles against a mock 4-chip Blackhole system descriptor.
Runtime verification needs real QB2 hardware; this gate is 'compiles'
per INTERFACE.md:214.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk.problem import ComputeKernelConfig
from tt_pk.resident import ResidentSchedule
from tt_pk.codegen import emit_ttnn_generic

import test_attention_toy_verify as av


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr}")


def _emit_fb(out_dir):
    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)
    core = ((0, 0),)
    import test_matmul_verify as mm
    ops = (
        av._mm_op("attn.q",  ("x",  "Wq"), "Q",  core, cfg, mm._COMPUTE_MATMUL),
        av._mm_op("attn.k",  ("x",  "Wk"), "K",  core, cfg, mm._COMPUTE_MATMUL),
        av._mm_op("attn.v",  ("x",  "Wv"), "V",  core, cfg, mm._COMPUTE_MATMUL),
        av._mm_op("attn.s",  ("Q",  "K"),  "S",  core, cfg, av._COMPUTE_MATMUL_T),
        av._softmax_op("attn.p",  ("S",),  "P",  core, cfg),
        av._mm_op("attn.o",  ("P",  "V"),  "O",  core, cfg, mm._COMPUTE_MATMUL),
        av._mm_op("attn.y",  ("O",  "Wo"), "y",  core, cfg, mm._COMPUTE_MATMUL),
    )
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="attn_tp4", hardware="bh_qb2", ops=ops),
        operand_memspace="l1_block_sharded",
        mesh_shape="1x4")

    mlir = os.path.join(out_dir, "attn_tp4.mlir")
    reg = os.path.join(out_dir, "attn_tp4_reg.mlir")
    fb = os.path.join(out_dir, "attn_tp4.ttnn")
    with open(mlir, "w") as f:
        f.write(text)
    # Mock 4-chip Blackhole system desc so ttmlir-opt can resolve the
    # mesh; real QB2 runtime is not yet available on this host.
    _run([_OPT,
          "--ttcore-register-device=mock-system-desc-arch=blackhole mesh-shape=1,4",
          mlir, "-o", reg])
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    return fb


def main():
    out_dir = os.environ.get("VERIFY_OUT_DIR") or tempfile.mkdtemp()
    os.makedirs(out_dir, exist_ok=True)
    fb_path = _emit_fb(out_dir)
    sz = os.path.getsize(fb_path)
    print(f"[emit] {fb_path} ({sz} B)")
    if sz > 0:
        print("[PASS] M2 compile: 7-op attention emits mesh_shape=1x4 flatbuffer")
    else:
        print("[FAIL] empty flatbuffer")
        sys.exit(1)


if __name__ == "__main__":
    main()
