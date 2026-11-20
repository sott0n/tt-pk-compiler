"""M5 gate: toy prefill + decode + handoff on BH.

Two ttnn.generic programs sharing a cache:
  prefill(x_ctx, Wk, Wv)  -> K_cache, V_cache
  decode (x_tok, Wq, K_cache, V_cache, Wo) -> y
K_cache / V_cache ride host memory between calls (the handoff).
Captures M5's prefill+decode+handoff shape with toy math; a true
PP-prefill + TP-decode split is M-later.
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

import test_matmul_verify as mm
import test_attention_toy_verify as at
import test_gdn_toy_verify as gd


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr}")


def _emit_prefill(out_dir, cfg):
    core = ((0, 0),)
    ops = (
        gd._two_in_op("pre.k", ("x_ctx", "Wk"), "K", core, cfg, mm._COMPUTE_MATMUL),
        gd._two_in_op("pre.v", ("x_ctx", "Wv"), "V", core, cfg, mm._COMPUTE_MATMUL),
    )
    # Multi-output: our codegen returns only the LAST op's produces. For
    # M5, run two single-output prefill programs instead (one K, one V).
    return ops


def _emit_single_mm(model_name, consumes, produces, out_dir, cfg, suffix):
    core = ((0, 0),)
    ops = (
        gd._two_in_op("op", consumes, produces, core, cfg, mm._COMPUTE_MATMUL),
    )
    text = emit_ttnn_generic(ResidentSchedule(
        model_name=model_name, hardware="bh_p150", ops=ops),
        operand_memspace="l1_block_sharded")
    mlir = os.path.join(out_dir, f"{suffix}.mlir")
    reg = os.path.join(out_dir, f"{suffix}_reg.mlir")
    fb = os.path.join(out_dir, f"{suffix}.ttnn")
    with open(mlir, "w") as f:
        f.write(text)
    sysdesc_arg = "--ttcore-register-device"
    sysdesc_env = os.environ.get("SYSTEM_DESC_PATH")
    if sysdesc_env:
        sysdesc_arg = f"--ttcore-register-device=system-desc-path={sysdesc_env}"
    _run([_OPT, sysdesc_arg, mlir, "-o", reg])
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    return fb


def _emit_decode(out_dir, cfg):
    core = ((0, 0),)
    # Decode attends to the cached K/V delivered by the prefill programs:
    #   Q = x_tok @ Wq,  S = Q @ K_cache^T,  P = softmax(S),
    #   O = P @ V_cache, y = O @ Wo.
    ops = (
        at._mm_op("dec.q",  ("x_tok", "Wq"), "Q",  core, cfg, mm._COMPUTE_MATMUL),
        at._mm_op("dec.s",  ("Q", "K_cache"), "S",  core, cfg, at._COMPUTE_MATMUL_T),
        at._softmax_op("dec.p", ("S",), "P", core, cfg),
        at._mm_op("dec.o",  ("P", "V_cache"), "O",  core, cfg, mm._COMPUTE_MATMUL),
        at._mm_op("dec.y",  ("O", "Wo"), "y",  core, cfg, mm._COMPUTE_MATMUL),
    )
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="decode", hardware="bh_p150", ops=ops),
        operand_memspace="l1_block_sharded")
    mlir = os.path.join(out_dir, "decode.mlir")
    reg = os.path.join(out_dir, "decode_reg.mlir")
    fb = os.path.join(out_dir, "decode.ttnn")
    with open(mlir, "w") as f:
        f.write(text)
    sysdesc_arg = "--ttcore-register-device"
    sysdesc_env = os.environ.get("SYSTEM_DESC_PATH")
    if sysdesc_env:
        sysdesc_arg = f"--ttcore-register-device=system-desc-path={sysdesc_env}"
    _run([_OPT, sysdesc_arg, mlir, "-o", reg])
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    return fb


def _torch_ref(x_ctx, x_tok, Wq, Wk, Wv, Wo):
    import torch
    def bf(t): return t.to(torch.bfloat16)
    K = bf(torch.matmul(x_ctx.float(), Wk.float()))
    V = bf(torch.matmul(x_ctx.float(), Wv.float()))
    Q = bf(torch.matmul(x_tok.float(), Wq.float()))
    S = bf(torch.matmul(Q.float(), K.float().T))
    P = bf(torch.softmax(S.float(), dim=-1))
    O = bf(torch.matmul(P.float(), V.float()))
    y = bf(torch.matmul(O.float(), Wo.float()))
    return K, V, y


def _run_prog(fb_path, inputs_torch, logger, fm):
    import ttrt, ttrt.binary, ttrt.runtime
    from ttrt.common.util import (
        Binary, create_tensor, convert_input_layouts,
        convert_runtime_to_torch_tensor,
    )
    bin_obj = Binary(logger, fm, fb_path)
    bin_obj.check_version()
    program = bin_obj.get_program(0)
    program.populate_inputs(lambda *_a, **_k: None, golden_inputs=inputs_torch)
    import torch as T
    program.populate_outputs(lambda shape, dtype: T.zeros(shape, dtype=dtype))
    msh = program.mesh_shape
    inputs = [create_tensor(ts, msh) for ts in program.input_tensors]
    mo = ttrt.runtime.MeshDeviceOptions()
    mo.mesh_shape = msh
    dev = ttrt.runtime.open_mesh_device(mo)
    try:
        inputs = convert_input_layouts(dev, inputs, bin_obj.fbb, 0)
        outs = ttrt.runtime.submit(dev, bin_obj.fbb, 0, inputs)
        ttrt.runtime.wait(outs)
        host = ttrt.runtime.to_host(outs[0], untilize=True)
        got = convert_runtime_to_torch_tensor(host[0])
    finally:
        ttrt.runtime.close_mesh_device(dev)
    return got


def main():
    import torch
    import ttrt
    from ttrt.common.util import FileManager, Logger

    out_dir = os.environ.get("VERIFY_OUT_DIR") or tempfile.mkdtemp()
    os.makedirs(out_dir, exist_ok=True)

    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)

    k_fb = _emit_single_mm("prefill_k", ("x_ctx", "Wk"), "K", out_dir, cfg, "prefill_k")
    v_fb = _emit_single_mm("prefill_v", ("x_ctx", "Wv"), "V", out_dir, cfg, "prefill_v")
    d_fb = _emit_decode(out_dir, cfg)
    print(f"[emit] prefill_k={os.path.getsize(k_fb)} B")
    print(f"[emit] prefill_v={os.path.getsize(v_fb)} B")
    print(f"[emit] decode   ={os.path.getsize(d_fb)} B")

    torch.manual_seed(0)
    x_ctx = torch.randn(32, 32).to(torch.bfloat16)
    x_tok = torch.randn(32, 32).to(torch.bfloat16)
    Wq = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wk = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wv = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wo = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    K_ref, V_ref, y_ref = _torch_ref(x_ctx, x_tok, Wq, Wk, Wv, Wo)

    logger = Logger()
    fm = FileManager(logger)

    K_cache = _run_prog(k_fb, [x_ctx, Wk], logger, fm)
    V_cache = _run_prog(v_fb, [x_ctx, Wv], logger, fm)
    k_err = (K_cache.float() - K_ref.float()).abs().max().item()
    v_err = (V_cache.float() - V_ref.float()).abs().max().item()
    print(f"[prefill] K max_abs={k_err:.3e}")
    print(f"[prefill] V max_abs={v_err:.3e}")

    # Handoff: feed prefill's K, V as decode's cache inputs.
    # Decode function arg order = consumed-but-not-produced across its ops:
    # x_tok, Wq, K_cache, V_cache, Wo.
    y = _run_prog(d_fb, [x_tok, Wq, K_cache, V_cache, Wo], logger, fm)

    y_f, r_f = y.float(), y_ref.float()
    max_abs = (y_f - r_f).abs().max().item()
    g = y_f.reshape(-1); r = r_f.reshape(-1)
    pcc = (g @ r / (g.norm() * r.norm())).item()
    print(f"[decode] max_abs={max_abs:.3e}  pcc={pcc:.6f}")
    print(f"[decode] ref[0,:4]={r_f[0,:4].tolist()}")
    print(f"[decode] got[0,:4]={y_f[0,:4].tolist()}")
    tol = float(os.environ.get("HANDOFF_TOL", "2e-2"))
    if max_abs < tol and pcc > 0.999:
        print(f"[PASS] M5: prefill->handoff->decode within +/-{tol}")
    else:
        print(f"[FAIL] max_abs={max_abs:.3e} tol={tol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
