"""M4 gate: stacked decoder of toy GDN + toy attention layers on BH.

Scales down M4's '48 GDN + 16 attention': uses 3 GDN and 1 attention
layer with residual add between each. Each layer reuses the kernel
primitives validated by M1 (attention) and M3 (GDN). Weights per layer
are distinct DRAM inputs; the residual path threads through all layers.
Not a real decoder: toy GDN math and toy attention math, as those
milestones framed them. M4 is the stacking + residual pattern.
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

import test_add_verify as av
import test_matmul_verify as mm
import test_attention_toy_verify as at
import test_gdn_toy_verify as gd
import test_add_launch as add_mod


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")

# Interleave pattern for the toy stack: 3 GDN per 1 attention mirrors
# the production 48:16 ratio at reduced depth.
_LAYERS = ["gdn", "attn", "gdn", "gdn"]


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr}")


def _attn_layer_ops(prefix, x_name, Wq, Wk, Wv, Wo, out_name, core, cfg):
    return (
        at._mm_op(f"{prefix}.q",  (x_name,  Wq), f"{prefix}_Q",  core, cfg, mm._COMPUTE_MATMUL),
        at._mm_op(f"{prefix}.k",  (x_name,  Wk), f"{prefix}_K",  core, cfg, mm._COMPUTE_MATMUL),
        at._mm_op(f"{prefix}.v",  (x_name,  Wv), f"{prefix}_V",  core, cfg, mm._COMPUTE_MATMUL),
        at._mm_op(f"{prefix}.s",  (f"{prefix}_Q", f"{prefix}_K"), f"{prefix}_S", core, cfg, at._COMPUTE_MATMUL_T),
        at._softmax_op(f"{prefix}.p",  (f"{prefix}_S",), f"{prefix}_P",  core, cfg),
        at._mm_op(f"{prefix}.o",  (f"{prefix}_P", f"{prefix}_V"), f"{prefix}_O", core, cfg, mm._COMPUTE_MATMUL),
        at._mm_op(f"{prefix}.y",  (f"{prefix}_O", Wo), out_name, core, cfg, mm._COMPUTE_MATMUL),
    )


def _gdn_layer_ops(prefix, x_name, S_in, Wq, Wk, Wv, Wg, Wo, out_name, core, cfg):
    return (
        gd._two_in_op(f"{prefix}.q",  (x_name,  Wq), f"{prefix}_Q",  core, cfg, mm._COMPUTE_MATMUL),
        gd._rms_op   (f"{prefix}.qn", (f"{prefix}_Q",), f"{prefix}_Qn", core, cfg),
        gd._two_in_op(f"{prefix}.g",  (x_name,  Wg), f"{prefix}_G",  core, cfg, mm._COMPUTE_MATMUL),
        gd._one_in_op(f"{prefix}.gs", (f"{prefix}_G",), f"{prefix}_Gs", core, cfg, gd._COMPUTE_SIGMOID),
        gd._two_in_op(f"{prefix}.kv", (Wk, Wv), f"{prefix}_KV", core, cfg, mm._COMPUTE_MATMUL),
        gd._two_in_op(f"{prefix}.s",  (S_in, f"{prefix}_KV"), f"{prefix}_S",  core, cfg, add_mod._COMPUTE),
        gd._two_in_op(f"{prefix}.o",  (f"{prefix}_Qn", f"{prefix}_S"), f"{prefix}_O", core, cfg, mm._COMPUTE_MATMUL),
        gd._two_in_op(f"{prefix}.og", (f"{prefix}_O", f"{prefix}_Gs"), f"{prefix}_Og", core, cfg, gd._COMPUTE_MUL),
        gd._two_in_op(f"{prefix}.y",  (f"{prefix}_Og", Wo), out_name, core, cfg, mm._COMPUTE_MATMUL),
    )


def _residual_add(prefix, a_name, b_name, out_name, core, cfg):
    return gd._two_in_op(prefix, (a_name, b_name), out_name, core, cfg, add_mod._COMPUTE)


def _build_schedule(layers, core, cfg):
    ops = []
    cur = "x0"
    weight_names = []
    s_names = []
    for i, kind in enumerate(layers):
        prefix = f"l{i}_{kind}"
        if kind == "attn":
            Wq, Wk, Wv, Wo = (f"{prefix}_Wq", f"{prefix}_Wk",
                              f"{prefix}_Wv", f"{prefix}_Wo")
            weight_names += [Wq, Wk, Wv, Wo]
            branch_out = f"{prefix}_out"
            ops += list(_attn_layer_ops(
                prefix, cur, Wq, Wk, Wv, Wo, branch_out, core, cfg))
        else:
            S_in = f"{prefix}_Sin"
            s_names.append(S_in)
            Wq, Wk, Wv, Wg, Wo = (f"{prefix}_Wq", f"{prefix}_Wk",
                                  f"{prefix}_Wv", f"{prefix}_Wg", f"{prefix}_Wo")
            weight_names += [Wq, Wk, Wv, Wg, Wo]
            branch_out = f"{prefix}_out"
            ops += list(_gdn_layer_ops(
                prefix, cur, S_in, Wq, Wk, Wv, Wg, Wo, branch_out, core, cfg))
        nxt = f"x{i+1}"
        ops.append(_residual_add(f"{prefix}.res", cur, branch_out, nxt, core, cfg))
        cur = nxt
    return tuple(ops), weight_names, s_names, cur


def _emit_fb(out_dir):
    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)
    core = ((0, 0),)
    ops, weights, s_names, final = _build_schedule(_LAYERS, core, cfg)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="decode_stack", hardware="bh_p150", ops=ops),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "stack.mlir")
    reg = os.path.join(out_dir, "stack_reg.mlir")
    fb = os.path.join(out_dir, "stack.ttnn")
    with open(mlir, "w") as f:
        f.write(text)
    sysdesc_arg = "--ttcore-register-device"
    sysdesc_env = os.environ.get("SYSTEM_DESC_PATH")
    if sysdesc_env:
        sysdesc_arg = f"--ttcore-register-device=system-desc-path={sysdesc_env}"
    _run([_OPT, sysdesc_arg, mlir, "-o", reg])
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    return fb, weights, s_names, final


def _torch_layer_attn(x, Wq, Wk, Wv, Wo):
    import torch
    def bf(t): return t.to(torch.bfloat16)
    Q = bf(torch.matmul(x.float(), Wq.float()))
    K = bf(torch.matmul(x.float(), Wk.float()))
    V = bf(torch.matmul(x.float(), Wv.float()))
    S = bf(torch.matmul(Q.float(), K.float().T))
    P = bf(torch.softmax(S.float(), dim=-1))
    O = bf(torch.matmul(P.float(), V.float()))
    return bf(torch.matmul(O.float(), Wo.float()))


def _torch_layer_gdn(x, S_in, Wq, Wk, Wv, Wg, Wo):
    import torch
    def bf(t): return t.to(torch.bfloat16)
    def rmsn(t):
        tf = t.float()
        return bf(tf * tf.pow(2).mean(dim=-1, keepdim=True).rsqrt())
    Q  = bf(torch.matmul(x.float(), Wq.float()))
    Qn = rmsn(Q)
    G  = bf(torch.matmul(x.float(), Wg.float()))
    Gs = bf(torch.sigmoid(G.float()))
    KV = bf(torch.matmul(Wk.float(), Wv.float()))
    S  = bf(S_in.float() + KV.float())
    O  = bf(torch.matmul(Qn.float(), S.float()))
    Og = bf(O.float() * Gs.float())
    return bf(torch.matmul(Og.float(), Wo.float()))


def main():
    import torch
    import ttrt, ttrt.binary, ttrt.runtime
    from ttrt.common.util import (
        Binary, FileManager, Logger, create_tensor, convert_input_layouts,
        convert_runtime_to_torch_tensor,
    )

    out_dir = os.environ.get("VERIFY_OUT_DIR") or tempfile.mkdtemp()
    os.makedirs(out_dir, exist_ok=True)
    fb_path, weight_names, s_names, final_name = _emit_fb(out_dir)
    print(f"[emit] {fb_path} ({os.path.getsize(fb_path)} B)")
    print(f"[emit] final tensor name: {final_name}")

    logger = Logger()
    fm = FileManager(logger)
    bin_obj = Binary(logger, fm, fb_path)
    bin_obj.check_version()
    program = bin_obj.get_program(0)
    print(f"[meta] num_inputs={program.num_inputs()} num_outputs={program.num_outputs()}")

    # Build per-tensor host-side values keyed by name so we can feed them
    # in the exact order the codegen decided.
    torch.manual_seed(0)
    def small():  return (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    def medium(): return torch.randn(32, 32).to(torch.bfloat16)
    tensors = {"x0": medium()}
    for name in weight_names:
        tensors[name] = small()
    for name in s_names:
        tensors[name] = small()

    # Walk torch ref alongside layers.
    x = tensors["x0"]
    for i, kind in enumerate(_LAYERS):
        prefix = f"l{i}_{kind}"
        if kind == "attn":
            y = _torch_layer_attn(x,
                tensors[f"{prefix}_Wq"], tensors[f"{prefix}_Wk"],
                tensors[f"{prefix}_Wv"], tensors[f"{prefix}_Wo"])
        else:
            y = _torch_layer_gdn(x, tensors[f"{prefix}_Sin"],
                tensors[f"{prefix}_Wq"], tensors[f"{prefix}_Wk"],
                tensors[f"{prefix}_Wv"], tensors[f"{prefix}_Wg"],
                tensors[f"{prefix}_Wo"])
        x = (x.float() + y.float()).to(torch.bfloat16)
    ref = x

    # The flatbuffer input list carries names implicitly by position; the
    # codegen collects inputs as consumed-but-not-produced in first-touch
    # order. Reproduce that walk here to assemble the golden vector.
    produced = set()
    input_order = []
    seen = set()
    all_ops, _, _, _ = _build_schedule(_LAYERS, ((0, 0),), None)
    for op in all_ops:
        produced.add(op.produces)
    for op in all_ops:
        for name in op.consumes:
            if name in produced or name in seen:
                continue
            input_order.append(name)
            seen.add(name)
    print(f"[meta] input_order ({len(input_order)}): {input_order[:6]}...")
    golden = [tensors[n] for n in input_order]

    program.populate_inputs(lambda *_a, **_k: None, golden_inputs=golden)
    program.populate_outputs(lambda shape, dtype: torch.zeros(shape, dtype=dtype))

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

    got_f, ref_f = got.float(), ref.float()
    max_abs = (got_f - ref_f).abs().max().item()
    g = got_f.reshape(-1); r = ref_f.reshape(-1)
    pcc = (g @ r / (g.norm() * r.norm())).item()
    print(f"[verify] max_abs={max_abs:.3e}  pcc={pcc:.6f}  shape={tuple(got.shape)}")
    print(f"[verify] ref[0,:4]={ref_f[0,:4].tolist()}")
    print(f"[verify] got[0,:4]={got_f[0,:4].tolist()}")
    tol = float(os.environ.get("STACK_TOL", "5e-2"))
    if max_abs < tol and pcc > 0.999:
        print(f"[PASS] M4: {len(_LAYERS)}-layer stack within +/-{tol}")
    else:
        print(f"[FAIL] max_abs={max_abs:.3e} tol={tol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
