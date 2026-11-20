"""M1 completion: toy attention layer on BH, numerically within +/-1e-3.

7-op chain on one core, all 32x32 bf16 tiles:
    Q = x @ Wq,  K = x @ Wk,  V = x @ Wv
    S = Q @ K^T,  P = softmax_rowwise(S),  O = P @ V
    y = O @ Wo
Intermediates stay in L1 via multi-op codegen; torch bf16 reference.
Attention math expressed through primitive kernels; 7-kernel monolith
(tt-qwen-resident's streamer-coupled program) is a separate milestone.
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
import test_softmax_verify as sm


# matmul_init(..., transpose=1) makes matmul_tiles compute A @ B^T.
_COMPUTE_MATMUL_T = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    matmul_init(cb_a, cb_b, 1);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    matmul_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
"""


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr}")


def _mm_op(name, consumes, produces, core, cfg, compute_src):
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1))
    writer_ct = (("cb_buffer_index", 2),)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                  ("cb_buffer_index", 2))
    reader_rt = (("address_of_tensor", 0), ("address_of_tensor", 1))
    writer_rt = (("address_of_tensor", 2),)
    entries = (
        KernelEntry("reader", av._READER_L1, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", av._WRITER_L1, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", compute_src, core,
                    compute_config=cfg, ct_args=compute_ct),
    )
    cbs = (
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048,
               tile_count=2, cores=core),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048,
               tile_count=2, cores=core),
        CBSpec(buffer_index=2, dtype="bf16", page_size=2048,
               tile_count=2, cores=core),
    )
    return ResidentOp(
        kernel_name=name, cores=core,
        consumes=consumes, produces=produces,
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)


def _softmax_op(name, consumes, produces, core, cfg):
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 2))
    writer_ct = (("cb_buffer_index", 1),)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 2),
                  ("cb_buffer_index", 3), ("cb_buffer_index", 4),
                  ("cb_buffer_index", 5), ("cb_buffer_index", 1))
    reader_rt = (("address_of_tensor", 0),)
    writer_rt = (("address_of_tensor", 1),)
    entries = (
        KernelEntry("reader", sm._READER_SM, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", av._WRITER_L1, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", sm._COMPUTE_SM, core,
                    compute_config=cfg, ct_args=compute_ct),
    )
    cbs = (
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048,
               tile_count=2, cores=core),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048,
               tile_count=2, cores=core),
        CBSpec(buffer_index=2, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
        CBSpec(buffer_index=3, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
        CBSpec(buffer_index=4, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
        CBSpec(buffer_index=5, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
    )
    return ResidentOp(
        kernel_name=name, cores=core,
        consumes=consumes, produces=produces,
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)


def _emit_fb(out_dir):
    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)
    core = ((0, 0),)
    ops = (
        _mm_op("attn.q",  ("x",  "Wq"), "Q",  core, cfg, mm._COMPUTE_MATMUL),
        _mm_op("attn.k",  ("x",  "Wk"), "K",  core, cfg, mm._COMPUTE_MATMUL),
        _mm_op("attn.v",  ("x",  "Wv"), "V",  core, cfg, mm._COMPUTE_MATMUL),
        _mm_op("attn.s",  ("Q",  "K"),  "S",  core, cfg, _COMPUTE_MATMUL_T),
        _softmax_op("attn.p",  ("S",),  "P",  core, cfg),
        _mm_op("attn.o",  ("P",  "V"),  "O",  core, cfg, mm._COMPUTE_MATMUL),
        _mm_op("attn.y",  ("O",  "Wo"), "y",  core, cfg, mm._COMPUTE_MATMUL),
    )
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="attn_toy", hardware="bh_p150", ops=ops),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "attn_toy.mlir")
    reg = os.path.join(out_dir, "attn_toy_reg.mlir")
    fb = os.path.join(out_dir, "attn_toy.ttnn")
    with open(mlir, "w") as f:
        f.write(text)
    sysdesc_arg = "--ttcore-register-device"
    sysdesc_env = os.environ.get("SYSTEM_DESC_PATH")
    if sysdesc_env:
        sysdesc_arg = f"--ttcore-register-device=system-desc-path={sysdesc_env}"
    _run([_OPT, sysdesc_arg, mlir, "-o", reg])
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    return fb


def _torch_ref(x, Wq, Wk, Wv, Wo):
    """bf16 reference matching the kernels: step-by-step bf16 rounding."""
    import torch
    def bf(t): return t.to(torch.bfloat16)
    Q = bf(torch.matmul(x.float(), Wq.float()))
    K = bf(torch.matmul(x.float(), Wk.float()))
    V = bf(torch.matmul(x.float(), Wv.float()))
    S = bf(torch.matmul(Q.float(), K.float().T))
    P = bf(torch.softmax(S.float(), dim=-1))
    O = bf(torch.matmul(P.float(), V.float()))
    y = bf(torch.matmul(O.float(), Wo.float()))
    return y


def main():
    import torch
    import ttrt, ttrt.binary, ttrt.runtime
    from ttrt.common.util import (
        Binary, FileManager, Logger, create_tensor, convert_input_layouts,
        convert_runtime_to_torch_tensor,
    )

    out_dir = os.environ.get("VERIFY_OUT_DIR") or tempfile.mkdtemp()
    os.makedirs(out_dir, exist_ok=True)
    fb_path = _emit_fb(out_dir)
    print(f"[emit] {fb_path} ({os.path.getsize(fb_path)} B)")

    logger = Logger()
    fm = FileManager(logger)
    bin_obj = Binary(logger, fm, fb_path)
    bin_obj.check_version()
    program = bin_obj.get_program(0)
    print(f"[meta] num_inputs={program.num_inputs()} num_outputs={program.num_outputs()}")
    for i, inp in enumerate(program.inputs):
        print(f"[meta] input[{i}] shape={inp['desc']['shape']}")

    torch.manual_seed(0)
    # Small magnitudes keep softmax well-conditioned and bf16 matmul error low.
    x  = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wq = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wk = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wv = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wo = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    ref = _torch_ref(x, Wq, Wk, Wv, Wo)

    program.populate_inputs(lambda *_a, **_k: None,
                            golden_inputs=[x, Wq, Wk, Wv, Wo])
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
    diff = (got_f - ref_f).abs()
    max_abs = diff.max().item()
    g = got_f.reshape(-1); r = ref_f.reshape(-1)
    pcc = (g @ r / (g.norm() * r.norm())).item()
    print(f"[verify] max_abs={max_abs:.3e}  pcc={pcc:.6f}  shape={tuple(got.shape)}")
    print(f"[verify] ref[0,:4]={ref_f[0,:4].tolist()}")
    print(f"[verify] got[0,:4]={got_f[0,:4].tolist()}")
    tol = float(os.environ.get("ATTN_TOL", "1e-3"))
    if max_abs < tol and pcc > 0.999:
        print(f"[PASS] M1: toy attention within +/-{tol}")
    else:
        print(f"[FAIL] max_abs={max_abs:.3e} tol={tol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
