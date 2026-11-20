"""M3 completion: toy GDN-like layer on BH, numerically within +/-2e-2.

Captures GDN's shape with primitives we have validated, not full math:
    Q = x @ Wq;  Qn = rmsnorm(Q)
    G = x @ Wg;  Gs = sigmoid(G)
    KV = Wk @ Wv              (stands in for k.T @ v outer product)
    S = S_in + KV             (recurrent state update)
    O = Qn @ S                (query times state)
    Og = O * Gs               (element-wise gate)
    y = Og @ Wo               (output projection)
Not GDN math: sigmoid-of-matmul != true GDN gate, and KV != true k.T @ v.
This is the M3 "compiles and runs" gate per INTERFACE.md:215, framed as
GDN-shape rather than GDN-truth.
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
import test_rmsnorm_verify as rms


_COMPUTE_SIGMOID = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    init_sfpu(cb_in, cb_out);
    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile_init(cb_in);
    copy_tile(cb_in, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in, 1);
}
"""


_COMPUTE_MUL = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    binary_op_init_common(cb_a, cb_b, cb_out);
    mul_tiles_init(cb_a, cb_b);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    mul_tiles(cb_a, cb_b, 0, 0, 0);
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


def _two_in_op(name, consumes, produces, core, cfg, compute_src):
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
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048, tile_count=2, cores=core),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048, tile_count=2, cores=core),
        CBSpec(buffer_index=2, dtype="bf16", page_size=2048, tile_count=2, cores=core),
    )
    return ResidentOp(
        kernel_name=name, cores=core,
        consumes=consumes, produces=produces,
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)


def _one_in_op(name, consumes, produces, core, cfg, compute_src):
    reader_ct = (("cb_buffer_index", 0),)
    writer_ct = (("cb_buffer_index", 1),)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1))
    reader_rt = (("address_of_tensor", 0),)
    writer_rt = (("address_of_tensor", 1),)

    reader_src = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t in_bytes = get_tile_size(cb_in);
    cb_reserve_back(cb_in, 1);
    uint64_t in_noc = get_noc_addr(my_x[noc_index], my_y[noc_index], in_addr);
    noc_async_read(in_noc, get_write_ptr(cb_in), in_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_in, 1);
}
"""
    entries = (
        KernelEntry("reader", reader_src, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", av._WRITER_L1, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", compute_src, core,
                    compute_config=cfg, ct_args=compute_ct),
    )
    cbs = (
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048, tile_count=2, cores=core),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048, tile_count=2, cores=core),
    )
    return ResidentOp(
        kernel_name=name, cores=core,
        consumes=consumes, produces=produces,
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)


def _rms_op(name, consumes, produces, core, cfg):
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 2))
    writer_ct = (("cb_buffer_index", 1),)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 2),
                  ("cb_buffer_index", 3), ("cb_buffer_index", 4),
                  ("cb_buffer_index", 1))
    reader_rt = (("address_of_tensor", 0),)
    writer_rt = (("address_of_tensor", 1),)
    entries = (
        KernelEntry("reader", rms._READER_RMS, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", av._WRITER_L1, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", rms._COMPUTE_RMS, core,
                    compute_config=cfg, ct_args=compute_ct),
    )
    cbs = (
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048, tile_count=2, cores=core),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048, tile_count=2, cores=core),
        CBSpec(buffer_index=2, dtype="bf16", page_size=2048, tile_count=1, cores=core),
        CBSpec(buffer_index=3, dtype="bf16", page_size=2048, tile_count=1, cores=core),
        CBSpec(buffer_index=4, dtype="bf16", page_size=2048, tile_count=1, cores=core),
    )
    return ResidentOp(
        kernel_name=name, cores=core,
        consumes=consumes, produces=produces,
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)


def _emit_fb(out_dir):
    import test_add_launch as add_mod
    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)
    core = ((0, 0),)
    ops = (
        _two_in_op("gdn.q",   ("x",  "Wq"), "Q",  core, cfg, mm._COMPUTE_MATMUL),
        _rms_op   ("gdn.qn",  ("Q",),       "Qn", core, cfg),
        _two_in_op("gdn.g",   ("x",  "Wg"), "G",  core, cfg, mm._COMPUTE_MATMUL),
        _one_in_op("gdn.gs",  ("G",),       "Gs", core, cfg, _COMPUTE_SIGMOID),
        _two_in_op("gdn.kv",  ("Wk", "Wv"), "KV", core, cfg, mm._COMPUTE_MATMUL),
        _two_in_op("gdn.s",   ("S_in","KV"),"S",  core, cfg, add_mod._COMPUTE),
        _two_in_op("gdn.o",   ("Qn", "S"),  "O",  core, cfg, mm._COMPUTE_MATMUL),
        _two_in_op("gdn.og",  ("O",  "Gs"), "Og", core, cfg, _COMPUTE_MUL),
        _two_in_op("gdn.y",   ("Og", "Wo"), "y",  core, cfg, mm._COMPUTE_MATMUL),
    )
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="gdn_toy", hardware="bh_p150", ops=ops),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "gdn.mlir")
    reg = os.path.join(out_dir, "gdn_reg.mlir")
    fb = os.path.join(out_dir, "gdn.ttnn")
    with open(mlir, "w") as f:
        f.write(text)
    sysdesc_arg = "--ttcore-register-device"
    sysdesc_env = os.environ.get("SYSTEM_DESC_PATH")
    if sysdesc_env:
        sysdesc_arg = f"--ttcore-register-device=system-desc-path={sysdesc_env}"
    _run([_OPT, sysdesc_arg, mlir, "-o", reg])
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    return fb


def _torch_ref(x, S_in, Wq, Wk, Wv, Wg, Wo):
    import torch
    def bf(t): return t.to(torch.bfloat16)
    def rmsn(t):
        tf = t.float()
        rms = tf.pow(2).mean(dim=-1, keepdim=True).rsqrt()
        return bf(tf * rms)
    Q  = bf(torch.matmul(x.float(), Wq.float()))
    Qn = rmsn(Q)
    G  = bf(torch.matmul(x.float(), Wg.float()))
    Gs = bf(torch.sigmoid(G.float()))
    KV = bf(torch.matmul(Wk.float(), Wv.float()))
    S  = bf(S_in.float() + KV.float())
    O  = bf(torch.matmul(Qn.float(), S.float()))
    Og = bf(O.float() * Gs.float())
    y  = bf(torch.matmul(Og.float(), Wo.float()))
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

    torch.manual_seed(0)
    x    = torch.randn(32, 32).to(torch.bfloat16)
    S_in = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wq   = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wk   = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wv   = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wg   = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    Wo   = (torch.randn(32, 32) * 0.1).to(torch.bfloat16)
    ref  = _torch_ref(x, S_in, Wq, Wk, Wv, Wg, Wo)

    # Function-argument order matches input_order rebuilt by the codegen:
    # consumed-but-not-produced tensors in first-touch order across ops.
    # Our op list consumes in this order: x, Wq, Wg, Wk, Wv, S_in, Wo.
    golden = [x, Wq, Wg, Wk, Wv, S_in, Wo]
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
    tol = float(os.environ.get("GDN_TOL", "2e-2"))
    if max_abs < tol and pcc > 0.999:
        print(f"[PASS] M3: toy GDN within +/-{tol}")
    else:
        print(f"[FAIL] max_abs={max_abs:.3e} tol={tol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
