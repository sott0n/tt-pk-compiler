"""M3 gate 1: 1-tile row-RMSnorm on silicon.

Per row of a 32x32 bf16 tile: sq = x*x, mean = row_sum(sq)/32,
out = x * rsqrt(mean). Scaler tile carries 1/32 so reduce_tile
produces the mean directly. Eps omitted; inputs are scaled so
mean(x^2) stays well above zero.
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


# bf16 1/32 = 0x3D00; scaler pattern fills row 0 and row 16 of the
# tile (face row 0 of each of 4 faces), matching tt-metal's reduce
# scaler expected layout.
_READER_RMS = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint16_t BF16_INV32 = 0x3D00;
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t in_bytes = get_tile_size(cb_in);

    cb_reserve_back(cb_scaler, 1);
    {
        uint32_t addr = get_write_ptr(cb_scaler);
        volatile tt_l1_ptr uint16_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t i = 0; i < 1024; ++i) p[i] = 0;
        for (uint32_t k = 0; k < 4; ++k) {
            uint32_t base = k << 8;
            for (uint32_t j = 0; j < 16; ++j) p[base + j] = BF16_INV32;
        }
    }
    cb_push_back(cb_scaler, 1);

    cb_reserve_back(cb_in, 1);
    uint64_t in_noc = get_noc_addr(my_x[noc_index], my_y[noc_index], in_addr);
    noc_async_read(in_noc, get_write_ptr(cb_in), in_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_in, 1);
}
"""


_COMPUTE_RMS = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sumsq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);

    compute_kernel_hw_startup(cb_in, cb_in, cb_sq);

    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_scaler, 1);

    cb_reserve_back(cb_sq, 1);
    tile_regs_acquire();
    binary_op_init_common(cb_in, cb_in, cb_sq);
    mul_tiles_init(cb_in, cb_in);
    mul_tiles(cb_in, cb_in, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sq);
    tile_regs_release();
    cb_push_back(cb_sq, 1);

    cb_wait_front(cb_sq, 1);

    cb_reserve_back(cb_sumsq, 1);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_sumsq);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, 0, 0, 0);
    reduce_uninit();
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sumsq);
    tile_regs_release();
    cb_push_back(cb_sumsq, 1);

    cb_wait_front(cb_sumsq, 1);

    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    mul_bcast_cols_init(cb_in, cb_sumsq);
    mul_tiles_bcast<BroadcastType::COL>(cb_in, cb_sumsq, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);

    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_sq, 1);
    cb_pop_front(cb_sumsq, 1);
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


def _emit_fb(out_dir):
    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)
    core = ((0, 0),)
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 2))
    writer_ct = (("cb_buffer_index", 1),)
    compute_ct = (("cb_buffer_index", 0),  # in
                  ("cb_buffer_index", 2),  # scaler
                  ("cb_buffer_index", 3),  # sq
                  ("cb_buffer_index", 4),  # sumsq / rsqrt
                  ("cb_buffer_index", 1))  # out
    reader_rt = (("address_of_tensor", 0),)
    writer_rt = (("address_of_tensor", 1),)
    entries = (
        KernelEntry("reader", _READER_RMS, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", av._WRITER_L1, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", _COMPUTE_RMS, core,
                    compute_config=cfg, ct_args=compute_ct),
    )
    cbs = (
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048, tile_count=2, cores=core),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048, tile_count=2, cores=core),
        CBSpec(buffer_index=2, dtype="bf16", page_size=2048, tile_count=1, cores=core),
        CBSpec(buffer_index=3, dtype="bf16", page_size=2048, tile_count=1, cores=core),
        CBSpec(buffer_index=4, dtype="bf16", page_size=2048, tile_count=1, cores=core),
    )
    op = ResidentOp(
        kernel_name="rmsnorm.verify",
        cores=core,
        consumes=("x",), produces="y",
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="rmsnorm_v", hardware="bh_p150", ops=(op,)),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "rms.mlir")
    reg = os.path.join(out_dir, "rms_reg.mlir")
    fb = os.path.join(out_dir, "rms.ttnn")
    with open(mlir, "w") as f:
        f.write(text)
    sysdesc_arg = "--ttcore-register-device"
    sysdesc_env = os.environ.get("SYSTEM_DESC_PATH")
    if sysdesc_env:
        sysdesc_arg = f"--ttcore-register-device=system-desc-path={sysdesc_env}"
    _run([_OPT, sysdesc_arg, mlir, "-o", reg])
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    return fb


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

    torch.manual_seed(0)
    # Pick magnitudes where mean(x^2) is O(1): no eps needed, rsqrt stays sane.
    x = torch.randn(32, 32).to(torch.bfloat16)
    xf = x.float()
    mean_sq = (xf * xf).mean(dim=-1, keepdim=True)
    ref = (xf * torch.rsqrt(mean_sq)).to(torch.bfloat16)

    program.populate_inputs(lambda *_a, **_k: None, golden_inputs=[x])
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
    print(f"[verify] max_abs={max_abs:.3e} pcc={pcc:.6f} shape={tuple(got.shape)}")
    print(f"[verify] ref[0,:4]={ref_f[0,:4].tolist()}")
    print(f"[verify] got[0,:4]={got_f[0,:4].tolist()}")
    if max_abs < 2e-2 and pcc > 0.999:
        print("[PASS] rmsnorm within 2e-2 / pcc>0.999")
    else:
        print("[FAIL] rmsnorm numerical")
        sys.exit(1)


if __name__ == "__main__":
    main()
