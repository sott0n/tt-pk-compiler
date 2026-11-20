"""M1 gate 11: 1-tile row-softmax on silicon.

Per row of a 32x32 bf16 tile: max, sub, exp, sum, recip, mul.
Reader fills a reduce scaler tile in a dedicated CB; compute uses
reduce_tile<MAX/SUM, REDUCE_ROW> + sub_bcast_cols + exp + recip.
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


# Reduce-scaler tile layout mirrors tt-metal's generate_reduce_scaler
# (dm_utils.hpp): 4 faces of 16x16, first row of each face = scaler,
# rest zero. reduce_tile consumes this in cb_scaler.
_READER_SM = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint16_t BF16_ONE = 0x3F80;
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
            for (uint32_t j = 0; j < 16; ++j) p[base + j] = BF16_ONE;
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


_COMPUTE_SM = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_max = get_compile_time_arg_val(2);
    constexpr uint32_t cb_exps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_max);

    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_scaler, 1);

    cb_reserve_back(cb_max, 1);
    tile_regs_acquire();
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_max);
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, 0);
    reduce_uninit();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_max);
    tile_regs_release();
    cb_push_back(cb_max, 1);

    cb_wait_front(cb_max, 1);

    cb_reserve_back(cb_exps, 1);
    tile_regs_acquire();
    sub_bcast_cols_init(cb_in, cb_max);
    sub_tiles_bcast<BroadcastType::COL>(cb_in, cb_max, 0, 0, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_exps);
    tile_regs_release();
    cb_push_back(cb_exps, 1);

    cb_wait_front(cb_exps, 1);

    cb_reserve_back(cb_sum, 1);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exps, cb_scaler, cb_sum);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exps, cb_scaler, 0, 0, 0);
    reduce_uninit();
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sum);
    tile_regs_release();
    cb_push_back(cb_sum, 1);

    cb_wait_front(cb_sum, 1);

    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    mul_bcast_cols_init(cb_exps, cb_sum);
    mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_sum, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);

    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_max, 1);
    cb_pop_front(cb_exps, 1);
    cb_pop_front(cb_sum, 1);
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
    # cb_in is backed by the input tensor; cb_out by the output tensor.
    # cb_scaler/max/exps/sum are internal, no tensor backing.
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 2))
    writer_ct = (("cb_buffer_index", 1),)
    compute_ct = (("cb_buffer_index", 0),  # cb_in
                  ("cb_buffer_index", 2),  # cb_scaler
                  ("cb_buffer_index", 3),  # cb_max
                  ("cb_buffer_index", 4),  # cb_exps
                  ("cb_buffer_index", 5),  # cb_sum
                  ("cb_buffer_index", 1))  # cb_out
    reader_rt = (("address_of_tensor", 0),)
    writer_rt = (("address_of_tensor", 1),)
    entries = (
        KernelEntry("reader", _READER_SM, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", av._WRITER_L1, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", _COMPUTE_SM, core,
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
    op = ResidentOp(
        kernel_name="softmax.verify",
        cores=core,
        consumes=("x",), produces="y",
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="softmax_v", hardware="bh_p150", ops=(op,)),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "sm.mlir")
    reg = os.path.join(out_dir, "sm_reg.mlir")
    fb = os.path.join(out_dir, "sm.ttnn")
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
    x = (torch.rand(32, 32, dtype=torch.float32) * 4 - 2).to(torch.bfloat16)
    ref = torch.softmax(x.float(), dim=-1).to(torch.bfloat16)

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
    row_sum_got = got_f.sum(dim=-1)
    print(f"[verify] max_abs={max_abs:.3e}  shape={tuple(got.shape)}")
    print(f"[verify] row sums (got): min={row_sum_got.min().item():.4f} max={row_sum_got.max().item():.4f}")
    print(f"[verify] ref[0,:4]={ref_f[0,:4].tolist()}")
    print(f"[verify] got[0,:4]={got_f[0,:4].tolist()}")
    if max_abs < 1e-2:
        print("[PASS] numerical 1e-2")
    else:
        print("[FAIL] numerical 1e-2")
        sys.exit(1)


if __name__ == "__main__":
    main()
