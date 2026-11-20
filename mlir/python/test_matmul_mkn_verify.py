"""M1e gate 10: 2x2x2-tile bf16 matmul on one core, numerically PASS.

A (64x64) @ B (64x64) = C (64x64) on one core, Mt=Nt=Kt=2.
Full triple-loop matmul kernel over the tile grid; validates writer
striding across multiple output tiles and reader order for A/B tiles.
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


# Reader streams A[mt,kt] and B[kt,nt] in the triple-loop order the
# compute kernel pops them in: for each (mt, nt), push Kt (a_tile, b_tile)
# pairs. Each L1 shard is row-major, so A[mt,kt] sits at mt*Kt+kt and
# B[kt,nt] at kt*Nt+nt.
_READER_MKN = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t Mt = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t Nt = get_compile_time_arg_val(4);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_a);
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_reserve_back(cb_a, 1);
                cb_reserve_back(cb_b, 1);
                uint64_t a_noc = get_noc_addr(
                    my_x[noc_index], my_y[noc_index],
                    a_addr + (mt * Kt + kt) * tile_bytes);
                uint64_t b_noc = get_noc_addr(
                    my_x[noc_index], my_y[noc_index],
                    b_addr + (kt * Nt + nt) * tile_bytes);
                noc_async_read(a_noc, get_write_ptr(cb_a), tile_bytes);
                noc_async_read(b_noc, get_write_ptr(cb_b), tile_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_a, 1);
                cb_push_back(cb_b, 1);
            }
        }
    }
}
"""


_WRITER_MKN = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_wait_front(cb_out, 1);
            uint64_t out_noc = get_noc_addr(
                my_x[noc_index], my_y[noc_index],
                out_addr + (mt * Nt + nt) * tile_bytes);
            noc_async_write(get_read_ptr(cb_out), out_noc, tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
"""


_COMPUTE_MKN = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t Mt = get_compile_time_arg_val(3);
    constexpr uint32_t Kt = get_compile_time_arg_val(4);
    constexpr uint32_t Nt = get_compile_time_arg_val(5);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    matmul_init(cb_a, cb_b);
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_a, 1);
                cb_wait_front(cb_b, 1);
                matmul_tiles(cb_a, cb_b, 0, 0, 0);
                cb_pop_front(cb_a, 1);
                cb_pop_front(cb_b, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    }
}
"""


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")

_MT, _KT, _NT = 2, 2, 2
_M, _K, _N = _MT * 32, _KT * 32, _NT * 32


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
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                 _MT, _KT, _NT)
    writer_ct = (("cb_buffer_index", 2), _MT, _NT)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                  ("cb_buffer_index", 2), _MT, _KT, _NT)
    reader_rt = (("address_of_tensor", 0), ("address_of_tensor", 1))
    writer_rt = (("address_of_tensor", 2),)
    entries = (
        KernelEntry("reader", _READER_MKN, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", _WRITER_MKN, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", _COMPUTE_MKN, core,
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
    op = ResidentOp(
        kernel_name="matmul.mkn",
        cores=core,
        consumes=("a", "b"), produces="c",
        k=_M, n=_N, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs,
        operand_shapes=(("a", (_M, _K)), ("b", (_K, _N)), ("c", (_M, _N))))
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="mm_mkn", hardware="bh_p150", ops=(op,)),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "mm_mkn.mlir")
    reg = os.path.join(out_dir, "mm_mkn_reg.mlir")
    fb = os.path.join(out_dir, "mm_mkn.ttnn")
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
    print(f"[meta] num_inputs={program.num_inputs()} num_outputs={program.num_outputs()}")
    for i, inp in enumerate(program.inputs):
        print(f"[meta] input[{i}] shape={inp['desc']['shape']}")
    for i, o in enumerate(program.outputs):
        print(f"[meta] output[{i}] shape={o['desc']['shape']}")

    torch.manual_seed(0)
    a = (torch.rand(_M, _K, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = (torch.rand(_K, _N, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    ref = torch.matmul(a.float(), b.float()).to(torch.bfloat16)

    program.populate_inputs(lambda *_a, **_k: None, golden_inputs=[a, b])
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
    print(f"[verify] ref[48,60:64]={ref_f[48,60:64].tolist()}")
    print(f"[verify] got[48,60:64]={got_f[48,60:64].tolist()}")
    if pcc > 0.999 and max_abs < 5e-2:
        print("[PASS] pcc>0.999 and max_abs<5e-2")
    else:
        print("[FAIL] numerical bound")
        sys.exit(1)


if __name__ == "__main__":
    main()
