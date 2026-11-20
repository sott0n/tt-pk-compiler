"""Real-M1 step 2: S = Q @ K^T at head_dim=128 on BH.

Q [32, 128], K [L*32, 128], S = Q @ K^T [32, L*32].
Mt=1, Kt=4 (head_dim), Nt=L (keys). matmul_init transpose_b=1
reads K tiles as their transpose on the fly. Scale 1/sqrt(128)
is folded into Q on the host, matching qwen36's
rmsnorm(q, wq/sqrt(HD)) convention.
"""

import os
import subprocess
import sys
import tempfile
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk.codegen_types import CBSpec
from tt_pk.problem import ComputeKernelConfig, KernelEntry
from tt_pk.resident import ResidentOp, ResidentSchedule
from tt_pk.codegen import emit_ttnn_generic


_HEAD_DIM = 128
_L = 2
_SEQ_Q = 32
_SEQ_K = _L * 32
_Mt = 1
_Kt = _HEAD_DIM // 32
_Nt = _L


# Q[mt,kt] and K[nt,kt] both index as (row * Kt + kt). Reader streams
# (q_tile, k_tile) pairs in the compute kernel's mt/nt/kt order.
_READER_QK = r"""
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
                    b_addr + (nt * Kt + kt) * tile_bytes);
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


# matmul_init(cb_a, cb_b, 1) sets transpose_b: each K tile is read as
# its transpose, so matmul_tiles computes the Q @ K^T contraction.
_COMPUTE_QK = r"""
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
    matmul_init(cb_a, cb_b, 1);
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
                 _Mt, _Kt, _Nt)
    writer_ct = (("cb_buffer_index", 2), _Mt, _Nt)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                  ("cb_buffer_index", 2), _Mt, _Kt, _Nt)
    reader_rt = (("address_of_tensor", 0), ("address_of_tensor", 1))
    writer_rt = (("address_of_tensor", 2),)
    entries = (
        KernelEntry("reader", _READER_QK, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", _WRITER_MKN, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", _COMPUTE_QK, core,
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
        kernel_name="qk.mm",
        cores=core,
        consumes=("q", "k"), produces="s",
        k=_SEQ_Q, n=_SEQ_K, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs,
        operand_shapes=(("q", (_SEQ_Q, _HEAD_DIM)),
                        ("k", (_SEQ_K, _HEAD_DIM)),
                        ("s", (_SEQ_Q, _SEQ_K))))
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="qk_mm", hardware="bh_p150", ops=(op,)),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "qk.mlir")
    reg = os.path.join(out_dir, "qk_reg.mlir")
    fb = os.path.join(out_dir, "qk.ttnn")
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
    for i, inp in enumerate(program.inputs):
        print(f"[meta] input[{i}] shape={inp['desc']['shape']}")
    for i, o in enumerate(program.outputs):
        print(f"[meta] output[{i}] shape={o['desc']['shape']}")

    torch.manual_seed(0)
    q_raw = (torch.rand(_SEQ_Q, _HEAD_DIM, dtype=torch.float32) * 0.2 - 0.1)
    k = (torch.rand(_SEQ_K, _HEAD_DIM, dtype=torch.float32) * 0.2 - 0.1).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    q = (q_raw * scale).to(torch.bfloat16)
    ref = torch.matmul(q.float(), k.float().T).to(torch.bfloat16)

    program.populate_inputs(lambda *_a, **_k: None, golden_inputs=[q, k])
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
    print(f"[verify] ref[0,-4:]={ref_f[0,-4:].tolist()}")
    print(f"[verify] got[0,-4:]={got_f[0,-4:].tolist()}")
    if pcc > 0.999 and max_abs < 5e-3:
        print(f"[PASS] Q@K^T Kt={_Kt} Nt={_Nt} within 5e-3")
    else:
        print(f"[FAIL] max_abs={max_abs:.3e} pcc={pcc:.6f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
