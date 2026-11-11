"""M1e gate 2: 1-tile elementwise add on silicon.

Two DRAM inputs, one DRAM output. Reader fans both inputs into CB0/CB1,
compute runs add_tiles into DST, writer pushes the sum back out.
Proves two-input ttnn.generic + real tt-metal compute kernel.
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


# Single reader pulls both inputs. tt-metal allows at most one reader
# (brisc) and one writer (ncrisc) per core, so the two-reader split used
# by copy does not work for a 2-input op.
_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    constexpr auto a_args = TensorAccessorArgs<2>();
    constexpr auto b_args =
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, a_addr, get_tile_size(cb_a));
    const auto b = TensorAccessor(b_args, b_addr, get_tile_size(cb_b));
    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);
    noc_async_read_tile(0, a, get_write_ptr(cb_a));
    noc_async_read_tile(0, b, get_write_ptr(cb_b));
    noc_async_read_barrier();
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
}
"""

_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    constexpr auto dst_args =
        TensorAccessorArgs<get_compile_time_arg_val(1)>();
    const auto dst = TensorAccessor(dst_args, dst_addr,
                                    get_tile_size(cb_id));
    cb_wait_front(cb_id, 1);
    noc_async_write_tile(0, dst, get_read_ptr(cb_id));
    noc_async_write_barrier();
    cb_pop_front(cb_id, 1);
}
"""

_COMPUTE = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    binary_op_init_common(cb_a, cb_b, cb_out);
    add_tiles_init(cb_a, cb_b);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
"""


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
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                 ("tensor_accessor_args", 0), ("tensor_accessor_args", 1))
    writer_ct = (("cb_buffer_index", 2), ("tensor_accessor_args", 2))
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                  ("cb_buffer_index", 2))
    reader_rt = (("address_of_tensor", 0), ("address_of_tensor", 1))
    writer_rt = (("address_of_tensor", 2),)

    entries = (
        KernelEntry("reader", _READER, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", _WRITER, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", _COMPUTE, core,
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
        kernel_name="add.single_tile",
        cores=core,
        consumes=("a", "b"), produces="y",
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="add_m1e", hardware="bh_p150", ops=(op,)))

    out_dir = os.environ.get("ADD_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        mlir = os.path.join(out_dir, "add.mlir")
        reg = os.path.join(out_dir, "add_reg.mlir")
        fb = os.path.join(out_dir, "add.ttnn")
    else:
        tmp_ctx = tempfile.TemporaryDirectory()
        mlir = os.path.join(tmp_ctx.name, "add.mlir")
        reg = os.path.join(tmp_ctx.name, "add_reg.mlir")
        fb = os.path.join(tmp_ctx.name, "add.ttnn")

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
