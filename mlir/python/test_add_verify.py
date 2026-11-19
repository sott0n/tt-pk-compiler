"""M1e gate 4: numerically verify 1-tile add on silicon.

Emits the add flatbuffer, submits it with known bf16 inputs via
ttrt.runtime, reads the output back, compares to a+b in torch.
First numerical-correctness check in the project; every later gate
reuses the same pattern.
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

import test_add_launch as add_mod


# L1-sharded path: reader/writer use raw get_noc_addr(my_x, my_y, l1_offset)
# against the operand's own L1 shard, matching upstream generic_op.mlir.
_READER_L1 = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t a_bytes = get_tile_size(cb_a);
    const uint32_t b_bytes = get_tile_size(cb_b);
    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);
    uint64_t a_noc = get_noc_addr(my_x[noc_index], my_y[noc_index], a_addr);
    uint64_t b_noc = get_noc_addr(my_x[noc_index], my_y[noc_index], b_addr);
    noc_async_read(a_noc, get_write_ptr(cb_a), a_bytes);
    noc_async_read(b_noc, get_write_ptr(cb_b), b_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
}
"""

_WRITER_L1 = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_bytes = get_tile_size(cb_out);
    cb_wait_front(cb_out, 1);
    uint64_t out_noc = get_noc_addr(my_x[noc_index], my_y[noc_index], out_addr);
    noc_async_write(get_read_ptr(cb_out), out_noc, out_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
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
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1))
    writer_ct = (("cb_buffer_index", 2),)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                  ("cb_buffer_index", 2))
    reader_rt = (("address_of_tensor", 0), ("address_of_tensor", 1))
    writer_rt = (("address_of_tensor", 2),)
    entries = (
        KernelEntry("reader", _READER_L1, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", _WRITER_L1, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", add_mod._COMPUTE, core,
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
        kernel_name="add.verify",
        cores=core,
        consumes=("a", "b"), produces="y",
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="add_verify", hardware="bh_p150", ops=(op,)),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "add_v.mlir")
    reg = os.path.join(out_dir, "add_v_reg.mlir")
    fb = os.path.join(out_dir, "add_v.ttnn")
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
    import ttrt
    import ttrt.binary
    import ttrt.runtime
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
        print(f"[meta] input[{i}] shape={inp['desc']['shape']} dtype={inp['desc']['layout']['memory_desc']['data_type']}")
    for i, out in enumerate(program.outputs):
        print(f"[meta] output[{i}] shape={out['desc']['shape']} dtype={out['desc']['layout']['memory_desc']['data_type']}")

    torch.manual_seed(0)
    a = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
    b = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    ref = (a.float() + b.float()).to(torch.bfloat16)

    program.populate_inputs(lambda *_a, **_k: None, golden_inputs=[a, b])
    program.populate_outputs(lambda shape, dtype: torch.zeros(shape, dtype=dtype))

    fb_mesh_shape = program.mesh_shape
    inputs = [create_tensor(ts, fb_mesh_shape) for ts in program.input_tensors]

    mesh_options = ttrt.runtime.MeshDeviceOptions()
    mesh_options.mesh_shape = fb_mesh_shape
    device = ttrt.runtime.open_mesh_device(mesh_options)
    try:
        inputs = convert_input_layouts(device, inputs, bin_obj.fbb, 0)
        outputs = ttrt.runtime.submit(device, bin_obj.fbb, 0, inputs)
        ttrt.runtime.wait(outputs)
        host = ttrt.runtime.to_host(outputs[0], untilize=True)
        got = convert_runtime_to_torch_tensor(host[0])
    finally:
        ttrt.runtime.close_mesh_device(device)

    got_f = got.float()
    ref_f = ref.float()
    max_abs = (got_f - ref_f).abs().max().item()
    rtol = ((got_f - ref_f).abs() / ref_f.abs().clamp_min(1e-6)).max().item()
    print(f"[verify] max_abs={max_abs:.3e}  max_rel={rtol:.3e}")
    print(f"[verify] ref[0,0]={ref_f[0,0].item():.4f}  got[0,0]={got_f[0,0].item():.4f}")
    print(f"[verify] a[0,0]={a.float()[0,0].item():.4f}  b[0,0]={b.float()[0,0].item():.4f}")
    print(f"[verify] ref row 0: {ref_f[0,:8].tolist()}")
    print(f"[verify] got row 0: {got_f[0,:8].tolist()}")
    print(f"[verify] got col 0: {got_f[:8,0].tolist()}")
    print(f"[verify] got shape={tuple(got.shape)}  dtype={got.dtype}")
    if max_abs < 1e-3:
        print("[PASS] numerical 1e-3")
    else:
        print("[FAIL] numerical 1e-3")
        sys.exit(1)


if __name__ == "__main__":
    main()
