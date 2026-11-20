"""M1e gate 8: 2-op L1 chain, numerically bit-exact.

y = (a + b) + c on one core, two ttnn.generic calls back to back.
Validates the multi-op L1 codegen path: intermediate x stays in L1,
no DRAM round-trip between ops.
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
import test_add_launch as add_mod


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr}")


def _make_add(name, consumes, produces, core, cfg):
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
    return ResidentOp(
        kernel_name=name,
        cores=core,
        consumes=consumes, produces=produces,
        k=32, n=32, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)


def _emit_fb(out_dir):
    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)
    core = ((0, 0),)
    op1 = _make_add("add.ab", ("a", "b"), "x", core, cfg)
    op2 = _make_add("add.xc", ("x", "c"), "y", core, cfg)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="add_chain", hardware="bh_p150", ops=(op1, op2)),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "chain.mlir")
    reg = os.path.join(out_dir, "chain_reg.mlir")
    fb = os.path.join(out_dir, "chain.ttnn")
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

    # Integer values exact in bf16; y = a + b + c should be exact.
    a = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
    b = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    c = torch.full((32, 32), 4.0, dtype=torch.bfloat16)
    ref = (a.float() + b.float() + c.float()).to(torch.bfloat16)

    program.populate_inputs(lambda *_a, **_k: None, golden_inputs=[a, b, c])
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

    max_abs = (got.float() - ref.float()).abs().max().item()
    print(f"[verify] max_abs={max_abs:.3e}  shape={tuple(got.shape)}")
    print(f"[verify] ref[0,0]={ref.float()[0,0].item():.4f}  "
          f"got[0,0]={got.float()[0,0].item():.4f}")
    if max_abs < 1e-3:
        print("[PASS] numerical 1e-3")
    else:
        print("[FAIL] numerical 1e-3")
        sys.exit(1)


if __name__ == "__main__":
    main()
