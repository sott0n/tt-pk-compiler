"""M1e gate 5: 4-core 2x2-tile elementwise add on silicon, numerically.

Four cores in a row, each owning one 32x32 tile via L1 block_sharded.
Same reader/writer the 1-tile verify uses, since the shard address
already points at this core's own tile; no per-core rt_args needed.
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


def _emit_fb(out_dir):
    cfg = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=False,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)
    cores = ((0, 0), (1, 0), (2, 0), (3, 0))
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1))
    writer_ct = (("cb_buffer_index", 2),)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                  ("cb_buffer_index", 2))
    reader_rt = (("address_of_tensor", 0), ("address_of_tensor", 1))
    writer_rt = (("address_of_tensor", 2),)
    entries = (
        KernelEntry("reader", av._READER_L1, cores,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", av._WRITER_L1, cores,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", add_mod._COMPUTE, cores,
                    compute_config=cfg, ct_args=compute_ct),
    )
    cbs = (
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048,
               tile_count=2, cores=cores),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048,
               tile_count=2, cores=cores),
        CBSpec(buffer_index=2, dtype="bf16", page_size=2048,
               tile_count=2, cores=cores),
    )
    op = ResidentOp(
        kernel_name="add.mc_verify",
        cores=cores,
        consumes=("a", "b"), produces="y",
        k=32, n=128, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="add_mc_v", hardware="bh_p150", ops=(op,)),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "add_mc_v.mlir")
    reg = os.path.join(out_dir, "add_mc_v_reg.mlir")
    fb = os.path.join(out_dir, "add_mc_v.ttnn")
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
    for i, out in enumerate(program.outputs):
        print(f"[meta] output[{i}] shape={out['desc']['shape']}")

    # Distinct per-tile values catch any inter-core shuffle bug.
    a = torch.zeros(32, 128, dtype=torch.bfloat16)
    b = torch.zeros(32, 128, dtype=torch.bfloat16)
    for t in range(4):
        a[:, 32 * t:32 * (t + 1)] = float(t + 1)
        b[:, 32 * t:32 * (t + 1)] = float(10 * (t + 1))
    ref = (a.float() + b.float()).to(torch.bfloat16)

    program.populate_inputs(lambda *x, **k: None, golden_inputs=[a, b])
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
    print(f"[verify] max_abs={max_abs:.3e} got shape={tuple(got.shape)}")
    for t in range(4):
        v_got = got.float()[0, 32 * t].item()
        v_ref = ref.float()[0, 32 * t].item()
        print(f"[verify] tile {t}: ref={v_ref:.2f}  got={v_got:.2f}")
    if max_abs < 1e-3:
        print("[PASS] numerical 1e-3")
    else:
        print("[FAIL] numerical 1e-3")
        sys.exit(1)


if __name__ == "__main__":
    main()
