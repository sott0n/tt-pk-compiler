"""M1b gate: emit tt-qwen-resident's 7-kernel attention program.

Reads the six C++ files from the `qwen36_resident` kernel directory,
assigns placeholder leader / worker / chunk / tail core grids, and
runs the whole thing through the compiler pipeline onto QB2. ct_args
and runtime args are deliberately empty — M1b only proves the plumbing.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk.problem import ComputeKernelConfig, KernelEntry
from tt_pk.resident import ResidentOp, ResidentSchedule
from tt_pk.codegen import emit_ttnn_generic


_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_OPT = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                    "ttmlir-opt")
_XLATE = os.path.join(_ROOT, "third_party", "tt-mlir", "build", "bin",
                      "ttmlir-translate")
_KDIR = "/home/kyamaguchi/tt-qwen-resident/qwen36_resident/kernels"


def _check(cond, label):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        raise SystemExit(1)


def _read(name):
    with open(os.path.join(_KDIR, name)) as f:
        return f.read()


# tt-metal's flatbuffer runtime hard-codes `compiler_include_paths = {}`
# (runtime/lib/ttnn/operations/generic/generic_op.cpp:307), so the JIT
# never sees _KDIR. Inline the project-local hpps into every source
# that pulls them. tt-metal `api/...` and `<...>` includes stay.
_LOCAL_HPPS = ("attn_common.hpp", "attn_compute_common.hpp")


def _inline_local_hpp(src: str) -> str:
    for name in _LOCAL_HPPS:
        body = _read(name)
        for inner in _LOCAL_HPPS:
            body = body.replace(f'#include "{inner}"', _read(inner))
        src = src.replace(f'#include "{name}"', body)
    return src


def _kernel_source(name: str) -> str:
    return _inline_local_hpp(_read(name))


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr}")


def main():
    if not os.path.exists(_OPT) or not os.path.exists(_XLATE):
        print(f"[SKIP] ttmlir-opt / ttmlir-translate missing"); return

    hifi4 = ComputeKernelConfig(
        math_fidelity="hifi4", fp32_dest_acc_en=True,
        dst_full_sync_en=True, bfp8_pack_precise=False,
        math_approx_mode=False)
    hifi2 = ComputeKernelConfig(
        math_fidelity="hifi2", fp32_dest_acc_en=True,
        dst_full_sync_en=False, bfp8_pack_precise=False,
        math_approx_mode=False)

    # Placeholder core grids. Real model.py picks these from chip
    # topology; M1b only needs disjoint non-reserved worker cores.
    lead = ((1, 1),)
    work = ((2, 2), (2, 3), (3, 2), (3, 3))
    chunk = ((4, 4),)
    tail = ((4, 5),)

    entries = (
        KernelEntry("reader", _kernel_source("attn_worker_reader.cpp"), work),
        KernelEntry("writer", _kernel_source("attn_worker_writer.cpp"), work),
        KernelEntry("compute", _kernel_source("attn_worker_compute.cpp"),
                    chunk, compute_config=hifi2),
        KernelEntry("compute", _kernel_source("attn_worker_compute.cpp"),
                    tail, compute_config=hifi4),
        KernelEntry("reader", _kernel_source("attn_leader_reader.cpp"), lead),
        KernelEntry("writer", _kernel_source("attn_leader_writer.cpp"), lead),
        KernelEntry("compute", _kernel_source("attn_leader_compute.cpp"),
                    lead, compute_config=hifi4),
    )
    _check(len(entries) == 7, "7 KernelEntry built from real cpp files")

    op = ResidentOp(
        kernel_name="qwen36.attention",
        cores=lead + work + chunk + tail,
        consumes=("x",), produces="y",
        k=128, n=128, dtype="bf16",
        compute_config=hifi4, kernels=entries)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="attn_m1b", hardware="bh_p150", ops=(op,)))

    _check(text.count("source_compute_kernel") == 3,
           "3 compute kernels emitted")
    _check(text.count("source_read_kernel") == 2,
           "2 reader kernels emitted")
    _check(text.count("source_write_kernel") == 2,
           "2 writer kernels emitted")

    with tempfile.TemporaryDirectory() as tmp:
        mlir = os.path.join(tmp, "attn.mlir")
        reg = os.path.join(tmp, "attn_reg.mlir")
        fb = os.path.join(tmp, "attn.ttnn")
        with open(mlir, "w") as f:
            f.write(text)
        _run([_OPT, "--ttcore-register-device", mlir, "-o", reg])
        _check(os.path.getsize(reg) > 0, "ttmlir-opt accepts the module")
        _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
        size = os.path.getsize(fb)
        _check(size > 0, f"flatbuffer emitted ({size} B)")

    print("OK")


if __name__ == "__main__":
    main()
