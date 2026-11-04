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

from tt_pk.codegen_types import CBSpec, SemSpec
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

    # 21 compile-time args attn_common.hpp:54-74 expects. Indices match
    # tt-qwen-resident model.py:1030-1045; values are a toy 1-chip BH
    # config so get_compile_time_arg_val(N) sees valid bits at JIT time
    # and the kernels compile. Numerical correctness is a separate step.
    import struct
    eps_bits = struct.unpack("<I", struct.pack("<f", 1e-5))[0]
    attn_ct = (
        int(os.environ.get("ATTN_LAYERS", "1")),  # 0 layers (n_attn)
        1,          # 1 copies (attn_copies)
        4,          # 2 workers
        1,          # 3 num_streamers (S)
        4,          # 4 chunk_max (kv_block)
        4,          # 5 heads (heads_a)
        4,          # 6 Dt (head_dim / 32)
        2,          # 7 rot_tiles (ROT / TILE)
        0, 1, 2, 3, # 8-11 sem_rows / sem_q / sem_m / sem_M
        4, 5,       # 12-13 sem_part / sem_heads
        eps_bits,   # 14 eps bits
        6,          # 15 sem_addr
        1,          # 16 fanin (slots_per_node)
        1,          # 17 banks
        7,          # 18 sem_tail (SEM_LOCAL)
        2,          # 19 batch (verify=1 forces 2)
        0,          # 20 verify
    )

    entries = (
        KernelEntry("reader", _kernel_source("attn_worker_reader.cpp"), work,
                    ct_args=attn_ct),
        KernelEntry("writer", _kernel_source("attn_worker_writer.cpp"), work,
                    ct_args=attn_ct),
        KernelEntry("compute", _kernel_source("attn_worker_compute.cpp"),
                    chunk, compute_config=hifi2, ct_args=attn_ct),
        KernelEntry("compute", _kernel_source("attn_worker_compute.cpp"),
                    tail, compute_config=hifi4, ct_args=attn_ct),
        KernelEntry("reader", _kernel_source("attn_leader_reader.cpp"), lead,
                    ct_args=attn_ct),
        KernelEntry("writer", _kernel_source("attn_leader_writer.cpp"), lead,
                    ct_args=attn_ct),
        KernelEntry("compute", _kernel_source("attn_leader_compute.cpp"),
                    lead, compute_config=hifi4, ct_args=attn_ct),
    )
    _check(len(entries) == 7, "7 KernelEntry built from real cpp files")

    # 8 semaphores the attn kernels reference by ID (SEM_ROWS=0,
    # SEM_Q=1, SEM_m=2, SEM_M=3, SEM_PART=4, SEM_HEADS=5, SEM_ADDR=6,
    # SEM_LOCAL=7). All on the union of all cores so every kernel can
    # `get_semaphore(id)`. Initial value 0 matches tt-qwen-resident.
    union_cores = lead + work + chunk + tail
    sems = tuple(SemSpec(id=i, initial_value=0, cores=union_cores)
                 for i in range(8))

    # Minimal CB map. Buffer indices the attn kernels touch (seen in
    # attn_common.hpp:89-120). Sizes are small placeholders — enough to
    # satisfy `cb_reserve_back` at launch; numerical correctness comes
    # later when CBs are sized from the real head / kv-block shapes.
    _BF16 = 2048
    used_cb_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                       14, 15, 16, 17, 18, 24, 26, 31)
    cbs = tuple(CBSpec(buffer_index=i, dtype="bf16",
                       page_size=_BF16, tile_count=1,
                       cores=union_cores)
                for i in used_cb_indices)

    op = ResidentOp(
        kernel_name="qwen36.attention",
        cores=union_cores,
        consumes=("x",), produces="y",
        k=128, n=128, dtype="bf16",
        compute_config=hifi4, kernels=entries,
        semaphores=sems, cbs=cbs)
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="attn_m1b", hardware="bh_p150", ops=(op,)))

    _check(text.count("source_compute_kernel") == 3,
           "3 compute kernels emitted")
    _check(text.count("source_read_kernel") == 2,
           "2 reader kernels emitted")
    _check(text.count("source_write_kernel") == 2,
           "2 writer kernels emitted")

    out_dir = os.environ.get("ATTN_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        mlir = os.path.join(out_dir, "attn.mlir")
        reg = os.path.join(out_dir, "attn_reg.mlir")
        fb = os.path.join(out_dir, "attn.ttnn")
    else:
        tmp_ctx = tempfile.TemporaryDirectory()
        mlir = os.path.join(tmp_ctx.name, "attn.mlir")
        reg = os.path.join(tmp_ctx.name, "attn_reg.mlir")
        fb = os.path.join(tmp_ctx.name, "attn.ttnn")

    with open(mlir, "w") as f:
        f.write(text)
    _run([_OPT, "--ttcore-register-device", mlir, "-o", reg])
    _check(os.path.getsize(reg) > 0, "ttmlir-opt accepts the module")
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    size = os.path.getsize(fb)
    _check(size > 0, f"flatbuffer emitted ({size} B)")
    if out_dir:
        print(f"saved: {fb}")

    print("OK")


if __name__ == "__main__":
    main()
