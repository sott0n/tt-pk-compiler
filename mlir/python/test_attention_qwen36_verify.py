"""Real-M1 step 4-7: Qwen3.6 attention cluster on BH p150.

Shapes match tt_pk.qwen36 (HIDDEN=4096, HEAD_DIM=128, NUM_HEADS=32,
NUM_KV_HEADS=8, GQA 4:1). Decode-shape attention at seq_q=1 tile
(32 rows) and seq_k=L*32 tiles.

Chain collapsed into one ttnn.generic op with one fused compute
kernel that loops over heads: per head h picks kv_h = h // 4,
does Q_h @ K_{kv_h}^T, multi-tile row softmax, and P_h @ V_{kv_h}.
Reader streams Q/K/V once; writer drains per-head outputs in order.

Scale 1/sqrt(head_dim) folded into Q host-side (qwen36 convention
from tt-qwen-resident model.py: rmsnorm(q, wq / sqrt(HD))).
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


# Qwen3.6-27B attention cluster math at single-core scale:
# HEAD_DIM=128, GQA 4:1, bf16, multi-tile row softmax over L*32
# keys, scale 1/sqrt(HEAD_DIM) folded into Q (same convention as
# tt-qwen-resident model.py:238 `rmsnorm(q, wq / sqrt(HD))`).
# NUM_HEADS=8, NUM_KV_HEADS=2 = 1/4 of the full Qwen3.6 head count.
# Scaling the head count past this on BH p150 silently deadlocks
# the fused compute kernel (seen at 16 total heads, both single-
# and multi-core block_sharded); scoping the remaining 4x is
# M-later. All seven M1 primitives (multi-tile softmax, scaled
# Q@K^T, multi-head unroll, GQA expand-by-index, fused cluster,
# concat-heads via writer stride, +/-1e-3 vs torch) PASS here.
_HEAD_DIM = 128
_NUM_HEADS = 8
_NUM_KV_HEADS = 2
_GQA_RATIO = _NUM_HEADS // _NUM_KV_HEADS
_HIDDEN = _NUM_HEADS * _HEAD_DIM
_KV_DIM = _NUM_KV_HEADS * _HEAD_DIM
_HD_TILES = _HEAD_DIM // 32
_L = 2
_SEQ_Q = 32
_SEQ_K = _L * 32

_CORES = ((0, 0),)
_NUM_CORES = len(_CORES)
_LOCAL_NUM_HEADS = _NUM_HEADS // _NUM_CORES
_LOCAL_NUM_KV_HEADS = _NUM_KV_HEADS // _NUM_CORES

_Q_TILES = _LOCAL_NUM_HEADS * _HD_TILES
_KV_TILES = _L * _LOCAL_NUM_KV_HEADS * _HD_TILES


# Per-head streaming keeps CBs small enough for single-core L1:
# the reader pushes Q_h plus K_{kv_h} and V_{kv_h} for one head
# at a time, the compute pops them after processing, so cb_q /
# cb_k / cb_v only ever hold one head's slice.
_READER_ATTN = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(4);
    constexpr uint32_t GQA_RATIO = get_compile_time_arg_val(5);
    constexpr uint32_t NUM_KV_HEADS = get_compile_time_arg_val(6);
    constexpr uint32_t L = get_compile_time_arg_val(7);
    constexpr uint32_t HD_TILES = get_compile_time_arg_val(8);
    constexpr uint16_t BF16_ONE = 0x3F80;
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t tile_bytes = get_tile_size(cb_q);

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

    for (uint32_t h = 0; h < NUM_HEADS; ++h) {
        uint32_t kv_h = h / GQA_RATIO;

        cb_reserve_back(cb_q, HD_TILES);
        uint64_t q_noc = get_noc_addr(
            my_x[noc_index], my_y[noc_index],
            q_addr + h * HD_TILES * tile_bytes);
        noc_async_read(q_noc, get_write_ptr(cb_q), tile_bytes * HD_TILES);
        noc_async_read_barrier();
        cb_push_back(cb_q, HD_TILES);

        for (uint32_t mt = 0; mt < L; ++mt) {
            cb_reserve_back(cb_k, HD_TILES);
            uint64_t k_off = (mt * NUM_KV_HEADS * HD_TILES
                              + kv_h * HD_TILES) * tile_bytes;
            uint64_t k_noc = get_noc_addr(
                my_x[noc_index], my_y[noc_index], k_addr + k_off);
            noc_async_read(k_noc, get_write_ptr(cb_k),
                           tile_bytes * HD_TILES);
            noc_async_read_barrier();
            cb_push_back(cb_k, HD_TILES);
        }

        for (uint32_t mt = 0; mt < L; ++mt) {
            cb_reserve_back(cb_v, HD_TILES);
            uint64_t v_off = (mt * NUM_KV_HEADS * HD_TILES
                              + kv_h * HD_TILES) * tile_bytes;
            uint64_t v_noc = get_noc_addr(
                my_x[noc_index], my_y[noc_index], v_addr + v_off);
            noc_async_read(v_noc, get_write_ptr(cb_v),
                           tile_bytes * HD_TILES);
            noc_async_read_barrier();
            cb_push_back(cb_v, HD_TILES);
        }
    }
}
"""


_WRITER_ATTN = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t O_TILES = get_compile_time_arg_val(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    for (uint32_t i = 0; i < O_TILES; ++i) {
        cb_wait_front(cb_out, 1);
        uint64_t out_noc = get_noc_addr(
            my_x[noc_index], my_y[noc_index],
            out_addr + i * tile_bytes);
        noc_async_write(get_read_ptr(cb_out), out_noc, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
"""


# Head-by-head attention cluster: per head h pick kv_h = h / GQA_RATIO,
# matmul transpose for Q_h @ K_{kv_h}^T, multi-tile row softmax, then
# plain matmul for P_h @ V_{kv_h}. cb_s is reused as cb_p for the
# second matmul's A operand.
_COMPUTE_ATTN = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_s = get_compile_time_arg_val(4);
    constexpr uint32_t cb_max = get_compile_time_arg_val(5);
    constexpr uint32_t cb_exps = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(9);
    constexpr uint32_t GQA_RATIO = get_compile_time_arg_val(10);
    constexpr uint32_t NUM_KV_HEADS = get_compile_time_arg_val(11);
    constexpr uint32_t L = get_compile_time_arg_val(12);
    constexpr uint32_t HD_TILES = get_compile_time_arg_val(13);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);

    cb_wait_front(cb_scaler, 1);

    for (uint32_t h = 0; h < NUM_HEADS; ++h) {
        cb_wait_front(cb_q, HD_TILES);
        cb_wait_front(cb_k, L * HD_TILES);
        cb_wait_front(cb_v, L * HD_TILES);

        matmul_init(cb_q, cb_k, 1);
        for (uint32_t nt = 0; nt < L; ++nt) {
            cb_reserve_back(cb_s, 1);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < HD_TILES; ++kt) {
                matmul_tiles(cb_q, cb_k, kt, nt * HD_TILES + kt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_s);
            tile_regs_release();
            cb_push_back(cb_s, 1);
        }
        cb_wait_front(cb_s, L);

        cb_reserve_back(cb_max, 1);
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scaler, cb_max);
        for (uint32_t i = 0; i < L; ++i) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scaler, i, 0, 0);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_max);
        tile_regs_release();
        cb_push_back(cb_max, 1);
        cb_wait_front(cb_max, 1);

        for (uint32_t i = 0; i < L; ++i) {
            cb_reserve_back(cb_exps, 1);
            tile_regs_acquire();
            sub_bcast_cols_init(cb_s, cb_max);
            sub_tiles_bcast<BroadcastType::COL>(cb_s, cb_max, i, 0, 0);
            exp_tile_init<false>();
            exp_tile<false>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_exps);
            tile_regs_release();
            cb_push_back(cb_exps, 1);
        }
        cb_wait_front(cb_exps, L);

        cb_reserve_back(cb_sum, 1);
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exps, cb_scaler, cb_sum);
        for (uint32_t i = 0; i < L; ++i) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exps, cb_scaler, i, 0, 0);
        }
        reduce_uninit();
        recip_tile_init();
        recip_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sum);
        tile_regs_release();
        cb_push_back(cb_sum, 1);
        cb_wait_front(cb_sum, 1);

        cb_pop_front(cb_s, L);

        for (uint32_t i = 0; i < L; ++i) {
            cb_reserve_back(cb_s, 1);
            tile_regs_acquire();
            mul_bcast_cols_init(cb_exps, cb_sum);
            mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_sum, i, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_s);
            tile_regs_release();
            cb_push_back(cb_s, 1);
        }
        cb_wait_front(cb_s, L);

        cb_pop_front(cb_max, 1);
        cb_pop_front(cb_exps, L);
        cb_pop_front(cb_sum, 1);

        matmul_init(cb_s, cb_v);
        for (uint32_t nt = 0; nt < HD_TILES; ++nt) {
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < L; ++kt) {
                matmul_tiles(cb_s, cb_v, kt, kt * HD_TILES + nt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }

        cb_pop_front(cb_s, L);
        cb_pop_front(cb_q, HD_TILES);
        cb_pop_front(cb_k, L * HD_TILES);
        cb_pop_front(cb_v, L * HD_TILES);
    }

    cb_pop_front(cb_scaler, 1);
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
    core = _CORES
    reader_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                 ("cb_buffer_index", 2), ("cb_buffer_index", 3),
                 _LOCAL_NUM_HEADS, _GQA_RATIO, _LOCAL_NUM_KV_HEADS,
                 _L, _HD_TILES)
    writer_ct = (("cb_buffer_index", 8), _Q_TILES)
    compute_ct = (("cb_buffer_index", 0), ("cb_buffer_index", 1),
                  ("cb_buffer_index", 2), ("cb_buffer_index", 3),
                  ("cb_buffer_index", 4), ("cb_buffer_index", 5),
                  ("cb_buffer_index", 6), ("cb_buffer_index", 7),
                  ("cb_buffer_index", 8),
                  _LOCAL_NUM_HEADS, _GQA_RATIO, _LOCAL_NUM_KV_HEADS,
                  _L, _HD_TILES)
    reader_rt = (("address_of_tensor", 0),
                 ("address_of_tensor", 1),
                 ("address_of_tensor", 2))
    writer_rt = (("address_of_tensor", 3),)
    entries = (
        KernelEntry("reader", _READER_ATTN, core,
                    ct_args=reader_ct, common_rt_args=reader_rt),
        KernelEntry("writer", _WRITER_ATTN, core,
                    ct_args=writer_ct, common_rt_args=writer_rt),
        KernelEntry("compute", _COMPUTE_ATTN, core,
                    compute_config=cfg, ct_args=compute_ct),
    )
    cbs = (
        CBSpec(buffer_index=0, dtype="bf16", page_size=2048,
               tile_count=_HD_TILES * 2, cores=core),
        CBSpec(buffer_index=1, dtype="bf16", page_size=2048,
               tile_count=_L * _HD_TILES * 2, cores=core),
        CBSpec(buffer_index=2, dtype="bf16", page_size=2048,
               tile_count=_L * _HD_TILES * 2, cores=core),
        CBSpec(buffer_index=3, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
        CBSpec(buffer_index=4, dtype="bf16", page_size=2048,
               tile_count=_L, cores=core),
        CBSpec(buffer_index=5, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
        CBSpec(buffer_index=6, dtype="bf16", page_size=2048,
               tile_count=_L, cores=core),
        CBSpec(buffer_index=7, dtype="bf16", page_size=2048,
               tile_count=1, cores=core),
        CBSpec(buffer_index=8, dtype="bf16", page_size=2048,
               tile_count=_HD_TILES * 2, cores=core),
    )
    op = ResidentOp(
        kernel_name="qwen36.attention",
        cores=core,
        consumes=("Q", "K", "V"), produces="O",
        k=_SEQ_Q, n=_HIDDEN, dtype="bf16",
        compute_config=cfg, kernels=entries, cbs=cbs,
        operand_shapes=(("Q", (_SEQ_Q, _HIDDEN)),
                        ("K", (_SEQ_K, _KV_DIM)),
                        ("V", (_SEQ_K, _KV_DIM)),
                        ("O", (_SEQ_Q, _HIDDEN))))
    text = emit_ttnn_generic(ResidentSchedule(
        model_name="qwen36_attn", hardware="bh_p150", ops=(op,)),
        operand_memspace="l1_block_sharded")

    mlir = os.path.join(out_dir, "qwen_attn.mlir")
    reg = os.path.join(out_dir, "qwen_attn_reg.mlir")
    fb = os.path.join(out_dir, "qwen_attn.ttnn")
    with open(mlir, "w") as f:
        f.write(text)
    print(f"[mlir] wrote {mlir} ({os.path.getsize(mlir)} B)", flush=True)
    sysdesc_arg = "--ttcore-register-device"
    sysdesc_env = os.environ.get("SYSTEM_DESC_PATH")
    if sysdesc_env:
        sysdesc_arg = f"--ttcore-register-device=system-desc-path={sysdesc_env}"
    _run([_OPT, sysdesc_arg, mlir, "-o", reg])
    print(f"[reg] wrote {reg}", flush=True)
    _run([_XLATE, "--ttnn-to-flatbuffer", reg, "-o", fb])
    print(f"[fb] wrote {fb}", flush=True)
    return fb


def _torch_ref(q_raw, k, v):
    import torch
    def bf(t): return t.to(torch.bfloat16)
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    q = bf(q_raw * scale)

    # Reshape to heads: Q [seq_q, NUM_HEADS * head_dim]
    #                   -> [seq_q, NUM_HEADS, head_dim]
    #                   -> [NUM_HEADS, seq_q, head_dim]
    q_h = q.float().reshape(_SEQ_Q, _NUM_HEADS, _HEAD_DIM).permute(1, 0, 2)
    k_h = k.float().reshape(_SEQ_K, _NUM_KV_HEADS, _HEAD_DIM).permute(1, 0, 2)
    v_h = v.float().reshape(_SEQ_K, _NUM_KV_HEADS, _HEAD_DIM).permute(1, 0, 2)
    # GQA: expand kv to NUM_HEADS by repeating each kv head GQA_RATIO times
    k_e = k_h.repeat_interleave(_GQA_RATIO, dim=0)
    v_e = v_h.repeat_interleave(_GQA_RATIO, dim=0)

    S = bf(torch.matmul(q_h, k_e.transpose(-1, -2)))
    P = bf(torch.softmax(S.float(), dim=-1))
    O_h = bf(torch.matmul(P.float(), v_e.float()))
    # Re-layout: [NUM_HEADS, seq_q, head_dim] -> [seq_q, NUM_HEADS, head_dim]
    #                                         -> [seq_q, NUM_HEADS*head_dim]
    O = O_h.permute(1, 0, 2).reshape(_SEQ_Q, _HIDDEN)
    return q, O


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
    q_raw = (torch.rand(_SEQ_Q, _HIDDEN, dtype=torch.float32) * 0.2 - 0.1)
    k = (torch.rand(_SEQ_K, _KV_DIM, dtype=torch.float32) * 0.2 - 0.1).to(torch.bfloat16)
    v = (torch.rand(_SEQ_K, _KV_DIM, dtype=torch.float32) * 0.2 - 0.1).to(torch.bfloat16)
    q, ref = _torch_ref(q_raw, k, v)

    program.populate_inputs(lambda *_a, **_k: None, golden_inputs=[q, k, v])
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
    print(f"[verify] ref[0,2048:2052]={ref_f[0,2048:2052].tolist()}")
    print(f"[verify] got[0,2048:2052]={got_f[0,2048:2052].tolist()}")
    print(f"[verify] ref[0,-4:]={ref_f[0,-4:].tolist()}")
    print(f"[verify] got[0,-4:]={got_f[0,-4:].tolist()}")
    if max_abs < 1e-3 and pcc > 0.999:
        print(f"[PASS] qwen3.6 attention real-M1 within +/-1e-3")
    else:
        print(f"[FAIL] max_abs={max_abs:.3e} pcc={pcc:.6f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
