"""CP10 gate: per-dtype CB page sizing + ttmlir-opt round-trip.

derive_cbs now looks the page size up by dtype rather than assuming
bf16. The expected numbers match tt-mlir's `ttcore::getTileSizeBytes`
in `lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp`. The second half emits a
tiny bfp_bf8 schedule through codegen and round-trips it through the
in-tree ttmlir-opt.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tt_pk import (CBSpec, derive_cbs, emit_ttnn_generic, ensure_available,
                   lint, TTNNLintError)
from tt_pk.problem import ComputeKernelConfig
from tt_pk.resident import ResidentOp, ResidentSchedule


_EXPECTED = {
    "f32":      4096,
    "f16":      2048,
    "bf16":     2048,
    "bfp_f8":   1088,
    "bfp_bf8":  1088,
    "bfp_f4":   576,
    "bfp_bf4":  576,
    "bfp_f2":   320,
    "bfp_bf2":  320,
    "u32":      4096,
    "u16":      2048,
    "u8":       1024,
    "si32":     4096,
    "i1":       1024,
}


_CFG = ComputeKernelConfig(
    math_fidelity="hifi4",
    fp32_dest_acc_en=False,
    dst_full_sync_en=False,
    bfp8_pack_precise=False,
    math_approx_mode=False,
)


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def _make_op(dtype: str) -> ResidentOp:
    return ResidentOp(
        kernel_name="demo.proj", cores=((1, 1),),
        consumes=("x",), produces="y",
        k=64, n=64, dtype=dtype, compute_config=_CFG)


def main() -> None:
    for dtype, want in _EXPECTED.items():
        cbs = derive_cbs(_make_op(dtype))
        for cb in cbs:
            _check(cb.dtype == dtype,
                   f"{dtype}: cb.dtype matches op.dtype")
            _check(cb.page_size == want,
                   f"{dtype}: page_size == {want} (got {cb.page_size})")

    try:
        derive_cbs(_make_op("float64"))
    except NotImplementedError as e:
        _check("float64" in str(e), "unknown dtype raises NotImplementedError")
    else:
        _check(False, "unknown dtype must raise")

    try:
        ensure_available()
    except TTNNLintError as e:
        print(f"[SKIP] ttmlir-opt not built: {e}")
        return

    for dtype in ("bfp_bf8", "f32"):
        schedule = ResidentSchedule(
            model_name="demo", hardware="bh_p150",
            ops=(_make_op(dtype),))
        text = emit_ttnn_generic(schedule)
        _check(f"dtype = {dtype}" in text,
               f"{dtype}: emitted dtype spelling matches tt-mlir")
        result = lint(text)
        _check(result.attrs_rejected == (),
               f"{dtype}: ttmlir-opt accepts emitted TTNN text "
               f"(rejected: {result.attrs_rejected})")

    print("OK")


if __name__ == "__main__":
    main()
