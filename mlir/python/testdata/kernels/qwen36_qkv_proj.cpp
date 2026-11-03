// Placeholder compute-kernel body for CP8 import tests.
// Real Resident kernels are substituted in at integration time;
// this file is small on purpose so test diffs stay readable.
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    mm_init(in0_cb, in1_cb, out_cb);
    acquire_dst();
    cb_wait_front(in0_cb, 1);
    cb_wait_front(in1_cb, 1);
    matmul_tiles(in0_cb, in1_cb, 0, 0, 0, false);
    cb_pop_front(in0_cb, 1);
    cb_pop_front(in1_cb, 1);
    cb_reserve_back(out_cb, 1);
    pack_tile(0, out_cb);
    cb_push_back(out_cb, 1);
    release_dst();
}
}
