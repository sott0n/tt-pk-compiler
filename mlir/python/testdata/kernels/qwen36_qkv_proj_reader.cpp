// Placeholder reader-kernel body for CP9 import tests.
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_in);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_reserve_back(cb_in, 1);
        uint64_t src = get_noc_addr(addr + t * tile_bytes);
        uint32_t dst = get_write_ptr(cb_in);
        noc_async_read(src, dst, tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
