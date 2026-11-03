// Placeholder writer-kernel body for CP9 import tests.
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_out, 1);
        uint64_t dst = get_noc_addr(addr + t * tile_bytes);
        uint32_t src = get_read_ptr(cb_out);
        noc_async_write(src, dst, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
