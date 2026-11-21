"""AnalysisProblem: what the compiler must place.

Stage 1 of the 3-stage IR. Pure data. Independent of hardware mapping and
of any particular placement choice. Downstream stages read it; they never
mutate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

# A kernel ct/rt_arg is either a literal uint (ct_arg_N value) or a typed
# tag that the codegen lowers to the matching #ttnn.kernel_arg_* attr:
#   ("address_of_tensor", N) | ("cb_buffer_index", N) |
#   ("tensor_accessor_args", N)
KernelArg = Union[int, Tuple[str, int]]


@dataclass(frozen=True)
class ComputeKernelConfig:
    """Compile-time config emitted on every TTNN compute_kernel attr.

    Values mirror the required fields of `TTNN_ComputeKernelAttr`. The
    compiler fills them from a per-kernel override or the hardware
    policy default; it never invents them in the backend emitter.
    """
    math_fidelity: str
    fp32_dest_acc_en: bool
    dst_full_sync_en: bool
    bfp8_pack_precise: bool
    math_approx_mode: bool


@dataclass(frozen=True)
class KernelEntry:
    """One kernel slot inside a Resident-style multi-kernel program.

    A ttnn.generic op can carry several reader / writer / compute kernels
    on different core ranges; tt-qwen-resident's attention is 7 of them.
    `cores` is per-entry so leader and worker roles land on their own
    grids; `compute_config` is only meaningful on compute entries.
    """
    role: str
    source: str
    cores: Tuple[Tuple[int, int], ...]
    compute_config: Optional[ComputeKernelConfig] = None
    ct_args: Tuple[KernelArg, ...] = ()
    common_rt_args: Tuple[KernelArg, ...] = ()
    # Per-core runtime args: tuple of ((x, y), (arg, arg, ...)) pairs.
    # Each core on `cores` typically gets one entry; cores without an
    # entry receive no per-core rt_args.
    rt_args: Tuple[Tuple[Tuple[int, int], Tuple[KernelArg, ...]], ...] = ()


@dataclass(frozen=True)
class KernelSpec:
    name: str
    k: int
    n: int
    produces: str = ""
    consumes: Tuple[str, ...] = ()
    dtype: str = "bf16"
    # Explicit per-token cost for the search's weighted share. When 0, the
    # search falls back to k*n, which is fine for projections but blind to
    # the num_heads / context-length axes that make attention expensive
    # at decode. Set this to override.
    cost: int = 0
    compute_config: Optional[ComputeKernelConfig] = None
    # Inline C++ body for the compute kernel. When set, codegen emits
    # `#ttnn.source_compute_kernel` and drops the private function
    # declaration. When None, codegen falls back to `#ttnn.compute_kernel`
    # with a symbol_ref to a private declaration.
    compute_kernel_source: Optional[str] = None
    # Optional reader/writer C++ bodies. Each one, when set, adds a
    # `#ttnn.source_read_kernel` or `#ttnn.source_write_kernel` entry to
    # the generic op's program.kernels list on the same core ranges as
    # the compute kernel.
    reader_kernel_source: Optional[str] = None
    writer_kernel_source: Optional[str] = None
    # Multi-kernel programs (e.g. tt-qwen-resident attention: 7 kernels
    # across leader + worker grids). When set, codegen ignores the single
    # compute / reader / writer slots above and emits one program.kernels
    # entry per KernelEntry here.
    kernels: Tuple[KernelEntry, ...] = ()

    def effective_cost(self) -> int:
        return self.cost if self.cost > 0 else max(1, self.k * self.n)


@dataclass(frozen=True)
class DecoderLayerSpec:
    kernels: Tuple[KernelSpec, ...]


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    grid_x: int
    grid_y: int
    reserved_cores: Tuple[Tuple[int, int], ...] = ()
    default_compute_config: Optional[ComputeKernelConfig] = None
    # Per-Tensix L1 budget used by strategy's constraint check. Default
    # fits Blackhole p150 (1.5 MiB usable).
    l1_bytes_per_core: int = 1536 * 1024

    def total_cores(self) -> int:
        return self.grid_x * self.grid_y - len(self.reserved_cores)


# v2 roles. See mlir/INTERFACE.md § Parametric inputs. Strategy chooses
# counts per role; topology only names which roles can appear.
Role = str
ROLES: Tuple[Role, ...] = ("streamer", "worker", "tail", "leader", "hub",
                           "head")


@dataclass(frozen=True)
class MeshSpec:
    tp_axis: int = 1
    pp_axis: int = 1


@dataclass(frozen=True)
class StateBufferSpec:
    name: str
    tile_count: int
    dtype: str = "bf16"
    # Empty binds the state to every worker core. A non-empty tuple binds
    # it only to the listed nodes' cores, keeping per-node resident state
    # (recurrent S, conv state) off cores that never read it.
    nodes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WeightRingSpec:
    name: str
    stages: int
    tile_count: int
    dtype: str = "bf16"
    # Same meaning as StateBufferSpec.nodes: empty = all workers, listed
    # = only those nodes carry this weight ring.
    nodes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisProblem:
    model_name: str
    layer: DecoderLayerSpec
    hardware: HardwareSpec
    mesh: MeshSpec = MeshSpec()
    state_buffers: Tuple[StateBufferSpec, ...] = ()
    weight_rings: Tuple[WeightRingSpec, ...] = ()
