// RUN: tt-pk-opt %s | FileCheck %s
// CP2 fixture: hand-written tt_pk program for one Qwen3.6-27B decoder layer
// on Blackhole P150. Core-set coordinates are placeholders; what the test
// pins is the shape (disjoint sets per kernel) and the parse+verify path.
//
// CHECK-LABEL: tt_pk.program
// CHECK: tt_pk.core_set
// CHECK: tt_pk.kernel "qwen36.qkv_proj"
// CHECK: tt_pk.kernel "qwen36.mlp_down"
tt_pk.program {
  %qkv   = tt_pk.core_set [0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0] : !tt_pk.core_set
  %attn  = tt_pk.core_set [0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1] : !tt_pk.core_set
  %o     = tt_pk.core_set [0, 2, 1, 2, 2, 2, 3, 2]                         : !tt_pk.core_set
  %gate  = tt_pk.core_set [0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 6, 3, 7, 3] : !tt_pk.core_set
  %up    = tt_pk.core_set [0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7, 4] : !tt_pk.core_set
  %down  = tt_pk.core_set [0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5, 7, 5] : !tt_pk.core_set

  tt_pk.kernel "qwen36.qkv_proj"  on %qkv   : !tt_pk.core_set
  tt_pk.kernel "qwen36.attention" on %attn  : !tt_pk.core_set
  tt_pk.kernel "qwen36.o_proj"    on %o     : !tt_pk.core_set
  tt_pk.kernel "qwen36.mlp_gate"  on %gate  : !tt_pk.core_set
  tt_pk.kernel "qwen36.mlp_up"    on %up    : !tt_pk.core_set
  tt_pk.kernel "qwen36.mlp_down"  on %down  : !tt_pk.core_set
}
