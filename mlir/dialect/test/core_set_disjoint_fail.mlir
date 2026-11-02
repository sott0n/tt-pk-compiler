// RUN: not tt-pk-opt %s 2>&1 | FileCheck %s
// CHECK: core (1,0) already assigned to another core_set
tt_pk.program {
  %c1 = tt_pk.core_set [0, 0, 1, 0] : !tt_pk.core_set
  %c2 = tt_pk.core_set [1, 0, 2, 0] : !tt_pk.core_set
  tt_pk.kernel "k1" on %c1 : !tt_pk.core_set
  tt_pk.kernel "k2" on %c2 : !tt_pk.core_set
}
