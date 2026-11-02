// RUN: not tt-pk-opt %s 2>&1 | FileCheck %s
// CHECK: core (1,0) listed more than once in the same core_set
tt_pk.program {
  %c1 = tt_pk.core_set [0, 0, 1, 0, 1, 0] : !tt_pk.core_set
  tt_pk.kernel "k1" on %c1 : !tt_pk.core_set
}
