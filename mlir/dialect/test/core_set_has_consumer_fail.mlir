// RUN: not tt-pk-opt %s 2>&1 | FileCheck %s
// CHECK: core_set is not launched by any tt_pk.kernel op
tt_pk.program {
  %c1 = tt_pk.core_set [0, 0] : !tt_pk.core_set
  %c2 = tt_pk.core_set [1, 0] : !tt_pk.core_set
  tt_pk.kernel "k1" on %c1 : !tt_pk.core_set
}
