// RUN: tt-pk-opt %s | FileCheck %s
// CHECK-LABEL: tt_pk.program
tt_pk.program {
  %c1 = tt_pk.core_set [0, 0, 1, 0] : !tt_pk.core_set
  %c2 = tt_pk.core_set [2, 0, 3, 0] : !tt_pk.core_set
  tt_pk.kernel "k1" on %c1 : !tt_pk.core_set
  tt_pk.kernel "k2" on %c2 : !tt_pk.core_set
}
