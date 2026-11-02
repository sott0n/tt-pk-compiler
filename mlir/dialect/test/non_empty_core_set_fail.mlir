// RUN: not tt-pk-opt %s 2>&1 | FileCheck %s
// CHECK: core_set must list at least one core
tt_pk.program {
  %c1 = tt_pk.core_set [] : !tt_pk.core_set
  tt_pk.kernel "k1" on %c1 : !tt_pk.core_set
}
