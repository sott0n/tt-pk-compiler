#ifndef TTPK_IR_TTPKOPS_H
#define TTPK_IR_TTPKOPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tt_pk/IR/TTPKTypes.h"

#define GET_OP_CLASSES
#include "tt_pk/IR/TTPKOps.h.inc"

#endif // TTPK_IR_TTPKOPS_H
