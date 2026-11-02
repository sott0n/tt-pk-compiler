#include "tt_pk/IR/TTPKDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "tt_pk/IR/TTPKOps.h"
#include "tt_pk/IR/TTPKTypes.h"

using namespace mlir;
using namespace mlir::tt_pk;

#include "tt_pk/IR/TTPKDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tt_pk/IR/TTPKTypes.cpp.inc"

#define GET_OP_CLASSES
#include "tt_pk/IR/TTPKOps.cpp.inc"

void TTPKDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tt_pk/IR/TTPKTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tt_pk/IR/TTPKOps.cpp.inc"
      >();
}
