#ifndef TTPK_IR_OPS
#define TTPK_IR_OPS

include "tt_pk/IR/TTPKBase.td"
include "mlir/IR/BuiltinAttributes.td"

def TTPK_CoreSetOp : TTPK_Op<"core_set", [Pure]> {
  let summary = "Define a set of Tensix cores";
  let description = [{
    The `cores` attribute is a flat `[x0, y0, x1, y1, ...]` list of
    non-negative core coordinates. Length must be even; order is not
    semantically significant.
  }];

  let arguments = (ins DenseI64ArrayAttr:$cores);
  let results = (outs TTPK_CoreSetType:$result);

  let assemblyFormat = "$cores attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TTPK_KernelOp : TTPK_Op<"kernel"> {
  let summary = "Launch a named resident kernel on a core set";
  let description = [{
    `kernel_name` identifies the kernel source; `core_set` is the operand
    producing the placement.
  }];

  let arguments = (ins StrAttr:$kernel_name, TTPK_CoreSetType:$core_set);
  let results = (outs);

  let assemblyFormat =
      "$kernel_name `on` $core_set attr-dict `:` type($core_set)";
}

def TTPK_ProgramOp : TTPK_Op<"program", [NoTerminator]> {
  let summary = "Container for a materialized Tensix Persistent Program";
  let description = [{
    Holds a region of placement and kernel ops. The verifier enforces that
    all contained `tt_pk.core_set` values are pairwise disjoint, so
    placement bugs surface at parse time.
  }];

  let regions = (region AnyRegion:$body);
  let results = (outs);

  let assemblyFormat = "$body attr-dict";
  let hasVerifier = 1;
}

#endif // TTPK_IR_OPS
