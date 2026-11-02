#ifndef TTPK_IR_BASE
#define TTPK_IR_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TTPK_Dialect : Dialect {
  let name = "tt_pk";
  let cppNamespace = "::mlir::tt_pk";
  let summary = "Tensix Persistent Kernel materialized-program dialect";
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 0;
}

class TTPK_Op<string mnemonic, list<Trait> traits = []>
    : Op<TTPK_Dialect, mnemonic, traits>;

class TTPK_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTPK_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTPK_CoreSetType : TTPK_Type<"CoreSet", "core_set"> {
  let summary = "A set of Tensix core coordinates";
  let description = [{
    Opaque handle for a set of Tensix cores. The producing op carries the
    coordinate list; this type threads the handle through SSA use-defs so
    passes can reason about placement without re-deriving sets.
  }];
}

#endif // TTPK_IR_BASE
