#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tt_pk/IR/TTPKDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::tt_pk::TTPKDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "tt-pk-opt driver\n", registry));
}
