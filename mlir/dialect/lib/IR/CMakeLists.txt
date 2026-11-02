add_mlir_dialect_library(MLIRTTPKDialect
  TTPKDialect.cpp
  TTPKOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/tt_pk

  DEPENDS
  TTPKOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
