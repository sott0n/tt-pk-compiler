get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(tt-pk-opt
  tt-pk-opt.cpp
  compat_arc4random.c
)
set_source_files_properties(compat_arc4random.c PROPERTIES LANGUAGE C)

llvm_update_compile_flags(tt-pk-opt)
target_link_libraries(tt-pk-opt
  PRIVATE
  MLIRTTPKDialect
  MLIRRegisterAllDialects
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
)

mlir_check_all_link_libraries(tt-pk-opt)
