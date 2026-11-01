cmake_minimum_required(VERSION 3.22)
project(tt_pk LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Point at an existing LLVM/MLIR build. Resolution order:
#   1. -DLLVM_DIR=... / -DMLIR_DIR=...
#   2. $ENV{TT_PK_LLVM_BUILD_DIR} / lib/cmake/{llvm,mlir}
#   3. third_party/tt-mlir/env/build/... (populated when the submodule
#      has been built in place)
if(NOT DEFINED LLVM_DIR OR NOT DEFINED MLIR_DIR)
  set(_candidates "")
  if(DEFINED ENV{TT_PK_LLVM_BUILD_DIR})
    list(APPEND _candidates "$ENV{TT_PK_LLVM_BUILD_DIR}")
  endif()
  list(APPEND _candidates
    "${CMAKE_SOURCE_DIR}/../third_party/tt-mlir/env/build/llvm-project-prefix/src/llvm-project-build")
  foreach(_llvm_build IN LISTS _candidates)
    if(EXISTS "${_llvm_build}/lib/cmake/mlir/MLIRConfig.cmake")
      if(NOT DEFINED LLVM_DIR)
        set(LLVM_DIR "${_llvm_build}/lib/cmake/llvm" CACHE PATH "")
      endif()
      if(NOT DEFINED MLIR_DIR)
        set(MLIR_DIR "${_llvm_build}/lib/cmake/mlir" CACHE PATH "")
      endif()
      break()
    endif()
  endforeach()
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "LLVM:     ${LLVM_DIR} (${LLVM_PACKAGE_VERSION})")
message(STATUS "MLIR:     ${MLIR_DIR}")
message(STATUS "LLVM_BIN: ${LLVM_TOOLS_BINARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(dialect)
