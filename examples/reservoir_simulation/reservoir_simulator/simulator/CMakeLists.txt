# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# NVRS-CUDA — GPU black-oil / compositional reservoir simulator.
# Links the FP32 (or FP64) HYPRE install the same way GEOS does: an imported
# static/shared libHYPRE plus the CUDA math libraries and MPI.
#
#   cmake -S . -B build \
#         -DHYPRE_DIR=$HYPRE_FP32_DIR \
#         -DCMAKE_CUDA_ARCHITECTURES=100 \
#         -DNVRS_MIXED_PRECISION=ON
#   cmake --build build -j
cmake_minimum_required(VERSION 3.18)

# --- CUDA architecture ------------------------------------------------------
# This MUST be decided before project(), because enabling the CUDA language is
# what sets CMAKE_CUDA_ARCHITECTURES to the compiler default -- 52 with this
# toolkit.  A later `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` is then always
# false, the detection never runs, and the build silently targets sm_52, which
# has no double-precision atomicAdd.  Every kernel that accumulates a residual
# then fails to compile with "no instance of overloaded function atomicAdd
# matches the argument list (Real *, double)", which points at the atomics and
# not at the architecture that caused it.
#
# Ask the GPU directly rather than using "native": that keyword needs CMake
# 3.24 and the cluster images ship 3.22.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _nvrs_cc RESULT_VARIABLE _nvrs_cc_rc
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_nvrs_cc_rc EQUAL 0 AND _nvrs_cc)
    string(REGEX MATCH "[0-9]+\.[0-9]+" _nvrs_cc0 "${_nvrs_cc}")
    string(REPLACE "." "" _nvrs_arch "${_nvrs_cc0}")
  endif()
  # Detecting the device is only half of it: this toolkit must also be able to
  # target it.  sm_100 (GB200) needs CUDA 12.8, and asking an older nvcc for it
  # fails with "Unsupported gpu architecture", which reads like a broken build
  # rather than a toolkit too old for the machine it is standing on.
  execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
    OUTPUT_VARIABLE _nvrs_known RESULT_VARIABLE _nvrs_known_rc
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT _nvrs_known_rc EQUAL 0 OR NOT _nvrs_known)
    execute_process(
      COMMAND nvcc --list-gpu-arch
      OUTPUT_VARIABLE _nvrs_known RESULT_VARIABLE _nvrs_known_rc
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  string(REPLACE "compute_" "" _nvrs_known "${_nvrs_known}")
  string(REPLACE "\n" ";" _nvrs_known "${_nvrs_known}")

  # A GPU newer than the toolkit.  Hecate's Vera Rubin reports 10.7 and no CUDA
  # yet has an sm_107 target: 13.1 knows 100, 103, 110, 120, 121 and nothing at
  # 107.
  #
  # The answer is PTX.  A "-virtual" entry makes CMake emit code=compute_NN
  # rather than code=sm_NN, and the driver JIT-compiles it at load.  It must be
  # the newest known architecture NOT NEWER THAN the device: the driver will JIT
  # compute_103 onto sm_107, but PTX for compute_121 cannot run on a 10.7 device
  # at all.
  #
  # The old branch fell through to the fat binary below and warned that kernels
  # would "run through PTX JIT if at all".  They would not have: that list is
  # SASS only, so on sm_107 the build succeeded and then failed at launch with
  # no cubin and no PTX to fall back on.
  set(_nvrs_ptx "")
  if(_nvrs_arch AND _nvrs_known AND NOT "${_nvrs_arch}" IN_LIST _nvrs_known)
    set(_nvrs_le "")
    foreach(_k IN LISTS _nvrs_known)
      if(_k MATCHES "^[0-9]+$" AND NOT _k GREATER _nvrs_arch)
        if(NOT _nvrs_le OR _k GREATER _nvrs_le)
          set(_nvrs_le "${_k}")
        endif()
      endif()
    endforeach()
    if(_nvrs_le)
      message(STATUS
        "This machine reports compute capability ${_nvrs_arch}, which this CUDA "
        "cannot target. Building SASS for sm_${_nvrs_le}, which loads "
        "directly, plus PTX for compute_${_nvrs_le} as a fallback.")
      # SASS first, PTX second.  The cubin is forward-compatible across
      # minor revisions within a major capability, so it loads on this
      # device with no JIT; the PTX is the fallback if it does not.
      set(_nvrs_ptx "${_nvrs_le};${_nvrs_le}-virtual")
    else()
      message(WARNING
        "This machine reports compute capability ${_nvrs_arch}, and this CUDA "
        "knows no architecture at or below it. Falling back to a fat binary, "
        "which will not run here.")
    endif()
    set(_nvrs_arch "")
  endif()

  if(_nvrs_arch)
    set(CMAKE_CUDA_ARCHITECTURES "${_nvrs_arch}")
  elseif(_nvrs_ptx)
    set(CMAKE_CUDA_ARCHITECTURES "${_nvrs_ptx}")
  else()
    # No GPU visible (a login node, or a container without the driver): a fat
    # binary covering the architectures this code actually runs on, kept to the
    # ones this toolkit knows.  sm_60 is the floor because of the double
    # atomicAdd; sm_100 is here because Ptyche is GB200.
    set(_nvrs_want 70 80 90 100 120)
    set(CMAKE_CUDA_ARCHITECTURES "")
    foreach(_a IN LISTS _nvrs_want)
      if(NOT _nvrs_known OR "${_a}" IN_LIST _nvrs_known)
        list(APPEND CMAKE_CUDA_ARCHITECTURES "${_a}")
      endif()
    endforeach()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
    endif()
    # PTX for the newest of them as well, so a binary built with no GPU in sight
    # still runs on hardware that did not exist when it was built.  SASS alone
    # cannot.
    set(_nvrs_top "")
    foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
      if(_a MATCHES "^[0-9]+$" AND (NOT _nvrs_top OR _a GREATER _nvrs_top))
        set(_nvrs_top "${_a}")
      endif()
    endforeach()
    if(_nvrs_top)
      list(APPEND CMAKE_CUDA_ARCHITECTURES "${_nvrs_top}-virtual")
    endif()
  endif()
endif()
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$" AND
   CMAKE_CUDA_ARCHITECTURES LESS 60)
  message(FATAL_ERROR
    "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} is below sm_60, "
    "which has no double-precision atomicAdd.  Pass the real architecture, "
    "e.g. -DCMAKE_CUDA_ARCHITECTURES=100 for GB200.")
endif()

project(nvrs_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

message(STATUS "CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NVRS_MIXED_PRECISION "FP32 HYPRE AMG preconditioner (mixed CPR)" ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --- HYPRE (GEOS-style imported library) -----------------------------------
# Point -DHYPRE_DIR at the install prefix produced by
# Mixed_precision/scripts/docker/install_hypre_sm100.sh (fp32/ or fp64/).
if(NOT HYPRE_DIR)
  message(FATAL_ERROR "Set -DHYPRE_DIR=<hypre install prefix> (contains include/ and lib/)")
endif()
find_path(HYPRE_INCLUDE_DIR HYPRE.h HINTS ${HYPRE_DIR}/include REQUIRED)
find_library(HYPRE_LIBRARY NAMES HYPRE libHYPRE.a HYPRE.a
             HINTS ${HYPRE_DIR}/lib ${HYPRE_DIR}/lib64 REQUIRED)
message(STATUS "HYPRE include : ${HYPRE_INCLUDE_DIR}")
message(STATUS "HYPRE library : ${HYPRE_LIBRARY}")

# --- AmgX (optional) -------------------------------------------------------
# -DAMGX_DIR=<prefix>, or the AMGX_DIR environment variable that
# benchmark/install_amgx.sh writes into amgx_env.sh.  Found -> LINSOLVER AMGX
# works and is the deck default.  Not found -> src/amgx_backend.cu compiles to
# nothing, the binary runs on HYPRE, and it says so at start-up rather than
# silently ignoring the deck.
if(NOT AMGX_DIR)
  set(AMGX_DIR "$ENV{AMGX_DIR}")
endif()
if(AMGX_DIR)
  find_path(AMGX_INCLUDE_DIR amgx_c.h HINTS ${AMGX_DIR}/include)
  find_library(AMGX_LIBRARY NAMES amgxsh amgx
               HINTS ${AMGX_DIR}/lib ${AMGX_DIR}/lib64)
endif()
if(AMGX_INCLUDE_DIR AND AMGX_LIBRARY)
  set(NVRS_HAVE_AMGX ON)
  message(STATUS "AmgX include  : ${AMGX_INCLUDE_DIR}")
  message(STATUS "AmgX library  : ${AMGX_LIBRARY}")
else()
  set(NVRS_HAVE_AMGX OFF)
  message(STATUS "AmgX          : not found -- building HYPRE-only")
endif()

# --- Ginkgo (optional) -----------------------------------------------------
# -DGINKGO_DIR=<prefix>, or the GINKGO_DIR environment variable that
# benchmark/install_ginkgo.sh writes into ginkgo_env.sh.  Found -> LINSOLVER
# GINKGO works, with an FP16 multigrid by default.  Not found -> the backend
# compiles to stubs and such a deck falls back to AmgX, saying so.
if(NOT GINKGO_DIR)
  set(GINKGO_DIR "$ENV{GINKGO_DIR}")
endif()
if(GINKGO_DIR)
  find_package(Ginkgo CONFIG QUIET
               HINTS ${GINKGO_DIR} ${GINKGO_DIR}/lib/cmake/Ginkgo
                     ${GINKGO_DIR}/lib64/cmake/Ginkgo)
endif()
if(Ginkgo_FOUND)
  set(NVRS_HAVE_GINKGO ON)
  message(STATUS "Ginkgo        : ${Ginkgo_VERSION} at ${GINKGO_DIR}")
else()
  set(NVRS_HAVE_GINKGO OFF)
  message(STATUS "Ginkgo        : not found -- LINSOLVER GINKGO falls back to AmgX")
endif()

# --- CUTLASS (optional, header only) ----------------------------------------
# -DCUTLASS_DIR=<checkout>.  Found -> JACOBIAN TENSORCORE can issue the batched
# operator product as a TF32 Tensor Core GEMM on SM80 and newer.  Not found ->
# the same path runs as an FP32 product on CUDA cores, which works on any GPU.
if(NOT CUTLASS_DIR)
  set(CUTLASS_DIR "$ENV{CUTLASS_DIR}")
endif()
if(CUTLASS_DIR AND EXISTS ${CUTLASS_DIR}/include/cutlass/cutlass.h)
  set(NVRS_HAVE_CUTLASS ON)
  message(STATUS "CUTLASS       : ${CUTLASS_DIR} (TENSORCORE uses TF32 on SM80+)")
else()
  set(NVRS_HAVE_CUTLASS OFF)
  message(STATUS "CUTLASS       : not found -- TENSORCORE runs FP32 on CUDA cores")
endif()

# --- NCCL (optional) --------------------------------------------------------
# Stream-ordered communication for the Krylov tail, so CUDAGRAPH can capture
# it on several ranks.  -DNCCL_DIR=<prefix> or NCCL_HOME; not found -> MPI.
if(NOT NCCL_DIR)
  set(NCCL_DIR "$ENV{NCCL_HOME}")
endif()
# The NVIDIA HPC SDK keeps NCCL in comm_libs/nccl beside its compilers.
get_filename_component(_nvcc_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
file(GLOB _hpcsdk_nccl "${_nvcc_bin}/../../comm_libs/nccl"
     "${_nvcc_bin}/../../../comm_libs/nccl" "${_nvcc_bin}/../../comm_libs/*/nccl"
     "$ENV{NVHPC_ROOT}/comm_libs/nccl"
     "/opt/nvidia/hpc_sdk/*/*/comm_libs/nccl" "/opt/nvidia/hpc_sdk/*/*/comm_libs/*/nccl")
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_DIR}/include ${CUDAToolkit_INCLUDE_DIRS}
          PATHS ${_hpcsdk_nccl} PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES nccl HINTS ${NCCL_DIR}/lib ${NCCL_DIR}/lib64
             ${CUDAToolkit_LIBRARY_DIR} PATHS ${_hpcsdk_nccl} PATH_SUFFIXES lib lib64)
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
  set(NVRS_HAVE_NCCL ON)
  message(STATUS "NCCL          : ${NCCL_LIBRARY}")
else()
  set(NVRS_HAVE_NCCL OFF)
  message(STATUS "NCCL          : not found -- multi-rank Krylov tail keeps MPI")
endif()

add_executable(nvrs_cuda
  src/main.cu
  src/deck.cpp
  src/sequential.cu
  src/ecl_io.cpp
  src/halo.cu
  src/physics_host.cu
  src/assemble_blackoil.cu
  src/assemble_compositional.cu
  src/cpr_solver.cu
  src/amgx_backend.cu
  src/tc_jacobian.cu
  src/ginkgo_backend.cpp
  src/ginkgo_kernels.cu
  src/field_model.cu
  src/halo_conn.cu
  src/field_partition.cpp
  src/units_field.cpp
  src/cpgrid.cpp
  src/schedule.cpp
  src/equil.cpp
  src/pvt_tables.cpp
  src/pvt_upload.cu
  src/io.cu
  src/well_report.cu
  src/nccl_comm.cu)

target_include_directories(nvrs_cuda PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${HYPRE_INCLUDE_DIR}
  ${MPI_CXX_INCLUDE_DIRS})
if(NVRS_HAVE_AMGX)
  target_include_directories(nvrs_cuda PRIVATE ${AMGX_INCLUDE_DIR})
endif()

target_compile_definitions(nvrs_cuda PRIVATE
  NVRS_FP32_PRECOND=$<IF:$<BOOL:${NVRS_MIXED_PRECISION}>,1,0>)
if(NVRS_HAVE_AMGX)
  target_compile_definitions(nvrs_cuda PRIVATE NVRS_HAVE_AMGX=1)
endif()
if(NVRS_HAVE_GINKGO)
  target_compile_definitions(nvrs_cuda PRIVATE NVRS_HAVE_GINKGO=1)
endif()
if(NVRS_HAVE_NCCL)
  target_include_directories(nvrs_cuda PRIVATE ${NCCL_INCLUDE_DIR})
  target_compile_definitions(nvrs_cuda PRIVATE NVRS_HAVE_NCCL=1)
  target_link_libraries(nvrs_cuda PRIVATE ${NCCL_LIBRARY})
endif()

# Separable compilation so device code across TUs links cleanly.
set_target_properties(nvrs_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(nvrs_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 --extended-lambda>)

if(NVRS_HAVE_CUTLASS)
  target_include_directories(nvrs_cuda PRIVATE ${CUTLASS_DIR}/include)
  target_compile_definitions(nvrs_cuda PRIVATE NVRS_HAVE_CUTLASS=1)
endif()

target_link_libraries(nvrs_cuda PRIVATE
  ${HYPRE_LIBRARY}
  CUDA::cusparse CUDA::cublas CUDA::cusolver CUDA::cudart
  MPI::MPI_CXX)
if(NVRS_HAVE_AMGX)
  target_link_libraries(nvrs_cuda PRIVATE ${AMGX_LIBRARY})
endif()
if(NVRS_HAVE_GINKGO)
  target_link_libraries(nvrs_cuda PRIVATE Ginkgo::ginkgo)
endif()

install(TARGETS nvrs_cuda RUNTIME DESTINATION bin)
