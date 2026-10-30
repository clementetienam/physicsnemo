#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# syntax_check.sh -- compile every CUDA source without MPI, HYPRE or a GPU.
#
# Why this exists
# ---------------
# The simulator links against MPI, HYPRE and optionally AmgX, so a full build
# needs a cluster.  That makes the edit/compile loop slow at exactly the moment
# it should be fast: after changing a kernel, when the question is only "does
# this compile", not "does it run".  A kernel edit that reaches the cluster
# with a typo in it costs a queue wait to find out.
#
# So: stub the three headers down to the symbols these sources actually name,
# and let nvcc do the rest.  Nothing here produces a runnable binary -- the
# objects are thrown away.  What it catches is what it is for: type errors,
# missing members, wrong signatures, and the arithmetic of a kernel body.
#
#   ./scripts/syntax_check.sh              # every CUDA and C++ source
#   ./scripts/syntax_check.sh src/main.cu  # just these
#   NVRS_SYNTAX_FLAGS=-DNVRS_HAVE_TENSORCORE=1 ./scripts/syntax_check.sh  # the option on
#
# CUDA_ARCH may be set; it defaults to a value with double atomicAdd, which
# the residual reduction in every assembly kernel needs.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
arch=${CUDA_ARCH:-70}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/stub"

cat > "$work/stub/mpi.h" <<'EOF'
#pragma once
#include <cstddef>
typedef int MPI_Comm; typedef int MPI_Datatype; typedef int MPI_Op;
typedef int MPI_Request; typedef int MPI_Info;
typedef struct { int a, b, c; } MPI_Status;
#define MPI_COMM_WORLD 0
#define MPI_DOUBLE 1
#define MPI_LONG 2
#define MPI_INT 3
#define MPI_BYTE 4
#define MPI_CHAR 6
#define MPI_LONG_LONG 5
#define MPI_SUM 10
#define MPI_MAX 11
#define MPI_MIN 12
#define MPI_IN_PLACE ((void*)-1)
#define MPI_STATUSES_IGNORE ((MPI_Status*)0)
#define MPI_STATUS_IGNORE ((MPI_Status*)0)
#define MPI_COMM_TYPE_SHARED 1
#define MPI_INFO_NULL 0
#define MPI_MAX_PROCESSOR_NAME 256
#define MPI_THREAD_FUNNELED 1
#define MPI_THREAD_SINGLE 0
#define MPI_THREAD_MULTIPLE 3
#define MPI_FLOAT 8
#define MPI_SUCCESS 0
extern "C" {
int MPI_Testall(int, MPI_Request*, int*, MPI_Status*);
int MPI_Cancel(MPI_Request*);
int MPI_Request_free(MPI_Request*);
int MPI_Gatherv(const void*, int, MPI_Datatype, void*, const int*, const int*,
                MPI_Datatype, int, MPI_Comm);
int MPI_Allreduce(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
int MPI_Allgather(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
int MPI_Reduce(const void*, void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm);
int MPI_Barrier(MPI_Comm);
int MPI_Comm_rank(MPI_Comm, int*);
int MPI_Comm_size(MPI_Comm, int*);
int MPI_Comm_split_type(MPI_Comm, int, int, MPI_Info, MPI_Comm*);
int MPI_Comm_free(MPI_Comm*);
int MPI_Get_processor_name(char*, int*);
int MPI_Isend(const void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*);
int MPI_Irecv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*);
int MPI_Waitall(int, MPI_Request*, MPI_Status*);
int MPI_Init(int*, char***);
int MPI_Init_thread(int*, char***, int, int*);
int MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm);
int MPI_Gather(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
int MPI_Abort(MPI_Comm, int);
int MPI_Finalize(void);
double MPI_Wtime(void);
}
EOF

cat > "$work/stub/HYPRE.h" <<'EOF'
#pragma once
typedef int HYPRE_Int; typedef double HYPRE_Real;
typedef long long HYPRE_BigInt;
// cpr_solver.cu static_asserts that PrecReal == HYPRE_Complex.  PrecReal
// follows NVRS_FP32_PRECOND, which defaults to 1, so the stub follows it too --
// a real build links an FP32 HYPRE to match.
#if !defined(NVRS_FP32_PRECOND) || NVRS_FP32_PRECOND
typedef float HYPRE_Complex;
#else
typedef double HYPRE_Complex;
#endif
typedef void* HYPRE_Solver; typedef void* HYPRE_ParCSRMatrix;
typedef void* HYPRE_ParVector;
#define HYPRE_RELEASE_VERSION "0.0.0-stub"
#define HYPRE_USING_CUDA 1
#define HYPRE_USING_GPU 1
#define HYPRE_MEMORY_DEVICE 1
#define HYPRE_EXEC_DEVICE 1
#define HYPRE_PARCSR 5555
#define HYPRE_DIRECT_VEC 1
extern "C" {
int HYPRE_Init(void);
int HYPRE_Finalize(void);
int HYPRE_SetMemoryLocation(int);
int HYPRE_SetExecutionPolicy(int);
int HYPRE_SetSpGemmUseVendor(int);
int HYPRE_SetSpMVUseVendor(int);
int HYPRE_SetGpuAwareMPI(int);
int HYPRE_BoomerAMGSetup(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
int HYPRE_BoomerAMGSolve(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
int HYPRE_BoomerAMGGetNumIterations(HYPRE_Solver, HYPRE_Int*);
}
EOF
# cpr_solver.cu reaches for HYPRE's IJ interface, which is a larger surface
# than the rest: matrices, vectors and the solvers built on them.
cat > "$work/stub/HYPRE_IJ_mv.h" <<'EOF'
#pragma once
#include "HYPRE.h"
typedef void* HYPRE_IJMatrix;
typedef void* HYPRE_IJVector;
extern "C" {
int HYPRE_IJMatrixCreate(MPI_Comm, HYPRE_BigInt, HYPRE_BigInt, HYPRE_BigInt, HYPRE_BigInt, HYPRE_IJMatrix*);
int HYPRE_IJMatrixDestroy(HYPRE_IJMatrix);
int HYPRE_IJMatrixSetObjectType(HYPRE_IJMatrix, HYPRE_Int);
int HYPRE_IJMatrixInitialize(HYPRE_IJMatrix);
int HYPRE_IJMatrixInitialize_v2(HYPRE_IJMatrix, HYPRE_Int);
int HYPRE_IJMatrixAssemble(HYPRE_IJMatrix);
int HYPRE_IJMatrixGetObject(HYPRE_IJMatrix, void**);
int HYPRE_IJMatrixSetValues(HYPRE_IJMatrix, HYPRE_Int, HYPRE_Int*, const HYPRE_BigInt*, const HYPRE_BigInt*, const HYPRE_Complex*);
int HYPRE_IJMatrixSetConstantValues(HYPRE_IJMatrix, HYPRE_Complex);
int HYPRE_IJMatrixSetOMPFlag(HYPRE_IJMatrix, HYPRE_Int);
int HYPRE_IJMatrixSetRowSizes(HYPRE_IJMatrix, const HYPRE_Int*);
int HYPRE_IJMatrixSetDiagOffdSizes(HYPRE_IJMatrix, const HYPRE_Int*, const HYPRE_Int*);
int HYPRE_IJVectorCreate(MPI_Comm, HYPRE_BigInt, HYPRE_BigInt, HYPRE_IJVector*);
int HYPRE_IJVectorDestroy(HYPRE_IJVector);
int HYPRE_IJVectorSetObjectType(HYPRE_IJVector, HYPRE_Int);
int HYPRE_IJVectorInitialize(HYPRE_IJVector);
int HYPRE_IJVectorInitialize_v2(HYPRE_IJVector, HYPRE_Int);
int HYPRE_IJVectorAssemble(HYPRE_IJVector);
int HYPRE_IJVectorGetObject(HYPRE_IJVector, void**);
int HYPRE_IJVectorSetValues(HYPRE_IJVector, HYPRE_Int, const HYPRE_BigInt*, const HYPRE_Complex*);
int HYPRE_IJVectorGetValues(HYPRE_IJVector, HYPRE_Int, const HYPRE_BigInt*, HYPRE_Complex*);
}
EOF
# BoomerAMG and MGR: cpr_solver.cu configures both, so the setters have to be
# declared even though nothing is ever called.  Uniform (solver, int) and
# (solver, double) shapes cover all but a handful.
cat > "$work/stub/HYPRE_parcsr_ls.h" <<'EOF'
#pragma once
#include "HYPRE.h"
#define NVRS_SET_I(n) extern "C" int n(HYPRE_Solver, HYPRE_Int);
#define NVRS_SET_D(n) extern "C" int n(HYPRE_Solver, HYPRE_Real);
extern "C" {
int HYPRE_BoomerAMGCreate(HYPRE_Solver*);
int HYPRE_BoomerAMGDestroy(HYPRE_Solver);
int HYPRE_MGRCreate(HYPRE_Solver*);
int HYPRE_MGRDestroy(HYPRE_Solver);
int HYPRE_MGRSetCoarseSolver(HYPRE_Solver, int (*)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector), int (*)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector), HYPRE_Solver);
int HYPRE_MGRSetCpointsByBlock(HYPRE_Solver, HYPRE_Int, HYPRE_Int, HYPRE_Int*, HYPRE_Int**);
int HYPRE_BoomerAMGGetNumIterations(HYPRE_Solver, HYPRE_Int*);
int HYPRE_MGRSetup(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
int HYPRE_MGRSolve(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
int HYPRE_BoomerAMGSetCycleRelaxType(HYPRE_Solver, HYPRE_Int, HYPRE_Int);
}
NVRS_SET_I(HYPRE_BoomerAMGSetAggNumLevels)
NVRS_SET_I(HYPRE_BoomerAMGSetAggPMaxElmts)
NVRS_SET_D(HYPRE_BoomerAMGSetAggTruncFactor)
NVRS_SET_I(HYPRE_BoomerAMGSetCoarsenType)
NVRS_SET_I(HYPRE_BoomerAMGSetInterpType)
NVRS_SET_I(HYPRE_BoomerAMGSetKeepTranspose)
NVRS_SET_I(HYPRE_BoomerAMGSetMaxCoarseSize)
NVRS_SET_I(HYPRE_BoomerAMGSetMaxIter)
NVRS_SET_I(HYPRE_BoomerAMGSetNumSweeps)
NVRS_SET_I(HYPRE_BoomerAMGSetPMaxElmts)
NVRS_SET_I(HYPRE_BoomerAMGSetPrintLevel)
NVRS_SET_I(HYPRE_BoomerAMGSetRelaxType)
NVRS_SET_D(HYPRE_BoomerAMGSetStrongThreshold)
NVRS_SET_D(HYPRE_BoomerAMGSetTol)
NVRS_SET_D(HYPRE_BoomerAMGSetTruncFactor)
NVRS_SET_I(HYPRE_MGRSetGlobalSmoothType)
NVRS_SET_I(HYPRE_MGRSetInterpType)
NVRS_SET_I(HYPRE_MGRSetMaxGlobalSmoothIters)
NVRS_SET_I(HYPRE_MGRSetMaxIter)
NVRS_SET_I(HYPRE_MGRSetNonCpointsToFpoints)
NVRS_SET_I(HYPRE_MGRSetNumRelaxSweeps)
NVRS_SET_I(HYPRE_MGRSetPrintLevel)
NVRS_SET_I(HYPRE_MGRSetRelaxType)
NVRS_SET_D(HYPRE_MGRSetTol)
EOF
cp "$work/stub/HYPRE.h" "$work/stub/HYPRE_utilities.h"

if [ $# -gt 0 ]; then
  files=("$@")
else
  mapfile -t files < <(cd "$here" && ls src/*.cu src/*.cpp)
fi

fail=0
for f in "${files[@]}"; do
  printf '  %-34s ' "$f"
  if out=$(cd "$here" && nvcc -std=c++17 -arch="sm_$arch" --extended-lambda ${NVRS_SYNTAX_FLAGS:-} \
             -I include -I "$work/stub" -c "$f" -o "$work/o.obj" 2>&1); then
    echo "ok"
  else
    echo "FAIL"
    echo "$out" | grep -E "error|warning: .*\[-W" | head -12 | sed 's/^/      /'
    fail=$((fail + 1))
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "all ${#files[@]} sources compile (sm_$arch, stubbed MPI/HYPRE)"
else
  echo "$fail of ${#files[@]} sources failed"
fi
exit "$fail"
