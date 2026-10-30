// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// cpr_solver.cu — implementation of the CPR-preconditioned mixed-precision
// Krylov solver.  Adapted and generalised from the validated reference
// Mixed_precision/src/hypre_cusolver_mixed_cpr.cu.
#include "halo_conn.hpp"
#include "linear_solver.hpp"
#include "block_spmv.cuh"
#include "nccl_comm.hpp"
#include "wellcell.hpp"

#include "halo.hpp"

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
// The residual has to reach HYPRE's vector and the correction has to come
// back, once per Krylov iteration.  Going through the IJ layer to do it costs
// 2.54 ms per apply around a 0.79 ms V-cycle -- more time than the solve it
// exists to feed.  With this header the local array can be written directly.
// If it is not installed, the IJ path below still works, just slowly.
#if defined(__has_include)
#  if __has_include(<_hypre_parcsr_mv.h>)
#    include <_hypre_parcsr_mv.h>
#    define NVRS_HYPRE_DIRECT_VEC 1
#  endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <vector>

// ===========================================================================
// HYPRE BoomerAMG wrapper on the (distributed) pressure sub-block.
//
// NOTE: this struct is deliberately defined at *global* scope, because
// linear_solver.hpp forward-declares `struct hypre_amg_impl;` outside
// `namespace nvrs` in order to keep the HYPRE headers out of the public
// interface.  Defining it inside the namespace would create a distinct type
// (`nvrs::hypre_amg_impl`) and leave `CprSolver::amg_` pointing at an
// incomplete `::hypre_amg_impl`.
// ===========================================================================
struct hypre_amg_impl {
  int cells = 0;            // local cells (= local pressure rows)
  long long row0 = 0;       // global first pressure row for this rank
  HYPRE_IJMatrix A = nullptr;
  HYPRE_ParCSRMatrix parA = nullptr;
  HYPRE_IJVector b = nullptr, x = nullptr;
  HYPRE_ParVector parb = nullptr, parx = nullptr;
  HYPRE_Solver solver = nullptr;
  // Everything handed to the HYPRE IJ interface must live in *device* memory,
  // because main.cu selects HYPRE_MEMORY_DEVICE.  These are built once (the
  // sparsity is static) and reused for every assembly.
  HYPRE_BigInt*  d_idx   = nullptr;  // global row ids,      length cells
  HYPRE_Int*     d_ncols = nullptr;  // entries per row,     length cells
  HYPRE_BigInt*  d_cols  = nullptr;  // global column ids,   length pnnz
  HYPRE_Complex* d_zero  = nullptr;  // zeros for vector init, length cells
  // HYPRE's own storage for b and x.  Taken once after assembly: the
  // partitioning never changes, so the pointers stay valid for the run.
  nvrs::PrecReal* b_dev = nullptr;
  nvrs::PrecReal* x_dev = nullptr;
  bool built = false;
};

// ===========================================================================
// MGR: HYPRE's multigrid-reduction preconditioner on the full block system.
//
// CPR above is hand-rolled: we extract a scalar pressure matrix with
// quasi-IMPES weights, run BoomerAMG on it, prolong, and clean up with a
// block-Jacobi second stage.  MGR does the same reduction inside HYPRE.
// Pressure is declared the C-point of each cell block, everything else is
// F-relaxed, and the reduced pressure system goes to BoomerAMG.
//
// Two reasons to have it.  It is what GEOS uses, so it is the configuration
// with the most reservoir mileage on it.  And the reduction is built from the
// real block structure rather than from a diagonal approximation of it, which
// is the part of our CPR most likely to be responsible for the outer solver
// stalling at 100 iterations on a hard step.
//
// The mixed-precision arrangement is unchanged: HYPRE is built FP32, so MGR is
// an FP32 preconditioner exactly as BoomerAMG was, wrapped in the same FP64
// FGMRES with the same df64-emulated operator.
//
// The one thing MGR needs that CPR did not is the FULL system in HYPRE, not
// just the pressure block.  Our coupled CSR carries LOCAL column indices
// (interior and ghost); HYPRE needs global ones.  They are recoverable without
// touching build_sparsity, because the pressure CSR already stores the global
// cell id of every member in the same order the block CSR stores its members:
//
//     block row  bs*c + a  has entries (m, v) at rowptr[bs*c+a] + m*bs + v
//     pressure row     c   has member m's global cell at pcolind[prowptr[c]+m]
//
// so the global block column is bs*pcolind[prowptr[c]+m] + v.
// ===========================================================================
struct hypre_mgr_impl {
  int cells = 0, bs = 0, nnz = 0;
  long long row0 = 0;                 // first global BLOCK row on this rank
  HYPRE_IJMatrix A = nullptr;
  HYPRE_ParCSRMatrix parA = nullptr;
  HYPRE_IJVector b = nullptr, x = nullptr;
  HYPRE_ParVector parb = nullptr, parx = nullptr;
  HYPRE_Solver mgr = nullptr;
  HYPRE_Solver coarse = nullptr;      // BoomerAMG on the reduced pressure system
  // Device metadata, built once: the sparsity never changes, only the values.
  HYPRE_BigInt* d_rows = nullptr;     // global row id per owned row
  HYPRE_Int*    d_ncols = nullptr;    // entries per row
  HYPRE_BigInt* d_cols = nullptr;     // global column id per entry
  nvrs::PrecReal* d_vals = nullptr;     // FP32 mirror of the FP64 operator
  nvrs::Real*     d_rhs = nullptr;      // FP32 rhs/solution staging
  nvrs::PrecReal* d_b32 = nullptr;
  nvrs::PrecReal* d_x32 = nullptr;
  HYPRE_Int     cidx[1] = {0};        // pressure is component 0 of each block
  HYPRE_Int*    cidx_p[1] = {nullptr};
  HYPRE_Int     ncoarse[1] = {1};
  bool built = false;
};


namespace nvrs {

namespace {
// Device-synchronised wall clock.  cudaDeviceSynchronize before both reads:
// without it the "elapsed" time is however long the launches took to queue.
//
// Only used where the region is entered once per Newton iteration (the AMG
// setup).  Anything inside the Krylov loop uses CUDA events instead -- see
// CprSolver::EventSpan.
inline double now_synced() {
  cudaDeviceSynchronize();
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

#ifndef NVRS_HAVE_AMGX
// Built without AmgX.  main.cu forces the backend back to HYPRE and says so,
// so these are unreachable -- they exist only to satisfy the linker, and
// abort rather than silently doing nothing if that guard is ever removed.
// Already inside namespace nvrs (opened above), so these definitions must not
// open it again: doing so declares nvrs::nvrs::CprSolver, which is not the
// class being defined and does not compile.
struct amgx_impl {};
void CprSolver::setup_amgx(const DeviceLinearSystem&) {
  std::fprintf(stderr, "NVRS: built without AmgX; use LINSOLVER HYPRE.\n");
  MPI_Abort(MPI_COMM_WORLD, 3);
}
void CprSolver::apply_amgx(const DeviceLinearSystem&, PrecReal*, PrecReal*) {}
void CprSolver::destroy_amgx() {}
#endif

// ---------------------------------------------------------------------------
// EventSpan: asynchronous timing of a region entered once per Krylov iteration.
// ---------------------------------------------------------------------------
void CprSolver::EventSpan::reserve_pair() {
  if (used < beg.size()) return;
  cudaEvent_t a = nullptr, b = nullptr;
  // Default flags: these events are timed, so cudaEventDisableTiming is not an
  // option.  cudaEventBlockingSync is deliberately NOT set -- the drain happens
  // once per solve and spinning briefly there is cheaper than a context switch.
  NVRS_CHECK_CUDA(cudaEventCreate(&a));
  NVRS_CHECK_CUDA(cudaEventCreate(&b));
  beg.push_back(a);
  end.push_back(b);
}

void CprSolver::EventSpan::tic() {
  reserve_pair();
  NVRS_CHECK_CUDA(cudaEventRecord(beg[used], 0));
}

void CprSolver::EventSpan::toc() {
  NVRS_CHECK_CUDA(cudaEventRecord(end[used], 0));
  ++used;
}

double CprSolver::EventSpan::drain() {
  if (used == 0) return 0.0;
  // One synchronisation for the whole batch: waiting on the last stop event
  // implies every earlier event in the same stream has completed.
  NVRS_CHECK_CUDA(cudaEventSynchronize(end[used - 1]));
  float total_ms = 0.0f;
  for (size_t i = 0; i < used; ++i) {
    float ms = 0.0f;
    NVRS_CHECK_CUDA(cudaEventElapsedTime(&ms, beg[i], end[i]));
    total_ms += ms;
  }
  used = 0;
  return (double)total_ms * 1e-3;
}

CprSolver::EventSpan::~EventSpan() {
  for (auto e : beg) cudaEventDestroy(e);
  for (auto e : end) cudaEventDestroy(e);
}


// The pressure block is handed to HYPRE as-is, so the preconditioner scalar
// type must match the scalar type HYPRE was built with.  Getting this wrong
// (an FP32 NVRS build against an FP64 HYPRE, or the reverse) silently
// reinterprets the matrix values and produces a garbage hierarchy, so catch it
// at compile time instead.
static_assert(sizeof(PrecReal) == sizeof(HYPRE_Complex),
              "PrecReal must match HYPRE_Complex: build NVRS with "
              "-DNVRS_MIXED_PRECISION=ON against an FP32 HYPRE, or OFF against "
              "an FP64 HYPRE.");

#if defined(HYPRE_USING_UMPIRE_DEVICE)
// Not an error — a CUDA-enabled Umpire works — but this configuration has a
// failure mode that is very hard to read from the symptoms, so name it here.
#pragma message( \
    "NVRS: this HYPRE is built with Umpire device pooling. If a run dies with " \
    "'CUDA ERROR (code = 1, invalid argument) at utilities/memory.c' followed " \
    "by a thrust 'invalid device ordinal', the Umpire build has no CUDA " \
    "support and its device allocator is returning host memory. Rebuild HYPRE " \
    "with HYPRE_UMPIRE=OFF (see Mixed_precision/scripts/docker/install_hypre_sm100.sh).")
#endif

// ===========================================================================
// Device kernels
// ===========================================================================
// Per-row entry counts for the pressure CSR, in HYPRE's integer type.
__global__ static void k_row_counts(const int* prowptr, HYPRE_Int* ncols,
                                    int cells) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < cells) ncols[r] = (HYPRE_Int)(prowptr[r + 1] - prowptr[r]);
}
// Widen the 32-bit global column ids to HYPRE_BigInt, which is 64-bit when
// HYPRE is configured --enable-bigint.
__global__ static void k_widen_cols(const int* pcolind, HYPRE_BigInt* out,
                                    int pnnz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pnnz) out[i] = (HYPRE_BigInt)pcolind[i];
}

__global__ static void k_residual(const Real* b, const Real* Ax, Real* r, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) r[i] = b[i] - Ax[i];
}
__global__ static void k_scale_copy(Real* dst, const Real* src, Real a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = a * src[i];
}
__global__ static void k_axpy(Real* y, Real a, const Real* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}
// Extract pressure residual r[bs*c] → rp[c]  (pressure is unknown 0 of block).
// The weights are the same quasi-IMPES weights used to build the pressure
// matrix.  CPR requires A_p = W A P with rhs = W r; building A_p with W but
// restricting the residual with e_0 (the raw first component) solves a
// different problem, and the outer Krylov stalls instead of converging.
__global__ static void k_extract_p(const Real* r, const PrecReal* w,
                                   PrecReal* rp, int cells, int bs) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const PrecReal* wc = w + (long long)bs * c;
  PrecReal s = 0;
  for (int a = 0; a < bs; ++a) s += wc[a] * (PrecReal)r[bs * c + a];
  rp[c] = s;
}
// Scatter AMG pressure correction dp[c] → z (zeroing the rest of the block).
//
// One thread owns one whole cell block.  The obvious formulation — "thread i
// zeroes z[i]; thread c also writes z[bs*c]" — is a data race across thread
// blocks: nothing orders the zeroing of z[bs*c] against the write of the
// pressure correction into that same slot, so the AMG stage would be randomly
// discarded depending on scheduling.
__global__ static void k_scatter_p(const PrecReal* dp, Real* z, int cells,
                                   int bs) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  Real* zb = z + (long long)bs * c;
  zb[0] = (Real)dp[c];
  for (int a = 1; a < bs; ++a) zb[a] = 0.0;
}
// Stage-2 additive block-Jacobi: z += Dinv (r - A z)   per bs×bs cell block.
__global__ static void k_stage2(const Real* r, Real* Az, const PrecReal* dinv,
                                Real* z, int cells, int bs) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const int b0 = bs * c;
  const PrecReal* D = dinv + (long long)c * bs * bs;
  // Local residual of this block.  Sized by kMaxBlockSize so the compositional
  // block size (NCOMP + 1, up to kMaxComponents + 1 = 9) cannot overrun it.
  PrecReal rr[kMaxBlockSize];
  for (int a = 0; a < bs; ++a) rr[a] = (PrecReal)(r[b0 + a] - Az[b0 + a]);
  for (int a = 0; a < bs; ++a) {
    PrecReal acc = 0;
    for (int q = 0; q < bs; ++q) acc += D[a * bs + q] * rr[q];
    // Az is dead after this kernel in the block-Jacobi path. Preserve D^-1 r
    // there for the optional per-well Woodbury correction rather than
    // allocating another full reservoir vector.
    Az[b0 + a] = (Real)acc;
    z[b0 + a] += (Real)acc;
  }
}
// ---------------------------------------------------------------------------
// ZLINE stage 2: block-tridiagonal line relaxation along vertical columns.
//
// SPE10-class grids couple ~100x more strongly in Z than areally (thin cells,
// large horizontal spacing), and the error a pressure multigrid cannot see is
// exactly the saturation error running along those columns.  A per-cell
// block-Jacobi correction never couples along the column; solving the block
// tridiagonal formed by each column's diagonal and Z-face blocks does, the
// columns are mutually independent (no colouring, one thread per column), and
// the factorisation is a plain block Thomas recursion.
// ---------------------------------------------------------------------------

// In-register Gauss-Jordan inverse of a bs x bs block, partial pivoting.
// Returns false on a (numerically) singular block.
__device__ static bool zl_invert(const PrecReal* M, PrecReal* Inv, int bs) {
  PrecReal A[kMaxBlockSize * kMaxBlockSize];
  for (int i = 0; i < bs * bs; ++i) A[i] = M[i];
  for (int i = 0; i < bs; ++i)
    for (int j = 0; j < bs; ++j) Inv[i * bs + j] = (i == j) ? (PrecReal)1 : (PrecReal)0;
  for (int col = 0; col < bs; ++col) {
    int piv = col;
    PrecReal best = fabsf((float)A[col * bs + col]);
    for (int rw = col + 1; rw < bs; ++rw) {
      PrecReal v = fabsf((float)A[rw * bs + col]);
      if (v > best) { best = v; piv = rw; }
    }
    if (!(best > (PrecReal)0)) return false;
    if (piv != col)
      for (int j = 0; j < bs; ++j) {
        PrecReal t = A[col * bs + j]; A[col * bs + j] = A[piv * bs + j]; A[piv * bs + j] = t;
        t = Inv[col * bs + j]; Inv[col * bs + j] = Inv[piv * bs + j]; Inv[piv * bs + j] = t;
      }
    const PrecReal d = (PrecReal)1 / A[col * bs + col];
    for (int j = 0; j < bs; ++j) { A[col * bs + j] *= d; Inv[col * bs + j] *= d; }
    for (int rw = 0; rw < bs; ++rw) {
      if (rw == col) continue;
      const PrecReal f = A[rw * bs + col];
      if (f == (PrecReal)0) continue;
      for (int j = 0; j < bs; ++j) {
        A[rw * bs + j] -= f * A[col * bs + j];
        Inv[rw * bs + j] -= f * Inv[col * bs + j];
      }
    }
  }
  return true;
}

// Extract, per cell, the diagonal block and the two vertical coupling blocks
// from the coupled scalar CSR.  D lands in `tinv` (overwritten by the factor
// kernel), L couples to the cell below (k-1), U to the cell above (k+1).
// A missing or off-rank neighbour contributes a zero block.
__global__ static void k_zline_extract(const int* __restrict__ rowptr,
                                       const Real* __restrict__ aval,
                                       const int* __restrict__ nbr7,
                                       const int* __restrict__ bcol7,
                                       PrecReal* __restrict__ tinv,
                                       PrecReal* __restrict__ L,
                                       PrecReal* __restrict__ U,
                                       int cells, int bs) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const long long o = (long long)c * bs * bs;
  // member indices in nbr7/bcol7: 0 = self, 1 + face; kFaceZlo = 4, kFaceZhi = 5.
  const int m_self = 0, m_lo = 1 + 4, m_hi = 1 + 5;
  const int n_lo = nbr7[7 * c + m_lo], n_hi = nbr7[7 * c + m_hi];
  const int off_self = bcol7[7 * c + m_self];
  const int off_lo = bcol7[7 * c + m_lo], off_hi = bcol7[7 * c + m_hi];
  for (int a = 0; a < bs; ++a) {
    const int base = rowptr[bs * c + a];
    for (int q = 0; q < bs; ++q) {
      tinv[o + a * bs + q] = (PrecReal)aval[base + off_self + q];
      L[o + a * bs + q] = (n_lo >= 0 && n_lo < cells)
                              ? (PrecReal)aval[base + off_lo + q] : (PrecReal)0;
      U[o + a * bs + q] = (n_hi >= 0 && n_hi < cells)
                              ? (PrecReal)aval[base + off_hi + q] : (PrecReal)0;
    }
  }
}

// Factor each column's block tridiagonal.  With TU_k := Tinv_k U_k the
// Thomas recursion closes over TU alone:
//     T_0 = D_0,   T_k = D_k - L_k TU_{k-1},
// and U_k is overwritten IN PLACE by TU_k as the recursion ascends, which is
// safe because T_{k+1} needs exactly TU_k and never U_k again.  The solve
// kernel then does one block-matvec less per cell and never touches tinv on
// the backward pass.  One thread per areal column; local index of (i,j,k) is
// (k*nyl + j)*nxl + i, so a column walks with stride nxl*nyl.  A singular
// modified diagonal falls back to the assembly's plain block-diagonal inverse
// for that cell, degrading that column toward block-Jacobi instead of
// poisoning the sweep.
__global__ static void k_zline_factor(PrecReal* __restrict__ tinv,
                                      const PrecReal* __restrict__ L,
                                      PrecReal* __restrict__ U,
                                      const PrecReal* __restrict__ dinv,
                                      int nxl, int nyl, int nzl, int bs) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nxl * nyl) return;
  const int il = t % nxl, jl = t / nxl;
  const long long stride = (long long)nxl * nyl;
  PrecReal prevTU[kMaxBlockSize * kMaxBlockSize];   // TU of the cell below
  PrecReal T[kMaxBlockSize * kMaxBlockSize];
  PrecReal Inv[kMaxBlockSize * kMaxBlockSize];
  for (int kl = 0; kl < nzl; ++kl) {
    const long long c = (long long)kl * stride + (long long)jl * nxl + il;
    const long long o = c * bs * bs;
    for (int i = 0; i < bs * bs; ++i) T[i] = tinv[o + i];
    if (kl > 0) {
      // T -= L_c * TU_below
      for (int a = 0; a < bs; ++a)
        for (int q = 0; q < bs; ++q) {
          PrecReal acc = 0;
          for (int m = 0; m < bs; ++m)
            acc += L[o + a * bs + m] * prevTU[m * bs + q];
          T[a * bs + q] -= acc;
        }
    }
    if (!zl_invert(T, Inv, bs))
      for (int i = 0; i < bs * bs; ++i) Inv[i] = dinv[o + i];
    for (int i = 0; i < bs * bs; ++i) tinv[o + i] = Inv[i];
    if (kl + 1 < nzl) {
      // TU_k = Tinv_k U_k, overwriting U_k.
      PrecReal Ur[kMaxBlockSize * kMaxBlockSize];
      for (int i = 0; i < bs * bs; ++i) Ur[i] = U[o + i];
      for (int a = 0; a < bs; ++a)
        for (int q = 0; q < bs; ++q) {
          PrecReal acc = 0;
          for (int m = 0; m < bs; ++m) acc += Inv[a * bs + m] * Ur[m * bs + q];
          U[o + a * bs + q] = acc;
          prevTU[a * bs + q] = acc;
        }
    }
  }
}

// Apply: z += T^{-1} (r - A z) column by column.  Forward substitution stores
// w_k = Tinv_k (rr_k - L_k w_{k-1}) in `y`; the backward pass is then just
// x_k = w_k - TU_k x_{k+1}, carrying x_{k+1} in registers.
__global__ static void k_zline_solve(const Real* __restrict__ r,
                                     const Real* __restrict__ Az,
                                     const PrecReal* __restrict__ tinv,
                                     const PrecReal* __restrict__ L,
                                     const PrecReal* __restrict__ TU,
                                     PrecReal* __restrict__ y,
                                     Real* __restrict__ z,
                                     int nxl, int nyl, int nzl, int bs) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nxl * nyl) return;
  const int il = t % nxl, jl = t / nxl;
  const long long stride = (long long)nxl * nyl;
  PrecReal w[kMaxBlockSize];         // w_{k-1}
  // forward
  for (int kl = 0; kl < nzl; ++kl) {
    const long long c = (long long)kl * stride + (long long)jl * nxl + il;
    const long long o = c * bs * bs, b0 = c * bs;
    PrecReal rr[kMaxBlockSize];
    for (int a = 0; a < bs; ++a)
      rr[a] = (PrecReal)(r[b0 + a] - Az[b0 + a]);
    if (kl > 0) {
      for (int a = 0; a < bs; ++a) {
        PrecReal acc = 0;
        for (int q = 0; q < bs; ++q) acc += L[o + a * bs + q] * w[q];
        rr[a] -= acc;
      }
    }
    PrecReal wk[kMaxBlockSize];
    for (int a = 0; a < bs; ++a) {
      PrecReal acc = 0;
      for (int q = 0; q < bs; ++q) acc += tinv[o + a * bs + q] * rr[q];
      wk[a] = acc;
    }
    for (int a = 0; a < bs; ++a) { y[b0 + a] = wk[a]; w[a] = wk[a]; }
  }
  // backward
  PrecReal xn[kMaxBlockSize];        // x_{k+1}
  for (int kl = nzl - 1; kl >= 0; --kl) {
    const long long c = (long long)kl * stride + (long long)jl * nxl + il;
    const long long o = c * bs * bs, b0 = c * bs;
    PrecReal x[kMaxBlockSize];
    for (int a = 0; a < bs; ++a) x[a] = y[b0 + a];
    if (kl + 1 < nzl) {
      for (int a = 0; a < bs; ++a) {
        PrecReal acc = 0;
        for (int q = 0; q < bs; ++q) acc += TU[o + a * bs + q] * xn[q];
        x[a] -= acc;
      }
    }
    for (int a = 0; a < bs; ++a) { z[b0 + a] += (Real)x[a]; xn[a] = x[a]; }
  }
}

// df64 emulated FP64 SpMV (see reference for the double-single algebra).
__device__ __forceinline__ float2 df_qsum(float a, float b) {
  float s = a + b; float e = b - (s - a); return make_float2(s, e);
}
__device__ __forceinline__ float2 df_sum(float a, float b) {
  float s = a + b, bb = s - a; float e = (a - (s - bb)) + (b - bb);
  return make_float2(s, e);
}
__device__ __forceinline__ float2 df_prod(float a, float b) {
  float p = a * b; float e = __fmaf_rn(a, b, -p); return make_float2(p, e);
}
__device__ __forceinline__ float2 df_add(float2 a, float2 b) {
  float2 s = df_sum(a.x, b.x), t = df_sum(a.y, b.y);
  s.y += t.x; s = df_qsum(s.x, s.y); s.y += t.y; s = df_qsum(s.x, s.y); return s;
}
__device__ __forceinline__ float2 df_mul(float2 a, float2 b) {
  float2 p = df_prod(a.x, b.x); p.y += a.x * b.y + a.y * b.x;
  p = df_qsum(p.x, p.y); return p;
}
__global__ static void k_spmv_df64(const int* rp, const int* ci, const float* ahi,
                                   const float* alo, const Real* x, Real* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float2 acc = make_float2(0.f, 0.f);
  for (int p = rp[i]; p < rp[i + 1]; ++p) {
    int c = ci[p];
    float2 a = make_float2(ahi[p], alo[p]);
    double xv = x[c]; float xh = (float)xv;
    float2 xf = make_float2(xh, (float)(xv - (double)xh));
    acc = df_add(acc, df_mul(a, xf));
  }
  y[i] = (double)acc.x + (double)acc.y;
}
// Scalar CSR apply with FP32-stored values (EMULATE 2), FP64 accumulation.
__global__ static void k_spmv_f32(const int* rp, const int* ci, const float* ahi,
                                  const Real* x, Real* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double acc = 0.0;
  for (int p = rp[i]; p < rp[i + 1]; ++p) acc = fma((double)ahi[p], x[ci[p]], acc);
  y[i] = acc;
}

__global__ static void k_well_schur_reduce(const WellSchurEntry* e, int ne,
                                            const Real* x, int bs, int modes,
                                            Real* dot) {
  // Entries are uploaded in well order. One block owns one group, locates its
  // contiguous range, and reduces it in a fixed tree. This removes the
  // run-to-run roundoff variation from atomicAdd without serialising a well
  // onto one CUDA thread.
  const int group = blockIdx.x;
  int lo = 0, hi = ne;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (e[mid].group < group) lo = mid + 1; else hi = mid;
  }
  const int first = lo;
  hi = ne;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (e[mid].group <= group) lo = mid + 1; else hi = mid;
  }
  const int last = lo;
  Real s[3] = {0, 0, 0};
  for (int i = first + threadIdx.x; i < last; i += blockDim.x) {
    const WellSchurEntry a = e[i];
    for (int mode = 0; mode < modes; ++mode)
      for (int v = 0; v < bs && v < 3; ++v)
        s[mode] += a.dg_du[3 * mode + v] *
                   x[(size_t)bs * a.cell + v];
  }
  __shared__ Real lane[3][128];
  for (int mode = 0; mode < modes; ++mode)
    lane[mode][threadIdx.x] = s[mode];
  __syncthreads();
  for (int stride = 64; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      for (int mode = 0; mode < modes; ++mode)
        lane[mode][threadIdx.x] += lane[mode][threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int mode = 0; mode < modes; ++mode)
      dot[modes * group + mode] = lane[mode][0];
}

__global__ static void k_well_schur_apply(const WellSchurEntry* e, int ne,
                                           const Real* dot, int bs, int modes,
                                           Real* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ne) return;
  const WellSchurEntry a = e[i];
  const Real* wg = dot + modes * a.group;
  const Real g[3] = {wg[0], modes > 1 ? wg[1] : (Real)0,
                     modes > 2 ? wg[2] : (Real)0};
  const int na = bs < 3 ? bs : 3;
  for (int eq = 0; eq < na; ++eq)
    atomicAdd(y + (size_t)bs * a.cell + eq,
              a.dr_dg[3 * eq] * g[0] +
              a.dr_dg[3 * eq + 1] * g[1] +
              a.dr_dg[3 * eq + 2] * g[2]);
}

// One column U(:,group,mode) of the eliminated-well low-rank block.  Several
// completions may occupy the same reservoir cell, hence the atomic add.
__global__ static void k_well_schur_column(const WellSchurEntry* e, int ne,
                                            int group, int mode, int bs,
                                            Real* u) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ne || e[i].group != group) return;
  const WellSchurEntry a = e[i];
  const int na = bs < 3 ? bs : 3;
  for (int eq = 0; eq < na; ++eq)
    atomicAdd(u + (size_t)bs * a.cell + eq, a.dr_dg[3 * eq + mode]);
}

// Woodbury correction for the block-Jacobi second CPR stage.  The coupled
// operator is D + U V^T, where D is the reservoir cell-block diagonal and
// U,V are the per-well eliminated modes already used by op_apply().  Applying
// only D^-1 makes the preconditioner blind to the strongest nonlocal well
// coupling.  Per well, form
//
//   h = V^T D^-1 r,   K = I + V^T D^-1 U,
//
// then subtract D^-1 U K^-1 h from the ordinary block-Jacobi correction.
// Three modes need only a 3x3 dense solve; mode one is its scalar special case.
__global__ static void k_well_prec_reduce(const WellSchurEntry* e, int ne,
                                           const Real* r,
                                           const Real* Az,
                                           const PrecReal* dinv, int bs,
                                           int modes, Real* small) {
  const int group = blockIdx.x;
  int lo = 0, hi = ne;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (e[mid].group < group) lo = mid + 1; else hi = mid;
  }
  const int first = lo;
  hi = ne;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (e[mid].group <= group) lo = mid + 1; else hi = mid;
  }
  const int last = lo;
  Real s[12] = {};
  const int nv = bs < 3 ? bs : 3;
  for (int i = first + threadIdx.x; i < last; i += blockDim.x) {
    const WellSchurEntry a = e[i];
    const PrecReal* Di = dinv + (long long)a.cell * bs * bs;
    Real DU[3][3] = {};
    for (int v = 0; v < nv; ++v)
      for (int n = 0; n < modes; ++n)
        for (int q = 0; q < nv; ++q)
          DU[v][n] += (Real)Di[v * bs + q] * a.dr_dg[3 * q + n];
    for (int m = 0; m < modes; ++m) {
      Real h = 0;
      for (int v = 0; v < nv; ++v) {
        Real yv = 0;
        for (int q = 0; q < nv; ++q)
          yv += (Real)Di[v * bs + q] *
                (r[(size_t)bs * a.cell + q] -
                 Az[(size_t)bs * a.cell + q]);
        h += a.dg_du[3 * m + v] * yv;
      }
      s[m] += h;
      for (int n = 0; n < modes; ++n)
        for (int v = 0; v < nv; ++v)
          s[3 + 3 * m + n] += a.dg_du[3 * m + v] * DU[v][n];
    }
  }
  __shared__ Real lane[12][128];
  for (int q = 0; q < 12; ++q) lane[q][threadIdx.x] = s[q];
  __syncthreads();
  for (int stride = 64; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      for (int q = 0; q < 12; ++q)
        lane[q][threadIdx.x] += lane[q][threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    Real* out = small + 12 * group;
    for (int q = 0; q < 12; ++q) out[q] = lane[q][0];
  }
}

__global__ static void k_well_prec_solve(Real* small, int groups, int modes) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= groups) return;
  Real* a = small + 12 * g;
  Real M[3][4] = {};
  for (int i = 0; i < modes; ++i) {
    for (int j = 0; j < modes; ++j)
      M[i][j] = a[3 + 3 * i + j] + (i == j ? (Real)1 : (Real)0);
    M[i][modes] = a[i];
  }
  for (int col = 0; col < modes; ++col) {
    int piv = col;
    Real best = fabs(M[col][col]);
    for (int row = col + 1; row < modes; ++row) {
      const Real v = fabs(M[row][col]);
      if (v > best) { best = v; piv = row; }
    }
    if (!(best > (Real)1e-20) || !isfinite(best)) {
      for (int i = 0; i < modes; ++i) a[i] = 0;
      return;
    }
    if (piv != col)
      for (int j = col; j <= modes; ++j) {
        const Real t = M[col][j]; M[col][j] = M[piv][j]; M[piv][j] = t;
      }
    const Real inv = (Real)1 / M[col][col];
    for (int j = col; j <= modes; ++j) M[col][j] *= inv;
    for (int row = 0; row < modes; ++row) {
      if (row == col) continue;
      const Real f = M[row][col];
      for (int j = col; j <= modes; ++j) M[row][j] -= f * M[col][j];
    }
  }
  for (int i = 0; i < modes; ++i) a[i] = M[i][modes];
}

__global__ static void k_well_prec_apply(const WellSchurEntry* e, int ne,
                                          const PrecReal* dinv, int bs,
                                          int modes, const Real* small,
                                          Real* z) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ne) return;
  const WellSchurEntry a = e[i];
  const Real* coeff = small + 12 * a.group;
  const PrecReal* Di = dinv + (long long)a.cell * bs * bs;
  const int nv = bs < 3 ? bs : 3;
  for (int v = 0; v < nv; ++v) {
    Real corr = 0;
    for (int n = 0; n < modes; ++n) {
      Real du = 0;
      for (int q = 0; q < nv; ++q)
        du += (Real)Di[v * bs + q] * a.dr_dg[3 * q + n];
      corr += du * coeff[n];
    }
    atomicAdd(z + (size_t)bs * a.cell + v, -corr);
  }
}

// Pressure-space factors for CPRW. With pressure prolongation P selecting the
// first primary and quasi-IMPES restriction W, an eliminated well contributes
// (W U) (V^T P) to the pressure operator.
__global__ static void k_cprw_column(const WellSchurEntry* e, int ne,
                                     const PrecReal* weight, int bs,
                                     int group, int mode, Real* u) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ne || e[i].group != group) return;
  const WellSchurEntry a = e[i];
  Real wu = 0;
  for (int eq = 0; eq < bs && eq < 3; ++eq)
    wu += (Real)weight[(size_t)bs * a.cell + eq]
          * a.dr_dg[3 * eq + mode];
  atomicAdd(u + a.cell, wu);
}

__global__ static void k_cprw_narrow(const Real* in, PrecReal* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (PrecReal)in[i];
}

__global__ static void k_cprw_reduce(const WellSchurEntry* e, int ne,
                                     const PrecReal* q, int modes, Real* dot) {
  const int group = blockIdx.x;
  int lo = 0, hi = ne;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (e[mid].group < group) lo = mid + 1; else hi = mid;
  }
  const int first = lo;
  hi = ne;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (e[mid].group <= group) lo = mid + 1; else hi = mid;
  }
  const int last = lo;
  Real s[3] = {0, 0, 0};
  for (int i = first + threadIdx.x; i < last; i += blockDim.x) {
    const WellSchurEntry a = e[i];
    const Real qi = (Real)q[a.cell];
    for (int m = 0; m < modes; ++m)
      s[m] += a.dg_du[3 * m] * qi;
  }
  __shared__ Real lane[3][128];
  for (int m = 0; m < modes; ++m) lane[m][threadIdx.x] = s[m];
  __syncthreads();
  for (int stride = 64; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      for (int m = 0; m < modes; ++m)
        lane[m][threadIdx.x] += lane[m][threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int m = 0; m < modes; ++m)
      dot[modes * group + m] = lane[m][0];
}

__global__ static void k_cprw_coeff(const Real* dot, const int* active,
                                    const Real* kinv, int na, Real* coeff) {
  if (blockIdx.x || threadIdx.x) return;
  for (int i = 0; i < na; ++i) {
    Real v = 0;
    for (int j = 0; j < na; ++j) v += kinv[(size_t)i * na + j] * dot[active[j]];
    coeff[i] = v;
  }
}

__global__ static void k_cprw_apply(PrecReal* x, const PrecReal* q,
                                    const Real* coeff, int cells, int na) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  Real corr = 0;
  for (int j = 0; j < na; ++j)
    corr += (Real)q[(size_t)j * cells + c] * coeff[j];
  x[c] = (PrecReal)((Real)x[c] - corr);
}

// ===========================================================================
// CprSolver
// ===========================================================================
CprSolver::CprSolver(const Grid& grid, int block_size, const SolverConfig& cfg)
    : grid_(grid), bs_(block_size), cfg_(cfg) {
  NVRS_CHECK_CUSPARSE(cusparseCreate(&sp_));
  NVRS_CHECK_CUBLAS(cublasCreate(&blas_));
  NVRS_CHECK_CUBLAS(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST));
  amg_ = new hypre_amg_impl();
  mgr_ = new hypre_mgr_impl();
  // amgx_ is allocated by setup_amgx and released by destroy_amgx, both of
  // which live in amgx_backend.cu.  amgx_impl is only a complete type there,
  // so new/delete cannot appear in this file.
}

CprSolver::~CprSolver() {
  destroy_amgx();   // frees amgx_ as well; see the note in the constructor
  destroy_ginkgo(); // likewise ginkgo_, which is complete only in its own file
  if (amg_) {
    if (amg_->built) {
      HYPRE_BoomerAMGDestroy(amg_->solver);
      HYPRE_IJVectorDestroy(amg_->x);
      HYPRE_IJVectorDestroy(amg_->b);
      HYPRE_IJMatrixDestroy(amg_->A);
      if (amg_->d_idx)   cudaFree(amg_->d_idx);
      if (amg_->d_ncols) cudaFree(amg_->d_ncols);
      if (amg_->d_cols)  cudaFree(amg_->d_cols);
      if (amg_->d_zero)  cudaFree(amg_->d_zero);
    }
    delete amg_;
  }
  if (matA_) cusparseDestroySpMat(matA_);
  if (vx_) cusparseDestroyDnVec(vx_);
  if (vy_) cusparseDestroyDnVec(vy_);
  if (spmv_buf_) cudaFree(spmv_buf_);
  if (redev_) cudaFree(redev_);
  cudaFree(r_); cudaFree(Ax_); cudaFree(tmp_); cudaFree(V_); cudaFree(Z_);
  cudaFree(hdev_); cudaFree(yv_); cudaFree(rp_); cudaFree(dp_);   // sdev_, ndev_ live in hdev_
  cudaFree(bcol_); cudaFree(bcol_bad_);
  graphs_clear();
  if (gs_) cudaStreamDestroy(gs_);
  cudaFree(cublas_ws_);
  cudaFree(p_); cudaFree(phat_); cudaFree(s_); cudaFree(shat_); cudaFree(t_);
  cudaFree(v_); cudaFree(rhat_);
  cudaFree(well_schur_dot_);
  cudaFree(well_lr_rhs_); cudaFree(well_lr_z_);
  cudaFree(cprw_u_); cudaFree(cprw_q_); cudaFree(cprw_kinv_);
  cudaFree(cprw_coeff_); cudaFree(cprw_active_);
  cudaFree(zl_tinv_); cudaFree(zl_l_); cudaFree(zl_u_); cudaFree(zl_y_);
  if (sp_) cusparseDestroy(sp_);
  if (blas_) cublasDestroy(blas_);
}

void CprSolver::ensure_scratch(const DeviceLinearSystem& sys) {
  const int n = sys.n, next = sys.n_ext;
  if (n <= n_alloc_ && next <= next_) return;
  auto realloc = [&](Real** p, size_t count) {
    if (*p) cudaFree(*p);
    NVRS_CHECK_CUDA(cudaMalloc(p, sizeof(Real) * count));
    NVRS_CHECK_CUDA(cudaMemset(*p, 0, sizeof(Real) * count));
  };
  // Vectors that are ever fed to the SpMV need the ghost tail (next), the rest
  // only need the owned length.  V_ and Z_ are column-major with stride next.
  realloc(&r_, n); realloc(&Ax_, n);
  realloc(&tmp_, next);                       // reused as an SpMV operand
  realloc(&V_, (size_t)(cfg_.restart + 1) * next);
  realloc(&Z_, (size_t)cfg_.restart * next);
  realloc(&yv_, cfg_.restart);
  // The two Gram-Schmidt projection columns and |w|^2 share one allocation, so
  // fgmres can bring all three to the host in a single copy per iteration.
  realloc(&hdev_, 2 * (size_t)(cfg_.restart + 1) + 1);
  sdev_ = hdev_ + (cfg_.restart + 1);
  ndev_ = hdev_ + 2 * (cfg_.restart + 1);
  realloc(&p_, next); realloc(&phat_, next);
  realloc(&s_, next);  realloc(&shat_, next);
  realloc(&t_, n); realloc(&v_, n); realloc(&rhat_, n);
  if (rp_) cudaFree(rp_);
  if (dp_) cudaFree(dp_);
  NVRS_CHECK_CUDA(cudaMalloc(&rp_, sizeof(PrecReal) * (sys.cells + 1)));
  NVRS_CHECK_CUDA(cudaMalloc(&dp_, sizeof(PrecReal) * (sys.cells + 1)));
  n_alloc_ = n;
  next_ = next;
}

// ---------------------------------------------------------------------------
// MPI-reduced inner products.
// ---------------------------------------------------------------------------
// Sum a device-resident vector across ranks, in place.
//
// The Arnoldi projections are produced on the device by cublasDgemv and
// consumed on the device by the next cublasDgemv, so staging them through the
// host costs two blocking copies per reduction and four per Arnoldi iteration.
// With a CUDA-aware MPI the reduction can read and write the device buffer
// directly; only a stream synchronisation is needed, to guarantee the gemv that
// produced the values has retired before MPI reads them.
void CprSolver::allreduce_dev(Real* dev, int count, std::vector<Real>& host) {
  if (grid_.nranks <= 1) return;
  // Stream-ordered: no synchronisation, and capturable in a CUDA graph.
  if (nccl_) { nccl_allreduce_sum(dev, count, st_); return; }
  if (halo_cuda_aware()) {
    // Distinct send and receive buffers, not MPI_IN_PLACE.  In-place collectives
    // on device memory are the least reliably implemented part of CUDA-aware
    // MPI, and this path is not worth a hang.
    if (!redev_ || redev_count_ < count) {
      if (redev_) cudaFree(redev_);
      NVRS_CHECK_CUDA(cudaMalloc(&redev_, sizeof(Real) * count));
      redev_count_ = count;
    }
    // The gemv that produced these values runs on the default stream, which is
    // what cuBLAS uses here; MPI must not read the buffer before it retires.
    NVRS_CHECK_CUDA(cudaStreamSynchronize(0));
    MPI_Allreduce(dev, redev_, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    NVRS_CHECK_CUDA(cudaMemcpy(dev, redev_, sizeof(Real) * count,
                               cudaMemcpyDeviceToDevice));
    return;
  }
  host.resize(count);
  NVRS_CHECK_CUDA(cudaMemcpy(host.data(), dev, sizeof(Real) * count,
                             cudaMemcpyDeviceToHost));
  MPI_Allreduce(MPI_IN_PLACE, host.data(), count, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  NVRS_CHECK_CUDA(cudaMemcpy(dev, host.data(), sizeof(Real) * count,
                             cudaMemcpyHostToDevice));
}

Real CprSolver::gdot(int n, const Real* a, const Real* b) {
  Real local = 0;
  NVRS_CHECK_CUBLAS(cublasDdot(blas_, n, a, 1, b, 1, &local));
  Real global = local;
  if (grid_.nranks > 1)
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global;
}
Real CprSolver::gnrm2(int n, const Real* a) { return std::sqrt(gdot(n, a, a)); }

// ---------------------------------------------------------------------------
// Coupled operator apply  y = A x   (cuSPARSE FP64 or df64 emulation).
//
// The operator is rectangular in the local index space: `n` owned rows by
// `n_ext` columns, the extra columns being the ghost cells across the K-slab
// seam.  Refreshing the ghost tail of x here is what makes the distributed SpMV
// exact — without it every rank would see a no-flow boundary at the seam.
// ---------------------------------------------------------------------------
void CprSolver::apply_well_schur(const DeviceLinearSystem& sys,
                                  const Real* x, Real* y) {
  if (!sys.well_schur || sys.well_schur_count <= 0 ||
      sys.well_schur_groups <= 0) return;
  const int modes = sys.well_schur_modes;
  const int nmode = modes * sys.well_schur_groups;
  if (well_schur_alloc_ < nmode) {
    cudaFree(well_schur_dot_);
    NVRS_CHECK_CUDA(cudaMalloc(&well_schur_dot_,
                               sizeof(Real) * nmode));
    well_schur_alloc_ = nmode;
  }
  const int nt = 128, nb = (sys.well_schur_count + nt - 1) / nt;
  const auto* e = static_cast<const WellSchurEntry*>(sys.well_schur);
  k_well_schur_reduce<<<sys.well_schur_groups, nt, 0, st_>>>(
      e, sys.well_schur_count, x, sys.bs, modes, well_schur_dot_);
  NVRS_CHECK_CUDA(cudaGetLastError());
  allreduce_dev(well_schur_dot_, nmode, redbuf_);
  k_well_schur_apply<<<nb, nt, 0, st_>>>(e, sys.well_schur_count, well_schur_dot_,
                                         sys.bs, modes, y);
  NVRS_CHECK_CUDA(cudaGetLastError());
}

// Build (once per sparsity pattern) and check the block column index.  The
// check costs one synchronisation, and only when the pattern changes.
bool CprSolver::block_ready(const DeviceLinearSystem& sys) {
  if (!cfg_.block_spmv || !bspmv::supported(sys.bs) || !sys.prowptr ||
      sys.cells <= 0)
    return false;
  if (bcol_state_ != 0 && bcol_rowptr_ == sys.rowptr &&
      bcol_colind_ == sys.colind && bcol_nnz_ == sys.nnz &&
      bcol_cells_ == sys.cells && bcol_bs_ == sys.bs)
    return bcol_state_ > 0;
  // A new pattern.  Never while a graph is being recorded: the check reads its
  // verdict back to the host.  graphs_usable() refuses to capture before this
  // has run, so this cannot happen, but say so rather than corrupt a capture.
  if (capturing_) return false;
  int pnnz = 0;
  NVRS_CHECK_CUDA(cudaMemcpy(&pnnz, sys.prowptr + sys.cells, sizeof(int),
                             cudaMemcpyDeviceToHost));
  if (pnnz * sys.bs * sys.bs != sys.nnz) {
    bcol_state_ = -1;
  } else {
    if (bcol_alloc_ < pnnz) {
      cudaFree(bcol_);
      NVRS_CHECK_CUDA(cudaMalloc(&bcol_, sizeof(int) * (size_t)pnnz));
      bcol_alloc_ = pnnz;
    }
    if (!bcol_bad_) NVRS_CHECK_CUDA(cudaMalloc(&bcol_bad_, sizeof(int)));
    bspmv::k_bcol_build<<<grid1d(sys.cells), 256>>>(
        sys.prowptr, sys.rowptr, sys.colind, bcol_, sys.cells, sys.bs);
    NVRS_CHECK_CUDA(cudaMemset(bcol_bad_, 0, sizeof(int)));
    bspmv::k_bspmv_check<<<grid1d(sys.cells), 256>>>(
        sys.prowptr, sys.rowptr, sys.colind, bcol_, sys.cells, sys.bs,
        bcol_bad_);
    NVRS_CHECK_CUDA(cudaGetLastError());
    int bad = 0;
    NVRS_CHECK_CUDA(cudaMemcpy(&bad, bcol_bad_, sizeof(int),
                               cudaMemcpyDeviceToHost));
    bcol_state_ = bad == 0 ? 1 : -1;
  }
  if (bcol_state_ < 0 && grid_.rank == 0) {
    static bool said = false;
    if (!said) {
      said = true;
      std::printf("  [operator] BLOCKSPMV: this sparsity pattern does not have "
                  "the %dx%d block layout; using the scalar CSR operator.\n",
                  sys.bs, sys.bs);
    }
  }
  bcol_rowptr_ = sys.rowptr; bcol_colind_ = sys.colind;
  bcol_nnz_ = sys.nnz; bcol_cells_ = sys.cells; bcol_bs_ = sys.bs;
  return bcol_state_ > 0;
}

void CprSolver::op_apply(const DeviceLinearSystem& sys, Real* x, Real* y) {
  if (!capturing_) ev_spmv_.tic();
  // The field path has its own halo: the ghosts of an unstructured partition
  // are not a K-slab seam and cannot be described by one.  Same branch the
  // assembly kernels take.
  if (nccl_) {
    // Stream-ordered ghosts: nothing here waits for the device.
    if (conn_halo_) {
      static_cast<HaloConn*>(conn_halo_)->exchange_nccl(x, sys.bs, st_);
    } else {
      if (!capturing_) halo_init(grid_, sys.bs);   // buffers exist before any capture
      halo_exchange_nccl(x, sys.bs, grid_, st_);
    }
  } else if (conn_halo_) {
    static_cast<HaloConn*>(conn_halo_)->exchange(x, sys.bs);
  } else if (grid_.nranks > 1) {
    halo_exchange(x, sys.bs, grid_);
  }

  // The df64 path needs the (hi, lo) FP32 slices of the operator.  A system
  // that does not maintain them — anything not going through
  // split_operator_df64 — leaves these null, and the emulated SpMV would then
  // dereference null on every thread.  That is an illegal access reported far
  // from its cause, so fall back to cuSPARSE instead of trusting the flag.
  const int mode = master_apply_ ? 0 : operator_mode();
  const bool fp32op = mode == 2 && sys.aval_hi;
  const bool df64 = mode == 1 && sys.aval_hi && sys.aval_lo;
  // Block storage: one column index per bs x bs block (include/block_spmv.cuh).
  if (block_ready(sys) && (df64 || fp32op || sys.aval64)) {
    if (fp32op)
      bspmv::launch_f32(sys.prowptr, bcol_, sys.aval_hi, x, y, sys.n, sys.bs,
                        st_);
    else if (df64)
      bspmv::launch_df64(sys.prowptr, bcol_, sys.aval_hi, sys.aval_lo, x, y,
                         sys.n, sys.bs, st_);
    else
      bspmv::launch_f64(sys.prowptr, bcol_, sys.aval64, x, y, sys.n, sys.bs,
                        st_);
    NVRS_CHECK_CUDA(cudaGetLastError());
    if (!capturing_) {                        // replays are counted by graph_phase
      ++timings_.spmv_block;
      if (fp32op) ++timings_.spmv_fp32;
    }
    apply_well_schur(sys, x, y);
    if (!capturing_) ev_spmv_.toc();
    return;
  }
  if (!capturing_) ++timings_.spmv_scalar;
  if (fp32op) {
    int threads = 256, nb = (sys.n + threads - 1) / threads;
    k_spmv_f32<<<nb, threads, 0, st_>>>(sys.rowptr, sys.colind, sys.aval_hi, x,
                                        y, sys.n);
    NVRS_CHECK_CUDA(cudaGetLastError());
    if (!capturing_) ++timings_.spmv_fp32;
    apply_well_schur(sys, x, y);
    if (!capturing_) ev_spmv_.toc();
    return;
  }
  if (df64) {
    int threads = 256, nb = (sys.n + threads - 1) / threads;
    k_spmv_df64<<<nb, threads, 0, st_>>>(sys.rowptr, sys.colind, sys.aval_hi,
                                         sys.aval_lo, x, y, sys.n);
    NVRS_CHECK_CUDA(cudaGetLastError());
    apply_well_schur(sys, x, y);
    if (!capturing_) ev_spmv_.toc();
    return;
  }
  const Real one = 1.0, zero = 0.0;
  // Rebuild the descriptor if this system has a different shape from the one
  // it was created for.
  if (matA_ && (mat_rows_ != sys.n || mat_cols_ != sys.n_ext ||
                mat_nnz_ != sys.nnz)) {
    cusparseDestroySpMat(matA_); matA_ = nullptr;
    if (vx_) { cusparseDestroyDnVec(vx_); vx_ = nullptr; }
    if (vy_) { cusparseDestroyDnVec(vy_); vy_ = nullptr; }
    if (spmv_buf_) { cudaFree(spmv_buf_); spmv_buf_ = nullptr; spmv_bufsz_ = 0; }
  }
  if (!matA_) {
    NVRS_CHECK_CUSPARSE(cusparseCreateCsr(
        &matA_, sys.n, sys.n_ext, sys.nnz, sys.rowptr, sys.colind, sys.aval64,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));
    NVRS_CHECK_CUSPARSE(cusparseCreateDnVec(&vx_, sys.n_ext, (void*)x, CUDA_R_64F));
    NVRS_CHECK_CUSPARSE(cusparseCreateDnVec(&vy_, sys.n, y, CUDA_R_64F));
    mat_rows_ = sys.n; mat_cols_ = sys.n_ext; mat_nnz_ = sys.nnz;
  } else {
    NVRS_CHECK_CUSPARSE(cusparseCsrSetPointers(matA_, sys.rowptr, sys.colind,
                                               sys.aval64));
    NVRS_CHECK_CUSPARSE(cusparseDnVecSetValues(vx_, (void*)x));
    NVRS_CHECK_CUSPARSE(cusparseDnVecSetValues(vy_, y));
  }
  if (!spmv_buf_) {
    NVRS_CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        sp_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA_, vx_, &zero, vy_,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_bufsz_));
    if (spmv_bufsz_) NVRS_CHECK_CUDA(cudaMalloc(&spmv_buf_, spmv_bufsz_));
  }
  NVRS_CHECK_CUSPARSE(cusparseSetStream(sp_, st_));
  NVRS_CHECK_CUSPARSE(cusparseSpMV(sp_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                   matA_, vx_, &zero, vy_, CUDA_R_64F,
                                   CUSPARSE_SPMV_ALG_DEFAULT, spmv_buf_));
  NVRS_CHECK_CUSPARSE(cusparseSetStream(sp_, 0));
  apply_well_schur(sys, x, y);
  if (!capturing_) ev_spmv_.toc();
}

// The residual that decides convergence is formed with the FP64 master values.
// Under EMULATE 0 and 1 that is exactly op_apply, so those runs are unchanged;
// under EMULATE 2 the Krylov matvec reads FP32 values and this one does not.
void CprSolver::op_apply_master(const DeviceLinearSystem& sys, Real* x, Real* y) {
  if (operator_mode() != 2 || !sys.aval64) { op_apply(sys, x, y); return; }
  master_apply_ = true;
  op_apply(sys, x, y);
  master_apply_ = false;
}

// ---------------------------------------------------------------------------
// Shared BoomerAMG configuration (see CprSolver::configure_amg).
// ---------------------------------------------------------------------------
void CprSolver::configure_amg(void* solver_v) const {
  HYPRE_Solver s = (HYPRE_Solver)solver_v;
  HYPRE_BoomerAMGSetPrintLevel(s, cfg_.amg_print_level);
  HYPRE_BoomerAMGSetMaxIter(s, 1);          // one V-cycle as a preconditioner
  HYPRE_BoomerAMGSetTol(s, 0.0);
  HYPRE_BoomerAMGSetCoarsenType(s, cfg_.coarsen_type);   // 8 = PMIS
  HYPRE_BoomerAMGSetInterpType(s, cfg_.interp_type);
  HYPRE_BoomerAMGSetStrongThreshold(s, cfg_.strong_threshold);
  HYPRE_BoomerAMGSetPMaxElmts(s, cfg_.pmax);             // P stays sparse
  HYPRE_BoomerAMGSetTruncFactor(s, cfg_.trunc_factor);
  HYPRE_BoomerAMGSetRelaxType(s, cfg_.relax_type);
  HYPRE_BoomerAMGSetNumSweeps(s, cfg_.num_sweeps);
  // Cycle position 3 is the coarsest grid.  Set after SetRelaxType, which
  // writes all three positions.
  if (cfg_.amg_coarse_relax >= 0)
    HYPRE_BoomerAMGSetCycleRelaxType(s, cfg_.amg_coarse_relax, 3);
  if (cfg_.keep_transpose) HYPRE_BoomerAMGSetKeepTranspose(s, 1);

  if (cfg_.agg_num_levels > 0) {
    HYPRE_BoomerAMGSetAggNumLevels(s, cfg_.agg_num_levels);
    // Interpolation on the aggressive levels is denser than elsewhere unless
    // it is truncated separately.
    if (cfg_.agg_trunc_factor > 0.0)
      HYPRE_BoomerAMGSetAggTruncFactor(s, cfg_.agg_trunc_factor);
    if (cfg_.agg_pmax > 0) HYPRE_BoomerAMGSetAggPMaxElmts(s, cfg_.agg_pmax);
  }

  // The 512 ceiling exists because of the coarsest-grid solver, not because of
  // the coarse grid itself.  With Gaussian elimination there, HYPRE gathers the
  // coarse operator onto every rank and factorises it densely on each, so the
  // cost grows as the cube of this number and an earlier attempt at 2000 was
  // catastrophic.  Any device-resident smoother makes the coarse grid ordinary
  // work again and the ceiling stops applying.
  //
  // This matters because the measurements say the V-cycle is dominated by
  // per-level overhead, not by arithmetic: at one GPU a cycle costs 2.07 ms
  // while level 0, the only level with real work in it, accounts for a few
  // hundred microseconds.  Stopping the coarsening earlier removes whole levels,
  // and removing levels is the only thing that reduces that overhead.
  int max_coarse = cfg_.amg_max_coarse > 0 ? cfg_.amg_max_coarse : 64;
  const bool coarse_is_ge = (cfg_.amg_coarse_relax == 9);
  if (coarse_is_ge && max_coarse > 512) {
    if (grid_.rank == 0) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        std::printf("  [warn] AMGMAXCOARSE %d capped at 512: the coarsest grid "
                    "still uses Gaussian elimination, which is gathered and "
                    "factorised on every rank. Set AMGCOARSERELAX 18 to lift "
                    "this.\n",
                    max_coarse);
      }
    }
    max_coarse = 512;
  }
  HYPRE_BoomerAMGSetMaxCoarseSize(s, max_coarse);

  // Gathering the small coarse levels onto one rank is NOT available on a GPU
  // build either, for exactly the same reason as non-Galerkin below.
  //
  // SetSeqThreshold sends BoomerAMGSetup down hypre_seqAMGSetup, which is a
  // host routine that walks the row pointers directly:
  //
  //   gen_redcs_mat.c:174  A_tmp_i[i] = A_diag_i[i] - A_diag_i[i-1] + ...
  //
  // Under HYPRE_MEMORY_DEVICE those arrays are device memory, so it dies with
  // SIGSEGV on the first rank that reaches it.  Single rank never gets there,
  // which is why this only showed up at two GPUs.
  //
  // This is the knob that most directly targeted the measured cost -- coarse
  // levels holding a few hundred rows spread over every GPU, traversed on every
  // V-cycle -- so losing it matters.  What remains that is GPU-safe is
  // coarsening faster, so those levels are never created: see agg_num_levels.
  if (cfg_.amg_seq_threshold > 0 && grid_.rank == 0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::printf("  [warn] AMGSEQTHRESH %d ignored: HYPRE's sequential coarse "
                  "solve is a host-only code path and segfaults against a "
                  "device build.\n",
                  cfg_.amg_seq_threshold);
    }
  }

  // Non-Galerkin dropping is NOT available on a GPU build of HYPRE.
  //
  // hypre_NonGalerkinSparsityPattern allocates with HYPRE_MEMORY_HOST and
  // then reads RAP_diag_i[i] in a host loop.  Under HYPRE_MEMORY_DEVICE that
  // array lives on the device, so the loop dereferences device memory from
  // the host and the process dies with SIGSEGV inside BoomerAMGSetup:
  //
  //   par_nongalerkin.c:1039  for (i = 0; i < num_variables; i++)
  //     rownz[i] = 1.2*(RAP_diag_i[i+1] - RAP_diag_i[i]) + ...
  //
  // This is a limitation of the library, not a tuning choice, so the knob is
  // refused rather than silently ignored.
  if (cfg_.amg_nongalerkin_tol > 0.0 && grid_.rank == 0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::printf("  [warn] AMGNONGALERKIN %.3g ignored: HYPRE's "
                  "non-Galerkin coarsening is a host-only code path and "
                  "segfaults against a device build.\n",
                  cfg_.amg_nongalerkin_tol);
    }
  }
}

// ---------------------------------------------------------------------------
// Force the next setup() to rebuild.  See the header for when to call it.
// ---------------------------------------------------------------------------
void CprSolver::invalidate_amg() {
  reuse_left_ = 0;
  iters_fresh_ = -1;
  amg_pval_ = nullptr;
}

// ---------------------------------------------------------------------------
// Build / refresh the BoomerAMG hierarchy from the pressure sub-block.
// ---------------------------------------------------------------------------
void CprSolver::setup(const DeviceLinearSystem& sys) {
  // The ZLINE stage-2 factor tracks the Jacobian, not the AMG hierarchy, so
  // it refreshes on every setup regardless of backend or hierarchy reuse.
  if (cfg_.stage2 == 1) setup_zline(sys);
  if (cfg_.stage2 == 2) setup_ginkgo_stage2(sys);
  if (cfg_.backend == SolverConfig::kGinkgo) {
    setup_ginkgo(sys);
    setup_cprw(sys);
    return;
  }
  if (cfg_.backend == SolverConfig::kAmgx) {
    setup_amgx(sys);
    setup_cprw(sys);
    return;
  }
  if (cfg_.use_mgr) {
    // Same reuse rule as CPR: the hierarchy is frozen for a few Newton
    // iterations and rebuilt when the Krylov count says it has gone stale.
    if (mgr_->built && reuse_left_ > 0 &&
        amg_pval_ == (const void*)sys.aval64 && amg_cells_ == sys.cells &&
        amg_pnnz_ == sys.nnz) {
      --reuse_left_;
      ++timings_.n_amg_reuse;
      amg_fresh_ = false;
      return;
    }
    setup_mgr(sys);
    return;
  }
  auto& A = *amg_;

  // --- reuse the hierarchy the last rebuild left behind? --------------------
  // Both the coarse-grid operators and the fine-level matrix are kept: HYPRE's
  // relaxation caches per-level data at setup time (the l1 norms, for relax
  // type 18), so refreshing the fine-level values under a hierarchy built from
  // the old ones would leave the smoother describing a matrix that is no longer
  // there.  Freezing the whole preconditioner instead is both cheaper and
  // easier to reason about -- and it skips the IJ assembly, which on N ranks is
  // itself a communication step.
  if (A.built && reuse_left_ > 0 && amg_pval_ == (const void*)sys.pval &&
      amg_cells_ == sys.cells && amg_pnnz_ == sys.pnnz) {
    --reuse_left_;
    ++timings_.n_amg_reuse;
    amg_fresh_ = false;
    setup_cprw(sys);
    return;
  }

  A.cells = sys.cells;
  A.row0 = sys.row0;
  const long long lo = sys.row0, hi = sys.row0 + sys.cells - 1;

  if (!A.built) {
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, lo, hi, lo, hi, &A.A);
    HYPRE_IJMatrixSetObjectType(A.A, HYPRE_PARCSR);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, lo, hi, &A.b);
    HYPRE_IJVectorSetObjectType(A.b, HYPRE_PARCSR);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, lo, hi, &A.x);
    HYPRE_IJVectorSetObjectType(A.x, HYPRE_PARCSR);
    HYPRE_BoomerAMGCreate(&A.solver);
    configure_amg(A.solver);
    if (grid_.rank == 0 && grid_.nranks > 1)
      std::printf("  AMG      : coarsening stops at %d rows (%d ranks)\n",
                  cfg_.amg_max_coarse > 0 ? cfg_.amg_max_coarse : 64,
                  grid_.nranks);
    // --- device-side index metadata, built once (the sparsity is static) ---
    std::vector<HYPRE_BigInt> idx(sys.cells);
    for (int i = 0; i < sys.cells; ++i) idx[i] = lo + i;
    NVRS_CHECK_CUDA(cudaMalloc(&A.d_idx, sizeof(HYPRE_BigInt) * sys.cells));
    NVRS_CHECK_CUDA(cudaMemcpy(A.d_idx, idx.data(),
                               sizeof(HYPRE_BigInt) * sys.cells,
                               cudaMemcpyHostToDevice));

    NVRS_CHECK_CUDA(cudaMalloc(&A.d_ncols, sizeof(HYPRE_Int) * sys.cells));
    k_row_counts<<<grid1d(sys.cells), 256>>>(sys.prowptr, A.d_ncols, sys.cells);
    NVRS_CHECK_CUDA(cudaGetLastError());

    NVRS_CHECK_CUDA(cudaMalloc(&A.d_cols, sizeof(HYPRE_BigInt) * sys.pnnz));
    k_widen_cols<<<grid1d(sys.pnnz), 256>>>(sys.pcolind, A.d_cols, sys.pnnz);
    NVRS_CHECK_CUDA(cudaGetLastError());

    NVRS_CHECK_CUDA(cudaMalloc(&A.d_zero, sizeof(HYPRE_Complex) * sys.cells));
    NVRS_CHECK_CUDA(cudaMemset(A.d_zero, 0, sizeof(HYPRE_Complex) * sys.cells));
    A.built = true;
  }

  // Load the whole pressure block in a single call, straight from device
  // memory.  Two things matter here:
  //
  //   * HYPRE_MEMORY_DEVICE (set in main.cu) means every array below must be a
  //     *device* pointer.  Passing host pointers — a stack `HYPRE_Int nc`, a
  //     stack `HYPRE_BigInt row`, and std::vector buffers, as this code used to
  //     — makes HYPRE dereference host addresses as device memory.  That is
  //     what produced "CUDA ERROR (code = 1, invalid argument)" inside HYPRE's
  //     memory.c, followed by a bogus thrust "invalid device ordinal".
  //   * One call for all rows instead of one call per row. The old loop issued
  //     `cells` HYPRE calls plus three device-to-host copies on *every* Newton
  //     iteration; the sparsity never changes, so only the values need to move,
  //     and they are already where HYPRE wants them.
  HYPRE_IJMatrixInitialize(A.A);
  HYPRE_IJMatrixSetValues(A.A, sys.cells, A.d_ncols, A.d_idx, A.d_cols,
                          reinterpret_cast<const HYPRE_Complex*>(sys.pval));
  HYPRE_IJMatrixAssemble(A.A);
  HYPRE_IJMatrixGetObject(A.A, (void**)&A.parA);

  HYPRE_IJVectorInitialize(A.b);
  HYPRE_IJVectorInitialize(A.x);
  HYPRE_IJVectorSetValues(A.b, sys.cells, A.d_idx, A.d_zero);
  HYPRE_IJVectorSetValues(A.x, sys.cells, A.d_idx, A.d_zero);
  HYPRE_IJVectorAssemble(A.b);
  HYPRE_IJVectorAssemble(A.x);
  HYPRE_IJVectorGetObject(A.b, (void**)&A.parb);
  HYPRE_IJVectorGetObject(A.x, (void**)&A.parx);
#ifdef NVRS_HYPRE_DIRECT_VEC
  A.b_dev = (nvrs::PrecReal*)hypre_VectorData(
      hypre_ParVectorLocalVector((hypre_ParVector*)A.parb));
  A.x_dev = (nvrs::PrecReal*)hypre_VectorData(
      hypre_ParVectorLocalVector((hypre_ParVector*)A.parx));
#endif
  if (grid_.rank == 0 && !vecpath_said_) {
    vecpath_said_ = true;
    std::printf("  Vectors  : %s\n",
                (A.b_dev && A.x_dev)
                    ? "written directly into HYPRE's storage"
                    : "via the IJ interface (_hypre_parcsr_mv.h not found)");
  }
  {
    const double t = now_synced();
    HYPRE_BoomerAMGSetup(A.solver, A.parA, A.parb, A.parx);
    timings_.amg_setup += now_synced() - t;
    ++timings_.n_amg_setup;
  }
  // AMGPRINT dumps HYPRE's level table -- number of levels, rows and non-zeros
  // per level, grid and operator complexity.  For the first setup only: a run
  // rebuilds the hierarchy of order a hundred times and would bury everything
  // else.  There is no public getter for the level count on its own, so the
  // table is how you read it.
  if (cfg_.amg_print_level > 0 && !amg_printed_) {
    amg_printed_ = true;
    HYPRE_BoomerAMGSetPrintLevel(A.solver, 0);
  }

  // Arm the reuse counter and record what this hierarchy was built from.  The
  // yardstick for the growth guard is cleared here and re-measured by the very
  // next solve(), so a rebuild triggered by a stale hierarchy is never judged
  // against the iteration count that made it stale.
  reuse_left_ = std::max(0, cfg_.amg_reuse);
  iters_fresh_ = -1;
  amg_fresh_ = true;
  amg_pval_ = (const void*)sys.pval;
  amg_cells_ = sys.cells;
  amg_pnnz_ = sys.pnnz;
  setup_cprw(sys);
}

// One stage-1 pressure apply, x = M_p^{-1} b, through the configured backend.
// AmgX and Ginkgo take our staging buffers directly.  HYPRE reads and writes
// its own vectors: straight into their storage where the headers expose it,
// through the IJ interface otherwise -- the same two paths apply_cpr uses.
void CprSolver::apply_stage1(const DeviceLinearSystem& sys, PrecReal* bp,
                             PrecReal* xp) {
  if (cfg_.backend == SolverConfig::kAmgx) { apply_amgx(sys, bp, xp); return; }
  if (cfg_.backend == SolverConfig::kGinkgo) {
    apply_ginkgo(sys, bp, xp);
    return;
  }
  auto& A = *amg_;
  const bool direct = A.b_dev != nullptr && A.x_dev != nullptr;
  if (direct) {
    NVRS_CHECK_CUDA(cudaMemcpy(A.b_dev, bp, sizeof(PrecReal) * sys.cells,
                               cudaMemcpyDeviceToDevice));
    NVRS_CHECK_CUDA(cudaMemset(A.x_dev, 0, sizeof(PrecReal) * sys.cells));
  } else {
    HYPRE_IJVectorSetValues(A.b, sys.cells, A.d_idx, (HYPRE_Complex*)bp);
    HYPRE_IJVectorSetValues(A.x, sys.cells, A.d_idx, A.d_zero);
    HYPRE_IJVectorAssemble(A.b);
    HYPRE_IJVectorAssemble(A.x);
  }
  HYPRE_BoomerAMGSolve(A.solver, A.parA, A.parb, A.parx);
  if (direct)
    NVRS_CHECK_CUDA(cudaMemcpy(xp, A.x_dev, sizeof(PrecReal) * sys.cells,
                               cudaMemcpyDeviceToDevice));
  else
    HYPRE_IJVectorGetValues(A.x, sys.cells, A.d_idx, (HYPRE_Complex*)xp);
}

// Build the Flow-style global well correction for the pressure stage.  The
// pressure restriction/prolongation turns the eliminated well/control block
// into a small Woodbury correction around the AMG pressure inverse.
void CprSolver::setup_cprw(const DeviceLinearSystem& sys) {
  cprw_na_ = cprw_nfull_ = 0;
  if (!cfg_.use_cprw || cfg_.use_mgr ||
      !sys.well_schur || sys.well_schur_count <= 0 ||
      sys.well_schur_groups <= 0 || sys.well_schur_modes <= 0)
    return;

  ensure_scratch(sys);
  const int modes = sys.well_schur_modes;
  const int nfull = modes * sys.well_schur_groups;
  if (nfull > 16) return;

  std::vector<WellSchurEntry> he(sys.well_schur_count);
  NVRS_CHECK_CUDA(cudaMemcpy(he.data(), sys.well_schur,
                             sizeof(WellSchurEntry) * he.size(),
                             cudaMemcpyDeviceToHost));
  std::vector<int> have_u(nfull, 0), have_v(nfull, 0);
  for (const auto& a : he)
    for (int m = 0; m < modes; ++m) {
      const int id = modes * a.group + m;
      for (int eq = 0; eq < sys.bs && eq < 3; ++eq)
        if (a.dr_dg[3 * eq + m] != 0) have_u[id] = 1;
      if (a.dg_du[3 * m] != 0) have_v[id] = 1;
    }
  if (grid_.nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, have_u.data(), nfull, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, have_v.data(), nfull, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
  }
  std::vector<int> active;
  for (int i = 0; i < nfull; ++i)
    if (have_u[i] && have_v[i]) active.push_back(i);
  if (active.empty()) return;
  const int na = (int)active.size();

  if (cprw_cells_alloc_ < sys.cells || cprw_modes_alloc_ < na) {
    cudaFree(cprw_u_); cudaFree(cprw_q_); cudaFree(cprw_kinv_);
    cudaFree(cprw_coeff_); cudaFree(cprw_active_);
    NVRS_CHECK_CUDA(cudaMalloc(&cprw_u_, sizeof(Real) * sys.cells));
    NVRS_CHECK_CUDA(cudaMalloc(&cprw_q_, sizeof(PrecReal) *
                               (size_t)sys.cells * na));
    NVRS_CHECK_CUDA(cudaMalloc(&cprw_kinv_, sizeof(Real) * na * na));
    NVRS_CHECK_CUDA(cudaMalloc(&cprw_coeff_, sizeof(Real) * na));
    NVRS_CHECK_CUDA(cudaMalloc(&cprw_active_, sizeof(int) * na));
    cprw_cells_alloc_ = sys.cells;
    cprw_modes_alloc_ = na;
  }
  if (well_schur_alloc_ < nfull) {
    cudaFree(well_schur_dot_);
    NVRS_CHECK_CUDA(cudaMalloc(&well_schur_dot_, sizeof(Real) * nfull));
    well_schur_alloc_ = nfull;
  }
  NVRS_CHECK_CUDA(cudaMemcpy(cprw_active_, active.data(), sizeof(int) * na,
                             cudaMemcpyHostToDevice));

  std::vector<Real> K((size_t)na * na, 0.0), dots(nfull);
  const auto* we = static_cast<const WellSchurEntry*>(sys.well_schur);
  for (int j = 0; j < na; ++j) {
    const int id = active[j], group = id / modes, mode = id % modes;
    NVRS_CHECK_CUDA(cudaMemset(cprw_u_, 0, sizeof(Real) * sys.cells));
    k_cprw_column<<<(sys.well_schur_count + 127) / 128, 128>>>(
        we, sys.well_schur_count, sys.pweight, sys.bs, group, mode, cprw_u_);
    k_cprw_narrow<<<grid1d(sys.cells), 256>>>(cprw_u_, rp_, sys.cells);
    NVRS_CHECK_CUDA(cudaMemset(dp_, 0, sizeof(PrecReal) * sys.cells));
    apply_stage1(sys, rp_, dp_);
    PrecReal* qj = cprw_q_ + (size_t)j * sys.cells;
    NVRS_CHECK_CUDA(cudaMemcpy(qj, dp_, sizeof(PrecReal) * sys.cells,
                               cudaMemcpyDeviceToDevice));
    k_cprw_reduce<<<sys.well_schur_groups, 128>>>(
        we, sys.well_schur_count, qj, modes, well_schur_dot_);
    allreduce_dev(well_schur_dot_, nfull, redbuf_);
    NVRS_CHECK_CUDA(cudaMemcpy(dots.data(), well_schur_dot_,
                               sizeof(Real) * nfull, cudaMemcpyDeviceToHost));
    for (int i = 0; i < na; ++i)
      K[(size_t)i * na + j] = (i == j ? 1.0 : 0.0) + dots[active[i]];
  }

  // Invert the tiny dense matrix with partial pivoting.
  std::vector<Real> inv((size_t)na * na, 0.0);
  for (int i = 0; i < na; ++i) inv[(size_t)i * na + i] = 1.0;
  bool good = true;
  for (int col = 0; col < na && good; ++col) {
    int piv = col;
    for (int row = col + 1; row < na; ++row)
      if (std::fabs(K[(size_t)row * na + col]) >
          std::fabs(K[(size_t)piv * na + col])) piv = row;
    const Real d = K[(size_t)piv * na + col];
    if (!std::isfinite((double)d) || std::fabs(d) < 1e-16) { good = false; break; }
    if (piv != col)
      for (int j = 0; j < na; ++j) {
        std::swap(K[(size_t)col * na + j], K[(size_t)piv * na + j]);
        std::swap(inv[(size_t)col * na + j], inv[(size_t)piv * na + j]);
      }
    const Real di = 1.0 / K[(size_t)col * na + col];
    for (int j = 0; j < na; ++j) {
      K[(size_t)col * na + j] *= di;
      inv[(size_t)col * na + j] *= di;
    }
    for (int row = 0; row < na; ++row) if (row != col) {
      const Real f = K[(size_t)row * na + col];
      for (int j = 0; j < na; ++j) {
        K[(size_t)row * na + j] -= f * K[(size_t)col * na + j];
        inv[(size_t)row * na + j] -= f * inv[(size_t)col * na + j];
      }
    }
  }
  if (!good) return;
  NVRS_CHECK_CUDA(cudaMemcpy(cprw_kinv_, inv.data(), sizeof(Real) * na * na,
                             cudaMemcpyHostToDevice));
  cprw_na_ = na;
  cprw_nfull_ = nfull;
  if (grid_.rank == 0) {
    static bool said = false;
    if (!said) {
      std::printf("  CPRW     : %d global eliminated-well pressure mode(s)\n", na);
      said = true;
    }
  }
}

void CprSolver::apply_cprw(const DeviceLinearSystem& sys,
                           PrecReal* pressure_update) {
  if (cprw_na_ <= 0) return;
  const auto* we = static_cast<const WellSchurEntry*>(sys.well_schur);
  k_cprw_reduce<<<sys.well_schur_groups, 128, 0, st_>>>(
      we, sys.well_schur_count, pressure_update, sys.well_schur_modes,
      well_schur_dot_);
  allreduce_dev(well_schur_dot_, cprw_nfull_, redbuf_);
  k_cprw_coeff<<<1, 1, 0, st_>>>(well_schur_dot_, cprw_active_, cprw_kinv_,
                                 cprw_na_, cprw_coeff_);
  k_cprw_apply<<<grid1d(sys.cells), 256, 0, st_>>>(pressure_update, cprw_q_,
                                                   cprw_coeff_, sys.cells,
                                                   cprw_na_);
  NVRS_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CPR preconditioner apply  z = M^{-1} r.
// ---------------------------------------------------------------------------
void CprSolver::apply_cpr(const DeviceLinearSystem& sys, const Real* r, Real* z,
                          Real* tmpAz) {
  if (cfg_.use_mgr) { apply_mgr(sys, r, z); return; }
  PrecReal* xslot = apply_cpr_head(sys, r);
  if (!capturing_) ev_stage2_.tic();
  apply_cpr_tail(sys, r, z, tmpAz, xslot);
  if (!capturing_) ev_stage2_.toc();
}

// Stage 1: restrict the residual and run the multigrid on the pressure block.
PrecReal* CprSolver::apply_cpr_head(const DeviceLinearSystem& sys,
                                    const Real* r) {
  const int threads = 256;
  const int cb = (sys.cells + threads - 1) / threads;
  // Take the block size from the *system*, not from the constructor argument.
  // One CprSolver instance is reused for systems of different block size -- the
  // sequential scheme hands it scalar (bs = 1) pressure and transport systems
  // while the solver was constructed for the bs = 3 coupled system -- and a
  // stale bs_ makes k_extract_p read r[3*c] out of a vector of length cells,
  // which is an illegal memory access rather than a wrong answer.
  const int bs = sys.bs;

  // Write the restricted residual straight into HYPRE's b and zero its x.
  // The IJ route did the same thing via SetValues, SetValues, Assemble,
  // Assemble, and it dominated the whole solver: 27.0 s of a 46.2 s linear
  // solve on one GPU, 37.0 s of 60.9 on two.  Assembling a vector whose
  // partitioning has not changed since setup is work with no result.
  const bool use_amgx = (cfg_.backend == SolverConfig::kAmgx);
  const bool use_ginkgo = (cfg_.backend == SolverConfig::kGinkgo);
  const bool use_lib = use_amgx || use_ginkgo;
  // AmgX and Ginkgo take the residual from our own staging buffers; only the
  // HYPRE path can write straight into the library's vector storage.
  const bool direct = !use_lib && amg_->b_dev != nullptr &&
                      amg_->x_dev != nullptr;
  PrecReal* bslot = direct ? amg_->b_dev : rp_;
  PrecReal* xslot = direct ? amg_->x_dev : dp_;
  k_extract_p<<<cb, threads>>>(r, sys.pweight, bslot, sys.cells, bs);
  NVRS_CHECK_CUDA(cudaMemset(xslot, 0, sizeof(PrecReal) * sys.cells));
  if (!use_lib && !direct) {
    ev_ijvec_.tic();
    HYPRE_IJVectorSetValues(amg_->b, sys.cells, amg_->d_idx,
                            (HYPRE_Complex*)rp_);
    HYPRE_IJVectorSetValues(amg_->x, sys.cells, amg_->d_idx,
                            (HYPRE_Complex*)dp_);
    HYPRE_IJVectorAssemble(amg_->b);
    HYPRE_IJVectorAssemble(amg_->x);
    ev_ijvec_.toc();
  }
  if (use_ginkgo) {
    // Same staging buffers and the same ev_amg_ timing as AmgX, so the three
    // backends report one quantity.
    apply_ginkgo(sys, bslot, xslot);
  } else if (use_amgx) {
    // apply_amgx does its own ev_amg_ timing, so the two backends report
    // the same quantity and the numbers are directly comparable.
    apply_amgx(sys, bslot, xslot);
  } else {
    ev_amg_.tic();
    HYPRE_BoomerAMGSolve(amg_->solver, amg_->parA, amg_->parb, amg_->parx);
    ev_amg_.toc();
    ++timings_.n_amg_solve;
    // How many V-cycles BoomerAMG actually took.  With max_iter = 1 this is
    // 1 per application and the sum is just the count, but it stops being so
    // the moment anyone raises the AMG iteration limit, and then the number
    // matters.
    HYPRE_Int its = 0;
    HYPRE_BoomerAMGGetNumIterations(amg_->solver, &its);
    timings_.amg_iters += (long)its;
  }
  if (!use_lib && !direct) {
    ev_ijvec_.tic();
    HYPRE_IJVectorGetValues(amg_->x, sys.cells, amg_->d_idx,
                            (HYPRE_Complex*)dp_);
    ev_ijvec_.toc();
  }
  return xslot;
}

// CPRW, prolongation and the second stage: our own kernels only, launched on
// st_, so that CUDAGRAPH can record them.  CPRW used to run inside the AmgX
// branch of the head and after the solve for the other two backends; it is
// the first thing here for all three, which is the same order.
void CprSolver::apply_cpr_tail(const DeviceLinearSystem& sys, const Real* r,
                               Real* z, Real* tmpAz, PrecReal* xslot) {
  const int threads = 256;
  const int cb = (sys.cells + threads - 1) / threads;
  const int bs = sys.bs;
  apply_cprw(sys, xslot);
  k_scatter_p<<<cb, threads, 0, st_>>>(xslot, z, sys.cells, bs);
  NVRS_CHECK_CUDA(cudaGetLastError());

  // Stage 2: correction on the full residual -- per-cell block-Jacobi, or the
  // per-column block tridiagonal when ZLINE is selected and factored.
  op_apply(sys, z, tmpAz);
  if (cfg_.stage2 == 1 && zl_tinv_) {
    const int cols = grid_.nxl * grid_.nyl;
    const int cb2 = (cols + threads - 1) / threads;
    k_zline_solve<<<cb2, threads, 0, st_>>>(r, tmpAz, zl_tinv_, zl_l_, zl_u_,
                                            zl_y_, z, grid_.nxl, grid_.nyl,
                                            grid_.nzl, bs);
  } else {
    if (cfg_.stage2 == 2)
      apply_ginkgo_stage2(sys, r, tmpAz, z);
    else
      k_stage2<<<cb, threads, 0, st_>>>(r, tmpAz, sys.dinv, z, sys.cells, bs);
    if (sys.well_schur_modes > 0 && sys.well_schur &&
        sys.well_schur_count > 0 &&
        sys.well_schur_groups > 0) {
      const int modes = sys.well_schur_modes;
      const int nsmall = 12 * sys.well_schur_groups;
      if (well_schur_alloc_ < nsmall) {
        cudaFree(well_schur_dot_);
        NVRS_CHECK_CUDA(cudaMalloc(&well_schur_dot_,
                                   sizeof(Real) * nsmall));
        well_schur_alloc_ = nsmall;
      }
      const auto* we = static_cast<const WellSchurEntry*>(sys.well_schur);
      k_well_prec_reduce<<<sys.well_schur_groups, 128, 0, st_>>>(
          we, sys.well_schur_count, r, tmpAz, sys.dinv, bs, modes,
          well_schur_dot_);
      NVRS_CHECK_CUDA(cudaGetLastError());
      allreduce_dev(well_schur_dot_, nsmall, redbuf_);
      k_well_prec_solve<<<(sys.well_schur_groups + 127) / 128, 128, 0, st_>>>(
          well_schur_dot_, sys.well_schur_groups, modes);
      k_well_prec_apply<<<(sys.well_schur_count + 127) / 128, 128, 0, st_>>>(
          we, sys.well_schur_count, sys.dinv, bs, modes,
          well_schur_dot_, z);
    }
  }
  NVRS_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CUDAGRAPH: the Arnoldi tail recorded once per basis column and replayed.
// ---------------------------------------------------------------------------
// Everything a captured graph bakes in: kernel arguments (pointers, sizes,
// counts) and the branch taken.  If any of it changes the graphs are stale and
// are dropped; they are recaptured on the next use.
void CprSolver::graph_signature(const DeviceLinearSystem& sys,
                                const PrecReal* xslot, bool cgs2,
                                std::vector<unsigned long long>& g) const {
  auto u = [&](const void* p) { g.push_back((unsigned long long)(uintptr_t)p); };
  auto v = [&](long long x) { g.push_back((unsigned long long)x); };
  g.clear();
  v(sys.n); v(sys.n_ext); v(sys.cells); v(sys.bs); v(sys.nnz);
  u(sys.rowptr); u(sys.colind); u(sys.aval64); u(sys.aval_hi); u(sys.aval_lo);
  u(sys.prowptr); u(sys.dinv);
  u(sys.well_schur); v(sys.well_schur_count); v(sys.well_schur_groups);
  v(sys.well_schur_modes);
  u(bcol_); v(bcol_state_);
  u(well_schur_dot_); v(well_schur_alloc_);
  v(cprw_na_); v(cprw_nfull_); u(cprw_q_); u(cprw_coeff_); u(cprw_kinv_);
  u(cprw_active_);
  u(xslot); u(V_); u(Z_); u(tmp_); u(hdev_); v(next_); v(cgs2 ? 1 : 0);
  v(cfg_.stage2); u(zl_tinv_); u(zl_l_); u(zl_u_); u(zl_y_);
  v(operator_mode()); v(cfg_.block_spmv ? 1 : 0);
  u(matA_); u(spmv_buf_);
  u(conn_halo_); v(nccl_ ? 1 : 0); v(grid_.nranks);
}

void CprSolver::graphs_clear() {
  for (auto& e : gexec_)
    if (e) cudaGraphExecDestroy(e);
  gexec_.clear();
  gsig_.clear();
}

// One rank, or several with NCCL carrying the tail's communication (host
// MPI cannot be recorded); our own stage 2 either way.
bool CprSolver::graphs_usable(const DeviceLinearSystem& sys) const {
  return cfg_.cuda_graph && !graphs_off_ && (grid_.nranks == 1 || nccl_) &&
         !cfg_.use_mgr && cfg_.stage2 != 2 && sys.n > 0;
}

// Size the small well buffers for every use in the tail before a capture, so
// no cudaMalloc can happen while a graph is being recorded.
void CprSolver::ensure_small_scratch(const DeviceLinearSystem& sys) {
  const int need = std::max({sys.well_schur_modes * sys.well_schur_groups,
                             12 * sys.well_schur_groups, cprw_nfull_, 1});
  if (well_schur_alloc_ < need) {
    cudaFree(well_schur_dot_);
    NVRS_CHECK_CUDA(cudaMalloc(&well_schur_dot_, sizeof(Real) * need));
    well_schur_alloc_ = need;
  }
  if (nccl_ && !conn_halo_) halo_init(grid_, sys.bs);
}

template <class F>
void CprSolver::graph_phase(int column, int phase, EventSpan& ev, F&& body) {
  const size_t slot = (size_t)column * 3 + phase;
  if (gexec_.size() <= slot) gexec_.resize(slot + 1, nullptr);
  if (!gs_) {
    NVRS_CHECK_CUDA(cudaStreamCreate(&gs_));   // blocking stream
    // A fixed workspace, so cuBLAS never has to allocate while recording.
    const size_t ws = 4u << 20;
    NVRS_CHECK_CUDA(cudaMalloc(&cublas_ws_, ws));
    NVRS_CHECK_CUBLAS(cublasSetWorkspace(blas_, cublas_ws_, ws));
  }
  ev.tic();
  if (!gexec_[slot]) {
    // Host time from the start of recording to an instantiated graph: the
    // price of one capture, which a replay has to earn back.
    const auto tcap = std::chrono::steady_clock::now();
    st_ = gs_;
    capturing_ = true;
    NVRS_CHECK_CUBLAS(cublasSetStream(blas_, gs_));
    cudaError_t err = cudaStreamBeginCapture(gs_, cudaStreamCaptureModeRelaxed);
    if (err == cudaSuccess) body();
    cudaGraph_t g = nullptr;
    const cudaError_t end = cudaStreamEndCapture(gs_, &g);
    if (err == cudaSuccess) err = end;
    st_ = 0;
    capturing_ = false;
    NVRS_CHECK_CUBLAS(cublasSetStream(blas_, 0));
    cudaGraphExec_t ex = nullptr;
    if (err == cudaSuccess && g) err = cudaGraphInstantiateWithFlags(&ex, g, 0);
    if (g) cudaGraphDestroy(g);
    if (err != cudaSuccess || !ex) {
      // Recording failed, so nothing ran.  Turn graphs off for the rest of the
      // run and do the work the ordinary way.
      (void)cudaGetLastError();
      graphs_off_ = true;
      graphs_clear();
      if (grid_.rank == 0)
        std::printf("  [graph] CUDAGRAPH: capture failed (%s); continuing "
                    "without graphs.\n", cudaGetErrorString(err));
      body();
      ev.toc();
      return;
    }
    gexec_[slot] = ex;
    ++timings_.graph_captures;
    timings_.graph_capture_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tcap)
            .count();
  }
  NVRS_CHECK_CUDA(cudaGraphLaunch(gexec_[slot], gs_));
  ++timings_.graph_replays;
  // Phases 0 and 1 each hold one operator apply.
  if (phase <= 1) {
    if (bcol_state_ > 0 && cfg_.block_spmv) ++timings_.spmv_block;
    else ++timings_.spmv_scalar;
    if (operator_mode() == 2) ++timings_.spmv_fp32;
  }
  ev.toc();
}

// Gram-Schmidt projections of w against V[:, 0:nc], and |w|^2 into ndev_.
void CprSolver::ortho_project(int n, int nc, Real* w, bool cgs2) {
  const Real one = 1.0, zero = 0.0, neg = -1.0;
  const int ld = (int)next_;
  // pass 1
  NVRS_CHECK_CUBLAS(cublasDgemv(blas_, CUBLAS_OP_T, n, nc, &one, V_, ld, w, 1,
                                &zero, hdev_, 1));
  allreduce_dev(hdev_, nc, redbuf_);
  NVRS_CHECK_CUBLAS(cublasDgemv(blas_, CUBLAS_OP_N, n, nc, &neg, V_, ld, hdev_,
                                1, &one, w, 1));
  // pass 2.  CGS2 re-projects the *residual* of pass 1, so the two
  // reductions are strictly sequential and cannot be merged into one.
  if (cgs2) {
    NVRS_CHECK_CUBLAS(cublasDgemv(blas_, CUBLAS_OP_T, n, nc, &one, V_, ld, w, 1,
                                  &zero, sdev_, 1));
    allreduce_dev(sdev_, nc, redbuf_);
    NVRS_CHECK_CUBLAS(cublasDgemv(blas_, CUBLAS_OP_N, n, nc, &neg, V_, ld,
                                  sdev_, 1, &one, w, 1));
  }
  // |w|^2 on the device: the same cublasDdot the host-mode norm used, written
  // beside the projections instead of read back on its own.
  NVRS_CHECK_CUBLAS(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_DEVICE));
  NVRS_CHECK_CUBLAS(cublasDdot(blas_, n, w, 1, w, 1, ndev_));
  NVRS_CHECK_CUBLAS(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST));
  allreduce_dev(ndev_, 1, redbuf_);
}

// Extract and factor the ZLINE stage-2 block tridiagonals.  Runs every
// setup() call: assembly refreshes the coupled CSR each Newton iterate, and
// the factor kernel is one launch over the areal columns (milliseconds),
// so unlike the AMG hierarchy there is nothing worth reusing.
void CprSolver::setup_zline(const DeviceLinearSystem& sys) {
  const int cells = sys.cells, bs = sys.bs;
  // A system without the face topology (the sequential scheme's scalar
  // systems) cannot be line-factored; apply_cpr then falls back to Jacobi
  // because zl_tinv_ stays null.
  // A corner-point grid on the connection list (CONNLIST) has no seven-point
  // neighbour map either, and there the fallback used to be silent: a
  // STAGE2 ZLINE run was block-Jacobi and said nothing.  Say so once.
  if (!sys.nbr7 || !sys.bcol7 || cells <= 0) {
    static bool warned = false;
    if (!warned && cfg_.stage2 == 1) {
      warned = true;
      std::fprintf(stderr, "  [warn] STAGE2 ZLINE needs the Cartesian seven-point "
                           "neighbour map, which this grid does not have; stage 2 "
                           "stays block-Jacobi.\n");
    }
    return;
  }
  if (zl_cells_ < cells) {
    if (zl_tinv_) cudaFree(zl_tinv_);
    if (zl_l_) cudaFree(zl_l_);
    if (zl_u_) cudaFree(zl_u_);
    if (zl_y_) cudaFree(zl_y_);
    const size_t nb = (size_t)cells * bs * bs;
    NVRS_CHECK_CUDA(cudaMalloc(&zl_tinv_, sizeof(PrecReal) * nb));
    NVRS_CHECK_CUDA(cudaMalloc(&zl_l_, sizeof(PrecReal) * nb));
    NVRS_CHECK_CUDA(cudaMalloc(&zl_u_, sizeof(PrecReal) * nb));
    NVRS_CHECK_CUDA(cudaMalloc(&zl_y_, sizeof(PrecReal) * (size_t)cells * bs));
    zl_cells_ = cells;
  }
  const int threads = 256;
  const int cb = (cells + threads - 1) / threads;
  k_zline_extract<<<cb, threads>>>(sys.rowptr, sys.aval64, sys.nbr7, sys.bcol7,
                                   zl_tinv_, zl_l_, zl_u_, cells, bs);
  const int cols = grid_.nxl * grid_.nyl;
  const int cb2 = (cols + threads - 1) / threads;
  k_zline_factor<<<cb2, threads>>>(zl_tinv_, zl_l_, zl_u_, sys.dinv,
                                   grid_.nxl, grid_.nyl, grid_.nzl, bs);
  NVRS_CHECK_CUDA(cudaGetLastError());
}

namespace {

// Global block column of every CSR entry, recovered from the pressure CSR.
__global__ void k_block_cols(const int* __restrict__ prowptr,
                             const int* __restrict__ pcolind,
                             const int* __restrict__ rowptr,
                             HYPRE_BigInt* __restrict__ bcol, int cells,
                             int bs) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const int p0 = prowptr[c], ncols = prowptr[c + 1] - p0;
  for (int a = 0; a < bs; ++a) {
    const int base = rowptr[bs * c + a];
    for (int m = 0; m < ncols; ++m) {
      const HYPRE_BigInt g = (HYPRE_BigInt)bs * (HYPRE_BigInt)pcolind[p0 + m];
      for (int v = 0; v < bs; ++v) bcol[base + m * bs + v] = g + v;
    }
  }
}

// Row ids and per-row entry counts.  Rows are contiguous because the cells are.
__global__ void k_block_rows(const int* __restrict__ rowptr,
                             HYPRE_BigInt* __restrict__ rows,
                             HYPRE_Int* __restrict__ ncols, int nrows,
                             long long row0) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows) return;
  rows[r] = (HYPRE_BigInt)(row0 + r);
  ncols[r] = (HYPRE_Int)(rowptr[r + 1] - rowptr[r]);
}

// The operator is FP64; HYPRE is FP32.  Narrowing here rather than storing a
// second copy in the system keeps the preconditioner's precision a property of
// the preconditioner, which is the whole argument of the mixed-precision design.
__global__ void k_narrow(const Real* __restrict__ src,
                         PrecReal* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (PrecReal)src[i];
}
__global__ void k_narrow_vec(const Real* __restrict__ src,
                             PrecReal* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (PrecReal)src[i];
}
__global__ void k_widen_vec(const PrecReal* __restrict__ src,
                            Real* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (Real)src[i];
}

}  // namespace

void CprSolver::setup_mgr(const DeviceLinearSystem& sys) {
  auto& M = *mgr_;
  const int bs = sys.bs, nrows = sys.n;

  if (!M.built) {
    M.cells = sys.cells; M.bs = bs; M.nnz = sys.nnz;
    M.row0 = (long long)bs * sys.row0;
    const long long lo = M.row0, hi = M.row0 + nrows - 1;

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, lo, hi, lo, hi, &M.A);
    HYPRE_IJMatrixSetObjectType(M.A, HYPRE_PARCSR);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, lo, hi, &M.b);
    HYPRE_IJVectorSetObjectType(M.b, HYPRE_PARCSR);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, lo, hi, &M.x);
    HYPRE_IJVectorSetObjectType(M.x, HYPRE_PARCSR);

    NVRS_CHECK_CUDA(cudaMalloc(&M.d_rows, sizeof(HYPRE_BigInt) * nrows));
    NVRS_CHECK_CUDA(cudaMalloc(&M.d_ncols, sizeof(HYPRE_Int) * nrows));
    NVRS_CHECK_CUDA(cudaMalloc(&M.d_cols, sizeof(HYPRE_BigInt) * sys.nnz));
    NVRS_CHECK_CUDA(cudaMalloc(&M.d_vals, sizeof(PrecReal) * sys.nnz));
    NVRS_CHECK_CUDA(cudaMalloc(&M.d_b32, sizeof(PrecReal) * nrows));
    NVRS_CHECK_CUDA(cudaMalloc(&M.d_x32, sizeof(PrecReal) * nrows));

    k_block_rows<<<grid1d(nrows), 256>>>(sys.rowptr, M.d_rows, M.d_ncols,
                                         nrows, M.row0);
    k_block_cols<<<grid1d(sys.cells), 256>>>(sys.prowptr, sys.pcolind,
                                             sys.rowptr, M.d_cols, sys.cells,
                                             bs);
    NVRS_CHECK_CUDA(cudaGetLastError());

    // --- the reduction itself ------------------------------------------
    HYPRE_MGRCreate(&M.mgr);
    M.cidx[0] = 0;                       // pressure is unknown 0 of the block
    M.cidx_p[0] = M.cidx;
    M.ncoarse[0] = 1;
    // One reduction level: C = pressure, F = saturations/compositions.
    HYPRE_MGRSetCpointsByBlock(M.mgr, bs, 1, M.ncoarse, M.cidx_p);
    HYPRE_MGRSetNonCpointsToFpoints(M.mgr, 1);
    // Jacobi F-relaxation: the alternatives that are stronger on a CPU are the
    // ones with sequential dependencies, which is the wrong trade on a GPU.
    HYPRE_MGRSetRelaxType(M.mgr, cfg_.mgr_relax_type);
    HYPRE_MGRSetNumRelaxSweeps(M.mgr, cfg_.mgr_relax_sweeps);
    HYPRE_MGRSetInterpType(M.mgr, cfg_.mgr_interp_type);
    HYPRE_MGRSetGlobalSmoothType(M.mgr, cfg_.mgr_global_smooth);
    HYPRE_MGRSetMaxGlobalSmoothIters(M.mgr, cfg_.mgr_global_sweeps);
    HYPRE_MGRSetMaxIter(M.mgr, 1);       // one application, as a preconditioner
    HYPRE_MGRSetTol(M.mgr, 0.0);
    HYPRE_MGRSetPrintLevel(M.mgr, cfg_.amg_print_level);

    // The reduced pressure system goes to BoomerAMG with the same settings the
    // hand-rolled CPR used, so a comparison between the two is a comparison of
    // the reduction and not of the multigrid underneath it.
    // Identical settings to the CPR pressure solve, so comparing the
    // two preconditioners compares the reduction and not the multigrid.
    HYPRE_BoomerAMGCreate(&M.coarse);
    configure_amg(M.coarse);
    HYPRE_MGRSetCoarseSolver(M.mgr, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup,
                             M.coarse);
    M.built = true;
    if (grid_.rank == 0)
      std::printf("  Precond  : MGR (C = pressure, F-relax %d x%d, coarse "
                  "BoomerAMG)\n",
                  cfg_.mgr_relax_type, cfg_.mgr_relax_sweeps);
  }

  // Values change every rebuild; the sparsity does not.
  k_narrow<<<grid1d(sys.nnz), 256>>>(sys.aval64, M.d_vals, sys.nnz);
  NVRS_CHECK_CUDA(cudaGetLastError());

  HYPRE_IJMatrixInitialize(M.A);
  HYPRE_IJMatrixSetValues(M.A, nrows, M.d_ncols, M.d_rows, M.d_cols,
                          reinterpret_cast<const HYPRE_Complex*>(M.d_vals));
  HYPRE_IJMatrixAssemble(M.A);
  HYPRE_IJMatrixGetObject(M.A, (void**)&M.parA);

  // Zero them explicitly.  IJVectorInitialize does not define the values,
  // and MGRSetup inspects the vectors, so leaving them at whatever cudaMalloc
  // returned put a NaN through the whole first solve.  The CPR path always
  // did this (its d_zero array); the MGR path was missing it.
  NVRS_CHECK_CUDA(cudaMemset(M.d_b32, 0, sizeof(PrecReal) * nrows));
  NVRS_CHECK_CUDA(cudaMemset(M.d_x32, 0, sizeof(PrecReal) * nrows));
  HYPRE_IJVectorInitialize(M.b);
  HYPRE_IJVectorInitialize(M.x);
  HYPRE_IJVectorSetValues(M.b, nrows, M.d_rows, (HYPRE_Complex*)M.d_b32);
  HYPRE_IJVectorSetValues(M.x, nrows, M.d_rows, (HYPRE_Complex*)M.d_x32);
  HYPRE_IJVectorAssemble(M.b);
  HYPRE_IJVectorAssemble(M.x);
  HYPRE_IJVectorGetObject(M.b, (void**)&M.parb);
  HYPRE_IJVectorGetObject(M.x, (void**)&M.parx);

  {
    const double t = now_synced();
    HYPRE_MGRSetup(M.mgr, M.parA, M.parb, M.parx);
    timings_.amg_setup += now_synced() - t;
    ++timings_.n_amg_setup;
  }
  reuse_left_ = std::max(0, cfg_.amg_reuse);
  iters_fresh_ = -1;
  amg_fresh_ = true;
  amg_pval_ = (const void*)sys.aval64;
  amg_cells_ = sys.cells;
  amg_pnnz_ = sys.nnz;
}

void CprSolver::apply_mgr(const DeviceLinearSystem& sys, const Real* r,
                          Real* z) {
  auto& M = *mgr_;
  const int nrows = sys.n;
  // FP64 residual in, FP32 for the preconditioner, FP64 correction out: the
  // narrowing is the mixed-precision boundary and it lives here, not in the
  // Krylov recurrences.
  k_narrow_vec<<<grid1d(nrows), 256>>>(r, M.d_b32, nrows);
  NVRS_CHECK_CUDA(cudaMemset(M.d_x32, 0, sizeof(PrecReal) * nrows));
  HYPRE_IJVectorSetValues(M.b, nrows, M.d_rows, (HYPRE_Complex*)M.d_b32);
  HYPRE_IJVectorSetValues(M.x, nrows, M.d_rows, (HYPRE_Complex*)M.d_x32);
  HYPRE_IJVectorAssemble(M.b);
  HYPRE_IJVectorAssemble(M.x);

  ev_amg_.tic();
  HYPRE_MGRSolve(M.mgr, M.parA, M.parb, M.parx);
  ev_amg_.toc();
  ++timings_.n_amg_solve;

  HYPRE_IJVectorGetValues(M.x, nrows, M.d_rows, (HYPRE_Complex*)M.d_x32);
  k_widen_vec<<<grid1d(nrows), 256>>>(M.d_x32, z, nrows);
  NVRS_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Outer FP64 FGMRES with right CPR preconditioning (Givens QR, CGS1/CGS2).
// ---------------------------------------------------------------------------
SolveStats CprSolver::fgmres(const DeviceLinearSystem& sys, Real* x,
                            bool zero_initial_guess) {
  const int n = sys.n, restart = cfg_.restart;
  const Real tiny = 1e-300;
  ensure_scratch(sys);
  const size_t vstride = (size_t)next_;   // column stride of V_ / Z_
  const int ldh = restart + 1;
  std::vector<Real> Hbar((size_t)ldh * restart, 0), cs(restart, 0), sn(restart, 0),
      g(restart + 1, 0), hcol(restart + 1, 0), yh(restart, 0);
  // CGS2 is the robust default. A one-pass screen is useful on large,
  // well-preconditioned systems where the second projection/reduction can
  // cost more than the loss of orthogonality; it remains explicitly opt-in.
  const char* cgs_passes_env = std::getenv("NVRS_CGS_PASSES");
  const int cgs_passes = cgs_passes_env ? std::atoi(cgs_passes_env)
                                        : cfg_.ortho_passes;
  const bool cgs2 = cgs_passes != 1;
  // Host copy of [h (ldh), s (ldh), |w|^2]: one read per Arnoldi iteration.
  std::vector<Real> pack(2 * (size_t)ldh + 1, 0.0);

  // CUDAGRAPH on several ranks needs NCCL for the tail.  Created here, at the
  // first solve, which every rank reaches together (nccl_init is collective
  // and gives every rank the same answer).
  if (cfg_.cuda_graph && grid_.nranks > 1 && !nccl_tried_) {
    nccl_tried_ = true;
    // NVRS_NCCL=0 keeps the tail on MPI (and uncaptured) on several ranks.
    const char* e = std::getenv("NVRS_NCCL");
    nccl_ = (e && e[0] == '0') ? false : nccl_init(grid_);
    if (!nccl_ && grid_.rank == 0)
      std::printf("  [graph] CUDAGRAPH on %d ranks needs NCCL (%s); the Krylov "
                  "tail keeps MPI and is not captured.\n", grid_.nranks,
                  nccl_built() ? "communicator creation failed"
                               : "this build has no NCCL");
  }
  // CUDAGRAPH: the tail of each Arnoldi step is recorded per basis column.
  // The block index is built (it reads back to the host) before any capture.
  bool graphs = graphs_usable(sys);

  Real bnorm = gnrm2(n, sys.b);
  if (bnorm < tiny) bnorm = tiny;
  SolveStats st;
  int total = 0;
  bool first_residual = true;
  // The Givens recursion's |r_j| / |b| at the point it declared convergence,
  // or 0 when it has not.  The stopping decision is then confirmed on the true
  // residual b - J x, formed with the FP64 operator: at the top of the next
  // pass if iterations remain, otherwise after the loop.  A variable
  // preconditioner can make Z_m numerically rank deficient, and then the
  // recursive estimate understates the true residual; when that happens the
  // solve restarts from the true residual instead of stopping short.
  Real recursive_rel = 0;
  Real true_rel_now = -1;          // >= 0 once the loop exits on a true check
  auto note_gap = [&](Real true_rel) {
    if (recursive_rel <= 0) return;
    const double gap = (double)true_rel / (double)recursive_rel;
    if (gap > timings_.fgmres_gap_max) timings_.fgmres_gap_max = gap;
    ++timings_.fgmres_gap_checks;
    if (true_rel > cfg_.tol) ++timings_.fgmres_gap_restarts;
    recursive_rel = 0;
  };

  while (total < cfg_.max_iters) {
    if (first_residual && zero_initial_guess) {
      NVRS_CHECK_CUDA(cudaMemcpy(r_, sys.b, sizeof(Real) * n,
                                 cudaMemcpyDeviceToDevice));
    } else {
      op_apply_master(sys, x, Ax_);
      k_residual<<<grid1d(n), 256>>>(sys.b, Ax_, r_, n);
    }
    first_residual = false;
    Real rnorm = gnrm2(n, r_);
    Real rel = rnorm / bnorm;
    note_gap(rel);
    if (rel <= cfg_.tol) {
      st.converged = true;
      true_rel_now = rel;          // x has not moved since: no need to re-form
      break;
    }

    const int mmax = std::min(restart, cfg_.max_iters - total);
    k_scale_copy<<<grid1d(n), 256>>>(V_, r_, 1.0 / rnorm, n);
    std::fill(g.begin(), g.end(), 0.0);
    g[0] = rnorm;
    int used = 0;

    for (int j = 0; j < mmax; ++j) {
      Real* vj = V_ + (size_t)j * vstride;
      Real* zj = Z_ + (size_t)j * vstride;
      // apply_cpr uses tmp_ as its own scratch, so the Arnoldi vector cannot
      // alias it; w is the (j+1)-th basis slot, overwritten by the SpMV below.
      Real* w = V_ + (size_t)(j + 1) * vstride;
      const int nc = j + 1;
      if (graphs && !cfg_.use_mgr) {
        // Stage 1 as usual (the AMG library synchronises inside), then the
        // three recorded phases.  Each keeps its own timer, so the report's
        // split of the linear solve is unchanged.
        PrecReal* xslot = apply_cpr_head(sys, vj);
        ensure_small_scratch(sys);
        (void)block_ready(sys);
        std::vector<unsigned long long> sig;
        graph_signature(sys, xslot, cgs2, sig);
        if (sig != gsig_) { graphs_clear(); gsig_ = sig; }
        graph_phase(j, 0, ev_stage2_,
                    [&] { apply_cpr_tail(sys, vj, zj, tmp_, xslot); });
        graph_phase(j, 1, ev_spmv_, [&] { op_apply(sys, zj, w); });
        graph_phase(j, 2, ev_ortho_, [&] { ortho_project(n, nc, w, cgs2); });
        // CUDAGRAPHMINREUSE: once enough captures have been seen, stop
        // capturing if the graphs are not being reused often enough to pay
        // for their recording.  graph_replays counts the launch that follows
        // each capture too, so reuses proper are replays - captures.
        if (cfg_.graph_min_reuse > 0 &&
            timings_.graph_captures >= cfg_.graph_guard_after &&
            !timings_.graph_guard_fired) {
          const double reuse =
              (double)(timings_.graph_replays - timings_.graph_captures) /
              (double)timings_.graph_captures;
          if (reuse < cfg_.graph_min_reuse) {
            timings_.graph_guard_fired = 1;
            timings_.graph_guard_at = timings_.graph_captures;
            graphs_off_ = true;
            graphs_clear();
            if (grid_.rank == 0)
              std::printf("  [graph] %.2f reuses per capture after %ld "
                          "captures (%.1f us each); below CUDAGRAPHMINREUSE "
                          "%.2f, so the Krylov tail is launched directly for "
                          "the rest of the run.\n",
                          reuse, timings_.graph_captures,
                          1e6 * timings_.graph_capture_seconds /
                              (double)timings_.graph_captures,
                          cfg_.graph_min_reuse);
          }
        }
        graphs = graphs_usable(sys);   // a failed capture turns them off
      } else {
        apply_cpr(sys, vj, zj, tmp_);
        op_apply(sys, zj, w);
        // Classical Gram-Schmidt orthogonalisation with MPI-reduced
        // projections, and |w|^2, all left on the device.
        ev_ortho_.tic();
        ortho_project(n, nc, w, cgs2);
        ev_ortho_.toc();
      }
      // The one host read of the iteration: both projection columns and the
      // squared norm come back together.  It used to be three synchronising
      // reads (two copies and a host-mode dot).
      NVRS_CHECK_CUDA(cudaMemcpy(pack.data(), hdev_, sizeof(Real) * pack.size(),
                                 cudaMemcpyDeviceToHost));
      for (int i = 0; i < nc; ++i)
        hcol[i] = pack[i] + (cgs2 ? pack[(size_t)ldh + i] : 0.0);
      Real hn = std::sqrt(pack[2 * (size_t)ldh]);
      // A non-finite Arnoldi norm means the preconditioner returned garbage.
      // Without this the run does not stop: every comparison against a NaN is
      // false, so `hn <= tiny` does not break, `rel` becomes NaN, and the loop
      // grinds out all max_iters iterations before reporting "rel nan" -- which
      // says nothing about where it came from.  Stop at the iteration that
      // produced it and name the preconditioner.
      if (!std::isfinite(hn)) {
        // Once.  A failing preconditioner fails on every solve of every Newton
        // of every chop, and several hundred identical lines bury the run.
        static bool said = false;
        if (grid_.rank == 0 && !said) {
          said = true;
          std::printf("  [error] %s returned a non-finite correction on Krylov "
                      "iteration %d; abandoning this solve. Further "
                      "occurrences are not reported.\n",
                      cfg_.use_mgr ? "MGR" : "CPR", total + 1);
        }
        st.iters = total;
        st.converged = false;
        st.final_residual = std::numeric_limits<Real>::quiet_NaN();
        st.sol_norm = 0.0;
        return st;
      }
      Real* Hj = &Hbar[(size_t)j * ldh];
      for (int i = 0; i < nc; ++i) Hj[i] = hcol[i];
      Hj[j + 1] = hn;
      // w already *is* basis slot j+1, so this normalises in place.
      if (hn > tiny)
        k_scale_copy<<<grid1d(n), 256>>>(w, w, 1.0 / hn, n);

      for (int i = 0; i < j; ++i) {
        Real t0 = cs[i] * Hj[i] + sn[i] * Hj[i + 1];
        Hj[i + 1] = -sn[i] * Hj[i] + cs[i] * Hj[i + 1];
        Hj[i] = t0;
      }
      Real denom = std::hypot(Hj[j], Hj[j + 1]);
      if (denom < tiny) denom = tiny;
      cs[j] = Hj[j] / denom;
      sn[j] = Hj[j + 1] / denom;
      Hj[j] = cs[j] * Hj[j] + sn[j] * Hj[j + 1];
      Hj[j + 1] = 0.0;
      g[j + 1] = -sn[j] * g[j];
      g[j] = cs[j] * g[j];
      rel = std::fabs(g[j + 1]) / bnorm;
      used = j + 1;
      total++;
      if (rel <= cfg_.tol || total >= cfg_.max_iters || hn <= tiny) break;
    }

    for (int i = used - 1; i >= 0; --i) {
      Real s = g[i];
      for (int k = i + 1; k < used; ++k) s -= Hbar[(size_t)i + (size_t)k * ldh] * yh[k];
      Real diag = Hbar[(size_t)i + (size_t)i * ldh];
      yh[i] = (std::fabs(diag) > tiny) ? s / diag : 0.0;
    }
    // x += Z y   (accumulate the preconditioned basis) as one gemv rather
    // than `used` separate axpy launches.
    if (used > 0) {
      NVRS_CHECK_CUDA(cudaMemcpy(yv_, yh.data(), sizeof(Real) * used,
                                 cudaMemcpyHostToDevice));
      const Real one = 1.0;
      NVRS_CHECK_CUBLAS(cublasDgemv(blas_, CUBLAS_OP_N, n, used, &one, Z_,
                                    (int)vstride, yv_, 1, &one, x, 1));
    }
    // The recursion says converged: confirm it on the true residual, at the
    // top of the next pass (which restarts if it is not in fact met).
    if (rel <= cfg_.tol) recursive_rel = rel > tiny ? rel : tiny;
  }

  if (true_rel_now >= 0) {
    st.final_residual = true_rel_now;
  } else {
    op_apply_master(sys, x, Ax_);
    k_residual<<<grid1d(n), 256>>>(sys.b, Ax_, r_, n);
    st.final_residual = gnrm2(n, r_) / bnorm;
    note_gap(st.final_residual);
  }
  st.sol_norm = gnrm2(n, x);
  st.iters = total;
  st.converged = st.final_residual <= cfg_.tol;
  return st;
}

// ---------------------------------------------------------------------------
// Outer FP64 BiCGSTAB with right CPR preconditioning.
// ---------------------------------------------------------------------------
SolveStats CprSolver::bicgstab(const DeviceLinearSystem& sys, Real* x,
                              bool zero_initial_guess) {
  const int n = sys.n;
  const Real tiny = 1e-300;
  ensure_scratch(sys);
  SolveStats st;

  if (zero_initial_guess) {
    NVRS_CHECK_CUDA(cudaMemcpy(r_, sys.b, sizeof(Real) * n,
                               cudaMemcpyDeviceToDevice));
  } else {
    op_apply(sys, x, Ax_);
    k_residual<<<grid1d(n), 256>>>(sys.b, Ax_, r_, n);
  }
  NVRS_CHECK_CUDA(cudaMemcpy(rhat_, r_, sizeof(Real) * n, cudaMemcpyDeviceToDevice));
  Real bnorm = gnrm2(n, sys.b);
  if (bnorm < tiny) bnorm = tiny;

  Real rho = 1, alpha = 1, omega = 1;
  NVRS_CHECK_CUDA(cudaMemset(p_, 0, sizeof(Real) * n));
  NVRS_CHECK_CUDA(cudaMemset(v_, 0, sizeof(Real) * n));

  int it = 0;
  Real rel = gnrm2(n, r_) / bnorm;
  for (; it < cfg_.max_iters && rel > cfg_.tol; ++it) {
    Real rho_new = gdot(n, rhat_, r_);
    Real beta = (rho_new / (rho + tiny)) * (alpha / (omega + tiny));
    rho = rho_new;
    // p = r + beta (p - omega v)
    k_axpy<<<grid1d(n), 256>>>(p_, -omega, v_, n);           // p -= omega v
    k_scale_copy<<<grid1d(n), 256>>>(tmp_, p_, beta, n);      // tmp = beta p
    k_axpy<<<grid1d(n), 256>>>(tmp_, 1.0, r_, n);            // tmp = r + beta p
    NVRS_CHECK_CUDA(cudaMemcpy(p_, tmp_, sizeof(Real) * n, cudaMemcpyDeviceToDevice));

    apply_cpr(sys, p_, phat_, tmp_);   // phat = M^{-1} p
    op_apply(sys, phat_, v_);          // v = A phat
    alpha = rho / (gdot(n, rhat_, v_) + tiny);
    // s = r - alpha v
    NVRS_CHECK_CUDA(cudaMemcpy(s_, r_, sizeof(Real) * n, cudaMemcpyDeviceToDevice));
    k_axpy<<<grid1d(n), 256>>>(s_, -alpha, v_, n);
    rel = gnrm2(n, s_) / bnorm;
    if (rel <= cfg_.tol) {
      k_axpy<<<grid1d(n), 256>>>(x, alpha, phat_, n);
      ++it; break;
    }
    apply_cpr(sys, s_, shat_, tmp_);   // shat = M^{-1} s
    op_apply(sys, shat_, t_);          // t = A shat
    omega = gdot(n, t_, s_) / (gdot(n, t_, t_) + tiny);
    k_axpy<<<grid1d(n), 256>>>(x, alpha, phat_, n);
    k_axpy<<<grid1d(n), 256>>>(x, omega, shat_, n);
    NVRS_CHECK_CUDA(cudaMemcpy(r_, s_, sizeof(Real) * n, cudaMemcpyDeviceToDevice));
    k_axpy<<<grid1d(n), 256>>>(r_, -omega, t_, n);
    rel = gnrm2(n, r_) / bnorm;
  }
  op_apply(sys, x, Ax_);
  k_residual<<<grid1d(n), 256>>>(sys.b, Ax_, r_, n);
  st.final_residual = gnrm2(n, r_) / bnorm;
  st.sol_norm = gnrm2(n, x);
  st.iters = it;
  st.converged = st.final_residual <= cfg_.tol;
  return st;
}

SolveStats CprSolver::solve(const DeviceLinearSystem& sys, Real* x,
                            bool zero_initial_guess) {
  SolveStats st;
  const char* wb_env = std::getenv("NVRS_WELL_WOODBURY");
  const bool use_wb = cfg_.krylov == 0 && sys.well_schur &&
      sys.well_schur_count > 0 && sys.well_schur_groups > 0 &&
      sys.well_schur_modes > 0 && wb_env && std::atoi(wb_env) != 0;

  if (use_wb) {
    // A = A0 + U V^T.  Solve against A0 with the same CPR hierarchy, form
    // K = I + V^T A0^-1 U, and x = A0^-1 b - A0^-1 U K^-1 V^T A0^-1 b.
    // The final coupled FGMRES call is normally only a residual cleanup; it
    // also makes this safe when the A0 solves stop at an inexact tolerance.
    const int modes = sys.well_schur_modes;
    const int nfull = modes * sys.well_schur_groups;
    std::vector<WellSchurEntry> he(sys.well_schur_count);
    NVRS_CHECK_CUDA(cudaMemcpy(he.data(), sys.well_schur,
                               sizeof(WellSchurEntry) * he.size(),
                               cudaMemcpyDeviceToHost));
    std::vector<int> present(nfull, 0);
    for (const auto& a : he)
      for (int m = 0; m < modes; ++m)
        for (int eq = 0; eq < sys.bs && eq < 3; ++eq)
          if (a.dr_dg[3 * eq + m] != 0) present[modes * a.group + m] = 1;
    if (grid_.nranks > 1)
      MPI_Allreduce(MPI_IN_PLACE, present.data(), nfull, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
    std::vector<int> active;
    for (int i = 0; i < nfull; ++i) if (present[i]) active.push_back(i);

    // A dense solve is tiny for normal well groups.  Cap it defensively: a
    // giant group-controlled network belongs in a dedicated global well
    // solver, and must not unexpectedly reserve hundreds of full GPU vectors.
    if (!active.empty() && active.size() <= 16) {
      const int na = (int)active.size();
      if (well_lr_next_alloc_ < sys.n_ext || well_lr_modes_alloc_ < na) {
        cudaFree(well_lr_rhs_); cudaFree(well_lr_z_);
        NVRS_CHECK_CUDA(cudaMalloc(&well_lr_rhs_, sizeof(Real) * sys.n_ext));
        NVRS_CHECK_CUDA(cudaMalloc(&well_lr_z_,
                                   sizeof(Real) * (size_t)sys.n_ext * na));
        well_lr_next_alloc_ = sys.n_ext;
        well_lr_modes_alloc_ = na;
      }
      DeviceLinearSystem base = sys;
      base.well_schur = nullptr;
      base.well_schur_count = base.well_schur_groups = 0;
      base.well_schur_modes = 0;

      NVRS_CHECK_CUDA(cudaMemset(x, 0, sizeof(Real) * sys.n_ext));
      SolveStats sb = fgmres(base, x, true);
      int work = sb.iters;
      bool good = sb.converged;
      std::vector<Real> vt_x(nfull), dots(nfull);
      if (good) {
        if (well_schur_alloc_ < nfull) {
          cudaFree(well_schur_dot_);
          NVRS_CHECK_CUDA(cudaMalloc(&well_schur_dot_,
                                     sizeof(Real) * nfull));
          well_schur_alloc_ = nfull;
        }
        k_well_schur_reduce<<<sys.well_schur_groups, 128>>>(
            static_cast<const WellSchurEntry*>(sys.well_schur),
            sys.well_schur_count, x, sys.bs, modes, well_schur_dot_);
        NVRS_CHECK_CUDA(cudaGetLastError());
        allreduce_dev(well_schur_dot_, nfull, redbuf_);
        NVRS_CHECK_CUDA(cudaMemcpy(vt_x.data(), well_schur_dot_,
                                   sizeof(Real) * nfull,
                                   cudaMemcpyDeviceToHost));
      }

      std::vector<Real> K((size_t)na * na, 0.0);
      for (int j = 0; good && j < na; ++j) {
        Real* zj = well_lr_z_ + (size_t)j * sys.n_ext;
        NVRS_CHECK_CUDA(cudaMemset(well_lr_rhs_, 0,
                                   sizeof(Real) * sys.n_ext));
        const int id = active[j], group = id / modes, mode = id % modes;
        k_well_schur_column<<<(sys.well_schur_count + 127) / 128, 128>>>(
            static_cast<const WellSchurEntry*>(sys.well_schur),
            sys.well_schur_count, group, mode, sys.bs, well_lr_rhs_);
        NVRS_CHECK_CUDA(cudaGetLastError());
        NVRS_CHECK_CUDA(cudaMemset(zj, 0, sizeof(Real) * sys.n_ext));
        base.b = well_lr_rhs_;
        SolveStats sz = fgmres(base, zj, true);
        work += sz.iters;
        good = sz.converged;
        if (!good) break;
        k_well_schur_reduce<<<sys.well_schur_groups, 128>>>(
            static_cast<const WellSchurEntry*>(sys.well_schur),
            sys.well_schur_count, zj, sys.bs, modes, well_schur_dot_);
        NVRS_CHECK_CUDA(cudaGetLastError());
        allreduce_dev(well_schur_dot_, nfull, redbuf_);
        NVRS_CHECK_CUDA(cudaMemcpy(dots.data(), well_schur_dot_,
                                   sizeof(Real) * nfull,
                                   cudaMemcpyDeviceToHost));
        for (int i = 0; i < na; ++i)
          K[(size_t)i * na + j] = (i == j ? 1.0 : 0.0) + dots[active[i]];
      }

      // Pivoted Gauss-Jordan solve K c = V^T A0^-1 b.
      std::vector<Real> c(na);
      for (int i = 0; i < na; ++i) c[i] = vt_x[active[i]];
      for (int k = 0; good && k < na; ++k) {
        int piv = k;
        for (int i = k + 1; i < na; ++i)
          if (std::fabs(K[(size_t)i * na + k]) >
              std::fabs(K[(size_t)piv * na + k])) piv = i;
        const Real d = K[(size_t)piv * na + k];
        if (!std::isfinite(d) || std::fabs(d) < 1e-14) { good = false; break; }
        if (piv != k) {
          for (int j = 0; j < na; ++j)
            std::swap(K[(size_t)k * na + j], K[(size_t)piv * na + j]);
          std::swap(c[k], c[piv]);
        }
        const Real inv = 1.0 / K[(size_t)k * na + k];
        for (int j = k; j < na; ++j) K[(size_t)k * na + j] *= inv;
        c[k] *= inv;
        for (int i = 0; i < na; ++i) if (i != k) {
          const Real q = K[(size_t)i * na + k];
          for (int j = k; j < na; ++j)
            K[(size_t)i * na + j] -= q * K[(size_t)k * na + j];
          c[i] -= q * c[k];
        }
      }
      if (good)
        for (int j = 0; j < na; ++j)
          k_axpy<<<grid1d(sys.n), 256>>>(
              x, -c[j], well_lr_z_ + (size_t)j * sys.n_ext, sys.n);
      else
        NVRS_CHECK_CUDA(cudaMemset(x, 0, sizeof(Real) * sys.n_ext));

      base.b = sys.b;
      st = fgmres(sys, x, false);
      st.iters += work;
    } else {
      st = fgmres(sys, x, zero_initial_guess);
    }
  } else {
    st = (cfg_.krylov == 1) ? bicgstab(sys, x, zero_initial_guess)
                            : fgmres(sys, x, zero_initial_guess);
  }

  // Fold the asynchronously recorded phase times in once per solve, rather
  // than synchronising inside the Krylov loop to read each one.
  timings_.amg_solve += ev_amg_.drain();
  timings_.stage2 += ev_stage2_.drain();
  timings_.spmv += ev_spmv_.drain();
  timings_.ortho += ev_ortho_.drain();
  timings_.ijvec += ev_ijvec_.drain();

  // --- growth guard on the reused hierarchy --------------------------------
  // Reuse is only a win while the frozen preconditioner still costs about as
  // many Krylov iterations as a fresh one.  Once it does not, the saved setup
  // is being paid for several times over in V-cycles, so stop reusing and let
  // the next setup() rebuild.  A solve that failed to converge is treated the
  // same way: the cap is the worst possible iteration count, so it is evidence
  // the preconditioner has gone stale, not evidence that it is holding up.
  if (cfg_.amg_reuse > 0) {
    if (amg_fresh_) {
      iters_fresh_ = st.iters;
    } else if (iters_fresh_ > 0) {
      const double limit = cfg_.amg_reuse_growth * (double)iters_fresh_;
      if (!st.converged || (double)st.iters > limit) {
        if (reuse_left_ > 0) ++timings_.n_amg_forced;
        reuse_left_ = 0;
      }
    }
  }
  return st;
}

}  // namespace nvrs
