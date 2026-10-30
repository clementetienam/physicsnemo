// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// common.hpp — shared types, error-checking macros and small utilities for the
// NVRS-CUDA reservoir simulator.  This is the C++/CUDA translation of the
// original CuPy `GPU_reservoir_simulator` (NVRS).  The numerical model, the
// TPFA finite-volume discretisation and the CPR-preconditioned Krylov solver
// mirror the Python reference one-to-one; only the execution layer changed.
#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvrs {

// ---------------------------------------------------------------------------
// Scalar types.  The state (pressure, saturations) and the *outer* Krylov
// solve run in FP64; the CPR preconditioner (HYPRE BoomerAMG + block smoother)
// runs in FP32 by default — exactly the mixed-precision split validated in the
// Mixed_precision reference.  Flip PrecReal to double to get an all-FP64 build.
// ---------------------------------------------------------------------------
using Real     = double;                 // primary state / operator precision
#ifndef NVRS_FP32_PRECOND
#define NVRS_FP32_PRECOND 1
#endif
#if NVRS_FP32_PRECOND
using PrecReal = float;                   // preconditioner precision (mixed)
#else
using PrecReal = double;                  // preconditioner precision (fp64)
#endif

// ---------------------------------------------------------------------------
// Error-checking macros (identical style to hypre_cusolver_mixed_cpr.cu).
// ---------------------------------------------------------------------------
#define NVRS_CHECK_CUDA(x)                                                      \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      std::ostringstream os_;                                                  \
      os_ << "CUDA error '" << cudaGetErrorString(e_) << "' at " << __FILE__   \
          << ":" << __LINE__;                                                  \
      throw std::runtime_error(os_.str());                                     \
    }                                                                          \
  } while (0)

#define NVRS_CHECK_CUSPARSE(x)                                                  \
  do {                                                                         \
    cusparseStatus_t e_ = (x);                                                 \
    if (e_ != CUSPARSE_STATUS_SUCCESS) {                                       \
      std::ostringstream os_;                                                  \
      os_ << "cuSPARSE error " << (int)e_ << " at " << __FILE__ << ":"         \
          << __LINE__;                                                         \
      throw std::runtime_error(os_.str());                                     \
    }                                                                          \
  } while (0)

#define NVRS_CHECK_CUBLAS(x)                                                    \
  do {                                                                         \
    cublasStatus_t e_ = (x);                                                   \
    if (e_ != CUBLAS_STATUS_SUCCESS) {                                         \
      std::ostringstream os_;                                                  \
      os_ << "cuBLAS error " << (int)e_ << " at " << __FILE__ << ":"           \
          << __LINE__;                                                         \
      throw std::runtime_error(os_.str());                                     \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Physical constants and field-unit conversions (Eclipse/OPM field units).
// ---------------------------------------------------------------------------
namespace units {
constexpr Real kPi          = 3.14159265358979323846;
// Darcy field-unit transmissibility constant: q[STB/D] = C * k[mD]*A/(mu*B*L).
constexpr Real kDarcyField  = 1.127e-3;   // (bbl·cP)/(mD·psi·day·ft)
constexpr Real kGravity     = 0.4335;     // psi/ft for 1.0 g/cc fluid (rho*g)
// Cell bulk volumes are computed in ft^3 from DX*DY*DZ, but ECLIPSE reports
// volumes in reservoir barrels in FIELD units, so anything written to an
// output file has to cross this.
constexpr Real kFt3PerBbl   = 5.614583;   // ft^3 per reservoir barrel
}  // namespace units

// ---------------------------------------------------------------------------
// Compile-time capacity limits.  The compositional flash keeps its per-cell
// working set in registers/local memory, so the component count is bounded at
// compile time.  Every kernel that sizes a per-thread array by the block size
// must use kMaxBlockSize, and every entry point that accepts NCOMP from a deck
// must validate against kMaxComponents.
// ---------------------------------------------------------------------------
constexpr int kMaxComponents = 8;                     // max NCOMP
constexpr int kMaxBlockSize  = kMaxComponents + 1;    // pressure + components

// A cell block-size = number of primary unknowns per grid cell.
//   2  → two-phase (pressure, Sw)         black oil
//   3  → three-phase (pressure, Sw, Sg)   black oil
//   nc+1 → compositional (pressure, N_1..N_nc overall molar densities)
enum class Model { BlackOil2P, BlackOil3P, Compositional };

inline int block_size(Model m, int ncomp = 0) {
  switch (m) {
    case Model::BlackOil2P:    return 2;
    case Model::BlackOil3P:    return 3;
    case Model::Compositional: return ncomp + 1;  // pressure + nc molar comps
  }
  return 1;
}

// Small host helper: launch config for a 1-D grid of `n` threads.
inline dim3 grid1d(int n, int block = 256) {
  return dim3((n + block - 1) / block);
}

}  // namespace nvrs
