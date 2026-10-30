// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// ginkgo_kernels.hpp -- the device kernels the Ginkgo backend needs around the
// library: symmetric Jacobi scaling of the pressure system, and the narrowing
// to and widening from IEEE half precision.
//
// They live in a CUDA translation unit of their own so that ginkgo_backend.cpp
// can be compiled by the host compiler against the Ginkgo headers, and so that
// they build whether or not Ginkgo is present.  Half-precision values cross
// this interface as raw IEEE binary16 bit patterns (std::uint16_t), which is
// exactly the layout of gko::half.
//
// Why the scaling is not optional in FP16: the pressure matrix is assembled in
// field units, and its entries run well past 65,504, the largest finite half.
// Narrowing it as it stands turns the hierarchy into infinities.  With
//     A~ = D^-1/2 A D^-1/2,   D = |diag A|,
// the diagonal is +-1 and a diagonally dominant row has every entry at most 1
// in magnitude, and the right-hand side D^-1/2 b is divided by its largest
// entry, so every half-precision number the multigrid touches is O(1).  The
// multigrid is linear in its right-hand side, so the correction is recovered
// exactly by x = s D^-1/2 y.
#pragma once

#include "common.hpp"

#include <cstdint>

namespace nvrs {
namespace gk {

// d[i] = 1 / sqrt(|a_ii|), or 1 where the diagonal is zero or not stored.
// Columns are global pressure ids; the diagonal of local row i is row0 + i.
void jacobi_scale(int cells, long long row0, const int* rowptr,
                  const int* colind, const PrecReal* val, float* d);

// out[k] = d[i] a_ik d[j], narrowed to half (clamped to +-65504) or to float.
void scale_matrix_f16(int cells, long long row0, const int* rowptr,
                      const int* colind, const PrecReal* val, const float* d,
                      std::uint16_t* out);
void scale_matrix_f32(int cells, long long row0, const int* rowptr,
                      const int* colind, const PrecReal* val, const float* d,
                      float* out);

// work[i] = d[i] b[i]; s = max |work|; out[i] = work[i] / s.  Returns s, or 1
// for an all-zero right-hand side.  One device synchronisation, for s.
float scale_rhs_f16(int cells, const PrecReal* b, const float* d, float* work,
                    std::uint16_t* out);
float scale_rhs_f32(int cells, const PrecReal* b, const float* d, float* out);

// x[i] = s d[i] y[i], widened from half or float.
void unscale_f16(int cells, const std::uint16_t* y, const float* d, float s,
                 PrecReal* x);
void unscale_f32(int cells, const float* y, const float* d, float s,
                 PrecReal* x);

// Stage 2 (STAGE2 GINKGO), on the owned n = bs * cells entries:
//   residual: Az := r - Az
//   update  : Az := y;  z += y
void stage2_residual(int n, const Real* r, Real* Az);
void stage2_update(int n, const Real* y, Real* Az, Real* z);

}  // namespace gk
}  // namespace nvrs
