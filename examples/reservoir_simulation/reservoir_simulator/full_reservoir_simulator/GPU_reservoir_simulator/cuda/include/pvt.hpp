// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// pvt.hpp — black-oil PVT correlations.  These are a 1:1 port of the CuPy
// helpers calc_mu_g / calc_rs / calc_dp / calc_bg / calc_bo in NVR.py, kept as
// header-only __host__ __device__ inlines so the same code runs on the CPU
// (deck echo, well post-processing) and inside CUDA kernels.
#pragma once

#include "common.hpp"
#include <cmath>

namespace nvrs {
namespace pvt {

// These correlations are exponential in pressure, so a Newton iterate that
// wanders far outside the physical range overflows them: exp(1.7e-3 * dp)
// reaches the double range at dp ~ 4e5, and B_o underflows to zero, after
// which 1/B_o is inf and the residual becomes NaN.  The NaN then propagates
// for the rest of the run and the only symptom is "|R0|=-nan".
//
// Clamping the *correlation argument* keeps the fluid properties finite and
// monotone without changing anything inside the physical envelope.  This is
// only used by refresh_pvt (once per report step, on the host); the Jacobian
// holds PVT frozen within a Newton step, so no derivative is affected.
constexpr Real kPmin = 1.0;        // [psia]
constexpr Real kPmax = 2.0e5;      // [psia]
__host__ __device__ inline Real clamp_p(Real p) {
  return (p < kPmin) ? kPmin : ((p > kPmax) ? kPmax : p);
}

// Gas viscosity [cP] — quadratic (Lee-Kesler style) fit in pressure [psi].
__host__ __device__ inline Real mu_g(Real p_in) {
  const Real p = clamp_p(p_in);
  return 3e-10 * p * p + 1e-6 * p + 0.0133;
}

// Solution gas-oil ratio RS [SCF/STB] (Standing 1947).  Constant above Pb.
__host__ __device__ inline Real rs(Real p_bub, Real p_in) {
  const Real p = clamp_p(p_in);
  const Real C = 178.11 * 178.11 / 5.615;  // ≈ 5658 SCF/STB
  return (p < p_bub) ? C * pow(p / p_bub, 1.3) : C;
}

// Effective pressure differential used by the FVF fits [psi].
__host__ __device__ inline Real dp(Real p_bub, Real p_atm, Real p_in) {
  const Real p = clamp_p(p_in);
  return (p < p_bub) ? (p_atm - p) : (p_atm - p_bub);
}

// Gas formation volume factor BG [RB/SCF].
__host__ __device__ inline Real bg(Real p_bub, Real p_atm, Real p) {
  return 1.0 / exp(1.7e-3 * dp(p_bub, p_atm, p));
}

// Oil formation volume factor BO [RB/STB].
__host__ __device__ inline Real bo(Real p_bub, Real p_atm, Real cfo, Real p_in) {
  const Real p = clamp_p(p_in);
  if (p < p_bub) return 1.0 / exp(-8e-5 * (p_atm - p));
  return 1.0 / (exp(-8e-5 * (p_atm - p_bub)) * exp(-cfo * (p - p_bub)));
}

}  // namespace pvt
}  // namespace nvrs
