// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// ginkgo_kernels.cu -- scaling and FP16 conversion around the Ginkgo backend.
// See include/ginkgo_kernels.hpp for what each kernel is for and why the
// scaling is required in half precision.
#include "ginkgo_kernels.hpp"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cmath>

namespace nvrs {
namespace gk {
namespace {

const int kThreads = 256;
inline int blocks(int n) { return (n + kThreads - 1) / kThreads; }

// IEEE binary32 <-> binary16 by bit manipulation, so the result does not
// depend on whether the device or the compiler has a native half type.
// Rounding is to nearest; values beyond the half range clamp to +-65504
// rather than becoming infinities, and NaN stays NaN.
union F32Bits {
  float f;
  std::uint32_t u;
};

__host__ __device__ inline std::uint16_t to_half(float f) {
  F32Bits b;
  b.f = f;
  const std::uint32_t x = b.u;
  const std::uint16_t sign = (std::uint16_t)((x >> 16) & 0x8000u);
  const std::uint32_t ebits = (x >> 23) & 0xffu;
  std::uint32_t m = x & 0x7fffffu;
  if (ebits == 0xffu)                       // inf or NaN
    return (std::uint16_t)(sign | (m ? 0x7e00u : 0x7c00u));
  const int e = (int)ebits - 127 + 15;
  if (e >= 31) return (std::uint16_t)(sign | 0x7bffu);   // clamp to max
  if (e <= 0) {                                          // subnormal or zero
    if (e < -10) return sign;
    m |= 0x800000u;
    const int shift = 14 - e;
    std::uint32_t hm = m >> shift;
    if ((m >> (shift - 1)) & 1u) ++hm;
    return (std::uint16_t)(sign | hm);
  }
  std::uint32_t h = (std::uint32_t)sign | ((std::uint32_t)e << 10) | (m >> 13);
  if (m & 0x1000u) {                                     // round to nearest
    ++h;
    if ((h & 0x7c00u) == 0x7c00u) h = (std::uint32_t)sign | 0x7bffu;
  }
  return (std::uint16_t)h;
}

__host__ __device__ inline float from_half(std::uint16_t h) {
  const std::uint32_t sign = (std::uint32_t)(h & 0x8000u) << 16;
  std::uint32_t e = (h >> 10) & 0x1fu;
  std::uint32_t m = h & 0x3ffu;
  std::uint32_t x;
  if (e == 0) {
    if (m == 0) {
      x = sign;
    } else {                                             // subnormal
      e = 127 - 15 + 1;
      while (!(m & 0x400u)) { m <<= 1; --e; }
      m &= 0x3ffu;
      x = sign | (e << 23) | (m << 13);
    }
  } else if (e == 31) {
    x = sign | 0x7f800000u | (m << 13);
  } else {
    x = sign | ((e - 15 + 127) << 23) | (m << 13);
  }
  F32Bits b;
  b.u = x;
  return b.f;
}

__global__ void k_jacobi(int cells, long long row0, const int* rowptr,
                         const int* colind, const PrecReal* val, float* d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  float dv = 1.0f;
  for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) {
    if ((long long)colind[k] == row0 + i) {
      const double a = std::fabs((double)val[k]);
      if (a > 0.0) dv = (float)(1.0 / std::sqrt(a));
      break;
    }
  }
  d[i] = dv;
}

__host__ __device__ inline float col_scale(const float* d, int cells,
                                           long long row0, int col) {
  const long long j = (long long)col - row0;
  return (j >= 0 && j < cells) ? d[j] : 1.0f;
}

__global__ void k_scale_f16(int cells, long long row0, const int* rowptr,
                            const int* colind, const PrecReal* val,
                            const float* d, std::uint16_t* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  for (int k = rowptr[i]; k < rowptr[i + 1]; ++k)
    out[k] = to_half((float)((double)d[i] * (double)val[k] *
                             (double)col_scale(d, cells, row0, colind[k])));
}

__global__ void k_scale_f32(int cells, long long row0, const int* rowptr,
                            const int* colind, const PrecReal* val,
                            const float* d, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  for (int k = rowptr[i]; k < rowptr[i + 1]; ++k)
    out[k] = (float)((double)d[i] * (double)val[k] *
                     (double)col_scale(d, cells, row0, colind[k]));
}

__global__ void k_rhs(int cells, const PrecReal* b, const float* d,
                      float* work) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  work[i] = (float)((double)d[i] * (double)b[i]);
}

__global__ void k_narrow16(int cells, const float* work, float inv_s,
                           std::uint16_t* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  out[i] = to_half(work[i] * inv_s);
}

__global__ void k_div32(int cells, float inv_s, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  out[i] *= inv_s;
}

__global__ void k_widen16(int cells, const std::uint16_t* y, const float* d,
                          float s, PrecReal* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  x[i] = (PrecReal)((double)s * (double)d[i] * (double)from_half(y[i]));
}

__global__ void k_widen32(int cells, const float* y, const float* d, float s,
                          PrecReal* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  x[i] = (PrecReal)((double)s * (double)d[i] * (double)y[i]);
}

__global__ void k_s2_res(int n, const Real* r, Real* Az) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Az[i] = r[i] - Az[i];
}

__global__ void k_s2_upd(int n, const Real* y, Real* Az, Real* z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Az[i] = y[i];
  z[i] += y[i];
}

struct AbsF {
  __host__ __device__ float operator()(float v) const {
    return v < 0.0f ? -v : v;
  }
};

float max_abs(int cells, const float* v) {
  if (cells <= 0) return 0.0f;
  return thrust::transform_reduce(thrust::cuda::par, v, v + cells, AbsF(),
                                  0.0f, thrust::maximum<float>());
}

inline void check() { NVRS_CHECK_CUDA(cudaGetLastError()); }

}  // namespace

void jacobi_scale(int cells, long long row0, const int* rowptr,
                  const int* colind, const PrecReal* val, float* d) {
  if (cells <= 0) return;
  k_jacobi<<<blocks(cells), kThreads>>>(cells, row0, rowptr, colind, val, d);
  check();
}

void scale_matrix_f16(int cells, long long row0, const int* rowptr,
                      const int* colind, const PrecReal* val, const float* d,
                      std::uint16_t* out) {
  if (cells <= 0) return;
  k_scale_f16<<<blocks(cells), kThreads>>>(cells, row0, rowptr, colind, val, d,
                                           out);
  check();
}

void scale_matrix_f32(int cells, long long row0, const int* rowptr,
                      const int* colind, const PrecReal* val, const float* d,
                      float* out) {
  if (cells <= 0) return;
  k_scale_f32<<<blocks(cells), kThreads>>>(cells, row0, rowptr, colind, val, d,
                                           out);
  check();
}

float scale_rhs_f16(int cells, const PrecReal* b, const float* d, float* work,
                    std::uint16_t* out) {
  if (cells <= 0) return 1.0f;
  k_rhs<<<blocks(cells), kThreads>>>(cells, b, d, work);
  check();
  float s = max_abs(cells, work);
  if (!(s > 0.0f) || !std::isfinite(s)) s = 1.0f;
  k_narrow16<<<blocks(cells), kThreads>>>(cells, work, 1.0f / s, out);
  check();
  return s;
}

float scale_rhs_f32(int cells, const PrecReal* b, const float* d, float* out) {
  if (cells <= 0) return 1.0f;
  k_rhs<<<blocks(cells), kThreads>>>(cells, b, d, out);
  check();
  float s = max_abs(cells, out);
  if (!(s > 0.0f) || !std::isfinite(s)) s = 1.0f;
  k_div32<<<blocks(cells), kThreads>>>(cells, 1.0f / s, out);
  check();
  return s;
}

void unscale_f16(int cells, const std::uint16_t* y, const float* d, float s,
                 PrecReal* x) {
  if (cells <= 0) return;
  k_widen16<<<blocks(cells), kThreads>>>(cells, y, d, s, x);
  check();
}

void unscale_f32(int cells, const float* y, const float* d, float s,
                 PrecReal* x) {
  if (cells <= 0) return;
  k_widen32<<<blocks(cells), kThreads>>>(cells, y, d, s, x);
  check();
}

void stage2_residual(int n, const Real* r, Real* Az) {
  if (n <= 0) return;
  k_s2_res<<<blocks(n), kThreads>>>(n, r, Az);
  check();
}

void stage2_update(int n, const Real* y, Real* Az, Real* z) {
  if (n <= 0) return;
  k_s2_upd<<<blocks(n), kThreads>>>(n, y, Az, z);
  check();
}

}  // namespace gk
}  // namespace nvrs
