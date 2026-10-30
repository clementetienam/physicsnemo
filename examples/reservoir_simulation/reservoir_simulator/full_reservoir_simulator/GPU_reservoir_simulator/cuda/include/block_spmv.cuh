// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// block_spmv.cuh -- the coupled operator applied block by block.
//
// The coupled Jacobian is stored as scalar CSR, one column index per scalar
// entry.  Both sparsity builders (build_sparsity and build_sparsity_field)
// emit it with a fixed block layout, which is what makes this file possible:
//
//   cell c owns rows bs*c .. bs*c+bs-1, each with the same ncols member cells
//   (ncols = prowptr[c+1] - prowptr[c]), and entry v of member m in row a sits
//   at   rowptr[bs*c] + (a*ncols + m)*bs + v,   with column  bs*member + v,
//   so   rowptr[bs*c] == bs*bs*prowptr[c].
//
// A block kernel therefore needs one cell-level column index per bs x bs block
// (bcol, pnnz entries, local cell ids) and no scalar rowptr or colind at all.
// Per nonzero that is 4/bs^2 bytes of index traffic instead of 4, and each
// thread reads its bs operand values as one contiguous run.
//
// k_bspmv_check verifies the layout on the device once per sparsity pattern;
// if any entry disagrees the solver keeps the scalar kernels.
#pragma once

#include <cuda_runtime.h>

namespace nvrs {
namespace bspmv {

// Local block column of every member of every cell, from the scalar CSR.
__global__ static void k_bcol_build(const int* __restrict__ prowptr,
                                    const int* __restrict__ rowptr,
                                    const int* __restrict__ colind,
                                    int* __restrict__ bcol, int cells, int bs) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const int p0 = prowptr[c], nc = prowptr[c + 1] - p0;
  const int base = rowptr[bs * c];
  for (int m = 0; m < nc; ++m) bcol[p0 + m] = colind[base + m * bs] / bs;
}

// Count entries that break the block layout; zero means the block kernels
// reproduce the scalar ones exactly.
__global__ static void k_bspmv_check(const int* __restrict__ prowptr,
                                     const int* __restrict__ rowptr,
                                     const int* __restrict__ colind,
                                     const int* __restrict__ bcol, int cells,
                                     int bs, int* __restrict__ bad) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const int p0 = prowptr[c], nc = prowptr[c + 1] - p0;
  int nbad = 0;
  if (rowptr[bs * c] != bs * bs * p0) ++nbad;
  for (int a = 0; a < bs; ++a) {
    const int r = bs * c + a;
    if (rowptr[r + 1] - rowptr[r] != nc * bs) { ++nbad; continue; }
    for (int m = 0; m < nc; ++m)
      for (int v = 0; v < bs; ++v)
        if (colind[rowptr[r] + m * bs + v] != bs * bcol[p0 + m] + v) ++nbad;
  }
  if (nbad) atomicAdd(bad, nbad);
}

__device__ __forceinline__ float2 bdf_sum(float a, float b) {
  float s = a + b, bb = s - a, e = (a - (s - bb)) + (b - bb);
  return make_float2(s, e);
}
__device__ __forceinline__ float2 bdf_qsum(float a, float b) {
  float s = a + b; return make_float2(s, b - (s - a));
}
__device__ __forceinline__ float2 bdf_prod(float a, float b) {
  float p = a * b; float e = __fmaf_rn(a, b, -p); return make_float2(p, e);
}
__device__ __forceinline__ float2 bdf_add(float2 a, float2 b) {
  float2 s = bdf_sum(a.x, b.x), t = bdf_sum(a.y, b.y);
  s.y += t.x; s = bdf_qsum(s.x, s.y); s.y += t.y; s = bdf_qsum(s.x, s.y); return s;
}
__device__ __forceinline__ float2 bdf_mul(float2 a, float2 b) {
  float2 p = bdf_prod(a.x, b.x); p.y += a.x * b.y + a.y * b.x;
  p = bdf_qsum(p.x, p.y); return p;
}

// Lanes per scalar row.  Four lanes share a row's member loop and a shuffle
// tree combines them.  Lane L reads the bs values of member L, L+4, ...: each
// lane's own values are contiguous and neighbouring lanes are bs values apart,
// so for bs = 3 one load instruction is strided, but the three unrolled loads
// of a four-lane group touch 12 contiguous values, which the L1/L2 sectors
// serve in full.  The saving comes from the sector reuse, not from a
// per-instruction coalesced access; a component-major layout would coalesce
// per instruction, and tests/bspmv_bench.cu measures it.  Note also that df64
// reads two value streams (hi, lo) where FP64 reads one: the same bytes, but
// twice the load requests.  Measured on the full 46 x 112 x 22 box with a
// seven-point pattern (340,032 rows, 7.0 M nonzeros -- the Norne bounding box,
// not Norne's 44,431 active cells): the df64 apply falls from 1.93 ms (scalar
// CSR, one thread per row) to 0.56 ms, and the FP64 apply from 0.49 ms
// (cuSPARSE CSR) to 0.28 ms; tests/bspmv_bench.cu reproduces both, adds
// cuSPARSE BSR, and runs on a real operator written by DUMPOP.  The combination order differs from
// the scalar loop, so results agree to rounding, not bitwise; the order is
// fixed, so a run is still deterministic.
constexpr int kLanes = 4;

// y = A x in df64.  BS > 0 fixes the block size at compile time; BS == 0 reads
// it at run time (at most 8).
template <int BS>
__global__ static void k_bspmv_df64(const int* __restrict__ prowptr,
                                    const int* __restrict__ bcol,
                                    const float* __restrict__ ahi,
                                    const float* __restrict__ alo,
                                    const double* __restrict__ x,
                                    double* __restrict__ y, int n, int bs_rt) {
  const int bs = BS > 0 ? BS : bs_rt;
  const long gt = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int r = (int)(gt / kLanes), lane = (int)(gt % kLanes);
  // No early return: the shuffles below name every lane of the warp, so a row
  // past the end runs an empty member loop instead of leaving.
  const bool live = r < n;
  const int c = live ? r / bs : 0, a = live ? r - c * bs : 0;
  const int p0 = prowptr[c], nc = live ? prowptr[c + 1] - p0 : 0;
  const int base = (bs * p0 + a * nc) * bs;
  float2 acc = make_float2(0.f, 0.f);
  for (int m = lane; m < nc; m += kLanes) {
    const int col = bs * bcol[p0 + m];
    const int q = base + m * bs;
#pragma unroll
    for (int v = 0; v < (BS > 0 ? BS : 8); ++v) {
      if (BS == 0 && v >= bs) break;
      const float2 av = make_float2(ahi[q + v], alo[q + v]);
      const double xv = x[col + v];
      const float xh = (float)xv;
      acc = bdf_add(acc, bdf_mul(av, make_float2(xh, (float)(xv - (double)xh))));
    }
  }
#pragma unroll
  for (int o = kLanes / 2; o > 0; o >>= 1) {
    const float2 t = make_float2(__shfl_down_sync(0xffffffffu, acc.x, o, kLanes),
                                 __shfl_down_sync(0xffffffffu, acc.y, o, kLanes));
    acc = bdf_add(acc, t);
  }
  if (live && lane == 0) y[r] = (double)acc.x + (double)acc.y;
}

// y = A x in native FP64, same indexing.
template <int BS>
__global__ static void k_bspmv_f64(const int* __restrict__ prowptr,
                                   const int* __restrict__ bcol,
                                   const double* __restrict__ aval,
                                   const double* __restrict__ x,
                                   double* __restrict__ y, int n, int bs_rt) {
  const int bs = BS > 0 ? BS : bs_rt;
  const long gt = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int r = (int)(gt / kLanes), lane = (int)(gt % kLanes);
  const bool live = r < n;
  const int c = live ? r / bs : 0, a = live ? r - c * bs : 0;
  const int p0 = prowptr[c], nc = live ? prowptr[c + 1] - p0 : 0;
  const int base = (bs * p0 + a * nc) * bs;
  double acc = 0.0;
  for (int m = lane; m < nc; m += kLanes) {
    const int col = bs * bcol[p0 + m];
    const int q = base + m * bs;
#pragma unroll
    for (int v = 0; v < (BS > 0 ? BS : 8); ++v) {
      if (BS == 0 && v >= bs) break;
      acc = fma(aval[q + v], x[col + v], acc);
    }
  }
#pragma unroll
  for (int o = kLanes / 2; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o, kLanes);
  if (live && lane == 0) y[r] = acc;
}

// y = A x with FP32-stored values (EMULATE 2): the hi slice of the df64 pair
// alone, promoted to FP64 before the multiply, and the sum kept in FP64.  Half
// the value bytes of the FP64 and df64 paths, which is the whole point: by the
// traffic model this is the one operator format that moves fewer bytes.
template <int BS>
__global__ static void k_bspmv_f32(const int* __restrict__ prowptr,
                                   const int* __restrict__ bcol,
                                   const float* __restrict__ ahi,
                                   const double* __restrict__ x,
                                   double* __restrict__ y, int n, int bs_rt) {
  const int bs = BS > 0 ? BS : bs_rt;
  const long gt = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int r = (int)(gt / kLanes), lane = (int)(gt % kLanes);
  const bool live = r < n;
  const int c = live ? r / bs : 0, a = live ? r - c * bs : 0;
  const int p0 = prowptr[c], nc = live ? prowptr[c + 1] - p0 : 0;
  const int base = (bs * p0 + a * nc) * bs;
  double acc = 0.0;
  for (int m = lane; m < nc; m += kLanes) {
    const int col = bs * bcol[p0 + m];
    const int q = base + m * bs;
#pragma unroll
    for (int v = 0; v < (BS > 0 ? BS : 8); ++v) {
      if (BS == 0 && v >= bs) break;
      acc = fma((double)ahi[q + v], x[col + v], acc);
    }
  }
#pragma unroll
  for (int o = kLanes / 2; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o, kLanes);
  if (live && lane == 0) y[r] = acc;
}

inline bool supported(int bs) { return bs >= 2 && bs <= 8; }

inline unsigned blocks_for(int n) {
  return (unsigned)(((long)n * kLanes + 255) / 256);
}

inline void launch_df64(const int* prowptr, const int* bcol, const float* ahi,
                        const float* alo, const double* x, double* y, int n,
                        int bs, cudaStream_t st) {
  const unsigned nb = blocks_for(n);
  if (bs == 3)      k_bspmv_df64<3><<<nb, 256, 0, st>>>(prowptr, bcol, ahi, alo, x, y, n, bs);
  else if (bs == 2) k_bspmv_df64<2><<<nb, 256, 0, st>>>(prowptr, bcol, ahi, alo, x, y, n, bs);
  else              k_bspmv_df64<0><<<nb, 256, 0, st>>>(prowptr, bcol, ahi, alo, x, y, n, bs);
}

inline void launch_f64(const int* prowptr, const int* bcol, const double* aval,
                       const double* x, double* y, int n, int bs,
                       cudaStream_t st) {
  const unsigned nb = blocks_for(n);
  if (bs == 3)      k_bspmv_f64<3><<<nb, 256, 0, st>>>(prowptr, bcol, aval, x, y, n, bs);
  else if (bs == 2) k_bspmv_f64<2><<<nb, 256, 0, st>>>(prowptr, bcol, aval, x, y, n, bs);
  else              k_bspmv_f64<0><<<nb, 256, 0, st>>>(prowptr, bcol, aval, x, y, n, bs);
}

inline void launch_f32(const int* prowptr, const int* bcol, const float* ahi,
                       const double* x, double* y, int n, int bs,
                       cudaStream_t st) {
  const unsigned nb = blocks_for(n);
  if (bs == 3)      k_bspmv_f32<3><<<nb, 256, 0, st>>>(prowptr, bcol, ahi, x, y, n, bs);
  else if (bs == 2) k_bspmv_f32<2><<<nb, 256, 0, st>>>(prowptr, bcol, ahi, x, y, n, bs);
  else              k_bspmv_f32<0><<<nb, 256, 0, st>>>(prowptr, bcol, ahi, x, y, n, bs);
}

}  // namespace bspmv
}  // namespace nvrs
