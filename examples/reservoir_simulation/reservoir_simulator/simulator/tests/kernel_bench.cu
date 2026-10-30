// SPDX-License-Identifier: Apache-2.0
// kernel_bench.cu -- the operator apply in isolation, on a fixed input, with
// repeats: every format the paper discusses, against both vendor baselines,
// next to the bandwidth the device actually achieves.
//
//   nvcc -O3 -std=c++17 -I../include -o kernel_bench kernel_bench.cu -lcusparse
//   ./kernel_bench --op OUTPUT_NORNE/op_J.bin --json kernels.json
//   ./kernel_bench 46 112 22 3 --json box.json        # synthetic 7-point box
//
// Options: --trials N (default 15)  --reps N applies per trial (default 200)
//
// What it answers (reviewer comments in brackets):
//   * df64 against native FP64 against FP32-stored values in the operator
//     apply, measured on a fixed input with n >= 15 trials, median and IQR
//     [R14], on the real Norne operator written by DUMPOP [R15];
//   * the block kernel against cuSPARSE CSR *and* cuSPARSE BSR, the honest
//     vendor baseline for a 3x3-block Jacobian [R18];
//   * lanes per row 1, 2, 4, 8 and a component-major (block-transposed) layout
//     in which consecutive lanes read consecutive values [R16];
//   * the achieved bandwidth of a STREAM triad on the same device, so the
//     roofline and the distance to it use a measured number, not the peak
//     [R13];
//   * the FP64 : FP32 FMA throughput ratio of the device, measured, so the
//     headroom the emulated path covers is a number [R14].
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "block_spmv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} } while (0)

using namespace nvrs;

// --- df64 helpers and the simulator's scalar df64 kernel -------------------
__device__ __forceinline__ float2 k_qsum(float a, float b) {
  float s = a + b; float e = b - (s - a); return make_float2(s, e);
}
__device__ __forceinline__ float2 k_sum(float a, float b) {
  float s = a + b, bb = s - a; float e = (a - (s - bb)) + (b - bb);
  return make_float2(s, e);
}
__device__ __forceinline__ float2 k_prod(float a, float b) {
  float p = a * b; float e = __fmaf_rn(a, b, -p); return make_float2(p, e);
}
__device__ __forceinline__ float2 k_add(float2 a, float2 b) {
  float2 s = k_sum(a.x, b.x), t = k_sum(a.y, b.y);
  s.y += t.x; s = k_qsum(s.x, s.y); s.y += t.y; s = k_qsum(s.x, s.y); return s;
}
__device__ __forceinline__ float2 k_mul(float2 a, float2 b) {
  float2 p = k_prod(a.x, b.x); p.y += a.x * b.y + a.y * b.x;
  p = k_qsum(p.x, p.y); return p;
}
__global__ void k_csr_df64(const int* rp, const int* ci, const float* ahi,
                           const float* alo, const double* x, double* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float2 acc = make_float2(0.f, 0.f);
  for (int p = rp[i]; p < rp[i + 1]; ++p) {
    const double xv = x[ci[p]]; const float xh = (float)xv;
    acc = k_add(acc, k_mul(make_float2(ahi[p], alo[p]),
                           make_float2(xh, (float)(xv - (double)xh))));
  }
  y[i] = (double)acc.x + (double)acc.y;
}

// Lanes-per-row variants of the block FP64 kernel, simulator layout.
template <int BS, int L>
__global__ void k_blk_f64_L(const int* __restrict__ prp, const int* __restrict__ bcol,
                            const double* __restrict__ a, const double* __restrict__ x,
                            double* __restrict__ y, int n) {
  const long gt = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int r = (int)(gt / L), lane = (int)(gt % L);
  const bool live = r < n;
  const int c = live ? r / BS : 0, e = live ? r - c * BS : 0;
  const int p0 = prp[c], nc = live ? prp[c + 1] - p0 : 0;
  const int base = (BS * p0 + e * nc) * BS;
  double acc = 0.0;
  for (int m = lane; m < nc; m += L) {
    const int col = BS * bcol[p0 + m], q = base + m * BS;
#pragma unroll
    for (int v = 0; v < BS; ++v) acc = fma(a[q + v], x[col + v], acc);
  }
#pragma unroll
  for (int o = L / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o, L);
  if (live && lane == 0) y[r] = acc;
}

// Component-major (block-transposed) layout: within a scalar row the values
// are stored [v][m], so for each component v consecutive lanes read
// consecutive values -- one coalesced access per instruction.
template <int BS, int L>
__global__ void k_blk_f64_cm(const int* __restrict__ prp, const int* __restrict__ bcol,
                             const double* __restrict__ a, const double* __restrict__ x,
                             double* __restrict__ y, int n) {
  const long gt = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int r = (int)(gt / L), lane = (int)(gt % L);
  const bool live = r < n;
  const int c = live ? r / BS : 0, e = live ? r - c * BS : 0;
  const int p0 = prp[c], nc = live ? prp[c + 1] - p0 : 0;
  const int base = (BS * p0 + e * nc) * BS;
  double acc = 0.0;
  for (int m = lane; m < nc; m += L) {
    const int col = BS * bcol[p0 + m];
#pragma unroll
    for (int v = 0; v < BS; ++v) acc = fma(a[base + v * nc + m], x[col + v], acc);
  }
#pragma unroll
  for (int o = L / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o, L);
  if (live && lane == 0) y[r] = acc;
}

// --- bandwidth and arithmetic probes ----------------------------------------
__global__ void k_triad(double* a, const double* b, const double* c, double s, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    a[i] = b[i] + s * c[i];
}
template <class T>
__global__ void k_fma(T* out, T s, int iters) {
  T a0 = (T)threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3,
    a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  for (int i = 0; i < iters; ++i) {
    a0 = fma(a0, s, s); a1 = fma(a1, s, s); a2 = fma(a2, s, s); a3 = fma(a3, s, s);
    a4 = fma(a4, s, s); a5 = fma(a5, s, s); a6 = fma(a6, s, s); a7 = fma(a7, s, s);
  }
  if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == (T)-1) out[0] = a0;
}

// --- timing ------------------------------------------------------------------
struct Stat { double med = 0, lo = 0, hi = 0, q1 = 0, q3 = 0; };
Stat stat(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  Stat s; const size_t n = v.size();
  if (!n) return s;
  auto q = [&](double f) { return v[(size_t)std::min<double>(n - 1, std::floor(f * (n - 1) + 0.5))]; };
  s.med = q(0.5); s.lo = v.front(); s.hi = v.back(); s.q1 = q(0.25); s.q3 = q(0.75);
  return s;
}
template <class F> Stat time_it(F f, int trials, int reps) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int w = 0; w < 3; ++w) f();                    // warm up
  CK(cudaDeviceSynchronize());
  std::vector<double> per;
  for (int t = 0; t < trials; ++t) {
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    per.push_back(1e3 * ms / reps);                   // microseconds per apply
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  return stat(per);
}

int main(int argc, char** argv) {
  std::string op_path, json_path;
  int trials = 15, reps = 200, nx = 46, ny = 112, nz = 22, bs = 3;
  std::vector<int> pos;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--op" && i + 1 < argc) op_path = argv[++i];
    else if (s == "--json" && i + 1 < argc) json_path = argv[++i];
    else if (s == "--trials" && i + 1 < argc) trials = std::max(3, std::atoi(argv[++i]));
    else if (s == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    else pos.push_back(std::atoi(argv[i]));
  }
  if (pos.size() >= 3) { nx = pos[0]; ny = pos[1]; nz = pos[2]; }
  if (pos.size() >= 4) bs = pos[3];

  // --- the operator: DUMPOP's op_J.bin, or the synthetic box ------------------
  int n = 0, cells = 0;
  std::vector<int> rp, ci;
  std::vector<double> av;
  std::string source;
  if (!op_path.empty()) {
    std::ifstream f(op_path, std::ios::binary);
    if (!f) { std::printf("cannot open %s\n", op_path.c_str()); return 2; }
    int32_t h[5]; f.read((char*)h, sizeof(h));
    n = h[0]; const int nnz = h[2]; bs = h[3]; cells = h[4];
    rp.resize(n + 1); ci.resize(nnz); av.resize(nnz);
    f.read((char*)rp.data(), 4 * (n + 1)); f.read((char*)ci.data(), 4 * (size_t)nnz);
    f.read((char*)av.data(), 8 * (size_t)nnz);
    if (!f) { std::printf("%s is truncated\n", op_path.c_str()); return 2; }
    if (h[1] != n) std::printf("[note] n_ext %d != n %d; ghost columns are "
                               "clamped (one-rank dumps have none)\n", h[1], n);
    for (auto& c : ci) c = std::min(c, n - 1);
    source = op_path;
  } else {
    cells = nx * ny * nz; n = bs * cells;
    auto lid = [&](int i, int j, int k) { return i + nx * (j + ny * k); };
    rp.assign(n + 1, 0);
    for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
      const int c = lid(i, j, k);
      std::vector<int> mem = {c};
      if (i > 0) mem.push_back(lid(i - 1, j, k)); if (i < nx - 1) mem.push_back(lid(i + 1, j, k));
      if (j > 0) mem.push_back(lid(i, j - 1, k)); if (j < ny - 1) mem.push_back(lid(i, j + 1, k));
      if (k > 0) mem.push_back(lid(i, j, k - 1)); if (k < nz - 1) mem.push_back(lid(i, j, k + 1));
      std::sort(mem.begin(), mem.end());
      for (int a = 0; a < bs; ++a) {
        rp[bs * c + a + 1] = (int)mem.size() * bs;
        for (int m : mem) for (int v = 0; v < bs; ++v) ci.push_back(bs * m + v);
      }
    }
    for (int r = 0; r < n; ++r) rp[r + 1] += rp[r];
    std::mt19937_64 rng(7); std::uniform_real_distribution<double> U(-1.0, 1.0);
    av.resize(ci.size());
    for (auto& v : av) v = U(rng) * std::pow(10.0, 3.0 * U(rng));
    char b[96]; std::snprintf(b, sizeof b, "synthetic %dx%dx%d box, 7-point", nx, ny, nz);
    source = b;
  }
  const int nnz = rp[n];

  // Block structure: cell c's scalar rows list nc blocks of bs contiguous
  // columns, in the same order for every row of the cell.
  std::vector<int> prp(cells + 1, 0), bcol;
  bool block_ok = (n == bs * cells);
  for (int c = 0; c < cells && block_ok; ++c) {
    const int len = rp[bs * c + 1] - rp[bs * c];
    if (len % bs) { block_ok = false; break; }
    const int nc = len / bs;
    for (int a = 0; a < bs && block_ok; ++a)
      if (rp[bs * c + a + 1] - rp[bs * c + a] != len) block_ok = false;
    for (int m = 0; m < nc && block_ok; ++m) {
      const int col0 = ci[rp[bs * c] + m * bs];
      if (col0 % bs) { block_ok = false; break; }
      for (int a = 0; a < bs; ++a)
        for (int v = 0; v < bs; ++v)
          if (ci[rp[bs * c + a] + m * bs + v] != col0 + v) block_ok = false;
      bcol.push_back(col0 / bs);
    }
    prp[c + 1] = prp[c] + nc;
  }
  if (!block_ok || bs != 3) {
    std::printf("the operator is not in the bs=3 block layout build_sparsity "
                "writes; only the scalar kernels would apply\n");
    return 3;
  }
  const int nnzb = prp[cells];
  std::printf("operator: %s\n  %d cells, %d rows, %d nonzeros, %d blocks, "
              "%.2f blocks per block row\n", source.c_str(), cells, n, nnz, nnzb,
              (double)nnzb / cells);

  // Value arrays: simulator layout (= CSR order), df64 slices, FP32,
  // cuSPARSE BSR row-major, and component-major.
  std::vector<float> hi(nnz), lo(nnz);
  for (int p = 0; p < nnz; ++p) { hi[p] = (float)av[p]; lo[p] = (float)(av[p] - (double)hi[p]); }
  std::vector<double> bsr((size_t)nnzb * 9), cm(nnz);
  for (int c = 0; c < cells; ++c) {
    const int p0 = prp[c], nc = prp[c + 1] - p0;
    for (int a = 0; a < 3; ++a)
      for (int m = 0; m < nc; ++m)
        for (int v = 0; v < 3; ++v) {
          const double val = av[(size_t)(3 * p0 + a * nc) * 3 + m * 3 + v];
          bsr[(size_t)(p0 + m) * 9 + a * 3 + v] = val;
          cm[(size_t)(3 * p0 + a * nc) * 3 + v * nc + m] = val;
        }
  }
  std::mt19937_64 rng(11); std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::vector<double> x(n); for (auto& v : x) v = U(rng) * 1e3;

  int *d_rp, *d_ci, *d_prp, *d_bcol;
  double *d_a, *d_bsr, *d_cm, *d_x, *d_y, *d_ref;
  float *d_hi, *d_lo;
  CK(cudaMalloc(&d_rp, 4 * (n + 1))); CK(cudaMalloc(&d_ci, 4 * (size_t)nnz));
  CK(cudaMalloc(&d_prp, 4 * (cells + 1))); CK(cudaMalloc(&d_bcol, 4 * (size_t)nnzb));
  CK(cudaMalloc(&d_a, 8 * (size_t)nnz)); CK(cudaMalloc(&d_bsr, 8 * bsr.size()));
  CK(cudaMalloc(&d_cm, 8 * (size_t)nnz));
  CK(cudaMalloc(&d_hi, 4 * (size_t)nnz)); CK(cudaMalloc(&d_lo, 4 * (size_t)nnz));
  CK(cudaMalloc(&d_x, 8 * n)); CK(cudaMalloc(&d_y, 8 * n)); CK(cudaMalloc(&d_ref, 8 * n));
  CK(cudaMemcpy(d_rp, rp.data(), 4 * (n + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_ci, ci.data(), 4 * (size_t)nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_prp, prp.data(), 4 * (cells + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bcol, bcol.data(), 4 * (size_t)nnzb, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_a, av.data(), 8 * (size_t)nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bsr, bsr.data(), 8 * bsr.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_cm, cm.data(), 8 * (size_t)nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_hi, hi.data(), 4 * (size_t)nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lo, lo.data(), 4 * (size_t)nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_x, x.data(), 8 * n, cudaMemcpyHostToDevice));

  // --- vendor baselines ----------------------------------------------------------
  cusparseHandle_t sp; cusparseCreate(&sp);
  cusparseSpMatDescr_t A; cusparseDnVecDescr_t vx, vy;
  cusparseCreateCsr(&A, n, n, nnz, d_rp, d_ci, d_a, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateDnVec(&vx, n, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&vy, n, d_y, CUDA_R_64F);
  const double one = 1.0, zero = 0.0;
  size_t bsz = 0; void* buf = nullptr;
  cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vx, &zero, vy,
                          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bsz);
  if (bsz) CK(cudaMalloc(&buf, bsz));
  auto cusp_csr = [&] {
    cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vx, &zero, vy,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf);
  };
  // Reference result: cuSPARSE CSR in FP64.
  {
    cusparseDnVecDescr_t vr; cusparseCreateDnVec(&vr, n, d_ref, CUDA_R_64F);
    cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vx, &zero, vr,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf);
    CK(cudaDeviceSynchronize());
    cusparseDestroyDnVec(vr);
  }
  // cuSPARSE BSR, 3x3 blocks, row-major within a block.
  bool have_bsr = false;
#if CUSPARSE_VER_MAJOR < 13
  cusparseMatDescr_t descr; cusparseCreateMatDescr(&descr);
  auto cusp_bsr = [&] {
    cusparseDbsrmv(sp, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   cells, cells, nnzb, &one, descr, d_bsr, d_prp, d_bcol, 3,
                   d_x, &zero, d_y);
  };
  have_bsr = true;
#else
  cusparseSpMatDescr_t Ab = nullptr; size_t bbsz = 0; void* bbuf = nullptr;
  have_bsr = cusparseCreateBsr(&Ab, cells, cells, nnzb, 3, 3, d_prp, d_bcol, d_bsr,
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F,
                               CUSPARSE_ORDER_ROW) == CUSPARSE_STATUS_SUCCESS &&
             cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, Ab,
                                     vx, &zero, vy, CUDA_R_64F,
                                     CUSPARSE_SPMV_ALG_DEFAULT,
                                     &bbsz) == CUSPARSE_STATUS_SUCCESS;
  if (have_bsr && bbsz) CK(cudaMalloc(&bbuf, bbsz));
  auto cusp_bsr = [&] {
    cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, Ab, vx, &zero, vy,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, bbuf);
  };
#endif

  // --- the variants ------------------------------------------------------------------
  const int T = 256;
  auto grid_for = [&](int L) { return (unsigned)(((long)n * L + T - 1) / T); };
  struct Variant { const char* name; const char* fmt; double bytes; Stat s; double err; };
  std::vector<Variant> out;
  const double xy = 8.0 * n * 2;                       // x read once, y written
  const double idx_scalar = 4.0 * nnz + 4.0 * (n + 1), idx_block = 4.0 * nnzb + 4.0 * (cells + 1);
  auto check = [&]() {
    std::vector<double> y(n), r(n);
    CK(cudaMemcpy(y.data(), d_y, 8 * n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r.data(), d_ref, 8 * n, cudaMemcpyDeviceToHost));
    double d = 0, m = 0;
    for (int i = 0; i < n; ++i) { d = std::max(d, std::fabs(y[i] - r[i])); m = std::max(m, std::fabs(r[i])); }
    return m > 0 ? d / m : d;
  };
  auto run = [&](const char* name, const char* fmt, double bytes, auto f) {
    CK(cudaMemset(d_y, 0, 8 * n)); f(); CK(cudaDeviceSynchronize());
    const double err = check();
    const Stat s = time_it(f, trials, reps);
    out.push_back({name, fmt, bytes, s, err});
  };
  run("cusparse_csr_fp64", "FP64", 8.0 * nnz + idx_scalar + xy, cusp_csr);
  if (have_bsr) run("cusparse_bsr_fp64", "FP64", 8.0 * nnz + idx_block + xy, cusp_bsr);
  run("scalar_csr_df64", "df64", 8.0 * nnz + idx_scalar + xy, [&] {
    k_csr_df64<<<(n + T - 1) / T, T>>>(d_rp, d_ci, d_hi, d_lo, d_x, d_y, n); });
  run("block_df64", "df64", 8.0 * nnz + idx_block + xy, [&] {
    bspmv::launch_df64(d_prp, d_bcol, d_hi, d_lo, d_x, d_y, n, 3, 0); });
  run("block_fp64", "FP64", 8.0 * nnz + idx_block + xy, [&] {
    bspmv::launch_f64(d_prp, d_bcol, d_a, d_x, d_y, n, 3, 0); });
  run("block_fp32_stored", "FP32", 4.0 * nnz + idx_block + xy, [&] {
    bspmv::launch_f32(d_prp, d_bcol, d_hi, d_x, d_y, n, 3, 0); });
  run("block_fp64_L1", "FP64", 8.0 * nnz + idx_block + xy, [&] {
    k_blk_f64_L<3, 1><<<grid_for(1), T>>>(d_prp, d_bcol, d_a, d_x, d_y, n); });
  run("block_fp64_L2", "FP64", 8.0 * nnz + idx_block + xy, [&] {
    k_blk_f64_L<3, 2><<<grid_for(2), T>>>(d_prp, d_bcol, d_a, d_x, d_y, n); });
  run("block_fp64_L8", "FP64", 8.0 * nnz + idx_block + xy, [&] {
    k_blk_f64_L<3, 8><<<grid_for(8), T>>>(d_prp, d_bcol, d_a, d_x, d_y, n); });
  run("block_fp64_component_major", "FP64", 8.0 * nnz + idx_block + xy, [&] {
    k_blk_f64_cm<3, 4><<<grid_for(4), T>>>(d_prp, d_bcol, d_cm, d_x, d_y, n); });

  // --- achieved bandwidth: STREAM triad on 3 x 64 M doubles --------------------
  const size_t N = (size_t)64 << 20;
  double *ta, *tb, *tc;
  CK(cudaMalloc(&ta, 8 * N)); CK(cudaMalloc(&tb, 8 * N)); CK(cudaMalloc(&tc, 8 * N));
  CK(cudaMemset(tb, 0, 8 * N)); CK(cudaMemset(tc, 0, 8 * N));
  int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const Stat tri = time_it([&] { k_triad<<<sms * 8, 512>>>(ta, tb, tc, 1.5, N); },
                           trials, 10);
  const double triad_gbs = 3.0 * 8.0 * N / (tri.med * 1e-6) / 1e9;

  // --- FP64 : FP32 FMA throughput ----------------------------------------------
  float* of; double* od; CK(cudaMalloc(&of, 8)); CK(cudaMalloc(&od, 8));
  const int it = 4096, blocks = sms * 16;
  const Stat f32 = time_it([&] { k_fma<float><<<blocks, 256>>>(of, 0.999f, it); }, trials, 3);
  const Stat f64 = time_it([&] { k_fma<double><<<blocks, 256>>>(od, 0.999, it); }, trials, 3);
  const double flops = 2.0 * 8 * it * (double)blocks * 256;
  const double tf32 = flops / (f32.med * 1e-6) / 1e12, tf64 = flops / (f64.med * 1e-6) / 1e12;

  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, sms);
  std::printf("achieved bandwidth (STREAM triad, median of %d): %.1f GB/s\n", trials, triad_gbs);
  std::printf("FMA throughput: FP32 %.1f TFLOP/s, FP64 %.1f TFLOP/s, ratio FP32:FP64 = %.1f : 1\n",
              tf32, tf64, tf32 / tf64);
  std::printf("\n%-28s %-5s %10s %10s %10s %9s %9s %10s\n", "kernel", "fmt", "median_us",
              "q1_us", "q3_us", "GB/s", "%triad", "rel_err");
  for (const auto& v : out) {
    const double gbs = v.bytes / (v.s.med * 1e-6) / 1e9;
    std::printf("%-28s %-5s %10.2f %10.2f %10.2f %9.1f %8.1f%% %10.2e\n", v.name, v.fmt,
                v.s.med, v.s.q1, v.s.q3, gbs, 100.0 * gbs / triad_gbs, v.err);
  }
  auto find = [&](const char* nm) -> const Variant* {
    for (const auto& v : out) if (!std::strcmp(v.name, nm)) return &v;
    return nullptr;
  };
  const Variant *bf = find("block_fp64"), *bd = find("block_df64"), *b32 = find("block_fp32_stored"),
                *cc = find("cusparse_csr_fp64"), *cb = find("cusparse_bsr_fp64");
  if (bf && bd) std::printf("\ndf64 / FP64 per apply (block): %.3f\n", bd->s.med / bf->s.med);
  if (bf && b32) std::printf("FP32-stored / FP64 per apply (block): %.3f\n", b32->s.med / bf->s.med);
  if (bf && cc) std::printf("block FP64 speed-up over cuSPARSE CSR: %.2fx\n", cc->s.med / bf->s.med);
  if (bf && cb) std::printf("block FP64 speed-up over cuSPARSE BSR: %.2fx\n", cb->s.med / bf->s.med);

  if (!json_path.empty()) {
    std::ofstream j(json_path);
    j << "{\n  \"device\": \"" << prop.name << "\", \"sm\": \"" << prop.major << prop.minor
      << "\", \"sms\": " << sms << ",\n";
    j << "  \"operator\": \"" << source << "\", \"cells\": " << cells << ", \"rows\": " << n
      << ", \"nnz\": " << nnz << ", \"blocks\": " << nnzb << ",\n";
    j << "  \"trials\": " << trials << ", \"reps\": " << reps << ",\n";
    j << "  \"triad_gbs\": " << triad_gbs << ",\n";
    j << "  \"fma_tflops_fp32\": " << tf32 << ", \"fma_tflops_fp64\": " << tf64 << ",\n";
    j << "  \"kernels\": [\n";
    for (size_t i = 0; i < out.size(); ++i) {
      const auto& v = out[i];
      j << "    {\"name\": \"" << v.name << "\", \"format\": \"" << v.fmt
        << "\", \"median_us\": " << v.s.med << ", \"q1_us\": " << v.s.q1
        << ", \"q3_us\": " << v.s.q3 << ", \"min_us\": " << v.s.lo << ", \"max_us\": " << v.s.hi
        << ", \"bytes\": " << v.bytes << ", \"rel_err_vs_cusparse_fp64\": " << v.err << "}"
        << (i + 1 < out.size() ? ",\n" : "\n");
    }
    j << "  ]\n}\n";
    std::printf("\nwrote %s\n", json_path.c_str());
  }
  return 0;
}
