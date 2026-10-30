// SPDX-License-Identifier: Apache-2.0
// bspmv_bench.cu -- block operator apply and CUDA graphs, checked and timed in
// isolation, on the CSR layout build_sparsity() emits.
//
//   nvcc -O3 -std=c++17 -I../include -o bspmv_bench bspmv_bench.cu -lcusparse -lcublas
//   ./bspmv_bench [nx ny nz] [bs]
//
// 1. Correctness: the block df64 kernel must agree with the scalar df64 kernel,
//    and the block FP64 kernel with cuSPARSE, to rounding.
// 2. Speed: scalar df64, block df64, cuSPARSE FP64 and block FP64.
// 3. Launch overhead: the post-AMG tail of one Arnoldi step (a dozen small
//    kernels and four cuBLAS gemv) launched kernel by kernel against one
//    CUDA-graph replay, and three host synchronisations per step against one.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "block_spmv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} } while (0)

using namespace nvrs;

// --- the simulator's scalar df64 kernel, verbatim --------------------------
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
__global__ void k_spmv_df64(const int* rp, const int* ci, const float* ahi,
                            const float* alo, const double* x, double* y, int n) {
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

// Small kernels standing in for the stage-2 / well / scaling work of one step.
__global__ void k_axpy(double* y, double a, const double* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += a * x[i];
}
__global__ void k_stage2(const double* r, double* az, double* z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) z[i] += 0.5 * (r[i] - az[i]);
}

// Sub-warp variant: L lanes per row share the member loop, so neighbouring
// lanes read neighbouring value runs (coalesced), then reduce by shuffles.
template <int BS, int L>
__global__ void k_bspmv_f64_sw(const int* __restrict__ prowptr, const int* __restrict__ bcol,
                               const double* __restrict__ aval, const double* __restrict__ x,
                               double* __restrict__ y, int n) {
  const int gt = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = gt / L, lane = gt % L;
  if (r >= n) return;
  const int c = r / BS, a = r - c * BS;
  const int p0 = prowptr[c], nc = prowptr[c + 1] - p0;
  const int base = (BS * p0 + a * nc) * BS;
  double acc = 0.0;
  for (int m = lane; m < nc; m += L) {
    const int col = BS * bcol[p0 + m];
    const int q = base + m * BS;
#pragma unroll
    for (int v = 0; v < BS; ++v) acc = fma(aval[q + v], x[col + v], acc);
  }
#pragma unroll
  for (int o = L / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o, L);
  if (lane == 0) y[r] = acc;
}
template <int BS, int L>
__global__ void k_bspmv_df64_sw(const int* __restrict__ prowptr, const int* __restrict__ bcol,
                                const float* __restrict__ ahi, const float* __restrict__ alo,
                                const double* __restrict__ x, double* __restrict__ y, int n) {
  const int gt = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = gt / L, lane = gt % L;
  if (r >= n) return;
  const int c = r / BS, a = r - c * BS;
  const int p0 = prowptr[c], nc = prowptr[c + 1] - p0;
  const int base = (BS * p0 + a * nc) * BS;
  float2 acc = make_float2(0.f, 0.f);
  for (int m = lane; m < nc; m += L) {
    const int col = BS * bcol[p0 + m];
    const int q = base + m * BS;
#pragma unroll
    for (int v = 0; v < BS; ++v) {
      const float2 av = make_float2(ahi[q + v], alo[q + v]);
      const double xv = x[col + v]; const float xh = (float)xv;
      acc = df_add(acc, df_mul(av, make_float2(xh, (float)(xv - (double)xh))));
    }
  }
#pragma unroll
  for (int o = L / 2; o > 0; o >>= 1) {
    float2 t = make_float2(__shfl_down_sync(0xffffffffu, acc.x, o, L),
                           __shfl_down_sync(0xffffffffu, acc.y, o, L));
    acc = df_add(acc, t);
  }
  if (lane == 0) y[r] = (double)acc.x + (double)acc.y;
}

template <class F> float time_ms(F f, int reps) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); CK(cudaDeviceSynchronize());
  cudaEventRecord(a);
  for (int i = 0; i < reps; ++i) f();
  cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / reps;
}

int main(int argc, char** argv) {
  int nx = 46, ny = 112, nz = 22, bs = 3;
  if (argc >= 4) { nx = std::atoi(argv[1]); ny = std::atoi(argv[2]); nz = std::atoi(argv[3]); }
  if (argc >= 5) bs = std::atoi(argv[4]);
  const int cells = nx * ny * nz, n = bs * cells;
  auto lid = [&](int i, int j, int k) { return i + nx * (j + ny * k); };

  // --- the sparsity exactly as build_sparsity() writes it ---------------------
  std::vector<int> rowptr(n + 1, 0), colind, prowptr(cells + 1, 0);
  for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
    const int c = lid(i, j, k);
    std::vector<int> mem = {c};
    if (i > 0) mem.push_back(lid(i - 1, j, k)); if (i < nx - 1) mem.push_back(lid(i + 1, j, k));
    if (j > 0) mem.push_back(lid(i, j - 1, k)); if (j < ny - 1) mem.push_back(lid(i, j + 1, k));
    if (k > 0) mem.push_back(lid(i, j, k - 1)); if (k < nz - 1) mem.push_back(lid(i, j, k + 1));
    std::sort(mem.begin(), mem.end());
    const int nc = (int)mem.size();
    for (int a = 0; a < bs; ++a) {
      rowptr[bs * c + a + 1] = nc * bs;
      for (int m : mem) for (int v = 0; v < bs; ++v) colind.push_back(bs * m + v);
    }
    prowptr[c + 1] = nc;
  }
  for (int r = 0; r < n; ++r) rowptr[r + 1] += rowptr[r];
  for (int c = 0; c < cells; ++c) prowptr[c + 1] += prowptr[c];
  const int nnz = rowptr[n];
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::vector<double> a64(nnz), x(n);
  for (auto& v : a64) v = U(rng) * std::pow(10.0, 3.0 * U(rng));   // wide dynamic range
  for (auto& v : x) v = U(rng) * 1e3;
  std::vector<float> ahi(nnz), alo(nnz);
  for (int p = 0; p < nnz; ++p) { ahi[p] = (float)a64[p]; alo[p] = (float)(a64[p] - (double)ahi[p]); }
  std::printf("grid %dx%dx%d, bs %d: %d cells, %d rows, %d nonzeros\n", nx, ny, nz, bs, cells, n, nnz);

  int *d_rp, *d_ci, *d_prp, *d_bcol, *d_bad;
  float *d_hi, *d_lo; double *d_a, *d_x, *d_y1, *d_y2, *d_y3, *d_y4;
  CK(cudaMalloc(&d_rp, 4 * (n + 1))); CK(cudaMalloc(&d_ci, 4 * nnz));
  CK(cudaMalloc(&d_prp, 4 * (cells + 1))); CK(cudaMalloc(&d_bcol, 4 * prowptr[cells]));
  CK(cudaMalloc(&d_bad, 4));
  CK(cudaMalloc(&d_hi, 4 * nnz)); CK(cudaMalloc(&d_lo, 4 * nnz)); CK(cudaMalloc(&d_a, 8 * nnz));
  CK(cudaMalloc(&d_x, 8 * n)); CK(cudaMalloc(&d_y1, 8 * n)); CK(cudaMalloc(&d_y2, 8 * n));
  CK(cudaMalloc(&d_y3, 8 * n)); CK(cudaMalloc(&d_y4, 8 * n));
  CK(cudaMemcpy(d_rp, rowptr.data(), 4 * (n + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_ci, colind.data(), 4 * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_prp, prowptr.data(), 4 * (cells + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_hi, ahi.data(), 4 * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lo, alo.data(), 4 * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_a, a64.data(), 8 * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_x, x.data(), 8 * n, cudaMemcpyHostToDevice));

  const int T = 256;
  bspmv::k_bcol_build<<<(cells + T - 1) / T, T>>>(d_prp, d_rp, d_ci, d_bcol, cells, bs);
  CK(cudaMemset(d_bad, 0, 4));
  bspmv::k_bspmv_check<<<(cells + T - 1) / T, T>>>(d_prp, d_rp, d_ci, d_bcol, cells, bs, d_bad);
  int bad = -1; CK(cudaMemcpy(&bad, d_bad, 4, cudaMemcpyDeviceToHost));
  std::printf("layout check: %d mismatching entries\n", bad);

  // --- correctness ------------------------------------------------------------
  k_spmv_df64<<<(n + T - 1) / T, T>>>(d_rp, d_ci, d_hi, d_lo, d_x, d_y1, n);
  bspmv::launch_df64(d_prp, d_bcol, d_hi, d_lo, d_x, d_y2, n, bs, 0);
  cusparseHandle_t sp; cusparseCreate(&sp);
  cusparseSpMatDescr_t A; cusparseDnVecDescr_t vx, vy;
  cusparseCreateCsr(&A, n, n, nnz, d_rp, d_ci, d_a, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateDnVec(&vx, n, d_x, CUDA_R_64F); cusparseCreateDnVec(&vy, n, d_y3, CUDA_R_64F);
  const double one = 1.0, zero = 0.0; size_t bsz = 0; void* buf = nullptr;
  cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vx, &zero, vy,
                          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bsz);
  if (bsz) CK(cudaMalloc(&buf, bsz));
  auto cusp = [&] { cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vx, &zero, vy,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf); };
  cusp();
  bspmv::launch_f64(d_prp, d_bcol, d_a, d_x, d_y4, n, bs, 0);
  CK(cudaDeviceSynchronize());
  std::vector<double> y1(n), y2(n), y3(n), y4(n);
  CK(cudaMemcpy(y1.data(), d_y1, 8 * n, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y2.data(), d_y2, 8 * n, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y3.data(), d_y3, 8 * n, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y4.data(), d_y4, 8 * n, cudaMemcpyDeviceToHost));
  double rel = 0, emu = 0, bdf = 0, nrm = 0;
  for (int i = 0; i < n; ++i) {
    rel = std::max(rel, std::fabs(y3[i] - y4[i]));
    emu = std::max(emu, std::fabs(y3[i] - y1[i]));
    bdf = std::max(bdf, std::fabs(y2[i] - y1[i]));
    nrm = std::max(nrm, std::fabs(y3[i]));
  }
  std::printf("block df64 vs scalar df64: max |diff| / max |y| = %.2e\n", bdf / nrm);
  std::printf("block FP64 vs cuSPARSE: max |diff| / max |y| = %.2e\n", rel / nrm);
  std::printf("df64 vs cuSPARSE FP64:  max |diff| / max |y| = %.2e\n", emu / nrm);

  // --- speed ------------------------------------------------------------------
  const int reps = 200;
  float t_sdf = time_ms([&] { k_spmv_df64<<<(n + T - 1) / T, T>>>(d_rp, d_ci, d_hi, d_lo, d_x, d_y1, n); }, reps);
  float t_bdf = time_ms([&] { bspmv::launch_df64(d_prp, d_bcol, d_hi, d_lo, d_x, d_y2, n, bs, 0); }, reps);
  float t_csp = time_ms(cusp, reps);
  float t_bf = time_ms([&] { bspmv::launch_f64(d_prp, d_bcol, d_a, d_x, d_y4, n, bs, 0); }, reps);
  std::printf("\nper apply        scalar df64 %.4f ms | block df64 %.4f ms (%.2fx)\n",
              t_sdf, t_bdf, t_sdf / t_bdf);
  std::printf("                 cuSPARSE FP64 %.4f ms | block FP64 %.4f ms (%.2fx)\n",
              t_csp, t_bf, t_csp / t_bf);
  const double bytes_s = (double)nnz * (4 + 8) + 8.0 * n * 2, bytes_b = (double)nnz * 8 + 4.0 * prowptr[cells] + 8.0 * n * 2;
  if (bs == 3) {
    std::vector<double> ys(n);
    auto sw = [&](auto kern, int L, double* yy) { const long th = (long)n * L;
      kern<<<(unsigned)((th + T - 1) / T), T>>>(d_prp, d_bcol, d_a, d_x, yy, n); };
    auto swd = [&](auto kern, int L, double* yy) { const long th = (long)n * L;
      kern<<<(unsigned)((th + T - 1) / T), T>>>(d_prp, d_bcol, d_hi, d_lo, d_x, yy, n); };
    float f2 = time_ms([&] { sw(k_bspmv_f64_sw<3, 2>, 2, d_y4); }, reps);
    float f4 = time_ms([&] { sw(k_bspmv_f64_sw<3, 4>, 4, d_y4); }, reps);
    float f8 = time_ms([&] { sw(k_bspmv_f64_sw<3, 8>, 8, d_y4); }, reps);
    CK(cudaMemcpy(ys.data(), d_y4, 8 * n, cudaMemcpyDeviceToHost));
    double d = 0; for (int i = 0; i < n; ++i) d = std::max(d, std::fabs(ys[i] - y3[i]));
    std::printf("  sub-warp FP64  L=2 %.4f | L=4 %.4f | L=8 %.4f ms   (vs cuSPARSE %.2e)\n", f2, f4, f8, d / nrm);
    float g2 = time_ms([&] { swd(k_bspmv_df64_sw<3, 2>, 2, d_y2); }, reps);
    float g4 = time_ms([&] { swd(k_bspmv_df64_sw<3, 4>, 4, d_y2); }, reps);
    float g8 = time_ms([&] { swd(k_bspmv_df64_sw<3, 8>, 8, d_y2); }, reps);
    CK(cudaMemcpy(ys.data(), d_y2, 8 * n, cudaMemcpyDeviceToHost));
    d = 0; for (int i = 0; i < n; ++i) d = std::max(d, std::fabs(ys[i] - y1[i]));
    std::printf("  sub-warp df64  L=2 %.4f | L=4 %.4f | L=8 %.4f ms   (vs scalar df64 %.2e)\n", g2, g4, g8, d / nrm);
  }
  std::printf("df64 bytes moved (model): scalar %.1f MB, block %.1f MB (%.2fx)\n",
              bytes_s / 1e6, bytes_b / 1e6, bytes_s / bytes_b);

  // --- launch overhead: one Arnoldi tail, kernel by kernel against a graph ----
  const int kcols = 20;                 // Arnoldi column index being orthogonalised
  double *V, *w, *z, *r, *az, *h, *s, *dn;
  CK(cudaMalloc(&V, 8 * (size_t)n * (kcols + 1))); CK(cudaMalloc(&w, 8 * n)); CK(cudaMalloc(&z, 8 * n));
  CK(cudaMalloc(&r, 8 * n)); CK(cudaMalloc(&az, 8 * n));
  CK(cudaMalloc(&h, 8 * (2 * 64 + 1))); s = h + 64; dn = h + 128;   // one packed buffer
  CK(cudaMemset(V, 0, 8 * (size_t)n * (kcols + 1))); CK(cudaMemset(r, 0, 8 * n)); CK(cudaMemset(z, 0, 8 * n));
  cublasHandle_t blas; cublasCreate(&blas);
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cublasSetStream(blas, st);
  const double neg = -1.0;
  auto tail = [&](cudaStream_t q, bool devnorm) {
    const int nb = (n + T - 1) / T;
    // stage 2: operator apply, block smoother, three well kernels
    bspmv::launch_df64(d_prp, d_bcol, d_hi, d_lo, z, az, n, bs, q);
    k_stage2<<<nb, T, 0, q>>>(r, az, z, n);
    k_axpy<<<nb, T, 0, q>>>(z, 1e-3, r, n);
    k_axpy<<<nb, T, 0, q>>>(z, 1e-3, az, n);
    k_axpy<<<nb, T, 0, q>>>(az, 1e-3, r, n);
    // Arnoldi operator apply and two well kernels
    bspmv::launch_df64(d_prp, d_bcol, d_hi, d_lo, z, w, n, bs, q);
    k_axpy<<<nb, T, 0, q>>>(w, 1e-3, z, n);
    k_axpy<<<nb, T, 0, q>>>(w, 1e-3, az, n);
    // CGS2
    cublasDgemv(blas, CUBLAS_OP_T, n, kcols, &one, V, n, w, 1, &zero, h, 1);
    cublasDgemv(blas, CUBLAS_OP_N, n, kcols, &neg, V, n, h, 1, &one, w, 1);
    cublasDgemv(blas, CUBLAS_OP_T, n, kcols, &one, V, n, w, 1, &zero, s, 1);
    cublasDgemv(blas, CUBLAS_OP_N, n, kcols, &neg, V, n, s, 1, &one, w, 1);
    if (devnorm) {
      cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE);
      cublasDdot(blas, n, w, 1, w, 1, dn);
      cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);
    }
  };
  // The solver's pattern: a blocking capture stream, relaxed capture, timing
  // events recorded on the legacy default stream around the replay, a fixed
  // cuBLAS workspace, and one synchronising read of the packed results.
  cudaStream_t gs; CK(cudaStreamCreate(&gs));
  void* ws; CK(cudaMalloc(&ws, 4u << 20)); cublasSetWorkspace(blas, ws, 4u << 20);
  cublasSetStream(blas, 0);
  std::vector<double> z0(n), r0(n), v0((size_t)n * (kcols + 1));
  for (auto& v : z0) v = U(rng); for (auto& v : r0) v = U(rng);
  for (auto& v : v0) v = U(rng) * 1e-3;
  auto reset = [&] {
    CK(cudaMemcpy(z, z0.data(), 8 * n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(r, r0.data(), 8 * n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(V, v0.data(), 8 * v0.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(az, 0, 8 * n)); CK(cudaMemset(w, 0, 8 * n));
  };
  std::vector<double> hh(2 * 64 + 1), ref(2 * 64 + 1), wref(n), wg(n);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  // (a) today: kernel by kernel on the legacy stream, three host reads
  auto step_today = [&] {
    tail(0, false);
    CK(cudaMemcpy(hh.data(), h, 8 * kcols, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hh.data() + 64, s, 8 * kcols, cudaMemcpyDeviceToHost));
    double nn; cublasDdot(blas, n, w, 1, w, 1, &nn); hh[128] = nn;
  };
  // (b) the same launches, one host read
  auto step_onesync = [&] {
    tail(0, true);
    CK(cudaMemcpy(hh.data(), h, 8 * hh.size(), cudaMemcpyDeviceToHost));
  };
  // (c) one host read, the tail replayed from a graph on the blocking stream
  cudaGraph_t g; cudaGraphExec_t ge;
  cublasSetStream(blas, gs);
  CK(cudaStreamBeginCapture(gs, cudaStreamCaptureModeRelaxed));
  tail(gs, true);
  CK(cudaStreamEndCapture(gs, &g));
  cublasSetStream(blas, 0);
  CK(cudaGraphInstantiateWithFlags(&ge, g, 0));
  auto step_graph = [&] {
    cudaEventRecord(e0, 0);
    CK(cudaGraphLaunch(ge, gs));
    cudaEventRecord(e1, 0);
    CK(cudaMemcpy(hh.data(), h, 8 * hh.size(), cudaMemcpyDeviceToHost));
  };

  // correctness of the ordering: the graph must reproduce the direct launches
  // bit for bit, and the legacy-stream read must see its results.
  reset(); step_onesync();
  ref = hh; CK(cudaMemcpy(wref.data(), w, 8 * n, cudaMemcpyDeviceToHost));
  reset(); std::fill(hh.begin(), hh.end(), -1.0); step_graph();
  CK(cudaMemcpy(wg.data(), w, 8 * n, cudaMemcpyDeviceToHost));
  const bool same = std::memcmp(ref.data(), hh.data(), 8 * hh.size()) == 0 &&
                    std::memcmp(wref.data(), wg.data(), 8 * (size_t)n) == 0;
  float ev_ms = 0; cudaEventElapsedTime(&ev_ms, e0, e1);
  std::printf("\ngraph replay == direct launches bitwise: %s (event span %.3f ms)\n",
              same ? "yes" : "NO", ev_ms);

  auto wall = [&](auto f, int k) {
    f(); CK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < k; ++i) f();
    CK(cudaDeviceSynchronize());
    return 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / k;
  };
  const double a0 = wall(step_today, 300), a1 = wall(step_onesync, 300), a2 = wall(step_graph, 300);
  std::printf("Arnoldi tail (10 kernels + 5 cuBLAS calls, %d columns):\n", kcols);
  std::printf("  today (3 host reads)       %.4f ms\n", a0);
  std::printf("  one host read              %.4f ms (%.2fx)\n", a1, a0 / a1);
  std::printf("  one read + CUDA graph      %.4f ms (%.2fx)\n", a2, a0 / a2);

  // The price of a capture, and how many replays repay it.  Record and
  // instantiate the same tail repeatedly, as the solver does every time the
  // system signature changes, and divide by the time one replay saves over
  // launching the tail directly with the same single host read.
  const int ncap = 50;
  auto tc0 = std::chrono::steady_clock::now();
  for (int k = 0; k < ncap; ++k) {
    cudaGraph_t g2; cudaGraphExec_t ge2;
    cublasSetStream(blas, gs);
    CK(cudaStreamBeginCapture(gs, cudaStreamCaptureModeRelaxed));
    tail(gs, true);
    CK(cudaStreamEndCapture(gs, &g2));
    cublasSetStream(blas, 0);
    CK(cudaGraphInstantiateWithFlags(&ge2, g2, 0));
    cudaGraphDestroy(g2);
    cudaGraphExecDestroy(ge2);
  }
  const double cap_ms = 1e3 * std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - tc0).count() / ncap;
  const double saving = a1 - a2;
  if (saving > 0)
    std::printf("capture + instantiate: %.4f ms each; a replay saves %.4f ms "
                "over direct launches, so a graph must be reused %.1f times to "
                "repay its capture\n", cap_ms, saving, cap_ms / saving);
  else
    std::printf("capture + instantiate: %.4f ms each; a replay saves nothing "
                "here (%.4f ms), so on this device and size a graph never "
                "repays its capture\n", cap_ms, saving);
  return 0;
}
