// hypre_cusolver_mixed_cpr.cu
//
// GPU mixed-precision CPR reference:
//   * FP64 outer restarted FGMRES on GPU
//   * cuSPARSE FP64 SpMV for true residual/full operator
//   * cuBLAS FP64 dot/norm/gemv operations
//   * HYPRE FP32 BoomerAMG pressure solve on GPU
//   * FP32 CPR stage-2 local 2x2 block smoother on GPU
//
// Build against the FP32 HYPRE install only. The full matrix is stored by this
// program in FP64/FP32 CUDA CSR arrays; HYPRE only owns the FP32 pressure AMG.
//
// ---------------------------------------------------------------------------
// GMRES optimization notes (vs. the original normal-equations implementation):
//
//   The inner Arnoldi loop now uses the standard Givens-rotation QR update of
//   the Hessenberg matrix instead of forming and LU-solving the normal
//   equations (H^T H) y = H^T (beta e1) every iteration. Consequences:
//
//     * The residual norm estimate |g[j+1]| is available for free each inner
//       step, so we no longer recompute the true residual with an extra SpMV
//       + nrm2 per iteration, and we never run cuSolver LU in the hot loop.
//     * The solution x = xbase + Z y is reconstructed once per restart cycle
//       (not once per inner iteration).
//     * Orthogonalization is classical Gram-Schmidt with one reorthogonal-
//       ization pass (CGS2), implemented as two batched cublasDgemv calls per
//       pass. This replaces modified Gram-Schmidt, which forced a host
//       synchronization (cublasDdot -> host) and a single-double cudaMemcpy on
//       every i in the inner loop. The Hessenberg/Givens bookkeeping lives on
//       the host (it is tiny: (restart+1) x restart), with one device->host
//       copy of the new Hessenberg column per inner step.
//     * cuSPARSE SpMV reuses persistent dense-vector descriptors and a cached
//       work buffer instead of creating/destroying descriptors and re-querying
//       the buffer size on every call.
//
//   Net effect in the solve loop: per inner iteration drops from
//   {2 SpMV + AMG + LU(getrf+getrs) + GEMM + j*(dot+axpy+tiny memcpy) + full x
//    rebuild + residual SpMV + nrm2}
//   to
//   {2 SpMV + AMG + 4 small GEMV + 1 column copy + 1 nrm2 + O(j) host flops},
//   with x rebuilt and the true residual checked only at restart boundaries.
//   The math (flexible/right-preconditioned GMRES, FP64 outer, FP32 precond)
//   is unchanged, as is the JSON output schema and the CLI.
// ---------------------------------------------------------------------------

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <mpi.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::ostringstream os; os << "CUDA error " << cudaGetErrorString(e) << " at line " << __LINE__; throw std::runtime_error(os.str()); }} while (0)
#define CHECK_CUBLAS(x) do { cublasStatus_t e = (x); if (e != CUBLAS_STATUS_SUCCESS) { \
  std::ostringstream os; os << "cuBLAS error " << (int)e << " at line " << __LINE__; throw std::runtime_error(os.str()); }} while (0)
#define CHECK_CUSPARSE(x) do { cusparseStatus_t e = (x); if (e != CUSPARSE_STATUS_SUCCESS) { \
  std::ostringstream os; os << "cuSPARSE error " << (int)e << " at line " << __LINE__; throw std::runtime_error(os.str()); }} while (0)

#ifndef CPR_USE_FP32_PRECOND
#define CPR_USE_FP32_PRECOND 1
#endif

#if CPR_USE_FP32_PRECOND
using PrecReal = float;
static constexpr const char* CPR_VARIANT = "mixed";
static constexpr const char* CPR_METHOD =
    "GPU mixed CPR: FP64 FGMRES (Givens) + HYPRE FP32 AMG";
#else
using PrecReal = double;
static constexpr const char* CPR_VARIANT = "fp64";
static constexpr const char* CPR_METHOD =
    "GPU FP64 CPR: FP64 FGMRES (Givens) + HYPRE FP64 AMG";
#endif

static const char* arg_value(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return argv[i + 1];
  return def;
}

static bool has_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == key) return true;
  return false;
}

static std::string jnum(double x) {
  if (!std::isfinite(x)) return "null";
  std::ostringstream os; os.precision(10); os << x; return os.str();
}

static double now_ms() {
  using clock = std::chrono::high_resolution_clock;
  static const auto t0 = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

struct HostSystem {
  int side = 0, cells = 0, n = 0, nnz = 0, pnnz = 0;
  std::vector<int> rowptr, colind, prowptr, pcolind;
  std::vector<double> aval64, b64;
  std::vector<float> aval32;
  std::vector<HYPRE_Complex> pval_hypre;
  std::vector<PrecReal> dinv_prec;
};

static int cid(int i, int j, int k, int s) { return (k * s + j) * s + i; }

static double perm_factor(int i, int j, int k, double sigma) {
  const double x = std::sin(0.173 * (i + 1)) + std::cos(0.119 * (j + 2)) + std::sin(0.071 * (k + 3));
  return std::exp(0.25 * sigma * x);
}

static HostSystem build_host_system(int side, double sigma) {
  HostSystem S;
  S.side = side; S.cells = side * side * side; S.n = 2 * S.cells;
  std::vector<std::vector<std::pair<int, double>>> rows(S.n), prows(S.cells);
  S.dinv_prec.assign(4 * S.cells, (PrecReal)0);
  const int di[6] = {1,-1,0,0,0,0}, dj[6] = {0,0,1,-1,0,0}, dk[6] = {0,0,0,0,1,-1};
  for (int k = 0; k < side; ++k) for (int j = 0; j < side; ++j) for (int i = 0; i < side; ++i) {
    const int c = cid(i,j,k,side), p = 2*c, q = p+1;
    double pdiag = 1e-3, sdiag = 2.0, ps = 0.05, sp = -0.02;
    for (int m = 0; m < 6; ++m) {
      const int ni = i + di[m], nj = j + dj[m], nk = k + dk[m];
      if (ni < 0 || nj < 0 || nk < 0 || ni >= side || nj >= side || nk >= side) continue;
      const int cn = cid(ni,nj,nk,side);
      const double t = 0.5 * (perm_factor(i,j,k,sigma) + perm_factor(ni,nj,nk,sigma));
      pdiag += t; sdiag += 0.04 * t;
      rows[p].push_back({2*cn, -t});
      rows[q].push_back({2*cn+1, -0.04*t});
      prows[c].push_back({cn, -t});
    }
    rows[p].push_back({p, pdiag}); rows[p].push_back({q, ps});
    rows[q].push_back({p, sp}); rows[q].push_back({q, sdiag});
    prows[c].push_back({c, pdiag});
    const double det = pdiag * sdiag - ps * sp;
    S.dinv_prec[4*c+0] = (PrecReal)(sdiag / det);
    S.dinv_prec[4*c+1] = (PrecReal)(-ps / det);
    S.dinv_prec[4*c+2] = (PrecReal)(-sp / det);
    S.dinv_prec[4*c+3] = (PrecReal)(pdiag / det);
  }

  S.rowptr.assign(S.n + 1, 0);
  for (int r = 0; r < S.n; ++r) {
    auto& row = rows[r];
    std::sort(row.begin(), row.end(), [](auto a, auto b){ return a.first < b.first; });
    S.rowptr[r+1] = S.rowptr[r] + (int)row.size();
    for (auto [c,v] : row) { S.colind.push_back(c); S.aval64.push_back(v); S.aval32.push_back((float)v); }
  }
  S.prowptr.assign(S.cells + 1, 0);
  for (int r = 0; r < S.cells; ++r) {
    auto& row = prows[r];
    std::sort(row.begin(), row.end(), [](auto a, auto b){ return a.first < b.first; });
    S.prowptr[r+1] = S.prowptr[r] + (int)row.size();
    for (auto [c,v] : row) { S.pcolind.push_back(c); S.pval_hypre.push_back((HYPRE_Complex)v); }
  }
  S.nnz = (int)S.colind.size(); S.pnnz = (int)S.pcolind.size();

  std::vector<double> xtrue(S.n);
  for (int c = 0; c < S.cells; ++c) {
    xtrue[2*c] = 1.0 + 0.01 * std::sin(0.01 * c);
    xtrue[2*c+1] = 0.2 + 0.01 * std::cos(0.013 * c);
  }
  S.b64.assign(S.n, 0.0);
  for (int r = 0; r < S.n; ++r) {
    double acc = 0.0;
    for (int p = S.rowptr[r]; p < S.rowptr[r+1]; ++p) acc += S.aval64[p] * xtrue[S.colind[p]];
    S.b64[r] = acc;
  }
  return S;
}

template<class T>
static T* dcopy_from_host(const std::vector<T>& h) {
  T* d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, sizeof(T) * h.size()));
  CHECK_CUDA(cudaMemcpy(d, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice));
  return d;
}

struct DeviceSystem {
  int cells = 0, n = 0, nnz = 0;
  int *rowptr = nullptr, *colind = nullptr;
  double *aval64 = nullptr, *b64 = nullptr;
  float *aval32 = nullptr, *dinv32 = nullptr;
  PrecReal *dinv_prec = nullptr;
  cusparseSpMatDescr_t A64 = nullptr;
};

static DeviceSystem make_device_system(const HostSystem& H) {
  DeviceSystem D; D.cells = H.cells; D.n = H.n; D.nnz = H.nnz;
  D.rowptr = dcopy_from_host(H.rowptr);
  D.colind = dcopy_from_host(H.colind);
  D.aval64 = dcopy_from_host(H.aval64);
  D.aval32 = dcopy_from_host(H.aval32);
  D.b64 = dcopy_from_host(H.b64);
  D.dinv_prec = dcopy_from_host(H.dinv_prec);
  CHECK_CUSPARSE(cusparseCreateCsr(&D.A64, D.n, D.n, D.nnz, D.rowptr, D.colind, D.aval64,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  return D;
}

__global__ static void fill_zero(double* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 0.0;
}

__global__ static void residual_kernel(const double* b, const double* Ax, double* r, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) r[i] = b[i] - Ax[i];
}

template<class T>
__global__ static void extract_pressure_rhs(const double* r, T* rp, int cells) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < cells) rp[c] = (T)r[2*c];
}

template<class T>
__global__ static void pressure_to_z(const T* dp, double* z, int n, int cells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] = 0.0;
  if (i < cells) z[2*i] = (double)dp[i];
}

template<class T>
__global__ static void stage2_block_jacobi(const double* r, const double* Az, const T* dinv, double* z, int cells) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  double r0 = r[2*c] - Az[2*c];
  double r1 = r[2*c+1] - Az[2*c+1];
  T d00 = dinv[4*c+0], d01 = dinv[4*c+1], d10 = dinv[4*c+2], d11 = dinv[4*c+3];
  z[2*c]   += (double)(d00 * (T)r0 + d01 * (T)r1);
  z[2*c+1] += (double)(d10 * (T)r0 + d11 * (T)r1);
}

__global__ static void scale_copy(double* dst, const double* src, double alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = alpha * src[i];
}

__global__ static void combine_z_from_base(double* x, const double* xbase, const double* Z, const double* y, int n, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double acc = xbase[i];
  for (int j = 0; j < m; ++j) acc += y[j] * Z[(size_t)j * n + i];
  x[i] = acc;
}

class PressureAMG {
 public:
  PressureAMG(const HostSystem& H, int amg_print_level, int coarsen_type, int relax_type)
      : n_(H.cells) {
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, n_-1, 0, n_-1, &A_);
    HYPRE_IJMatrixSetObjectType(A_, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_);
    for (int r = 0; r < n_; ++r) {
      HYPRE_Int ncols = H.prowptr[r+1] - H.prowptr[r];
      std::vector<HYPRE_BigInt> cols(ncols);
      std::vector<HYPRE_Complex> vals(ncols);
      for (int j = 0; j < ncols; ++j) {
        int p = H.prowptr[r] + j;
        cols[j] = H.pcolind[p];
        vals[j] = H.pval_hypre[p];
      }
      HYPRE_BigInt row = r;
      HYPRE_IJMatrixSetValues(A_, 1, &ncols, &row, cols.data(), vals.data());
    }
    HYPRE_IJMatrixAssemble(A_);
    HYPRE_IJMatrixGetObject(A_, (void**)&par_A_);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n_-1, &b_);
    HYPRE_IJVectorSetObjectType(b_, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b_);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n_-1, &x_);
    HYPRE_IJVectorSetObjectType(x_, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x_);
    std::vector<HYPRE_BigInt> idx(n_);
    std::vector<HYPRE_Complex> zero(n_, (HYPRE_Complex)0.0);
    for (int i = 0; i < n_; ++i) idx[i] = i;
    HYPRE_IJVectorSetValues(b_, n_, idx.data(), zero.data());
    HYPRE_IJVectorSetValues(x_, n_, idx.data(), zero.data());
    HYPRE_IJVectorAssemble(b_);
    HYPRE_IJVectorAssemble(x_);
    HYPRE_IJVectorGetObject(b_, (void**)&par_b_);
    HYPRE_IJVectorGetObject(x_, (void**)&par_x_);

    HYPRE_BoomerAMGCreate(&solver_);
    HYPRE_BoomerAMGSetPrintLevel(solver_, amg_print_level);
    HYPRE_BoomerAMGSetLogging(solver_, 1);
    HYPRE_BoomerAMGSetMaxIter(solver_, 1);
    HYPRE_BoomerAMGSetTol(solver_, 0.0);
    HYPRE_BoomerAMGSetCoarsenType(solver_, coarsen_type);
    HYPRE_BoomerAMGSetRelaxType(solver_, relax_type);
    HYPRE_BoomerAMGSetNumSweeps(solver_, 1);
    HYPRE_BoomerAMGSetup(solver_, par_A_, par_b_, par_x_);

    CHECK_CUDA(cudaMalloc(&d_idx_, sizeof(HYPRE_BigInt) * n_));
    CHECK_CUDA(cudaMemcpy(d_idx_, idx.data(), sizeof(HYPRE_BigInt) * n_, cudaMemcpyHostToDevice));
  }

  ~PressureAMG() {
    if (d_idx_) cudaFree(d_idx_);
    HYPRE_BoomerAMGDestroy(solver_);
    HYPRE_IJVectorDestroy(x_);
    HYPRE_IJVectorDestroy(b_);
    HYPRE_IJMatrixDestroy(A_);
  }

  void solve(PrecReal* d_rhs, PrecReal* d_x) {
    HYPRE_IJVectorSetValues(b_, n_, d_idx_, (HYPRE_Complex*)d_rhs);
    HYPRE_IJVectorSetValues(x_, n_, d_idx_, (HYPRE_Complex*)d_x);
    HYPRE_IJVectorAssemble(b_);
    HYPRE_IJVectorAssemble(x_);
    HYPRE_BoomerAMGSolve(solver_, par_A_, par_b_, par_x_);
    HYPRE_IJVectorGetValues(x_, n_, d_idx_, (HYPRE_Complex*)d_x);
  }

 private:
  int n_;
  HYPRE_IJMatrix A_ = nullptr;
  HYPRE_ParCSRMatrix par_A_ = nullptr;
  HYPRE_IJVector b_ = nullptr, x_ = nullptr;
  HYPRE_ParVector par_b_ = nullptr, par_x_ = nullptr;
  HYPRE_Solver solver_ = nullptr;
  HYPRE_BigInt* d_idx_ = nullptr;
};

// ---------------------------------------------------------------------------
// Persistent SpMV: reuse dense-vector descriptors and the work buffer across
// all calls. Only re-query/allocate the buffer when it first proves too small.
// ---------------------------------------------------------------------------
struct SpMV {
  cusparseHandle_t sp = nullptr;
  cusparseSpMatDescr_t A = nullptr;
  cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
  void* buf = nullptr;
  size_t bufsz = 0;
  int n = 0;
};

static void spmv_init(SpMV& S, cusparseHandle_t sp, const DeviceSystem& D) {
  S.sp = sp; S.A = D.A64; S.n = D.n;
  // Placeholder pointers; reset per call via cusparseDnVecSetValues.
  CHECK_CUSPARSE(cusparseCreateDnVec(&S.vx, D.n, D.b64, CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&S.vy, D.n, D.b64, CUDA_R_64F));
}

static void spmv_free(SpMV& S) {
  if (S.vx) cusparseDestroyDnVec(S.vx);
  if (S.vy) cusparseDestroyDnVec(S.vy);
  if (S.buf) cudaFree(S.buf);
  S.vx = S.vy = nullptr; S.buf = nullptr; S.bufsz = 0;
}

// y <- A * x   (FP64)
static void spmv_apply(SpMV& S, const double* x, double* y) {
  const double alpha = 1.0, beta = 0.0;
  CHECK_CUSPARSE(cusparseDnVecSetValues(S.vx, (void*)x));
  CHECK_CUSPARSE(cusparseDnVecSetValues(S.vy, y));
  if (S.buf == nullptr) {
    size_t need = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(S.sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, S.A, S.vx, &beta, S.vy,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &need));
    if (need > 0) { CHECK_CUDA(cudaMalloc(&S.buf, need)); }
    S.bufsz = need;
  }
  CHECK_CUSPARSE(cusparseSpMV(S.sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, S.A, S.vx,
                              &beta, S.vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, S.buf));
}

// CPR preconditioner: z = M^{-1} r.
//   stage 1: AMG on the pressure block, scattered back into z
//   stage 2: local 2x2 block-Jacobi correction on the full residual
static void apply_cpr(SpMV& spm, const DeviceSystem& D, PressureAMG& amg,
                      const double* r, double* z, double* tmp_Az,
                      PrecReal* rp, PrecReal* dp) {
  const int threads = 256;
  const int nb = (D.n + threads - 1) / threads;
  const int cb = (D.cells + threads - 1) / threads;
  extract_pressure_rhs<PrecReal><<<cb, threads>>>(r, rp, D.cells);
  CHECK_CUDA(cudaMemset(dp, 0, sizeof(PrecReal) * D.cells));
  amg.solve(rp, dp);
  pressure_to_z<PrecReal><<<std::max(nb, cb), threads>>>(dp, z, D.n, D.cells);
  spmv_apply(spm, z, tmp_Az);
  stage2_block_jacobi<PrecReal><<<cb, threads>>>(r, tmp_Az, D.dinv_prec, z, D.cells);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  HYPRE_Init();
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

  try {
    const int side = std::atoi(arg_value(argc, argv, "--n-side", "32"));
    const double sigma = std::atof(arg_value(argc, argv, "--sigma", "1.5"));
    const double tol = std::atof(arg_value(argc, argv, "--tol", "1e-8"));
    const int restart = std::atoi(arg_value(argc, argv, "--restart", "30"));
    const int max_iters = std::atoi(arg_value(argc, argv, "--max-iters", "80"));
    const int amg_print = std::atoi(arg_value(argc, argv, "--amg-print-level", "0"));
    const int coarsen = std::atoi(arg_value(argc, argv, "--coarsen-type", "10"));
    const int relax = std::atoi(arg_value(argc, argv, "--relax-type", "18"));
    const bool progress = has_flag(argc, argv, "--progress");

    const double tb0 = now_ms();
    HostSystem H = build_host_system(side, sigma);
    DeviceSystem D = make_device_system(H);
    PressureAMG pressure(H, amg_print, coarsen, relax);
    const double build_ms = now_ms() - tb0;

    cusparseHandle_t sp; cublasHandle_t blas;
    CHECK_CUSPARSE(cusparseCreate(&sp));
    CHECK_CUBLAS(cublasCreate(&blas));
    CHECK_CUBLAS(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));

    SpMV spm; spmv_init(spm, sp, D);

    // Device work vectors.
    double *x=nullptr, *xbase=nullptr, *r=nullptr, *Ax=nullptr, *tmp=nullptr;
    double *V=nullptr, *Z=nullptr, *y=nullptr, *h_dev=nullptr, *s_dev=nullptr;
    PrecReal *rp=nullptr, *dp=nullptr;
    CHECK_CUDA(cudaMalloc(&x, sizeof(double)*D.n));
    CHECK_CUDA(cudaMalloc(&xbase, sizeof(double)*D.n));
    CHECK_CUDA(cudaMalloc(&r, sizeof(double)*D.n));
    CHECK_CUDA(cudaMalloc(&Ax, sizeof(double)*D.n));
    CHECK_CUDA(cudaMalloc(&tmp, sizeof(double)*D.n));
    CHECK_CUDA(cudaMalloc(&V, sizeof(double)*(size_t)(restart+1)*D.n));
    CHECK_CUDA(cudaMalloc(&Z, sizeof(double)*(size_t)restart*D.n));
    CHECK_CUDA(cudaMalloc(&y, sizeof(double)*restart));
    CHECK_CUDA(cudaMalloc(&h_dev, sizeof(double)*(restart+1)));
    CHECK_CUDA(cudaMalloc(&s_dev, sizeof(double)*(restart+1)));
    CHECK_CUDA(cudaMalloc(&rp, sizeof(PrecReal)*D.cells));
    CHECK_CUDA(cudaMalloc(&dp, sizeof(PrecReal)*D.cells));
    fill_zero<<<(D.n+255)/256,256>>>(x, D.n);

    // Host Hessenberg + Givens bookkeeping (tiny: (restart+1) x restart).
    const int ldh = restart + 1;
    std::vector<double> Hbar((size_t)ldh * restart, 0.0);  // column-major
    std::vector<double> cs(restart, 0.0), sn(restart, 0.0);
    std::vector<double> g(restart + 1, 0.0);
    std::vector<double> hcol(restart + 1, 0.0);
    std::vector<double> y_host(restart, 0.0);

    const double one = 1.0, neg_one = -1.0, zero = 0.0;
    const double tiny = 1e-300;

    double bnorm = 0.0;
    CHECK_CUBLAS(cublasDnrm2(blas, D.n, D.b64, 1, &bnorm));
    std::vector<double> hist;
    int total_iters = 0;
    const double ts0 = now_ms();

    while (total_iters < max_iters) {
      // True residual at the start of this restart cycle.
      spmv_apply(spm, x, Ax);
      residual_kernel<<<(D.n+255)/256,256>>>(D.b64, Ax, r, D.n);
      double rnorm = 0.0;
      CHECK_CUBLAS(cublasDnrm2(blas, D.n, r, 1, &rnorm));
      double rel = rnorm / std::max(bnorm, tiny);
      if (hist.empty()) hist.push_back(rel);   // record the initial true residual once
      if (progress) std::cerr << "[HYPRE+cuSolver " << CPR_VARIANT << " CPR] iter=" << total_iters
                              << " residual=" << rel << std::endl;
      if (rel <= tol) break;

      const int mmax = std::min(restart, max_iters - total_iters);
      CHECK_CUDA(cudaMemcpy(xbase, x, sizeof(double)*D.n, cudaMemcpyDeviceToDevice));

      // v_0 = r / ||r||
      scale_copy<<<(D.n+255)/256,256>>>(V, r, 1.0/rnorm, D.n);
      std::fill(g.begin(), g.end(), 0.0);
      g[0] = rnorm;

      int used = 0;
      for (int j = 0; j < mmax; ++j) {
        double* vj = V + (size_t)j*D.n;
        double* zj = Z + (size_t)j*D.n;

        // z_j = M^{-1} v_j  (apply_cpr internally uses `tmp` as scratch)
        apply_cpr(spm, D, pressure, vj, zj, tmp, rp, dp);

        // w = A z_j   (tmp is free again after apply_cpr returns)
        double* w = tmp;
        spmv_apply(spm, zj, w);

        // --- Orthogonalize w against V[:,0..j] : classical GS + 1 reorth (CGS2) ---
        const int nc = j + 1;
        // pass 1:  h = V^T w ;  w -= V h
        CHECK_CUBLAS(cublasDgemv(blas, CUBLAS_OP_T, D.n, nc, &one,  V, D.n, w, 1, &zero, h_dev, 1));
        CHECK_CUBLAS(cublasDgemv(blas, CUBLAS_OP_N, D.n, nc, &neg_one, V, D.n, h_dev, 1, &one, w, 1));
        // pass 2:  s = V^T w ;  w -= V s ;  h += s
        CHECK_CUBLAS(cublasDgemv(blas, CUBLAS_OP_T, D.n, nc, &one,  V, D.n, w, 1, &zero, s_dev, 1));
        CHECK_CUBLAS(cublasDgemv(blas, CUBLAS_OP_N, D.n, nc, &neg_one, V, D.n, s_dev, 1, &one, w, 1));
        CHECK_CUBLAS(cublasDaxpy(blas, nc, &one, s_dev, 1, h_dev, 1));

        // one device->host copy of the new Hessenberg column (nc doubles)
        CHECK_CUDA(cudaMemcpy(hcol.data(), h_dev, sizeof(double)*nc, cudaMemcpyDeviceToHost));

        double hn = 0.0;
        CHECK_CUBLAS(cublasDnrm2(blas, D.n, w, 1, &hn));

        // Store column j of the (unrotated) Hessenberg.
        double* Hj = &Hbar[(size_t)j*ldh];
        for (int i = 0; i < nc; ++i) Hj[i] = hcol[i];
        Hj[j+1] = hn;

        // v_{j+1} = w / hn
        if (hn > tiny) {
          scale_copy<<<(D.n+255)/256,256>>>(V + (size_t)(j+1)*D.n, w, 1.0/hn, D.n);
        }

        // Apply previous Givens rotations to column j.
        for (int i = 0; i < j; ++i) {
          double t0 = cs[i]*Hj[i] + sn[i]*Hj[i+1];
          Hj[i+1]   = -sn[i]*Hj[i] + cs[i]*Hj[i+1];
          Hj[i]     = t0;
        }
        // New rotation eliminating Hj[j+1].
        double denom = std::hypot(Hj[j], Hj[j+1]);
        if (denom < tiny) denom = tiny;
        cs[j] = Hj[j]   / denom;
        sn[j] = Hj[j+1] / denom;
        Hj[j]   = cs[j]*Hj[j] + sn[j]*Hj[j+1];   // = denom
        Hj[j+1] = 0.0;
        // Rotate the RHS vector g.
        g[j+1] = -sn[j]*g[j];
        g[j]   =  cs[j]*g[j];

        rel = std::fabs(g[j+1]) / std::max(bnorm, tiny);
        used = j + 1;
        total_iters++;
        hist.push_back(rel);   // per-iteration FGMRES residual estimate (free from Givens)
        if (progress) std::cerr << "[HYPRE+cuSolver " << CPR_VARIANT << " CPR]   inner=" << used
                                << " residual=" << rel << std::endl;
        if (rel <= tol || total_iters >= max_iters || hn <= tiny) break;
      }

      // Solve the (used x used) upper-triangular system R y = g by back-substitution.
      for (int i = used - 1; i >= 0; --i) {
        double s = g[i];
        for (int k = i + 1; k < used; ++k) s -= Hbar[(size_t)i + (size_t)k*ldh] * y_host[k];
        double diag = Hbar[(size_t)i + (size_t)i*ldh];
        y_host[i] = (std::fabs(diag) > tiny) ? s / diag : 0.0;
      }
      CHECK_CUDA(cudaMemcpy(y, y_host.data(), sizeof(double)*used, cudaMemcpyHostToDevice));
      combine_z_from_base<<<(D.n+255)/256,256>>>(x, xbase, Z, y, D.n, used);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    const double solve_ms = now_ms() - ts0;

    spmv_apply(spm, x, Ax);
    residual_kernel<<<(D.n+255)/256,256>>>(D.b64, Ax, r, D.n);
    double rnorm = 0.0; CHECK_CUBLAS(cublasDnrm2(blas, D.n, r, 1, &rnorm));
    const double final_rel = rnorm / std::max(bnorm, tiny);

    std::cout << "{\n";
    std::cout << "  \"method\": \"" << CPR_METHOD << "\",\n";
    std::cout << "  \"variant\": \"" << CPR_VARIANT << "\",\n";
    std::cout << "  \"n_side\": " << side << ", \"cells\": " << H.cells << ", \"unknowns\": " << H.n << ",\n";
    std::cout << "  \"nnz\": " << H.nnz << ", \"pressure_nnz\": " << H.pnnz << ",\n";
    std::cout << "  \"build_setup_ms\": " << jnum(build_ms) << ", \"solve_ms\": " << jnum(solve_ms)
              << ", \"total_ms\": " << jnum(build_ms + solve_ms) << ",\n";
    std::cout << "  \"iters\": " << total_iters << ", \"converged\": " << (final_rel <= tol ? "true" : "false") << ",\n";
    std::cout << "  \"final_residual\": " << jnum(final_rel) << ",\n";
    std::cout << "  \"residual_history\": [";
    for (size_t i = 0; i < hist.size(); ++i) std::cout << (i ? ", " : "") << jnum(hist[i]);
    std::cout << "]\n}\n";

    spmv_free(spm);
    cudaFree(dp); cudaFree(rp); cudaFree(s_dev); cudaFree(h_dev); cudaFree(y);
    cudaFree(Z); cudaFree(V); cudaFree(tmp); cudaFree(Ax); cudaFree(r); cudaFree(xbase); cudaFree(x);
    cublasDestroy(blas); cusparseDestroy(sp);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    HYPRE_Finalize(); MPI_Finalize(); return 2;
  }

  HYPRE_Finalize();
  MPI_Finalize();
  return 0;
}
