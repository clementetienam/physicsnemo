// SPDX-License-Identifier: Apache-2.0
//
// tc_jacobian.cu -- the batched operator product behind JACOBIAN TENSORCORE.
// Compiled only with TENSORCORE=1.
//
//   TF32   strided batched GEMM, Tensor Core multiply with FP32 accumulation.
//          Needs CUTLASS at build time and compute capability 8.0 at run time.
//   FP32   a tiled product on CUDA cores, on any GPU and with no CUTLASS.
#include "tc_jacobian.hpp"

#include <stdexcept>
#include <string>

#if defined(NVRS_HAVE_CUTLASS)
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#endif

namespace nvrs {
namespace tc {

// One block per connection, one thread per output element.
__global__ void k_batched_product(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int batch) {
  const int q = blockIdx.x;
  if (q >= batch) return;
  const int row = threadIdx.y, col = threadIdx.x;
  const size_t off = (size_t)q * kTile * kTile;

  __shared__ float sa[kTile][kTile];
  __shared__ float sb[kTile][kTile];
  sa[row][col] = a[off + row * kTile + col];
  sb[row][col] = b[off + row * kTile + col];
  __syncthreads();

  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kTile; ++k) acc = fmaf(sa[row][k], sb[k][col], acc);
  c[off + row * kTile + col] = acc;
}

#if defined(NVRS_HAVE_CUTLASS)
// A 128x128 threadblock with 128-bit alignment, the usual choice for large
// GEMMs, cannot be instantiated for a 16x16 problem; these shapes can.
using GemmTF32 = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3, 4, 4,
    cutlass::arch::OpMultiplyAddFastF32>;

static void run_tf32(const float* a, const float* b, float* c, int batch) {
  constexpr int n = kTile, stride = kTile * kTile;
  GemmTF32::Arguments args({n, n, n}, {a, n}, stride, {b, n}, stride,
                           {c, n}, stride, {c, n}, stride,
                           {1.0f, 0.0f}, batch);
  GemmTF32 op;
  if (op.can_implement(args) != cutlass::Status::kSuccess)
    throw std::runtime_error("TENSORCORE: CUTLASS rejected the 16x16 batch");
  const size_t bytes = op.get_workspace_size(args);
  void* ws = nullptr;
  if (bytes && cudaMalloc(&ws, bytes) != cudaSuccess)
    throw std::runtime_error("TENSORCORE: no memory for the CUTLASS workspace");
  const bool ok = op.initialize(args, ws) == cutlass::Status::kSuccess &&
                  op() == cutlass::Status::kSuccess;
  if (ws) cudaFree(ws);
  if (!ok) throw std::runtime_error("TENSORCORE: the CUTLASS GEMM failed");
}
#endif

static bool device_has_tf32() {
  int dev = 0, major = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) !=
      cudaSuccess)
    return false;
  return major >= 8;
}

static bool g_enabled = false;
int g_precision = kPrecTF32;
int g_validate = 0;

bool enabled() { return g_enabled; }

const char* backend_name(int precision) {
  return precision == kPrecTF32
             ? "TF32 Tensor Core with FP32 accumulation (CUTLASS)"
             : "FP32 on CUDA cores";
}

void enable(int precision, int validate) {
  if (precision == kPrecTF32) {
#if !defined(NVRS_HAVE_CUTLASS)
    throw std::runtime_error(
        "TCPRECISION TF32 needs CUTLASS. Rebuild with make TENSORCORE=1 "
        "CUTLASS_DIR=<checkout>, or set TCPRECISION FP32.");
#endif
    if (!device_has_tf32())
      throw std::runtime_error(
          "TCPRECISION TF32 needs compute capability 8.0 or newer and this "
          "GPU is older. Set TCPRECISION FP32.");
  }
  g_precision = precision;
  g_validate = validate;
  g_enabled = true;
}

void batched_operator_product(const float* a, const float* b, float* c,
                              int batch, int precision) {
  if (batch <= 0) return;
#if defined(NVRS_HAVE_CUTLASS)
  if (precision == kPrecTF32) {
    run_tf32(a, b, c, batch);
    return;
  }
#else
  (void)precision;
#endif
  k_batched_product<<<batch, dim3(kTile, kTile)>>>(a, b, c, batch);
}

}  // namespace tc
}  // namespace nvrs
