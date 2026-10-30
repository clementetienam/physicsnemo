// SPDX-License-Identifier: Apache-2.0
//
// tc_jacobian.cu -- batched operator product for the TENSORCORE Jacobian.
//
// Two backends, chosen at build time and at run time:
//
//   CUTLASS TF32   strided-batched GEMM, Tensor Core multiply with FP32
//                  accumulation.  Requires NVRS_HAVE_CUTLASS at build time and
//                  a device of compute capability 8.0 or newer at run time.
//   FP32 fallback  a plain tiled product on CUDA cores.  Always available, so
//                  the path is testable on any GPU and the build never depends
//                  on CUTLASS being present.
//
// The fallback is not a stub.  It computes the same product to FP32 accuracy
// and exists so that the factorisation and the scatter can be validated on
// hardware that has no TF32 datapath, which is most hardware that is not a
// datacentre GPU.
#include "tc_jacobian.hpp"

#include <cstdio>

#if defined(NVRS_HAVE_CUTLASS)
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#endif

namespace nvrs {
namespace tc {

bool cutlass_available() {
#if defined(NVRS_HAVE_CUTLASS)
  return true;
#else
  return false;
#endif
}

const char* tc_backend_name() {
#if defined(NVRS_HAVE_CUTLASS)
  return "CUTLASS TF32 Tensor Core (FP32 accumulate), FP32 fallback below SM80";
#else
  return "FP32 CUDA cores (built without CUTLASS)";
#endif
}

// --- FP32 fallback ---------------------------------------------------------
// One block per connection, one thread per output element.  The tile is 16 or
// 32 on a side, so a block is 256 or 1024 threads and the operands fit
// comfortably in shared memory.
template <int TILE>
__global__ void k_batched_product(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int batch) {
  const int q = blockIdx.x;
  if (q >= batch) return;
  const int row = threadIdx.y, col = threadIdx.x;
  const size_t off = (size_t)q * TILE * TILE;

  __shared__ float sa[TILE][TILE];
  __shared__ float sb[TILE][TILE];
  sa[row][col] = a[off + row * TILE + col];
  sb[row][col] = b[off + row * TILE + col];
  __syncthreads();

  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < TILE; ++k) acc = fmaf(sa[row][k], sb[k][col], acc);
  c[off + row * TILE + col] = acc;
}

#if defined(NVRS_HAVE_CUTLASS)
// TF32 Tensor Core multiply with FP32 accumulation.  The tile and alignment
// are sized for the 16x16 and 32x32 problems this path issues; a 128x128
// threadblock with 128-bit alignment, which is the natural choice for large
// GEMMs, cannot be instantiated for these shapes.
template <int TILE>
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

template <int TILE>
static bool run_tf32(const float* a, const float* b, float* c, int batch,
                     cudaStream_t stream) {
  using Gemm = GemmTF32<TILE>;
  constexpr int n = TILE, stride = TILE * TILE;
  typename Gemm::Arguments args({n, n, n}, {a, n}, stride, {b, n}, stride,
                                {c, n}, stride, {c, n}, stride,
                                {1.0f, 0.0f}, batch);
  Gemm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t bytes = op.get_workspace_size(args);
  void* ws = nullptr;
  if (bytes && cudaMalloc(&ws, bytes) != cudaSuccess) return false;
  bool ok = op.initialize(args, ws, stream) == cutlass::Status::kSuccess &&
            op(stream) == cutlass::Status::kSuccess;
  if (ws) cudaFree(ws);
  return ok;
}
#endif

// True when the device running this process can issue TF32 Tensor Core MMA.
static bool device_has_tf32() {
  int dev = 0, major = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) !=
      cudaSuccess)
    return false;
  return major >= 8;
}

void batched_operator_product(const float* l_flux, const float* l_props,
                              float* out, int batch, int tile,
                              bool prefer_tf32, cudaStream_t stream) {
  if (batch <= 0) return;

#if defined(NVRS_HAVE_CUTLASS)
  if (prefer_tf32 && device_has_tf32()) {
    const bool ok = (tile == kTile3) ? run_tf32<kTile3>(l_flux, l_props, out, batch, stream)
                                     : run_tf32<kTile2>(l_flux, l_props, out, batch, stream);
    if (ok) return;
    // Falling through is deliberate: a CUTLASS configuration that cannot be
    // instantiated for this shape is a build-time property, not a run-time
    // failure to hide.  Say so once, then use the FP32 path.
    static bool said = false;
    if (!said) {
      std::printf("  [warn] TENSORCORE: CUTLASS could not issue the %dx%d "
                  "batched GEMM; using the FP32 path.\n", tile, tile);
      said = true;
    }
  }
#else
  (void)prefer_tf32;
#endif

  if (tile == kTile3) {
    dim3 blk(kTile3, kTile3);
    k_batched_product<kTile3><<<batch, blk, 0, stream>>>(l_flux, l_props, out, batch);
  } else {
    dim3 blk(kTile2, kTile2);
    k_batched_product<kTile2><<<batch, blk, 0, stream>>>(l_flux, l_props, out, batch);
  }
}

}  // namespace tc
}  // namespace nvrs
