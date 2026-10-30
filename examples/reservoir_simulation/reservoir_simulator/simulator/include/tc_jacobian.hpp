// SPDX-License-Identifier: Apache-2.0
//
// tc_jacobian.hpp -- the operator-product ("Tensor Core") Jacobian assembly.
//
// Selected by JACOBIAN TENSORCORE.  Instead of computing each local connection
// block with scalar arithmetic inside the per-cell assembly kernel, this path
// writes, for every oriented connection, two padded operators
//
//     L_props = d(features) / d(primaries)
//     L_flux  = d(local residual) / d(features)
//
// into structure-of-arrays buffers, runs the whole batch as one strided GEMM,
// and scatters the leading physical block of each product into the same CSR
// the analytic path fills.  The product
//
//     J_local = L_flux * L_props
//
// is the chain rule written as a matrix product, so the arithmetic is dense
// and regular and can be issued on Tensor Cores.  The trade is extra FLOPs on
// the padding lanes for regular memory access; whether it wins is a question
// for measurement on a given device, not for argument.
//
// Block sizes.  bs = 2 is oil-water: primaries (p, Sw) per cell, four per
// connection, so the physical block is 4x4 inside a 16x16 tile.  bs = 3 is
// three-phase black oil: primaries (p, Sw, Sg-or-Rs) per cell, six per
// connection, so the physical block is 6x6 inside a 32x32 tile.  Both grid
// paths are covered: the connection list used by corner-point geometry, and
// the fixed six-neighbour stencil used by Cartesian grids.
#pragma once

#include "common.hpp"

namespace nvrs {
namespace tc {

// Tile width per block size.  Padding to a native Tensor Core tile is the
// point of the layout, not an accident of it.
constexpr int kTile2 = 16;  // bs = 2: 4 primaries, 12 physical features
constexpr int kTile3 = 32;  // bs = 3: 6 primaries, 27 physical features

constexpr int tile_for(int bs) { return bs == 3 ? kTile3 : kTile2; }

// Feature layout for bs = 2.  The first four lanes are the primaries
// themselves, then the per-side phase densities and mobilities, then the two
// phase potentials.  The remaining lanes are padding and stay zero; they exist
// so the tile matches the hardware and so the feature set can grow without
// changing the kernel shape.
enum Feature2 : int {
  F2_P_I, F2_SW_I, F2_P_J, F2_SW_J,
  F2_RHOW_I, F2_RHOO_I, F2_LW_I, F2_LO_I,
  F2_RHOW_J, F2_RHOO_J, F2_LW_J, F2_LO_J,
  F2_DPHIW, F2_DPHIO, F2_PAD0, F2_PAD1, F2_COUNT
};
static_assert(F2_COUNT == kTile2, "bs=2 feature set must fill the tile");

// Whether this binary was built against CUTLASS, and therefore whether the
// TF32 Tensor Core path exists at all.  Without it the batch still runs, as
// an FP32 product on CUDA cores, and the banner says so.
bool cutlass_available();
const char* tc_backend_name();

// Run the batched operator product for `batch` connections.  `l_flux` and
// `l_props` are device buffers of `batch * tile * tile` floats in row-major
// order with a batch stride of `tile * tile`; `out` receives the products in
// the same layout.  `prefer_tf32` selects the Tensor Core path where the
// device and the build both support it.
void batched_operator_product(const float* l_flux, const float* l_props,
                              float* out, int batch, int tile,
                              bool prefer_tf32, cudaStream_t stream);

}  // namespace tc
}  // namespace nvrs
