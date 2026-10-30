// SPDX-License-Identifier: Apache-2.0
//
// tc_jacobian.hpp -- optional Tensor Core Jacobian, built with TENSORCORE=1.
//
// The default build does not compile this path and is unchanged by it.  With
// the option on, JACOBIAN TENSORCORE forms each connection block as the chain
// rule written as a matrix product,
//
//     J_conn = L_flux * L_props,
//     L_props = d(features) / d(primaries of both cells),
//     L_flux  = d(connection residual) / d(features),
//
// padded to 16x16 tiles and issued as one strided batched GEMM.  The features
// are the eleven quantities the black-oil flux is a product of (see
// src/tc_assemble.inc), so bs = 2 and bs = 3 share one tile shape.
#pragma once

#include "common.hpp"

namespace nvrs {
namespace tc {

constexpr int kTile = 16;

// Deck TCPRECISION values.
constexpr int kPrecFP32 = 0;
constexpr int kPrecTF32 = 32;

// Turn the path on for this run.  Throws if TF32 was asked for and this binary
// or this GPU cannot do it, rather than run FP32 under a TF32 label.
void enable(int precision, int validate);
bool enabled();
extern int g_precision;
extern int g_validate;

// Fill-product-scatter seconds and, under TCVALIDATE, the measured difference
// from the analytic blocks.  Prints on the calling rank.
void report();

const char* backend_name(int precision);

// c[q] = a[q] * b[q] for `batch` row-major 16x16 tiles with stride 256.
void batched_operator_product(const float* a, const float* b, float* c,
                              int batch, int precision);

}  // namespace tc
}  // namespace nvrs
