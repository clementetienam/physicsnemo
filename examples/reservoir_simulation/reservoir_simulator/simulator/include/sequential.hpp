// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// sequential.hpp — the sequential (IMPES / sequential-implicit) scheme, ported
// one-for-one from the CuPy reference NVRS/NVR.py.
//
// This is the scheme that produced GPU_reservoir_simulator/RESULTS, and it is a
// different formulation from the fully-coupled FIM path in assemble_blackoil.cu.
// Each timestep does:
//
//   1. Total mobility, exactly the reference's weighting
//          Mt = Mw/Bw + Mo/Bo (1 + Rs) + Mg/Bg
//
//   2. An elliptic pressure solve.  Harmonic face transmissibilities of K*Mt on
//      the reference's *normalised* unit-cube geometry,
//          T_x = 2 (hy hz / hx) / (1/(K Mt)_i + 1/(K Mt)_j),   h = 1/N,
//      assembled as a symmetric 7-point operator whose diagonal is the sum of
//      its face transmissibilities:
//          -div(lambda_t K grad p) = q.
//      There is no accumulation term and no gravity: the pressure field is the
//      steady response to the wells at the current mobility.  Wells enter as
//      *fixed rate sources*, injectors positive and producers negative, so the
//      system is pure Neumann and singular; the reference pins it by adding
//      sum(K) at cell 0, and so do we.
//
//   3. Darcy face velocities from the pressure,  v_face = (p_i - p_j) T_face.
//
//   4. A saturation update by fractional flow.  With f_a = M_a / sum_b M_b and
//      the reference's upwind flux operator (NVR.py::GenA), water and gas are
//      advanced *independently* as two scalar systems
//          G(S) = S - S^n - (B f(S) + q_inj dt/pv + q_prod dt/pv f(S)) = 0,
//          dG   = I - B diag(df/dS),
//      Newton-solved with alpha-halving line search and joint clipping
//      (Sw in [Swi,1], Sg in [0, 1-Sw]).
//
// Why this matters: each of those systems is scalar and identity-dominant, so
// it is far better conditioned than the coupled 3x3 block system the FIM path
// builds.  That is the real reason the reference converges where a naive FIM
// implementation does not.
//
// The mixed-precision machinery is retained in full: both the pressure solve
// and each transport solve go through the same CprSolver — FP64 outer Krylov,
// FP32 BoomerAMG preconditioner, optional df64-emulated operator.  With a
// scalar system CPR degenerates to exactly what it should be, AMG on the whole
// operator plus a diagonal smoother.
#pragma once

#include "common.hpp"
#include "linear_solver.hpp"
#include "physics.hpp"

namespace nvrs {

// Workspace for the sequential scheme.  Built once, reused every timestep.
struct SeqSolver {
  DeviceLinearSystem sys;      // scalar (bs = 1) system, refilled per solve

  Real* p      = nullptr;      // cell pressure                     [cells_ext]
  Real* mt     = nullptr;      // total mobility                    [cells_ext]
  Real* tface  = nullptr;      // face transmissibility of K*Mt     [cells*6]
  Real* vface  = nullptr;      // face velocity, outward positive   [cells*6]
  Real* qtot   = nullptr;      // total well source  (inj + prod)   [cells]
  Real* qwat   = nullptr;      // water injection source            [cells]
  Real* pv     = nullptr;      // pore volume                       [cells]
  Real* s_old  = nullptr;      // saturation at the start of a step [cells]
  Real* work   = nullptr;      // Newton scratch                    [cells_ext]
  // Position of each stencil member within its CSR row, cells*7, -1 absent.
  int*  pos    = nullptr;

  // Face transmissibility geometry, on the reference's normalised unit cube.
  Real tx = 0, ty = 0, tz = 0;
};

// Allocate the workspace and precompute everything that does not change.
void seq_build(const Reservoir& r, const Deck& d, SeqSolver& S);
void seq_free(SeqSolver& S);

// Advance one timestep of length `dt` days.  Returns false if the transport
// Newton failed to converge, which the caller answers with a timestep chop.
bool seq_step(Reservoir& r, const Deck& d, SeqSolver& S, CprSolver& solver,
              Real dt, int* newton_iters, int* linear_iters, Real* residual);

}  // namespace nvrs
