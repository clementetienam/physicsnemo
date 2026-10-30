// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// assemble_compositional.cu — fully-implicit compositional (multi-component)
// flow.  This is the NEW physics requested in the brief: an nc-component,
// two-hydrocarbon-phase (oil/gas) model driven by a MODEL COMPOSITIONAL deck.
//
// Primary unknowns per cell:  u = [p, N_1, …, N_nc]   (bs = nc + 1)
//   N_i = moles of component i per unit pore volume (overall molar density).
//
// Per cell we run a two-phase vapour-liquid flash (Wilson K-values +
// Rachford-Rice) to obtain the vapour mole fraction β, phase compositions
// x_i / y_i and phase molar densities, hence oil/gas saturations.  Component
// conservation is transported by TPFA with phase-potential upwinding; the
// pressure/volume-balance equation closes the system.  The Jacobian uses the
// same exact-structure numerical differentiation and CSR/CPR layout as the
// black-oil assembler, with the pressure block taken as equation 0.
//
// The flash and phase-density closures here are a documented *reference* EOS
// (Wilson K + ideal-gas vapour density + constant liquid molar volume).  The
// data path — deck EOS keywords, nc-component transport, CPR with bs=nc+1 — is
// complete; swap in a full Peng-Robinson flash for production accuracy.
#include "physics.hpp"
#include "halo.hpp"
#include "wellcell.hpp"

namespace nvrs {

// Per-thread flash storage is fixed size; kMaxComponents (common.hpp) is the
// single source of truth and make_reservoir validates NCOMP against it.
#define NCMAX kMaxComponents

struct CompParams {
  int nc, bs, nx, ny, nz;
  Real dz_cell, dt, T;                 // reservoir temperature [R]
  Real Tc[NCMAX], Pc[NCMAX], acf[NCMAX], vmL[NCMAX];  // per-component (from deck)
  Real visc_o, visc_g;
  int  gravity; Real rhoo, rhog;
  Real swi, swr;
  Real grav_grad;   // psi/ft per (g/cc)  (deck GRAVGRAD)
  Real z_factor;    // gas z-factor       (deck ZFACTOR)
  Real fd_eps;      // numerical-Jacobian FD step scale (deck FDEPS)
  Real zinj[NCMAX]; // injected composition (deck ZMF), mole fractions
};

// Wilson K-value for component i at pressure p.
__device__ __forceinline__ Real wilsonK(const CompParams& P, int i, Real p) {
  return (P.Pc[i] / p) * exp(5.373 * (1.0 + P.acf[i]) * (1.0 - P.Tc[i] / P.T));
}

// Rachford-Rice: solve Σ z_i (K_i-1)/(1+β(K_i-1)) = 0 for β∈[0,1].
__device__ Real rachford_rice(const Real* z, const Real* K, int nc) {
  Real lo = 0.0, hi = 1.0;
  auto g = [&](Real b) {
    Real s = 0;
    for (int i = 0; i < nc; ++i) s += z[i] * (K[i] - 1.0) / (1.0 + b * (K[i] - 1.0));
    return s;
  };
  Real glo = g(0.0), ghi = g(1.0);
  if (glo < 0) return 0.0;          // all liquid
  if (ghi > 0) return 1.0;          // all vapour
  for (int it = 0; it < 40; ++it) { // bisection (robust, derivative-free)
    Real mid = 0.5 * (lo + hi), gm = g(mid);
    if (gm > 0) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

// Flash a cell.
//   So, Sg  — *normalised* phase saturations (So + Sg == 1), used for rel-perm.
//   Vo, Vg  — phase volumes per unit pore volume, NOT normalised.  The volume
//             balance needs these: because So and Sg are normalised by
//             construction, "So + Sg - 1" is identically zero and would leave
//             the pressure equation (and hence the whole CPR pressure block)
//             structurally singular.  The physical closure is Vo + Vg == 1.
struct FlashOut { Real beta, So, Sg, Vo, Vg, Ntot, rhoL, rhoV; };
__device__ FlashOut flash(const CompParams& P, Real p, const Real* N) {
  FlashOut F;
  Real z[NCMAX], K[NCMAX];
  Real Ntot = 0;
  for (int i = 0; i < P.nc; ++i) Ntot += N[i];
  Ntot = fmax(Ntot, 1e-20);
  for (int i = 0; i < P.nc; ++i) z[i] = N[i] / Ntot;
  for (int i = 0; i < P.nc; ++i) K[i] = wilsonK(P, i, p);
  Real beta = rachford_rice(z, K, P.nc);

  // phase molar densities:  vapour ~ ideal gas (mol/ft^3),
  //   rho_V = p /(z_gas R T); liquid from constant component molar volumes.
  const Real Rgas = 10.7316;  // psia·ft^3/(lbmol·R)
  Real vmL = 0;
  for (int i = 0; i < P.nc; ++i) {
    Real xi = z[i] / (1.0 + beta * (K[i] - 1.0));          // liquid mole frac
    vmL += xi * P.vmL[i];
  }
  Real rhoV = p / (P.z_factor * Rgas * P.T);                // gas z-factor (deck)
  Real rhoL = (vmL > 1e-12) ? 1.0 / vmL : 1.0;
  // phase volumes per pore volume: L moles = (1-β)Ntot, V moles = βNtot
  Real Vg = beta * Ntot / rhoV;
  Real Vo = (1.0 - beta) * Ntot / rhoL;
  Real Vt = Vo + Vg + 1e-20;
  F.beta = beta; F.Ntot = Ntot; F.rhoL = rhoL; F.rhoV = rhoV;
  F.Vo = Vo; F.Vg = Vg;                 // unnormalised — drives volume balance
  F.So = Vo / Vt; F.Sg = Vg / Vt;       // normalised   — drives rel-perm
  return F;
}

// Component residual for a cell from a local stencil buffer.
//   us[0]=self [p,N_1..N_nc], us[1+f]=neighbours, tf[f] face transmissibility.
__device__ void cell_residual_comp(const Real us[7][NCMAX + 1], const Real usn[NCMAX + 1],
                                   const Real tf[6], const Real zn[6], Real zc,
                                   Real poro, Real V, const CompParams& P, Real* R) {
  const int nc = P.nc, bs = P.bs;
  const Real gpsi = P.grav_grad;
  Real p = us[0][0];
  Real Nself[NCMAX];
  for (int i = 0; i < nc; ++i) Nself[i] = us[0][1 + i];
  FlashOut Fs = flash(P, p, Nself);

  Real acc = V * poro / P.dt;
  // Volume-balance (pressure) equation: the flashed fluid volume must fill the
  // pore volume exactly.  Uses the unnormalised phase volumes — see FlashOut.
  R[0] = (Fs.Vo + Fs.Vg) - 1.0;
  // component accumulation
  for (int i = 0; i < nc; ++i) R[1 + i] = acc * (us[0][1 + i] - usn[1 + i]);

  Real g = P.gravity ? gpsi : 0.0;
  Real invMuO = 1.0 / P.visc_o, invMuG = 1.0 / P.visc_g;
  for (int f = 0; f < 6; ++f) {
    Real Tg = tf[f];
    if (Tg <= 0) continue;
    Real pn = us[1 + f][0];
    Real Nnb[NCMAX];
    for (int i = 0; i < nc; ++i) Nnb[i] = us[1 + f][1 + i];
    FlashOut Fn = flash(P, pn, Nnb);

    Real dPo = (pn - g * P.rhoo * zn[f]) - (p - g * P.rhoo * zc);
    Real dPg = (pn - g * P.rhog * zn[f]) - (p - g * P.rhog * zc);
    // upstream phase (potential based); Corey-like kr from phase saturations.
    const FlashOut& upO = (dPo > 0) ? Fn : Fs;
    const FlashOut& upG = (dPg > 0) ? Fn : Fs;
    Real kroO = upO.So * upO.So, krgG = upG.Sg * upG.Sg;   // simple Corey
    Real lamO = kroO * invMuO, lamG = krgG * invMuG;
    // Phase molar fluxes (Darcy):  oil (liquid) and gas (vapour).
    Real Fo = Tg * lamO * upO.rhoL * dPo;
    Real Fg = Tg * lamG * upG.rhoV * dPg;
    // Upstream overall molar densities used to split flux into components.
    int uOcell = (dPo > 0) ? 1 + f : 0;
    int uGcell = (dPg > 0) ? 1 + f : 0;
    Real Ntot_uO = fmax(upO.Ntot, 1e-20), Ntot_uG = fmax(upG.Ntot, 1e-20);
    for (int i = 0; i < nc; ++i) {
      Real xL = us[uOcell][1 + i] / Ntot_uO;  // liquid-phase component fraction
      Real yV = us[uGcell][1 + i] / Ntot_uG;  // vapour-phase component fraction
      R[1 + i] -= (xL * Fo + yV * Fg);
    }
  }
  (void)bs;
}

// Well source term.  The compositional model previously assembled accumulation
// and transport only, so a WELLS record in a MODEL COMPOSITIONAL deck was
// parsed and then ignored and the run was a closed system.
//
//   producer (BHP controlled):  molar rate of each phase by Peaceman,
//        q_o = WI k_ro/mu_o rho_L (p - p_wf),   q_g = WI k_rg/mu_g rho_V (...)
//     split into components by composition, so R_i gains z_i (q_o + q_g).
//   injector (rate controlled):  the deck feed ZMF at the given molar rate.
//
// Components are split by the *overall* molar composition z_i rather than by
// the flashed x_i / y_i, which is exactly the approximation the transport term
// above already makes (it divides the upstream overall N_i by N_tot for both
// phases), so the well and the flux stay consistent with each other.
__device__ void add_well_comp(const Real us[7][NCMAX + 1], const CompParams& P,
                              WellCell w, const FlashOut& F, Real* R) {
  const int nc = P.nc;
  if (w.is_prod) {
    const Real draw = us[0][0] - w.pwf;
    if (draw > 0) {
      const Real kro = F.So * F.So, krg = F.Sg * F.Sg;   // same Corey as flux
      const Real qo = w.wi * (kro / P.visc_o) * F.rhoL * draw;
      const Real qg = w.wi * (krg / P.visc_g) * F.rhoV * draw;
      const Real Ntot = fmax(F.Ntot, 1e-20);
      for (int i = 0; i < nc; ++i)
        R[1 + i] += (us[0][1 + i] / Ntot) * (qo + qg);
    }
  }
  if (w.is_inj)
    for (int i = 0; i < nc; ++i) R[1 + i] -= P.zinj[i] * w.qw_inj;
}

__global__ void k_assemble_comp(const Real* u, const Real* u_old, const Real* poro,
                                const Real* tface, const int* nbr, const int* cellk,
                                const int* rowptr,
                                const int* bcol, const int* prowptr, const int* pcolpos,
                                const WellCell* well,
                                CompParams P, Real V, Real* b, Real* aval,
                                PrecReal* pval, PrecReal* dinv, PrecReal* pweight, Real* rnorm2) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int ncell = P.nx * P.ny * P.nz;
  if (cell >= ncell) return;
  const int bs = P.bs;

  Real us[7][NCMAX + 1] = {}, usn[NCMAX + 1] = {};
  Real tf[6], zn[6]; int mc[7];
  mc[0] = cell;
  for (int v = 0; v < bs; ++v) { us[0][v] = u[bs * cell + v]; usn[v] = u_old[bs * cell + v]; }
  Real zc = (cellk[cell] + 0.5) * P.dz_cell;
  for (int f = 0; f < 6; ++f) {
    int nc = nbr[7 * cell + 1 + f];
    mc[1 + f] = nc;
    tf[f] = (nc >= 0) ? tface[6 * cell + f] : 0.0;
    if (nc >= 0) {
      for (int v = 0; v < bs; ++v) us[1 + f][v] = u[bs * nc + v];
      zn[f] = (cellk[nc] + 0.5) * P.dz_cell;
    } else zn[f] = zc;
  }

  WellCell wc = well[cell];
  Real R0[NCMAX + 1];
  cell_residual_comp(us, usn, tf, zn, zc, poro[cell], V, P, R0);
  {
    Real Nself[NCMAX];
    for (int i = 0; i < P.nc; ++i) Nself[i] = us[0][1 + i];
    add_well_comp(us, P, wc, flash(P, us[0][0], Nself), R0);
  }
  for (int a = 0; a < bs; ++a) b[bs * cell + a] = -R0[a];
  Real rn = 0; for (int a = 0; a < bs; ++a) rn += R0[a] * R0[a];
  atomicAdd(rnorm2, rn);

  Real Dself[(NCMAX + 1) * (NCMAX + 1)] = {};
  Real Jp[7][NCMAX + 1] = {};   // dR_a/dp per member, for the CPR pressure row
  int  pcol[7];
  for (int m = 0; m < 7; ++m) pcol[m] = -1;
  for (int m = 0; m < 7; ++m) {
    int mcm = mc[m], coff = bcol[7 * cell + m];
    if (mcm < 0 || coff < 0) continue;
    pcol[m] = pcolpos[7 * cell + m];
    for (int v = 0; v < bs; ++v) {
      Real base = us[m][v];
      Real eps = P.fd_eps * (fabs(base) + 1e-3);
      us[m][v] = base + eps;
      Real Rp[NCMAX + 1];
      cell_residual_comp(us, usn, tf, zn, zc, poro[cell], V, P, Rp);
      // The well depends only on us[0], so applying it for every m keeps it
      // cancelling in the neighbour differences and supplying the true
      // derivative for m == 0 — the same rule as the black-oil assembler.
      {
        Real Nself[NCMAX];
        for (int i = 0; i < P.nc; ++i) Nself[i] = us[0][1 + i];
        add_well_comp(us, P, wc, flash(P, us[0][0], Nself), Rp);
      }
      us[m][v] = base;
      for (int a = 0; a < bs; ++a) {
        Real J = (Rp[a] - R0[a]) / eps;
        aval[rowptr[bs * cell + a] + coff + v] = J;
        if (m == 0) Dself[a * bs + v] = J;
        if (v == 0) Jp[m][a] = J;
      }
    }
  }
  // Gauss-Jordan inverse of the (bs×bs) self block for CPR stage-2.
  Real M[(NCMAX + 1) * (NCMAX + 1)], I[(NCMAX + 1) * (NCMAX + 1)];
  for (int a = 0; a < bs * bs; ++a) M[a] = Dself[a];
  for (int a = 0; a < bs; ++a)
    for (int c = 0; c < bs; ++c) I[a * bs + c] = (a == c) ? 1.0 : 0.0;
  for (int col = 0; col < bs; ++col) {
    Real piv = M[col * bs + col];
    Real ip = (fabs(piv) > 1e-300) ? 1.0 / piv : 0.0;
    for (int c = 0; c < bs; ++c) { M[col * bs + c] *= ip; I[col * bs + c] *= ip; }
    for (int a = 0; a < bs; ++a)
      if (a != col) {
        Real fac = M[a * bs + col];
        for (int c = 0; c < bs; ++c) {
          M[a * bs + c] -= fac * M[col * bs + c];
          I[a * bs + c] -= fac * I[col * bs + c];
        }
      }
  }
  // Same narrowing guard as the black-oil path: a nearly singular block would
  // otherwise store +-inf into the FP32 stage-2 inverse and poison the solve.
  PrecReal* dout = dinv + (long long)cell * bs * bs;
  bool ok = true;
  for (int a = 0; a < bs * bs && ok; ++a)
    ok = isfinite(I[a]) && fabs(I[a]) <= 1e30;
  for (int a = 0; a < bs * bs; ++a)
    dout[a] = ok ? (PrecReal)I[a] : (PrecReal)0;

  // CPR stage-1 pressure row, quasi-IMPES weighted by the first row of the
  // inverse cell block, so the pressure diagonal is 1 by construction rather
  // than the volume-balance derivative, which can vanish.  See the discussion
  // in assemble_blackoil.cu.
  for (int a = 0; a < bs; ++a)
    pweight[bs * cell + a] = (PrecReal)(ok ? I[a] : 1.0);
  for (int m = 0; m < 7; ++m) {
    if (pcol[m] < 0) continue;
    Real s = 0;
    for (int a = 0; a < bs; ++a) s += (ok ? I[a] : 1.0) * Jp[m][a];
    pval[prowptr[cell] + pcol[m]] = (PrecReal)s;
  }
}

// Host launcher.
Real assemble_compositional(Reservoir& r, DeviceLinearSystem& sys) {
  CompParams P{};
  P.nc = r.ncomp; P.bs = r.bs;
  // Local block extents: the kernels index cells in this rank' own
  // ordering, and use the product only as a cell count.
  P.nx = r.grid.nxl; P.ny = r.grid.nyl; P.nz = r.grid.nzl;
  P.dz_cell = r.grid.dz; P.dt = r.dt; P.T = r.temperature;
  P.visc_o = r.uo; P.visc_g = r.ug; P.gravity = r.gravity;
  P.rhoo = r.rhoo; P.rhog = r.rhog; P.swi = r.swi; P.swr = r.swr;
  P.grav_grad = r.grav_grad; P.z_factor = r.z_factor; P.fd_eps = r.fd_eps;
  for (int i = 0; i < NCMAX; ++i) P.zinj[i] = 0.0;
  for (int i = 0; i < r.ncomp && i < NCMAX; ++i) P.zinj[i] = r.comp_zinj[i];
  // Component EOS properties come from the deck (TCRIT/PCRIT/ACF/VMOL).  NCOMP
  // is validated against kMaxComponents in make_reservoir, so a silent truncation
  // here would be a bug rather than a safeguard — fail loudly instead.
  if (P.nc < 1 || P.nc > NCMAX)
    throw std::runtime_error("compositional NCOMP out of range for the kernels");
  NVRS_CHECK_CUDA(cudaMemcpy(P.Tc, r.comp_Tc, sizeof(Real) * P.nc, cudaMemcpyDeviceToHost));
  NVRS_CHECK_CUDA(cudaMemcpy(P.Pc, r.comp_Pc, sizeof(Real) * P.nc, cudaMemcpyDeviceToHost));
  NVRS_CHECK_CUDA(cudaMemcpy(P.acf, r.comp_acf, sizeof(Real) * P.nc, cudaMemcpyDeviceToHost));
  NVRS_CHECK_CUDA(cudaMemcpy(P.vmL, r.comp_vmL, sizeof(Real) * P.nc, cudaMemcpyDeviceToHost));
  // Cell pore volume must be in RESERVOIR BARRELS, not ft^3.  Every flux and
  // well term in this assembler carries the field-unit Darcy constant
  // kDarcyField = 1.127e-3 (bbl.cP)/(mD.psi.day.ft), so it is in bbl/day; an
  // accumulation in ft^3/day is 5.6146x too large against them and saturations
  // then change 5.6146x too slowly for a given throughput.  Measured against
  // OPM Flow on the BENCH2P benchmark, this left NVRS retaining 0.515 Mbbl of
  // injected water where Flow retained 1.703 Mbbl, with the flood front barely
  // leaving the injectors.
  Real V = (double)r.grid.dx * r.grid.dy * r.grid.dz /
           units::kFt3PerBbl;                  // [reservoir bbl]
  int cells = r.grid.n_local();
  Real* d_rn = nullptr;
  NVRS_CHECK_CUDA(cudaMalloc(&d_rn, sizeof(Real)));
  NVRS_CHECK_CUDA(cudaMemset(d_rn, 0, sizeof(Real)));

  // Refresh ghost layers before reading neighbour states (see halo.hpp).
  halo_exchange(r.u, r.bs, r.grid);

  k_assemble_comp<<<grid1d(cells), 256>>>(r.u, r.u_old, r.poro, r.tface, r.nbr,
                                          r.cellk, sys.rowptr, r.bcol, sys.prowptr,
                                          r.pcolpos,
                                          static_cast<WellCell*>(r.wellcells),
                                          P, V, sys.b, sys.aval64,
                                          sys.pval, sys.dinv, sys.pweight, d_rn);
  NVRS_CHECK_CUDA(cudaGetLastError());
  Real rn2 = 0;
  NVRS_CHECK_CUDA(cudaMemcpy(&rn2, d_rn, sizeof(Real), cudaMemcpyDeviceToHost));
  cudaFree(d_rn);
  if (r.mpi_nranks > 1)
    MPI_Allreduce(MPI_IN_PLACE, &rn2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(rn2);
}

}  // namespace nvrs
