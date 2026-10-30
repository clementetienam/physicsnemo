// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// assemble_blackoil.cu — fully-implicit (FIM) black-oil residual and Jacobian.
//
// Formulation
// -----------
// Primary unknowns per cell:  u = [p, Sw]      (2-phase, bs=2)
//                             u = [p, Sw, Sg]  (3-phase, bs=3)
// Component conservation (surface volumes, b_ph = 1/B_ph):
//   R_oil   = (Vφ/dt)((b_o So+Rv b_g Sg)−(…)^n) − Σ_f(b_o F_o+Rv b_g F_g)−q_o
//   R_water = (Vφ/dt)(b_w Sw − b_w Sw^n)          − Σ_f b_w F_w          − q_w
//   R_gas   = (Vφ/dt)((b_g Sg+Rs b_o So)−(…)^n)   − Σ_f(b_g F_g+Rs b_o F_o)−q_g
// TPFA phase flux  F_ph = Tface · λ_ph^up · (Φ_ph,nb − Φ_ph,cell),
// phase potential  Φ_ph = p − ρ_ph g z, single-point upstream weighting.
//
// Note what λ must and must not contain.  F above is a RESERVOIR volume rate,
// which the b_ph in front of it converts to surface volumes; so λ_ph = kr/μ,
// with no formation volume factor in it.  relperm3() in fluid.hpp returns
// kr/(μB) instead -- a surface-volume mobility -- so calling it with the deck's
// B and then multiplying by b = 1/B applies the conversion twice.  Dimensions
// make the error plain: T·M·ΔΦ is already scf/day, and scaling that by scf/rb
// leaves scf²/(rb·day) against an accumulation in scf/day.
//
// The factor is 1/B, so it hid for a long time.  Every deck in decks/ except
// Norne sets BW 1.0, BO 1.2, BG 0.005; water was exact, oil was 20% out, and
// nothing three-phase had a realistic Bg.  Norne's Bg is 8.12e-4 rb/scf, which
// makes 1/Bg = 1231 -- the gas flux came out three orders of magnitude too
// large, which is survivable while the gas sits still and is not survivable
// the moment a gas injector opens.  The B arguments below are therefore passed
// as 1, leaving relperm3 returning kr/μ and the b_ph applied once.
//
// The Jacobian is assembled by exact-structure numerical differentiation of the
// analytic per-cell residual.  Each thread copies its 7-member stencil (self +
// ≤6 face neighbours, ≤21 doubles) into registers, evaluates the base residual,
// then perturbs one stencil unknown at a time and finite-differences.  Because a
// cell residual couples only to its direct neighbours this reproduces the exact
// 7-point block-CSR Jacobian with no hand-coded derivatives.  Row 0 (the oil
// equation) is exported as the CPR pressure block.
#include "physics.hpp"
#include "halo.hpp"
#include "halo_conn.hpp"
#include "pvt.hpp"
#include "pvt_tables.hpp"
#include "wellcell.hpp"
#ifdef NVRS_HAVE_TENSORCORE
#include "tc_jacobian.hpp"
#include <chrono>
#include <stdexcept>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace nvrs {

struct BOParams {
  int bs, nx, ny, nz;
  Real dz_cell, dt;
  // Strict CNV tolerance, so the kernel can weigh the cells that violate it.
  Real cnv_tol;
  // CPR stage-1 decoupling: 0 quasi-IMPES, 1 dynamic row sum, 2 row sum.
  int  cpr_decouple;
  Real drs_eps;
  // Total compressibility [1/psi]: rock plus fluid, deck keyword CT.
  //
  // With PVTCONST the formation volume factors are constants, so without this
  // the accumulation term has no pressure derivative at all and the system is
  // exactly incompressible.  That is not merely stiff, it is unsolvable for a
  // field under depletion: producing from a closed incompressible reservoir
  // with no injection has no solution, and the pressure operator degenerates
  // to a singular Neumann Laplacian -- |diagonal| equal to the sum of the
  // off-diagonals in every row, with the constant vector in its null space.
  // A Krylov method stalls on that at a fixed relative residual however many
  // iterations it is given, and Newton fails at every timestep however small.
  Real ct;
  int have_rock;
  Real rock_pref;
  Real uw, uo, ug, bw, bo, bg, rs, swi, swr;
  Real rhow, rhoo, rhog; int gravity;
  // PVTNUM: per-region tables and each cell's region.  Null means one
  // region for the whole field, which is `pvt` below.
  const DevicePvt* pvt_regions; const int* pvt_cell;
  Real grav_grad;   // psi/ft per (g/cc)  (deck GRAVGRAD)
  Real fd_eps;      // numerical-Jacobian FD step scale (deck FDEPS)

  // Tabulated PVT, when the deck supplies it.  When it does, the b-factors
  // above are unused and ct falls back to being rock compressibility alone:
  // the fluid's share of it now comes from db/dp, evaluated per cell, which is
  // both larger and pressure-dependent.  Norne's gas at 4200 psi has a
  // compressibility of 1.8e-4 /psi against the 2e-5 the lumped constant was
  // standing in with, and it is that difference a gas injector runs into.
  DevicePvt pvt;
  Real rs_max;      // dissolved gas an undersaturated oil retains
  // Per-cell version of the same, from RSVD, or null for the scalar.  The
  // anchor VARIES with depth and equilibration region -- Norne's five RSVD
  // tables span 675 to 531 scf/stb -- and a single number puts the crest value
  // over the whole field, giving every cell in the leaner regions 27% more
  // dissolved gas than the deck states.
  const Real* rs_max_cell;
  // DRSDT: the current per-cell ceiling on dissolved gas, walked down after
  // every accepted step.  Takes precedence over rs_max_cell when present.
  const Real* rs_cap;
  const Real* so_max;
  const Real* gas_primary;
  const Real* gas_primary_old;
  Real vap1;
};

// Residual of one cell from a local stencil buffer.
//   us[0]   = self [p,Sw,Sg]     usn = self^n
//   us[1+f] = neighbour f vars   (f=0..5 → x-,x+,y-,y+,z-,z+)
//   tf[f]   = face transmissibility (0 if no neighbour)
//   zc, zn[f] = cell-centre depths
__device__ void cell_residual_bo(const Real us[7][3], const Real usn[3],
                                 const Real tf[6], const Real zn[6], Real zc,
                                 Real poro, Real V, const SatTables& T,
                                 const BOParams& P, Real* R) {
  const int bs = P.bs;
  const Real gpsi = P.grav_grad;            // psi/ft per g/cc (deck GRAVGRAD)
  Real p = us[0][0], Sw = us[0][1], Sg = (bs == 3) ? us[0][2] : 0.0;
  Real So = 1.0 - Sw - Sg;
  Real Swn = usn[1], Sgn = (bs == 3) ? usn[2] : 0.0, Son = 1.0 - Swn - Sgn;
  Real bw = 1.0 / P.bw, bo = 1.0 / P.bo, bg = 1.0 / P.bg, rs = P.rs;

  Real acc = V * poro / P.dt;
  R[0] = acc * (bo * So - bo * Son);                              // oil
  R[1] = acc * (bw * Sw - bw * Swn);                              // water
  if (bs == 3)
    R[2] = acc * ((bg * Sg + rs * bo * So) - (bg * Sgn + rs * bo * Son));

  Real g = P.gravity ? gpsi : 0.0;
  for (int f = 0; f < 6; ++f) {
    Real Tg = tf[f];
    if (Tg <= 0) continue;
    Real pn = us[1 + f][0], Swnb = us[1 + f][1],
         Sgnb = (bs == 3) ? us[1 + f][2] : 0.0;
    Real dPw = (pn - g * P.rhow * zn[f]) - (p - g * P.rhow * zc);
    Real dPo = (pn - g * P.rhoo * zn[f]) - (p - g * P.rhoo * zc);
    Real dPg = (pn - g * P.rhog * zn[f]) - (p - g * P.rhog * zc);
    // upstream saturations (per phase potential)
    Real SwU = (dPw > 0) ? Swnb : Sw, SgU = (dPg > 0) ? Sgnb : Sg;
    Real SwOu = (dPo > 0) ? Swnb : Sw, SgOu = (dPo > 0) ? Sgnb : Sg;
    Real Mw, Mo, Mg;
    if (bs == 2) {
      Mw = relperm2_table(SwU, T, P.uw, P.uo, 1.0, 1.0,
                          P.swi, P.swr).Mw;
      Mo = relperm2_table(SwOu, T, P.uw, P.uo, 1.0, 1.0,
                          P.swi, P.swr).Mo;
      Mg = 0;
    } else {
      Mw = relperm3(SwU, SgU, T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr).Mw;
      Mo = relperm3(SwOu, SgOu, T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr).Mo;
      Mg = relperm3(SwU, SgU, T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr).Mg;
    }
    Real Fw = Tg * Mw * dPw, Fo = Tg * Mo * dPo, Fg = Tg * Mg * dPg;
    R[0] -= bo * Fo;
    R[1] -= bw * Fw;
    if (bs == 3) R[2] -= (bg * Fg + rs * bo * Fo);
  }
}

// Add well contribution (depends on self only) to residual R.
__device__ void add_well_bo(const Real us[7][3], const SatTables& T,
                            const BOParams& P, WellCell w, Real* R) {
  const int bs = P.bs;
  Real p = us[0][0], Sw = us[0][1], Sg = (bs == 3) ? us[0][2] : 0.0;
  Real bw = 1.0 / P.bw, bo = 1.0 / P.bo, bg = 1.0 / P.bg, rs = P.rs;
  if (w.is_prod) {
    Real draw = p - w.pwf;
    Real Mw, Mo, Mg = 0;
    if (bs == 2) {
      Mob2 m = relperm2_table(Sw, T, P.uw, P.uo, 1.0, 1.0,
                              P.swi, P.swr);
      Mw = m.Mw; Mo = m.Mo;
    } else {
      Mob3 m = relperm3(Sw, Sg, T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr);
      Mw = m.Mw; Mo = m.Mo; Mg = m.Mg;
    }
    R[0] += bo * w.wi * Mo * draw;
    R[1] += bw * w.wi * Mw * draw;
    if (bs == 3) R[2] += (bg * w.wi * Mg + rs * bo * w.wi * Mo) * draw;
  }
  if (w.is_inj) R[1] -= w.qw_inj;  // water injection into water eq
}

// Invert a bs×bs matrix (bs≤3) in registers → out (row-major).  Used for the
// CPR stage-2 block-Jacobi diagonal inverse.
// Narrow a double to the preconditioner precision, refusing anything the
// target type cannot hold.  The old code guarded the determinant against
// 1e-300 — the *double* underflow limit — and then cast the result to float.
// A block with det ~ 1e-300 therefore produced entries of ~1e300, which become
// +-inf as soon as they are stored as FP32, and the stage-2 smoother turns
// those into NaN across the whole solve.  A cell whose diagonal block is that
// close to singular contributes nothing useful anyway, so drop it: CPR stage 2
// is a correction, and zero there simply leaves that cell to the AMG stage.
// The PVT tables governing one cell.
//
// PVTNUM is a per-cell region index; without it every cell was evaluated
// against region 1, which on a two-region deck like Norne is the wrong oil
// compressibility, viscosity and dissolved gas everywhere outside it.
__device__ __forceinline__ const DevicePvt& pvt_of(const BOParams& P, int c) {
  if (P.pvt_regions && P.pvt_cell) {
    const int r = P.pvt_cell[c];
    if (r >= 0) return P.pvt_regions[r];
  }
  return P.pvt;
}

__device__ __forceinline__ bool narrow_ok(Real v, PrecReal* dst) {
  // 1e30 is comfortably inside the FP32 range (max ~3.4e38) and far beyond any
  // physically meaningful Jacobian entry.
  if (!isfinite(v) || fabs(v) > 1e30) return false;
  *dst = (PrecReal)v;
  return true;
}

// Invert the bs x bs cell block in full precision.  Returns false when the
// block is too close to singular to be useful.
__device__ bool invert_block(const Real* A, int bs, Real* inv) {
  Real det;
  if (bs == 2) {
    det = A[0] * A[3] - A[1] * A[2];
    if (!(fabs(det) > 1e-300)) return false;
    const Real id = 1.0 / det;
    inv[0] =  A[3] * id;  inv[1] = -A[1] * id;
    inv[2] = -A[2] * id;  inv[3] =  A[0] * id;
  } else {  // bs == 3, cofactor inverse
    const Real a = A[0], b = A[1], c = A[2], d = A[3], e = A[4], f = A[5],
               g = A[6], h = A[7], i = A[8];
    const Real A00 = e * i - f * h, A01 = -(d * i - f * g), A02 = d * h - e * g;
    det = a * A00 + b * A01 + c * A02;
    if (!(fabs(det) > 1e-300)) return false;
    const Real id = 1.0 / det;
    inv[0] = A00 * id;
    inv[1] = -(b * i - c * h) * id;
    inv[2] = (b * f - c * e) * id;
    inv[3] = A01 * id;
    inv[4] = (a * i - c * g) * id;
    inv[5] = -(a * f - c * d) * id;
    inv[6] = A02 * id;
    inv[7] = -(a * h - b * g) * id;
    inv[8] = (a * e - b * d) * id;
  }
  for (int k = 0; k < bs * bs; ++k)
    if (!isfinite(inv[k])) return false;
  return true;
}

// Store the block inverse for the CPR stage-2 smoother, narrowing to the
// preconditioner precision.  Anything the target type cannot hold drops the
// block to zero, which merely leaves that cell to the AMG stage; storing +-inf
// instead would poison the whole solve.
__device__ void store_dinv(const Real* inv, bool ok, PrecReal* out, int bs) {
  const int n2 = bs * bs;
  bool good = ok;
  for (int k = 0; k < n2 && good; ++k) good = narrow_ok(inv[k], &out[k]);
  if (!good)
    for (int k = 0; k < n2; ++k) out[k] = (PrecReal)0;
}

// ---------------------------------------------------------------------------
// CPR stage-1 pressure row: quasi-IMPES weighting.
//
// The pressure operator handed to BoomerAMG used to be the raw oil-equation
// pressure derivative, dR_oil/dp.  That entry is proportional to the oil
// mobility, so as soon as a cell runs out of mobile oil (So -> 0 after a Newton
// update, which the saturation clamps make easy to reach) its *diagonal* goes
// to zero.  AMG then smooths against a zero pivot and returns NaN — observed as
// 40 zero entries appearing in pval at Newton 1, immediately followed by a NaN
// linear residual.
//
// The standard remedy is to build the pressure equation from a weighted
// combination of all the component equations rather than from one of them.
// Taking the weights as the first row of the inverse cell block (quasi-IMPES)
// gives, for the diagonal member,
//
//     sum_a (D^-1)[0][a] * D[a][0] = (D^-1 D)[0][0] = 1,
//
// so every diagonal is exactly one by construction and can never vanish,
// whatever the local phase state. Off-diagonals become the corresponding
// weighted couplings.
//
//   w    : first row of the inverse cell block (length bs)
//   Jp   : dR_a/dp for this stencil member (length bs)
// ---------------------------------------------------------------------------
__device__ __forceinline__ Real pressure_row_entry(const Real* w, const Real* Jp,
                                                   int bs) {
  Real s = 0;
  for (int a = 0; a < bs; ++a) s += w[a] * Jp[a];
  return s;
}

// Weights to use when the cell block could not be inverted.  Summing the
// equations is far more robust than picking one of them, and this path is rare.
__device__ __forceinline__ void fallback_weights(Real* w, int bs) {
  for (int a = 0; a < bs; ++a) w[a] = 1.0;
}

// Assemble kernel.  CSR scatter uses:
//   rowptr[bs*cell+a]         : start of scalar row (a = component equation)
//   bcol[7*cell+m]            : scalar column offset (within row) of block m,
//                              = (rank of neighbour m among sorted cols) * bs; -1 absent
//   prowptr[cell], pcolpos[7*cell+m] : pressure sub-block (row=cell) layout
// Entry (row a, block m, var v)  →  aval[rowptr[bs*cell+a] + bcol[7*cell+m] + v]
__global__ void k_assemble_bo(const Real* u, const Real* u_old, const Real* poro,
                              const Real* tface, const int* nbr, const int* cellk,
                              const int* rowptr,
                              const int* bcol, const int* prowptr, const int* pcolpos,
                              const WellCell* well, SatTables T, BOParams P, Real V,
                              Real* b, Real* aval, PrecReal* pval, PrecReal* dinv, PrecReal* pweight,
                              Real* rnorm2) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int ncell = P.nx * P.ny * P.nz;
  if (cell >= ncell) return;
  const int bs = P.bs;

  Real us[7][3] = {}, usn[3] = {};
  Real tf[6], zn[6];
  int  mc[7];
  mc[0] = cell;
  for (int v = 0; v < bs; ++v) { us[0][v] = u[bs * cell + v]; usn[v] = u_old[bs * cell + v]; }
  // Depth is measured from the *global* datum via cellk, so gravity does not
  // depend on how the grid was decomposed across ranks.
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

  WellCell w = well[cell];
  Real R0[3];
  cell_residual_bo(us, usn, tf, zn, zc, poro[cell], V, T, P, R0);
  add_well_bo(us, T, P, w, R0);

  for (int a = 0; a < bs; ++a) b[bs * cell + a] = -R0[a];
  Real rn = 0; for (int a = 0; a < bs; ++a) rn += R0[a] * R0[a];
  atomicAdd(rnorm2, rn);

  Real Dself[9] = {};  // bs×bs self diagonal block (row-major) for CPR inverse
  Real Jp[7][3] = {};  // dR_a/dp per stencil member, for the CPR pressure row
  int  pcol[7];
  for (int m = 0; m < 7; ++m) pcol[m] = -1;
  for (int m = 0; m < 7; ++m) {
    int mcm = mc[m];
    int coff = bcol[7 * cell + m];
    if (mcm < 0 || coff < 0) continue;
    int pcp = pcolpos[7 * cell + m];   // pressure-block column position
    pcol[m] = pcp;
    for (int v = 0; v < bs; ++v) {
      Real base = us[m][v];
      Real eps = P.fd_eps * (fabs(base) + 1e-3);
      us[m][v] = base + eps;
      Real Rp[3];
      cell_residual_bo(us, usn, tf, zn, zc, poro[cell], V, T, P, Rp);
      // The well term MUST be included for every stencil member, not just the
      // self block.  R0 contains it, so omitting it from Rp leaves a spurious
      // -Q_well/eps in every neighbour entry of the Jacobian.  With the gas
      // equation's Q_well ~ 4e6 and an Sg perturbation of eps ~ 2e-9 that is an
      // entry of order 1e15, which is exactly what the analytic-vs-numerical
      // cross-check reported.  add_well_bo depends only on us[0], so for m != 0
      // it contributes the same constant as in R0 and cancels in the
      // difference; for m == 0 it supplies the true derivative.
      add_well_bo(us, T, P, w, Rp);
      us[m][v] = base;
      for (int a = 0; a < bs; ++a) {
        Real J = (Rp[a] - R0[a]) / eps;
        aval[rowptr[bs * cell + a] + coff + v] = J;
        if (m == 0) Dself[a * bs + v] = J;
        if (v == 0) Jp[m][a] = J;        // ∂R_a/∂p of block m
      }
    }
  }

  // CPR stage-2 block-diagonal inverse, and the quasi-IMPES pressure row that
  // reuses the same inverse (see pressure_row_entry).
  Real Inv[9];
  const bool ok = invert_block(Dself, bs, Inv);
  store_dinv(Inv, ok, dinv + (long long)cell * bs * bs, bs);

  Real wgt[3];
  if (ok) { for (int a = 0; a < bs; ++a) wgt[a] = Inv[a]; }
  else    { fallback_weights(wgt, bs); }
  for (int a = 0; a < bs; ++a) pweight[bs * cell + a] = (PrecReal)wgt[a];
  for (int m = 0; m < 7; ++m)
    if (pcol[m] >= 0)
      pval[prowptr[cell] + pcol[m]] =
          (PrecReal)pressure_row_entry(wgt, Jp[m], bs);
}

// ===========================================================================
// Analytic Jacobian — exact derivatives for the two-phase (oil/water) black-oil
// model (bs = 2).  Same CSR/pressure/dinv layout as the numerical kernel, so it
// is a drop-in replacement selected by JACOBIAN ANALYTIC.  Derived term by term
// from the residual (accumulation, TPFA upstream flux, Peaceman well):
//   ∂R_oil/∂p_i = +b_o Σ_f T_f M_o^up ;  ∂R_oil/∂p_j = −b_o T_f M_o^up
//   ∂R_oil/∂Sw^up = −b_o T_f ΔΦ_o dM_o(S^up)   (to self or neighbour)
// and the water row analogously; PVT (b_o,b_w) frozen within the Newton step.
// ===========================================================================
__global__ void k_assemble_bo_analytic(const Real* u, const Real* u_old,
                                       const Real* poro, const Real* tface,
                                       const int* nbr, const int* cellk,
                                       const int* rowptr,
                                       const int* bcol, const int* prowptr,
                                       const int* pcolpos, const WellCell* well,
                                       const Real* aimflag,
                                       SatTables T, BOParams P, Real V, Real* b,
                                       Real* aval, PrecReal* pval, PrecReal* dinv, PrecReal* pweight,
                                       Real* rnorm2) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int ncell = P.nx * P.ny * P.nz;
  if (cell >= ncell) return;
  // This exact path covers bs == 2 (bs == 3 handled by k_assemble_bo3_analytic).
  const Real gpsi = P.grav_grad;
  const Real bo = 1.0 / P.bo, bw = 1.0 / P.bw;

  int mc[7];
  Real us[7][2] = {}, usn[2] = {};
  Real tf[6], zn[6];
  // AIM: a face's mobility comes from its donor (upstream) cell; when the
  // donor is flagged explicit its OLD saturation is used and no saturation
  // derivative is written, so the flux is linear in the implicit unknowns.
  Real uso_n[6] = {};
  bool exn[6] = {};
  const bool ex_i = aimflag[cell] < 0.5;
  mc[0] = cell;
  us[0][0] = u[2 * cell]; us[0][1] = u[2 * cell + 1];
  usn[0] = u_old[2 * cell]; usn[1] = u_old[2 * cell + 1];
  Real zc = (cellk[cell] + 0.5) * P.dz_cell;
  for (int f = 0; f < 6; ++f) {
    int nc = nbr[7 * cell + 1 + f];
    mc[1 + f] = nc;
    tf[f] = (nc >= 0) ? tface[6 * cell + f] : 0.0;
    if (nc >= 0) {
      us[1 + f][0] = u[2 * nc]; us[1 + f][1] = u[2 * nc + 1];
      zn[f] = (cellk[nc] + 0.5) * P.dz_cell;
      uso_n[f] = u_old[2 * nc + 1];
      exn[f] = aimflag[nc] < 0.5;
    } else zn[f] = zc;
  }

  Real p_i = us[0][0], Sw_i = us[0][1];
  Real So_i = 1.0 - Sw_i, So_n = 1.0 - usn[1];
  Real acc = V * poro[cell] / P.dt;

  // --- residual (for rhs and norm) ---
  Real R_oil = acc * (bo * So_i - bo * So_n);
  Real R_wat = acc * (bw * Sw_i - bw * usn[1]);

  // --- Jacobian accumulators ---
  Real sJ[2][2] = {};          // self block  ∂R_i/∂u_i
  Real nJ[6][2][2] = {};       // neighbour blocks ∂R_i/∂u_j
  // accumulation contributions (self only)
  sJ[0][1] += acc * bo * (-1.0);   // ∂R_oil/∂Sw
  sJ[1][1] += acc * bw;            // ∂R_wat/∂Sw

  Real g = P.gravity ? gpsi : 0.0;
  for (int f = 0; f < 6; ++f) {
    Real Tg = tf[f];
    if (Tg <= 0) continue;
    Real p_j = us[1 + f][0], Sw_j = us[1 + f][1];
    Real dPo = (p_j - g * P.rhoo * zn[f]) - (p_i - g * P.rhoo * zc);
    Real dPw = (p_j - g * P.rhow * zn[f]) - (p_i - g * P.rhow * zc);
    bool upo_j = dPo > 0.0;      // oil upstream is neighbour?
    bool upw_j = dPw > 0.0;
    // AIM: explicit donor -> old saturation, and no dM term below.
    const bool ex_o = upo_j ? exn[f] : ex_i;
    const bool ex_w = upw_j ? exn[f] : ex_i;
    Real Sw_uo = upo_j ? (ex_o ? uso_n[f] : Sw_j) : (ex_o ? usn[1] : Sw_i);
    Real Sw_uw = upw_j ? (ex_w ? uso_n[f] : Sw_j) : (ex_w ? usn[1] : Sw_i);
    Mob2 mo = relperm2_table(Sw_uo, T, P.uw, P.uo, 1.0, 1.0,
                             P.swi, P.swr);
    Mob2 mw = relperm2_table(Sw_uw, T, P.uw, P.uo, 1.0, 1.0,
                             P.swi, P.swr);

    Real Fo = Tg * mo.Mo * dPo, Fw = Tg * mw.Mw * dPw;
    R_oil -= bo * Fo;
    R_wat -= bw * Fw;

    // pressure derivatives (mobility independent of p)
    sJ[0][0] += bo * Tg * mo.Mo;         // ∂R_oil/∂p_i
    nJ[f][0][0] -= bo * Tg * mo.Mo;      // ∂R_oil/∂p_j
    sJ[1][0] += bw * Tg * mw.Mw;         // ∂R_wat/∂p_i
    nJ[f][1][0] -= bw * Tg * mw.Mw;      // ∂R_wat/∂p_j
    // saturation derivatives (only the upstream cell, and only if implicit)
    Real dOil_dSwUp = -bo * Tg * dPo * mo.dMo;
    Real dWat_dSwUp = -bw * Tg * dPw * mw.dMw;
    if (!ex_o) { if (upo_j) nJ[f][0][1] += dOil_dSwUp; else sJ[0][1] += dOil_dSwUp; }
    if (!ex_w) { if (upw_j) nJ[f][1][1] += dWat_dSwUp; else sJ[1][1] += dWat_dSwUp; }
  }

  // --- wells (self) ---
  WellCell w = well[cell];
  if (w.is_prod) {
    Real draw = p_i - w.pwf;
    Mob2 m = relperm2_table(Sw_i, T, P.uw, P.uo, 1.0, 1.0,
                            P.swi, P.swr);
    R_oil += bo * w.wi * m.Mo * draw;
    R_wat += bw * w.wi * m.Mw * draw;
    sJ[0][0] += bo * w.wi * m.Mo;
    sJ[0][1] += bo * w.wi * m.dMo * draw;
    sJ[1][0] += bw * w.wi * m.Mw;
    sJ[1][1] += bw * w.wi * m.dMw * draw;
  }
  if (w.is_inj) R_wat -= w.qw_inj;

  // --- write rhs and norm ---
  b[2 * cell] = -R_oil;
  b[2 * cell + 1] = -R_wat;
  atomicAdd(rnorm2, R_oil * R_oil + R_wat * R_wat);

  // --- CPR: stage-2 block inverse and quasi-IMPES pressure weights ---
  Real flat[4] = {sJ[0][0], sJ[0][1], sJ[1][0], sJ[1][1]};
  Real Inv[9];
  const bool ok = invert_block(flat, 2, Inv);
  store_dinv(Inv, ok, dinv + (long long)cell * 4, 2);
  Real wgt[3];
  if (ok) { wgt[0] = Inv[0]; wgt[1] = Inv[1]; }
  else    { fallback_weights(wgt, 2); }
  pweight[2 * cell + 0] = (PrecReal)wgt[0];
  pweight[2 * cell + 1] = (PrecReal)wgt[1];

  // --- scatter blocks into CSR ---
  auto scatter = [&](int m, const Real J[2][2]) {
    int coff = bcol[7 * cell + m];
    if (coff < 0) return;
    for (int a = 0; a < 2; ++a)
      for (int v = 0; v < 2; ++v)
        aval[rowptr[2 * cell + a] + coff + v] = J[a][v];
    int pcp = pcolpos[7 * cell + m];
    if (pcp >= 0) {
      Real Jp[2] = {J[0][0], J[1][0]};
      pval[prowptr[cell] + pcp] = (PrecReal)pressure_row_entry(wgt, Jp, 2);
    }
  };
  scatter(0, sJ);
  for (int f = 0; f < 6; ++f)
    if (mc[1 + f] >= 0) scatter(1 + f, nJ[f]);
}

// ===========================================================================
// Analytic Jacobian — exact derivatives for THREE-phase black oil (bs = 3),
// primary vars [p, Sw, Sg], equations [oil, water, gas].  Phase mobilities and
// their partials come from relperm3 (OPM-default three-phase KRO); each phase flux is
// differentiated w.r.t. its own phase-potential-upstream cell, so saturation
// derivatives route to self or neighbour independently per phase — matching the
// numerical residual term-for-term (validation cross-check).
// ===========================================================================
__global__ void k_assemble_bo3_analytic(const Real* u, const Real* u_old,
                                        const Real* poro, const Real* tface,
                                        const int* nbr, const int* cellk,
                                        const int* rowptr,
                                        const int* bcol, const int* prowptr,
                                        const int* pcolpos, const WellCell* well,
                                        const Real* aimflag,
                                        SatTables T, BOParams P, Real V, Real* b,
                                        Real* aval, PrecReal* pval, PrecReal* dinv, PrecReal* pweight,
                                        Real* rnorm2) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int ncell = P.nx * P.ny * P.nz;
  if (cell >= ncell) return;
  const Real gpsi = P.grav_grad;
  const Real bo = 1.0 / P.bo, bw = 1.0 / P.bw, bg = 1.0 / P.bg, rs = P.rs;

  int mc[7];
  Real us[7][3] = {}, usn[3] = {};
  Real tf[6], zn[6];
  // AIM donor data: neighbours' old (Sw, Sg) and explicit flags; see the
  // two-phase kernel for the rule.
  Real uso_n[6][2] = {};
  bool exn[6] = {};
  const bool ex_i = aimflag[cell] < 0.5;
  mc[0] = cell;
  for (int v = 0; v < 3; ++v) { us[0][v] = u[3 * cell + v]; usn[v] = u_old[3 * cell + v]; }
  Real zc = (cellk[cell] + 0.5) * P.dz_cell;
  for (int f = 0; f < 6; ++f) {
    int nc = nbr[7 * cell + 1 + f];
    mc[1 + f] = nc;
    tf[f] = (nc >= 0) ? tface[6 * cell + f] : 0.0;
    if (nc >= 0) {
      for (int v = 0; v < 3; ++v) us[1 + f][v] = u[3 * nc + v];
      zn[f] = (cellk[nc] + 0.5) * P.dz_cell;
      uso_n[f][0] = u_old[3 * nc + 1];
      uso_n[f][1] = u_old[3 * nc + 2];
      exn[f] = aimflag[nc] < 0.5;
    } else zn[f] = zc;
  }

  Real p_i = us[0][0], Sw_i = us[0][1], Sg_i = us[0][2];
  Real So_i = 1.0 - Sw_i - Sg_i;
  Real So_n = 1.0 - usn[1] - usn[2];
  Real acc = V * poro[cell] / P.dt;

  Real R_oil = acc * (bo * So_i - bo * So_n);
  Real R_wat = acc * (bw * Sw_i - bw * usn[1]);
  Real R_gas = acc * ((bg * Sg_i + rs * bo * So_i) - (bg * usn[2] + rs * bo * So_n));

  Real sJ[3][3] = {}, nJ[6][3][3] = {};
  // accumulation (self)
  sJ[0][1] += -acc * bo;          sJ[0][2] += -acc * bo;      // ∂R_oil/∂Sw,∂Sg
  sJ[1][1] += acc * bw;                                        // ∂R_wat/∂Sw
  sJ[2][1] += -acc * rs * bo;     sJ[2][2] += acc * (bg - rs * bo);

  Real g = P.gravity ? gpsi : 0.0;
  for (int f = 0; f < 6; ++f) {
    Real Tg = tf[f];
    if (Tg <= 0) continue;
    Real p_j = us[1 + f][0], Sw_j = us[1 + f][1], Sg_j = us[1 + f][2];
    Real dPw = (p_j - g * P.rhow * zn[f]) - (p_i - g * P.rhow * zc);
    Real dPo = (p_j - g * P.rhoo * zn[f]) - (p_i - g * P.rhoo * zc);
    Real dPg = (p_j - g * P.rhog * zn[f]) - (p_i - g * P.rhog * zc);
    bool wj = dPw > 0.0, oj = dPo > 0.0, gj = dPg > 0.0;   // upstream=neighbour?
    // AIM: each phase's donor decides; an explicit donor contributes its old
    // saturations and no derivative for that phase's mobility.
    const bool ex_w = wj ? exn[f] : ex_i;
    const bool ex_o = oj ? exn[f] : ex_i;
    const bool ex_g = gj ? exn[f] : ex_i;
    Real SwU = wj ? (ex_w ? uso_n[f][0] : Sw_j) : (ex_w ? usn[1] : Sw_i);
    Real SgU = gj ? (ex_g ? uso_n[f][1] : Sg_j) : (ex_g ? usn[2] : Sg_i);
    Real SwO = oj ? (ex_o ? uso_n[f][0] : Sw_j) : (ex_o ? usn[1] : Sw_i);
    Real SgO = oj ? (ex_o ? uso_n[f][1] : Sg_j) : (ex_o ? usn[2] : Sg_i);
    Mob3 A = relperm3(SwU, SgU, T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr);
    Mob3 B = relperm3(SwO, SgO, T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr);
    Real Fw = Tg * A.Mw * dPw, Fo = Tg * B.Mo * dPo, Fg = Tg * A.Mg * dPg;
    R_oil -= bo * Fo;
    R_wat -= bw * Fw;
    R_gas -= (bg * Fg + rs * bo * Fo);

    // --- oil equation ---
    sJ[0][0] += bo * Tg * B.Mo;  nJ[f][0][0] -= bo * Tg * B.Mo;
    Real dOilSw = -bo * Tg * dPo * B.dMo_dSw, dOilSg = -bo * Tg * dPo * B.dMo_dSg;
    if (!ex_o) {
      (oj ? nJ[f][0][1] : sJ[0][1]) += dOilSw;
      (oj ? nJ[f][0][2] : sJ[0][2]) += dOilSg;
    }
    // --- water equation ---
    sJ[1][0] += bw * Tg * A.Mw;  nJ[f][1][0] -= bw * Tg * A.Mw;
    Real dWatSw = -bw * Tg * dPw * A.dMw_dSw;
    if (!ex_w) (wj ? nJ[f][1][1] : sJ[1][1]) += dWatSw;
    // --- gas equation:  bg*Fg + rs*bo*Fo ---
    sJ[2][0] += bg * Tg * A.Mg + rs * bo * Tg * B.Mo;
    nJ[f][2][0] -= bg * Tg * A.Mg + rs * bo * Tg * B.Mo;
    Real dGasSg_g = -bg * Tg * dPg * A.dMg_dSg;         // from Fg
    if (!ex_g) (gj ? nJ[f][2][2] : sJ[2][2]) += dGasSg_g;
    Real dGasSw_o = -rs * bo * Tg * dPo * B.dMo_dSw;    // from rs*bo*Fo
    Real dGasSg_o = -rs * bo * Tg * dPo * B.dMo_dSg;
    if (!ex_o) {
      (oj ? nJ[f][2][1] : sJ[2][1]) += dGasSw_o;
      (oj ? nJ[f][2][2] : sJ[2][2]) += dGasSg_o;
    }
  }

  // --- wells (self) ---
  WellCell w = well[cell];
  if (w.is_prod) {
    Real draw = p_i - w.pwf;
    Mob3 m = relperm3(Sw_i, Sg_i, T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr);
    R_oil += bo * w.wi * m.Mo * draw;
    R_wat += bw * w.wi * m.Mw * draw;
    R_gas += (bg * w.wi * m.Mg + rs * bo * w.wi * m.Mo) * draw;
    sJ[0][0] += bo * w.wi * m.Mo;
    sJ[0][1] += bo * w.wi * m.dMo_dSw * draw;
    sJ[0][2] += bo * w.wi * m.dMo_dSg * draw;
    sJ[1][0] += bw * w.wi * m.Mw;
    sJ[1][1] += bw * w.wi * m.dMw_dSw * draw;
    sJ[2][0] += bg * w.wi * m.Mg + rs * bo * w.wi * m.Mo;
    sJ[2][1] += rs * bo * w.wi * m.dMo_dSw * draw;
    sJ[2][2] += (bg * w.wi * m.dMg_dSg + rs * bo * w.wi * m.dMo_dSg) * draw;
  }
  // Injection enters the equation of the phase being injected.  A rate-
  // controlled injector is a pure source: it contributes no derivative, so
  // nothing is added to the Jacobian here.
  if (w.is_inj) {
    R_wat -= w.qw_inj;
    R_oil -= w.qo_inj;
    R_gas -= w.qg_inj;
  }

  b[3 * cell] = -R_oil; b[3 * cell + 1] = -R_wat; b[3 * cell + 2] = -R_gas;
  atomicAdd(rnorm2, R_oil * R_oil + R_wat * R_wat + R_gas * R_gas);

  // --- CPR: stage-2 block inverse and quasi-IMPES pressure weights ---
  Real flat[9] = {sJ[0][0], sJ[0][1], sJ[0][2], sJ[1][0], sJ[1][1],
                  sJ[1][2], sJ[2][0], sJ[2][1], sJ[2][2]};
  Real Inv[9];
  const bool ok = invert_block(flat, 3, Inv);
  store_dinv(Inv, ok, dinv + (long long)cell * 9, 3);
  Real wgt[3];
  if (ok) { wgt[0] = Inv[0]; wgt[1] = Inv[1]; wgt[2] = Inv[2]; }
  else    { fallback_weights(wgt, 3); }
  for (int a = 0; a < 3; ++a) pweight[3 * cell + a] = (PrecReal)wgt[a];

  auto scatter3 = [&](int m, const Real J[3][3]) {
    int coff = bcol[7 * cell + m];
    if (coff < 0) return;
    for (int a = 0; a < 3; ++a)
      for (int v = 0; v < 3; ++v)
        aval[rowptr[3 * cell + a] + coff + v] = J[a][v];
    int pcp = pcolpos[7 * cell + m];
    if (pcp >= 0) {
      Real Jp[3] = {J[0][0], J[1][0], J[2][0]};
      pval[prowptr[cell] + pcp] = (PrecReal)pressure_row_entry(wgt, Jp, 3);
    }
  };
  scatter3(0, sJ);
  for (int f = 0; f < 6; ++f)
    if (mc[1 + f] >= 0) scatter3(1 + f, nJ[f]);
}

// ===========================================================================
// The same three-phase analytic assembly, over a general connection list.
//
// Why a second kernel rather than a rewrite of the one above
// ----------------------------------------------------------
// The two must agree on a structured grid -- that is the test that says this
// generalisation is faithful -- so both exist and either can be selected.  On
// a Cartesian box the connection list holds exactly the six faces the fixed
// stencil holds, and the residual and Jacobian come out equal to round-off.
// Once that is established, this one also runs on grids the other cannot
// describe: corner-point geometry with faults, where a cell may have anywhere
// from one to seventeen neighbours because a throw juxtaposes it against parts
// of several.
//
// Three differences of substance:
//
// 1. Gravity comes from a per-connection depth difference rather than from a
//    face index.  The old form asked "is this face index >= 4", which is a
//    question about a Cartesian stencil and has no answer for a fault
//    connection that is neither vertical nor lateral.  Since
//      (p_j - g rho z_j) - (p_i - g rho z_i)  ==  (p_j - p_i) - g rho (z_j - z_i)
//    the depth difference is the only geometric quantity the flux ever needed,
//    and differencing the depths avoids subtracting two large g*rho*z terms.
//
// 2. Nothing is held per neighbour.  The old kernel accumulated nJ[6][3][3]
//    and scattered it at the end; at degree seventeen that array would be 612
//    bytes a thread and spill to local memory.  But a neighbour's block is
//    complete the moment its own connection has been processed -- only the
//    self block accumulates across connections -- so each is written straight
//    to the CSR inside the loop.  Register use is then independent of degree.
//
// 3. The pressure row needs the CPR weights, which are not known until the
//    self block is finished.  Rather than keep the blocks around for a second
//    pass, the column-0 entries are read back out of the CSR after the weights
//    are formed.  Same thread, already written: no race, and no storage.
// ===========================================================================

// atomicMax for a non-negative double.  IEEE-754 orders positive doubles the
// same way their bit patterns order as unsigned integers, so reinterpreting is
// exact and needs no compare-and-swap loop.  Used for the CNV norm, which is a
// maximum over cells and cannot be formed by the atomicAdd the other norms use.
__device__ static inline void atomic_max_pos(unsigned long long* addr, Real v) {
  if (!(v > 0)) return;
  atomicMax(addr, (unsigned long long)__double_as_longlong((double)v));
}

// Eclipse THPRES dead band.  Once a phase-potential difference exceeds the
// region barrier, only the excess drives flow; below it the face is closed.
__device__ static inline Real threshold_potential(Real dphi, Real threshold) {
  if (!(threshold > 0)) return dphi;
  if (dphi > threshold) return dphi - threshold;
  if (dphi < -threshold) return dphi + threshold;
  return (Real)0;
}

// Form a face potential in one canonical (low-cell -> high-cell) orientation,
// then negate it for the reverse directed connection.  Evaluating a physical
// face independently in both directions is only algebraically antisymmetric;
// different floating-point operation order leaves a field-wide conservation
// error after millions of large fluxes cancel.
__device__ static inline Real canonical_potential(
    int cell, int neighbour, Real phi_i, Real phi_j, Real rho_head, Real dz) {
  const bool forward = cell < neighbour;
  const Real dp = forward ? (phi_j - phi_i) : (phi_i - phi_j);
  const Real dz_forward = forward ? dz : -dz;
  const Real value = dp - rho_head * dz_forward;
  return forward ? value : -value;
}

// PVT is a cell property at the current state, but the connection kernel used
// to re-evaluate a neighbour's tabulated oil/gas/water state for every directed
// connection.  Norne averages 5.8 neighbours per cell, so most table searches
// and centred pressure differences were repeated about six times.  Evaluate
// each owned or ghost cell once and let the connection kernel consume the
// exact same PvtCell values from a dense cache.
struct AssemblyCellProps {
  PvtCell pvt;
  Cap cap;
  Mob3 mob;
};

static BOParams make_bo_params(const Reservoir& r) {
  BOParams P{};
  P.bs = r.bs; P.nx = r.grid.nxl; P.ny = r.grid.nyl; P.nz = r.grid.nzl;
  P.dz_cell = r.grid.dz; P.dt = r.dt; P.ct = r.ct;
  P.have_rock = r.have_rock; P.rock_pref = r.rock_pref;
  P.cnv_tol = r.cnv_tol;
  P.cpr_decouple = r.cpr_decouple;
  P.drs_eps = r.drs_eps;
  P.uw = r.uw; P.uo = r.uo; P.ug = r.ug;
  P.bw = r.bw; P.bo = r.bo; P.bg = r.bg; P.rs = r.rs;
  P.pvt = r.pvt;
  P.pvt_regions = r.pvt_regions; P.pvt_cell = r.pvt_cell;
  P.rs_max = (r.rs_max > 0) ? r.rs_max : r.rs;
  P.rs_max_cell = r.rs_max_cell; P.rs_cap = r.rs_cap;
  P.so_max = r.so_max; P.vap1 = r.vap1;
  P.gas_primary = r.gas_primary;
  P.gas_primary_old = r.gas_primary_old;
  P.swi = r.swi; P.swr = r.swr;
  P.rhow = r.rhow; P.rhoo = r.rhoo; P.rhog = r.rhog;
  P.gravity = r.gravity; P.grav_grad = r.grav_grad; P.fd_eps = r.fd_eps;
  return P;
}

__global__ static void k_prepare_pvt_cache(const Real* u, const Real* u_old,
                                           const Real* aimflag, int n,
                                           SatTables T, BOParams P,
                                           AssemblyCellProps* out) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n) return;
  const Real p = u[3 * cell];
  const Real sw = u[3 * cell + 1];
  const Real meaning = P.gas_primary ? P.gas_primary[cell] : (Real)0;
  const bool rs_primary = meaning > (Real)0.5 && meaning < (Real)1.5;
  const bool rv_primary = meaning > (Real)1.5;
  const Real x = u[3 * cell + 2];
  const Real sg = rs_primary ? (Real)0
                              : (rv_primary ? (Real)1 - sw : x);
  const Real so = (Real)1 - sw - sg;
  const Real rsanchor = P.rs_cap        ? P.rs_cap[cell]
                        : P.rs_max_cell ? P.rs_max_cell[cell]
                                        : P.rs_max;
  const Real rsmax = (!P.rs_cap && sg > (Real)1e-7)
                         ? (Real)1e30 : rsanchor;
  const DevicePvt& pv = pvt_of(P, cell);
  const Real somax = P.so_max ? P.so_max[cell] : so;
  out[cell].pvt = pv.valid
                          ? (rs_primary
                                 ? dev_pvt_cell_rs(pv, p, x, so, somax, P.vap1)
                                 : (rv_primary
                                        ? dev_pvt_cell_rv(pv, p, x, somax,
                                                          P.vap1)
                                        : dev_pvt_cell(pv, p, rsmax, so,
                                                       somax, P.vap1)))
                          : const_pvt_cell(P.bo, P.bw, P.bg, P.rs,
                                           P.uo, P.uw, P.ug);
  if (!rs_primary && !rv_primary) {
    // On the Sg branch x changes So with derivative -1.  Record the complete
    // VAPPARS chain once so accumulation, faces and wells can all use it.
    out[cell].pvt.drv_dx = -out[cell].pvt.drv_dso;
    out[cell].pvt.dbg_dx = -out[cell].pvt.dbg_dso;
    out[cell].pvt.dmug_dx = -out[cell].pvt.dmug_dso;
  }
  if (!rv_primary) {
    out[cell].pvt.drv_dsw = -out[cell].pvt.drv_dso;
    out[cell].pvt.dbg_dsw = -out[cell].pvt.dbg_dso;
    out[cell].pvt.dmug_dsw = -out[cell].pvt.dmug_dso;
  }
  out[cell].cap = capillary(sw, sg, T, cell);
  const bool explicit_cell = aimflag[cell] < (Real)0.5;
  const Real sw_m = explicit_cell ? u_old[3 * cell + 1] : sw;
  const Real old_meaning = P.gas_primary_old ? P.gas_primary_old[cell]
                                               : (Real)0;
  const bool rs_old = old_meaning > (Real)0.5 && old_meaning < (Real)1.5;
  const bool rv_old = old_meaning > (Real)1.5;
  const Real sg_old = rs_old ? (Real)0
                      : (rv_old ? (Real)1 - u_old[3 * cell + 1]
                                : u_old[3 * cell + 2]);
  const Real sg_m = explicit_cell ? sg_old : sg;
  const PvtCell& f = out[cell].pvt;
  out[cell].mob = relperm3(sw_m, sg_m, T, f.muw, f.muo, f.mug,
                           (Real)1, (Real)1, (Real)1, P.swi, P.swr, cell);
}

// One fixed-order partial sum per block.  This replaces a synchronous copy of
// the complete primary-variable field followed by 44k host PVT table searches.
// The host combines the small partial array in block order, so unlike atomics
// the result is repeatable from run to run.
__global__ static void k_region_average(
    const Real* u, const Real* poro, const Real* cellvol, Real uniform_volume,
    const AssemblyCellProps* props, int cells, double* partial) {
  constexpr int B = 256;
  __shared__ double sum[8 * B];
  const int lane = threadIdx.x;
  const int c = blockIdx.x * blockDim.x + lane;
  double v[8] = {};
  if (c < cells) {
    const double pv = (double)(cellvol ? cellvol[c] : uniform_volume) *
                      (double)poro[c];
    if (pv > 0.0) {
      const double p = (double)u[3 * c];
      const double sw = (double)u[3 * c + 1];
      const double hpv = pv * fmax(0.0, 1.0 - sw);
      const PvtCell& f = props[c].pvt;
      v[0] = hpv; v[1] = hpv * p;
      v[2] = hpv * (double)f.rs; v[3] = hpv * (double)f.rv;
      v[4] = pv; v[5] = pv * p;
      v[6] = pv * (double)f.rs; v[7] = pv * (double)f.rv;
    }
  }
  #pragma unroll
  for (int q = 0; q < 8; ++q) sum[q * B + lane] = v[q];
  __syncthreads();
  for (int stride = B / 2; stride; stride >>= 1) {
    if (lane < stride) {
      #pragma unroll
      for (int q = 0; q < 8; ++q)
        sum[q * B + lane] += sum[q * B + lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    #pragma unroll
    for (int q = 0; q < 8; ++q)
      partial[(size_t)blockIdx.x * 8 + q] = sum[q * B];
  }
}

static AssemblyCellProps* prepare_assembly_props(Reservoir& r,
                                                  const BOParams& P) {
  const int cells_ext = r.grid.n_local_ext();
  if (cells_ext > r.assembly_pvt_capacity) {
    if (r.assembly_pvt_cache) cudaFree(r.assembly_pvt_cache);
    NVRS_CHECK_CUDA(cudaMalloc(&r.assembly_pvt_cache,
                               sizeof(AssemblyCellProps) * (size_t)cells_ext));
    r.assembly_pvt_capacity = cells_ext;
  }
  auto* props = static_cast<AssemblyCellProps*>(r.assembly_pvt_cache);
  k_prepare_pvt_cache<<<grid1d(cells_ext), 256>>>(
      r.u, r.u_old, r.aimflag, cells_ext, r.tables, P, props);
  NVRS_CHECK_CUDA(cudaGetLastError());
  return props;
}

bool blackoil_region_average(Reservoir& r, double out[8]) {
  if (!out || r.bs != 3 || r.jacobian != 1 || !r.use_conn_list ||
      !r.conn_ptr || !r.u || !r.poro)
    return false;
  const BOParams P = make_bo_params(r);
  if (r.halo_conn)
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.u, r.bs);
  else
    halo_exchange(r.u, r.bs, r.grid);
  AssemblyCellProps* props = prepare_assembly_props(r, P);
  const int cells = r.grid.n_local();
  const int blocks = (cells + 255) / 256;
  if (blocks > r.well_region_capacity) {
    if (r.well_region_partials) cudaFree(r.well_region_partials);
    NVRS_CHECK_CUDA(cudaMalloc(&r.well_region_partials,
                               sizeof(double) * (size_t)blocks * 8));
    r.well_region_capacity = blocks;
  }
  const Real V = (double)r.grid.dx * r.grid.dy * r.grid.dz /
                 units::kFt3PerBbl;
  k_region_average<<<blocks, 256>>>(r.u, r.poro, r.cellvol, V, props, cells,
                                    r.well_region_partials);
  NVRS_CHECK_CUDA(cudaGetLastError());
  static thread_local std::vector<double> partial;
  partial.resize((size_t)blocks * 8);
  NVRS_CHECK_CUDA(cudaMemcpy(partial.data(), r.well_region_partials,
                             sizeof(double) * partial.size(),
                             cudaMemcpyDeviceToHost));
  for (int q = 0; q < 8; ++q) out[q] = 0.0;
  for (int b = 0; b < blocks; ++b)
    for (int q = 0; q < 8; ++q) out[q] += partial[(size_t)b * 8 + q];
  r.assembly_pvt_ready = 1;
  return true;
}

// Exact analytic two-phase assembly for an arbitrary connection list.  This is
// the field-grid counterpart of k_assemble_bo_analytic: it uses the imported
// transmissibilities, depth differences and per-cell accumulation volumes.
__global__ void k_assemble_bo2_analytic_conn(
    const Real* u, const Real* u_old, const Real* poro,
    const int* conn_ptr, const int* conn_nbr, const Real* conn_trans,
    const Real* conn_dz, const Real* conn_threshold,
    const int* conn_bcol, const int* conn_pcol,
    const int* self_bcol, const int* self_pcol,
    const int* rowptr, const int* prowptr, const WellCell* well,
    const Real* aimflag, const Real* cellvol, SatTables T, BOParams P, Real V,
    Real* b,
    Real* aval, PrecReal* pval, PrecReal* dinv, PrecReal* pweight,
    Real* rnorm2, unsigned long long* cnv) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int ncell = P.nx * P.ny * P.nz;
  if (cell >= ncell) return;

  const Real Vc = cellvol ? cellvol[cell] : V;
  const Real p_i = u[2 * cell];
  const Real sw_i = u[2 * cell + 1];
  const Real p_old = u_old[2 * cell];
  const Real sw_old = u_old[2 * cell + 1];
  const Real bo = (Real)1 / P.bo;
  const Real bw = (Real)1 / P.bw;
  const Real acc = Vc * poro[cell] / P.dt;
  const Real pvfac = P.have_rock
                         ? rock_pv_mult(p_i, P.rock_pref, P.ct)
                         : (Real)1 + P.ct * (p_i - p_old);
  const Real pvfac_old = P.have_rock
                             ? rock_pv_mult(p_old, P.rock_pref, P.ct)
                             : (Real)1;
  const Real dpvfac_dp = P.have_rock
                             ? rock_pv_mult_deriv(p_i, P.rock_pref, P.ct)
                             : P.ct;

  Real R_oil = acc * bo *
               ((Real)(1 - sw_i) * pvfac -
                (Real)(1 - sw_old) * pvfac_old);
  Real R_wat = acc * bw * (sw_i * pvfac - sw_old * pvfac_old);
  // Internal face fluxes cancel analytically. Keep them out of the global MB
  // reduction: summing millions of large opposing divergences loses enough
  // low bits to create a false O(10^3) STB/day SPE10 imbalance.
  Real MB_oil = R_oil;
  Real MB_wat = R_wat;
  Real sJ[2][2] = {};
  sJ[0][0] = acc * bo * (1 - sw_i) * dpvfac_dp;
  sJ[0][1] = -acc * bo * pvfac;
  sJ[1][0] = acc * bw * sw_i * dpvfac_dp;
  sJ[1][1] = acc * bw * pvfac;

  const bool ex_i = aimflag[cell] < (Real)0.5;
  const int c0 = conn_ptr[cell], c1 = conn_ptr[cell + 1];
  const Real grav = P.gravity ? P.grav_grad : (Real)0;
  for (int m = c0; m < c1; ++m) {
    const Real trans = conn_trans[m];
    if (!(trans > 0)) continue;
    const int nc = conn_nbr[m];
    const Real p_j = u[2 * nc];
    const Real sw_j = u[2 * nc + 1];
    const Real dz = conn_dz[m];
    const Real th = conn_threshold ? conn_threshold[m] : (Real)0;
    const Real dpo0 = canonical_potential(
        cell, nc, p_i, p_j, grav * P.rhoo, dz);
    const Real dpw0 = canonical_potential(
        cell, nc, p_i, p_j, grav * P.rhow, dz);
    const Real dpo = threshold_potential(dpo0, th);
    const Real dpw = threshold_potential(dpw0, th);
    // At zero potential an ordinary face still has derivative T*lambda.
    // Dropping it disconnected every equal-pressure horizontal SPE10 cell
    // from the Newton matrix.  Only a positive THPRES dead band makes the
    // flux locally constant and its derivative genuinely zero.
    const bool flow_o = !(th > 0 && fabs(dpo0) <= th);
    const bool flow_w = !(th > 0 && fabs(dpw0) <= th);
    const bool upo_j = dpo > 0;
    const bool upw_j = dpw > 0;
    const bool ex_j = aimflag[nc] < (Real)0.5;
    const bool ex_o = upo_j ? ex_j : ex_i;
    const bool ex_w = upw_j ? ex_j : ex_i;
    const Real sw_o = upo_j
                          ? (ex_o ? u_old[2 * nc + 1] : sw_j)
                          : (ex_o ? sw_old : sw_i);
    const Real sw_w = upw_j
                          ? (ex_w ? u_old[2 * nc + 1] : sw_j)
                          : (ex_w ? sw_old : sw_i);
    Mob2 mo = relperm2_table(sw_o, T, P.uw, P.uo, 1, 1, P.swi, P.swr);
    Mob2 mw = relperm2_table(sw_w, T, P.uw, P.uo, 1, 1, P.swi, P.swr);
    if (!flow_o) mo.Mo = mo.dMo = 0;
    if (!flow_w) mw.Mw = mw.dMw = 0;

    const Real fo = trans * mo.Mo * dpo;
    const Real fw = trans * mw.Mw * dpw;
    R_oil -= bo * fo;
    R_wat -= bw * fw;

    Real nJ[2][2] = {};
    sJ[0][0] += bo * trans * mo.Mo;
    nJ[0][0] -= bo * trans * mo.Mo;
    sJ[1][0] += bw * trans * mw.Mw;
    nJ[1][0] -= bw * trans * mw.Mw;
    const Real do_dsw = -bo * trans * dpo * mo.dMo;
    const Real dw_dsw = -bw * trans * dpw * mw.dMw;
    if (!ex_o) (upo_j ? nJ[0][1] : sJ[0][1]) += do_dsw;
    if (!ex_w) (upw_j ? nJ[1][1] : sJ[1][1]) += dw_dsw;

    const int off = conn_bcol[m];
    if (off >= 0) {
      for (int eq = 0; eq < 2; ++eq)
        for (int v = 0; v < 2; ++v)
          aval[rowptr[2 * cell + eq] + off + v] = nJ[eq][v];
    }
  }

  const WellCell w = well[cell];
  if (w.is_prod) {
    const Real draw = p_i - w.pwf;
    const Mob2 mob =
        relperm2_table(sw_i, T, P.uw, P.uo, 1, 1, P.swi, P.swr);
    const Real qo = bo * w.wi * mob.Mo * draw;
    const Real qw = bw * w.wi * mob.Mw * draw;
    R_oil += qo; MB_oil += qo;
    R_wat += qw; MB_wat += qw;
    atomicAdd(rnorm2 + 17, qo);
    atomicAdd(rnorm2 + 18, qw);
    sJ[0][0] += bo * w.wi * mob.Mo;
    sJ[0][1] += bo * w.wi * mob.dMo * draw;
    sJ[1][0] += bw * w.wi * mob.Mw;
    sJ[1][1] += bw * w.wi * mob.dMw * draw;
  }
  if (w.is_inj) {
    R_oil -= w.qo_inj;
    R_wat -= w.qw_inj;
    MB_oil -= w.qo_inj;
    MB_wat -= w.qw_inj;
    atomicAdd(rnorm2 + 20, w.qo_inj);
    atomicAdd(rnorm2 + 21, w.qw_inj);
    if (w.inj_rate_coupled) {
      sJ[1][0] -= w.inj_dqdu[0];
      sJ[1][1] -= w.inj_dqdu[1];
    }
    if (w.inj_xflow_coupled) {
      for (int eq = 0; eq < 2; ++eq)
        for (int v = 0; v < 2; ++v)
          sJ[eq][v] -= w.inj_direct[3 * eq + v];
    }
  }

  b[2 * cell] = -R_oil;
  b[2 * cell + 1] = -R_wat;
  atomicAdd(rnorm2, R_oil * R_oil + R_wat * R_wat);
  atomicAdd(rnorm2 + 1, R_oil * R_oil);
  atomicAdd(rnorm2 + 2, R_wat * R_wat);
  atomicAdd(rnorm2 + 11, R_oil);
  atomicAdd(rnorm2 + 12, R_wat);
  atomicAdd(rnorm2 + 14, MB_oil);
  atomicAdd(rnorm2 + 15, MB_wat);
  const Real pvc = Vc * poro[cell];
  atomicAdd(rnorm2 + 6, pvc);
  if (pvc > 0 && cnv) {
    const Real sc = P.dt / pvc;
    const Real co = fabs(R_oil) * sc / (bo > 0 ? bo : 1);
    const Real cw = fabs(R_wat) * sc / (bw > 0 ? bw : 1);
    atomic_max_pos(cnv + 0, co);
    atomic_max_pos(cnv + 1, cw);
    if (P.cnv_tol > 0) {
      if (co > P.cnv_tol) atomicAdd(rnorm2 + 7, pvc);
      if (cw > P.cnv_tol) atomicAdd(rnorm2 + 8, pvc);
      if (fmax(co, cw) > P.cnv_tol) atomicAdd(rnorm2 + 10, pvc);
    }
  }

  Real flat[4] = {sJ[0][0], sJ[0][1], sJ[1][0], sJ[1][1]};
  Real Inv[9];
  const bool ok = invert_block(flat, 2, Inv);
  store_dinv(Inv, ok, dinv + (long long)cell * 4, 2);
  if (!ok) atomicAdd(rnorm2 + 4, (Real)1);
  Real wgt[2];
  if (ok) {
    wgt[0] = Inv[0];
    wgt[1] = Inv[1];
  } else {
    fallback_weights(wgt, 2);
  }
  // Flow's default CPRW uses true-IMPES weights, formed from storage rather
  // than the full diagonal Jacobian.  For [oil,water] conservation with
  // primaries [p,Sw], the saturation column of storage is [-bo,+bw].  Its
  // left null vector is therefore proportional to [1,bo/bw].  Normalizing by
  // the largest component matches OPM and produces the total-component
  // pressure equation without importing transport/front stiffness into the
  // pressure restriction.  Keep Inv above for the fine-grid block solve.
  wgt[0] = (Real)1;
  wgt[1] = bw > (Real)0 ? bo / bw : (Real)1;
  const Real wtmax = fmax(fabs(wgt[0]), fabs(wgt[1]));
  if (wtmax > (Real)0) {
    wgt[0] /= wtmax;
    wgt[1] /= wtmax;
  }
  pweight[2 * cell] = (PrecReal)wgt[0];
  pweight[2 * cell + 1] = (PrecReal)wgt[1];

  const int soff = self_bcol[cell];
  if (soff >= 0) {
    for (int eq = 0; eq < 2; ++eq)
      for (int v = 0; v < 2; ++v)
        aval[rowptr[2 * cell + eq] + soff + v] = sJ[eq][v];
    const int spc = self_pcol[cell];
    if (spc >= 0) {
      const Real jp[2] = {sJ[0][0], sJ[1][0]};
      pval[prowptr[cell] + spc] =
          (PrecReal)pressure_row_entry(wgt, jp, 2);
    }
  }
  for (int m = c0; m < c1; ++m) {
    const int pcp = conn_pcol[m];
    const int off = conn_bcol[m];
    if (pcp < 0 || off < 0) continue;
    const Real jp[2] = {aval[rowptr[2 * cell] + off],
                        aval[rowptr[2 * cell + 1] + off]};
    pval[prowptr[cell] + pcp] =
        (PrecReal)pressure_row_entry(wgt, jp, 2);
  }
}

__global__ void k_assemble_bo3_analytic_conn(
    const Real* u, const Real* u_old, const Real* poro,
    const int* conn_ptr, const int* conn_nbr, const Real* conn_trans,
    const Real* conn_dz, const Real* conn_threshold,
    const int* conn_bcol, const int* conn_pcol,
    const int* self_bcol, const int* self_pcol,
    const int* rowptr, const int* prowptr, const WellCell* well,
    const Real* aimflag, const Real* cellvol,
    const AssemblyCellProps* cell_props,
    SatTables T, BOParams P,
    Real V, Real* b,
    Real* aval, PrecReal* pval, PrecReal* dinv, PrecReal* pweight,
    Real* rnorm2, unsigned long long* cnv) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int ncell = P.nx * P.ny * P.nz;
  if (cell >= ncell) return;

  // Accumulation volume.  A uniform grid has one value for every cell and
  // passes it as scalar V.  A corner-point field supplies PORV/(PORO*NTG), so
  // Vc*poro below is the deck's final PORV including MULTPV and MINPV.  It is
  // deliberately not always the geometric bulk volume.
  const Real Vc = cellvol ? cellvol[cell] : V;

  const Real gpsi = P.grav_grad;

  const Real p_i  = u[3 * cell];
  const Real Sw_i = u[3 * cell + 1];
  const Real meaning_i = P.gas_primary ? P.gas_primary[cell] : (Real)0;
  const Real meaning_old_i = P.gas_primary_old
                                 ? P.gas_primary_old[cell] : (Real)0;
  const bool rs_i = meaning_i > (Real)0.5 && meaning_i < (Real)1.5;
  const bool rv_i = meaning_i > (Real)1.5;
  const bool rs_old_i = meaning_old_i > (Real)0.5 &&
                        meaning_old_i < (Real)1.5;
  const bool rv_old_i = meaning_old_i > (Real)1.5;
  const Real x_i = u[3 * cell + 2];
  const Real Sg_i = rs_i ? (Real)0 : (rv_i ? (Real)1 - Sw_i : x_i);
  const Real usn1 = u_old[3 * cell + 1];
  const Real usn2 = rs_old_i ? (Real)0
                     : (rv_old_i ? (Real)1 - usn1
                                 : u_old[3 * cell + 2]);
  const Real So_i = 1.0 - Sw_i - Sg_i;
  const Real So_n = 1.0 - usn1 - usn2;
  const bool ex_i = aimflag[cell] < 0.5;
  const Real p_old = u_old[3 * cell];

  // Fluid properties at this cell.  With tables these are functions of
  // pressure and carry their own slopes; without, they are the deck's
  // constants with zero slopes and P.ct remains the only compressibility.
  // The undersaturated anchor for THIS cell.
  const Real rsanchor_i = P.rs_cap        ? P.rs_cap[cell]
                          : P.rs_max_cell ? P.rs_max_cell[cell]
                                          : P.rs_max;
  const DevicePvt& PVi = pvt_of(P, cell);
  const Real somax_i = P.so_max ? P.so_max[cell] : So_i;
  const PvtCell& F = cell_props[cell].pvt;
  const Real bo = F.bo, bw = F.bw, bg = F.bg, rs = F.rs, rv = F.rv;

  // The old-time term must be evaluated at the OLD pressure, not this one.
  // Using the current b there would make the accumulation a difference of
  // saturations scaled by one factor rather than a difference of two masses,
  // and the entire compressibility of the cell would cancel out of it.
  const Real x_old_i = u_old[3 * cell + 2];
  const Real rsmax_n = rs_old_i ? x_old_i
                       : (!P.rs_cap && usn2 > (Real)1e-7)
                             ? (Real)1e30 : rsanchor_i;
  const PvtCell Fn = PVi.valid
                         ? (rs_old_i
                                ? dev_pvt_cell_rs(PVi, p_old, x_old_i, So_n,
                                                  somax_i, P.vap1)
                                : (rv_old_i
                                       ? dev_pvt_cell_rv(PVi, p_old, x_old_i,
                                                         somax_i, P.vap1)
                                       : dev_pvt_cell(PVi, p_old, rsmax_n,
                                                      So_n, somax_i,
                                                      P.vap1)))
                         : F;

  const Real acc = Vc * poro[cell] / P.dt;

  // ROCK is an absolute pore-volume law about its deck reference pressure.
  // Both time levels must use that same reference or accumulation does not
  // telescope over timesteps.  PVTCONST decks without ROCK retain the older
  // incremental total-compressibility fallback needed to avoid a singular
  // closed incompressible pressure system.
  const Real pvfac = P.have_rock
                         ? rock_pv_mult(p_i, P.rock_pref, P.ct)
                         : (Real)1 + P.ct * (p_i - p_old);
  const Real pvfac_old = P.have_rock
                             ? rock_pv_mult(p_old, P.rock_pref, P.ct)
                             : (Real)1;
  const Real dpvfac_dp = P.have_rock
                             ? rock_pv_mult_deriv(p_i, P.rock_pref, P.ct)
                             : P.ct;

  // With VAPOIL, stock-tank oil is present in both phases: bo*So in the oil
  // phase and Rv*bg*Sg in the gas phase.  This is OPM's black-oil component
  // form; omitting the second term loses condensate from accumulation,
  // transport, wells and every reported oil/gas ratio.
  Real R_oil = acc * ((bo * So_i + rv * bg * Sg_i) * pvfac
                      - (Fn.bo * So_n + Fn.rv * Fn.bg * usn2) * pvfac_old);
  Real R_wat = acc * (bw * Sw_i * pvfac - Fn.bw * usn1 * pvfac_old);
  Real R_gas = acc * ((bg * Sg_i + rs * bo * So_i) * pvfac
                      - (Fn.bg * usn2 + Fn.rs * Fn.bo * So_n) * pvfac_old);

  Real sJ[3][3] = {};
  const Real dvb_dso = F.drv_dso * bg + rv * F.dbg_dso;
  if (P.gas_primary) {
    const Real dso_sw = rv_i ? (Real)0 : (Real)-1;
    const Real dsg_sw = rv_i ? (Real)-1 : (Real)0;
    sJ[0][1] += acc *
        (F.dbo_dsw * So_i + bo * dso_sw +
         (F.drv_dsw * bg + rv * F.dbg_dsw) * Sg_i +
         rv * bg * dsg_sw) * pvfac;
  } else {
    sJ[0][1] += acc * (-bo - dvb_dso * Sg_i) * pvfac;
  }
  sJ[0][2] += acc *
      ((F.dbo_dx * So_i + (F.drv_dx * bg + rv * F.dbg_dx) * Sg_i +
        ((rs_i || rv_i) ? (Real)0 : (rv * bg - bo))) * pvfac);
  sJ[1][1] += acc * bw * pvfac;
  if (P.gas_primary) {
    const Real dso_sw = rv_i ? (Real)0 : (Real)-1;
    const Real dsg_sw = rv_i ? (Real)-1 : (Real)0;
    sJ[2][1] += acc *
        (F.dbg_dsw * Sg_i + bg * dsg_sw +
         (F.drs_dsw * bo + rs * F.dbo_dsw) * So_i +
         rs * bo * dso_sw) * pvfac;
  } else {
    sJ[2][1] += acc * (-rs * bo - F.dbg_dso * Sg_i) * pvfac;
  }
  sJ[2][2] += acc *
      ((F.dbg_dx * Sg_i + (F.drs_dx * bo + rs * F.dbo_dx) * So_i +
        ((rs_i || rv_i) ? (Real)0 : (bg - rs * bo))) * pvfac);

  // The pressure derivative of the accumulation: what makes the system
  // compressible, and the pressure operator non-singular.  Two terms now --
  // the pore volume opening up (P.ct) and the fluid itself densifying (db/dp)
  // -- where the constant-PVT path had only the first, standing in for both.
  //
  // For gas the second term dominates by an order of magnitude, and it is the
  // one that decides whether a gas injector is solvable: it sets how much
  // pressure a given injected volume must raise, and a value 9x too small asks
  // Newton to push the pressure 9x too far on every iteration.
  const Real d_bo = F.dbo, d_bw = F.dbw, d_bg = F.dbg, d_rs = F.drs;
  const Real d_rv = F.drv;
  sJ[0][0] += acc * ((bo * So_i + rv * bg * Sg_i) * dpvfac_dp
                     + pvfac * (d_bo * So_i +
                               (d_rv * bg + rv * d_bg) * Sg_i));
  sJ[1][0] += acc * (bw * Sw_i * dpvfac_dp + pvfac * d_bw * Sw_i);
  sJ[2][0] += acc * ((bg * Sg_i + rs * bo * So_i) * dpvfac_dp
                     + pvfac * (d_bg * Sg_i + (d_rs * bo + rs * d_bo) * So_i));

  const Real g = P.gravity ? gpsi : 0.0;
  const int c0 = conn_ptr[cell], c1 = conn_ptr[cell + 1];

  for (int m = c0; m < c1; ++m) {
    const Real Tg = conn_trans[m];
    if (Tg <= 0) continue;
    const int nc = conn_nbr[m];
    const Real dz = conn_dz[m];

    const Real meaning_j = P.gas_primary ? P.gas_primary[nc] : (Real)0;
    const bool rs_j = meaning_j > (Real)0.5 && meaning_j < (Real)1.5;
    const bool rv_j = meaning_j > (Real)1.5;
    const Real p_j = u[3 * nc], Sw_j = u[3 * nc + 1];
    const Real Sg_j = rs_j ? (Real)0
                    : (rv_j ? (Real)1 - Sw_j : u[3 * nc + 2]);
    // Evaluate the neighbour before forming phase potentials: gravity needs
    // reservoir phase densities, which depend on both cells' PVT state.
    const PvtCell& N = cell_props[nc].pvt;

    // Phase pressures differ by the capillary pressures.
    //
    // p is the OIL pressure; SWOF column four gives Pcow = p_o - p_w and
    // SGOF column four Pcog = p_g - p_o, so
    //
    //     p_w = p - Pcow(Sw)      p_g = p + Pcog(Sg)
    //
    // and each phase's potential difference carries the difference of ITS
    // capillary pressure across the face.  Both are zero, and every line
    // below collapses to what it was, when the deck gives three columns.
    const Cap& Ci = cell_props[cell].cap;
    const Cap& Cj = cell_props[nc].cap;
    // DENSITY gives surface densities.  OPM forms the phase potential with
    // reservoir densities: multiply by the shrinkage factor and include the
    // mass of dissolved gas in oil and vaporised oil in wet gas.  The old
    // constant P.rhog was 0.00085 for Norne; the actual reservoir gas is
    // about 0.19 SG, so gas segregation and completion crossflow were nearly
    // absent.  Arithmetic face averaging is OPM's standard TPFA treatment.
    constexpr Real ft3_per_bbl = (Real)5.614583333333333;
    const Real rho_w_i = P.rhow * F.bw;
    const Real rho_w_j = P.rhow * N.bw;
    const Real rho_o_i = (P.rhoo + F.rs * P.rhog / ft3_per_bbl) * F.bo;
    const Real rho_o_j = (P.rhoo + N.rs * P.rhog / ft3_per_bbl) * N.bo;
    const Real rho_g_i = (P.rhog / ft3_per_bbl + F.rv * P.rhoo) * F.bg;
    const Real rho_g_j = (P.rhog / ft3_per_bbl + N.rv * P.rhoo) * N.bg;
    const Real rho_w = (Real)0.5 * (rho_w_i + rho_w_j);
    const Real rho_o = (Real)0.5 * (rho_o_i + rho_o_j);
    const Real rho_g = (Real)0.5 * (rho_g_i + rho_g_j);
    const Real drho_w_i = P.rhow * F.dbw;
    const Real drho_w_j = P.rhow * N.dbw;
    const Real oil_mass_i = P.rhoo + F.rs * P.rhog / ft3_per_bbl;
    const Real oil_mass_j = P.rhoo + N.rs * P.rhog / ft3_per_bbl;
    const Real drho_o_i = F.drs * P.rhog / ft3_per_bbl * F.bo +
                          oil_mass_i * F.dbo;
    const Real drho_o_j = N.drs * P.rhog / ft3_per_bbl * N.bo +
                          oil_mass_j * N.dbo;
    const Real gas_mass_i = P.rhog / ft3_per_bbl + F.rv * P.rhoo;
    const Real gas_mass_j = P.rhog / ft3_per_bbl + N.rv * P.rhoo;
    const Real drho_g_i = F.drv * P.rhoo * F.bg + gas_mass_i * F.dbg;
    const Real drho_g_j = N.drv * P.rhoo * N.bg + gas_mass_j * N.dbg;
    const Real th = conn_threshold ? conn_threshold[m] : (Real)0;
    const Real po_i = p_i - (rv_i ? Ci.pcog : (Real)0);
    const Real po_j = p_j - (rv_j ? Cj.pcog : (Real)0);
    const Real dPw0 = canonical_potential(
        cell, nc, po_i - Ci.pcow, po_j - Cj.pcow, g * rho_w, dz);
    const Real dPo0 = canonical_potential(
        cell, nc, po_i, po_j, g * rho_o, dz);
    const Real dPg0 = canonical_potential(
        cell, nc, po_i + Ci.pcog, po_j + Cj.pcog, g * rho_g, dz);
    const Real dPw = threshold_potential(dPw0, th);
    const Real dPo = threshold_potential(dPo0, th);
    const Real dPg = threshold_potential(dPg0, th);
    // Zero potential is an active face with a nonzero pressure derivative.
    // Only THPRES makes a finite interval genuinely inactive.
    const bool flow_w = !(th > 0 && fabs(dPw0) <= th);
    const bool flow_o = !(th > 0 && fabs(dPo0) <= th);
    const bool flow_g = !(th > 0 && fabs(dPg0) <= th);
    // Magnitudes of -d(Phi_j-Phi_i)/dp_i and
    // +d(Phi_j-Phi_i)/dp_j.  Keeping these density slopes in the Jacobian
    // makes Newton see the same PVT-dependent gravity function as the
    // residual instead of a frozen-density approximation.
    const Real kw_i = 1 + g * (Real)0.5 * drho_w_i * dz;
    const Real kw_j = 1 - g * (Real)0.5 * drho_w_j * dz;
    const Real ko_i = 1 + g * (Real)0.5 * drho_o_i * dz;
    const Real ko_j = 1 - g * (Real)0.5 * drho_o_j * dz;
    const Real kg_i = 1 + g * (Real)0.5 * drho_g_i * dz;
    const Real kg_j = 1 - g * (Real)0.5 * drho_g_j * dz;
    const bool wj = dPw > 0.0, oj = dPo > 0.0, gj = dPg > 0.0;

    // AIM: each phase's donor decides; an explicit donor contributes its old
    // saturations and no derivative for that phase's mobility.
    const bool exn = aimflag[nc] < 0.5;
    const Real son0 = u_old[3 * nc + 1];
    const Real meaning_old_j = P.gas_primary_old
                                   ? P.gas_primary_old[nc] : (Real)0;
    const bool rs_old_j = meaning_old_j > (Real)0.5 &&
                          meaning_old_j < (Real)1.5;
    const bool rv_old_j = meaning_old_j > (Real)1.5;
    const Real son1 = rs_old_j ? (Real)0
                      : (rv_old_j ? (Real)1 - son0 : u_old[3 * nc + 2]);
    const bool ex_w = wj ? exn : ex_i;
    const bool ex_o = oj ? exn : ex_i;
    const bool ex_g = gj ? exn : ex_i;
    const Real SwU = wj ? (ex_w ? son0 : Sw_j) : (ex_w ? usn1 : Sw_i);
    const Real SgU = gj ? (ex_g ? son1 : Sg_j) : (ex_g ? usn2 : Sg_i);
    const Real SwO = oj ? (ex_o ? son0 : Sw_j) : (ex_o ? usn1 : Sw_i);
    const Real SgO = oj ? (ex_o ? son1 : Sg_j) : (ex_o ? usn2 : Sg_i);

    // Each phase converts its flux at the cell it flowed OUT of, since that is
    // where its reservoir volume was measured.  Taking the b-factor at the
    // downstream cell instead loses mass across every face where pressure --
    // and so density -- differs, which for gas across a 500 psi drawdown is a
    // factor of order the pressure ratio.
    const PvtCell& Uw = wj ? N : F;
    const PvtCell& Uo = oj ? N : F;
    const PvtCell& Ug = gj ? N : F;
    const Real bo_u = Uo.bo, bw_u = Uw.bw, bg_u = Ug.bg, rs_u = Uo.rs;
    const Real rv_u = Ug.rv;

    // Unit B arguments: relperm3 then returns kr/mu, a reservoir mobility, and
    // the b_ph above applies the surface conversion exactly once.  See the
    // note at the top of this file.
    // ENDSCALE: a mobility is a property of the cell the phase flowed OUT
    // of, so it reads that cell's endpoints, exactly as the saturation and
    // the b-factor above already do.  A carries water and gas, which can in
    // principle be upwind of different cells; it follows the water donor,
    // which is the same approximation SwU/SgU already make.
    const int upw = wj ? nc : cell;
    const int upo = oj ? nc : cell;
    Mob3 A, B;
    if (wj == oj && oj == gj) {
      // With a common phase-potential direction the saturation (current or
      // AIM-frozen), endpoint cell and viscosities all belong to one donor.
      // The preparation kernel evaluated this exact cell property once.
      A = cell_props[upw].mob;
      B = A;
    } else {
      A = relperm3(SwU, SgU, T, Uw.muw, Uo.muo, Ug.mug,
                   1.0, 1.0, 1.0, P.swi, P.swr, upw);
      B = relperm3(SwO, SgO, T, Uw.muw, Uo.muo, Ug.mug,
                   1.0, 1.0, 1.0, P.swi, P.swr, upo);
    }
    // Inside the dead band the phase flux is identically zero, including its
    // pressure, capillary and mobility derivatives.  The switching surface is
    // nonsmooth by definition; away from it the derivative is unchanged.
    if (!flow_w) A.Mw = A.dMw_dSw = 0;
    if (Ug.mug > 0) {
      const Real dmu = A.Mg * Ug.dmug_dso / Ug.mug;
      A.dMg_dSw += dmu;
      A.dMg_dSg += dmu;
    }
    if (!flow_g) A.Mg = A.dMg_dSw = A.dMg_dSg = 0;
    if (!flow_o) B.Mo = B.dMo_dSw = B.dMo_dSg = 0;
    const Real Fw = Tg * A.Mw * dPw, Fo = Tg * B.Mo * dPo, Fg = Tg * A.Mg * dPg;
    R_oil -= (bo_u * Fo + rv_u * bg_u * Fg);
    R_wat -= bw_u * Fw;
    R_gas -= (bg_u * Fg + rs_u * bo_u * Fo);

    // The upstream b-factor depends on pressure too, and on WHOSE pressure
    // depends on which way the phase is flowing.  These slopes land on the
    // side the phase came from and are zero on the other.
    const Real dbo_i = oj ? 0.0 : F.dbo,  dbo_j = oj ? N.dbo : 0.0;
    const Real dbw_i = wj ? 0.0 : F.dbw,  dbw_j = wj ? N.dbw : 0.0;
    const Real dbg_i = gj ? 0.0 : F.dbg,  dbg_j = gj ? N.dbg : 0.0;
    // d(Rv.bg)/dp on the gas upstream side.
    const Real drvbg_i = gj ? 0.0 : (F.drv * bg_u + rv_u * F.dbg);
    const Real drvbg_j = gj ? (N.drv * bg_u + rv_u * N.dbg) : 0.0;
    // d(Rs.bo)/dp on the oil's upstream side, by the product rule.
    const Real drb_i = oj ? 0.0 : (F.drs * bo_u + rs_u * F.dbo);
    const Real drb_j = oj ? (N.drs * bo_u + rs_u * N.dbo) : 0.0;
    // This neighbour's block, complete here and written straight out.
    Real nJ[3][3] = {};
    sJ[0][0] += bo_u * Tg * B.Mo * ko_i +
                rv_u * bg_u * Tg * A.Mg * kg_i
                - dbo_i * Fo - drvbg_i * Fg;
    nJ[0][0] -= bo_u * Tg * B.Mo * ko_j +
                rv_u * bg_u * Tg * A.Mg * kg_j
                + dbo_j * Fo + drvbg_j * Fg;
    const Real dOilSw = -bo_u * Tg * dPo * B.dMo_dSw;
    const Real dOilSg = -bo_u * Tg * dPo * B.dMo_dSg;
    if (!ex_o) {
      if (!P.gas_primary) (oj ? nJ[0][1] : sJ[0][1]) += dOilSw;
      if (!P.gas_primary) (oj ? nJ[0][2] : sJ[0][2]) += dOilSg;
    }
    const Real dOilSw_g = -rv_u * bg_u * Tg * dPg * A.dMg_dSw;
    const Real dOilSg_g = -rv_u * bg_u * Tg * dPg * A.dMg_dSg;
    if (!ex_g) {
      if (!P.gas_primary) (gj ? nJ[0][1] : sJ[0][1]) += dOilSw_g;
      if (!P.gas_primary) (gj ? nJ[0][2] : sJ[0][2]) += dOilSg_g;
    }
    // R_oil -= Rv(So).bg(So).Fg and dSo/dSw=dSo/dSg=-1. Rv and
    // Bg are evaluated at the current gas donor even when AIM freezes mobility.
    const Real dRvBgSo_g =
        (Ug.drv_dso * bg_u + rv_u * Ug.dbg_dso) * Fg;
    if (!P.gas_primary)
      (gj ? nJ[0][1] : sJ[0][1]) += dRvBgSo_g;
    if (!P.gas_primary)
      (gj ? nJ[0][2] : sJ[0][2]) += dRvBgSo_g;
    if (T.have_pc) {
      if (!P.gas_primary) {
        sJ[0][2] += rv_u * bg_u * Tg * A.Mg * Ci.dpcog;
        nJ[0][2] += -rv_u * bg_u * Tg * A.Mg * Cj.dpcog;
      }
    }
    sJ[1][0] += bw_u * Tg * A.Mw * kw_i - dbw_i * Fw;
    nJ[1][0] -= bw_u * Tg * A.Mw * kw_j + dbw_j * Fw;
    const Real dWatSw = -bw_u * Tg * dPw * A.dMw_dSw;
    if (!ex_w && !P.gas_primary)
      (wj ? nJ[1][1] : sJ[1][1]) += dWatSw;
    // The potential itself depends on BOTH cells' saturations now, whichever
    // way the phase happens to be flowing, so these are not upstream-weighted
    // like the mobility terms above.  R_wat -= bw.Fw and Fw = Tg.Mw.dPw, with
    // d(dPw)/dSw_i = +Pcow'(Sw_i) and d(dPw)/dSw_j = -Pcow'(Sw_j).
    if (T.have_pc) {
      if (!P.gas_primary) {
        sJ[1][1] += -bw_u * Tg * A.Mw * Ci.dpcow;
        nJ[1][1] += bw_u * Tg * A.Mw * Cj.dpcow;
      }
    }
    sJ[2][0] += bg_u * Tg * A.Mg * kg_i +
                rs_u * bo_u * Tg * B.Mo * ko_i
                - dbg_i * Fg - drb_i * Fo;
    nJ[2][0] -= bg_u * Tg * A.Mg * kg_j +
                rs_u * bo_u * Tg * B.Mo * ko_j
                + dbg_j * Fg + drb_j * Fo;
    const Real dGasSw_g = -bg_u * Tg * dPg * A.dMg_dSw;
    const Real dGasSg_g = -bg_u * Tg * dPg * A.dMg_dSg;
    if (!ex_g) {
      if (!P.gas_primary) (gj ? nJ[2][1] : sJ[2][1]) += dGasSw_g;
      if (!P.gas_primary) (gj ? nJ[2][2] : sJ[2][2]) += dGasSg_g;
    }
    const Real dBgSo_g = Ug.dbg_dso * Fg;
    if (!P.gas_primary)
      (gj ? nJ[2][1] : sJ[2][1]) += dBgSo_g;
    if (!P.gas_primary)
      (gj ? nJ[2][2] : sJ[2][2]) += dBgSo_g;
    // Gas carries the opposite sign: dPg = dp + (Pcog_j - Pcog_i) - g.rho.dz,
    // so d(dPg)/dSg_i = -Pcog'(Sg_i) and d(dPg)/dSg_j = +Pcog'(Sg_j).
    if (T.have_pc) {
      if (!P.gas_primary) {
        sJ[2][2] += bg_u * Tg * A.Mg * Ci.dpcog;
        nJ[2][2] += -bg_u * Tg * A.Mg * Cj.dpcog;
      }
    }
    const Real dGasSw_o = -rs_u * bo_u * Tg * dPo * B.dMo_dSw;
    const Real dGasSg_o = -rs_u * bo_u * Tg * dPo * B.dMo_dSg;
    if (!ex_o) {
      if (!P.gas_primary) (oj ? nJ[2][1] : sJ[2][1]) += dGasSw_o;
      if (!P.gas_primary) (oj ? nJ[2][2] : sJ[2][2]) += dGasSg_o;
    }

    if (P.gas_primary) {
      const Real dsg_sw_i = rv_i ? (Real)-1 : (Real)0;
      const Real dsg_sw_j = rv_j ? (Real)-1 : (Real)0;
      const Real drho_w_sw_i = (Real)0, drho_w_sw_j = (Real)0;
      const Real drho_o_sw_i = F.drs_dsw * P.rhog / ft3_per_bbl * F.bo +
                               oil_mass_i * F.dbo_dsw;
      const Real drho_o_sw_j = N.drs_dsw * P.rhog / ft3_per_bbl * N.bo +
                               oil_mass_j * N.dbo_dsw;
      const Real drho_g_sw_i = F.drv_dsw * P.rhoo * F.bg +
                               gas_mass_i * F.dbg_dsw;
      const Real drho_g_sw_j = N.drv_dsw * P.rhoo * N.bg +
                               gas_mass_j * N.dbg_dsw;
      // These are derivatives of the FACE potential difference
      // Phi_j-Phi_i.  The self phase pressure therefore enters with a minus
      // sign while the neighbour enters with a plus sign.  On an Rv cell
      // Pg is primary and Po=Pg-Pcog(1-Sw), so dPo/dSw=+Pcog'; hence
      // d(Po_j-Po_i)/dSw_i=-Pcog' and the gas-potential derivative is zero.
      const Real dpo_phase_sw_i =
          rv_i ? -Ci.dpcog * dsg_sw_i : (Real)0;
      const Real dpo_phase_sw_j =
          rv_j ? -Cj.dpcog * dsg_sw_j : (Real)0;
      const Real dpo_sw_i = -dpo_phase_sw_i -
                             g * (Real)0.5 * drho_o_sw_i * dz;
      const Real dpo_sw_j = dpo_phase_sw_j -
                             g * (Real)0.5 * drho_o_sw_j * dz;
      const Real dpw_sw_i = -dpo_phase_sw_i + Ci.dpcow -
                             g * (Real)0.5 * drho_w_sw_i * dz;
      const Real dpw_sw_j = dpo_phase_sw_j - Cj.dpcow -
                             g * (Real)0.5 * drho_w_sw_j * dz;
      const Real dpg_sw_i = -dpo_phase_sw_i - Ci.dpcog * dsg_sw_i -
                             g * (Real)0.5 * drho_g_sw_i * dz;
      const Real dpg_sw_j = dpo_phase_sw_j + Cj.dpcog * dsg_sw_j -
                             g * (Real)0.5 * drho_g_sw_j * dz;
      const PvtCell& Yo = oj ? N : F;
      const Real dsg_sw_o = oj ? dsg_sw_j : dsg_sw_i;
      const Real dsg_sw_g = gj ? dsg_sw_j : dsg_sw_i;
      const Real dmw_sw = A.dMw_dSw;
      Real dmo_sw = B.dMo_dSw + B.dMo_dSg * dsg_sw_o;
      if (Yo.muo > 0) dmo_sw -= B.Mo * Yo.dmuo_dsw / Yo.muo;
      Real dmg_sw = A.dMg_dSw + A.dMg_dSg * dsg_sw_g;
      const Real dFw_i = Tg * (A.Mw * dpw_sw_i +
          ((!wj && !ex_w) ? dmw_sw * dPw : (Real)0));
      const Real dFw_j = Tg * (A.Mw * dpw_sw_j +
          (( wj && !ex_w) ? dmw_sw * dPw : (Real)0));
      const Real dFo_i_sw = Tg * (B.Mo * dpo_sw_i +
          ((!oj && !ex_o) ? dmo_sw * dPo : (Real)0));
      const Real dFo_j_sw = Tg * (B.Mo * dpo_sw_j +
          (( oj && !ex_o) ? dmo_sw * dPo : (Real)0));
      const Real dFg_i_sw = Tg * (A.Mg * dpg_sw_i +
          ((!gj && !ex_g) ? dmg_sw * dPg : (Real)0));
      const Real dFg_j_sw = Tg * (A.Mg * dpg_sw_j +
          (( gj && !ex_g) ? dmg_sw * dPg : (Real)0));
      const Real dbo_sw_i = oj ? (Real)0 : F.dbo_dsw;
      const Real dbo_sw_j = oj ? N.dbo_dsw : (Real)0;
      const Real dbw_sw_i = (Real)0, dbw_sw_j = (Real)0;
      const Real drvbg_sw_i = gj ? (Real)0 :
          (F.drv_dsw * bg_u + rv_u * F.dbg_dsw);
      const Real drvbg_sw_j = gj ?
          (N.drv_dsw * bg_u + rv_u * N.dbg_dsw) : (Real)0;
      const Real dbg_sw_i = gj ? (Real)0 : F.dbg_dsw;
      const Real dbg_sw_j = gj ? N.dbg_dsw : (Real)0;
      const Real drsbo_sw_i = oj ? (Real)0 :
          (F.drs_dsw * bo_u + rs_u * F.dbo_dsw);
      const Real drsbo_sw_j = oj ?
          (N.drs_dsw * bo_u + rs_u * N.dbo_dsw) : (Real)0;
      sJ[0][1] += -(dbo_sw_i * Fo + bo_u * dFo_i_sw +
                     drvbg_sw_i * Fg + rv_u * bg_u * dFg_i_sw);
      nJ[0][1] += -(dbo_sw_j * Fo + bo_u * dFo_j_sw +
                     drvbg_sw_j * Fg + rv_u * bg_u * dFg_j_sw);
      sJ[1][1] += -(dbw_sw_i * Fw + bw_u * dFw_i);
      nJ[1][1] += -(dbw_sw_j * Fw + bw_u * dFw_j);
      sJ[2][1] += -(dbg_sw_i * Fg + bg_u * dFg_i_sw +
                     drsbo_sw_i * Fo + rs_u * bo_u * dFo_i_sw);
      nJ[2][1] += -(dbg_sw_j * Fg + bg_u * dFg_j_sw +
                     drsbo_sw_j * Fo + rs_u * bo_u * dFo_j_sw);

      // Exact third-column chain rule.  x is Sg on saturated cells and Rs on
      // oil-only cells.  Form flux derivatives directly; this includes PVTO
      // composition, phase-potential density, capillary, mobility and the
      // surface-volume conversion at the actual upstream cell.
      const Real ds_i = (rs_i || rv_i) ? (Real)0 : (Real)1;
      const Real ds_j = (rs_j || rv_j) ? (Real)0 : (Real)1;
      const Real drho_o_x_i = F.drs_dx * P.rhog / ft3_per_bbl * F.bo +
                              oil_mass_i * F.dbo_dx;
      const Real drho_o_x_j = N.drs_dx * P.rhog / ft3_per_bbl * N.bo +
                              oil_mass_j * N.dbo_dx;
      const Real drho_g_x_i = F.drv_dx * P.rhoo * F.bg +
                              gas_mass_i * F.dbg_dx;
      const Real drho_g_x_j = N.drv_dx * P.rhoo * N.bg +
                              gas_mass_j * N.dbg_dx;
      const Real dpo_x_i = -g * (Real)0.5 * drho_o_x_i * dz;
      const Real dpo_x_j = -g * (Real)0.5 * drho_o_x_j * dz;
      const Real dpg_x_i = -Ci.dpcog * ds_i -
                            g * (Real)0.5 * drho_g_x_i * dz;
      const Real dpg_x_j = Cj.dpcog * ds_j -
                            g * (Real)0.5 * drho_g_x_j * dz;
      const PvtCell& Xo = oj ? N : F;
      const PvtCell& Xg = gj ? N : F;
      const bool xg_rv = gj ? rv_j : rv_i;
      const Real ds_o = oj ? ds_j : ds_i;
      const Real ds_g = gj ? ds_j : ds_i;
      Real dmo_x = B.dMo_dSg * ds_o;
      if (Xo.muo > 0) dmo_x -= B.Mo * Xo.dmuo_dx / Xo.muo;
      Real dmg_x = A.dMg_dSg * ds_g;
      // A.dMg_dSg already contains the VAPPARS viscosity chain on Sg cells.
      // Rv is different: saturation is fixed while gas viscosity varies
      // directly with the composition primary.  Omitting this very large
      // (per-unit-Rv) term leaves the Rv column inconsistent and makes phase
      // transitions stall even at tiny timesteps.
      if (xg_rv && Xg.mug > 0)
        dmg_x -= A.Mg * Xg.dmug_dx / Xg.mug;
      const Real dFo_i = Tg * (B.Mo * dpo_x_i +
          ((!oj && !ex_o) ? dmo_x * dPo : (Real)0));
      const Real dFo_j = Tg * (B.Mo * dpo_x_j +
          (( oj && !ex_o) ? dmo_x * dPo : (Real)0));
      const Real dFg_i = Tg * (A.Mg * dpg_x_i +
          ((!gj && !ex_g) ? dmg_x * dPg : (Real)0));
      const Real dFg_j = Tg * (A.Mg * dpg_x_j +
          (( gj && !ex_g) ? dmg_x * dPg : (Real)0));
      const Real dbo_u_i = oj ? (Real)0 : F.dbo_dx;
      const Real dbo_u_j = oj ? N.dbo_dx : (Real)0;
      const Real drvbg_u_i = gj ? (Real)0 :
          (F.drv_dx * bg_u + rv_u * F.dbg_dx);
      const Real drvbg_u_j = gj ?
          (N.drv_dx * bg_u + rv_u * N.dbg_dx) : (Real)0;
      const Real dbg_u_i = gj ? (Real)0 : F.dbg_dx;
      const Real dbg_u_j = gj ? N.dbg_dx : (Real)0;
      const Real drsbo_u_i = oj ? (Real)0 :
          (F.drs_dx * bo_u + rs_u * F.dbo_dx);
      const Real drsbo_u_j = oj ?
          (N.drs_dx * bo_u + rs_u * N.dbo_dx) : (Real)0;
      sJ[0][2] += -(dbo_u_i * Fo + bo_u * dFo_i +
                     drvbg_u_i * Fg + rv_u * bg_u * dFg_i);
      nJ[0][2] += -(dbo_u_j * Fo + bo_u * dFo_j +
                     drvbg_u_j * Fg + rv_u * bg_u * dFg_j);
      sJ[2][2] += -(dbg_u_i * Fg + bg_u * dFg_i +
                     drsbo_u_i * Fo + rs_u * bo_u * dFo_i);
      nJ[2][2] += -(dbg_u_j * Fg + bg_u * dFg_j +
                     drsbo_u_j * Fo + rs_u * bo_u * dFo_j);
    }

    const int coff = conn_bcol[m];
    if (coff >= 0)
      for (int a = 0; a < 3; ++a)
        for (int v = 0; v < 3; ++v)
          aval[rowptr[3 * cell + a] + coff + v] = nJ[a][v];
  }

  // --- wells (self) ---
  const WellCell w = well[cell];
  const Real well_po = p_i - (rv_i ? cell_props[cell].cap.pcog : (Real)0);
  const Real ddraw_prod_sw = rv_i ? cell_props[cell].cap.dpcog : (Real)0;
  // The residual before the producer terms, so the rate those terms add can
  // be reduced exactly as the residual sees it (slots 17..19, surface
  // units/day, cross-flow included).  main.cu integrates these over accepted
  // substeps for the NVRS_SUBSTEP_BALANCE ledger.
  const Real R_oil_pre_well = R_oil, R_wat_pre_well = R_wat,
             R_gas_pre_well = R_gas;
  if (w.is_prod) {
    const Real draw = well_po - w.pwf;
    Mob3 m = relperm3(Sw_i, Sg_i, T, F.muw, F.muo, F.mug,
                      1.0, 1.0, 1.0, P.swi, P.swr, cell);
    if (F.mug > 0) {
      const Real dmu = m.Mg * F.dmug_dso / F.mug;
      m.dMg_dSw += dmu;
      m.dMg_dSg += dmu;
    }
    if (draw >= 0 || !w.xflow_model) {
      // Forward flow, or the legacy reverse-flow formula (WELLXFLOW 1, or an
      // uncontrolled well): each phase at its own LOCAL formation mobility.
      // Unchanged from before crossflow modelling existed -- see
      // well_control.hpp for why that is wrong for a constrained well's
      // reverse completions, and right for everything else.
      R_oil += (bo * w.wi * m.Mo + rv * bg * w.wi * m.Mg) * draw;
      R_wat += bw * w.wi * m.Mw * draw;
      R_gas += (bg * w.wi * m.Mg + rs * bo * w.wi * m.Mo) * draw;
      sJ[0][0] += bo * w.wi * m.Mo + rv * bg * w.wi * m.Mg
                  + (d_bo * m.Mo + (d_rv * bg + rv * d_bg) * m.Mg)
                        * w.wi * draw;
      const Real dVapSo = (F.drv_dso * bg + rv * F.dbg_dso) *
                          w.wi * m.Mg * draw;
      sJ[0][1] += (bo * m.dMo_dSw + rv * bg * m.dMg_dSw) *
                  w.wi * draw - dVapSo;
      if (!P.gas_primary)
        sJ[0][2] += (bo * w.wi * m.dMo_dSg +
                     rv * bg * w.wi * m.dMg_dSg) * draw - dVapSo;
      sJ[1][0] += bw * w.wi * m.Mw + d_bw * w.wi * m.Mw * draw;
      sJ[1][1] += bw * w.wi * m.dMw_dSw * draw;
      sJ[2][0] += bg * w.wi * m.Mg + rs * bo * w.wi * m.Mo
                  + (d_bg * m.Mg + (d_rs * bo + rs * d_bo) * m.Mo) * w.wi * draw;
      const Real dBgSo = F.dbg_dso * w.wi * m.Mg * draw;
      sJ[2][1] += (bg * m.dMg_dSw + rs * bo * m.dMo_dSw) *
                  w.wi * draw - dBgSo;
      if (rv_i) {
        // Pg is primary, Po=Pg-Pcog(1-Sw), and Sg=1-Sw.  Complete
        // d(C_phase*draw)/dSw for the transformed Rv state.
        const Real co = bo * m.Mo + rv * bg * m.Mg;
        const Real cw = bw * m.Mw;
        const Real cg = bg * m.Mg + rs * bo * m.Mo;
        sJ[0][1] += w.wi *
            (draw * (-bo * m.dMo_dSg - rv * bg * m.dMg_dSg) +
             co * ddraw_prod_sw);
        sJ[1][1] += w.wi * cw * ddraw_prod_sw;
        sJ[2][1] += w.wi *
            (draw * (-bg * m.dMg_dSg - rs * bo * m.dMo_dSg) +
             cg * ddraw_prod_sw);
      }
      if (!P.gas_primary) {
        sJ[2][2] += (bg * w.wi * m.dMg_dSg +
                     rs * bo * w.wi * m.dMo_dSg) * draw;
        sJ[2][2] -= dBgSo;
      } else {
        const Real dsx = rs_i ? (Real)0 : (Real)1;
        Real dmo = m.dMo_dSg * dsx;
        if (F.muo > 0) dmo -= m.Mo * F.dmuo_dx / F.muo;
        Real dmg = m.dMg_dSg * dsx;
        if (rv_i && F.mug > 0) dmg -= m.Mg * F.dmug_dx / F.mug;
        const Real drvbg = F.drv_dx * bg + rv * F.dbg_dx;
        const Real drsbo = F.drs_dx * bo + rs * F.dbo_dx;
        sJ[0][2] += w.wi * draw *
            (F.dbo_dx * m.Mo + bo * dmo + drvbg * m.Mg + rv * bg * dmg);
        sJ[2][2] += w.wi * draw *
            (F.dbg_dx * m.Mg + bg * dmg + drsbo * m.Mo + rs * bo * dmo);
      }
    } else {
      // Crossflow, OPM's cmix_s branch (StandardWell::computePerfRate).
      // mix_{o,w,g} are surface-component fractions.  Convert one unit of
      // surface mixture to reservoir volume at THIS receiving completion:
      //
      //   Vr/Vs = cw/bw + (co-rv*cg)/(bo*(1-rs*rv))
      //                    + (cg-rs*co)/(bg*(1-rs*rv)).
      //
      // The old code treated mix as reservoir-phase fractions and multiplied
      // by b.  That is only valid when source and receiving PVT are identical,
      // and it is not the well equation Flow differentiates.
      const Real lam = m.Mo + m.Mw + m.Mg;
      const Real co = w.mix_o, cw = w.mix_w, cg = w.mix_g;
      constexpr Real kvol = (Real)5.614583333333333;
      // Fractions use one physical surface-volume basis. Oil and water are
      // already STB; convert gas scf to STB-equivalent volume.
      const Real bgc = bg / kvol, rsc = rs / kvol, rvc = rv * kvol;
      const Real dis = fmax((Real)1e-12, (Real)1 - rs * rv);
      const Real ao = co - rvc * cg;
      const Real ag = cg - rsc * co;
      const Real vro = ao / (bo * dis);
      const Real vrw = cw / bw;
      const Real vrg = ag / (bgc * dis);
      const Real volume_ratio = fmax((Real)1e-20, vro + vrw + vrg);
      const Real q_res = w.wi * lam * draw; // <= 0 reservoir bbl/day
      const Real q_s = q_res / volume_ratio;
      R_oil += co * q_s;
      R_wat += cw * q_s;
      R_gas += kvol * cg * q_s;

      Real dlam[3] = {
          (Real)0,
          m.dMo_dSw + m.dMw_dSw + m.dMg_dSw,
          m.dMo_dSg + m.dMg_dSg};
      Real ddraw[3] = {(Real)1, (Real)0, (Real)0};
      if (rv_i) {
        dlam[1] -= m.dMo_dSg + m.dMg_dSg;
        ddraw[1] = ddraw_prod_sw;
      }
      if (P.gas_primary) {
        const Real dsx = rs_i ? (Real)0 : (Real)1;
        Real dmo = m.dMo_dSg * dsx;
        if (F.muo > 0) dmo -= m.Mo * F.dmuo_dx / F.muo;
        Real dmg = m.dMg_dSg * dsx;
        if (rv_i && F.mug > 0) dmg -= m.Mg * F.dmug_dx / F.mug;
        dlam[2] = dmo + dmg;
      }
      const Real dbo_v[3] = {F.dbo, F.dbo_dsw, F.dbo_dx};
      const Real dbw_v[3] = {F.dbw, (Real)0, (Real)0};
      const Real dbg_v[3] = {F.dbg / kvol, F.dbg_dsw / kvol,
                              F.dbg_dx / kvol};
      const Real drs_v[3] = {F.drs / kvol, F.drs_dsw / kvol,
                              F.drs_dx / kvol};
      const Real drv_v[3] = {F.drv * kvol, F.drv_dsw * kvol,
                              F.drv_dx * kvol};
      const Real cmix[3] = {co, cw, kvol * cg};
      for (int v = 0; v < 3; ++v) {
        const Real ddis = -(drs_v[v] * rvc + rsc * drv_v[v]);
        const Real dao = -drv_v[v] * cg;
        const Real dag = -drs_v[v] * co;
        const Real dvro = dao / (bo * dis) -
            vro * (dbo_v[v] / bo + ddis / dis);
        const Real dvrw = -cw * dbw_v[v] / (bw * bw);
        const Real dvrg = dag / (bgc * dis) -
            vrg * (dbg_v[v] / bgc + ddis / dis);
        const Real dvr = dvro + dvrw + dvrg;
        const Real dqres =
            w.wi * (dlam[v] * draw + lam * ddraw[v]);
        const Real dqs =
            (dqres * volume_ratio - q_res * dvr) /
            (volume_ratio * volume_ratio);
        sJ[0][v] += cmix[0] * dqs;
        sJ[1][v] += cmix[1] * dqs;
        sJ[2][v] += cmix[2] * dqs;
      }
    }
  }
  if (w.is_prod) {
    atomicAdd(rnorm2 + 17, R_oil - R_oil_pre_well);
    atomicAdd(rnorm2 + 18, R_wat - R_wat_pre_well);
    atomicAdd(rnorm2 + 19, R_gas - R_gas_pre_well);
  }
  // Injection enters the equation of the phase being injected.
  //
  // No b-factor on the rate, and this is not an oversight.  A WCONINJE rate is
  // a SURFACE rate and these equations are already in surface volumes, so it
  // enters as it stands.  Converting it again with b = 1/B multiplies it by
  // 1231 for Norne's gas.  The producer terms above do carry b, because there
  // the well index and mobility produce a RESERVOIR rate that still needs
  // converting.
  //
  // The rate is a REQUEST, not a fact.  An injector also carries a bottom-hole
  // limit -- 600 bar for C-4H -- and cannot put away more than the formation
  // will take at that pressure:
  //
  //     q_reservoir  <=  WI . lambda_total . (p_bhp - p_cell)
  //
  // Without that cap a rate-controlled injector is a source that cannot be
  // argued with: it delivers in full however high the cell pressure climbs,
  // nothing pushes back, and no timestep is small enough to rescue it.  With
  // it the well simply moves onto pressure control when the formation tightens,
  // which is what a real one does -- and the cap brings a derivative with it,
  // putting a positive term on the pressure diagonal exactly where the
  // injector is.
  if (w.is_inj) {
    Real q_w = w.qw_inj, q_o = w.qo_inj, q_g = w.qg_inj;
    Real dRo_dp = 0, dRw_dp = 0, dRg_dp = 0;
    Real dRo_dSw = 0, dRo_dSg = 0;
    Real dRw_dSw = 0, dRw_dSg = 0;
    Real dRg_dSw = 0, dRg_dSg = 0;

    if (w.pwf > 0 && w.wi > 0) {
      Mob3 m = relperm3(Sw_i, Sg_i, T, F.muw, F.muo, F.mug,
                        1.0, 1.0, 1.0, P.swi, P.swr, cell);
      // Injectivity is against the TOTAL mobility: what goes in has to push
      // everything already there out of the way.
      const Real lam = m.Mw + m.Mo + m.Mg;
      const Real dlam_dSw = m.dMw_dSw + m.dMo_dSw + m.dMg_dSw -
                            (rv_i ? m.dMo_dSg + m.dMg_dSg : (Real)0);
      const Real dlam_dSg = m.dMo_dSg + m.dMg_dSg;
      Real draw = w.pwf - well_po;
      if (draw < 0) draw = 0;                  // at the limit, nothing goes in
      const Real q_res = w.wi * lam * draw;    // reservoir bbl/day
      const Real dqres_dsw = draw > 0
          ? w.wi * (dlam_dSw * draw - lam * ddraw_prod_sw) : (Real)0;

      if (q_g > 0) {
        const Real cap = q_res * bg;           // the same, as scf/day
        if (cap < q_g) {
          atomicAdd(rnorm2 + 5, (Real)1);      // on pressure control
          q_g = cap;
          dRg_dp  = bg * w.wi * lam;           // d(-q)/dp, positive
          dRg_dSw = -bg * dqres_dsw - F.dbg_dsw * q_res;
          if (!P.gas_primary)
            dRg_dSg = -bg * w.wi * draw * dlam_dSg;
          else {
            const Real dsx = rs_i ? (Real)0 : (Real)1;
            Real dmo = m.dMo_dSg * dsx;
            if (F.muo > 0) dmo -= m.Mo * F.dmuo_dx / F.muo;
            Real dmg = m.dMg_dSg * dsx;
            if (rv_i && F.mug > 0) dmg -= m.Mg * F.dmug_dx / F.mug;
            dRg_dSg = -(F.dbg_dx * q_res +
                         bg * w.wi * draw * (dmo + dmg));
          }
        }
      }
      if (q_w > 0) {
        const Real cap = q_res * bw;           // stb/day
        if (cap < q_w) {
          q_w = cap;
          dRw_dp  = bw * w.wi * lam;
          dRw_dSw = -bw * dqres_dsw;
        }
      }
      if (q_o > 0) {
        const Real cap = q_res * bo;
        if (cap < q_o) {
          q_o = cap;
          dRo_dp = bo * w.wi * lam;
          dRo_dSw = -bo * dqres_dsw - F.dbo_dsw * q_res;
        }
      }
    }

    if (w.inj_rate_coupled && w.inj_dqdp > 0) {
      if (q_g > 0) {
        dRg_dp -= w.inj_dqdu[0];
        dRg_dSw -= w.inj_dqdu[1];
        dRg_dSg -= w.inj_dqdu[2];
      } else if (q_w > 0) {
        dRw_dp -= w.inj_dqdu[0];
        dRw_dSw -= w.inj_dqdu[1];
        dRw_dSg -= w.inj_dqdu[2];
      } else if (q_o > 0) {
        dRo_dp -= w.inj_dqdu[0];
        dRo_dSw -= w.inj_dqdu[1];
        dRo_dSg -= w.inj_dqdu[2];
      }
    }
    if (w.inj_xflow_coupled) {
      for (int eq = 0; eq < 3; ++eq)
        for (int v = 0; v < 3; ++v)
          sJ[eq][v] -= w.inj_direct[3 * eq + v];
    }

    R_wat -= q_w;
    R_oil -= q_o;
    R_gas -= q_g;
    // Injected surface rates, after the formation cap, for the ledger.
    atomicAdd(rnorm2 + 20, q_o);
    atomicAdd(rnorm2 + 21, q_w);
    atomicAdd(rnorm2 + 22, q_g);
    sJ[0][0] += dRo_dp;  sJ[0][1] += dRo_dSw;  sJ[0][2] += dRo_dSg;
    sJ[1][0] += dRw_dp;  sJ[1][1] += dRw_dSw;  sJ[1][2] += dRw_dSg;
    sJ[2][0] += dRg_dp;  sJ[2][1] += dRg_dSw;  sJ[2][2] += dRg_dSg;
  }

  b[3 * cell] = -R_oil; b[3 * cell + 1] = -R_wat; b[3 * cell + 2] = -R_gas;
  atomicAdd(rnorm2, R_oil * R_oil + R_wat * R_wat + R_gas * R_gas);
  // Per-equation, because the three are not on the same scale and it matters.
  // In surface volumes a reservoir barrel of gas is ~1231 scf where a barrel
  // of oil is ~0.75 stb, so the gas row is inherently three orders larger.
  // Newton measures one norm over all three and therefore mostly measures gas;
  // a badly converged oil equation can hide under a tolerance the gas row set.
  atomicAdd(rnorm2 + 1, R_oil * R_oil);
  atomicAdd(rnorm2 + 2, R_wat * R_wat);
  atomicAdd(rnorm2 + 3, R_gas * R_gas);
  atomicAdd(rnorm2 + 11, R_oil);
  atomicAdd(rnorm2 + 12, R_wat);
  atomicAdd(rnorm2 + 13, R_gas);
  const Real pvc = Vc * poro[cell];
  atomicAdd(rnorm2 + 6, pvc);              // pore volume [rb]
  // CNV, after OPM: the worst single cell, as a fraction of its own contents.
  // A material balance is a sum and lets a badly wrong cell hide behind
  // everything that is right; this does not.  Both are needed, which is why
  // Flow tests both.
  if (pvc > 0 && cnv) {
    const Real sc = P.dt / pvc;
    const Real c_oil = fabs(R_oil) * sc / (bo > 0 ? bo : 1);
    const Real c_wat = fabs(R_wat) * sc / (bw > 0 ? bw : 1);
    const Real c_gas = fabs(R_gas) * sc / (bg > 0 ? bg : 1);
    atomic_max_pos(cnv + 0, c_oil);
    atomic_max_pos(cnv + 1, c_wat);
    atomic_max_pos(cnv + 2, c_gas);
    // How much of the field is responsible for a CNV over tolerance.  OPM
    // accepts a component whose worst cell fails so long as the cells that
    // fail hold a small enough share of the pore volume, so the share has to
    // be measured while the per-cell residual is still in a register.
    if (P.cnv_tol > 0) {
      if (c_oil > P.cnv_tol) atomicAdd(rnorm2 + 7, pvc);
      if (c_wat > P.cnv_tol) atomicAdd(rnorm2 + 8, pvc);
      if (c_gas > P.cnv_tol) atomicAdd(rnorm2 + 9, pvc);
      if (fmax(c_oil, fmax(c_wat, c_gas)) > P.cnv_tol)
        atomicAdd(rnorm2 + 10, pvc);
    }
  }

  // --- CPR: stage-2 block inverse and quasi-IMPES pressure weights ---
  Real flat[9] = {sJ[0][0], sJ[0][1], sJ[0][2], sJ[1][0], sJ[1][1],
                  sJ[1][2], sJ[2][0], sJ[2][1], sJ[2][2]};
  Real Inv[9];
  const bool ok = invert_block(flat, 3, Inv);
  store_dinv(Inv, ok, dinv + (long long)cell * 9, 3);
  // A cell block that will not invert is a hole in the preconditioner: stage 2
  // falls back to fixed weights there, and the CPR restriction stops matching
  // the operator it is meant to precondition.  Count them.
  if (!ok) atomicAdd(rnorm2 + 4, (Real)1);
  Real wgt[3];
  if (ok) { wgt[0] = Inv[0]; wgt[1] = Inv[1]; wgt[2] = Inv[2]; }
  else    { fallback_weights(wgt, 3); }

  // CPRDECOUPLE DRS / SUM.  The component equations are in surface volumes,
  // so each is first put on a reservoir-volume scale (multiplied by its FVF,
  // 1/b); a sum of those is a volume balance, the IMPES pressure equation.
  // DRS keeps an equation in the sum only where its own pressure row is
  // diagonally dominant to within drs_eps -- a_pp >= drs_eps * sum_j |a_pj| --
  // which is what stops a row sum from destroying the M-matrix character AMG
  // relies on where a phase appears or vanishes.  The weights are normalised
  // so the pressure diagonal is one, as it is under quasi-IMPES.  Where no
  // equation qualifies, quasi-IMPES stands.
  if (P.cpr_decouple != 0) {
    const Real sc[3] = {bo > 0 ? 1 / bo : (Real)1, bw > 0 ? 1 / bw : (Real)1,
                        bg > 0 ? 1 / bg : (Real)1};
    Real w2[3] = {0, 0, 0};
    if (P.cpr_decouple == 2) {
      for (int a = 0; a < 3; ++a) w2[a] = sc[a];
    } else {
      for (int a = 0; a < 3; ++a) {
        Real off = 0;
        for (int m = c0; m < c1; ++m) {
          const int coff = conn_bcol[m];
          if (coff >= 0) off += fabs(aval[rowptr[3 * cell + a] + coff]);
        }
        const Real d = sJ[a][0];
        if (d > 0 && d >= P.drs_eps * off) w2[a] = sc[a];
      }
    }
    const Real diag = w2[0] * sJ[0][0] + w2[1] * sJ[1][0] + w2[2] * sJ[2][0];
    if (diag > (Real)1e-300 && isfinite(diag))
      for (int a = 0; a < 3; ++a) wgt[a] = w2[a] / diag;
  }
  for (int a = 0; a < 3; ++a) pweight[3 * cell + a] = (PrecReal)wgt[a];

  // self block
  const int soff = self_bcol[cell];
  if (soff >= 0) {
    for (int a = 0; a < 3; ++a)
      for (int v = 0; v < 3; ++v)
        aval[rowptr[3 * cell + a] + soff + v] = sJ[a][v];
    const int spc = self_pcol[cell];
    if (spc >= 0) {
      const Real Jp[3] = {sJ[0][0], sJ[1][0], sJ[2][0]};
      pval[prowptr[cell] + spc] = (PrecReal)pressure_row_entry(wgt, Jp, 3);
    }
  }

  // Pressure row for the neighbours.  Their blocks are already in the CSR, so
  // the column-0 entries -- which are exactly dR_a/dp_j -- are read back
  // rather than kept in registers.
  for (int m = c0; m < c1; ++m) {
    const int pcp = conn_pcol[m];
    const int coff = conn_bcol[m];
    if (pcp < 0 || coff < 0) continue;
    const Real Jp[3] = {aval[rowptr[3 * cell + 0] + coff],
                        aval[rowptr[3 * cell + 1] + coff],
                        aval[rowptr[3 * cell + 2] + coff]};
    pval[prowptr[cell] + pcp] = (PrecReal)pressure_row_entry(wgt, Jp, 3);
  }
}

#ifdef NVRS_HAVE_TENSORCORE
#include "tc_assemble.inc"
#endif

// Host launcher: fills J and rhs=-R for the current state; returns global ||R||.
Real assemble_blackoil(Reservoir& r, DeviceLinearSystem& sys) {
  sys.well_schur = r.well_schur_enabled ? r.well_schur : nullptr;
  sys.well_schur_count = r.well_schur_enabled ? r.well_schur_count : 0;
  sys.well_schur_groups = r.well_schur_enabled ? r.well_schur_groups : 0;
  sys.well_schur_modes = (r.well_schur_enabled == 1 ||
                          r.well_schur_enabled == 8)
                             ? 1
                             : ((r.well_schur_enabled == 9 && r.bs == 2) ? 2 : 3);
  BOParams P = make_bo_params(r);
  // Local block extents: the kernels index cells in this rank' own
  // ordering, and use the product only as a cell count.
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
  // [0] total, [1..3] per equation, [4] cell blocks that would not invert,
  // [5] injector cells that hit their bottom-hole limit, [6] pore volume.
  Real*& d_rn = r.assembly_residual_stats;
  // [7..9] strict-CNV violating PV per component, [10] their cell-wise union.
  if (!d_rn) NVRS_CHECK_CUDA(cudaMalloc(&d_rn, sizeof(Real) * 23));
  NVRS_CHECK_CUDA(cudaMemset(d_rn, 0, sizeof(Real) * 23));
  unsigned long long*& d_cnv = r.assembly_cnv_stats;
  if (!d_cnv)
    NVRS_CHECK_CUDA(cudaMalloc(&d_cnv, sizeof(unsigned long long) * 3));
  NVRS_CHECK_CUDA(cudaMemset(d_cnv, 0, sizeof(unsigned long long) * 3));

  // Refresh the ghost layers: the assembly reads neighbour primary variables
  // and the Newton update only touched the owned cells.  An unstructured grid
  // exchanges arbitrary cell lists; a structured one exchanges slabs.
  if (r.halo_conn)
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.u, r.bs);
  else
    halo_exchange(r.u, r.bs, r.grid);

  AssemblyCellProps* cell_props = nullptr;
  if (r.jacobian == 1 && P.bs == 3 && r.use_conn_list && r.conn_ptr) {
    if (r.assembly_pvt_ready) {
      cell_props = static_cast<AssemblyCellProps*>(r.assembly_pvt_cache);
      r.assembly_pvt_ready = 0;
    } else {
      cell_props = prepare_assembly_props(r, P);
    }
  }

  auto* well = static_cast<WellCell*>(r.wellcells);

  // A grid with no fixed stencil can only be assembled by the connection-list
  // kernel.  Every other path below reads nbr[], tface[] and cellk[], which
  // are null for such a grid; launching one dereferences null on the device,
  // which does not reliably fault and instead returns garbage that looks like
  // a hard reservoir.  Refuse instead, and name the reason.
  if (!r.nbr && !(r.use_conn_list && r.conn_ptr && r.jacobian == 1 &&
                  (P.bs == 2 || P.bs == 3))) {
    throw std::runtime_error(
        "this grid has no six-face stencil, so it can only be assembled by "
        "the connection-list kernel, which covers two/three-phase black oil "
        "with JACOBIAN ANALYTIC. Select MODEL BLACKOIL and JACOBIAN "
        "ANALYTIC, or run a structured grid.");
  }

#ifdef NVRS_HAVE_TENSORCORE
  const double tc_t0 = tc::enabled() ? tc_clock() : 0.0;
#endif
  // Analytic paths are exact for two- and three-phase black oil; the numerical
  // (finite-difference) assembly is the fallback and covers all block sizes.
  if (r.jacobian == 1 && P.bs == 2 && r.use_conn_list && r.conn_ptr) {
    k_assemble_bo2_analytic_conn<<<grid1d(cells), 256>>>(
        r.u, r.u_old, r.poro, r.conn_ptr, r.conn_nbr, r.conn_trans,
        r.conn_dz, r.conn_threshold, r.conn_bcol, r.conn_pcol,
        r.self_bcol, r.self_pcol, sys.rowptr, sys.prowptr, well, r.aimflag,
        r.cellvol, r.tables, P, V, sys.b, sys.aval64, sys.pval, sys.dinv,
        sys.pweight, d_rn, d_cnv);
  } else if (r.jacobian == 1 && P.bs == 2) {
    k_assemble_bo_analytic<<<grid1d(cells), 256>>>(
        r.u, r.u_old, r.poro, r.tface, r.nbr, r.cellk, sys.rowptr, r.bcol,
        sys.prowptr, r.pcolpos, well, r.aimflag, r.tables, P, V, sys.b,
        sys.aval64, sys.pval, sys.dinv, sys.pweight, d_rn);
  } else if (r.jacobian == 1 && P.bs == 3 && r.use_conn_list && r.conn_ptr) {
    // The connection-list path.  Required for any grid whose cells do not
    // have exactly six neighbours -- corner-point geometry with faults -- and
    // equal to the fixed-stencil path to round-off on a grid that does.
    k_assemble_bo3_analytic_conn<<<grid1d(cells), 256>>>(
        r.u, r.u_old, r.poro, r.conn_ptr, r.conn_nbr, r.conn_trans, r.conn_dz,
        r.conn_threshold, r.conn_bcol, r.conn_pcol, r.self_bcol, r.self_pcol, sys.rowptr,
        sys.prowptr, well, r.aimflag, r.cellvol, cell_props, r.tables, P, V,
        sys.b,
        sys.aval64, sys.pval, sys.dinv, sys.pweight, d_rn, d_cnv);
  } else if (r.jacobian == 1 && P.bs == 3) {
    k_assemble_bo3_analytic<<<grid1d(cells), 256>>>(
        r.u, r.u_old, r.poro, r.tface, r.nbr, r.cellk, sys.rowptr, r.bcol,
        sys.prowptr, r.pcolpos, well, r.aimflag, r.tables, P, V, sys.b,
        sys.aval64, sys.pval, sys.dinv, sys.pweight, d_rn);
  } else {
    k_assemble_bo<<<grid1d(cells), 256>>>(
        r.u, r.u_old, r.poro, r.tface, r.nbr, r.cellk, sys.rowptr, r.bcol,
        sys.prowptr, r.pcolpos, well, r.tables, P, V, sys.b, sys.aval64,
        sys.pval, sys.dinv, sys.pweight, d_rn);
  }
  NVRS_CHECK_CUDA(cudaGetLastError());
#ifdef NVRS_HAVE_TENSORCORE
  if (tc::enabled())
    tc_replace_connection_blocks(r, sys, P, cell_props, tc_t0);
#endif

  Real rn2 = 0;
  Real rn_all[23] = {};
  NVRS_CHECK_CUDA(cudaMemcpy(rn_all, d_rn, sizeof(Real) * 23,
                             cudaMemcpyDeviceToHost));
  unsigned long long cnv_bits_pre[3] = {};
  NVRS_CHECK_CUDA(cudaMemcpy(cnv_bits_pre, d_cnv, sizeof(cnv_bits_pre),
                             cudaMemcpyDeviceToHost));

  // Make every number the Newton test reads a GLOBAL one, before anything is
  // derived from it.
  //
  // Reducing only the total norm was not enough: main.cu builds the OPM-style
  // material-balance and CNV criteria out of res_eq[] and cnv[], and those
  // were per-rank.  Four ranks then answered "has this Newton step converged"
  // four different ways, left the loop at different iterations, and deadlocked
  // in whatever collective the fastest rank had already moved past.  The sums
  // of squares reduce with SUM (the norm is over the whole field); CNV is a
  // per-cell maximum and reduces with MAX.
  if (r.mpi_nranks > 1) {
    // Slots 7..10 are pore volumes and 11..13 signed residuals; all reduce
    // with SUM like the squared norms.
    double buf[23];
    for (int a = 0; a < 23; ++a) buf[a] = (double)rn_all[a];
    MPI_Allreduce(MPI_IN_PLACE, buf, 23, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int a = 0; a < 23; ++a) rn_all[a] = (Real)buf[a];

    double cbuf[3];
    for (int a = 0; a < 3; ++a) {
      const long long b = (long long)cnv_bits_pre[a];
      double v; std::memcpy(&v, &b, sizeof(v));
      cbuf[a] = (cnv_bits_pre[a] == 0) ? 0.0 : v;
    }
    MPI_Allreduce(MPI_IN_PLACE, cbuf, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for (int a = 0; a < 3; ++a) {
      long long b; std::memcpy(&b, &cbuf[a], sizeof(b));
      cnv_bits_pre[a] = (cbuf[a] == 0.0) ? 0ull : (unsigned long long)b;
    }
  }

  rn2 = rn_all[0];
  r.res_eq[0] = std::sqrt((double)rn_all[1]);
  r.res_eq[1] = std::sqrt((double)rn_all[2]);
  r.res_eq[2] = std::sqrt((double)rn_all[3]);
  r.n_singular = (int)rn_all[4];
  r.n_inj_bhp = (int)rn_all[5];
  r.pv_local = rn_all[6];
  for (int a = 0; a < 3; ++a) r.cnv_pv[a] = rn_all[7 + a];
  r.cnv_bad_pv = rn_all[10];
  const bool stable_two_phase_ledger =
      r.bs == 2 && r.jacobian == 1 && r.use_conn_list && r.conn_ptr;
  if (stable_two_phase_ledger) {
    r.res_sum[0] = rn_all[14];
    r.res_sum[1] = rn_all[15];
    r.res_sum[2] = 0;
  } else {
    for (int a = 0; a < 3; ++a) r.res_sum[a] = rn_all[11 + a];
  }
  for (int a = 0; a < 3; ++a) {
    r.well_prod[a] = rn_all[17 + a];
    r.well_inj[a] = rn_all[20 + a];
  }
  unsigned long long cnv_bits[3] = {cnv_bits_pre[0], cnv_bits_pre[1],
                                    cnv_bits_pre[2]};
  for (int a = 0; a < 3; ++a) {
    const long long b = (long long)cnv_bits[a];
    double v; std::memcpy(&v, &b, sizeof(v));
    r.cnv[a] = (cnv_bits[a] == 0) ? (Real)0 : (Real)v;
  }

  // Once per run, say how the three equations compare.  NVRS_RES_DIAG=n for
  // the first n assemblies.
  {
    static int budget = -1;
    if (budget < 0) {
      const char* e = std::getenv("NVRS_RES_DIAG");
      budget = e ? std::atoi(e) : 1;
    }
    const bool want = (budget > 0) || (r.diag_budget > 0);
    if (want && P.bs == 3 && r.use_conn_list && r.mpi_rank == 0) {
      if (budget > 0) --budget; else --r.diag_budget;
      const double ro = std::sqrt((double)rn_all[1]);
      const double rw = std::sqrt((double)rn_all[2]);
      const double rg = std::sqrt((double)rn_all[3]);
      std::printf("\n  [res] residual by equation\n");
      std::printf("        oil %.4e   water %.4e   gas %.4e\n",
                  ro, rw, rg);
      std::printf("        gas/oil %.4g   gas/water %.4g\n",
                  ro > 0 ? rg / ro : 0.0, rw > 0 ? rg / rw : 0.0);
      if (rn_all[4] > 0)
        std::printf("        cell blocks that would not invert: %.0f of %d\n",
                    (double)rn_all[4], cells);
      std::fflush(stdout);
    }
  }
  // rn2 was reduced with the rest of rn_all above; reducing it again here
  // would square the field's residual on every assembly.

  // The df64 (hi,lo) slices of the operator are refreshed by the caller via
  // split_operator_df64() once the values are final; see main.cu.
  return std::sqrt(rn2);
}

// ===========================================================================
// Adaptive implicit (AIM 1): per-cell implicitness flags.
//
// A cell is explicit-in-saturation when its volumetric throughput per pore
// volume over the coming dt is below AIMCFL: the saturation there moves too
// slowly for implicitness to matter, so its face mobilities are frozen at
// t^n and its saturation couplings leave the Jacobian (see the assembly
// kernels).  Cells above the threshold, and every well cell, stay fully
// implicit — fronts and wells are exactly where FIM earns its keep.  The
// accumulation term stays implicit in every cell, so each row remains
// well-posed and the same Newton loop, residual test and CPR solve run
// unchanged; explicit cells simply stop feeding mobility derivatives in.
// ===========================================================================
// A note on the b-factors below, which look like the double count fixed at the
// top of this file and are not.
//
// This kernel does not conserve anything.  It measures how fast a saturation
// could change -- a throughput per pore volume -- and compares it against eta,
// a tuned constant, to decide which cells may be advanced explicitly.  Only the
// RATIO of rate to eta means anything, so the scale of the mobility is a
// calibration, not a dimension.
//
// Keeping kr/(mu.B) here keeps that calibration where it was.  Changing it to
// kr/mu -- which is what the residual needs -- divides the gas contribution by
// 1/Bg = 1231 on Norne, and the CFL then cannot see gas moving at all: the
// explicit fraction went from 57-65% to 75-95%, gas saturations were advanced
// explicitly in cells where gas was moving fast, and the Newton solve came
// apart.  eta would have to be retuned by three orders of magnitude to mean
// the same thing, so the b-factors stay and eta keeps its meaning.
__global__ static void k_aim_rate(const Real* u, const Real* poro,
                                  const Real* tface, const int* nbr,
                                  const int* cellk, SatTables T, BOParams P,
                                  Real V, Real* rate) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int ncell = P.nx * P.ny * P.nz;
  if (cell >= ncell) return;
  const int bs = P.bs;
  const Real g = P.gravity ? P.grav_grad : 0.0;
  const Real p_i = u[bs * cell];
  const Real Sw_i = u[bs * cell + 1];
  const Real Sg_i = (bs == 3) ? u[bs * cell + 2] : 0.0;
  const Real zc = (cellk[cell] + 0.5) * P.dz_cell;
  // Volumetric throughput [rb/day], counting BOTH directions with the
  // DONOR's mobility per phase per face.  Outflux alone is not enough: the
  // cell just downstream of a front has almost no outflux (its own water is
  // still immobile) but a large influx from a mobile implicit donor, and
  // flagging it explicit lets the frozen-rate influx overfill it past the
  // physical bounds — the saturation clamp then blocks the residual from
  // closing and Newton stalls at any dt.  A cell's saturation moves as fast
  // as the larger of what enters and what leaves.
  Real thru = 0.0;
  for (int f = 0; f < 6; ++f) {
    int nc = nbr[7 * cell + 1 + f];
    if (nc < 0) continue;
    Real Tg = tface[6 * cell + f];
    if (Tg <= 0) continue;
    Real p_j = u[bs * nc];
    Real Sw_j = u[bs * nc + 1];
    Real Sg_j = (bs == 3) ? u[bs * nc + 2] : 0.0;
    Real zn = (cellk[nc] + 0.5) * P.dz_cell;
    Real dPw = (p_j - g * P.rhow * zn) - (p_i - g * P.rhow * zc);
    Real dPo = (p_j - g * P.rhoo * zn) - (p_i - g * P.rhoo * zc);
    // The stability limit of an explicit saturation update is set by the
    // SENSITIVITY of the flux to saturation, dM/dS, not by the mobility
    // itself: the linearised criterion is dt/PV * sum_f T |dPhi| |dM/dS| < 1.
    // Near a Corey front dM/dS is several times M, so using M underestimates
    // the CFL, marks front-adjacent cells safe when they are not, and the
    // timestep controller then collapses to whatever those cells tolerate
    // (measured: dt stuck at 0.39 days, 2381 Newtons for a 400-day slice).
    if (bs == 2) {
      Mob2 mi = relperm2_table(Sw_i, T, P.uw, P.uo, P.bw, P.bo,
                               P.swi, P.swr);
      Mob2 mj = relperm2_table(Sw_j, T, P.uw, P.uo, P.bw, P.bo,
                               P.swi, P.swr);
      thru += Tg * fabs(dPw > 0 ? mj.dMw : mi.dMw) * fabs(dPw);
      thru += Tg * fabs(dPo > 0 ? mj.dMo : mi.dMo) * fabs(dPo);
    } else {
      Mob3 mi = relperm3(Sw_i, Sg_i, T, P.uw, P.uo, P.ug, P.bw, P.bo, P.bg,
                         P.swi, P.swr);
      Mob3 mj = relperm3(Sw_j, Sg_j, T, P.uw, P.uo, P.ug, P.bw, P.bo, P.bg,
                         P.swi, P.swr);
      Real dPg = (p_j - g * P.rhog * zn) - (p_i - g * P.rhog * zc);
      const Mob3& dw = (dPw > 0) ? mj : mi;
      const Mob3& dn = (dPo > 0) ? mj : mi;
      const Mob3& dg = (dPg > 0) ? mj : mi;
      thru += Tg * fabs(dw.dMw_dSw) * fabs(dPw);
      // Oil mobility moves with both saturations; take the worst case.
      thru += Tg * (fabs(dn.dMo_dSw) + fabs(dn.dMo_dSg)) * fabs(dPo);
      thru += Tg * fabs(dg.dMg_dSg) * fabs(dPg);
    }
  }
  const Real pv = V * poro[cell];
  rate[cell] = thru / (pv > (Real)0 ? pv : (Real)1e-30);   // [pore volumes/day]
}

// The same throughput measure over a general connection list.
//
// This has to exist, not merely be nice to have.  The fixed-stencil version
// above reads nbr[], tface[] and cellk[], and a corner-point model has none of
// them -- build_sparsity_field leaves all three null because there is no
// six-face stencil to fill them from.  Calling it anyway dereferences null on
// the device, which does not reliably fault; it returns garbage.  AIM then
// flags cells explicit from that garbage, the residual freezes donor
// mobilities in cells the Jacobian does not treat as explicit, and the two
// stop describing the same function.  Newton cannot converge on an
// inconsistent pair at any timestep, however small -- which is precisely how
// this presented: linear solves converging, Newton failing down to a tenth of
// a day.
__global__ static void k_aim_rate_conn(const Real* u, const Real* poro,
                                       const int* conn_ptr, const int* conn_nbr,
                                       const Real* conn_trans,
                                       const Real* conn_dz,
                                       const Real* cellvol, SatTables T,
                                       BOParams P, Real V, Real* rate) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int ncell = P.nx * P.ny * P.nz;
  if (cell >= ncell) return;
  const int bs = P.bs;
  const Real g = P.gravity ? P.grav_grad : 0.0;
  const Real p_i = u[bs * cell];
  const Real Sw_i = u[bs * cell + 1];
  const Real Sg_i = (bs == 3) ? u[bs * cell + 2] : 0.0;

  Real thru = 0.0;
  const int c0 = conn_ptr[cell], c1 = conn_ptr[cell + 1];
  for (int m = c0; m < c1; ++m) {
    const Real Tg = conn_trans[m];
    if (Tg <= 0) continue;
    const int nc = conn_nbr[m];
    const Real dz = conn_dz[m];
    const Real p_j = u[bs * nc];
    const Real Sw_j = u[bs * nc + 1];
    const Real Sg_j = (bs == 3) ? u[bs * nc + 2] : 0.0;
    const Real dp = p_j - p_i;
    const Real dPw = dp - g * P.rhow * dz;
    const Real dPo = dp - g * P.rhoo * dz;
    if (bs == 2) {
      Mob2 mi = relperm2_table(Sw_i, T, P.uw, P.uo, P.bw, P.bo,
                               P.swi, P.swr);
      Mob2 mj = relperm2_table(Sw_j, T, P.uw, P.uo, P.bw, P.bo,
                               P.swi, P.swr);
      thru += Tg * fabs(dPw > 0 ? mj.dMw : mi.dMw) * fabs(dPw);
      thru += Tg * fabs(dPo > 0 ? mj.dMo : mi.dMo) * fabs(dPo);
    } else {
      Mob3 mi = relperm3(Sw_i, Sg_i, T, P.uw, P.uo, P.ug, P.bw, P.bo, P.bg,
                         P.swi, P.swr, cell);
      Mob3 mj = relperm3(Sw_j, Sg_j, T, P.uw, P.uo, P.ug, P.bw, P.bo, P.bg,
                         P.swi, P.swr, nc);
      const Real dPg = dp - g * P.rhog * dz;
      const Mob3& dw = (dPw > 0) ? mj : mi;
      const Mob3& dn = (dPo > 0) ? mj : mi;
      const Mob3& dg = (dPg > 0) ? mj : mi;
      thru += Tg * fabs(dw.dMw_dSw) * fabs(dPw);
      thru += Tg * (fabs(dn.dMo_dSw) + fabs(dn.dMo_dSg)) * fabs(dPo);
      thru += Tg * fabs(dg.dMg_dSg) * fabs(dPg);
    }
  }
  const Real Vc = cellvol ? cellvol[cell] : V;
  const Real pv = Vc * poro[cell];
  rate[cell] = thru / (pv > (Real)0 ? pv : (Real)1e-30);   // [pore volumes/day]
}

// DRSDT: walk each cell's dissolved-gas ceiling after an accepted step.
//
//     Rs(t) <= Rs(t-dt) + DRSDT * dt        and    Rs(t) <= Rs_sat(p(t))
//
// so the ceiling is min(cap + drsdt*dt, Rs_sat(p)).  The accepted state's Rs,
// not the original RSVD value, is the reference for the next step.  This is
// important in an initially two-phase cell: RSVD describes undersaturated oil
// below the gas/oil contact, while oil coexisting with free gas starts at
// Rs_sat even when its extrapolated RSVD value is smaller.
//
// With DRSDT 0 this is simply "the cap never increases": gas that came out of
// solution when a cell depleted stays out.  Run once per accepted timestep,
// which is nowhere near the linear solve.
__global__ static void k_drsdt_update(const Real* u, int bs, BOParams P,
                                      Real dt, Real drsdt, int cells,
                                      Real* cap) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const DevicePvt& pv = pvt_of(P, c);
  if (!pv.valid) return;
  const Real p = u[(long long)bs * c];
  const Real rs_sat = dev_rs_sat(pv, p);
  Real lim;
  if (drsdt >= 0) {
    lim = cap[c] + drsdt * dt;             // explicit dissolution-rate limit
    if (rs_sat < lim) lim = rs_sat;
  } else {
    lim = fmin(cap[c], rs_sat);
  }
  if (lim < 0) lim = 0;
  cap[c] = lim;
}

// Establish OPM/ECLIPSE's initial `lastRs`: undersaturated oil takes its RSVD
// charge, but any cell containing free gas is on the saturated PVTO line.
__global__ static void k_drsdt_initialize(const Real* u, int bs, BOParams P,
                                          int cells, Real* cap) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const DevicePvt& pv = pvt_of(P, c);
  if (!pv.valid) return;
  const Real p = u[(long long)bs * c];
  const Real sg = (bs >= 3) ? u[(long long)bs * c + 2] : (Real)0;
  const Real anchor = P.rs_max_cell ? P.rs_max_cell[c] : P.rs_max;
  cap[c] = dev_rs_initial(pv, p, sg, anchor);
}

__global__ static void k_vappars_update(const Real* u, int bs, int cells,
                                        const Real* meaning, Real* so_max) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const Real sw = u[(long long)bs * c + 1];
  const Real kind = meaning ? meaning[c] : (Real)0;
  const Real sg = bs < 3 ? (Real)0
                  : kind > (Real)1.5 ? (Real)1 - sw
                  : kind > (Real)0.5 ? (Real)0
                                     : u[(long long)bs * c + 2];
  const Real so = fmax((Real)0, (Real)1 - sw - sg);
  if (so > so_max[c]) so_max[c] = so;
}

// flag = 1 (implicit) unless dt * rate < eta.  Wells are always implicit.
__global__ static void k_aim_flags(const Real* rate, const WellCell* well,
                                   Real dt, Real eta, int cells, Real* flag,
                                   int* nexp) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cells) return;
  WellCell w = well[cell];
  const bool ex = !(w.is_inj || w.is_prod) && (dt * rate[cell] < eta);
  flag[cell] = ex ? 0.0 : 1.0;
  if (ex) atomicAdd(nexp, 1);
}

__global__ static void k_aim_all_implicit(Real* flag, int cells) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < cells) flag[c] = 1.0;
}

// All cells implicit: the state the run starts in and returns to after a
// timestep chop, since the rates measured during a discarded attempt
// describe a state that was rolled back.
void aim_reset_implicit(Reservoir& r) {
  if (!r.aimflag) return;
  const int cells_ext = r.grid.n_local_ext();
  k_aim_all_implicit<<<grid1d(cells_ext), 256>>>(r.aimflag, cells_ext);
  NVRS_CHECK_CUDA(cudaGetLastError());
  r.aim_have_rate = 0;
  r.aim_expfrac = 0;
}

// Measure each cell's throughput rate [pore volumes/day] from a CONVERGED
// state.  This is the half of the CFL that depends on the pressure field,
// and it must come from a solved one: at the start of a substep the
// pressures are whatever the last solve left (on the very first substep,
// the uniform initial condition), where every potential difference is zero,
// the rate looks like zero everywhere, and the whole field would be flagged
// explicit -- including the cells about to be filled by the injector.
void aim_measure_rates(Reservoir& r) {
  if (!r.aim || !r.aim_rate) return;
  if (r.model == Model::Compositional) return;
  BOParams P;
  P.bs = r.bs; P.nx = r.grid.nxl; P.ny = r.grid.nyl; P.nz = r.grid.nzl;
  P.dz_cell = r.grid.dz; P.dt = r.dt; P.ct = r.ct;
  P.have_rock = r.have_rock; P.rock_pref = r.rock_pref;
  P.cnv_tol = r.cnv_tol;
  P.cpr_decouple = r.cpr_decouple;
  P.drs_eps = r.drs_eps;
  P.uw = r.uw; P.uo = r.uo; P.ug = r.ug;
  P.bw = r.bw; P.bo = r.bo; P.bg = r.bg; P.rs = r.rs;
  P.pvt = r.pvt;
  P.pvt_regions = r.pvt_regions; P.pvt_cell = r.pvt_cell;
  // With no explicit undersaturated anchor, the deck's constant RS is it.
  P.rs_max = (r.rs_max > 0) ? r.rs_max : r.rs;
  P.rs_max_cell = r.rs_max_cell;
  P.rs_cap = r.rs_cap;
  P.so_max = r.so_max; P.vap1 = r.vap1;
  P.swi = r.swi; P.swr = r.swr;
  P.rhow = r.rhow; P.rhoo = r.rhoo; P.rhog = r.rhog; P.gravity = r.gravity;
  P.grav_grad = r.grav_grad; P.fd_eps = r.fd_eps;
  Real V = (double)r.grid.dx * r.grid.dy * r.grid.dz / units::kFt3PerBbl;

  // Same choice of connectivity as the assembly.  It has to be the same: the
  // flags this produces decide which cells the residual treats explicitly, and
  // the Jacobian applies the identical rule.  Measuring the throughput from a
  // stencil the grid does not have would flag cells from garbage and leave the
  // residual and Jacobian describing different functions.
  if (r.halo_conn)
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.u, r.bs);
  else
    halo_exchange(r.u, r.bs, r.grid);

  if (r.use_conn_list && r.conn_ptr) {
    k_aim_rate_conn<<<grid1d(r.grid.n_local()), 256>>>(
        r.u, r.poro, r.conn_ptr, r.conn_nbr, r.conn_trans, r.conn_dz,
        r.cellvol, r.tables, P, V, r.aim_rate);
  } else {
    k_aim_rate<<<grid1d(r.grid.n_local()), 256>>>(r.u, r.poro, r.tface, r.nbr,
                                                  r.cellk, r.tables, P, V,
                                                  r.aim_rate);
  }
  NVRS_CHECK_CUDA(cudaGetLastError());
  r.aim_have_rate = 1;
}

// Turn the measured rates into flags for the coming substep at its dt.
// Until a converged state has been measured the run stays fully implicit,
// which is what makes the first substep of every report step a plain FIM
// step rather than a guess.
void aim_update_flags(Reservoir& r, int verbose) {
  if (!r.aim || !r.aimflag) return;
  if (r.model == Model::Compositional) return;
  if (!r.aim_have_rate) return;          // no converged measurement yet
  const int cells = r.grid.n_local();
  int* d_n = nullptr;
  NVRS_CHECK_CUDA(cudaMalloc(&d_n, sizeof(int)));
  NVRS_CHECK_CUDA(cudaMemset(d_n, 0, sizeof(int)));
  auto* well = static_cast<WellCell*>(r.wellcells);
  k_aim_flags<<<grid1d(cells), 256>>>(r.aim_rate, well, r.dt,
                                      (Real)r.aim_cfl, cells, r.aimflag, d_n);
  NVRS_CHECK_CUDA(cudaGetLastError());
  // Donor decisions at rank seams read the neighbour's flag.
  halo_exchange(r.aimflag, 1, r.grid);
  int nexp = 0;
  NVRS_CHECK_CUDA(cudaMemcpy(&nexp, d_n, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_n);
  long tot[2] = {(long)nexp, (long)cells};
  if (r.mpi_nranks > 1)
    MPI_Allreduce(MPI_IN_PLACE, tot, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  r.aim_expfrac = tot[1] > 0 ? (Real)tot[0] / (Real)tot[1] : (Real)0;
  if (verbose && r.grid.rank == 0)
    std::printf("  [aim] %.1f%% of cells explicit (throughput CFL < %.2g at "
                "dt = %g days)\n",
                100.0 * (double)r.aim_expfrac, (double)r.aim_cfl,
                (double)r.dt);
}

// ---------------------------------------------------------------------------
// What the pressure operator actually looks like
// ---------------------------------------------------------------------------
void report_pressure_operator(const DeviceLinearSystem& sys, const char* when) {
  static int budget = -1;
  if (budget < 0) {
    const char* e = std::getenv("NVRS_CPR_DIAG");
    budget = e ? std::atoi(e) : 1;
  }
  if (budget <= 0) return;
  --budget;

  const int n = sys.cells;
  if (n <= 0 || !sys.prowptr || !sys.pval || !sys.pcolind) return;

  std::vector<int> rp(n + 1);
  NVRS_CHECK_CUDA(cudaMemcpy(rp.data(), sys.prowptr, sizeof(int) * (n + 1),
                             cudaMemcpyDeviceToHost));
  const int nnz = rp[n];
  if (nnz <= 0) return;
  std::vector<PrecReal> av(nnz);
  std::vector<int> ci(nnz);
  NVRS_CHECK_CUDA(cudaMemcpy(av.data(), sys.pval, sizeof(PrecReal) * nnz,
                             cudaMemcpyDeviceToHost));
  NVRS_CHECK_CUDA(cudaMemcpy(ci.data(), sys.pcolind, sizeof(int) * nnz,
                             cudaMemcpyDeviceToHost));

  long long bad_diag = 0, bad_dom = 0, pos_off = 0, empty = 0;
  double worst_ratio = 1e300, sum_ratio = 0;
  double dmin = 1e300, dmax = -1e300;
  int worst_row = -1;
  long long counted = 0;

  for (int i = 0; i < n; ++i) {
    const long long gi = sys.row0 + i;
    double diag = 0, offsum = 0, offpos = 0;
    bool found = false;
    for (int k = rp[i]; k < rp[i + 1]; ++k) {
      const double v = (double)av[k];
      if ((long long)ci[k] == gi) { diag += v; found = true; }
      else { offsum += std::fabs(v); if (v > 0) offpos += v; }
    }
    if (!found || rp[i + 1] == rp[i]) { ++empty; continue; }
    if (!(diag > 0)) ++bad_diag;
    if (offpos > 0) ++pos_off;
    dmin = std::min(dmin, diag);
    dmax = std::max(dmax, diag);
    // Weak diagonal dominance: |diag| >= sum |offdiag|.  Equality is the
    // singular Neumann case; below it the row is not dominant at all.
    const double ratio = (offsum > 0) ? std::fabs(diag) / offsum : 1e300;
    if (ratio < 1.0 - 1e-12) ++bad_dom;
    if (ratio < worst_ratio) { worst_ratio = ratio; worst_row = i; }
    if (ratio < 1e299) { sum_ratio += ratio; ++counted; }
  }

  std::printf("\n  [cpr] pressure operator at %s\n", when);
  std::printf("        rows %d, nonzeros %d\n", n, nnz);
  std::printf("        diagonal            : %.4g .. %.4g\n", dmin, dmax);
  std::printf("        non-positive diag   : %lld  (AMG needs > 0)\n", bad_diag);
  std::printf("        positive off-diag   : %lld rows  (M-matrix wants none)\n",
              pos_off);
  std::printf("        not diag. dominant  : %lld rows of %d\n", bad_dom, n);
  std::printf("        |diag|/sum|offdiag| : worst %.6f at row %d, mean %.4f\n",
              worst_ratio, worst_row, counted ? sum_ratio / (double)counted : 0.0);
  if (empty) std::printf("        rows with no diagonal entry: %lld\n", empty);
  // 1.000000 exactly is the tell for an incompressible system: the diagonal is
  // the sum of the off-diagonals, the constant vector is in the null space, and
  // no amount of Krylov iteration will converge.
  if (worst_ratio > 0.999999 && worst_ratio < 1.000001)
    std::printf("        ^ ratio is 1.000000: the operator is a singular "
                "Neumann Laplacian.\n");
  std::fflush(stdout);
}


// ---------------------------------------------------------------------------
__global__ static void k_pvswitch_initialize(Real* u, int bs, BOParams P,
                                              SatTables T, int cells,
                                              Real* meaning) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const DevicePvt& pv = pvt_of(P, c);
  if (!pv.valid || bs != 3) { meaning[c] = 0; return; }
  const Real p = u[(long long)bs * c];
  const Real sg = u[(long long)bs * c + 2];
  const Real so = (Real)1 - u[(long long)bs * c + 1] - sg;
  if (sg <= (Real)1e-7 && so > (Real)1e-7) {
    const Real anchor = P.rs_max_cell ? P.rs_max_cell[c] : P.rs_max;
    u[(long long)bs * c + 2] = dev_rs_state(pv, p, anchor);
    meaning[c] = 1;
  } else if (so <= (Real)1e-7 && sg > (Real)1e-7) {
    const Real somax = P.so_max ? P.so_max[c] : (Real)0;
    const Cap cap = capillary(u[(long long)bs * c + 1], sg, T, c);
    const Real pg = p + cap.pcog;
    const PvtCell f = dev_pvt_cell(pv, pg, (Real)1e30, (Real)0, somax,
                                   P.vap1);
    u[(long long)bs * c] = pg;  // Po -> Pg
    u[(long long)bs * c + 2] = f.rv;
    meaning[c] = 2;
  } else {
    meaning[c] = 0;
  }
}

__global__ static void k_pvswitch_adapt(Real* u, int bs, BOParams P,
                                         SatTables T, int cells,
                                         Real* meaning) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells || bs != 3) return;
  const DevicePvt& pv = pvt_of(P, c);
  if (!pv.valid) return;
  const long long q = (long long)bs * c;
  const Real p = u[q];
  const Real sw = u[q + 1];
  Real x = u[q + 2];
  // OPM's PriVarOscilationThreshold default. Applying it in both directions
  // is conservative and prevents a cell from swapping meaning on successive
  // Newton/line-search trials at the saturated boundary.
  constexpr Real eps = (Real)1e-5;
  // Match OPM's water-only special case.  With neither hydrocarbon phase
  // present, Rs or Rv is not an admissible composition primary and its local
  // equation block becomes singular.  Return to Sg=0; convert Pg back to Po
  // first when leaving the Rv pressure convention.
  if (sw >= (Real)1) {
    if (meaning[c] > (Real)1.5) {
      const Cap cap = capillary((Real)1, (Real)0, T, c);
      u[q] = p - cap.pcog;
    }
    u[q + 1] = (Real)1;
    u[q + 2] = (Real)0;
    meaning[c] = (Real)0;
    return;
  }
  if (meaning[c] < (Real)0.5) {
    // OPM: a negative gas saturation with oil present means that the gas phase
    // disappeared.  Replace the same storage slot by the saturated (or
    // DRSDT-limited) dissolved-gas ratio; component inventory remains in the
    // nonlinear unknown rather than an accepted-step side update.
    if (x < -eps && (Real)1 - sw > 0) {
      Real lim = dev_rs_sat(pv, p);
      if (P.rs_cap && P.rs_cap[c] < lim) lim = P.rs_cap[c];
      u[q + 2] = fmax((Real)0, lim);
      meaning[c] = 1;
    } else if ((Real)1 - sw - x < -eps && x > 0) {
      const Real sg = x;
      const Real somax = P.so_max ? P.so_max[c] : (Real)0;
      const Cap cap = capillary(sw, sg, T, c);
      const Real pg = p + cap.pcog;
      const PvtCell f = dev_pvt_cell(pv, pg, (Real)1e30, (Real)0, somax,
                                     P.vap1);
      u[q] = pg;                            // Po -> Pg
      u[q + 2] = f.rv;
      meaning[c] = 2;
    } else if (x < 0) {
      u[q + 2] = 0;
    }
  } else if (meaning[c] < (Real)1.5) {
    Real lim = dev_rs_sat(pv, p);
    if (P.rs_cap && P.rs_cap[c] < lim) lim = P.rs_cap[c];
    if (x > lim * ((Real)1 + eps)) {
      // Dissolved gas exceeds what saturated oil can hold: create the free-gas
      // phase at zero saturation and let the next Newton iteration distribute
      // the excess through the gas component equation.
      u[q + 2] = 0;
      meaning[c] = 0;
    }
  } else {
    const Real sg = fmax((Real)0, (Real)1 - sw);
    const Real somax = P.so_max ? P.so_max[c] : (Real)0;
    const PvtCell sat = dev_pvt_cell(pv, p, (Real)1e30, (Real)0, somax,
                                     P.vap1);
    if (x > sat.rv * ((Real)1 + eps)) {
      const Cap cap = capillary(sw, sg, T, c);
      u[q] = p - cap.pcog;                 // Pg -> Po
      u[q + 2] = sg;
      meaning[c] = 0;
    }
  }
}

void blackoil_pvswitch_initialize(Reservoir& r) {
  if (!r.pv_switch || r.bs != 3 || !r.pvt.valid || r.gas_primary) return;
  const int ext = r.grid.n_local_ext();
  NVRS_CHECK_CUDA(cudaMalloc(&r.gas_primary, sizeof(Real) * (size_t)ext));
  NVRS_CHECK_CUDA(cudaMalloc(&r.gas_primary_old, sizeof(Real) * (size_t)ext));
  BOParams P = make_bo_params(r);
  const int cells = r.grid.n_local();
  k_pvswitch_initialize<<<grid1d(cells), 256>>>(r.u, r.bs, P, r.tables, cells,
                                                 r.gas_primary);
  NVRS_CHECK_CUDA(cudaGetLastError());
  if (r.halo_conn) {
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.u, r.bs);
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.gas_primary, 1);
  } else {
    halo_exchange(r.u, r.bs, r.grid);
    halo_exchange(r.gas_primary, 1, r.grid);
  }
  NVRS_CHECK_CUDA(cudaMemcpy(r.u_old, r.u,
      sizeof(Real) * (size_t)r.bs * ext, cudaMemcpyDeviceToDevice));
  NVRS_CHECK_CUDA(cudaMemcpy(r.gas_primary_old, r.gas_primary,
      sizeof(Real) * (size_t)ext, cudaMemcpyDeviceToDevice));
}

void blackoil_pvswitch_adapt(Reservoir& r) {
  if (!r.gas_primary) return;
  BOParams P = make_bo_params(r);
  const int cells = r.grid.n_local();
  k_pvswitch_adapt<<<grid1d(cells), 256>>>(r.u, r.bs, P, r.tables, cells,
                                           r.gas_primary);
  NVRS_CHECK_CUDA(cudaGetLastError());
  if (r.halo_conn)
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.gas_primary, 1);
  else
    halo_exchange(r.gas_primary, 1, r.grid);
}

void drsdt_initialize(Reservoir& r) {
  if (!r.rs_cap) return;
  BOParams P{};
  P.bs = r.bs;
  P.pvt = r.pvt;
  P.pvt_regions = r.pvt_regions; P.pvt_cell = r.pvt_cell;
  P.rs_max = (r.rs_max > 0) ? r.rs_max : r.rs;
  P.rs_max_cell = r.rs_max_cell;
  const int cells = r.grid.n_local();
  k_drsdt_initialize<<<grid1d(cells), 256>>>(r.u, r.bs, P, cells, r.rs_cap);
  NVRS_CHECK_CUDA(cudaGetLastError());
  if (r.halo_conn)
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.rs_cap, 1);
  else
    halo_exchange(r.rs_cap, 1, r.grid);
}

void drsdt_update(Reservoir& r, Real dt) {
  if (!r.rs_cap || dt <= 0) return;
  BOParams P{};
  P.bs = r.bs;
  P.pvt = r.pvt;
  P.pvt_regions = r.pvt_regions; P.pvt_cell = r.pvt_cell;
  P.rs_max = (r.rs_max > 0) ? r.rs_max : r.rs;
  const int cells = r.grid.n_local();
  k_drsdt_update<<<grid1d(cells), 256>>>(r.u, r.bs, P, dt, r.drsdt,
                                         cells, r.rs_cap);
  NVRS_CHECK_CUDA(cudaGetLastError());
  if (r.halo_conn)
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.rs_cap, 1);
  else
    halo_exchange(r.rs_cap, 1, r.grid);
}

void vappars_update(Reservoir& r) {
  if (!r.so_max || !(r.vap1 > 0)) return;
  const int cells = r.grid.n_local();
  k_vappars_update<<<grid1d(cells), 256>>>(r.u, r.bs, cells,
                                           r.gas_primary, r.so_max);
  NVRS_CHECK_CUDA(cudaGetLastError());
  if (r.halo_conn)
    static_cast<HaloConn*>(r.halo_conn)->exchange(r.so_max, 1);
  else
    halo_exchange(r.so_max, 1, r.grid);
}

// EHYSTR: walk both OPM Carlson histories after an accepted step: maximum Sg
// for gas/oil and minimum Sw+Sg (= 1-So) for oil/water.
__global__ static void k_hyst_update(const Real* u, int bs, int cells,
                                     const Real* meaning, Real* sg_hy,
                                     Real* sw_ow_hy) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const Real sw = u[(long long)bs * c + 1];
  const Real kind = meaning ? meaning[c] : (Real)0;
  const Real sg = kind > (Real)1.5 ? (Real)1 - sw
                  : kind > (Real)0.5 ? (Real)0
                                     : u[(long long)bs * c + 2];
  if (sg > sg_hy[c]) sg_hy[c] = sg;
  const Real sw_ow = sw + sg;
  if (sw_ow < sw_ow_hy[c]) sw_ow_hy[c] = sw_ow;
}

void hyst_update(Reservoir& r) {
  if (!r.sg_hy || !r.sw_ow_hy || !r.use_hysteresis || r.bs < 3) return;
  const int cells = r.grid.n_local();
  k_hyst_update<<<grid1d(cells), 256>>>(r.u, r.bs, cells, r.gas_primary,
                                        r.sg_hy, r.sw_ow_hy);
  NVRS_CHECK_CUDA(cudaGetLastError());
  halo_exchange(r.sg_hy, 1, r.grid);
  halo_exchange(r.sw_ow_hy, 1, r.grid);
}

}  // namespace nvrs
