// SPDX-License-Identifier: Apache-2.0
//
// tc_assemble.inc -- JACOBIAN TENSORCORE.  Included by assemble_blackoil.cu
// when built with TENSORCORE=1, and absent from the default build.
//
// The analytic kernel runs first, as it always does, and supplies the residual,
// the accumulation and well parts of the Jacobian, the norms and the pressure
// weights.  This file then replaces what comes from the connections.  For each
// oriented connection (i, j) the residual contribution is a function of eleven
// features z, the phase mobilities, potentials and upstream PVT factors,
//
//     R_o = -(b_o F_o + R_v b_g F_g)
//     R_w = -b_w F_w                     F_a = T M_a dPhi_a
//     R_g = -(b_g F_g + R_s b_o F_o)
//
// so that dR/dx = (dR/dz)(dz/dx) = L_flux L_props, with x the primaries of both
// cells.  L_flux is bs x 11 and L_props is 11 x 2bs, padded to 16x16.  Column v
// of the product is the coupling to primary v of cell i, column bs+v to primary
// v of cell j.  In a two-phase run R_g, M_g, dPhi_g, b_g, R_s and R_v are absent
// and the PVT lanes of L_props are zero, since b_o and b_w are constants there.
//
// The four fill kernels below mirror the connection loops of the four analytic
// kernels, derivative for derivative.  TCVALIDATE 1 measures the result against
// those kernels on every assembly, and tests/test_tc_jacobian.cu checks all four
// on random states.
//
// The self block is corrected rather than rebuilt:
//
//     J_ii = J_ii^analytic + sum_conn (J_conn,i^product - J_conn,i^FP64)
//
// where J_conn,i^FP64 is the same factorised product in FP64.  Accumulation and
// wells therefore keep FP64 and only the connection part carries the rounding
// of the product.

namespace {

enum TcFeature : int {
  TZ_MW, TZ_MO, TZ_MG, TZ_DPW, TZ_DPO, TZ_DPG,
  TZ_BW, TZ_BO, TZ_BG, TZ_RS, TZ_RV, TZ_COUNT
};
static_assert(TZ_COUNT <= tc::kTile, "features must fit the tile");

struct TcOps {
  Real f[3][TZ_COUNT];   // L_flux,  rows oil, water, gas
  Real p[TZ_COUNT][6];   // L_props, columns primaries of i then of j
};

// Everything the kernels read, for both grid descriptions.
struct TcView {
  const Real* u; const Real* u_old; const Real* aimflag;
  bool conn;
  const int* conn_ptr; const int* conn_owner; const int* conn_nbr;
  const Real* conn_trans; const Real* conn_dz; const Real* conn_threshold;
  const int* conn_bcol; const int* conn_pcol;
  const int* self_bcol; const int* self_pcol;
  const Real* tface; const int* nbr; const int* cellk;
  const int* bcol; const int* pcolpos;
  const int* rowptr; const int* prowptr;
  const AssemblyCellProps* props;
  SatTables T; BOParams P;
  float* lflux; float* lprops; const float* prod;
  Real* dself; Real* aval; PrecReal* pval; PrecReal* dinv;
  const PrecReal* pweight;
  Real* stats; unsigned long long* worst;
};

// L_flux for the three component equations.  Row 2 is not emitted when bs = 2.
__device__ __forceinline__ void tc_flux(TcOps& L, Real T, Real Mw, Real Mo,
                                        Real Mg, Real dPw, Real dPo, Real dPg,
                                        Real bw, Real bo, Real bg, Real rs,
                                        Real rv) {
  const Real Fw = T * Mw * dPw, Fo = T * Mo * dPo, Fg = T * Mg * dPg;
  L.f[0][TZ_MO] = -bo * T * dPo;      L.f[0][TZ_DPO] = -bo * T * Mo;
  L.f[0][TZ_MG] = -rv * bg * T * dPg; L.f[0][TZ_DPG] = -rv * bg * T * Mg;
  L.f[0][TZ_BO] = -Fo;  L.f[0][TZ_BG] = -rv * Fg;  L.f[0][TZ_RV] = -bg * Fg;
  L.f[1][TZ_MW] = -bw * T * dPw;      L.f[1][TZ_DPW] = -bw * T * Mw;
  L.f[1][TZ_BW] = -Fw;
  L.f[2][TZ_MG] = -bg * T * dPg;      L.f[2][TZ_DPG] = -bg * T * Mg;
  L.f[2][TZ_MO] = -rs * bo * T * dPo; L.f[2][TZ_DPO] = -rs * bo * T * Mo;
  L.f[2][TZ_BG] = -Fg;  L.f[2][TZ_BO] = -rs * Fo;  L.f[2][TZ_RS] = -bo * Fo;
}

// Write both tiles in FP32 and subtract the FP64 self contribution.  The tiles
// were cleared by the host, so only physical lanes are stored.
__device__ __forceinline__ void tc_emit(const TcOps& L, int bs, int q, int cell,
                                        const TcView& V) {
  const size_t off = (size_t)q * tc::kTile * tc::kTile;
  for (int a = 0; a < bs; ++a)
    for (int z = 0; z < TZ_COUNT; ++z)
      V.lflux[off + a * tc::kTile + z] = (float)L.f[a][z];
  for (int z = 0; z < TZ_COUNT; ++z)
    for (int c = 0; c < 2 * bs; ++c)
      V.lprops[off + z * tc::kTile + c] = (float)L.p[z][c];
  for (int a = 0; a < bs; ++a)
    for (int v = 0; v < bs; ++v) {
      Real s = 0;
      for (int z = 0; z < TZ_COUNT; ++z) s += L.f[a][z] * L.p[z][v];
      atomicAdd(V.dself + (long long)cell * bs * bs + a * bs + v, -s);
    }
}

// --- two-phase, connection list (k_assemble_bo2_analytic_conn) ---------------
__global__ void k_tc_fill_bo2_conn(TcView V, long long first, int count) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const long long m = first + q;
  const Real trans = V.conn_trans[m];
  if (!(trans > 0)) return;
  const BOParams& P = V.P;
  const int cell = V.conn_owner[m], nc = V.conn_nbr[m];
  const Real bo = (Real)1 / P.bo, bw = (Real)1 / P.bw;
  const Real p_i = V.u[2 * cell], sw_i = V.u[2 * cell + 1];
  const Real sw_old = V.u_old[2 * cell + 1];
  const Real p_j = V.u[2 * nc], sw_j = V.u[2 * nc + 1];
  const bool ex_i = V.aimflag[cell] < (Real)0.5;
  const bool ex_j = V.aimflag[nc] < (Real)0.5;
  const Real grav = P.gravity ? P.grav_grad : (Real)0;
  const Real dz = V.conn_dz[m];
  const Real th = V.conn_threshold ? V.conn_threshold[m] : (Real)0;
  const Real dpo0 = canonical_potential(cell, nc, p_i, p_j, grav * P.rhoo, dz);
  const Real dpw0 = canonical_potential(cell, nc, p_i, p_j, grav * P.rhow, dz);
  const Real dpo = threshold_potential(dpo0, th);
  const Real dpw = threshold_potential(dpw0, th);
  const bool flow_o = !(th > 0 && fabs(dpo0) <= th);
  const bool flow_w = !(th > 0 && fabs(dpw0) <= th);
  const bool upo_j = dpo > 0, upw_j = dpw > 0;
  const bool ex_o = upo_j ? ex_j : ex_i;
  const bool ex_w = upw_j ? ex_j : ex_i;
  const Real sw_o = upo_j ? (ex_o ? V.u_old[2 * nc + 1] : sw_j)
                          : (ex_o ? sw_old : sw_i);
  const Real sw_w = upw_j ? (ex_w ? V.u_old[2 * nc + 1] : sw_j)
                          : (ex_w ? sw_old : sw_i);
  Mob2 mo = relperm2_table(sw_o, V.T, P.uw, P.uo, 1, 1, P.swi, P.swr);
  Mob2 mw = relperm2_table(sw_w, V.T, P.uw, P.uo, 1, 1, P.swi, P.swr);
  if (!flow_o) mo.Mo = mo.dMo = 0;
  if (!flow_w) mw.Mw = mw.dMw = 0;

  TcOps L = {};
  tc_flux(L, trans, mw.Mw, mo.Mo, 0, dpw, dpo, 0, bw, bo, 0, 0, 0);
  L.p[TZ_DPO][0] = -1;  L.p[TZ_DPO][2] = 1;
  L.p[TZ_DPW][0] = -1;  L.p[TZ_DPW][2] = 1;
  if (!ex_o) L.p[TZ_MO][upo_j ? 3 : 1] = mo.dMo;
  if (!ex_w) L.p[TZ_MW][upw_j ? 3 : 1] = mw.dMw;
  tc_emit(L, 2, q, cell, V);
}

// --- two-phase, fixed stencil (k_assemble_bo_analytic) -----------------------
__global__ void k_tc_fill_bo2_cart(TcView V, long long first, int count) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const long long m = first + q;
  const int cell = (int)(m / 6), f = (int)(m % 6);
  const int nc = V.nbr[7 * cell + 1 + f];
  if (nc < 0) return;
  const Real Tg = V.tface[6 * cell + f];
  if (Tg <= 0) return;
  const BOParams& P = V.P;
  const Real bo = 1.0 / P.bo, bw = 1.0 / P.bw;
  const Real p_i = V.u[2 * cell], Sw_i = V.u[2 * cell + 1];
  const Real p_j = V.u[2 * nc], Sw_j = V.u[2 * nc + 1];
  const Real g = P.gravity ? P.grav_grad : (Real)0;
  const Real zc = (V.cellk[cell] + 0.5) * P.dz_cell;
  const Real zn = (V.cellk[nc] + 0.5) * P.dz_cell;
  const Real dPo = (p_j - g * P.rhoo * zn) - (p_i - g * P.rhoo * zc);
  const Real dPw = (p_j - g * P.rhow * zn) - (p_i - g * P.rhow * zc);
  const bool upo_j = dPo > 0.0, upw_j = dPw > 0.0;
  const bool ex_i = V.aimflag[cell] < 0.5, ex_j = V.aimflag[nc] < 0.5;
  const bool ex_o = upo_j ? ex_j : ex_i;
  const bool ex_w = upw_j ? ex_j : ex_i;
  const Real old_i = V.u_old[2 * cell + 1], old_j = V.u_old[2 * nc + 1];
  const Real Sw_uo = upo_j ? (ex_o ? old_j : Sw_j) : (ex_o ? old_i : Sw_i);
  const Real Sw_uw = upw_j ? (ex_w ? old_j : Sw_j) : (ex_w ? old_i : Sw_i);
  const Mob2 mo = relperm2_table(Sw_uo, V.T, P.uw, P.uo, 1.0, 1.0, P.swi, P.swr);
  const Mob2 mw = relperm2_table(Sw_uw, V.T, P.uw, P.uo, 1.0, 1.0, P.swi, P.swr);

  TcOps L = {};
  tc_flux(L, Tg, mw.Mw, mo.Mo, 0, dPw, dPo, 0, bw, bo, 0, 0, 0);
  L.p[TZ_DPO][0] = -1;  L.p[TZ_DPO][2] = 1;
  L.p[TZ_DPW][0] = -1;  L.p[TZ_DPW][2] = 1;
  if (!ex_o) L.p[TZ_MO][upo_j ? 3 : 1] = mo.dMo;
  if (!ex_w) L.p[TZ_MW][upw_j ? 3 : 1] = mw.dMw;
  tc_emit(L, 2, q, cell, V);
}

// --- three-phase, fixed stencil (k_assemble_bo3_analytic) --------------------
__global__ void k_tc_fill_bo3_cart(TcView V, long long first, int count) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const long long m = first + q;
  const int cell = (int)(m / 6), f = (int)(m % 6);
  const int nc = V.nbr[7 * cell + 1 + f];
  if (nc < 0) return;
  const Real Tg = V.tface[6 * cell + f];
  if (Tg <= 0) return;
  const BOParams& P = V.P;
  const Real bo = 1.0 / P.bo, bw = 1.0 / P.bw, bg = 1.0 / P.bg, rs = P.rs;
  const Real p_i = V.u[3 * cell], Sw_i = V.u[3 * cell + 1], Sg_i = V.u[3 * cell + 2];
  const Real p_j = V.u[3 * nc], Sw_j = V.u[3 * nc + 1], Sg_j = V.u[3 * nc + 2];
  const Real g = P.gravity ? P.grav_grad : (Real)0;
  const Real zc = (V.cellk[cell] + 0.5) * P.dz_cell;
  const Real zn = (V.cellk[nc] + 0.5) * P.dz_cell;
  const Real dPw = (p_j - g * P.rhow * zn) - (p_i - g * P.rhow * zc);
  const Real dPo = (p_j - g * P.rhoo * zn) - (p_i - g * P.rhoo * zc);
  const Real dPg = (p_j - g * P.rhog * zn) - (p_i - g * P.rhog * zc);
  const bool wj = dPw > 0.0, oj = dPo > 0.0, gj = dPg > 0.0;
  const bool ex_i = V.aimflag[cell] < 0.5, ex_j = V.aimflag[nc] < 0.5;
  const bool ex_w = wj ? ex_j : ex_i;
  const bool ex_o = oj ? ex_j : ex_i;
  const bool ex_g = gj ? ex_j : ex_i;
  const Real w0 = V.u_old[3 * cell + 1], g0 = V.u_old[3 * cell + 2];
  const Real w1 = V.u_old[3 * nc + 1], g1 = V.u_old[3 * nc + 2];
  const Real SwU = wj ? (ex_w ? w1 : Sw_j) : (ex_w ? w0 : Sw_i);
  const Real SgU = gj ? (ex_g ? g1 : Sg_j) : (ex_g ? g0 : Sg_i);
  const Real SwO = oj ? (ex_o ? w1 : Sw_j) : (ex_o ? w0 : Sw_i);
  const Real SgO = oj ? (ex_o ? g1 : Sg_j) : (ex_o ? g0 : Sg_i);
  const Mob3 A = relperm3(SwU, SgU, V.T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr);
  const Mob3 B = relperm3(SwO, SgO, V.T, P.uw, P.uo, P.ug, 1.0, 1.0, 1.0, P.swi, P.swr);

  TcOps L = {};
  tc_flux(L, Tg, A.Mw, B.Mo, A.Mg, dPw, dPo, dPg, bw, bo, bg, rs, 0);
  L.p[TZ_DPW][0] = -1;  L.p[TZ_DPW][3] = 1;
  L.p[TZ_DPO][0] = -1;  L.p[TZ_DPO][3] = 1;
  L.p[TZ_DPG][0] = -1;  L.p[TZ_DPG][3] = 1;
  if (!ex_o) {
    L.p[TZ_MO][oj ? 4 : 1] = B.dMo_dSw;
    L.p[TZ_MO][oj ? 5 : 2] = B.dMo_dSg;
  }
  if (!ex_w) L.p[TZ_MW][wj ? 4 : 1] = A.dMw_dSw;
  if (!ex_g) L.p[TZ_MG][gj ? 5 : 2] = A.dMg_dSg;
  tc_emit(L, 3, q, cell, V);
}

// --- three-phase, connection list (k_assemble_bo3_analytic_conn) -------------
__global__ void k_tc_fill_bo3_conn(TcView V, long long first, int count) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const long long m = first + q;
  const Real Tg = V.conn_trans[m];
  if (Tg <= 0) return;
  const BOParams& P = V.P;
  const SatTables& T = V.T;
  const int cell = V.conn_owner[m], nc = V.conn_nbr[m];
  const Real* u = V.u;
  const Real* u_old = V.u_old;
  const AssemblyCellProps* cell_props = V.props;
  const Real g = P.gravity ? P.grav_grad : (Real)0;
  const Real dz = V.conn_dz[m];

  const Real p_i = u[3 * cell], Sw_i = u[3 * cell + 1];
  const Real meaning_i = P.gas_primary ? P.gas_primary[cell] : (Real)0;
  const Real meaning_old_i = P.gas_primary_old ? P.gas_primary_old[cell] : (Real)0;
  const bool rs_i = meaning_i > (Real)0.5 && meaning_i < (Real)1.5;
  const bool rv_i = meaning_i > (Real)1.5;
  const bool rs_old_i = meaning_old_i > (Real)0.5 && meaning_old_i < (Real)1.5;
  const bool rv_old_i = meaning_old_i > (Real)1.5;
  const Real Sg_i = rs_i ? (Real)0 : (rv_i ? (Real)1 - Sw_i : u[3 * cell + 2]);
  const Real usn1 = u_old[3 * cell + 1];
  const Real usn2 = rs_old_i ? (Real)0
                    : (rv_old_i ? (Real)1 - usn1 : u_old[3 * cell + 2]);
  const bool ex_i = V.aimflag[cell] < 0.5;

  const Real meaning_j = P.gas_primary ? P.gas_primary[nc] : (Real)0;
  const bool rs_j = meaning_j > (Real)0.5 && meaning_j < (Real)1.5;
  const bool rv_j = meaning_j > (Real)1.5;
  const Real p_j = u[3 * nc], Sw_j = u[3 * nc + 1];
  const Real Sg_j = rs_j ? (Real)0 : (rv_j ? (Real)1 - Sw_j : u[3 * nc + 2]);

  const PvtCell& F = cell_props[cell].pvt;
  const PvtCell& N = cell_props[nc].pvt;
  const Cap& Ci = cell_props[cell].cap;
  const Cap& Cj = cell_props[nc].cap;

  constexpr Real ft3_per_bbl = (Real)5.614583333333333;
  const Real rho_w = (Real)0.5 * (P.rhow * F.bw + P.rhow * N.bw);
  const Real oil_mass_i = P.rhoo + F.rs * P.rhog / ft3_per_bbl;
  const Real oil_mass_j = P.rhoo + N.rs * P.rhog / ft3_per_bbl;
  const Real gas_mass_i = P.rhog / ft3_per_bbl + F.rv * P.rhoo;
  const Real gas_mass_j = P.rhog / ft3_per_bbl + N.rv * P.rhoo;
  const Real rho_o = (Real)0.5 * (oil_mass_i * F.bo + oil_mass_j * N.bo);
  const Real rho_g = (Real)0.5 * (gas_mass_i * F.bg + gas_mass_j * N.bg);
  const Real drho_w_i = P.rhow * F.dbw, drho_w_j = P.rhow * N.dbw;
  const Real drho_o_i = F.drs * P.rhog / ft3_per_bbl * F.bo + oil_mass_i * F.dbo;
  const Real drho_o_j = N.drs * P.rhog / ft3_per_bbl * N.bo + oil_mass_j * N.dbo;
  const Real drho_g_i = F.drv * P.rhoo * F.bg + gas_mass_i * F.dbg;
  const Real drho_g_j = N.drv * P.rhoo * N.bg + gas_mass_j * N.dbg;
  const Real th = V.conn_threshold ? V.conn_threshold[m] : (Real)0;
  const Real po_i = p_i - (rv_i ? Ci.pcog : (Real)0);
  const Real po_j = p_j - (rv_j ? Cj.pcog : (Real)0);
  const Real dPw0 = canonical_potential(cell, nc, po_i - Ci.pcow, po_j - Cj.pcow,
                                        g * rho_w, dz);
  const Real dPo0 = canonical_potential(cell, nc, po_i, po_j, g * rho_o, dz);
  const Real dPg0 = canonical_potential(cell, nc, po_i + Ci.pcog, po_j + Cj.pcog,
                                        g * rho_g, dz);
  const Real dPw = threshold_potential(dPw0, th);
  const Real dPo = threshold_potential(dPo0, th);
  const Real dPg = threshold_potential(dPg0, th);
  const bool flow_w = !(th > 0 && fabs(dPw0) <= th);
  const bool flow_o = !(th > 0 && fabs(dPo0) <= th);
  const bool flow_g = !(th > 0 && fabs(dPg0) <= th);
  const Real kw_i = 1 + g * (Real)0.5 * drho_w_i * dz;
  const Real kw_j = 1 - g * (Real)0.5 * drho_w_j * dz;
  const Real ko_i = 1 + g * (Real)0.5 * drho_o_i * dz;
  const Real ko_j = 1 - g * (Real)0.5 * drho_o_j * dz;
  const Real kg_i = 1 + g * (Real)0.5 * drho_g_i * dz;
  const Real kg_j = 1 - g * (Real)0.5 * drho_g_j * dz;
  const bool wj = dPw > 0.0, oj = dPo > 0.0, gj = dPg > 0.0;

  const bool exn = V.aimflag[nc] < 0.5;
  const Real son0 = u_old[3 * nc + 1];
  const Real meaning_old_j = P.gas_primary_old ? P.gas_primary_old[nc] : (Real)0;
  const bool rs_old_j = meaning_old_j > (Real)0.5 && meaning_old_j < (Real)1.5;
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

  const PvtCell& Uw = wj ? N : F;
  const PvtCell& Uo = oj ? N : F;
  const PvtCell& Ug = gj ? N : F;
  const Real bo_u = Uo.bo, bw_u = Uw.bw, bg_u = Ug.bg, rs_u = Uo.rs, rv_u = Ug.rv;

  const int upw = wj ? nc : cell;
  const int upo = oj ? nc : cell;
  Mob3 A, B;
  if (wj == oj && oj == gj) {
    A = cell_props[upw].mob;
    B = A;
  } else {
    A = relperm3(SwU, SgU, T, Uw.muw, Uo.muo, Ug.mug,
                 1.0, 1.0, 1.0, P.swi, P.swr, upw);
    B = relperm3(SwO, SgO, T, Uw.muw, Uo.muo, Ug.mug,
                 1.0, 1.0, 1.0, P.swi, P.swr, upo);
  }
  if (!flow_w) A.Mw = A.dMw_dSw = 0;
  if (Ug.mug > 0) {
    const Real dmu = A.Mg * Ug.dmug_dso / Ug.mug;
    A.dMg_dSw += dmu;
    A.dMg_dSg += dmu;
  }
  if (!flow_g) A.Mg = A.dMg_dSw = A.dMg_dSg = 0;
  if (!flow_o) B.Mo = B.dMo_dSw = B.dMo_dSg = 0;

  TcOps L = {};
  tc_flux(L, Tg, A.Mw, B.Mo, A.Mg, dPw, dPo, dPg, bw_u, bo_u, bg_u, rs_u, rv_u);

  // Pressure column of both cells.
  L.p[TZ_DPW][0] = -kw_i;  L.p[TZ_DPW][3] = kw_j;
  L.p[TZ_DPO][0] = -ko_i;  L.p[TZ_DPO][3] = ko_j;
  L.p[TZ_DPG][0] = -kg_i;  L.p[TZ_DPG][3] = kg_j;
  L.p[TZ_BW][wj ? 3 : 0] = Uw.dbw;
  L.p[TZ_BO][oj ? 3 : 0] = Uo.dbo;
  L.p[TZ_BG][gj ? 3 : 0] = Ug.dbg;
  L.p[TZ_RS][oj ? 3 : 0] = Uo.drs;
  L.p[TZ_RV][gj ? 3 : 0] = Ug.drv;

  if (!P.gas_primary) {
    if (!ex_o) {
      L.p[TZ_MO][oj ? 4 : 1] = B.dMo_dSw;
      L.p[TZ_MO][oj ? 5 : 2] = B.dMo_dSg;
    }
    if (!ex_g) {
      L.p[TZ_MG][gj ? 4 : 1] = A.dMg_dSw;
      L.p[TZ_MG][gj ? 5 : 2] = A.dMg_dSg;
    }
    if (!ex_w) L.p[TZ_MW][wj ? 4 : 1] = A.dMw_dSw;
    // Rv and 1/Bg respond to So at fixed history, and dSo/dSw = dSo/dSg = -1.
    L.p[TZ_RV][gj ? 4 : 1] = -Ug.drv_dso;
    L.p[TZ_RV][gj ? 5 : 2] = -Ug.drv_dso;
    L.p[TZ_BG][gj ? 4 : 1] = -Ug.dbg_dso;
    L.p[TZ_BG][gj ? 5 : 2] = -Ug.dbg_dso;
    if (T.have_pc) {
      L.p[TZ_DPW][1] = Ci.dpcow;   L.p[TZ_DPW][4] = -Cj.dpcow;
      L.p[TZ_DPG][2] = -Ci.dpcog;  L.p[TZ_DPG][5] = Cj.dpcog;
    }
  } else {
    // Variable substitution: the third primary is Sg, Rs or Rv, cell by cell.
    const Real dsg_sw_i = rv_i ? (Real)-1 : (Real)0;
    const Real dsg_sw_j = rv_j ? (Real)-1 : (Real)0;
    const Real drho_o_sw_i = F.drs_dsw * P.rhog / ft3_per_bbl * F.bo +
                             oil_mass_i * F.dbo_dsw;
    const Real drho_o_sw_j = N.drs_dsw * P.rhog / ft3_per_bbl * N.bo +
                             oil_mass_j * N.dbo_dsw;
    const Real drho_g_sw_i = F.drv_dsw * P.rhoo * F.bg + gas_mass_i * F.dbg_dsw;
    const Real drho_g_sw_j = N.drv_dsw * P.rhoo * N.bg + gas_mass_j * N.dbg_dsw;
    const Real dpo_phase_sw_i = rv_i ? -Ci.dpcog * dsg_sw_i : (Real)0;
    const Real dpo_phase_sw_j = rv_j ? -Cj.dpcog * dsg_sw_j : (Real)0;
    L.p[TZ_DPO][1] = -dpo_phase_sw_i - g * (Real)0.5 * drho_o_sw_i * dz;
    L.p[TZ_DPO][4] = dpo_phase_sw_j - g * (Real)0.5 * drho_o_sw_j * dz;
    L.p[TZ_DPW][1] = -dpo_phase_sw_i + Ci.dpcow;
    L.p[TZ_DPW][4] = dpo_phase_sw_j - Cj.dpcow;
    L.p[TZ_DPG][1] = -dpo_phase_sw_i - Ci.dpcog * dsg_sw_i -
                     g * (Real)0.5 * drho_g_sw_i * dz;
    L.p[TZ_DPG][4] = dpo_phase_sw_j + Cj.dpcog * dsg_sw_j -
                     g * (Real)0.5 * drho_g_sw_j * dz;
    const Real dsg_sw_o = oj ? dsg_sw_j : dsg_sw_i;
    const Real dsg_sw_g = gj ? dsg_sw_j : dsg_sw_i;
    Real dmo_sw = B.dMo_dSw + B.dMo_dSg * dsg_sw_o;
    if (Uo.muo > 0) dmo_sw -= B.Mo * Uo.dmuo_dsw / Uo.muo;
    const Real dmg_sw = A.dMg_dSw + A.dMg_dSg * dsg_sw_g;
    if (!ex_w) L.p[TZ_MW][wj ? 4 : 1] = A.dMw_dSw;
    if (!ex_o) L.p[TZ_MO][oj ? 4 : 1] = dmo_sw;
    if (!ex_g) L.p[TZ_MG][gj ? 4 : 1] = dmg_sw;
    L.p[TZ_BO][oj ? 4 : 1] = Uo.dbo_dsw;
    L.p[TZ_BG][gj ? 4 : 1] = Ug.dbg_dsw;
    L.p[TZ_RS][oj ? 4 : 1] = Uo.drs_dsw;
    L.p[TZ_RV][gj ? 4 : 1] = Ug.drv_dsw;

    const Real ds_i = (rs_i || rv_i) ? (Real)0 : (Real)1;
    const Real ds_j = (rs_j || rv_j) ? (Real)0 : (Real)1;
    const Real drho_o_x_i = F.drs_dx * P.rhog / ft3_per_bbl * F.bo +
                            oil_mass_i * F.dbo_dx;
    const Real drho_o_x_j = N.drs_dx * P.rhog / ft3_per_bbl * N.bo +
                            oil_mass_j * N.dbo_dx;
    const Real drho_g_x_i = F.drv_dx * P.rhoo * F.bg + gas_mass_i * F.dbg_dx;
    const Real drho_g_x_j = N.drv_dx * P.rhoo * N.bg + gas_mass_j * N.dbg_dx;
    L.p[TZ_DPO][2] = -g * (Real)0.5 * drho_o_x_i * dz;
    L.p[TZ_DPO][5] = -g * (Real)0.5 * drho_o_x_j * dz;
    L.p[TZ_DPG][2] = -Ci.dpcog * ds_i - g * (Real)0.5 * drho_g_x_i * dz;
    L.p[TZ_DPG][5] = Cj.dpcog * ds_j - g * (Real)0.5 * drho_g_x_j * dz;
    const bool xg_rv = gj ? rv_j : rv_i;
    const Real ds_o = oj ? ds_j : ds_i;
    const Real ds_g = gj ? ds_j : ds_i;
    Real dmo_x = B.dMo_dSg * ds_o;
    if (Uo.muo > 0) dmo_x -= B.Mo * Uo.dmuo_dx / Uo.muo;
    Real dmg_x = A.dMg_dSg * ds_g;
    if (xg_rv && Ug.mug > 0) dmg_x -= A.Mg * Ug.dmug_dx / Ug.mug;
    if (!ex_o) L.p[TZ_MO][oj ? 5 : 2] = dmo_x;
    if (!ex_g) L.p[TZ_MG][gj ? 5 : 2] = dmg_x;
    L.p[TZ_BO][oj ? 5 : 2] = Uo.dbo_dx;
    L.p[TZ_BG][gj ? 5 : 2] = Ug.dbg_dx;
    L.p[TZ_RS][oj ? 5 : 2] = Uo.drs_dx;
    L.p[TZ_RV][gj ? 5 : 2] = Ug.drv_dx;
  }
  tc_emit(L, 3, q, cell, V);
}

// The owning cell of each connection, so a connection can be its own thread.
__global__ void k_tc_owner(const int* conn_ptr, int* owner, int cells) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cells) return;
  for (int m = conn_ptr[cell]; m < conn_ptr[cell + 1]; ++m) owner[m] = cell;
}

// Which cell a connection belongs to, where its block sits in the row, and
// whether the analytic kernel assembled it at all.
__device__ __forceinline__ bool tc_locate(const TcView& V, long long m, int& cell,
                                          int& off) {
  if (V.conn) {
    cell = V.conn_owner[m];
    off = V.conn_bcol[m];
    return V.conn_trans[m] > 0;
  }
  cell = (int)(m / 6);
  const int f = (int)(m % 6);
  off = V.nbr[7 * cell + 1 + f] >= 0 ? V.bcol[7 * cell + 1 + f] : -1;
  return V.nbr[7 * cell + 1 + f] >= 0 && V.tface[6 * cell + f] > 0;
}

// TCVALIDATE: the product against the analytic neighbour block, still in the CSR.
__global__ void k_tc_validate(TcView V, long long first, int count, int bs) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= count) return;
  int cell, off;
  if (!tc_locate(V, first + q, cell, off) || off < 0) return;
  const float* tile = V.prod + (size_t)q * tc::kTile * tc::kTile;
  Real d2 = 0, n2 = 0, dmax = 0, nmax = 0;
  for (int a = 0; a < bs; ++a)
    for (int v = 0; v < bs; ++v) {
      const Real ref = V.aval[V.rowptr[bs * cell + a] + off + v];
      const Real d = (Real)tile[a * tc::kTile + bs + v] - ref;
      d2 += d * d;
      n2 += ref * ref;
      dmax = fmax(dmax, fabs(d));
      nmax = fmax(nmax, fabs(ref));
    }
  atomicAdd(V.stats + 0, d2);
  atomicAdd(V.stats + 1, n2);
  if (nmax > 0) atomic_max_pos(V.worst, dmax / nmax);
}

// Neighbour blocks from the product, and its share of the self block.
__global__ void k_tc_scatter(TcView V, long long first, int count, int bs) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= count) return;
  int cell, off;
  if (!tc_locate(V, first + q, cell, off)) return;
  const float* tile = V.prod + (size_t)q * tc::kTile * tc::kTile;
  for (int a = 0; a < bs; ++a)
    for (int v = 0; v < bs; ++v) {
      atomicAdd(V.dself + (long long)cell * bs * bs + a * bs + v,
                (Real)tile[a * tc::kTile + v]);
      if (off >= 0)
        V.aval[V.rowptr[bs * cell + a] + off + v] =
            (Real)tile[a * tc::kTile + bs + v];
    }
}

// Apply the self correction and rebuild what depends on the blocks: the stage-2
// inverse and the pressure row.  The CPR weights stay as the analytic kernel
// chose them, which is what the DRS decoupling and true-IMPES paths need.
__global__ void k_tc_finish(TcView V, int cells, int bs, int validate) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cells) return;
  const int soff = V.conn ? V.self_bcol[cell] : V.bcol[7 * cell];
  const int spc = V.conn ? V.self_pcol[cell] : V.pcolpos[7 * cell];
  const Real* d = V.dself + (long long)cell * bs * bs;
  Real wgt[3] = {0, 0, 0};
  for (int a = 0; a < bs; ++a) wgt[a] = (Real)V.pweight[bs * cell + a];

  Real flat[9] = {};
  if (soff >= 0) {
    Real d2 = 0, n2 = 0, dmax = 0, nmax = 0;
    for (int a = 0; a < bs; ++a)
      for (int v = 0; v < bs; ++v) {
        Real& e = V.aval[V.rowptr[bs * cell + a] + soff + v];
        d2 += d[a * bs + v] * d[a * bs + v];
        n2 += e * e;
        dmax = fmax(dmax, fabs(d[a * bs + v]));
        nmax = fmax(nmax, fabs(e));
        e += d[a * bs + v];
        flat[a * bs + v] = e;
      }
    if (validate) {
      atomicAdd(V.stats + 2, d2);
      atomicAdd(V.stats + 3, n2);
      if (nmax > 0) atomic_max_pos(V.worst + 1, dmax / nmax);
    }
    Real Inv[9];
    const bool ok = invert_block(flat, bs, Inv);
    store_dinv(Inv, ok, V.dinv + (long long)cell * bs * bs, bs);
    if (spc >= 0) {
      Real jp[3];
      for (int a = 0; a < bs; ++a) jp[a] = flat[a * bs];
      V.pval[V.prowptr[cell] + spc] = (PrecReal)pressure_row_entry(wgt, jp, bs);
    }
  }
  const int nm = V.conn ? V.conn_ptr[cell + 1] - V.conn_ptr[cell] : 6;
  for (int k = 0; k < nm; ++k) {
    int off, pcp;
    if (V.conn) {
      const int m = V.conn_ptr[cell] + k;
      off = V.conn_bcol[m];
      pcp = V.conn_pcol[m];
    } else {
      if (V.nbr[7 * cell + 1 + k] < 0) continue;
      off = V.bcol[7 * cell + 1 + k];
      pcp = V.pcolpos[7 * cell + 1 + k];
    }
    if (off < 0 || pcp < 0) continue;
    Real jp[3];
    for (int a = 0; a < bs; ++a) jp[a] = V.aval[V.rowptr[bs * cell + a] + off];
    V.pval[V.prowptr[cell] + pcp] = (PrecReal)pressure_row_entry(wgt, jp, bs);
  }
}

struct TcState {
  int* conn_owner = nullptr;
  float* lflux = nullptr; float* lprops = nullptr; float* prod = nullptr;
  Real* dself = nullptr; Real* stats = nullptr;
  unsigned long long* worst = nullptr;
  long long nconn = 0;
  int capacity = 0;
  long long assemblies = 0, products = 0;
  double t_analytic = 0, t_fill = 0, t_product = 0, t_scatter = 0;
  double v_neighbour = 0, v_self = 0, v_worst_neighbour = 0, v_worst_self = 0;
} g_tc;

double tc_clock() {
  NVRS_CHECK_CUDA(cudaDeviceSynchronize());
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// How many connections fit: three 16x16 FP32 tiles each, within a share of the
// free memory, so a large field runs in several batches instead of failing.
int tc_capacity(long long nconn) {
  size_t free_b = 0, total_b = 0;
  NVRS_CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
  const size_t tile_b = 3 * tc::kTile * tc::kTile * sizeof(float);
  const long long fit = (long long)(free_b / 4 / tile_b);
  const long long cap = std::min<long long>(nconn, std::min<long long>(fit, 1 << 24));
  if (cap < 1)
    throw std::runtime_error("TENSORCORE: not enough GPU memory for one batch");
  return (int)cap;
}

TcView tc_view(const Reservoir& r, DeviceLinearSystem& sys, const BOParams& P,
               const AssemblyCellProps* props) {
  TcView V{};
  V.u = r.u; V.u_old = r.u_old; V.aimflag = r.aimflag;
  V.conn = r.use_conn_list && r.conn_ptr;
  V.conn_ptr = r.conn_ptr; V.conn_owner = g_tc.conn_owner; V.conn_nbr = r.conn_nbr;
  V.conn_trans = r.conn_trans; V.conn_dz = r.conn_dz;
  V.conn_threshold = r.conn_threshold;
  V.conn_bcol = r.conn_bcol; V.conn_pcol = r.conn_pcol;
  V.self_bcol = r.self_bcol; V.self_pcol = r.self_pcol;
  V.tface = r.tface; V.nbr = r.nbr; V.cellk = r.cellk;
  V.bcol = r.bcol; V.pcolpos = r.pcolpos;
  V.rowptr = sys.rowptr; V.prowptr = sys.prowptr;
  V.props = props; V.T = r.tables; V.P = P;
  V.lflux = g_tc.lflux; V.lprops = g_tc.lprops; V.prod = g_tc.prod;
  V.dself = g_tc.dself; V.aval = sys.aval64; V.pval = sys.pval; V.dinv = sys.dinv;
  V.pweight = sys.pweight;
  V.stats = g_tc.stats; V.worst = g_tc.worst;
  return V;
}

void tc_allocate(const Reservoir& r, int cells) {
  const bool conn = r.use_conn_list && r.conn_ptr;
  int n = 6 * cells;
  if (conn) {
    NVRS_CHECK_CUDA(cudaMemcpy(&n, r.conn_ptr + cells, sizeof(int),
                               cudaMemcpyDeviceToHost));
    NVRS_CHECK_CUDA(cudaMalloc(&g_tc.conn_owner, sizeof(int) * (size_t)(n + 1)));
    k_tc_owner<<<grid1d(cells), 256>>>(r.conn_ptr, g_tc.conn_owner, cells);
  }
  g_tc.nconn = n;
  g_tc.capacity = tc_capacity(n);
  const size_t tile_b = (size_t)g_tc.capacity * tc::kTile * tc::kTile * sizeof(float);
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.lflux, tile_b));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.lprops, tile_b));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.prod, tile_b));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.dself, sizeof(Real) * (size_t)cells * r.bs * r.bs));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.stats, sizeof(Real) * 4));
  NVRS_CHECK_CUDA(cudaMalloc(&g_tc.worst, sizeof(unsigned long long) * 2));
}

// Fill, multiply, validate and scatter every connection, then finish the cells.
// Split out from the driver below so that the test can run it on bare arrays.
void tc_run_connections(const TcView& V, int cells, int bs) {
  const size_t tile_b = (size_t)g_tc.capacity * tc::kTile * tc::kTile * sizeof(float);
  double t = tc_clock();
  NVRS_CHECK_CUDA(cudaMemset(g_tc.dself, 0, sizeof(Real) * (size_t)cells * bs * bs));
  if (tc::g_validate) {
    NVRS_CHECK_CUDA(cudaMemset(g_tc.stats, 0, sizeof(Real) * 4));
    NVRS_CHECK_CUDA(cudaMemset(g_tc.worst, 0, sizeof(unsigned long long) * 2));
  }
  for (long long first = 0; first < g_tc.nconn; first += g_tc.capacity) {
    const int count = (int)std::min<long long>(g_tc.capacity, g_tc.nconn - first);
    const dim3 grid = grid1d(count);
    NVRS_CHECK_CUDA(cudaMemset(g_tc.lflux, 0, tile_b));
    NVRS_CHECK_CUDA(cudaMemset(g_tc.lprops, 0, tile_b));
    if (bs == 2 && V.conn)  k_tc_fill_bo2_conn<<<grid, 256>>>(V, first, count);
    else if (bs == 2)       k_tc_fill_bo2_cart<<<grid, 256>>>(V, first, count);
    else if (V.conn)        k_tc_fill_bo3_conn<<<grid, 256>>>(V, first, count);
    else                    k_tc_fill_bo3_cart<<<grid, 256>>>(V, first, count);
    NVRS_CHECK_CUDA(cudaGetLastError());
    const double t1 = tc_clock();
    g_tc.t_fill += t1 - t;

    tc::batched_operator_product(g_tc.lflux, g_tc.lprops, g_tc.prod, count,
                                 tc::g_precision);
    NVRS_CHECK_CUDA(cudaGetLastError());
    t = tc_clock();
    g_tc.t_product += t - t1;
    g_tc.products += count;

    if (tc::g_validate) k_tc_validate<<<grid, 256>>>(V, first, count, bs);
    k_tc_scatter<<<grid, 256>>>(V, first, count, bs);
    NVRS_CHECK_CUDA(cudaGetLastError());
  }
  k_tc_finish<<<grid1d(cells), 256>>>(V, cells, bs, tc::g_validate);
  NVRS_CHECK_CUDA(cudaGetLastError());
  g_tc.t_scatter += tc_clock() - t;

  if (tc::g_validate) {
    Real s[4];
    unsigned long long w[2];
    NVRS_CHECK_CUDA(cudaMemcpy(s, g_tc.stats, sizeof(s), cudaMemcpyDeviceToHost));
    NVRS_CHECK_CUDA(cudaMemcpy(w, g_tc.worst, sizeof(w), cudaMemcpyDeviceToHost));
    const auto bits = [](unsigned long long b) {
      double v; std::memcpy(&v, &b, sizeof(v)); return v;
    };
    g_tc.v_neighbour = std::max(g_tc.v_neighbour,
                                s[1] > 0 ? std::sqrt((double)(s[0] / s[1])) : 0.0);
    g_tc.v_self = std::max(g_tc.v_self,
                           s[3] > 0 ? std::sqrt((double)(s[2] / s[3])) : 0.0);
    g_tc.v_worst_neighbour = std::max(g_tc.v_worst_neighbour, bits(w[0]));
    g_tc.v_worst_self = std::max(g_tc.v_worst_self, bits(w[1]));
  }
}

// Replace the connection part of the system the analytic kernel just assembled.
// Runs before the residual statistics are read and before anything uses the
// matrix, so the rest of the Newton step is unaware of it.
void tc_replace_connection_blocks(const Reservoir& r, DeviceLinearSystem& sys,
                                  const BOParams& P,
                                  const AssemblyCellProps* props, double t0) {
  if (P.bs != 2 && P.bs != 3)
    throw std::runtime_error("JACOBIAN TENSORCORE covers two- and three-phase "
                             "black oil only");
  const int cells = r.grid.n_local();
  if (!g_tc.lflux) tc_allocate(r, cells);
  g_tc.t_analytic += tc_clock() - t0;
  ++g_tc.assemblies;
  tc_run_connections(tc_view(r, sys, P, props), cells, P.bs);
}

}  // namespace

void tc::report() {
  if (!g_tc.assemblies) return;
  std::printf("\n  [tc] Jacobian product : %s\n", tc::backend_name(tc::g_precision));
  std::printf("  [tc] assemblies %lld, connections %lld, tiles per batch %d, "
              "buffers %.2f GB\n", g_tc.assemblies, g_tc.nconn, g_tc.capacity,
              3.0 * g_tc.capacity * tc::kTile * tc::kTile * sizeof(float) / 1e9);
  std::printf("  [tc] seconds  analytic kernel %.3f  fill %.3f  product %.3f  "
              "scatter %.3f\n", g_tc.t_analytic, g_tc.t_fill, g_tc.t_product,
              g_tc.t_scatter);
  if (g_tc.t_product > 0)
    std::printf("  [tc] product rate %.3g tiles/s\n",
                (double)g_tc.products / g_tc.t_product);
  if (tc::g_validate)
    std::printf("  [tc] validate, worst assembly, product against analytic:\n"
                "       neighbour blocks  relative Frobenius %.3e  worst block "
                "%.3e\n       self blocks       relative Frobenius %.3e  worst "
                "block %.3e\n", g_tc.v_neighbour, g_tc.v_worst_neighbour,
                g_tc.v_self, g_tc.v_worst_self);
  std::fflush(stdout);
}
