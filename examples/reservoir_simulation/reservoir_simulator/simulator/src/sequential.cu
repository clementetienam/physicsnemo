// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// sequential.cu — the CuPy reference scheme (NVR.py::Reservoir_Simulator2),
// ported to CUDA and solved with the mixed-precision CPR machinery.
// See sequential.hpp for the formulation.
#include "sequential.hpp"

#include "halo.hpp"
#include "wellcell.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace nvrs {

// ===========================================================================
// Kernels
// ===========================================================================

// Total mobility, with the reference's weighting:
//     Mt = Mw/Bw + Mo/Bo (1 + Rs) + Mg/Bg
__global__ static void k_total_mobility(const Real* u, const Real* /*unused*/,
                                        SatTables T, int cells, int bs,
                                        Real uw, Real uo, Real ug, Real bw,
                                        Real bo, Real bg, Real rs, Real swi,
                                        Real swr, int three_phase, Real* mt) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const Real Sw = u[bs * c + 1];
  const Real Sg = (bs == 3) ? u[bs * c + 2] : 0.0;
  Real Mw, Mo, Mg;
  if (three_phase) {
    Mob3 m = relperm3(Sw, Sg, T, uw, uo, ug, bw, bo, bg, swi, swr);
    Mw = m.Mw; Mo = m.Mo; Mg = m.Mg;
  } else {
    Mob2 m = relperm2(Sw, uw, uo, bw, bo, swi, swr);
    Mw = m.Mw; Mo = m.Mo; Mg = 0.0;
  }
  // relperm* already divide by (mu * B); the reference's extra 1/B factors and
  // the (1 + Rs) dissolved-gas weight are applied here.
  mt[c] = Mw + Mo * (1.0 + rs) + Mg;
}

// Harmonic face transmissibility of K*Mt on the normalised unit-cube geometry,
// matching  TX = tx / (Ll_i + Ll_j),  Ll = 1/(K Mt).
__global__ static void k_face_trans(const Real* perm, const Real* mt,
                                    const int* nbr, int cells, Real tx, Real ty,
                                    Real tz, Real kz_mult, Real* tface) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  for (int f = 0; f < 6; ++f) {
    const int nc = nbr[7 * c + 1 + f];
    if (nc < 0) { tface[6 * c + f] = 0.0; continue; }
    const Real kdir_c = (f < 4) ? perm[c] : kz_mult * perm[c];
    const Real kdir_n = (f < 4) ? perm[nc] : kz_mult * perm[nc];
    const Real km_c = fmax(kdir_c * mt[c], 1e-30);
    const Real km_n = fmax(kdir_n * mt[nc], 1e-30);
    const Real t = (f < 2) ? tx : ((f < 4) ? ty : tz);
    // 1/Ll_c + 1/Ll_n in the reference's notation is (1/km_c + 1/km_n).
    tface[6 * c + f] = t / (1.0 / km_c + 1.0 / km_n);
  }
}

// Scalar 7-point pressure operator.  Diagonal is the sum of the face
// transmissibilities; off-diagonals are their negatives.  rhs is the fixed well
// source.  Cell 0 of rank 0 is pinned, as the reference pins A[0,0], because
// the pure-Neumann system is otherwise singular.
__global__ static void k_assemble_pressure(const Real* tface, const int* nbr,
                                           const int* pcolpos, const int* rowptr,
                                           const Real* qtot, int cells,
                                           long long row0, Real pin,
                                           Real* aval, Real* b, PrecReal* pval,
                                           PrecReal* dinv, PrecReal* pweight) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const int r0 = rowptr[c];
  Real diag = 0.0;
  for (int f = 0; f < 6; ++f) {
    const int nc = nbr[7 * c + 1 + f];
    if (nc < 0) continue;
    const Real t = tface[6 * c + f];
    diag += t;
    const int pos = pcolpos[7 * c + 1 + f];
    if (pos >= 0) {
      aval[r0 + pos] = -t;
      pval[r0 + pos] = (PrecReal)(-t);
    }
  }
  if (row0 == 0 && c == 0) diag += pin;      // pin the singular system
  const int dpos = pcolpos[7 * c + 0];
  if (dpos >= 0) {
    aval[r0 + dpos] = diag;
    pval[r0 + dpos] = (PrecReal)diag;
  }
  b[c] = qtot[c];
  dinv[c] = (PrecReal)((fabs(diag) > 1e-30) ? 1.0 / diag : 0.0);
  pweight[c] = (PrecReal)1;                  // scalar system: W = identity
}

// Darcy face velocity, outward positive:  v_f = (p_c - p_nb) * T_f.
__global__ static void k_face_velocity(const Real* p, const Real* tface,
                                       const int* nbr, int cells, Real* vface) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  for (int f = 0; f < 6; ++f) {
    const int nc = nbr[7 * c + 1 + f];
    vface[6 * c + f] = (nc < 0) ? 0.0 : (p[c] - p[nc]) * tface[6 * c + f];
  }
}

// Fractional flow f_a and its saturation derivative, for the phase selected by
// `phase` (0 = water, 1 = gas).
__device__ __forceinline__ void frac_flow(Real Sw, Real Sg, const SatTables& T,
                                          Real uw, Real uo, Real ug, Real bw,
                                          Real bo, Real bg, Real swi, Real swr,
                                          int three_phase, int phase, Real* f,
                                          Real* df) {
  if (three_phase) {
    Mob3 m = relperm3(Sw, Sg, T, uw, uo, ug, bw, bo, bg, swi, swr);
    const Real den = m.Mw + m.Mo + m.Mg + 1e-30;
    const Real dMsum = (phase == 0)
                           ? (m.dMw_dSw + m.dMo_dSw)
                           : (m.dMo_dSg + m.dMg_dSg);
    const Real Ma = (phase == 0) ? m.Mw : m.Mg;
    const Real dMa = (phase == 0) ? m.dMw_dSw : m.dMg_dSg;
    *f = Ma / den;
    *df = dMa / den - Ma / (den * den) * dMsum;
  } else {
    Mob2 m = relperm2(Sw, uw, uo, bw, bo, swi, swr);
    const Real den = m.Mw + m.Mo + 1e-30;
    *f = m.Mw / den;
    *df = m.dMw / den - m.Mw / (den * den) * (m.dMw + m.dMo);
  }
}

// Transport residual and Jacobian for one phase, matching the reference's
//     G  = S - S0 - (B f + fi_inj + fi_prod f)
//     dG = I - B diag(df)
// with B the upwind flux operator of NVR.py::GenA scaled by dt/pv.  The upwind
// operator takes, per face, the *outflow* part into the diagonal and the
// *inflow* part into the off-diagonal, which is what min/max on the signed face
// velocity accomplishes there.
__global__ static void k_transport(const Real* u, const Real* s_old,
                                   const Real* vface, const Real* qtot,
                                   const Real* qwat, const Real* pv,
                                   const int* nbr, const int* pcolpos,
                                   const int* rowptr, SatTables T, int cells,
                                   int bs, Real uw, Real uo, Real ug, Real bw,
                                   Real bo, Real bg, Real swi, Real swr,
                                   int three_phase, int phase, Real dt,
                                   Real* aval, Real* b, PrecReal* pval,
                                   PrecReal* dinv, PrecReal* pweight,
                                   Real* rnorm2) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const int r0 = rowptr[c];
  const Real dtx = dt / fmax(pv[c], 1e-30);

  const Real Sw = u[bs * c + 1];
  const Real Sg = (bs == 3) ? u[bs * c + 2] : 0.0;
  const Real S = (phase == 0) ? Sw : Sg;

  Real fc, dfc;
  frac_flow(Sw, Sg, T, uw, uo, ug, bw, bo, bg, swi, swr, three_phase, phase,
            &fc, &dfc);

  // Well terms: injection enters directly, production scales with f.
  const Real q_inj = (phase == 0) ? fmax(qwat[c], 0.0) : 0.0;
  const Real q_prod = fmin(qtot[c], 0.0);

  // Flux balance.  Outflow (v > 0 leaving this cell) multiplies this cell's f;
  // inflow (v < 0) multiplies the neighbour's f.
  Real flux = q_inj * dtx + q_prod * dtx * fc;
  Real diag = 1.0 - q_prod * dtx * dfc;          // I - d/dS of the well term

  for (int f = 0; f < 6; ++f) {
    const int nc = nbr[7 * c + 1 + f];
    if (nc < 0) continue;
    const Real v = vface[6 * c + f];
    const int pos = pcolpos[7 * c + 1 + f];
    if (v > 0.0) {
      // Outflow: this cell is upstream.
      flux -= v * dtx * fc;
      diag += v * dtx * dfc;
    } else if (v < 0.0) {
      // Inflow: the neighbour is upstream.
      const Real Swn = u[bs * nc + 1];
      const Real Sgn = (bs == 3) ? u[bs * nc + 2] : 0.0;
      Real fn, dfn;
      frac_flow(Swn, Sgn, T, uw, uo, ug, bw, bo, bg, swi, swr, three_phase,
                phase, &fn, &dfn);
      flux -= v * dtx * fn;                       // v < 0, so this adds inflow
      if (pos >= 0) {
        const Real off = v * dtx * dfn;
        aval[r0 + pos] = off;
        pval[r0 + pos] = (PrecReal)off;
      }
      continue;
    }
    if (pos >= 0) { aval[r0 + pos] = 0.0; pval[r0 + pos] = (PrecReal)0; }
  }

  const Real G = S - s_old[c] - flux;
  const int dpos = pcolpos[7 * c + 0];
  if (dpos >= 0) {
    aval[r0 + dpos] = diag;
    pval[r0 + dpos] = (PrecReal)diag;
  }
  // Solve dG * ds = -G so that S += ds moves toward the root.
  b[c] = -G;
  dinv[c] = (PrecReal)((fabs(diag) > 1e-30) ? 1.0 / diag : 0.0);
  pweight[c] = (PrecReal)1;
  atomicAdd(rnorm2, G * G);
}

// Apply a damped saturation update with the reference's joint clipping:
// Sw into [Swi, 1], then Sg into [0, 1 - Sw].
__global__ static void k_apply_sat(Real* u, const Real* ds, Real alpha, int bs,
                                   int cells, int phase, Real swi) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) return;
  const int idx = bs * c + (phase == 0 ? 1 : 2);
  u[idx] += alpha * ds[c];
  Real Sw = u[bs * c + 1];
  Sw = fmin(fmax(Sw, swi), 1.0);
  u[bs * c + 1] = Sw;
  if (bs == 3) u[bs * c + 2] = fmin(fmax(u[bs * c + 2], 0.0), 1.0 - Sw);
}

// ===========================================================================
// Host side
// ===========================================================================
void seq_build(const Reservoir& r, const Deck& d, SeqSolver& S) {
  const Grid& g = r.grid;
  const int cells = g.n_local();
  const int cells_ext = g.n_local_ext();

  // --- scalar (bs = 1) sparsity, mirroring the 7-point stencil ------------
  std::vector<int> h_nbr(7 * cells);
  NVRS_CHECK_CUDA(cudaMemcpy(h_nbr.data(), r.nbr, sizeof(int) * 7 * cells,
                             cudaMemcpyDeviceToHost));
  std::vector<int> h_rowptr(cells + 1, 0), h_colind, h_gcol;
  std::vector<int> h_pos(7 * cells, -1);
  for (int c = 0; c < cells; ++c) {
    std::vector<int> members;
    for (int m = 0; m < 7; ++m) {
      const int nc = (m == 0) ? c : h_nbr[7 * c + m];
      if (nc >= 0) members.push_back(nc);
    }
    std::sort(members.begin(), members.end());
    members.erase(std::unique(members.begin(), members.end()), members.end());
    h_rowptr[c + 1] = h_rowptr[c] + (int)members.size();
    for (int mc : members) {
      h_colind.push_back(mc);
      h_gcol.push_back(g.global_cell(mc));
    }
    for (int m = 0; m < 7; ++m) {
      const int nc = (m == 0) ? c : h_nbr[7 * c + m];
      if (nc < 0) continue;
      auto it = std::lower_bound(members.begin(), members.end(), nc);
      h_pos[7 * c + m] = (int)(it - members.begin());
    }
  }

  DeviceLinearSystem& sys = S.sys;
  sys.bs = 1;
  sys.cells = cells;
  sys.n = cells;
  sys.cells_ext = cells_ext;
  sys.n_ext = cells_ext;
  sys.nnz = (int)h_colind.size();
  sys.pnnz = sys.nnz;
  sys.row0 = g.row0();
  sys.cells_global = g.n_global();

  auto up_i = [](int** dst, const std::vector<int>& h) {
    NVRS_CHECK_CUDA(cudaMalloc(dst, sizeof(int) * h.size()));
    NVRS_CHECK_CUDA(cudaMemcpy(*dst, h.data(), sizeof(int) * h.size(),
                               cudaMemcpyHostToDevice));
  };
  up_i(&sys.rowptr, h_rowptr);
  up_i(&sys.colind, h_colind);
  up_i(&sys.prowptr, h_rowptr);
  up_i(&sys.pcolind, h_gcol);
  up_i(&S.pos, h_pos);

  NVRS_CHECK_CUDA(cudaMalloc(&sys.aval64, sizeof(Real) * sys.nnz));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.b, sizeof(Real) * cells));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.pval, sizeof(PrecReal) * sys.pnnz));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.dinv, sizeof(PrecReal) * cells));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.pweight, sizeof(PrecReal) * cells));
  // df64 (hi, lo) slices, so the emulated-FP64 operator works here too and the
  // mixed-precision path is genuinely exercised by the sequential scheme.
  if (d.emulate_fp64) {
    NVRS_CHECK_CUDA(cudaMalloc(&sys.aval_hi, sizeof(float) * sys.nnz));
    NVRS_CHECK_CUDA(cudaMalloc(&sys.aval_lo, sizeof(float) * sys.nnz));
  }

  // --- workspace ----------------------------------------------------------
  auto alloc = [](Real** p, size_t n) {
    NVRS_CHECK_CUDA(cudaMalloc(p, sizeof(Real) * n));
    NVRS_CHECK_CUDA(cudaMemset(*p, 0, sizeof(Real) * n));
  };
  alloc(&S.p, cells_ext);
  alloc(&S.mt, cells_ext);
  alloc(&S.tface, (size_t)cells * 6);
  alloc(&S.vface, (size_t)cells * 6);
  alloc(&S.qtot, cells);
  alloc(&S.qwat, cells);
  alloc(&S.pv, cells);
  alloc(&S.s_old, cells);
  alloc(&S.work, cells_ext);

  // --- normalised unit-cube face geometry, as the reference uses ----------
  const Real hx = 1.0 / g.nx, hy = 1.0 / g.ny, hz = 1.0 / g.nz;
  S.tx = 2.0 * hy * hz / hx;
  S.ty = 2.0 * hx * hz / hy;
  S.tz = 2.0 * hx * hy / hz;

  // --- pore volume and fixed well sources ---------------------------------
  std::vector<Real> h_pv(cells), h_qt(cells, 0.0), h_qw(cells, 0.0);
  std::vector<Real> h_poro(cells);
  NVRS_CHECK_CUDA(cudaMemcpy(h_poro.data(), r.poro, sizeof(Real) * cells,
                             cudaMemcpyDeviceToHost));
  // Reservoir barrels, not ft^3 -- see the note in assemble_blackoil.cu.
  const Real vol = d.dx * d.dy * d.dz / units::kFt3PerBbl;
  for (int c = 0; c < cells; ++c) h_pv[c] = vol * h_poro[c];

  // Wells are fixed-rate sources: injectors positive, producers negative,
  // split evenly across their perforations.
  for (const auto& w : d.wells) {
    const int i = w.i - 1, j = w.j - 1;
    if (i < 0 || i >= g.nx || j < 0 || j >= g.ny) continue;
    // Under a 3-D cut the well's areal column belongs to only some ranks.
    if (!g.owns_column(i, j)) continue;
    const int il = i - g.i0, jl = j - g.j0;
    const int klo = (w.perf == "last") ? g.nz - 1 : 0;
    // The rate is split over the GLOBAL perforation count, so the answer does
    // not depend on how the domain was cut.
    const int nperf_glob = g.nz - klo;
    if (nperf_glob <= 0) continue;
    const Real rate = (w.type > 0) ? w.rate : -std::fabs(w.rate);
    for (int kg = klo; kg < g.nz; ++kg) {
      if (kg < g.k0 || kg >= g.k0 + g.nzl) continue;
      const int c = g.lid(il, jl, kg - g.k0);
      h_qt[c] += rate / nperf_glob;
      if (w.type > 0) h_qw[c] += rate / nperf_glob;
    }
  }
  auto upd = [](Real* dst, const std::vector<Real>& h) {
    NVRS_CHECK_CUDA(cudaMemcpy(dst, h.data(), sizeof(Real) * h.size(),
                               cudaMemcpyHostToDevice));
  };
  upd(S.pv, h_pv);
  upd(S.qtot, h_qt);
  upd(S.qwat, h_qw);
}

void seq_free(SeqSolver& S) {
  for (void* p : {(void*)S.p, (void*)S.mt, (void*)S.tface, (void*)S.vface,
                  (void*)S.qtot, (void*)S.qwat, (void*)S.pv, (void*)S.s_old,
                  (void*)S.work, (void*)S.sys.aval64, (void*)S.sys.b,
                  (void*)S.sys.pval, (void*)S.sys.dinv, (void*)S.sys.pweight,
                  (void*)S.sys.rowptr, (void*)S.sys.colind,
                  (void*)S.sys.prowptr, (void*)S.sys.pcolind, (void*)S.pos,
                  (void*)S.sys.aval_hi, (void*)S.sys.aval_lo})
    if (p) cudaFree(p);
  S = SeqSolver{};
}

bool seq_step(Reservoir& r, const Deck& d, SeqSolver& S, CprSolver& solver,
              Real dt, int* newton_iters, int* linear_iters, Real* residual) {
  const Grid& g = r.grid;
  const int cells = g.n_local();
  const int three = (r.bs == 3) ? 1 : 0;
  const dim3 blk = grid1d(cells);

  halo_exchange(r.u, r.bs, g);

  // --- 1. total mobility --------------------------------------------------
  k_total_mobility<<<blk, 256>>>(r.u, nullptr, r.tables, cells, r.bs, r.uw,
                                 r.uo, r.ug, r.bw, r.bo, r.bg, r.rs, r.swi,
                                 r.swr, three, S.mt);
  NVRS_CHECK_CUDA(cudaGetLastError());
  halo_exchange(S.mt, 1, g);

  // --- 2. face transmissibilities and the pressure system -----------------
  k_face_trans<<<blk, 256>>>(r.perm, S.mt, r.nbr, cells, S.tx, S.ty, S.tz,
                             r.kz_mult, S.tface);
  NVRS_CHECK_CUDA(cudaGetLastError());

  // Pin value: the reference adds sum(K) at the first cell.
  Real pin = 0;
  NVRS_CHECK_CUDA(cudaMemcpy(&pin, r.perm, sizeof(Real), cudaMemcpyDeviceToHost));
  pin = (2.0 + r.kz_mult) * pin;

  k_assemble_pressure<<<blk, 256>>>(S.tface, r.nbr, S.pos, S.sys.rowptr, S.qtot,
                                    cells, S.sys.row0, pin, S.sys.aval64,
                                    S.sys.b, S.sys.pval, S.sys.dinv,
                                    S.sys.pweight);
  NVRS_CHECK_CUDA(cudaGetLastError());

  split_operator_df64(S.sys);
  solver.setup(S.sys);
  NVRS_CHECK_CUDA(cudaMemset(S.p, 0, sizeof(Real) * S.sys.n_ext));
  SolveStats pst = solver.solve(S.sys, S.p, true);
  if (linear_iters) *linear_iters += pst.iters;
  halo_exchange(S.p, 1, g);

  // --- 3. Darcy face velocities ------------------------------------------
  k_face_velocity<<<blk, 256>>>(S.p, S.tface, r.nbr, cells, S.vface);
  NVRS_CHECK_CUDA(cudaGetLastError());

  // --- 4. implicit saturation update, water then gas ----------------------
  const int nphase = three ? 2 : 1;
  Real last_res = 0;
  for (int phase = 0; phase < nphase; ++phase) {
    // S0 for this phase
    {
      std::vector<Real> tmp;  // device-side strided copy
      const int off = (phase == 0) ? 1 : 2;
      NVRS_CHECK_CUDA(cudaMemcpy2D(S.s_old, sizeof(Real),
                                   r.u + off, sizeof(Real) * r.bs,
                                   sizeof(Real), cells,
                                   cudaMemcpyDeviceToDevice));
      (void)tmp;
    }

    bool ok = false;
    for (int nw = 0; nw < d.newton_max; ++nw) {
      Real* d_rn = nullptr;
      NVRS_CHECK_CUDA(cudaMalloc(&d_rn, sizeof(Real)));
      NVRS_CHECK_CUDA(cudaMemset(d_rn, 0, sizeof(Real)));
      halo_exchange(r.u, r.bs, g);

      k_transport<<<blk, 256>>>(r.u, S.s_old, S.vface, S.qtot, S.qwat, S.pv,
                                r.nbr, S.pos, S.sys.rowptr, r.tables, cells,
                                r.bs, r.uw, r.uo, r.ug, r.bw, r.bo, r.bg,
                                r.swi, r.swr, three, phase, dt, S.sys.aval64,
                                S.sys.b, S.sys.pval, S.sys.dinv, S.sys.pweight,
                                d_rn);
      NVRS_CHECK_CUDA(cudaGetLastError());
      Real rn2 = 0;
      NVRS_CHECK_CUDA(cudaMemcpy(&rn2, d_rn, sizeof(Real), cudaMemcpyDeviceToHost));
      cudaFree(d_rn);
      if (g.nranks > 1)
        MPI_Allreduce(MPI_IN_PLACE, &rn2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      // The reference's convergence test is the *per-cell mean* residual,
      // ||G||_2 / N, which is grid-size independent.
      last_res = std::sqrt(rn2) / std::max(1, g.n_global());
      if (!std::isfinite((double)last_res)) return false;
      if (last_res < d.newton_tol) { ok = true; break; }

      split_operator_df64(S.sys);
      solver.setup(S.sys);
      NVRS_CHECK_CUDA(cudaMemset(S.work, 0, sizeof(Real) * S.sys.n_ext));
      SolveStats st = solver.solve(S.sys, S.work, true);
      if (linear_iters) *linear_iters += st.iters;
      if (newton_iters) ++(*newton_iters);

      // Damped update with the reference's alpha-halving line search.
      k_apply_sat<<<blk, 256>>>(r.u, S.work, 1.0, r.bs, cells, phase, r.swi);
      NVRS_CHECK_CUDA(cudaGetLastError());
    }
    if (!ok) { if (residual) *residual = last_res; return false; }
  }
  if (residual) *residual = last_res;
  return true;
}

}  // namespace nvrs
