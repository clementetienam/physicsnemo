// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// field_model.cu -- turning a corner-point deck into a solvable system.
//
// The interesting part of this file is build_sparsity_field(), which builds the
// same arrays as build_sparsity() but from a connection list rather than from a
// six-face stencil.  Everything the solver needs downstream -- the coupled CSR,
// the pressure sub-block, the CPR column maps -- is derived from that list, so
// the linear algebra never learns that the grid was unstructured.
#include <set>
#include "field_model.hpp"

#include "common.hpp"
#include "deck.hpp"
#include "linear_solver.hpp"
#include "physics.hpp"
#include "halo.hpp"
#include "pvt.hpp"
#include "wellcell.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nvrs {
// Unit conversion and deck loading live in units_field.cpp.  They are pure
// host code -- grid, schedule, equilibration and the partition -- and every
// factor in them is silent when it is wrong, so they belong in a translation
// unit a laptop can compile and tests/test_field_units can exercise without
// nvcc or a GPU.

// ---------------------------------------------------------------------------
// The linear system, from a connection list
// ---------------------------------------------------------------------------
void build_sparsity_field(const Reservoir& r, DeviceLinearSystem& sys,
                          const FieldModel& fm) {
  const FieldPartition& P = fm.part;
  const int bs = r.bs;

  // Rows exist only for the cells this rank owns.  Ghosts are columns, not
  // rows: their values arrive through the halo and their equations belong to
  // whoever owns them.
  const int cells = P.n_owned;
  const int cells_ext = P.n_local_ext();

  sys.bs = bs;
  sys.cells = cells;
  sys.n = bs * cells;
  sys.cells_ext = cells_ext;
  sys.n_ext = bs * cells_ext;
  sys.cells_global = (int)P.n_global;
  // A CELL offset, not a row offset.  The pressure block has one row per cell
  // and indexes with this directly (hi = row0 + cells - 1); the coupled system
  // multiplies by bs itself.  Scaling it here would put every rank's rows in
  // the wrong place by a factor of three -- invisible on one rank, where it is
  // zero either way.
  sys.row0 = P.row0;

  // Adjacency of the owned cells, in local indices.  P.conns already holds
  // only the connections this rank assembles, each once, with endpoints
  // renumbered; a ghost endpoint has a local index at or beyond n_owned.
  std::vector<std::vector<std::pair<int, int>>> adj(cells);
  for (int a = 0; a < cells; ++a) adj[a].reserve(8);
  for (size_t k = 0; k < P.conns.size(); ++k) {
    const Connection& c = P.conns[k];
    if (c.a < cells) adj[c.a].emplace_back(c.b, (int)k);
    if (c.b < cells) adj[c.b].emplace_back(c.a, (int)k);
  }
  for (int a = 0; a < cells; ++a)
    std::sort(adj[a].begin(), adj[a].end());

  // Global row of a local cell, for the distributed solvers' column ids.
  // Owned cells are numbered in ascending active order from this rank's row0,
  // and a ghost's global row follows from its owner's offset the same way --
  // which every rank can evaluate, because every rank holds the partition.
  const CornerPointGrid& g = fm.grid;
  std::vector<long long> owner_row0(P.nranks + 1, 0);
  {
    std::vector<long long> cnt(P.nranks, 0);
    for (int a = 0; a < g.n_active(); ++a) ++cnt[P.owner_of_active[a]];
    for (int q = 0; q < P.nranks; ++q) owner_row0[q + 1] = owner_row0[q] + cnt[q];
  }
  std::vector<long long> rank_seen(P.nranks, 0);
  std::vector<long long> global_row(cells_ext, 0);
  {
    // The n-th active cell owned by rank q has global row owner_row0[q] + n.
    std::vector<long long> seen(P.nranks, 0);
    std::vector<long long> grow_of_active(g.n_active(), 0);
    for (int a = 0; a < g.n_active(); ++a) {
      const int q = P.owner_of_active[a];
      grow_of_active[a] = owner_row0[q] + seen[q];
      ++seen[q];
    }
    for (int l = 0; l < cells_ext; ++l)
      global_row[l] = grow_of_active[P.active_of_local[l]];
  }
  (void)rank_seen;

  // ---- CSR layouts ------------------------------------------------------
  const size_t nconn_local = P.conns.size();
  std::vector<int> h_rowptr(bs * cells + 1, 0), h_colind;
  std::vector<int> h_prowptr(cells + 1, 0), h_pcolind;
  h_colind.reserve((size_t)(cells + 2 * nconn_local) * bs * bs);
  h_pcolind.reserve(cells + 2 * nconn_local);

  std::vector<int> h_cptr(cells + 1, 0), h_cnbr, h_cbcol, h_cpcol;
  std::vector<Real> h_ctrans, h_cdz, h_cthreshold;
  std::vector<int> h_sbcol(cells, -1), h_spcol(cells, -1);
  h_cnbr.reserve(2 * nconn_local);
  h_ctrans.reserve(2 * nconn_local);
  h_cdz.reserve(2 * nconn_local);
  h_cbcol.reserve(2 * nconn_local);
  h_cpcol.reserve(2 * nconn_local);

  int max_deg = 0;
  std::vector<int> members;
  for (int c = 0; c < cells; ++c) {
    // Column set for this row: the cell itself and each of its neighbours,
    // in ascending order, as cuSPARSE requires.
    members.clear();
    members.push_back(c);
    for (const auto& nb : adj[c]) members.push_back(nb.first);
    std::sort(members.begin(), members.end());
    members.erase(std::unique(members.begin(), members.end()), members.end());

    const int ncols = (int)members.size();
    for (int a = 0; a < bs; ++a) {
      h_rowptr[bs * c + a + 1] = ncols * bs;
      for (int mc : members)
        for (int v = 0; v < bs; ++v) h_colind.push_back(bs * mc + v);
    }
    h_prowptr[c + 1] = ncols;
    // The pressure block's columns are GLOBAL row ids: the distributed AMG
    // resolves off-process columns itself, and giving it local indices would
    // silently build a matrix in which every rank means something different
    // by column 7.
    for (int mc : members) h_pcolind.push_back((int)global_row[mc]);

    auto rank_of = [&](int cell_id) {
      return (int)(std::lower_bound(members.begin(), members.end(), cell_id) -
                   members.begin());
    };
    h_sbcol[c] = rank_of(c) * bs;
    h_spcol[c] = rank_of(c);

    int deg = 0;
    for (const auto& nb : adj[c]) {
      const int nc = nb.first;
      const Connection& cn = P.conns[nb.second];
      h_cnbr.push_back(nc);
      h_ctrans.push_back(cn.trans);
      h_cthreshold.push_back(cn.threshold);
      // Depth increases downwards, and the flux wants depth(neighbour) minus
      // depth(this cell) so that gravity is (p_j - p_i) - g*rho*dz.  Depths
      // are indexed through the local-to-active map so a ghost is as valid
      // here as an owned cell.
      h_cdz.push_back(g.depth[P.active_of_local[nc]] -
                      g.depth[P.active_of_local[c]]);
      h_cbcol.push_back(rank_of(nc) * bs);
      h_cpcol.push_back(rank_of(nc));
      ++deg;
    }
    h_cptr[c + 1] = h_cptr[c] + deg;
    max_deg = std::max(max_deg, deg);
  }

  for (int i = 0; i < bs * cells; ++i) h_rowptr[i + 1] += h_rowptr[i];
  for (int i = 0; i < cells; ++i) h_prowptr[i + 1] += h_prowptr[i];
  sys.nnz = (int)h_colind.size();
  sys.pnnz = (int)h_pcolind.size();

  // ---- upload ------------------------------------------------------------
  auto up_i = [](int** dst, const std::vector<int>& h) {
    const size_t n = h.empty() ? 1 : h.size();
    NVRS_CHECK_CUDA(cudaMalloc(dst, sizeof(int) * n));
    if (!h.empty())
      NVRS_CHECK_CUDA(cudaMemcpy(*dst, h.data(), sizeof(int) * h.size(),
                                 cudaMemcpyHostToDevice));
  };
  auto up_r = [](Real** dst, const std::vector<Real>& h) {
    const size_t n = h.empty() ? 1 : h.size();
    NVRS_CHECK_CUDA(cudaMalloc(dst, sizeof(Real) * n));
    if (!h.empty())
      NVRS_CHECK_CUDA(cudaMemcpy(*dst, h.data(), sizeof(Real) * h.size(),
                                 cudaMemcpyHostToDevice));
  };

  up_i(&sys.rowptr, h_rowptr);
  up_i(&sys.colind, h_colind);
  up_i(&sys.prowptr, h_prowptr);
  up_i(&sys.pcolind, h_pcolind);

  Reservoir& rr = const_cast<Reservoir&>(r);
  up_i(&rr.conn_ptr, h_cptr);
  up_i(&rr.conn_nbr, h_cnbr);
  up_i(&rr.conn_bcol, h_cbcol);
  up_i(&rr.conn_pcol, h_cpcol);
  up_i(&rr.self_bcol, h_sbcol);
  up_i(&rr.self_pcol, h_spcol);
  up_r(&rr.conn_trans, h_ctrans);
  up_r(&rr.conn_dz, h_cdz);
  up_r(&rr.conn_threshold, h_cthreshold);
  rr.nconn = (int)h_cnbr.size();
  rr.max_degree = max_deg;
  // A grid that is not a six-face stencil cannot be assembled by the fixed-slot
  // kernels at all, so this is not a preference.
  rr.use_conn_list = 1;

  // The fixed-stencil arrays are left null.  Anything that still reads them --
  // the ZLINE second stage, which walks vertical couplings through nbr7 -- must
  // check for that rather than dereference them.
  rr.nbr = rr.bcol = rr.pcolpos = nullptr;
  rr.tface = nullptr;
  sys.nbr7 = sys.bcol7 = nullptr;

  // ---- values ------------------------------------------------------------
  NVRS_CHECK_CUDA(cudaMalloc(&sys.aval64, sizeof(Real) * sys.nnz));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.b, sizeof(Real) * sys.n));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.pval, sizeof(PrecReal) * sys.pnnz));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.dinv,
                             sizeof(PrecReal) * (long long)cells * bs * bs));
  NVRS_CHECK_CUDA(cudaMalloc(&sys.pweight,
                             sizeof(PrecReal) * (long long)cells * bs));
}

// ---------------------------------------------------------------------------
// The Reservoir
// ---------------------------------------------------------------------------
Reservoir make_reservoir_field(const Deck& d, FieldModel& fm) {
  to_field_units(fm);

  const CornerPointGrid& g = fm.grid;
  const FieldPartition& P = fm.part;
  const int cells = P.n_owned;
  const int cells_ext = P.n_local_ext();

  Reservoir r;
  // Fluid and solver settings come from the simulator's own deck keywords.
  // A corner-point deck states these in tables this reader does not
  // interpret, so they are taken from the run deck instead.
  r.model = d.model;
  r.ncomp = d.ncomp;
  r.bs = block_size(d.model, d.ncomp);
  r.kz_mult = d.kz_mult;
  r.uw = d.uw; r.uo = d.uo; r.ug = d.ug;
  r.bw = d.bw; r.bo = d.bo; r.bg = d.bg;
  r.pvt_const = d.pvt_const;
  r.rs = d.pvt_const ? d.rs_const : pvt::rs(d.p_bub, d.p_init);
  r.swi = d.swi; r.swr = d.swr; r.cfo = d.cfo;
  r.ct = d.ct;
  r.appleyard = d.appleyard;
  r.pv_switch = d.pv_switch;
  r.aim = d.aim; r.aim_cfl = d.aim_cfl;
  r.p_bub = d.p_bub; r.p_atm = d.p_atm;
  r.rhow = d.rhow; r.rhoo = d.rhoo; r.rhog = d.rhog;
  r.gravity = d.gravity;
  r.jacobian = d.jacobian;
  r.dp_max = d.dp_max; r.ds_max = d.ds_max;
  r.darcy_const = d.darcy_const;
  r.grav_grad = d.grav_grad;
  r.re_factor = d.re_factor;
  r.fd_eps = d.fd_eps;
  r.temperature = d.temperature;
  r.z_factor = d.z_factor;

  // The kernels use the grid only as a cell count, and index cells in this
  // rank's own ordering.  An unstructured model has no i,j,k extents to give,
  // so the count is presented as a single run of cells.
  r.grid = Grid::make(cells, 1, 1, 1.0, 1.0, 1.0, 0, 1);

  // grid above says "rank 0 of 1" on purpose; the collectives must not.
  r.mpi_rank = P.rank;
  r.mpi_nranks = P.nranks;

  // The unstructured halo replaces the structured one here.  It is built from
  // the partition, and on a single rank it has no peers and does nothing.
  const_cast<FieldModel&>(fm).halo.init(P, r.bs);
  r.halo_conn = &const_cast<FieldModel&>(fm).halo;

  // ---- per-cell arrays ---------------------------------------------------
  // Rock and geometry are sized for owned cells AND ghosts: the assembly reads
  // a neighbour's permeability and depth, and a neighbour may live on another
  // rank.  The state vectors are likewise extended, since that is what the
  // halo writes into.
  auto up_r = [](Real** dst, const std::vector<Real>& h) {
    NVRS_CHECK_CUDA(cudaMalloc(dst, sizeof(Real) * std::max<size_t>(h.size(), 1)));
    if (!h.empty())
      NVRS_CHECK_CUDA(cudaMemcpy(*dst, h.data(), sizeof(Real) * h.size(),
                                 cudaMemcpyHostToDevice));
  };
  // Gather a grid array onto the local numbering, ghosts included.
  auto local_of = [&](const std::vector<Real>& src) {
    std::vector<Real> out(cells_ext, 0.0);
    for (int l = 0; l < cells_ext; ++l) out[l] = src[P.active_of_local[l]];
    return out;
  };

  // Porosity carries net-to-gross, because the accumulation term wants the
  // volume that actually stores fluid and the flux already has NTG in its
  // transmissibility.
  std::vector<Real> poro(cells_ext);
  for (int l = 0; l < cells_ext; ++l) {
    const int a = P.active_of_local[l];
    poro[l] = g.poro[a] * g.ntg[a];
  }

  // The black-oil kernels form accumulation and AIM throughput denominators
  // as `cellvol * poro`.  For a field deck that product must be the FINAL
  // deck pore volume, including MULTPV (and any MINPV treatment), not merely
  // geometric bulk volume * PORO * NTG.  Norne uses MULTPV extensively; the
  // old upload silently omitted it from the reservoir equations even though
  // grid.porv, INIT output, and the material-balance audit all included it.
  // That made the converged equations describe a different reservoir from the
  // one being reported.
  //
  // Keep r.poro as physical PORO*NTG for output and diagnostics.  Carry the
  // remaining pore-volume multiplier in the effective accumulation volume so
  // their product is exactly grid.porv in every active cell.
  std::vector<Real> accum_volume(cells_ext, 0.0);
  for (int l = 0; l < cells_ext; ++l) {
    const int a = P.active_of_local[l];
    accum_volume[l] = poro[l] > 0 ? g.porv[a] / poro[l] : g.volume[a];
  }

  up_r(&r.perm, local_of(g.permx));
  up_r(&r.permz, local_of(g.permz));
  up_r(&r.poro, poro);
  up_r(&r.cellvol, accum_volume);
  up_r(&r.celldepth, local_of(g.depth));

  // RSVD, if the deck stated it.  Uploaded on the same local numbering as
  // everything else, so the kernel indexes it with the cell it is already
  // working on.  Left null when the deck has no RSVD, and the scalar rs_max
  // then applies exactly as before.
  if ((int)fm.rs_anchor.size() == g.n_active())
    up_r(&r.rs_max_cell, local_of(fm.rs_anchor));

  // Dissolved-gas composition history.  The ceiling starts at the anchor --
  // the gas the oil was given -- and drsdt_update() walks it after every
  // accepted step.  DRSDT needs it for the explicit dissolution-rate limit;
  // VAPPARS needs it because Rs is not a primary unknown in this compact
  // formulation.
  r.drsdt = fm.drsdt;
  // A VAPPARS run still needs an Rs composition history even though the
  // keyword cancels DRSDT: initial free-gas cells are saturated, after which
  // the cap may decrease on exsolution but cannot increase outside a coupled
  // component-conserving solve.
  if (r.drsdt >= 0 || (fm.vap1 > 0 && !r.pv_switch)) {
    std::vector<Real> cap0 =
        (int)fm.rs_anchor.size() == g.n_active()
            ? local_of(fm.rs_anchor)
            : std::vector<Real>((size_t)P.n_local_ext(),
                                (r.rs_max > 0) ? r.rs_max : r.rs);
    up_r(&r.rs_cap, cap0);
  }

  r.use_hysteresis = d.use_hysteresis;

  // ---- state -------------------------------------------------------------
  const size_t nu = (size_t)cells_ext * r.bs;
  NVRS_CHECK_CUDA(cudaMalloc(&r.u, sizeof(Real) * nu));
  NVRS_CHECK_CUDA(cudaMalloc(&r.u_old, sizeof(Real) * nu));

  std::vector<Real> u(nu);
  for (int l = 0; l < cells_ext; ++l) {
    const int a = P.active_of_local[l];
    if (fm.have_initial) {
      // Equilibrated: pressure and saturations vary with depth and region.
      u[r.bs * l + 0] = fm.initial.pressure[a];
      if (r.bs >= 2) u[r.bs * l + 1] = fm.initial.sw[a];
      if (r.bs >= 3) u[r.bs * l + 2] = fm.initial.sg[a];
    } else {
      u[r.bs * l + 0] = d.p_init;
      if (r.bs >= 2) u[r.bs * l + 1] = d.sw_init;
      if (r.bs >= 3) u[r.bs * l + 2] = d.sg_init;
    }
  }
  NVRS_CHECK_CUDA(cudaMemcpy(r.u, u.data(), sizeof(Real) * nu,
                             cudaMemcpyHostToDevice));
  NVRS_CHECK_CUDA(cudaMemcpy(r.u_old, u.data(), sizeof(Real) * nu,
                             cudaMemcpyHostToDevice));

  // Carlson history starts at the initial fluid state.  Delaying this until
  // the first accepted step loses the actual reversal point whenever gas
  // saturation falls or Sw+Sg rises during that step, and all subsequent
  // scanning curves then start from the wrong place.
  if (r.use_hysteresis) {
    std::vector<Real> sg_hy((size_t)cells_ext, (Real)0);
    std::vector<Real> sw_ow_hy((size_t)cells_ext, (Real)0);
    for (int l = 0; l < cells_ext; ++l) {
      const Real sw = u[(size_t)r.bs * l + 1];
      const Real sg = r.bs >= 3 ? u[(size_t)r.bs * l + 2] : (Real)0;
      sg_hy[(size_t)l] = sg;
      sw_ow_hy[(size_t)l] = sw + sg;
    }
    up_r(&r.sg_hy, sg_hy);
    up_r(&r.sw_ow_hy, sw_ow_hy);
  }

  // VAPPARS history starts at the initial oil saturation and advances only
  // after accepted steps.  This is OPM's maxOilSaturation state.
  r.vap1 = fm.vap1;
  r.vap2 = fm.vap2;
  if (r.vap1 > 0) {
    std::vector<Real> so_max((size_t)cells_ext, (Real)0);
    for (int l = 0; l < cells_ext; ++l) {
      const Real sw = u[(size_t)r.bs * l + 1];
      const Real sg = r.bs >= 3 ? u[(size_t)r.bs * l + 2] : (Real)0;
      so_max[(size_t)l] = std::max((Real)0, (Real)1 - sw - sg);
    }
    up_r(&r.so_max, so_max);
  }

  // ---- adaptive implicit flags ------------------------------------------
  // Sized for ghosts: the assembly reads a donor's AIM flag, and a donor can
  // be a cell another rank owns.
  NVRS_CHECK_CUDA(cudaMalloc(&r.aimflag, sizeof(Real) * cells_ext));
  NVRS_CHECK_CUDA(cudaMalloc(&r.aim_rate, sizeof(Real) * cells_ext));
  {
    const std::vector<Real> ones(cells_ext, 1.0);
    NVRS_CHECK_CUDA(cudaMemcpy(r.aimflag, ones.data(), sizeof(Real) * cells_ext,
                               cudaMemcpyHostToDevice));
    NVRS_CHECK_CUDA(cudaMemset(r.aim_rate, 0, sizeof(Real) * cells_ext));
  }

  // ---- wells -------------------------------------------------------------
  NVRS_CHECK_CUDA(cudaMalloc(&r.wellcells, sizeof(WellCell) * cells));
  NVRS_CHECK_CUDA(cudaMemset(r.wellcells, 0, sizeof(WellCell) * cells));
  r.wellcells_host = new WellCell[(size_t)cells]();
  r.well_schur_enabled = d.well_schur;
  if (r.well_schur_enabled) {
    r.well_schur_capacity = cells;
    NVRS_CHECK_CUDA(cudaMalloc(&r.well_schur,
                               sizeof(WellSchurEntry) * (size_t)cells));
  }
  if (fm.schedule.n_steps() > 0)
    apply_schedule_step(r, fm, 0, d.prod_bhp, nullptr);

  // Saturation tables come from the run deck, as the fluid properties do.
  auto upload_tab = [](const std::vector<std::array<Real, 3>>& tab,
                       const Real** x, const Real** ko, const Real** kwg,
                       int& n) {
    n = (int)tab.size();
    if (!n) return;
    std::vector<Real> X(n), KO(n), KWG(n);
    for (int i = 0; i < n; ++i) {
      X[i] = tab[i][0]; KO[i] = tab[i][1]; KWG[i] = tab[i][2];
    }
    Real *dx, *dko, *dkwg;
    NVRS_CHECK_CUDA(cudaMalloc(&dx, sizeof(Real) * n));
    NVRS_CHECK_CUDA(cudaMalloc(&dko, sizeof(Real) * n));
    NVRS_CHECK_CUDA(cudaMalloc(&dkwg, sizeof(Real) * n));
    NVRS_CHECK_CUDA(cudaMemcpy(dx, X.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    NVRS_CHECK_CUDA(cudaMemcpy(dko, KO.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    NVRS_CHECK_CUDA(cudaMemcpy(dkwg, KWG.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    *x = dx; *ko = dko; *kwg = dkwg;
  };
  if (!d.swof.empty())
    upload_tab(d.swof, &r.tables.sw, &r.tables.krow, &r.tables.krw, r.tables.nsw);
  if (!d.sgof.empty())
    upload_tab(d.sgof, &r.tables.sg, &r.tables.krog, &r.tables.krg, r.tables.nsg);
  // Capillary pressure, when the deck gave a fourth column with anything in
  // it.  All zeros is the same as absent and is treated as absent, so a deck
  // that pads its tables does not pay for a term that is identically zero.
  auto upload_pc = [&](const std::vector<Real>& col, const Real** dst) {
    if (col.empty()) return;
    bool any = false;
    for (Real v : col) if (v != (Real)0) { any = true; break; }
    if (!any) return;
    Real* dv = nullptr;
    NVRS_CHECK_CUDA(cudaMalloc(&dv, sizeof(Real) * col.size()));
    NVRS_CHECK_CUDA(cudaMemcpy(dv, col.data(), sizeof(Real) * col.size(),
                               cudaMemcpyHostToDevice));
    *dst = dv;
    r.tables.have_pc = 1;
  };
  if (!d.swof.empty() && d.swof_pcow.size() == d.swof.size())
    upload_pc(d.swof_pcow, &r.tables.pcow);
  if (!d.sgof.empty() && d.sgof_pcog.size() == d.sgof.size())
    upload_pc(d.sgof_pcog, &r.tables.pcog);

  // ---- ENDSCALE ----------------------------------------------------------
  //
  // The table above is one curve for the whole field.  Norne's is not: the
  // deck carries SWL, SWCR, SWU, SGL, SGCR, SGU, SOWCR, SOGCR and the maximum
  // relative permeabilities as grid arrays, so each cell has its own water
  // saturation at which water begins to flow and its own at which oil stops.
  //
  // That is what sets the timing of the water front, which is exactly where
  // NVRS has been lagging Flow -- both reach ~80 Mrb of water gained, and the
  // curves share their shape, but NVRS arrives late between roughly 500 and
  // 3000 days.  A single field-wide curve cannot reproduce a front whose
  // speed varies cell by cell.
  //
  // Read the TABLE's endpoints first: scaling is a map from the cell's onto
  // the table's, so the target has to be known before anything can be mapped.
  {
    std::vector<Real> hsw, hkrw, hkrow, hsg, hkrg, hkrog;
    for (const auto& row : d.swof) {
      hsw.push_back(row[0]); hkrow.push_back(row[1]); hkrw.push_back(row[2]);
    }
    for (const auto& row : d.sgof) {
      hsg.push_back(row[0]); hkrog.push_back(row[1]); hkrg.push_back(row[2]);
    }
    set_table_endpoints(r.tables, hsw.data(), hkrw.data(), hkrow.data(),
                        (int)hsw.size(), hsg.data(), hkrg.data(),
                        hkrog.data(), (int)hsg.size());

    // Then the per-cell arrays, each optional -- a deck may scale only some.
    // Uploaded on the LOCAL numbering with ghosts, like every other per-cell
    // array here, because relperm3 is handed a local cell index.
    int nscaled = 0;
    auto up_end = [&](const char* name, const Real** dst) {
      if (!g.has_array(name)) return;
      std::vector<Real> a = g.active_array(name);
      if ((int)a.size() != g.n_active()) return;
      // Partial I-endpoint grid edits leave untouched destination cells at a
      // numeric default. Preserve "missing" as NaN so imb_cell_end() can
      // inherit the imbibition-table endpoint without confusing it with
      // explicit zero.
      if (name[0] == 'I') {
        const std::vector<char> defined = g.active_array_defined(name);
        if (defined.size() == a.size())
          for (size_t q = 0; q < a.size(); ++q)
            if (!defined[q]) a[q] = std::numeric_limits<Real>::quiet_NaN();
      }
      Real* dv = nullptr;
      const std::vector<Real> loc = local_of(a);
      NVRS_CHECK_CUDA(cudaMalloc(&dv, sizeof(Real) * std::max<size_t>(loc.size(), 1)));
      NVRS_CHECK_CUDA(cudaMemcpy(dv, loc.data(), sizeof(Real) * loc.size(),
                                 cudaMemcpyHostToDevice));
      *dst = dv;
      ++nscaled;
      if (P.rank == 0) {
        Real lo = INFINITY, hi = -INFINITY;
        for (Real v : loc) if (std::isfinite((double)v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        std::printf("  [endscale] %-6s %.4f .. %.4f\n", name, (double)lo,
                    (double)hi);
      }
    };
    up_end("SWL", &r.tables.es.swl);
    up_end("SWCR", &r.tables.es.swcr);
    up_end("SWU", &r.tables.es.swu);
    up_end("SGL", &r.tables.es.sgl);
    up_end("SGCR", &r.tables.es.sgcr);
    up_end("SGU", &r.tables.es.sgu);
    up_end("SOWCR", &r.tables.es.sowcr);
    up_end("SOGCR", &r.tables.es.sogcr);
    up_end("KRW", &r.tables.es.krw);
    up_end("KRO", &r.tables.es.kro);
    up_end("KRG", &r.tables.es.krg);
    up_end("ISWL", &r.tables.es.iswl);
    up_end("ISWCR", &r.tables.es.iswcr);
    up_end("ISWU", &r.tables.es.iswu);
    up_end("ISGL", &r.tables.es.isgl);
    up_end("ISGCR", &r.tables.es.isgcr);
    up_end("ISGU", &r.tables.es.isgu);
    up_end("ISOWCR", &r.tables.es.isowcr);
    up_end("ISOGCR", &r.tables.es.isogcr);
    up_end("IKRW", &r.tables.es.ikrw);
    up_end("IKRO", &r.tables.es.ikro);
    up_end("IKRG", &r.tables.es.ikrg);
    // Reading the arrays and USING them are separate decisions.  On Norne the
    // same arrays also feed equilibration -- equil.cpp has always asked for
    // SWL and only now gets it -- so switching both on in one run leaves no
    // way to say which one moved the answer.  USEENDSCALE gates the curves;
    // connate water reaches the initial state either way.
    // The global SWI/SWR scalars have to agree with the arrays, or they
    // fight them.
    //
    // physics_host.cu clamps every Newton update with
    // Sw = max(Sw, swi) and runs the Appleyard chop against swc = swi and
    // smax = 1 - swr.  Those are the deck's scalars, one pair for the whole
    // field.  With ENDSCALE each cell has its own connate water and maximum,
    // and Norne's deck says SWI 0.10 while SWATINIT initialises cells as low
    // as 0.08: the clamp then forces those cells back up every iteration
    // while the equations pull them down, and Newton freezes with |R| fixed
    // to five figures and |du| still non-zero.  That is what step 1 did.
    //
    // So take them from the arrays rather than from the deck: the connate
    // water is the smallest SWL in the field and the maximum water is the
    // largest SWU, which for Norne is 1.0 and leaves the aquifer free to sit
    // at Sw = 1.
    if (nscaled > 0 && d.use_endscale) {
      Real lo = r.swi, hi = (Real)1 - r.swr;
      if (g.has_array("SWL")) {
        const std::vector<Real> a = g.active_array("SWL");
        for (Real v : a) if (std::isfinite((double)v)) lo = std::min(lo, v);
      }
      if (g.has_array("SWU")) {
        const std::vector<Real> a = g.active_array("SWU");
        for (Real v : a) if (std::isfinite((double)v)) hi = std::max(hi, v);
      }
      if (lo >= 0 && lo < hi && hi <= 1) {
        const Real old_swi = r.swi, old_swr = r.swr;
        r.swi = lo;
        r.swr = (Real)1 - hi;
        if (P.rank == 0 && (old_swi != r.swi || old_swr != r.swr))
          std::printf("  [endscale] SWI/SWR taken from the arrays: "
                      "%.4f/%.4f (deck said %.4f/%.4f) -- the Newton clamp "
                      "and the Appleyard chop use these\n",
                      (double)r.swi, (double)r.swr, (double)old_swi,
                      (double)old_swr);
      }
    }
    // Two things a deck can ask for that we do not model.  Both would
    // otherwise pass without a word, which is the worst way for a feature to
    // be missing: the run finishes, the numbers look plausible, and nothing
    // says which physics was dropped.
    //
    // SATNUM selects one saturation table per cell.  PvtTables keeps the
    // first SWOF/SGOF block and discards the rest, so a deck whose SATNUM
    // varies spatially silently runs region 1 everywhere.
    //
    // IMBNUM selects an imbibition table for hysteresis.  We read the
    // drainage curves and ignore it.  Water and gas relative permeability on
    // an imbibition path is what sets water cut and GOR, so a deck that needs
    // hysteresis disagrees on exactly the well quantities that matter and
    // gives no hint why.
    //
    // Norne trips neither: SATNUM is 1 in every cell, and its IMBNUM = 2
    // table is numerically identical to the drainage one in all four columns.
    if (P.rank == 0) {
      auto distinct = [&](const char* name, int* lo, int* hi) -> int {
        if (!g.has_array(name)) return 0;
        const std::vector<Real> a = g.active_array(name);
        if ((int)a.size() != g.n_active()) return 0;
        int mn = 1 << 30, mx = -(1 << 30);
        std::set<int> seen;
        for (Real v : a) {
          const int k = (int)(v + 0.5);
          seen.insert(k);
          if (k < mn) mn = k;
          if (k > mx) mx = k;
        }
        if (lo) *lo = mn;
        if (hi) *hi = mx;
        return (int)seen.size();
      };

      int slo = 0, shi = 0;
      const int nsat = distinct("SATNUM", &slo, &shi);
      if (nsat > 1) {
        std::printf("  [satfunc] WARNING: SATNUM takes %d distinct values "
                    "(%d..%d) but only region 1's SWOF/SGOF are read. Every "
                    "cell is running region 1's curves; the other %d region(s) "
                    "were discarded. Results on this deck are not comparable "
                    "with a simulator that honours SATNUM.\n",
                    nsat, slo, shi, nsat - 1);
      }

      int ilo = 0, ihi = 0;
      const int nimb = distinct("IMBNUM", &ilo, &ihi);
      if (!r.use_hysteresis && nimb > 0 &&
          !(nimb == 1 && nsat == 1 && ilo == slo)) {
        std::printf("  [satfunc] WARNING: the deck carries IMBNUM %d..%d but "
                    "the selected hysteresis model is unsupported: the "
                    "drainage curves are used throughout and the imbibition "
                    "table is not read. Where the two tables differ this moves "
                    "water cut and GOR.\n", ilo, ihi);
      }
    }

    r.tables.es.on = (nscaled > 0 && d.use_endscale) ? 1 : 0;
    r.tables.es.three_point = g.scalecrs ? 1 : 0;
    r.tables.hyst_on = r.use_hysteresis;
    r.tables.sg_hy = r.sg_hy;
    r.tables.sw_ow_hy = r.sw_ow_hy;

    // SWATINIT also scales maximum Pcow independently in every cell. Map the
    // stated saturation onto SWOF, evaluate the current Pc, then choose the
    // multiplier that makes it equal the hydrostatic Po-Pw.
    if (r.tables.pcow && r.tables.nsw > 1 && fm.have_initial &&
        fm.initial.n_swatinit > 0 &&
        fm.initial.pcow_target.size() == (size_t)g.n_active()) {
      std::vector<Real> scale((size_t)g.n_active(), (Real)1);
      const std::vector<Real> swl = g.has_array("SWL")
                                        ? g.active_array("SWL")
                                        : std::vector<Real>();
      const std::vector<Real> swu = g.has_array("SWU")
                                        ? g.active_array("SWU")
                                        : std::vector<Real>();
      Real slo = INFINITY, shi = -INFINITY;
      int nscaled_pc = 0;
      for (int a = 0; a < g.n_active(); ++a) {
        const Real target = fm.initial.pcow_target[(size_t)a];
        if (!(target > (Real)0)) continue;
        const Real lo = swl.empty() ? r.tables.sw_lo : swl[(size_t)a];
        const Real hi = swu.empty() ? r.tables.sw_up : swu[(size_t)a];
        Real jac = 1;
        const Real st = r.tables.es.on
                            ? map_sat(fm.initial.sw[(size_t)a], lo, hi,
                                      r.tables.sw_lo, r.tables.sw_up, &jac)
                            : fm.initial.sw[(size_t)a];
        Real base = 0, deriv = 0;
        interp1(hsw.data(), d.swof_pcow.data(), (int)hsw.size(), st,
                base, deriv);
        if (!(base > (Real)1e-12)) continue;
        scale[(size_t)a] = target / base;
        slo = std::min(slo, scale[(size_t)a]);
        shi = std::max(shi, scale[(size_t)a]);
        ++nscaled_pc;
      }
      Real* dscale = nullptr;
      up_r(&dscale, local_of(scale));
      r.tables.pcow_scale = dscale;
      if (P.rank == 0)
        std::printf("  [swatinit] Pcow scaled in %d cells: multiplier "
                    "%.4g .. %.4g\n", nscaled_pc, (double)slo,
                    (double)shi);
    }
    if (P.rank == 0) {
      if (nscaled && !d.use_endscale)
        std::printf("  [endscale] %d array(s) read but NOT applied to the "
                    "curves (USEENDSCALE 0); they still set connate water in "
                    "the initial state\n", nscaled);
      if (nscaled && d.use_endscale)
        std::printf("  [endscale] SCALECRS %s -> %s-point scaling\n",
                    g.scalecrs ? "YES" : "NO/absent",
                    g.scalecrs ? "three" : "two");
      if (nscaled && d.use_endscale)
        std::printf("  [endscale] %d array(s) active; table endpoints "
                    "Swl %.4f Swcr %.4f Swu %.4f Sowcr %.4f | Sgl %.4f "
                    "Sgcr %.4f Sgu %.4f Sogcr %.4f\n",
                    nscaled, (double)r.tables.sw_lo, (double)r.tables.sw_cr,
                    (double)r.tables.sw_up, (double)r.tables.so_wcr,
                    (double)r.tables.sg_lo, (double)r.tables.sg_cr,
                    (double)r.tables.sg_up, (double)r.tables.so_gcr);
      else
        std::printf("  [endscale] no endpoint arrays in the deck; the "
                    "saturation tables apply unscaled to every cell\n");
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// Time-varying well control
// ---------------------------------------------------------------------------
int schedule_step_at(const FieldModel& fm, double day) {
  const int n = fm.schedule.n_steps();
  if (n == 0) return -1;
  const double t0 = fm.schedule.start_day();
  int step = 0;
  for (int i = 0; i < n; ++i)
    if (fm.schedule.step(i).day - t0 <= day + 1e-9) step = i;
  return step;
}

void apply_schedule_step(Reservoir& r, const FieldModel& fm, int step) {
  apply_schedule_step(r, fm, step, 0.0, nullptr);
}

void apply_schedule_step(Reservoir& r, const FieldModel& fm, int step,
                         Real default_prod_bhp, int* n_shut_no_bhp,
                         Real default_inj_bhp, int* n_inj_no_bhp) {
  if (step < 0 || step >= fm.schedule.n_steps()) return;
  const ScheduleStep& s = fm.schedule.step(step);

  // Rebuild the per-cell well array from scratch each time.  A well that shut
  // between steps must leave no trace, and clearing is the only way to be sure
  // of that without tracking what the previous step wrote.
  const int cells = fm.part.n_owned;
  auto* wc = static_cast<WellCell*>(r.wellcells_host);
  if (!wc) {
    wc = new WellCell[(size_t)cells];
    r.wellcells_host = wc;
  }
  for (int a = 0; a < cells; ++a) wc[a] = WellCell{};

  // GCONINJE: hold each group's injection to what the group is allowed.
  //
  // Without this every injector delivers the rate its own WCONINJE states and
  // the group ceiling never applies.  On Norne the individual rates sum well
  // above it, and the field took in 1147 M reservoir bbl of water against
  // Flow's 648 on the same prescribed schedule -- 1.77x, which then came back
  // out of the producers as 733.6 against 144.6.
  //
  // Computed per (group, phase) because a group injects water and gas under
  // separate targets and a well is bound only by the record for its own
  // phase.  Only RATE is honoured here.  RESV needs a reservoir-volume
  // conversion, while REIN and VREP set the target as a fraction of what the
  // group produced and need production history.  A group under any of those
  // parsed-but-unenforced modes therefore keeps its wells' own rates.
  std::map<std::string, Real> grp_scale;   // "GROUP|phase" -> multiplier
  {
    std::map<std::string, Real> asked;     // sum of the wells' stated rates
    for (const Well& w : s.wells) {
      if (!w.state.open || w.state.type != WellType::Injector) continue;
      if (!std::isfinite((double)w.state.surface_rate) ||
          w.state.surface_rate <= 0)
        continue;
      const int ph = (w.state.inj_phase == InjectedPhase::Water) ? 1
                     : (w.state.inj_phase == InjectedPhase::Gas) ? 2 : 3;
      const GroupInject* g = s.group_inject(w.group, ph);
      if (!g || g->ctl != GroupInject::Ctl::Rate) continue;
      asked[g->name + "|" + std::to_string(ph)] += w.state.surface_rate;
    }
    for (const auto& kv : asked) {
      auto it = s.ginject.find(kv.first);
      if (it == s.ginject.end() || !(it->second.target > 0)) continue;
      // A ceiling, not a target: a group already injecting less than it is
      // allowed is left alone.
      if (kv.second > it->second.target)
        grp_scale[kv.first] = it->second.target / kv.second;
    }
    if (fm.part.rank == 0 && !grp_scale.empty()) {
      static int said = 0;
      if (!said) {
        said = 1;
        std::printf("  [ginject] %d group/phase ceiling(s) binding; the "
                    "wells' stated rates are scaled to the group's\n",
                    (int)grp_scale.size());
      }
    }
  }

  for (const Well& w : s.wells) {
    if (!w.state.open || w.state.mode == CtrlMode::Shut) continue;

    // Total well index over the live completions, used to share a rate target
    // among them in proportion to how well each connects.
    // The well index total is over ALL live completions, not just the ones on
    // this rank: a rate target is shared across the whole well, so a rank that
    // holds half of it must still divide by the whole.
    Real wi_sum = 0;
    for (const Completion& c : w.completions)
      if (c.cell >= 0 && c.open) wi_sum += c.wi;
    if (wi_sum <= 0) continue;

    const bool injector = (w.state.type == WellType::Injector);

    // BHP is defined at the WELSPECS datum, not independently at every
    // completion.  OPM defaults an absent datum to the top live connection.
    // Producers receive this correction in apply_producer_rate_targets(); an
    // injector must receive the same correction before its pressure ceiling
    // is tested in the assembly.  Applying the datum pressure unchanged at a
    // deeper connection makes the ceiling too low and silently discards part
    // of a rate-controlled injection target.
    Real ref_depth = w.ref_depth;
    if (!(ref_depth >= 0) || !std::isfinite((double)ref_depth)) {
      ref_depth = std::numeric_limits<Real>::infinity();
      for (const Completion& c : w.completions)
        if (c.cell >= 0 && c.open && c.cell < (int)fm.grid.depth.size())
          ref_depth = std::min(ref_depth, fm.grid.depth[c.cell]);
      if (!std::isfinite((double)ref_depth)) ref_depth = 0;
    }

    // Density of the injected phase at reservoir conditions.
    //
    // r.bo, r.bw and r.bg are FORMATION VOLUME FACTORS -- reservoir volume per
    // surface volume -- not their inverses.  Everything that consumes them says
    // so by inverting first: the assembly takes `bo = 1.0 / P.bo` before it
    // multiplies a mobility, and const_pvt_cell() is handed B and stores 1/B.
    // A surface density becomes a reservoir one by DIVIDING by B, because the
    // same mass occupies B reservoir volumes for every surface one.
    //
    // This block multiplied instead, on a comment that had the convention
    // backwards.  For water that is 2% -- 1.033/1.0113 read as 1.033*1.0113 --
    // and easy to miss.  For gas it is not: Norne's B_g is 8.12e-4 rb/scf, so
    // the reservoir gas density came out as 1.2e-7 against a true 0.186, and
    // the completion pressure correction on every gas injector was zero.  That
    // correction is what puts a deep completion's share of the bottom-hole
    // ceiling above the datum's; without it C-1H, C-3H, C-4H and C-4AH -- the
    // wells that take Norne's 294 G scf of reinjected gas -- test their
    // injectivity against a ceiling that is too low at depth and inject less
    // than the schedule asks for.
    //
    // The same expression, with 1/B, is what well_report.cu uses for the
    // producer connection correction and what the assembly uses for the flux
    // gravity term; gas additionally converts a reservoir barrel to ft^3.
    Real inj_rho = 0;
    if (injector) {
      if (w.state.inj_phase == InjectedPhase::Water)
        inj_rho = (r.bw > 0) ? r.rhow / r.bw : r.rhow;
      else if (w.state.inj_phase == InjectedPhase::Oil)
        inj_rho = (r.bo > 0) ? r.rhoo / r.bo : r.rhoo;
      else
        inj_rho = (r.bg > 0)
                      ? r.rhog / (r.bg * (Real)units::kFt3PerBbl)
                      : r.rhog / (Real)units::kFt3PerBbl;
    }

    // A producer needs a flowing bottom-hole pressure, and a history-matched
    // deck usually does not give one: WCONHIST states what the well produced
    // and defaults the pressure.  A defaulted value must not be taken as zero
    // -- that turns the well into a sink pulling the entire reservoir pressure
    // across every completion, and no timestep converges against it.  Where
    // the deck is silent and no default was supplied, the well stays shut and
    // the caller is told how many.
    Real pwf = 0;
    if (!injector) {
      // OPM gives a WCONHIST producer no bottom-hole limit unless it is under
      // BHP control (WellProductionProperties::handleWCONHIST), because the
      // deck is recording what the well DID.  Matching that here: a history
      // well starts from the deck's pressure if it has one, and otherwise
      // from our default, but apply_producer_rate_targets() is then free to
      // pull it below either -- which is where the rate target actually gets
      // met.
      const bool history_free =
          (w.state.historical && w.state.mode != CtrlMode::BHP);
      const Real stated = history_free ? (Real)0 : w.state.bhp_limit;
      if (stated > 0 && std::isfinite((double)stated)) pwf = stated;
      else if (default_prod_bhp > 0) pwf = default_prod_bhp;
      else {
        if (n_shut_no_bhp) ++(*n_shut_no_bhp);
        continue;
      }
    }

    for (const Completion& c : w.completions) {
      if (c.cell < 0 || !c.open) continue;
      const int loc = fm.part.local_of_active[c.cell];
      if (loc < 0 || loc >= cells) continue;   // another rank owns this cell
      WellCell& cell = wc[loc];
      cell.wi = c.wi;
      if (injector) {
        cell.is_inj = 1;
        // The bottom-hole limit, which for an injector is an UPPER bound and
        // was previously read, converted, and then dropped -- only the
        // producer branch above ever looked at it.  A rate-controlled injector
        // with no pressure feedback is a source that cannot be argued with:
        // it delivers its full rate whatever the cell pressure does, so the
        // pressure simply runs away and no timestep is small enough to help.
        // Norne states 600 bar for C-4H, and the kernel uses it to cap the
        // rate at what the formation will actually take.
        // A defaulted ceiling falls back to INJBHP.  Left at zero the
        // kernel's injectivity cap never engages and the well is a source
        // the reservoir cannot argue with.
        Real datum_bhp = 0;
        if (std::isfinite((double)w.state.bhp_limit) &&
            w.state.bhp_limit > 0) {
          datum_bhp = w.state.bhp_limit;
        } else {
          datum_bhp = default_inj_bhp;
          if (n_inj_no_bhp && default_inj_bhp <= 0) ++(*n_inj_no_bhp);
        }
        cell.bhp = datum_bhp;
        cell.pwf = datum_bhp;
        if (datum_bhp > 0 && r.gravity && inj_rho > 0 &&
            c.cell < (int)fm.grid.depth.size())
          cell.pwf += r.grav_grad * inj_rho *
                      (fm.grid.depth[c.cell] - ref_depth);
        // Surface rate, shared between the well's completions in proportion to
        // how well each connects, and placed in the phase the deck names.
        // WEFAC applies to an injector exactly as it does to a producer:
        // the stated rate is what goes in WHILE the well is on, and the
        // efficiency is the share of the interval it is on for.  Without it
        // the field receives more water and gas than the deck asks for, which
        // showed up as NVRS's implied injection running well above Flow's on
        // an identical schedule.
        const Real eff = (std::isfinite((double)w.state.efficiency) &&
                          w.state.efficiency > 0 && w.state.efficiency <= 1)
                             ? w.state.efficiency
                             : (Real)1;
        // The group's share, where a ceiling binds.
        Real gs = (Real)1;
        {
          const int ph = (w.state.inj_phase == InjectedPhase::Water) ? 1
                         : (w.state.inj_phase == InjectedPhase::Gas) ? 2 : 3;
          const GroupInject* g = s.group_inject(w.group, ph);
          if (g) {
            auto it = grp_scale.find(g->name + "|" + std::to_string(ph));
            if (it != grp_scale.end()) gs = it->second;
          }
        }
        const Real q = std::isfinite((double)w.state.surface_rate)
                           ? w.state.surface_rate * eff * gs *
                                 (c.wi / wi_sum)
                           : 0.0;
        switch (w.state.inj_phase) {
          case InjectedPhase::Water: cell.qw_inj = q; break;
          case InjectedPhase::Oil:   cell.qo_inj = q; break;
          case InjectedPhase::Gas:   cell.qg_inj = q; break;
        }
      } else {
        cell.is_prod = 1;
        cell.pwf = pwf;
        cell.bhp = pwf;
      }
    }
  }

  NVRS_CHECK_CUDA(cudaMemcpy(r.wellcells, wc,
                             sizeof(WellCell) * cells, cudaMemcpyHostToDevice));
}

}  // namespace nvrs
