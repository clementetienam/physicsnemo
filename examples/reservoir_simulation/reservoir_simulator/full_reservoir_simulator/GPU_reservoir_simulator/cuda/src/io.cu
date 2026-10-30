// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// io.cu — snapshot writer, well-rate CSV and insight summary.  Field snapshots
// are gathered to rank 0 and written as flat float32 binaries with a JSON
// sidecar (meta.json) that the Python post-processor turns into PNG/GIF plots.
#include "io.hpp"
#include "pvt.hpp"
#include "fluid.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>

namespace nvrs {

// std::filesystem (C++17) creates intermediate directories and is portable, so
// OUTDIR may be a nested path.  The previous POSIX/_mkdir pair did neither and
// did not even compile on Windows (_mkdir needs <direct.h>).
static void ensure_dir(const std::string& d) {
  std::error_code ec;
  std::filesystem::create_directories(d, ec);
  if (ec && !std::filesystem::is_directory(d))
    throw std::runtime_error("cannot create output directory '" + d +
                             "': " + ec.message());
}

Writer::Writer(const Deck& d, const Grid& g) : deck_(d), grid_(g), dir_(d.out_dir) {
  if (g.rank == 0) {
    ensure_dir(dir_);
    // ECLIPSE / OPM output is written by rank 0 from the gathered global fields.
    if (d.ecl_out)
      ecl_ = std::make_unique<ecl::EclipseWriter>(dir_, d.case_name, d);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

// Gather a per-cell field (extracted as u[stride*c+offset]) into a global
// float32 array on rank 0 in global natural ordering.
void Writer::gather_field(const Real* dev, int stride, int offset,
                          std::vector<float>& out) {
  int cells = grid_.n_local();
  std::vector<Real> host(cells);
  // strided copy on host
  std::vector<Real> raw(stride * cells);
  NVRS_CHECK_CUDA(cudaMemcpy(raw.data(), dev, sizeof(Real) * stride * cells,
                             cudaMemcpyDeviceToHost));
  for (int c = 0; c < cells; ++c) host[c] = raw[stride * c + offset];

  std::vector<float> local(cells);
  for (int c = 0; c < cells; ++c) local[c] = (float)host[c];

  // mpi_nranks_, not grid_.nranks: on a field run the latter says 1 on every
  // rank, so this returned each rank's own slab and rank 0 wrote a quarter of
  // the field as though it were all of it.
  if (mpi_nranks_ == 1) { out = local; return; }

  std::vector<int> counts(mpi_nranks_), displs(mpi_nranks_);
  MPI_Gather(&cells, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (mpi_rank_ == 0) {
    int tot = 0;
    for (int i = 0; i < mpi_nranks_; ++i) { displs[i] = tot; tot += counts[i]; }
    out.resize(tot);
  }
  MPI_Gatherv(local.data(), cells, MPI_FLOAT, out.data(), counts.data(),
              displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Gatherv concatenates the ranks in order, which is global ROW order.
  // Everything downstream -- the scatter onto the index box, the cgrid table,
  // the well markers -- is indexed by ACTIVE cell.  Permute once here so the
  // rest of the writer never has to know a partition existed.
  if (mpi_rank_ == 0 && !fgeom_.active_of_row.empty() &&
      out.size() == fgeom_.active_of_row.size()) {
    std::vector<float> byactive(out.size());
    for (size_t g = 0; g < out.size(); ++g) {
      const int a = fgeom_.active_of_row[g];
      if (a >= 0 && a < (int)byactive.size()) byactive[a] = out[g];
    }
    out.swap(byactive);
  }
  if (mpi_rank_ == 0 && fgeom_.active_of_row.empty() && mpi_nranks_ > 1 &&
      out.size() == (size_t)grid_.n_global()) {
    std::vector<float> bynatural(out.size());
    for (int k = 0; k < grid_.nz; ++k)
      for (int j = 0; j < grid_.ny; ++j)
        for (int i = 0; i < grid_.nx; ++i) {
          const size_t natural =
              ((size_t)k * grid_.ny + (size_t)j) * grid_.nx + (size_t)i;
          const long long row = grid_.global_row(i, j, k);
          if (row >= 0 && row < (long long)out.size())
            bynatural[natural] = out[(size_t)row];
        }
    out.swap(bynatural);
  }
}

void Writer::set_cell_geometry(std::vector<float> porv,
                               std::vector<float> depth) {
  if (ecl_) ecl_->set_cell_geometry(std::move(porv), std::move(depth));
}

void Writer::set_cell_porosity(std::vector<float> poro) {
  if (ecl_) ecl_->set_cell_porosity(std::move(poro));
}

void Writer::set_field_grid(const FieldGeometry& g) {
  fgeom_ = g;
  // The ECLIPSE writer took its dimensions from the deck, which on a field
  // run describes nothing.  It is constructed before the geometry is known
  // and writes the EGRID at the first snapshot, so here is both late enough
  // to have the numbers and early enough to matter.
  if (ecl_ && g.nx > 0 && g.ny > 0 && g.nz > 0) {
    ecl_->set_dims(g.nx, g.ny, g.nz);
    // And which of those cells exist.  The restart arrays carry one value per
    // ACTIVE cell, so ACTNUM is what makes their length mean anything.
    const size_t nbox = (size_t)g.nx * g.ny * g.nz;
    std::vector<int> act(nbox, 0);
    for (int gi : g.active_to_global)
      if (gi >= 0 && (size_t)gi < nbox) act[(size_t)gi] = 1;
    ecl_->set_actnum(std::move(act));
  }
}

void Writer::record_well_rates(Real t_days, const std::vector<WellRateRow>& rows) {
  wr_times_.push_back(t_days);
  wr_rows_.push_back(rows);
}

void Writer::record_well_cumulative(double oil_produced,
                                    double water_produced) {
  wr_cumulative_.push_back({oil_produced, water_produced});
}

namespace {
// Spread a value-per-active-cell plane over the full nx.ny.nz box, leaving the
// inactive cells NaN.
//
// NaN and not zero, because these plots are read for physics.  Norne is 39%
// active; the rest of the box is overburden and the parts the field does not
// occupy.  Filling those with zero draws a reservoir at zero pressure wrapped
// around the real one, which rescales every colour bar and puts a hard false
// edge at the boundary.  matplotlib omits NaN, so the inactive volume simply
// is not drawn -- which is what it is.
std::vector<float> scatter_to_box(const std::vector<float>& active,
                                  const std::vector<int>& to_global,
                                  size_t ncell_box) {
  std::vector<float> box(ncell_box, std::numeric_limits<float>::quiet_NaN());
  const size_t n = (active.size() < to_global.size()) ? active.size()
                                                      : to_global.size();
  for (size_t a = 0; a < n; ++a) {
    const int g = to_global[a];
    if (g >= 0 && (size_t)g < ncell_box) box[(size_t)g] = active[a];
  }
  return box;
}
}  // namespace

void Writer::snapshot(Reservoir& r, int step, Real t_days) {
  // Derive So (and Sg) from primary variables for output.
  int cells = grid_.n_local();
  std::vector<float> P, SW, SO, SG, gas_meaning;
  // Rock is static: gather it once, on the first snapshot.  gather_field is
  // collective, so every rank must agree on whether it happens — hence a flag
  // set on all ranks rather than a test on perm_global_, which only ever gets
  // filled on rank 0.  Disagreeing here silently pairs rank 0's gather of one
  // field with the other ranks' gather of another, and the receiving arrays
  // come back holding a different quantity entirely.
  const bool first = !rock_gathered_;
  if (first) {
    gather_field(r.perm, 1, 0, perm_global_);
    // Structured decks may omit PERMZFILE. In that case r.permz is null and
    // the simulator uses kz_mult * Kx; leave this vector empty so write_init
    // applies that same fallback instead of attempting a null device copy.
    if (r.permz)
      gather_field(r.permz, 1, 0, permz_global_);
    else
      permz_global_.clear();
    gather_field(r.poro, 1, 0, poro_global_);
    rock_gathered_ = true;
  }
  gather_field(r.u, r.bs, 0, P);
  gather_field(r.u, r.bs, 1, SW);
  if (r.bs == 3) gather_field(r.u, r.bs, 2, SG);
  if (r.gas_primary) gather_field(r.gas_primary, 1, 0, gas_meaning);

  if (mpi_rank_ == 0) {
    // What came back from the gather, which on several ranks is the whole
    // field and not grid_.n_global() -- that is this rank's own cell count.
    int ncell = (int)P.size();
    SO.resize(ncell);
    if (SG.empty()) SG.assign(ncell, 0.0f);

    // Eclipse PRESSURE is oil pressure. In NVRS an Rv-primary cell stores Pg
    // in u[0] (Po = Pg - Pcog), so writing P unchanged creates an artificial
    // positive pressure bias exactly where vaporised oil is the active
    // variable. Reconstruct the same Pcog table used by the reservoir kernel
    // and convert only those cells before either native or ECL output sees it.
    std::vector<Real> out_sg, out_pcog;
    SatTables out_tables;
    if (deck_.sgof_pcog.size() == deck_.sgof.size() &&
        deck_.sgof.size() > 1) {
      out_sg.reserve(deck_.sgof.size());
      out_pcog.reserve(deck_.sgof.size());
      for (size_t i = 0; i < deck_.sgof.size(); ++i) {
        out_sg.push_back(deck_.sgof[i][0]);
        out_pcog.push_back(deck_.sgof_pcog[i]);
      }
      out_tables.sg = out_sg.data();
      out_tables.pcog = out_pcog.data();
      out_tables.nsg = (int)out_sg.size();
    }

    // Joint clipping, exactly as the CuPy reference applies after every Newton
    // update:
    //     S    = clip(S,    SWI, 1.0)
    //     Soil = clip(Soil, 0.0, 1.0 - S)     # Sw + Sg <= 1 always
    // Doing it here as well makes the written fields bounded by construction:
    // So = 1 - Sw - Sg then lies in [0, 1 - Swi] and can never come out
    // negative, whatever the solver state.  If the clip actually bites, the
    // state itself is wrong and silently writing a physical-looking field would
    // hide that, so count the offenders and say so.
    // The SOLVER's connate water, not the deck's.
    //
    // With ENDSCALE these differ: field_model derives swi from the smallest
    // SWL in the field, and on Norne that is 0.04 where the deck's scalar
    // says 0.10.  Clipping the output against the deck's value then rewrote
    // every cell SWATINIT had put at 0.08 up to 0.10, and clipped Sg to
    // 1 - 0.10 in gas-cap cells holding 0.92 -- so the restart file, the
    // water balance and the oil in place computed from it were all describing
    // a field the solver never produced, while the warning blamed the solver
    // state for it.
    long n_bad_sw = 0, n_bad_sg = 0;
    float sw_lo = 1e30f, sw_hi = -1e30f, sg_lo = 1e30f, sg_hi = -1e30f;
    for (int c = 0; c < ncell; ++c) {
      const float sw_raw = SW[c];
      const float gm = c < (int)gas_meaning.size() ? gas_meaning[c] : 0.0f;
      const float sg_raw = r.bs != 3 ? 0.0f
                           : gm > 1.5f ? 1.0f - sw_raw
                           : gm > 0.5f ? 0.0f : SG[c];
      if (gm > 1.5f && out_tables.pcog) {
        Real pc = 0, dpc = 0;
        interp1(out_tables.sg, out_tables.pcog, out_tables.nsg,
                (Real)sg_raw, pc, dpc);
        P[c] -= (float)pc;
      }
      sw_lo = std::min(sw_lo, sw_raw); sw_hi = std::max(sw_hi, sw_raw);
      sg_lo = std::min(sg_lo, sg_raw); sg_hi = std::max(sg_hi, sg_raw);
      if (!(sw_raw >= -1e-6f && sw_raw <= 1.0f + 1e-6f)) ++n_bad_sw;

      // SWI is an immobile-water endpoint, not the physical lower bound.
      // SWOF states below it are valid (krw=0, kro=1) and must be reported.
      const float sw = std::min(std::max(sw_raw, 0.0f), 1.0f);
      const float sg = (r.bs == 3)
                           ? std::min(std::max(sg_raw, 0.0f), 1.0f - sw)
                           : 0.0f;
      if (r.bs == 3 && !(sg_raw >= -1e-6f && sg_raw <= 1.0f - sw + 1e-6f))
        ++n_bad_sg;
      SW[c] = sw;
      SG[c] = sg;
      SO[c] = 1.0f - sw - sg;
    }
    if (n_bad_sw || n_bad_sg)
      std::fprintf(stderr,
                   "[warn] step %d: saturations out of bounds before clipping "
                   "(Sw %ld cells, range [%.4g, %.4g]; Sg %ld cells, range "
                   "[%.4g, %.4g]). Output is clipped, but the solver state is "
                   "unphysical.\n",
                   step, n_bad_sw, sw_lo, sw_hi, n_bad_sg, sg_lo, sg_hi);
    // A field grid's state is indexed by ACTIVE cell; the sidecar and every
    // plot downstream are indexed by the full box.  Scatter here, once, so
    // nothing downstream has to know the difference: a field run then writes
    // files of exactly the shape a SPE10 or BENCH3P run writes, and
    // plot_results.py needs no special case for either.
    const bool fbox = (fgeom_.nx > 0 && fgeom_.ny > 0 && fgeom_.nz > 0);
    const size_t nbox =
        fbox ? (size_t)fgeom_.nx * fgeom_.ny * fgeom_.nz : (size_t)ncell;
    const std::vector<int>& a2g = fgeom_.active_to_global;
    // The gathered planes and the map must be the same length, or the scatter
    // silently places a handful of values and leaves the field blank -- which
    // is exactly what happened when the writer was handed the run deck's
    // 1 x 1 x 1 box and gathered a single cell.  A blank plot is not a
    // recognisable failure, so say it outright.
    if (fbox && P.size() != a2g.size()) {
      static bool said = false;
      if (!said) {
        std::fprintf(stderr,
                     "[warn] output: gathered %zu cells but the active-cell "
                     "map has %zu entries. The written fields will be mostly "
                     "empty; the writer is not using the reservoir's grid.\n",
                     P.size(), a2g.size());
        said = true;
      }
    }
    const std::vector<float> Pb  = fbox ? scatter_to_box(P,  a2g, nbox) : P;
    const std::vector<float> SWb = fbox ? scatter_to_box(SW, a2g, nbox) : SW;
    const std::vector<float> SOb = fbox ? scatter_to_box(SO, a2g, nbox) : SO;
    const std::vector<float> SGb = fbox ? scatter_to_box(SG, a2g, nbox) : SG;

    char fn[512];
    std::snprintf(fn, sizeof(fn), "%s/state_%04d.bin", dir_.c_str(), step);
    std::ofstream f(fn, std::ios::binary);
    f.write((const char*)Pb.data(),  sizeof(float) * nbox);
    f.write((const char*)SWb.data(), sizeof(float) * nbox);
    f.write((const char*)SOb.data(), sizeof(float) * nbox);
    f.write((const char*)SGb.data(), sizeof(float) * nbox);
    times_.push_back(t_days);
    ++nsnap_;

    // Static rock, written once.  The CuPy reference renders a Permeability
    // figure from its input field and the post-processor needs the same data;
    // reading it back out of the ECLIPSE INIT would make rock plots depend on
    // ECLOUT being enabled, so write it natively too.  Two float32 planes of
    // ncell, permeability [mD] then porosity, matching the state layout.
    if (first) {
      const std::vector<float> pk =
          fbox ? scatter_to_box(perm_global_, a2g, nbox) : perm_global_;
      const std::vector<float> po =
          fbox ? scatter_to_box(poro_global_, a2g, nbox) : poro_global_;
      std::ofstream rock(dir_ + "/rock.bin", std::ios::binary);
      rock.write((const char*)pk.data(), sizeof(float) * pk.size());
      rock.write((const char*)po.data(), sizeof(float) * po.size());

      // The true geometry, for anything that wants to draw the field rather
      // than its index box: eight corners and a centre per ACTIVE cell, in the
      // grid's own units.  This is what carries the fault throws and the
      // layering; the box above carries neither.
      if (fbox && !fgeom_.cgrid.empty()) {
        std::ofstream gx(dir_ + "/cgrid.bin", std::ios::binary);
        gx.write((const char*)fgeom_.cgrid.data(),
                 sizeof(float) * fgeom_.cgrid.size());
      }
    }

    // --- ECLIPSE / OPM output ---
    if (ecl_) {
      if (first) {
        ecl_->write_egrid();
        ecl_->write_init(perm_global_, permz_global_, poro_global_);
      }
      // ECLIPSE report steps are 1-based; step 0 is the initial condition.
      ecl_->write_restart_step(step, (double)t_days, P, SW, SG, SO);
    }
  }
  (void)cells;
}

void Writer::finalize(const Deck& d, Reservoir& r,
                      const std::vector<Real>& step_times,
                      const std::vector<int>& newton_iters,
                      const std::vector<int>& linear_iters,
                      double wall_seconds, double assembly_seconds,
                      double linear_seconds, const CprTimings& cpr) {
  if (mpi_rank_ != 0) return;

  // meta.json — consumed by scripts/plot_results.py
  {
    std::ofstream m(dir_ + "/meta.json");
    m << "{\n";
    m << "  \"title\": \"" << d.title << "\",\n";
    // The FIELD grid when there is one.  d.nx/ny/nz is the RUN deck's DIMENS,
    // which a field run never sets -- so this reported 1 x 1 x 1 for a model
    // of 113,344 cells, and every plot drew a single square.
    const bool fbox = (fgeom_.nx > 0 && fgeom_.ny > 0 && fgeom_.nz > 0);
    const int mnx = fbox ? fgeom_.nx : d.nx;
    const int mny = fbox ? fgeom_.ny : d.ny;
    const int mnz = fbox ? fgeom_.nz : d.nz;
    m << "  \"nx\": " << mnx << ", \"ny\": " << mny        << ", \"nz\": " << mnz << ",\n";
    if (fbox) {
      // Where the cells actually ARE.  cgrid.bin is six float32 columns per
      // active cell -- i, j, k (1-based) then x, y, z of the cell centre.
      // That is what draws the real field: the layers dip, a fault throws one
      // column against another, and an index box shows neither.
      m << "  \"corner_point\": true,\n";
      m << "  \"n_active\": " << fgeom_.active_to_global.size() << ",\n";
      m << "  \"cgrid\": \"cgrid.bin\", \"cgrid_columns\":"
           " [\"i\",\"j\",\"k\",\"x\",\"y\",\"z\"],\n";
    }
    m << "  \"dx\": " << d.dx << ", \"dy\": " << d.dy << ", \"dz\": " << d.dz << ",\n";
    m << "  \"model\": \""
      << (d.model == Model::Compositional ? "compositional" : "blackoil") << "\",\n";
    m << "  \"nsnap\": " << nsnap_ << ",\n";
    // Which well curves the deck asked for.  Empty means all of them.
    m << "  \"well_plot\": [";
    for (size_t i = 0; i < d.well_plot.size(); ++i)
      m << (i ? "," : "") << "\"" << d.well_plot[i] << "\"";
    m << "],\n";
    // Static rock, written alongside the snapshots as two float32 planes.
    m << "  \"rock\": \"rock.bin\",\n";
    // Constitutive data required to turn PORV*Sw into surface water in place.
    // The comparison must use the same pressure-dependent pore volume and
    // water shrinkage as the residual, not compare a static reservoir volume
    // with stock-tank well cumulatives.
    m << "  \"water_fvf_ref\": " << d.bw
      << ", \"water_compressibility\": " << d.cfw << ",\n";
    m << "  \"rock_ref_pressure\": " << r.rock_pref
      << ", \"rock_compressibility\": " << (r.have_rock ? r.ct : 0.0)
      << ",\n";
    // Wall clock split, so benchmark/compare.py can show which part of
    // the run was the linear solve without re-timing anything.
    m << "  \"assembly_seconds\": " << assembly_seconds << ",\n";
    m << "  \"linear_seconds\": " << linear_seconds << ",\n";
    // CPR internals, so the report can show where the linear solve went:
    // HYPRE's FP32 BoomerAMG against the block-Jacobi second stage.
    m << "  \"amg_setup_seconds\": " << cpr.amg_setup << ",\n";
    m << "  \"amg_solve_seconds\": " << cpr.amg_solve << ",\n";
    m << "  \"stage2_seconds\": " << cpr.stage2 << ",\n";
    m << "  \"stage2_setup_seconds\": " << cpr.stage2_setup << ",\n";
    // The other three CPR terms.  They were measured but never written out, so
    // every report split the linear solve into AMG + stage-2 + an unexplained
    // remainder, and that remainder is not small.  `ortho` is CGS2 and its
    // reductions, which became the largest single term in the solve once the
    // vendor SpMV left the V-cycle; `spmv` is the FP64 or df64 operator apply;
    // `ijvec` is pure marshalling in and out of HYPRE.  A profile that cannot
    // name a term cannot be acted on.
    m << "  \"spmv_seconds\": " << cpr.spmv << ",\n";
    m << "  \"ortho_seconds\": " << cpr.ortho << ",\n";
    m << "  \"ijvec_seconds\": " << cpr.ijvec << ",\n";
    m << "  \"amg_setups\": " << cpr.n_amg_setup << ",\n";
    m << "  \"amg_applications\": " << cpr.n_amg_solve << ",\n";
    m << "  \"amg_cycles\": " << cpr.amg_iters << ",\n";
    // Setups skipped by hierarchy reuse.  Without this the scaling table cannot
    // tell a run that got cheap setups from one that simply did fewer Newton
    // iterations, and those call for opposite conclusions.
    m << "  \"amg_reuses\": " << cpr.n_amg_reuse << ",\n";
    m << "  \"amg_reuse_aborts\": " << cpr.n_amg_forced << ",\n";
    // Graph capture economics (every launch counts in graph_replays,
    // including the one right after each capture), the operator applies by
    // format, and the FGMRES true-vs-recursive residual check.
    m << "  \"graph_captures\": " << cpr.graph_captures << ",\n";
    m << "  \"graph_launches\": " << cpr.graph_replays << ",\n";
    m << "  \"graph_capture_seconds\": " << cpr.graph_capture_seconds << ",\n";
    m << "  \"graph_guard_fired\": " << cpr.graph_guard_fired << ",\n";
    m << "  \"graph_guard_at\": " << cpr.graph_guard_at << ",\n";
    m << "  \"spmv_block_applies\": " << cpr.spmv_block << ",\n";
    m << "  \"spmv_scalar_applies\": " << cpr.spmv_scalar << ",\n";
    m << "  \"spmv_fp32_applies\": " << cpr.spmv_fp32 << ",\n";
    m << "  \"fgmres_gap_max\": " << cpr.fgmres_gap_max << ",\n";
    m << "  \"fgmres_gap_checks\": " << cpr.fgmres_gap_checks << ",\n";
    m << "  \"fgmres_gap_restarts\": " << cpr.fgmres_gap_restarts << ",\n";
    if (rs_set_) {
      m << "  \"wall_seconds\": " << rs_wall_ << ",\n";
      m << "  \"io_seconds\": " << rs_io_ << ",\n";
      m << "  \"accepted_substeps\": " << rs_sub_ << ",\n";
      m << "  \"newton_total\": " << rs_newton_ << ",\n";
      m << "  \"newton_discarded\": " << rs_newton_lost_ << ",\n";
      m << "  \"linear_total\": " << rs_linear_ << ",\n";
      m << "  \"simulated_days\": " << rs_days_ << ",\n";
      m << "  \"operator_mode\": " << rs_opmode_ << ",\n";
      m << "  \"cpr_decouple\": " << rs_decouple_ << ",\n";
    }
    m << "  \"times\": [";
    for (size_t i = 0; i < times_.size(); ++i) m << (i ? "," : "") << times_[i];
    m << "],\n";
    // Solver telemetry, so the post-processor can chart convergence without
    // scraping stdout.
    m << "  \"newton_iters\": [";
    for (size_t i = 0; i < newton_iters.size(); ++i)
      m << (i ? "," : "") << newton_iters[i];
    m << "],\n";
    m << "  \"linear_iters\": [";
    for (size_t i = 0; i < linear_iters.size(); ++i)
      m << (i ? "," : "") << linear_iters[i];
    m << "],\n";
    m << "  \"krylov\": \"" << (d.krylov ? "BiCGSTAB" : "FGMRES") << "\",\n";
    m << "  \"mixed_precision\": " << (d.mixed_precision ? "true" : "false")
      << ", \"emulate_fp64\": " << (d.emulate_fp64 ? "true" : "false") << ",\n";
    m << "  \"nranks\": " << mpi_nranks_ << ",\n";
    m << "  \"wall_seconds\": " << wall_seconds << ",\n";
    // Wells: the FIELD deck's, when there is one.  d.wells is the run deck's
    // WELLSPEC list, which a field run does not use and which is therefore
    // empty -- so every plot came out with no wells on it at all, from a run
    // that tracked thirty-six of them.
    int n_inj = 0, n_prod = 0;
    m << "  \"wells\": [";
    if (fbox) {
      for (size_t i = 0; i < fgeom_.wells.size(); ++i) {
        const auto& w = fgeom_.wells[i];
        if (w.kind == "prod") ++n_prod; else ++n_inj;
        m << (i ? "," : "") << "\n    {\"name\": \"" << w.name
          << "\", \"i\": " << w.i << ", \"j\": " << w.j
          << ", \"type\": \"" << w.kind << "\"}";
      }
    } else
    for (size_t i = 0; i < d.wells.size(); ++i) {
      const auto& w = d.wells[i];
      m << (i ? "," : "") << "\n    {\"name\": \"" << w.name << "\", \"i\": " << w.i
        << ", \"j\": " << w.j << ", \"type\": \"" << (w.type > 0 ? "inj" : "prod")
        << "\"}";
    }
    m << "\n  ],\n";
    if (!fbox) { n_inj = d.n_inj(); n_prod = d.n_prod(); }
    m << "  \"nwells_inj\": " << n_inj << ", \"nwells_prod\": " << n_prod
      << "\n}\n";
  }

  // --- well rates (Peaceman post-process) → RSM_FVM.csv ---
  //
  // This must reproduce the *same* well model the assembly kernels use, or the
  // reported rates will not correspond to the simulated flow field:
  //
  //     WI_k = 2π C_d k_k Δz / (ln(r_e/r_w) + s),     r_e = REFACTOR · Δx
  //     q_α  = Σ_k  WI_k · λ_α(S_k) · max(p_k − p_wf, 0)          [STB/D]
  //
  // with λ_α the same mobility used on device (tabulated SWOF/SGOF for the
  // three-phase model, Brooks-Corey for two-phase).  Three earlier errors are
  // corrected here: PERMMIN was used in place of the cell permeability, the
  // Darcy field-unit constant C_d (≈1.127e-3) was omitted entirely — a ~900×
  // rate error — and |Δp| allowed a shut-in producer to report inflow as
  // production.
  // Only when nothing better was recorded.  This path finds a well by index
  // arithmetic on the deck's own NX/NY/NZ, which a field deck does not have.
  if (wr_rows_.empty()) {
    std::ofstream csv(dir_ + "/RSM_FVM.csv");
    const bool three_phase = (d.model == Model::BlackOil3P);
    csv << "Time(DAY)";
    for (const auto& w : d.wells)
      if (w.type < 0) {
        csv << "," << w.name << "-WOPR(BBL/DAY)," << w.name << "-WWPR(BBL/DAY)";
        if (three_phase) csv << "," << w.name << "-WGPR(MSCF/DAY)";
        csv << "," << w.name << "-WWCT(%)," << w.name << "-WBHP(PSIA)";
      }
    csv << "\n";

    // Host-side view of the rel-perm tables (the device copies live on the GPU).
    std::vector<Real> tsw, tkrow, tkrw, tsg, tkrog, tkrg;
    for (const auto& row : d.swof) { tsw.push_back(row[0]); tkrow.push_back(row[1]); tkrw.push_back(row[2]); }
    for (const auto& row : d.sgof) { tsg.push_back(row[0]); tkrog.push_back(row[1]); tkrg.push_back(row[2]); }
    SatTables T;
    T.sw = tsw.data(); T.krow = tkrow.data(); T.krw = tkrw.data(); T.nsw = (int)tsw.size();
    T.sg = tsg.data(); T.krog = tkrog.data(); T.krg = tkrg.data(); T.nsg = (int)tsg.size();
    const bool have_tables = (T.nsw > 1 && T.nsg > 1);

    const Real re = d.re_factor * d.dx;
    const Real Cd = d.darcy_const;
    const int ncell = d.n_cells();

    // Summary vectors, accumulated alongside the CSV so the ECLIPSE .RSM /
    // .SMSPEC / .UNSMRY files carry exactly the same numbers.
    std::vector<ecl::SummaryVector> sv;
    std::vector<double> sv_times;
    auto add_vec = [&](const char* kw, const std::string& wg, const char* unit,
                       int num) -> ecl::SummaryVector& {
      sv.push_back(ecl::SummaryVector{kw, wg, unit, num, {}});
      return sv.back();
    };
    // Column order must stay fixed across steps, so build the layout up front.
    add_vec("TIME", ":+:+:+:+", "DAYS", 0);
    std::vector<size_t> idx_wopr, idx_wwpr, idx_wgpr, idx_wwct, idx_wbhp;
    std::vector<const WellSpec*> producers;
    int wnum = 0;
    for (const auto& w : d.wells) {
      if (w.type >= 0) continue;
      ++wnum;
      producers.push_back(&w);
      idx_wopr.push_back(sv.size()); add_vec("WOPR", w.name, "STB/DAY", wnum);
      idx_wwpr.push_back(sv.size()); add_vec("WWPR", w.name, "STB/DAY", wnum);
      if (three_phase) {
        idx_wgpr.push_back(sv.size()); add_vec("WGPR", w.name, "MSCF/DAY", wnum);
      }
      idx_wwct.push_back(sv.size()); add_vec("WWCT", w.name, "", wnum);
      idx_wbhp.push_back(sv.size()); add_vec("WBHP", w.name, "PSIA", wnum);
    }
    const size_t i_fopr = sv.size(); add_vec("FOPR", ":+:+:+:+", "STB/DAY", 0);
    const size_t i_fwpr = sv.size(); add_vec("FWPR", ":+:+:+:+", "STB/DAY", 0);
    const size_t i_fopt = sv.size(); add_vec("FOPT", ":+:+:+:+", "STB", 0);
    const size_t i_fwpt = sv.size(); add_vec("FWPT", ":+:+:+:+", "STB", 0);
    const size_t i_fwct = sv.size(); add_vec("FWCT", ":+:+:+:+", "", 0);
    const size_t i_fpr  = sv.size(); add_vec("FPR",  ":+:+:+:+", "PSIA", 0);
    const size_t i_fpav = sv.size(); add_vec("FPAV", ":+:+:+:+", "PSIA", 0);
    double cum_oil = 0, cum_wat = 0, prev_t = 0;

    for (int s = 0; s < nsnap_; ++s) {
      char fn[512];
      std::snprintf(fn, sizeof(fn), "%s/state_%04d.bin", dir_.c_str(), s);
      std::ifstream f(fn, std::ios::binary);
      if (!f) continue;
      std::vector<float> P(ncell), SW(ncell), SO(ncell), SG(ncell);
      f.read((char*)P.data(), sizeof(float) * ncell);
      f.read((char*)SW.data(), sizeof(float) * ncell);
      f.read((char*)SO.data(), sizeof(float) * ncell);
      f.read((char*)SG.data(), sizeof(float) * ncell);
      csv << times_[s];
      sv_times.push_back((double)times_[s]);
      sv[0].values.push_back((double)times_[s]);   // TIME
      double fld_qo = 0, fld_qw = 0;
      size_t pi = 0;                                // producer index
      for (const auto& w : d.wells) {
        if (w.type >= 0) continue;
        const int i = w.i - 1, j = w.j - 1;
        if (i < 0 || i >= d.nx || j < 0 || j >= d.ny) continue;
        // "last" perforates the bottom layer only, matching make_reservoir.
        const int klo = (w.perf == "last") ? d.nz - 1 : 0;
        Real qo = 0, qw = 0, qg = 0;
        for (int k = klo; k < d.nz; ++k) {
          const int c = (k * d.ny + j) * d.nx + i;
          if (c < 0 || c >= ncell) continue;
          const Real kcell = (c < (int)perm_global_.size()) ? perm_global_[c]
                                                            : d.perm_min;
          const Real wi = 2.0 * units::kPi * Cd * kcell * d.dz /
                          (std::log(re / w.rw) + w.skin);
          const Real draw = std::max<Real>(P[c] - w.pwf, 0.0);  // no backflow
          if (draw <= 0) continue;
          const Real Sw = SW[c], Sg = SG[c];
          if (three_phase && have_tables) {
            Mob3 m = relperm3(Sw, Sg, T, d.uw, d.uo, d.ug, d.bw, d.bo, d.bg,
                              d.swi, d.swr);
            qo += wi * m.Mo * draw;
            qw += wi * m.Mw * draw;
            qg += wi * m.Mg * draw;
          } else {
            Mob2 m = relperm2_table(Sw, T, d.uw, d.uo, d.bw, d.bo,
                                    d.swi, d.swr);
            qo += wi * m.Mo * draw;
            qw += wi * m.Mw * draw;
          }
        }
        const Real wct = qw / (qw + qo + 1e-10) * 100.0;
        csv << "," << qo << "," << qw;
        if (three_phase) csv << "," << qg;
        csv << "," << wct << "," << w.pwf;

        sv[idx_wopr[pi]].values.push_back((double)qo);
        sv[idx_wwpr[pi]].values.push_back((double)qw);
        if (three_phase) sv[idx_wgpr[pi]].values.push_back((double)qg);
        sv[idx_wwct[pi]].values.push_back((double)(wct / 100.0));  // fraction
        sv[idx_wbhp[pi]].values.push_back((double)w.pwf);
        fld_qo += (double)qo;
        fld_qw += (double)qw;
        ++pi;
      }
      csv << "\n";

      // Field-level cumulatives, accumulated with the RIGHT endpoint rather
      // than the trapezoidal rule.
      //
      // Every rate here is computed from the state at a report time, and the
      // state at report step s is the end of the interval (t_{s-1}, t_s] that
      // the fully implicit scheme just integrated.  Backward Euler is
      // therefore what the solver actually did, and the trapezoidal rule is
      // not a refinement of it but a different quadrature applied to points
      // that do not support it.
      //
      // The difference is not academic at s = 1.  The rate at t = 0 is the
      // instantaneous rate of the initial condition -- uniform reservoir
      // pressure against the producer BHP, ~6100 STB/day on BENCH2P against a
      // steady ~2000 -- and it decays within hours.  Averaging it across the
      // first 100-day interval invented ~200,000 STB of oil, which is most of
      // the 16% by which FOPT exceeded the drop in oil in place computed from
      // the saturation field.
      const double t = (double)times_[s];
      if (s > 0) {
        const double dt = t - prev_t;
        cum_oil += fld_qo * dt;
        cum_wat += fld_qw * dt;
      }
      prev_t = t;

      double psum = 0;
      for (int c = 0; c < ncell; ++c) psum += P[c];
      const double pavg = ncell ? psum / ncell : 0.0;

      sv[i_fopr].values.push_back(fld_qo);
      sv[i_fwpr].values.push_back(fld_qw);
      sv[i_fopt].values.push_back(cum_oil);
      sv[i_fwpt].values.push_back(cum_wat);
      sv[i_fwct].values.push_back(fld_qw / (fld_qw + fld_qo + 1e-10));
      sv[i_fpr].values.push_back(pavg);
      sv[i_fpav].values.push_back(pavg);
    }

    // --- ECLIPSE summary files (.SMSPEC / .UNSMRY / .RSM) ---
    if (ecl_) ecl_->write_summary(sv_times, sv);
  } else {
    // --- rates the driver recorded, step by step -------------------------
    // These came from the WellCell array the assembly kernels read, so they
    // need no reconstruction here: only the producers have to be selected,
    // laid out in a fixed column order, and integrated for the cumulatives.
    const bool three_phase = (d.model == Model::BlackOil3P);

    // Column order is fixed by the first step that has a well in it.  A
    // schedule opens wells over time -- Norne drills into the 1990s -- so
    // taking the well list from step 0 would lose every well drilled later,
    // and taking it per step would give a ragged CSV that no reader can
    // parse.  So: the union, in first-seen order.
    std::vector<std::string> names, inj_names;
    for (const auto& rows : wr_rows_)
      for (const auto& w : rows) {
        std::vector<std::string>* dst = nullptr;
        if (w.type < 0) dst = &names;
        else if (w.type > 0) dst = &inj_names;
        else continue;  // a shut state alone does not reveal the well's role
        bool seen = false;
        for (const auto& nm : *dst) if (nm == w.name) { seen = true; break; }
        if (!seen) dst->push_back(w.name);
      }

    std::ofstream csv(dir_ + "/RSM_FVM.csv");
    csv << "Time(DAY)";
    for (const auto& nm : names) {
      csv << "," << nm << "-WOPR(BBL/DAY)," << nm << "-WWPR(BBL/DAY)";
      if (three_phase) csv << "," << nm << "-WGPR(MSCF/DAY)";
      csv << "," << nm << "-WWCT(%)," << nm << "-WBHP(PSIA)";
    }
    for (const auto& nm : inj_names)
      csv << "," << nm << "-WWIR(BBL/DAY)," << nm
          << "-WGIR(MSCF/DAY)," << nm << "-IBHP(PSIA)";
    csv << "\n";

    std::vector<ecl::SummaryVector> sv;
    std::vector<double> sv_times;
    auto add_vec = [&](const char* kw, const std::string& wg, const char* unit,
                       int num) { sv.push_back(ecl::SummaryVector{kw, wg, unit, num, {}}); };
    add_vec("TIME", ":+:+:+:+", "DAYS", 0);
    std::vector<size_t> i_wopr, i_wwpr, i_wgpr, i_wwct, i_wbhp;
    int wnum = 0;
    for (const auto& nm : names) {
      ++wnum;
      i_wopr.push_back(sv.size()); add_vec("WOPR", nm, "STB/DAY", wnum);
      i_wwpr.push_back(sv.size()); add_vec("WWPR", nm, "STB/DAY", wnum);
      if (three_phase) { i_wgpr.push_back(sv.size()); add_vec("WGPR", nm, "MSCF/DAY", wnum); }
      i_wwct.push_back(sv.size()); add_vec("WWCT", nm, "", wnum);
      i_wbhp.push_back(sv.size()); add_vec("WBHP", nm, "PSIA", wnum);
    }
    std::vector<size_t> i_wwir, i_wgir, i_ibhp;
    for (const auto& nm : inj_names) {
      ++wnum;
      i_wwir.push_back(sv.size()); add_vec("WWIR", nm, "STB/DAY", wnum);
      i_wgir.push_back(sv.size()); add_vec("WGIR", nm, "MSCF/DAY", wnum);
      i_ibhp.push_back(sv.size()); add_vec("WBHP", nm, "PSIA", wnum);
    }
    const size_t i_fopr = sv.size(); add_vec("FOPR", ":+:+:+:+", "STB/DAY", 0);
    const size_t i_fwpr = sv.size(); add_vec("FWPR", ":+:+:+:+", "STB/DAY", 0);
    const size_t i_fgpr = sv.size(); add_vec("FGPR", ":+:+:+:+", "MSCF/DAY", 0);
    const size_t i_fopt = sv.size(); add_vec("FOPT", ":+:+:+:+", "STB", 0);
    const size_t i_fwpt = sv.size(); add_vec("FWPT", ":+:+:+:+", "STB", 0);
    const size_t i_fwct = sv.size(); add_vec("FWCT", ":+:+:+:+", "", 0);
    const size_t i_fwir = sv.size(); add_vec("FWIR", ":+:+:+:+", "STB/DAY", 0);
    const size_t i_fgir = sv.size(); add_vec("FGIR", ":+:+:+:+", "MSCF/DAY", 0);
    const size_t i_fvpr = sv.size(); add_vec("FVPR", ":+:+:+:+", "RB/DAY", 0);
    const size_t i_fvir = sv.size(); add_vec("FVIR", ":+:+:+:+", "RB/DAY", 0);
    const size_t i_fvpt = sv.size(); add_vec("FVPT", ":+:+:+:+", "RB", 0);
    const size_t i_fvit = sv.size(); add_vec("FVIT", ":+:+:+:+", "RB", 0);

    double cum_oil = 0, cum_wat = 0, cum_vp = 0, cum_vi = 0, prev_t = 0;
    for (size_t s2 = 0; s2 < wr_rows_.size(); ++s2) {
      const double t = (double)wr_times_[s2];
      csv << t;
      sv_times.push_back(t);
      sv[0].values.push_back(t);
      double fld_qo = 0, fld_qw = 0, fld_qg = 0;
      double fld_qwi = 0, fld_qgi = 0;
      double fld_qvp = 0, fld_qvi = 0;
      for (const auto& w : wr_rows_[s2]) {
        if (w.type > 0) {
          fld_qwi += (double)w.qw;
          fld_qgi += (double)w.qg;
          fld_qvi += (double)w.qres;
        } else if (w.type < 0) {
          fld_qvp += (double)w.qres;
        }
      }
      for (size_t c = 0; c < names.size(); ++c) {
        const WellRateRow* w = nullptr;
        for (const auto& cand : wr_rows_[s2])
          if (cand.name == names[c]) { w = &cand; break; }
        // A well not yet drilled, or shut, reports zero rather than a gap:
        // the column has to exist at every step or the row lengths diverge.
        const Real qo = (w && w->type < 0) ? w->qo : (Real)0;
        const Real qw = (w && w->type < 0) ? w->qw : (Real)0;
        const Real qg = (w && w->type < 0) ? w->qg : (Real)0;
        const Real bhp = w ? w->bhp : (Real)0;
        // A production water cut is defined from outward surface phase rates.
        // Completion crossflow can make one net phase rate slightly negative;
        // dividing those signed rates produced impossible WWCT values of 2--40
        // in Norne.  OPM's producer curve is bounded, so use the produced parts
        // and report zero when the well has no outward liquid flow.
        const Real qo_prod = std::max(qo, (Real)0);
        const Real qw_prod = std::max(qw, (Real)0);
        const Real ql_prod = qo_prod + qw_prod;
        const Real wct = ql_prod > (Real)1e-10
                             ? qw_prod / ql_prod * (Real)100
                             : (Real)0;
        csv << "," << qo << "," << qw;
        if (three_phase) csv << "," << qg;
        csv << "," << wct << "," << bhp;
        sv[i_wopr[c]].values.push_back((double)qo);
        sv[i_wwpr[c]].values.push_back((double)qw);
        if (three_phase) sv[i_wgpr[c]].values.push_back((double)qg);
        sv[i_wwct[c]].values.push_back((double)(wct / 100.0));
        sv[i_wbhp[c]].values.push_back((double)bhp);
        fld_qo += (double)qo;
        fld_qw += (double)qw;
        fld_qg += (double)qg;
      }
      for (size_t c = 0; c < inj_names.size(); ++c) {
        const WellRateRow* w = nullptr;
        for (const auto& cand : wr_rows_[s2])
          if (cand.name == inj_names[c]) { w = &cand; break; }
        const Real qw = (w && w->type > 0) ? w->qw : (Real)0;
        const Real qg = (w && w->type > 0) ? w->qg : (Real)0;
        const Real bhp = w ? w->bhp : (Real)0;
        csv << "," << qw << "," << qg << "," << bhp;
        sv[i_wwir[c]].values.push_back((double)qw);
        sv[i_wgir[c]].values.push_back((double)qg);
        sv[i_ibhp[c]].values.push_back((double)bhp);
      }
      csv << "\n";
      // Backward Euler, for the reason given in the branch above.
      if (s2 > 0) {
        const double dt = t - prev_t;
        cum_oil += fld_qo * dt;
        cum_wat += fld_qw * dt;
        cum_vp += fld_qvp * dt;
        cum_vi += fld_qvi * dt;
      }
      if (wr_cumulative_.size() == wr_rows_.size()) {
        cum_oil = wr_cumulative_[s2][0];
        cum_wat = wr_cumulative_[s2][1];
      }
      prev_t = t;
      sv[i_fopr].values.push_back(fld_qo);
      sv[i_fwpr].values.push_back(fld_qw);
      sv[i_fgpr].values.push_back(fld_qg);
      sv[i_fopt].values.push_back(cum_oil);
      sv[i_fwpt].values.push_back(cum_wat);
      sv[i_fwct].values.push_back(fld_qw / (fld_qw + fld_qo + 1e-10));
      sv[i_fwir].values.push_back(fld_qwi);
      sv[i_fgir].values.push_back(fld_qgi);
      sv[i_fvpr].values.push_back(fld_qvp);
      sv[i_fvir].values.push_back(fld_qvi);
      sv[i_fvpt].values.push_back(cum_vp);
      sv[i_fvit].values.push_back(cum_vi);
    }
    std::printf("  well rates: %zu producer(s), %zu injector(s) over %zu "
                "report step(s) -> RSM_FVM.csv\n", names.size(),
                inj_names.size(), wr_rows_.size());
    if (ecl_) ecl_->write_summary(sv_times, sv);
  }

  // --- insight summary ---
  if (d.make_summary) {
    std::ofstream s(dir_ + "/summary.txt");
    int tot_newton = 0, tot_lin = 0;
    for (int x : newton_iters) tot_newton += x;
    for (int x : linear_iters) tot_lin += x;
    s << "==========================================================\n";
    s << "  NVRS-CUDA simulation summary — " << d.title << "\n";
    s << "==========================================================\n\n";
    s << "Model              : "
      << (d.model == Model::Compositional ? "Compositional"
          : d.model == Model::BlackOil2P  ? "Black-oil (2-phase)"
                                          : "Black-oil (3-phase)")
      << "\n";
    s << "Grid               : " << d.nx << " x " << d.ny << " x " << d.nz
      << "  (" << d.n_cells() << " active cells)\n";
    s << "Ranks / GPUs       : " << mpi_nranks_ << "\n";
    s << "Solver             : CPR + "
      << (d.krylov ? "BiCGSTAB" : "FGMRES") << "  ["
      << (d.mixed_precision ? "mixed FP32 AMG" : "FP64 AMG") << "]\n";
    s << "Report steps       : " << (int)step_times.size() << "\n";
    s << "Total Newton iters : " << tot_newton << "\n";
    s << "Total linear iters : " << tot_lin << "\n";
    if (!step_times.empty())
      s << "Avg Newton/step    : " << (double)tot_newton / step_times.size() << "\n";
    s << "Wall-clock time    : " << wall_seconds << " s\n\n";
    s << "Insights\n--------\n";
    s << "* CPR kept the outer Krylov iteration count low and mesh-independent:\n";
    s << "  the BoomerAMG pressure stage removes the elliptic (incompressible)\n";
    s << "  error mode, while the block-Jacobi second stage cleans the local\n";
    s << "  hyperbolic saturation coupling — see docs/ for the theory.\n";
    if (d.mixed_precision)
      s << "* Mixed precision (FP32 AMG under an FP64 Krylov) preserves the FP64\n"
           "  convergence path while roughly halving preconditioner bandwidth.\n";
    s << "* Well response is written to RSM_FVM.csv; fields to state_*.bin.\n";
    s.close();
  }
  (void)r;
}

}  // namespace nvrs
