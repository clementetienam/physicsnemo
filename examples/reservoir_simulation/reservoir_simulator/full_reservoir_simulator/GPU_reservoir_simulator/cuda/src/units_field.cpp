// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// units_field.cpp -- a field deck's own unit system, converted once.
//
// A corner-point deck is ordinarily METRIC and the solver works in field units
// throughout, because every Darcy constant, well index and accumulation term in
// it was written that way.  Converting once here, when the model is loaded, is
// what lets the kernels stay unit-agnostic: after this runs the solver cannot
// tell what system the deck came from.
//
// This lives in its own HOST-ONLY translation unit rather than beside the rest
// of field_model.cu, and that is the point of the file.  Every factor below is
// silent when it is wrong: a metric rate read as a field one simulates a
// different field, converges, and reports success.  Norne is the case that
// makes this concrete -- 12,811 sm3/day read as stb/day is a sixth of the
// intended injection, and nothing downstream can tell.  The only defence is a
// test, and a test cannot run on a laptop if the conversion is compiled by
// nvcc and linked against CUDA.  Keeping it here means tests/test_field_units
// exercises the very function the simulator calls.
#include "field_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nvrs {
namespace {

// ---------------------------------------------------------------------------
// Unit conversion
// ---------------------------------------------------------------------------
// Metric to field.  Only the quantities that cross into the solver appear
// here; permeability is mD in both systems and needs none.
constexpr Real kM3ToBbl   = 6.2898107704;      // m^3   -> reservoir bbl
constexpr Real kMToFt     = 3.280839895;       // m     -> ft
constexpr Real kBarToPsi  = 14.503773773;      // bar   -> psi
// Transmissibility carries volume/pressure, so it converts by their ratio:
//   F[bbl/day] = T_f * M * dP[psi]  and  F[m3/day] = T_m * M * dP[bar]
// with F[bbl] = F[m3]*kM3ToBbl and dP[psi] = dP[bar]*kBarToPsi.
constexpr Real kTransMetricToField = kM3ToBbl / kBarToPsi;

}  // namespace

// ---------------------------------------------------------------------------
std::unique_ptr<FieldModel> load_field_model(const std::string& deck_path,
                                             const Real* gradients,
                                             Real connate_water,
                                             int rank, int nranks,
                                             int props_arrays,
                                             Real trans_drop_rel) {
  auto fm = std::make_unique<FieldModel>();
  fm->deck_path = deck_path;

  // Decided before the read, because it changes what the read takes in.
  fm->grid.read_props_arrays = (props_arrays != 0);
  fm->grid.read_grdecl(deck_path);
  fm->grid.set_drop_threshold(trans_drop_rel);
  fm->grid.build();
  // Remembered before anything converts it, so results can be reported in the
  // system the deck was actually written in.
  fm->input_units = fm->grid.units();

  fm->schedule.read(deck_path);
  fm->schedule.resolve(fm->grid);

  fm->equil.read(deck_path);
  if (gradients && fm->equil.n_regions() > 0) {
    fm->equil.set_gradients(gradients[0], gradients[1], gradients[2]);
    fm->initial = fm->equil.initialise(fm->grid, connate_water);
    fm->have_initial = true;
  }

  // RSVD -> one Rs anchor per active cell, built HERE because it is the last
  // moment the grid's depths and the RSVD tables are still in the same length
  // unit.  to_field_units() below moves the grid to feet and leaves the tables
  // where they are, so building it afterwards would look up a metre depth in a
  // table indexed in metres using a number that is now in feet.
  //
  // The dissolved gas itself converts by the same factor the PVT tables use:
  // sm3/sm3 -> scf/stb is (35.3146667/1000)/6.2898107704 * 1000 = 5.6146.
  {
    const Real kRsMetricToField =
        (fm->grid.units() == Units::Metric)
            ? (Real)((35.3146667 / 1000.0) / 6.2898107704 * 1000.0)
            : (Real)1000.0;   // a FIELD deck states Mscf/stb; the solver wants scf
    fm->rs_anchor = fm->equil.rs_anchor(fm->grid, kRsMetricToField);
    // DRSDT is a RATE of change of Rs, so it carries Rs's units and converts
    // by the same factor.
    if (fm->schedule.have_drsdt()) {
      const Real d = fm->schedule.drsdt();
      fm->drsdt = (d >= 0) ? d * kRsMetricToField : (Real)-1;
      if (rank == 0)
        std::printf("  [drsdt] dissolved gas may rise by at most %.4g "
                    "scf/stb/day%s\n", (double)fm->drsdt,
                    (fm->drsdt == 0) ? "  -- free gas may never redissolve"
                                     : "");
    }
    if (fm->schedule.have_vappars()) {
      fm->drsdt = -1; // VAPPARS cancels an earlier DRSDT by deck definition.
      fm->vap1 = fm->schedule.vap1();
      fm->vap2 = fm->schedule.vap2();
      if (rank == 0)
        std::printf("  [vappars] Rv depletion exponent %.4g, Rs exponent %.4g"
                    " (replaces DRSDT/DRVDT)\n",
                    (double)fm->vap1, (double)fm->vap2);
    }
    if (rank == 0 && !fm->rs_anchor.empty()) {
      Real lo = fm->rs_anchor[0], hi = fm->rs_anchor[0];
      for (Real v : fm->rs_anchor) { lo = std::min(lo, v); hi = std::max(hi, v); }
      std::printf("  [rsvd] dissolved-gas anchor from RSVD: %d table(s), "
                  "%.1f .. %.1f scf/stb over %d cells (deck RS is a single "
                  "number and is now only the fallback)\n",
                  (int)fm->equil.rsvd().size(), (double)lo, (double)hi,
                  (int)fm->rs_anchor.size());
    }
  }

  // Convert to the solver's units BEFORE partitioning.  partition_field takes
  // a copy of the connections, so converting afterwards would update the grid
  // and leave the partition holding the deck's units -- and the partition is
  // what the assembly actually reads.  That mismatch is a factor of 0.43 on
  // every transmissibility in a METRIC deck, with nothing to show for it.
  to_field_units(*fm);

  // How the grid is split, and the exchange that follows.  On one rank the
  // partition owns everything and the halo has no peers, so there is no
  // separate single-rank path to keep working.
  fm->part = partition_field(fm->grid, rank, nranks);
  return fm;
}

// ---------------------------------------------------------------------------
void to_field_units(FieldModel& fm) {
  if (fm.grid.units() == Units::Field) return;   // already there, or idempotent

  CornerPointGrid& g = fm.grid;
  const int na = g.n_active();

  for (int a = 0; a < na; ++a) {
    g.volume[a] *= kM3ToBbl;
    g.porv[a]   *= kM3ToBbl;
    g.depth[a]  *= kMToFt;
    g.dx[a]     *= kMToFt;
    g.dy[a]     *= kMToFt;
    g.dz[a]     *= kMToFt;
    g.centroid_x[a] *= kMToFt;
    g.centroid_y[a] *= kMToFt;
  }
  for (Connection& c : g.conns) {
    c.trans *= kTransMetricToField;
    c.threshold *= kBarToPsi;
  }

  // The well index has the same units as a transmissibility.  A connection
  // factor the deck stated is in the deck's units and converts the same way.
  //
  // The well CONTROLS need converting too, and for the same reason: a rate of
  // 12811 in a METRIC deck is 12811 sm3/day, which the solver would otherwise
  // read as 12811 stb/day -- a sixth of the intended injection.  Rates convert
  // by phase, because field gas is Mscf/day while field liquids are stb/day.
  const Real kLiq = kM3ToBbl;                     // sm3/day -> stb/day
  // sm3/day -> scf/day, NOT Mscf/day.
  //
  // Mscf is the FIELD unit a deck reports gas in, and it is the wrong one to
  // hand the solver.  The gas equation is in scf: bg is scf/rb and Rs is
  // scf/stb, so its accumulation and flux are both scf/day, and a source term
  // in Mscf/day enters a thousand times too small.  Norne's C-4H asks for
  // 392181 sm3/day, which is 1.385e7 scf/day; as 1.385e4 it is five orders
  // below the gas residual and the well does nothing at all.
  const Real kGas = 35.3146667;                   // sm3/day -> scf/day
  auto conv_p = [](Real p) {
    return std::isfinite((double)p) ? p * kBarToPsi : p;
  };
  auto conv_l = [&](Real q) {
    return std::isfinite((double)q) ? q * kLiq : q;
  };
  auto conv_g = [&](Real q) {
    return std::isfinite((double)q) ? q * kGas : q;
  };

  for (ScheduleStep& s : fm.schedule.steps_mutable()) {
    // Group targets are rates in the deck's units too, and are useless in
    // the wrong ones: a metric oil target read as STB would allocate the
    // field six times the production it asked for.
    for (auto& kv : s.groups) {
      GroupControl& g = kv.second;
      if (g.mode == CtrlMode::GRAT) g.target = conv_g(g.target);
      else                          g.target = conv_l(g.target);
    }
    // GCONINJE targets come from the same METRIC schedule.  In particular,
    // Norne's late FIELD/WATER RATE of 42000 is sm3/day (264172 stb/day), not
    // 42000 stb/day.  Leaving it raw clamps every water injector to one sixth
    // of OPM's intended group ceiling from day 2917 onward.
    convert_group_inject_targets(s, kLiq, kGas);
    for (Well& w : s.wells) {
      // WELSPECS datum depth is in the deck's length unit.  OPM reports and
      // controls BHP at this datum, while the connection pressure is corrected
      // to each completion depth.  Leaving a metric depth in metres beside a
      // grid converted to feet makes that correction wrong by 3.28084.
      if (std::isfinite((double)w.ref_depth) && w.ref_depth >= 0)
        w.ref_depth *= kMToFt;
      for (Completion& c : w.completions) c.wi *= kTransMetricToField;

      WellState& t = w.state;
      t.orat = conv_l(t.orat);
      t.wrat = conv_l(t.wrat);
      t.lrat = conv_l(t.lrat);
      t.grat = conv_g(t.grat);
      t.resv = conv_l(t.resv);          // reservoir volume, so the liquid factor
      t.resv_rate = conv_l(t.resv_rate);
      t.hist_orat = conv_l(t.hist_orat);
      t.hist_wrat = conv_l(t.hist_wrat);
      t.hist_grat = conv_g(t.hist_grat);
      // An injector's surface rate is in the units of the phase it injects.
      t.surface_rate = (t.inj_phase == InjectedPhase::Gas)
                           ? conv_g(t.surface_rate)
                           : conv_l(t.surface_rate);
      t.bhp_limit = conv_p(t.bhp_limit);
      t.thp_limit = conv_p(t.thp_limit);
      t.hist_bhp = conv_p(t.hist_bhp);
      t.hist_thp = conv_p(t.hist_thp);
    }
  }

  if (fm.have_initial) {
    for (Real& p : fm.initial.pressure) p *= kBarToPsi;
    for (Real& p : fm.initial.pcow_target) p *= kBarToPsi;
    // Saturations are dimensionless.
  }

  g.set_units(Units::Field);
}

}  // namespace nvrs
