// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_field_units.cpp -- METRIC field decks, and the well controls in them.
//
// tests/test_units.cpp already round-trips the RUN deck.  This is the other
// half, and on Norne it is the half that matters: the run deck is a handful of
// solver settings, while the grid, the wells and every control that drives them
// come from a corner-point FIELD deck which is METRIC throughout -- rates in
// sm3/day, pressures in bar, depths in metres, connection factors in
// cP.rm3/(day.bar).  Those are the numbers the well model is handed.
//
// The property tested is the only one that catches a wrong factor: write the
// SAME field twice, once in metric and once in field units, load both through
// load_field_model(), and require every quantity the solver goes on to use to
// agree.  A conversion that is merely self-consistent passes no test at all --
// every wrong factor is self-consistent too.  It simply simulates a different
// field, quietly, and the run converges and reports success.
//
// Norne is why each of these is checked rather than a plausible-looking subset:
//
//   - a gas injection rate is sm3/day and the gas equation is scf/day, a
//     factor of 35.31 and not the liquid 6.29;
//   - WCONHIST's observed rates drive nearly every producer, and its GRAT
//     column is gas where ORAT and WRAT are liquid;
//   - a WCONINJE bottom-hole ceiling of 600 is 600 BAR, and read as 600 psi
//     the injector is throttled to nothing;
//   - the WELSPECS datum is metres, and the completion pressure correction is
//     wrong by 3.28 without it;
//   - COMPDAT's connection factor is a transmissibility, so it carries volume
//     over pressure and converts by neither factor alone.
//
//   g++ -O2 -std=c++17 -I../include test_field_units.cpp ../src/units_field.cpp
//       ../src/cpgrid.cpp ../src/schedule.cpp ../src/equil.cpp
//       ../src/field_partition.cpp -o t && ./t

#include "field_model.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using namespace nvrs;

namespace {

int g_failures = 0, g_checks = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) { ++g_failures; std::printf("  FAIL  %s\n", what); }
  else      std::printf("  ok    %s\n", what);
}

// Relative agreement.  The tolerance absorbs the decimal rounding in the
// hand-written field deck below and nothing else: every factor this test exists
// to catch is out by 3.28, 5.61, 6.29, 14.5 or 35.3.
void same(double metric_side, double field_side, const char* what,
          double tol = 1e-5) {
  ++g_checks;
  const double d = std::fabs(field_side) > 1e-30
                       ? std::fabs(metric_side - field_side) / std::fabs(field_side)
                       : std::fabs(metric_side - field_side);
  const bool ok = (d <= tol);
  if (!ok) ++g_failures;
  std::printf("  %-5s %-40s %14.7g  vs %14.7g   (%.2g)\n",
              ok ? "ok" : "FAIL", what, metric_side, field_side, d);
}

std::string write_file(const std::string& path, const std::string& body) {
  std::ofstream f(path);
  f << body;
  return path;
}

// The same 5x5x3 box, the same two wells and the same controls, in both unit
// systems.  Every metric number below is the field one divided by its factor,
// written to enough digits that this is testing the code and not the arithmetic
// in this file.
//
//   length      1 m   = 3.280839895 ft
//   pressure    1 bar = 14.503773773 psi
//   liquid vol  1 sm3 = 6.2898107704 stb
//   gas vol     1 sm3 = 35.3146667 scf
//   transmis.   1 cP.rm3/(day.bar) = 6.2898107704 / 14.503773773 field
std::string deck_body(bool metric) {
  const int nx = 5, ny = 5, nz = 3;
  const double L = metric ? 1.0 : 3.280839895;
  const double dx = 20.0 * L, dy = 20.0 * L, dz = 4.0 * L;
  const double top = 2000.0 * L;

  std::ostringstream f;
  f.precision(12);
  f << (metric ? "METRIC\n\n" : "FIELD\n\n");
  f << "SPECGRID\n " << nx << " " << ny << " " << nz << " 1 F /\n\nCOORD\n";
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i)
      f << " " << i * dx << " " << j * dy << " " << (top - 100 * L) << " "
        << i * dx << " " << j * dy << " " << (top + nz * dz + 100 * L) << "\n";
  f << "/\n\nZCORN\n";
  for (int k = 0; k < nz; ++k)
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < ny; ++j)
        for (int dup = 0; dup < 2; ++dup)
          for (int i = 0; i < nx; ++i)
            for (int d = 0; d < 2; ++d) f << " " << (top + (k + h) * dz);
  f << "\n/\n\n";
  f << "PORO\n " << (nx * ny * nz) << "*0.25 /\n\n";
  f << "PERMX\n " << (nx * ny * nz) << "*500 /\n\n";   // mD in both systems
  f << "PERMY\n " << (nx * ny * nz) << "*500 /\n\n";
  f << "PERMZ\n " << (nx * ny * nz) << "*50 /\n\n";
  f << "NTG\n " << (nx * ny * nz) << "*1 /\n\n";

  // Equilibration, so the initial state compared below is a converted one
  // rather than a default.  The datum, the contacts and the datum pressure are
  // all in the deck's own units, which is the point of stating them.  This
  // comes before SCHEDULE because the equilibration reader stops there.
  const double eq_depth = metric ? 2600.0 : 8530.1837270;   // m   / ft
  const double eq_press = metric ? 270.0  : 3916.0189187;   // bar / psi
  const double eq_woc   = metric ? 2700.0 : 8858.2677165;
  const double eq_goc   = metric ? 2500.0 : 8202.0997375;
  f << "SOLUTION\n\nEQUIL\n  " << eq_depth << " " << eq_press << " " << eq_woc
    << " 0 " << eq_goc << " 0 1 1 1 /\n\n";

  // --- the wells, and the controls that drive them -------------------------
  //
  // The datum depth, the connection factors, the observed history rates, the
  // injector surface rates and both bottom-hole limits are all stated, so every
  // conversion the well model depends on is exercised at once.
  const double datum    = metric ? 2500.0     : 8202.0997375;    // m    / ft
  const double cf_p     = metric ? 120.0      : 52.0400624183;   // transmis.
  const double cf_i     = metric ? 80.0       : 34.6933749455;
  const double diam     = metric ? 0.216      : 0.708661417;     // m    / ft
  const double prod_bhp = metric ? 150.0      : 2175.5660659;    // bar  / psi
  const double inj_bhp  = metric ? 600.0      : 8702.2642638;
  const double h_orat   = metric ? 5093.7     : 32038.4091212;   // sm3  / stb
  const double h_wrat   = metric ? 318.4      : 2002.6757493;
  const double h_grat   = metric ? 565401.375 : 19966961.1098;   // sm3  / scf
  const double inj_wat  = metric ? 4200.0     : 26417.2052357;   // sm3  / stb
  const double inj_gas  = metric ? 392180.973 : 13849740.3476;   // sm3  / scf
  const double resv_p   = metric ? 7000.0     : 44028.6753928;   // rm3  / rb
  const double gconinje = metric ? 42000.0    : 264172.0523568;  // sm3  / stb

  f << "START\n  1 " << "'JAN'" << " 2000 /\n\nSCHEDULE\n";
  f << "WELSPECS\n"
    << "  'P1' 'G' 5 5 " << datum << " 'OIL'   7* /\n"
    << "  'I1' 'G' 1 1 " << datum << " 'WATER' 7* /\n/\n\n";
  f << "COMPDAT\n"
    << "  'P1' 5 5 1 3 'OPEN' 1* " << cf_p << " " << diam
    << " 1* 0.0 1* 'Z' 1* /\n"
    << "  'I1' 1 1 1 3 'OPEN' 1* " << cf_i << " " << diam
    << " 1* 0.0 1* 'Z' 1* /\n/\n\n";
  // A history producer: the Norne case, where the observation IS the control.
  f << "WCONHIST\n"
    << "  'P1' 'OPEN' 'RESV' " << h_orat << " " << h_wrat << " " << h_grat
    << " 5* /\n/\n\n";
  f << "WCONINJE\n"
    << "  'I1' 'WATER' 'OPEN' 'RATE' " << inj_wat << " 1* " << inj_bhp
    << " /\n/\n\n";
  f << "DATES\n  1 'FEB' 2000 /\n/\n\n";
  // A prediction producer, so the WCONPROD columns are covered too, and the
  // injector converted to gas -- whose rate must take the gas factor.
  f << "WCONPROD\n"
    << "  'P1' 'OPEN' 'RESV' 1* 1* 1* 1* " << resv_p << " " << prod_bhp
    << " /\n/\n\n";
  f << "WCONINJE\n"
    << "  'I1' 'GAS' 'OPEN' 'RATE' " << inj_gas << " 1* " << inj_bhp
    << " /\n/\n\n";
  // A group injection ceiling: Norne states exactly one, and in sm3/day.
  f << "GCONINJE\n"
    << "  'G' 'WATER' 'RATE' " << gconinje << " 9* /\n/\n\n";
  f << "DATES\n  1 'MAR' 2000 /\n/\n";
  return f.str();
}

const Well* well_at(const FieldModel& fm, int step, const char* name) {
  if (step < 0 || step >= fm.schedule.n_steps()) return nullptr;
  return fm.schedule.step(step).find(name);
}

}  // namespace

int main() {
  std::printf("test_field_units -- a METRIC field deck against its field twin\n");

  const std::string dm = write_file("_fu_metric.DATA", deck_body(true));
  const std::string df = write_file("_fu_field.DATA", deck_body(false));

  // Equilibration needs the phase gradients, and they are stated in the deck's
  // own pressure-over-length unit, so they convert with everything else.
  const Real grad_m[3] = {(Real)0.0706, (Real)0.1001, (Real)0.0186};  // bar/m
  const double k_grad = 14.503773773 / 3.280839895;
  const Real grad_f[3] = {(Real)(0.0706 * k_grad), (Real)(0.1001 * k_grad),
                          (Real)(0.0186 * k_grad)};                   // psi/ft

  std::unique_ptr<FieldModel> m = load_field_model(dm, grad_m, (Real)0.1);
  std::unique_ptr<FieldModel> g = load_field_model(df, grad_f, (Real)0.1);

  check(!!m && !!g, "both decks loaded");
  if (!m || !g) { std::printf("\ncannot continue\n"); return 1; }

  check(m->input_units == Units::Metric, "the metric deck was recognised");
  check(g->input_units == Units::Field, "the field deck was recognised");
  check(m->n_active() == g->n_active(), "both grids have the same active cells");

  // --- geometry ------------------------------------------------------------
  std::printf("\n[1] geometry\n");
  const int na = m->n_active();
  double vol_m = 0, vol_f = 0, pv_m = 0, pv_f = 0, dep_m = 0, dep_f = 0;
  for (int a = 0; a < na; ++a) {
    vol_m += m->grid.volume[a]; vol_f += g->grid.volume[a];
    pv_m  += m->grid.porv[a];   pv_f  += g->grid.porv[a];
    dep_m += m->grid.depth[a];  dep_f += g->grid.depth[a];
  }
  same(vol_m, vol_f, "total bulk volume [rb]");
  same(pv_m,  pv_f,  "total pore volume [rb]");
  same(dep_m, dep_f, "summed cell depth [ft]");
  double tr_m = 0, tr_f = 0;
  for (const Connection& c : m->grid.conns) tr_m += c.trans;
  for (const Connection& c : g->grid.conns) tr_f += c.trans;
  // The two Darcy constants in cpgrid.cpp -- 0.0085267014 metric and
  // 0.00112712 field -- are consistent with each other only to 4e-5 relative;
  // an exactly consistent metric partner for 0.00112712 is 0.00852705.  Four
  // parts in a hundred thousand is physically nothing, but it is real, so the
  // tolerance here names it rather than hiding it in a round number.
  same(tr_m, tr_f, "total transmissibility", 1e-4);

  // --- well geometry and controls -----------------------------------------
  std::printf("\n[2] well geometry\n");
  const Well* pm = well_at(*m, 0, "P1");
  const Well* pf = well_at(*g, 0, "P1");
  const Well* im = well_at(*m, 0, "I1");
  const Well* jf = well_at(*g, 0, "I1");
  check(pm && pf && im && jf, "both wells present in both decks");
  if (!(pm && pf && im && jf)) { std::printf("\ncannot continue\n"); return 1; }

  same(pm->ref_depth, pf->ref_depth, "WELSPECS datum depth [ft]");
  same(im->ref_depth, jf->ref_depth, "injector datum depth [ft]");
  double wi_m = 0, wi_f = 0;
  for (const Completion& c : pm->completions) wi_m += c.wi;
  for (const Completion& c : pf->completions) wi_f += c.wi;
  same(wi_m, wi_f, "COMPDAT connection factor [field]");

  std::printf("\n[3] history controls (WCONHIST)\n");
  same(pm->state.hist_orat, pf->state.hist_orat, "WCONHIST ORAT [stb/day]");
  same(pm->state.hist_wrat, pf->state.hist_wrat, "WCONHIST WRAT [stb/day]");
  same(pm->state.hist_grat, pf->state.hist_grat, "WCONHIST GRAT [scf/day]");
  same(pm->state.lrat, pf->state.lrat, "derived LRAT [stb/day]");
  check(pm->state.historical && pf->state.historical,
        "both read the record as a history control");
  check(pm->state.mode == pf->state.mode, "both selected the same CMODE");

  std::printf("\n[4] injector controls (WCONINJE)\n");
  same(im->state.surface_rate, jf->state.surface_rate, "water RATE [stb/day]");
  same(im->state.bhp_limit, jf->state.bhp_limit, "injector BHP ceiling [psi]");
  check(im->state.inj_phase == jf->state.inj_phase, "same injected phase");

  const Well* im2 = well_at(*m, 1, "I1");
  const Well* jf2 = well_at(*g, 1, "I1");
  check(im2 && jf2, "the second interval has the injector");
  if (im2 && jf2) {
    same(im2->state.surface_rate, jf2->state.surface_rate, "gas RATE [scf/day]");
    check(im2->state.inj_phase == InjectedPhase::Gas &&
          jf2->state.inj_phase == InjectedPhase::Gas,
          "and it converted to a gas injector");
    // The one that is easy to get wrong.  6.29 instead of 35.31 is 5.6x too
    // little gas, which on Norne is a gas injector that does nothing at all.
    check(std::fabs((double)im2->state.surface_rate / 6.2898107704 -
                    392180.973) > 1.0,
          "the gas rate did not silently take the liquid factor");
  }

  std::printf("\n[5] prediction controls (WCONPROD)\n");
  const Well* pm2 = well_at(*m, 1, "P1");
  const Well* pf2 = well_at(*g, 1, "P1");
  check(pm2 && pf2, "the second interval has the producer");
  if (pm2 && pf2) {
    same(pm2->state.resv, pf2->state.resv, "WCONPROD RESV [rb/day]");
    same(pm2->state.bhp_limit, pf2->state.bhp_limit, "producer BHP floor [psi]");
  }

  std::printf("\n[6] group injection ceiling (GCONINJE)\n");
  {
    const GroupInject* gm = m->schedule.step(1).group_inject("G", 1);
    const GroupInject* gf = g->schedule.step(1).group_inject("G", 1);
    check(gm && gf, "the water ceiling was read on both sides");
    if (gm && gf) same(gm->target, gf->target, "GCONINJE water RATE [stb/day]");
  }

  std::printf("\n[7] initial state\n");
  check(m->have_initial && g->have_initial, "both were equilibrated");
  if (m->have_initial && g->have_initial &&
      m->initial.pressure.size() == g->initial.pressure.size()) {
    double pm_sum = 0, pf_sum = 0;
    for (size_t i = 0; i < m->initial.pressure.size(); ++i) {
      pm_sum += (double)m->initial.pressure[i];
      pf_sum += (double)g->initial.pressure[i];
    }
    // Equilibration integrates a gradient down a converted depth, so this is
    // looser than the algebraic conversions above.
    same(pm_sum, pf_sum, "summed initial pressure [psi]", 1e-4);
  }

  std::printf("\n[8] idempotence\n");
  {
    // The run converts once, but nothing in the signature says so, and a second
    // application would be silent.
    const double pv_before = m->grid.porv.empty() ? 0.0 : (double)m->grid.porv[0];
    const Well* w0 = well_at(*m, 0, "P1");
    const double q_before = w0 ? (double)w0->state.hist_orat : 0.0;
    to_field_units(*m);
    const double pv_after = m->grid.porv.empty() ? 0.0 : (double)m->grid.porv[0];
    const Well* w1 = well_at(*m, 0, "P1");
    const double q_after = w1 ? (double)w1->state.hist_orat : 0.0;
    same(pv_after, pv_before, "pore volume unchanged by a second call");
    same(q_after, q_before, "well rate unchanged by a second call");
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
