// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_pvt.cpp -- tabulated PVT: reading it, and evaluating it.
//
// The property that matters most here is monotonicity, because it is what
// distinguishes a correctly read table from a plausibly wrong one.  Oil shrinks
// as pressure falls below the bubble point and swells above it; gas expands
// enormously as pressure falls; viscosity moves the other way.  A reader that
// interleaves the Rs curves of PVTO -- the natural mistake, since the file
// looks like a flat list -- produces a table that still interpolates, still
// returns sensible-looking numbers, and has the oil compressing in the wrong
// direction over part of its range.
//
//   g++ -O2 -std=c++17 -I../include test_pvt.cpp ../src/pvt_tables.cpp -o t
//   ./t [path/to/PVT-WET-GAS.INC]

#include "pvt_tables.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace nvrs;

namespace {

int g_failures = 0, g_checks = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) { ++g_failures; std::printf("  FAIL  %s\n", what); }
  else      std::printf("  ok    %s\n", what);
}

void check_near(double got, double want, double tol, const char* what) {
  ++g_checks;
  const double rel = (std::fabs(want) > 0) ? std::fabs(got - want) / std::fabs(want)
                                           : std::fabs(got - want);
  if (rel > tol) {
    ++g_failures;
    std::printf("  FAIL  %s: got %.10g, want %.10g (rel %.3g)\n", what, got, want, rel);
  } else {
    std::printf("  ok    %s = %.10g\n", what, got);
  }
}

std::string tmpdir() {
  const char* t = std::getenv("TMPDIR");
  if (!t) t = std::getenv("TEMP");
  if (!t) t = "/tmp";
  return std::string(t);
}

std::string write_file(const std::string& name, const std::string& body) {
  const std::string p = tmpdir() + "/" + name;
  std::ofstream f(p);
  f << body;
  return p;
}

// ---------------------------------------------------------------------------
void test_shape() {
  std::printf("\n[1] the curve structure of PVTO\n");
  // Two Rs curves, each with an undersaturated branch.  Read as one flat list
  // the pressures would run 50, 75, 100, 70, 95, 120 -- not monotone -- and
  // every interpolation above 70 would mix the two curves.
  const std::string p = write_file("nvrs_pvt1.DATA", R"(
METRIC
PVTO
   20.0   50.0   1.100   1.20
          75.0   1.095   1.25
         100.0   1.090   1.30 /
   40.0   70.0   1.200   1.00
          95.0   1.194   1.05
         120.0   1.188   1.10 /
/
PVTG
   50.0   0.0005  0.0250  0.0144
          0.0000  0.0250  0.0144 /
  100.0   0.0011  0.0125  0.0150
          0.0000  0.0125  0.0150 /
/
PVTW
  250.0  1.020  4.0e-5  0.32  0.0 /
DENSITY
  859.5  1033.0  0.854 /
)");

  PvtTables t;
  t.read(p);
  std::printf("%s", t.summary().c_str());

  check(t.have_pvto(), "PVTO was read");
  check(t.have_pvtg(), "PVTG was read");
  check(t.have_pvtw(), "PVTW was read");
  check(t.have_density(), "DENSITY was read");

  const auto& o = t.pvto(0);
  check(o.size() == 2u, "two Rs curves, not six flat rows");
  if (o.size() == 2) {
    check_near(o[0].rs, 20.0, 1e-12, "first curve Rs");
    check_near(o[1].rs, 40.0, 1e-12, "second curve Rs");
    check(o[0].p.size() == 3u && o[1].p.size() == 3u,
          "each curve kept its three points");
    check_near(o[0].p.front(), 50.0, 1e-12, "first bubble point");
    check_near(o[1].p.front(), 70.0, 1e-12, "second bubble point");
  }

  const auto& g = t.pvtg(0);
  check(g.size() == 2u, "two PVTG pressure nodes");
  if (g.size() == 2) {
    check_near(g[0].p, 50.0, 1e-12, "first gas pressure");
    check(g[0].rv.size() == 2u, "with its vaporised-oil branch");
  }
}

void test_evaluation() {
  std::printf("\n[2] evaluating it\n");
  const std::string p = write_file("nvrs_pvt2.DATA", R"(
METRIC
PVTO
   20.0   50.0   1.100   1.20
          75.0   1.095   1.25
         100.0   1.090   1.30 /
   40.0   70.0   1.200   1.00
          95.0   1.194   1.05
         120.0   1.188   1.10 /
/
PVTW
  250.0  1.020  4.0e-5  0.32  0.0 /
)");
  PvtTables t;
  t.read(p);

  // On the saturated line: at a bubble point, the table's own value.
  Real bo, mu;
  t.oil(50.0, 20.0, bo, mu);
  check_near(bo, 1.100, 1e-12, "Bo at the first bubble point");
  check_near(mu, 1.20, 1e-12, "viscosity there");

  t.oil(70.0, 40.0, bo, mu);
  check_near(bo, 1.200, 1e-12, "Bo at the second bubble point");

  // Saturated Rs rises with pressure.
  check_near(t.rs_sat(50.0), 20.0, 1e-12, "Rs at 50 bar");
  check_near(t.rs_sat(70.0), 40.0, 1e-12, "Rs at 70 bar");
  check(t.rs_sat(60.0) > 20.0 && t.rs_sat(60.0) < 40.0,
        "Rs interpolates between the bubble points");

  // Undersaturated: oil holds its Rs and SHRINKS as pressure rises, because
  // no more gas can dissolve.  This is the direction a flat read gets wrong.
  Real bo_lo, bo_hi, m;
  t.oil(50.0, 20.0, bo_lo, m);
  t.oil(100.0, 20.0, bo_hi, m);
  check(bo_hi < bo_lo,
        "above its bubble point the oil shrinks with pressure");
  std::printf("  note  Bo at Rs=20: %.5f at 50 bar, %.5f at 100 bar\n",
              bo_lo, bo_hi);

  // Water compresses.
  Real bw1, bw2, mw;
  t.water(250.0, bw1, mw);
  t.water(350.0, bw2, mw);
  check_near(bw1, 1.020, 1e-9, "Bw at its reference pressure");
  check(bw2 < bw1, "water shrinks as pressure rises");
}

void test_units() {
  std::printf("\n[3] converting to field units\n");
  const std::string p = write_file("nvrs_pvt3.DATA", R"(
METRIC
PVTO
   20.0   50.0   1.100   1.20 /
/
PVTG
   50.0   0.0005  0.0250  0.0144 /
/
PVTW
  250.0  1.020  4.0e-5  0.32  0.0 /
)");
  PvtTables t;
  t.read(p);
  const Real rs_m = t.pvto(0)[0].rs;
  const Real p_m = t.pvto(0)[0].p.front();
  const Real bg_m = t.pvtg(0)[0].bg.front();

  t.to_field_units();
  check(t.is_field(), "marked as field units");
  check_near(t.pvto(0)[0].p.front(), p_m * 14.503773773, 1e-9,
             "bubble point bar -> psi");
  // Rs: sm3/sm3 -> Mscf/stb.
  check_near(t.pvto(0)[0].rs, rs_m * (35.3146667 / 1000.0) / 6.2898107704,
             1e-9, "Rs sm3/sm3 -> Mscf/stb");
  // Bo is a ratio of like volumes and does not convert.
  check_near(t.pvto(0)[0].bo.front(), 1.100, 1e-12, "Bo is unchanged");
  // Bg: rm3/sm3 -> rb/scf.
  check_near(t.pvtg(0)[0].bg.front(),
             bg_m * 6.2898107704 / 35.3146667, 1e-9, "Bg rm3/sm3 -> rb/scf");
  check_near(t.pvtw(0).cw, 4.0e-5 / 14.503773773, 1e-9, "cw 1/bar -> 1/psi");

  // Idempotent: converting twice must not convert twice.
  const Real once = t.pvto(0)[0].p.front();
  t.to_field_units();
  check_near(t.pvto(0)[0].p.front(), once, 1e-12,
             "converting again changes nothing");
}

void test_ehystr() {
  std::printf("\nEHYSTR configuration\n");
  const std::string p = write_file("nvrs_pvt_ehystr.DATA", R"(
PROPS
EHYSTR
  0.1  0  0.1  1*  KR /
)");
  PvtTables t;
  t.read(p);
  check(t.have_ehystr(), "EHYSTR was read");
  check(t.ehystr_kr_model() == 0,
        "Norne's second EHYSTR item selects Carlson hysteresis");
}

// ---------------------------------------------------------------------------
void test_norne(const std::string& path) {
  std::printf("\n[4] Norne's PVT\n");
  { std::ifstream f(path);
    if (!f) { std::printf("  SKIP  not found: %s\n", path.c_str()); return; } }

  PvtTables t;
  t.read(path);
  std::printf("%s", t.summary().c_str());

  check(t.have_pvto(), "PVTO read");
  check(t.have_pvtg(), "PVTG read");
  check(t.have_density(), "DENSITY read");

  // Norne carries two PVT regions.  Stopping at the first region's terminating
  // slash reads a table that looks complete and silently applies region 1's
  // fluid everywhere -- so assert the count, not just that something was read.
  check(t.n_regions() == 2, "both PVT regions read, not just the first");

  const auto& o = t.pvto(0);
  check(o.size() > 5, "several Rs curves");

  // Rs and bubble point must both increase down the table: that ordering is
  // what the binary search in the device evaluator relies on.
  bool rs_up = true, pb_up = true, branch_ok = true;
  for (size_t i = 1; i < o.size(); ++i) {
    if (!(o[i].rs > o[i - 1].rs)) rs_up = false;
    if (!(o[i].p.front() > o[i - 1].p.front())) pb_up = false;
  }
  for (const PvtoRecord& r : o)
    for (size_t k = 1; k < r.p.size(); ++k)
      if (!(r.p[k] > r.p[k - 1])) branch_ok = false;
  check(rs_up, "Rs increases down the table");
  check(pb_up, "and so does the bubble-point pressure");
  check(branch_ok, "each undersaturated branch rises in pressure");

  // Oil swells with dissolved gas along the saturated line.
  bool bo_up = true;
  for (size_t i = 1; i < o.size(); ++i)
    if (!(o[i].bo.front() > o[i - 1].bo.front())) bo_up = false;
  check(bo_up, "saturated Bo grows with Rs, as dissolved gas swells the oil");

  // Gas expands as pressure falls.  The branch to test this on is the DRY one
  // -- the last entry of each node, where Rv = 0 -- because that is gas alone,
  // and it must be monotone.  The saturated branch (first entry, richest in
  // condensate) need not be: in a wet gas approaching the dew-point envelope
  // the reservoir fluid carries so much condensate that its volume per surface
  // sm3 of gas turns back up.  Norne does exactly this above ~535 bar, and a
  // reader that "fixed" it would be corrupting the table.
  const auto& g = t.pvtg(0);
  bool bg_down = true;
  for (size_t i = 1; i < g.size(); ++i)
    if (!(g[i].bg.back() < g[i - 1].bg.back())) bg_down = false;
  check(bg_down, "dry-gas Bg falls monotonically with pressure");

  size_t turn = 0;
  for (size_t i = 1; i < g.size() && !turn; ++i)
    if (g[i].bg.front() >= g[i - 1].bg.front()) turn = i;
  if (turn)
    std::printf("  note  saturated Bg turns up at %.1f bar -- condensate "
                "loading, as a wet gas does\n", (double)g[turn].p);

  if (g.size() > 1) {
    const double ratio = (double)g.front().bg.back() / g.back().bg.back();
    std::printf("  note  gas expands %.1fx from the highest to the lowest "
                "tabulated pressure\n", ratio);
    check(ratio > 5, "and by a large factor, as a real gas does");
  }

  // Evaluation stays inside the tabulated range.
  t.to_field_units();
  const auto& of = t.pvto(0);
  const Real p_lo = of.front().p.front(), p_hi = of.back().p.front();
  bool sane = true;
  for (int i = 0; i <= 20; ++i) {
    const Real pp = p_lo + (p_hi - p_lo) * i / 20.0;
    const Real rsb = t.rs_sat(pp);
    Real bo, mu;
    t.oil(pp, rsb, bo, mu);
    if (!(bo > 0.5 && bo < 3.0 && mu > 0 && mu < 100)) sane = false;
    Real bg, mg;
    t.gas(pp, -1, bg, mg);
    if (!(bg > 0 && bg < 1 && mg > 0 && mg < 1)) sane = false;
  }
  check(sane, "every evaluation across the range is physical");
  std::printf("  note  at %.0f psi: Rs %.4f Mscf/stb\n", (double)p_hi,
              (double)t.rs_sat(p_hi));
}

// ---------------------------------------------------------------------------
// The device evaluators, run on the host against the same flattened arrays the
// kernel would see.  What is being checked is not the interpolation -- [2] did
// that -- but the pressure derivatives, which are what the Jacobian is built
// from and what a constant-PVT run gets wrong by setting them all to zero.
void test_device_path(const std::string& path) {
  std::printf("\n[5] the device evaluator and its derivatives\n");
  { std::ifstream f(path);
    if (!f) { std::printf("  SKIP  not found\n"); return; } }

  PvtTables t;
  t.read(path);
  t.to_field_units();
  const HostPvt h = flatten_pvt(t, 0);
  const DevicePvt d = h.view();
  check(d.valid, "flattened tables are usable by the device evaluator");

  // Rs must come out of the flattening in the SOLVER's units, not the deck's.
  //
  // A FIELD deck states Rs in Mscf/stb; the gas equation needs scf/stb, since
  // it pairs Rs with bo [stb/rb] against a bg [scf/rb] term.  Getting this
  // wrong is silent -- every curve still interpolates, every value is still
  // positive and monotone, and the only symptom is an oil that carries a
  // thousandth of the gas it should.  So pin it against the number the deck's
  // own RS keyword carries for this field: 674 scf/stb, near enough to the
  // saturated value at Norne's initial 4018 psi.
  const Real rs_init = dev_rs_sat(d, 4018.0);
  check(rs_init > 400 && rs_init < 1200,
        "Rs at initial pressure is scf/stb, not Mscf/stb");
  std::printf("  note  Rs(4018 psi) = %.1f scf/stb; the deck's RS is 674\n",
              (double)rs_init);

  // VAPOIL takes the saturated PVTG branch.  In Norne's reservoir-pressure
  // range that gas compresses normally; above about 535 bar the wet-gas curve
  // legitimately turns as condensate loading grows.  Requiring monotonicity
  // to 9800 psi would reject the deck's real saturated branch and previously
  // caused the device path to substitute dry gas (Rv=0).
  bool cg_pos_reservoir = true;
  Real cg_at_4000 = 0;
  std::printf("   psi      bo       bg      Rs(scf/stb)   c_gas(1/psi)\n");
  for (int i = 1; i <= 14; ++i) {
    const Real p = 700.0 * i;
    const PvtCell c = dev_pvt_cell(d, p, 1e30);
    const Real cg = (c.bg > 0) ? c.dbg / c.bg : 0.0;   // (1/b)(db/dp)
    if (p <= 6000.0 && !(cg > 0)) cg_pos_reservoir = false;
    if (i == 6) cg_at_4000 = cg;
    if (i % 3 == 0 || i == 6)
      std::printf("  %5.0f  %7.4f  %8.2f  %9.1f     %.3e\n", (double)p,
                  (double)c.bo, (double)c.bg, (double)c.rs,
                  (double)cg);
  }
  check(cg_pos_reservoir,
        "wet-gas compressibility is positive over the reservoir range");
  check(cg_at_4000 > 1e-4 && cg_at_4000 < 1e-3,
        "and near 4000 psi is of order 1e-4 /psi, as a real gas is");
  const PvtCell vap = dev_pvt_cell(d, 4018.0, 1e30);
  check(vap.rv > 0 && vap.drv > 0,
        "VAPOIL carries saturated Rv and its pressure derivative");
  check(vap.rs * vap.rv < 1.0,
        "the coupled oil/gas surface conversion has positive determinant");
  std::printf("  note  the deck's lumped CT was 2e-5 /psi -- %.0fx smaller\n",
              (double)(cg_at_4000 / 2e-5));

  // Oil below the bubble point takes up gas as pressure rises, so it swells:
  // b_o falls.  This is the term a constant Bo throws away entirely.
  const PvtCell lo = dev_pvt_cell(d, 2000.0, 1e30);
  check(lo.dbo < 0, "b_o falls with pressure below the bubble point");
  check(lo.drs > 0, "and Rs rises, as more gas goes into solution");

  // The derivative must be the slope of the function the residual evaluates.
  // Comparing against a much coarser difference of the evaluator itself is the
  // check that matters: if they disagree, Newton is stepping on a curve the
  // residual does not lie on.
  //
  // The two will not agree exactly, and should not.  b = 1/B is convex where B
  // is piecewise linear, so a difference taken over a window wide enough to
  // cross table nodes reads slightly low -- 2.4% over +/-60 psi here.  That is
  // curvature, not error.  The tolerance is set to let it through while still
  // catching what this check is for: a derivative wrong by a factor or a sign.
  const Real p0 = 3000.0, H = 25.0;
  const PvtCell a = dev_pvt_cell(d, p0 - H, 1e30);
  const PvtCell b = dev_pvt_cell(d, p0 + H, 1e30);
  const PvtCell m = dev_pvt_cell(d, p0, 1e30);
  check_near((double)m.dbg, (double)(b.bg - a.bg) / (2 * H), 3e-2,
             "dbg/dp agrees with a coarse difference of bg");
  check_near((double)m.drs, (double)(b.rs - a.rs) / (2 * H), 3e-2,
             "drs/dp agrees with a coarse difference of Rs");

  // The undersaturated cap, and the derivative that goes with it.  Above the
  // pressure where the bubble-point line reaches rs_max the oil can take no
  // more gas, so Rs is flat and its slope is zero; below it Rs follows the
  // table.  Both limbs must hold, and so must the kink between them: a cap
  // that leaked would put dRs/dp where there should be none.
  const Real rs_cap = 400.0;                       // scf/stb
  const PvtCell capped   = dev_pvt_cell(d, 6000.0, rs_cap);
  const PvtCell uncapped = dev_pvt_cell(d, 2000.0, rs_cap);
  check_near((double)capped.rs, (double)rs_cap, 1e-9,
             "above the cap the oil holds exactly the Rs it was given");
  check(capped.drs == 0, "and takes up no more, so dRs/dp is zero there");
  check(uncapped.rs < rs_cap && uncapped.drs > 0,
        "below it Rs follows the bubble-point line and still rises");

  // RSVD describes oil below the contact.  It must not cap an initially
  // two-phase gas/oil cell, which instead starts on the saturated PVTO line.
  const Real p_gas = 4018.0;
  const Real low_anchor = 400.0;
  check_near((double)dev_rs_initial(d, p_gas, 0.2, low_anchor),
             (double)dev_rs_sat(d, p_gas), 1e-9,
             "initial free-gas cell ignores a lower RSVD anchor");
  check_near((double)dev_rs_initial(d, p_gas, 0.0, low_anchor),
             (double)low_anchor, 1e-9,
             "initial oil-only cell retains its RSVD anchor");

  // VAPPARS(1): saturated Rv is depleted with (So/So_max)^vap1, using
  // Eclipse's 0.001 floor.  Norne states vap1=0.5.
  const PvtCell rv_full = dev_pvt_cell(d, p_gas, 1e30, 0.5, 0.5, 0.5);
  const PvtCell rv_half = dev_pvt_cell(d, p_gas, 1e30, 0.125, 0.5, 0.5);
  check_near((double)rv_half.rv, 0.5 * (double)rv_full.rv, 1e-12,
             "VAPPARS scales saturated Rv by (So/So_max)^vap1");
  check_near((double)rv_half.drv, 0.5 * (double)rv_full.drv, 1e-12,
             "VAPPARS applies the same factor to dRv/dp");
  const double dh = 1e-6;
  const double rp = dev_pvt_cell(d, p_gas, 1e30, 0.125 + dh, 0.5, 0.5).rv;
  const double rm = dev_pvt_cell(d, p_gas, 1e30, 0.125 - dh, 0.5, 0.5).rv;
  check_near((double)rv_half.drv_dso, (rp-rm)/(2*dh), 1e-8,
             "analytic VAPPARS dRv/dSo matches a centered difference");

  Real bg_sat = 1, mu_sat = 1, bg_rv = 1, mu_rv = 1;
  dev_gas(d, p_gas, bg_sat, mu_sat);
  dev_gas(d, p_gas, dev_rv_sat(d, p_gas), bg_rv, mu_rv);
  check_near((double)bg_rv, (double)bg_sat, 1e-12,
             "2-D PVTG agrees with the saturated curve at saturated Rv");
  check_near((double)mu_rv, (double)mu_sat, 1e-12,
             "wet-gas viscosity is continuous at saturated Rv");
  check(std::abs((double)rv_half.bg - (double)rv_full.bg) > 1e-8,
        "VAPPARS-reduced Rv selects an undersaturated gas FVF");
  check(std::abs((double)rv_half.mug - (double)rv_full.mug) > 1e-8,
        "VAPPARS-reduced Rv selects an undersaturated gas viscosity");
  const PvtCell vp = dev_pvt_cell(d, p_gas, 1e30, 0.125 + dh, 0.5, 0.5);
  const PvtCell vm = dev_pvt_cell(d, p_gas, 1e30, 0.125 - dh, 0.5, 0.5);
  check_near((double)rv_half.dbg_dso,
             ((double)vp.bg - (double)vm.bg) / (2 * dh), 1e-6,
             "VAPPARS d(1/Bg)/dSo matches a centered difference");
  check_near((double)rv_half.dmug_dso,
             ((double)vp.mug - (double)vm.mug) / (2 * dh), 1e-6,
             "VAPPARS d(mu_g)/dSo matches a centered difference");
}

}  // namespace

// ---------------------------------------------------------------------------
// Dead oil and dry gas
// ---------------------------------------------------------------------------
// PVTO and PVTG describe fluids that exchange components. PVDO and PVDG
// describe the same two phases when they do not, and most published decks use
// them: SPE1 and SPE9 both take their gas that way.
//
// They are read into the same structures rather than into a second pair,
// because a fluid that exchanges nothing is the degenerate case of one that
// does. These checks pin that mapping: a dead oil must arrive as one saturated
// node at Rs = 0 carrying the whole pressure range, and a dry gas as one node
// per pressure with a single vaporised-oil ratio of zero.
static void test_dead_oil_and_dry_gas() {
  const std::string p = write_file("pvt_dead.DATA", R"(
RUNSPEC
FIELD
PROPS
PVDO
   4000  1.0500  1.20
   5000  1.0400  1.25
   6000  1.0300  1.30 /
PVDG
   4000  0.00080  0.0200
   5000  0.00065  0.0220
   6000  0.00055  0.0240 /
PVTW
   4000  1.0300  3.0e-6  0.31  0.0 /
ROCK
   14.7  4.0e-6 /
DENSITY
   53.0  64.0  0.0570 /
)");

  PvtTables t;
  t.read(p);

  check(t.have_pvto(), "PVDO was read into the oil tables");
  check(t.have_pvtg(), "PVDG was read into the gas tables");

  const std::vector<PvtoRecord>& o = t.pvto(0);
  check(o.size() == 1u, "a dead oil is one saturated node, not one per row");
  if (o.size() == 1u) {
    check(o[0].rs == 0.0, "that node carries no dissolved gas");
    check(o[0].p.size() == 3u,
          "its undersaturated branch spans every pressure in the table");
    check(std::fabs(o[0].p.front() - 4000.0) < 1e-9 &&
          std::fabs(o[0].p.back() - 6000.0) < 1e-9,
          "the branch runs from the first pressure to the last");
    check(o[0].bo.front() > o[0].bo.back(),
          "oil shrinks as pressure rises, with no gas to come out of it");
    check(std::fabs(o[0].mu.front() - 1.20) < 1e-9,
          "viscosity is taken from the third column");

    const HostPvt h = flatten_pvt(t, 0);
    Real bo = 0, mu = 0;
    dev_oil(h.view(), 4500.0, 0.0, bo, mu);
    const double b = 0.5 / 1.0500 + 0.5 / 1.0400;
    const double bm = 0.5 / (1.0500 * 1.20) +
                      0.5 / (1.0400 * 1.25);
    check_near((double)bo, 1.0 / b, 1e-12,
               "one-curve PVDO interpolates inverse B without overrun");
    check_near((double)mu, b / bm, 1e-12,
               "one-curve PVDO interpolates inverse B-mu consistently");
  }

  // ROCK sits in the same PROPS section and is read into ct, which means rock
  // compressibility alone once tables are present.  It is checked here
  // because the failure mode is silent: a wrong compressibility does not stop
  // a run, it changes the answer.
  check(t.have_rock(), "ROCK was read");
  if (t.have_rock()) {
    check(std::fabs(t.rock_cr() - 4.0e-6) < 1e-12,
          "the compressibility is the second column");
    check(std::fabs(t.rock_pref() - 14.7) < 1e-9,
          "the reference pressure is the first");
    check_near(rock_pv_mult(5014.7, 14.7, 4.0e-6),
               1.0202, 1e-12, "ROCK quadratic pore-volume multiplier");
    check_near(rock_pv_mult_deriv(5014.7, 14.7, 4.0e-6),
               4.08e-6, 1e-12, "ROCK multiplier pressure derivative");
  }

  const std::vector<PvtgRecord>& g = t.pvtg(0);
  check(g.size() == 3u, "a dry gas is one node per pressure row");
  if (g.size() == 3u) {
    bool all_dry = true;
    for (const PvtgRecord& r : g)
      if (r.rv.size() != 1u || r.rv[0] != 0.0) all_dry = false;
    check(all_dry, "every node carries the single vaporised-oil ratio zero");
    check(std::fabs(g.front().p - 4000.0) < 1e-9 &&
          std::fabs(g.back().p - 6000.0) < 1e-9,
          "the nodes run from the first pressure to the last");
    check(g.front().bg.front() > g.back().bg.front(),
          "gas is denser at higher pressure, so Bg falls");
  }
}

static void test_guided_pvto() {
  std::printf("\n[7] PVTO left-extreme guided interpolation\n");
  HostPvt h;
  h.valid = true;
  h.sat_rs = {100, 200};
  h.sat_p = {1000, 2000};
  h.sat_bo = {1.20, 1.40};
  h.sat_mu = {1.00, 2.00};
  h.us_off = {0, 3, 6};
  h.us_p = {1000, 2000, 3000, 2000, 3000, 4000};
  h.us_bo = {1.20, 1.10, 1.00, 1.40, 1.30, 1.20};
  h.us_mu = {1.00, 1.10, 1.20, 2.00, 2.10, 2.20};

  Real bo = 0, mu = 0;
  dev_oil(h.view(), 3000.0, 150.0, bo, mu);
  // At w=0.5 the two branches are sampled at 2500 and 3500 psi, not both at
  // 3000 psi, then inverse B and inverse(B*mu) are interpolated.
  const double bo0 = 1.05, bo1 = 1.25;
  const double mu0 = 1.15, mu1 = 2.15;
  const double b = 0.5 / bo0 + 0.5 / bo1;
  const double bm = 0.5 / (bo0 * mu0) + 0.5 / (bo1 * mu1);
  check_near((double)bo, 1.0 / b, 1e-12,
             "guided PVTO shifts pressure parallel to the bubble-point line");
  check_near((double)mu, b / bm, 1e-12,
             "guided PVTO interpolates inverse B-mu after shifting");
}

int main(int argc, char** argv) {
  test_dead_oil_and_dry_gas();
  test_guided_pvto();
  std::printf("NVRS PVT table tests\n");
  std::printf("====================\n");
  try {
    test_shape();
    test_evaluation();
    test_units();
    test_ehystr();
    if (argc > 1) { test_norne(argv[1]); test_device_path(argv[1]); }
    else std::printf("\n[4] Norne: skipped (pass PVT-WET-GAS.INC to run)\n");
  } catch (const std::exception& e) {
    std::printf("\nEXCEPTION: %s\n", e.what());
    return 2;
  }
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
