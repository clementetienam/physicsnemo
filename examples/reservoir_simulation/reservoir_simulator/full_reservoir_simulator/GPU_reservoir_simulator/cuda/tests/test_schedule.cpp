// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_schedule.cpp -- wells, completions, control switching, and the schedule.
//
//   g++ -O2 -std=c++17 -I../include test_schedule.cpp ../src/schedule.cpp
//       ../src/cpgrid.cpp -o t
//   ./t [path/to/NORNE_ATW2013.DATA]

#include "schedule.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
void test_dates() {
  std::printf("\n[1] the calendar\n");
  // A known pair: 1 Jan 1970 is day 0 of the Unix epoch.
  check_near(days_from_civil(1970, 1, 1), 0.0, 1e-12, "1 JAN 1970 is day 0");
  check_near(days_from_civil(1970, 12, 31), 364.0, 1e-12, "1970 had 365 days");
  // 2000 was a leap year; 1900 was not.
  check_near(days_from_civil(2000, 3, 1) - days_from_civil(2000, 2, 28), 2.0,
             1e-12, "2000 was a leap year");
  check_near(days_from_civil(1900, 3, 1) - days_from_civil(1900, 2, 28), 1.0,
             1e-12, "1900 was not a leap year");
  // Norne's own span: 6 Nov 1997 to 1 Dec 2006.
  check_near(days_from_civil(2006, 12, 1) - days_from_civil(1997, 11, 6),
             3312.0, 1e-12, "Norne's history is 3312 days");
}

void test_peaceman() {
  std::printf("\n[2] the well index, vertical and horizontal\n");
  const double C = 0.0085267014;

  // An isotropic square cell: r0 collapses to 0.28 * sqrt(2) * dx / 2, the
  // textbook value, and the answer must not depend on which way the well runs.
  const double k = 100, d = 20, rw = 0.1;
  double r0 = 0;
  const double wi_z = peaceman_wi(PenetrationDir::Z, k, k, k, d, d, d, rw, 0,
                                  Units::Metric, &r0);
  check_near(r0, 0.28 * std::sqrt(2.0) * d / 2.0, 1e-12,
             "r0 on an isotropic square cell");
  const double want = 2.0 * M_PI * C * k * d / std::log(r0 / rw);
  check_near(wi_z, want, 1e-12, "WI matches the closed form");

  const double wi_x = peaceman_wi(PenetrationDir::X, k, k, k, d, d, d, rw, 0,
                                  Units::Metric);
  const double wi_y = peaceman_wi(PenetrationDir::Y, k, k, k, d, d, d, rw, 0,
                                  Units::Metric);
  check_near(wi_x, wi_z, 1e-12, "direction is irrelevant when the cell is a cube");
  check_near(wi_y, wi_z, 1e-12, "and likewise for Y");

  // Now the case that matters.  A layered reservoir with kv/kh = 0.1 in flat
  // cells: a horizontal well sees a different pair of permeabilities and a
  // different pair of extents than a vertical one, so the two must differ.
  // Getting this wrong is silent -- both numbers look reasonable.
  const double kh = 500, kv = 50, dx = 100, dy = 100, dz = 5;
  const double v = peaceman_wi(PenetrationDir::Z, kh, kh, kv, dx, dy, dz, rw, 0,
                               Units::Metric);
  const double h = peaceman_wi(PenetrationDir::X, kh, kh, kv, dx, dy, dz, rw, 0,
                               Units::Metric);
  check(v > 0 && h > 0, "both orientations give a positive index");
  check(std::fabs(h / v - 1.0) > 0.5,
        "a horizontal completion differs sharply from a vertical one");
  std::printf("  note  vertical WI %.4g, horizontal WI %.4g, ratio %.3f\n",
              v, h, h / v);

  // The horizontal index should be computable by hand from the perpendicular
  // pair (ky, kz) and (dy, dz), with dx as the exposed length.
  const double r12 = std::sqrt(kh / kv), r21 = std::sqrt(kv / kh);
  const double r0h = 0.28 * std::sqrt(r21 * dy * dy + r12 * dz * dz) /
                     (std::pow(kv / kh, 0.25) + std::pow(kh / kv, 0.25));
  const double want_h = 2.0 * M_PI * C * std::sqrt(kh * kv) * dx /
                        std::log(r0h / rw);
  check_near(h, want_h, 1e-12, "horizontal WI uses (ky,kz) and (dy,dz)");

  // Skin reduces productivity; a negative skin (a stimulated well) raises it.
  const double s_pos = peaceman_wi(PenetrationDir::Z, k, k, k, d, d, d, rw, 5.0,
                                   Units::Metric);
  const double s_neg = peaceman_wi(PenetrationDir::Z, k, k, k, d, d, d, rw, -2.0,
                                   Units::Metric);
  check(s_pos < wi_z && s_neg > wi_z, "skin moves the index the right way");
}

void test_control_switching() {
  std::printf("\n[3] control: targets, limits, and which one binds\n");
  const double inf = std::numeric_limits<double>::infinity();

  // An injector on rate with a pressure ceiling.  It holds the rate while the
  // formation takes it, and switches to BHP when it will not.
  WellState inj;
  inj.defined = true;
  inj.open = true;
  inj.type = WellType::Injector;
  inj.mode = CtrlMode::RATE;
  inj.surface_rate = 5000;
  inj.bhp_limit = 350;
  check(inj.effective_mode(5000, 300) == CtrlMode::RATE,
        "injector below its pressure ceiling stays on rate");
  check(inj.effective_mode(5000, 380) == CtrlMode::BHP,
        "injector at its pressure ceiling switches to BHP");
  check_near(inj.target(), 5000, 1e-12, "injector rate target");

  // A producer held on bottom-hole pressure: the common case, and it should
  // stay there whatever rate comes out, absent a ceiling.
  WellState prod;
  prod.defined = true;
  prod.open = true;
  prod.type = WellType::Producer;
  prod.mode = CtrlMode::BHP;
  prod.bhp_limit = 150;
  prod.orat = inf;
  prod.lrat = inf;
  check(prod.effective_mode(1e6, 150) == CtrlMode::BHP,
        "producer on BHP stays on BHP when no rate ceiling was given");

  // The same producer with an oil ceiling: exceeding it moves control to ORAT.
  prod.orat = 2000;
  check(prod.effective_mode(2500, 150) == CtrlMode::ORAT,
        "producer on BHP switches to ORAT when it exceeds the oil ceiling");
  check(prod.effective_mode(1500, 150) == CtrlMode::BHP,
        "and stays on BHP while it does not");

  // A producer on an oil target with a pressure floor -- the other common
  // pairing.  It holds the target until it can no longer sustain it.
  WellState p2;
  p2.defined = true;
  p2.open = true;
  p2.type = WellType::Producer;
  p2.mode = CtrlMode::ORAT;
  p2.orat = 3000;
  p2.bhp_limit = 120;
  check(p2.effective_mode(3000, 200) == CtrlMode::ORAT,
        "producer on ORAT holds its target above the pressure floor");
  check(p2.effective_mode(3000, 110) == CtrlMode::BHP,
        "and drops to BHP when it falls through the floor");
  check_near(p2.target(), 3000, 1e-12, "producer oil target");

  // A shut well is shut regardless.
  WellState shut;
  shut.defined = true;
  shut.open = false;
  check(shut.effective_mode(0, 0) == CtrlMode::Shut, "a closed well stays shut");
}

void test_schedule_parsing() {
  std::printf("\n[4] reading a schedule\n");
  const std::string deck = write_file("nvrs_sched.DATA", R"(
START
  1 'JAN' 2000 /

TUNING
  1.25 12.5 6* /
  5* 0.1 0.0001 0.02 0.02 /
  2* 40 1* 15 /
/

SCHEDULE

WELSPECS
  'PROD1' 'G1'  5  5 1* 'OIL'  7* /
  'INJ1'  'G1'  1  1 1* 'WATER' 7* /
/

COMPDAT
-- vertical producer, three layers
  'PROD1'  5  5  1  3  'OPEN' 1*  12.5  0.216  1*  0.0  1*  'Z'  1* /
-- horizontal injector running along I
  'INJ1'   1  1  2  2  'OPEN' 1*  1*    0.216  1*  0.0  1*  'X'  1* /
  'INJ1'   2  1  2  2  'OPEN' 1*  1*    0.216  1*  0.0  1*  'X'  1* /
/

WCONPROD
  'PROD1' 'OPEN' 'ORAT'  1000  4*  90 /
/

WCONINJE
  'INJ1' 'WATER' 'OPEN' 'RATE'  1500  1*  320 /
/

GCONINJE
  'G1' 'WATER' 'RATE' 42000 /
  'G1' 'GAS'   'RATE' 1000 /
/

DATES
  1 'FEB' 2000 /
  1 'MAR' 2000 /
/

WCONPROD
  'PROD1' 'OPEN' 'BHP'  5*  110 /
/

WELOPEN
  'INJ1' 'SHUT' /
/

DATES
  1 'APR' 2000 /
/

WELOPEN
  'INJ1' 'OPEN' /
/

DATES
  1 'MAY' 2000 /
/
)");

  Schedule s;
  s.read(deck);
  std::printf("%s", s.summary().c_str());

  check(s.n_steps() == 4, "four report steps");
  check(s.well_names().size() == 2u, "two wells");
  check_near(s.start_day(), days_from_civil(2000, 1, 1), 1e-12, "START parsed");
  check_near(s.tuning_initial_step(), 1.25, 1e-12,
             "TUNING TSINIT parsed");
  check_near(s.tuning_max_step(), 12.5, 1e-12,
             "TUNING TSMAXZ parsed");

  if (s.n_steps() >= 3) {
    check_near(s.step(0).day - s.start_day(), 31.0, 1e-12,
               "first step is 31 days after START");
    check_near(s.step(2).day - s.start_day(), 91.0, 1e-12,
               "third step is 91 days after START");
    check(s.step(0).date == "01 FEB 2000", "the date is carried through");

    const Well* p0 = s.step(0).find("PROD1");
    check(p0 && p0->state.mode == CtrlMode::ORAT, "PROD1 starts on ORAT");
    check(p0 && std::fabs(p0->state.orat - 1000) < 1e-12, "with a 1000 target");
    check(p0 && std::fabs(p0->state.bhp_limit - 90) < 1e-12,
          "and a 90 pressure floor");
    check(p0 && p0->completions.size() == 3u,
          "K1..K2 expanded to three completions");
    check(p0 && p0->completions[0].dir == PenetrationDir::Z,
          "the producer is vertical");
    check(p0 && std::fabs(p0->completions[0].cf - 12.5) < 1e-12,
          "the deck's connection factor was kept");

    const Well* i0 = s.step(0).find("INJ1");
    check(i0 && i0->state.type == WellType::Injector, "INJ1 is an injector");
    check(i0 && i0->state.inj_phase == InjectedPhase::Water, "injecting water");
    check(i0 && i0->state.mode == CtrlMode::RATE, "on rate control");
    check(i0 && std::fabs(i0->state.surface_rate - 1500) < 1e-12, "at 1500");
    check(i0 && std::fabs(i0->state.bhp_limit - 320) < 1e-12,
          "with a 320 pressure ceiling");
    check(i0 && i0->completions.size() == 2u, "two completions");
    check(i0 && i0->completions[0].dir == PenetrationDir::X,
          "the injector is horizontal, running along I");
    check(i0 && i0->completions[0].cf < 0,
          "its connection factor was defaulted, to be computed");

    ScheduleStep converted = s.step(0);
    convert_group_inject_targets(converted, 6.2898107704, 35.3146667);
    const GroupInject* water = converted.group_inject("G1", 1);
    const GroupInject* gas = converted.group_inject("G1", 2);
    check_near(water ? water->target : -1.0, 42000.0 * 6.2898107704,
               1e-12, "metric GCONINJE water RATE converts to stb/day");
    check_near(gas ? gas->target : -1.0, 1000.0 * 35.3146667,
               1e-12, "metric GCONINJE gas RATE converts to scf/day");

    // The control change and the shut-in must apply from the third step, and
    // not before: this is the whole point of a time-varying schedule.
    const Well* p2 = s.step(2).find("PROD1");
    check(p2 && p2->state.mode == CtrlMode::BHP,
          "PROD1 switched to BHP control at the later date");
    check(p2 && std::fabs(p2->state.bhp_limit - 110) < 1e-12, "at 110");
    const Well* p1 = s.step(1).find("PROD1");
    check(p1 && p1->state.mode == CtrlMode::ORAT,
          "and was still on ORAT at the step before");

    const Well* i2 = s.step(2).find("INJ1");
    check(i2 && !i2->state.open, "WELOPEN shut the injector");
    check(i2 && i2->state.mode == CtrlMode::RATE,
          "a shut well retains its rate control");
    const Well* i1 = s.step(1).find("INJ1");
    check(i1 && i1->state.open, "and it was open at the step before");
    const Well* i3 = s.step(3).find("INJ1");
    check(i3 && i3->state.open && i3->state.mode == CtrlMode::RATE,
          "WELOPEN reopens the injector on its retained control");
  }
}

void test_history_controls() {
  std::printf("\n[5] OPM-compatible history controls\n");
  const std::string deck = write_file("nvrs_history.DATA", R"(
START
  1 'JAN' 2000 /
SCHEDULE
WELSPECS
  'P1' 'FIELD' 1 1 1* 'OIL' 7* /
/
WCONPROD
  'P1' 'OPEN' 'BHP' 5* 90 /
/
WCONHIST
  'P1' 'OPEN' 'RESV' 1000 20 150000 3* 250 /
/
DATES
  1 'FEB' 2000 /
/
WHISTCTL
  'GRAT' /
WCONHIST
  'P1' 'OPEN' 'RESV' 1100 30 160000 5* /
/
DATES
  1 'MAR' 2000 /
/
WHISTCTL
  'NONE' /
WCONHIST
  'P1' 'OPEN' 'BHP' 1200 40 170000 3* 175 /
/
DATES
  1 'APR' 2000 /
/
)");

  Schedule s;
  s.read(deck);
  check(s.n_steps() == 3, "three history-control snapshots");
  if (s.n_steps() != 3) return;

  const Well* p0 = s.step(0).find("P1");
  check(p0 && p0->state.historical, "WCONHIST enters history mode");
  check(p0 && p0->state.mode == CtrlMode::RESV,
        "the WCONHIST CMODE is preserved");
  check(p0 && p0->state.hist_orat == 1000 &&
              p0->state.hist_wrat == 20 &&
              p0->state.hist_grat == 150000,
        "all three observed phase rates are retained");
  check(p0 && p0->state.bhp_limit == 0,
        "observed BHP is not a RESV well's pressure limit");

  const Well* p1 = s.step(1).find("P1");
  check(p1 && p1->state.mode == CtrlMode::GRAT,
        "WHISTCTL overrides the WCONHIST CMODE");

  const Well* p2 = s.step(2).find("P1");
  check(p2 && p2->state.mode == CtrlMode::BHP,
        "WHISTCTL NONE restores the record CMODE");
  check(p2 && p2->state.bhp_limit == 175,
        "observed BHP is the target when CMODE is BHP");
}

void test_shut_status_retains_control() {
  std::printf("\n[6] shut status retains the last control\n");
  const std::string deck = write_file("nvrs_reopen.DATA", R"(
START
  1 'JAN' 2000 /
SCHEDULE
WELSPECS
  'PH' 'FIELD' 1 1 1* 'OIL'   7* /
  'PP' 'FIELD' 1 1 1* 'OIL'   7* /
  'IW' 'FIELD' 1 1 1* 'WATER' 7* /
/
WCONHIST
  'PH' 'SHUT' 'RESV' 1000 20 150000 /
/
WCONPROD
  'PP' 'SHUT' 'ORAT' 900 /
/
WCONINJE
  'IW' 'WATER' 'SHUT' 'RATE' 1500 1* 320 /
/
DATES
  1 'FEB' 2000 /
/
WELOPEN
  'PH' 'OPEN' /
  'PP' 'OPEN' /
  'IW' 'OPEN' /
/
DATES
  1 'MAR' 2000 /
/
)");

  Schedule s;
  s.read(deck);
  check(s.n_steps() == 2, "two reopen snapshots");
  if (s.n_steps() != 2) return;
  const Well* ph0 = s.step(0).find("PH");
  const Well* pp0 = s.step(0).find("PP");
  const Well* iw0 = s.step(0).find("IW");
  check(ph0 && !ph0->state.open && ph0->state.mode == CtrlMode::RESV,
        "shut WCONHIST retains RESV");
  check(pp0 && !pp0->state.open && pp0->state.mode == CtrlMode::ORAT,
        "shut WCONPROD retains ORAT");
  check(iw0 && !iw0->state.open && iw0->state.mode == CtrlMode::RATE,
        "shut WCONINJE retains RATE");
  const Well* ph1 = s.step(1).find("PH");
  const Well* pp1 = s.step(1).find("PP");
  const Well* iw1 = s.step(1).find("IW");
  check(ph1 && ph1->state.open && ph1->state.mode == CtrlMode::RESV,
        "WELOPEN resumes RESV history control");
  check(pp1 && pp1->state.open && pp1->state.mode == CtrlMode::ORAT,
        "WELOPEN resumes ORAT prediction control");
  check(iw1 && iw1->state.open && iw1->state.mode == CtrlMode::RATE,
        "WELOPEN resumes injector RATE control");
}

void test_completion_squeeze_and_reperforation() {
  std::printf("\n[6] explicit-zero WELOPEN completion squeeze\n");
  const std::string deck = write_file("nvrs_squeeze.DATA", R"(
START
  1 'JAN' 2000 /
SCHEDULE
WELSPECS
  'P1' 'FIELD' 1 1 1* 'OIL' 7* /
/
COMPDAT
  'P1' 1 1 1 3 'OPEN' 1* 10 /
/
WCONHIST
  'P1' 'OPEN' 'RESV' 1000 0 100000 /
/
DATES
  1 'FEB' 2000 /
/
WELOPEN
  'P1' 'SHUT' 0 0 0 2* /
/
COMPDAT
  'P1' 1 1 2 2 'OPEN' 1* 20 /
/
WCONHIST
  'P1' 'OPEN' 'RESV' 900 0 90000 /
/
DATES
  1 'MAR' 2000 /
/
)");
  Schedule s;
  s.read(deck);
  check(s.n_steps() == 2, "two squeeze snapshots");
  if (s.n_steps() != 2) return;
  const Well* before = s.step(0).find("P1");
  const Well* after = s.step(1).find("P1");
  check(before && before->state.open, "well is open before squeeze");
  check(before && before->completions.size() == 3u,
        "three completions before squeeze");
  check(after && after->state.open, "WCONHIST reopens the well");
  check(after && after->completions.size() == 3u,
        "squeeze retains completion identities");
  if (after && after->completions.size() == 3u) {
    check(!after->completions[0].open && after->completions[1].open &&
              !after->completions[2].open,
          "only the COMPDAT-reperforated cell reopens");
    check_near(after->completions[1].cf, 20.0, 1e-12,
               "reperforation replaces its connection factor");
  }
}

void test_resolve_against_grid() {
  std::printf("\n[6] resolving completions onto a grid\n");
  // A small box, with one cell deactivated under the producer so that the
  // orphan path is exercised rather than assumed.
  const std::string grd = write_file("nvrs_sw.grdecl", [] {
    std::ostringstream f;
    const int nx = 5, ny = 5, nz = 3;
    const double dx = 20, dy = 20, dz = 4, top = 2000;
    f << "SPECGRID\n " << nx << " " << ny << " " << nz << " 1 F /\n\nCOORD\n";
    for (int j = 0; j <= ny; ++j)
      for (int i = 0; i <= nx; ++i)
        f << " " << i * dx << " " << j * dy << " " << (top - 100) << " "
          << i * dx << " " << j * dy << " " << (top + nz * dz + 100) << "\n";
    f << "/\n\nZCORN\n";
    for (int k = 0; k < nz; ++k)
      for (int h = 0; h < 2; ++h)
        for (int j = 0; j < ny; ++j)
          for (int dup = 0; dup < 2; ++dup)
            for (int i = 0; i < nx; ++i)
              for (int d = 0; d < 2; ++d) f << " " << (top + (k + h) * dz);
    f << "\n/\n\nPORO\n " << (nx * ny * nz) << "*0.25 /\n\n";
    f << "PERMX\n " << (nx * ny * nz) << "*500 /\n\n";
    f << "PERMY\n " << (nx * ny * nz) << "*500 /\n\n";
    f << "PERMZ\n " << (nx * ny * nz) << "*50 /\n\n";
    f << "NTG\n " << (nx * ny * nz) << "*1 /\n\n";
    // Kill the middle layer under (5,5) -- 0-based (4,4,1).
    f << "ACTNUM\n";
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          f << " " << ((i == 4 && j == 4 && k == 1) ? 0 : 1);
    f << "\n/\n";
    return f.str();
  }());

  CornerPointGrid g;
  g.read_grdecl(grd);
  g.build();
  check(g.n_active() == 74, "74 active cells (one killed)");

  const std::string deck = write_file("nvrs_sw.DATA", R"(
START
  1 'JAN' 2000 /
SCHEDULE
WELSPECS
  'P1' 'G' 5 5 1* 'OIL' 7* /
  'I1' 'G' 1 1 1* 'WATER' 7* /
/
COMPDAT
  'P1'  5  5  1  3  'OPEN' 1* 1* 0.2 1* 0.0 1* 'Z' 1* /
  'I1'  1  1  2  2  'OPEN' 1* 1* 0.2 1* 0.0 1* 'X' 1* /
/
WCONPROD
  'P1' 'OPEN' 'BHP' 5* 150 /
/
WCONINJE
  'I1' 'WATER' 'OPEN' 'RATE' 800 1* 400 /
/
DATES
  1 'FEB' 2000 /
/
)");

  Schedule s;
  s.read(deck);
  s.resolve(g);
  std::printf("%s", s.summary().c_str());

  const Well* p = s.step(0).find("P1");
  check(p && p->completions.size() == 3u, "the producer has three completions");
  int live = 0;
  for (const Completion& c : p->completions) if (c.cell >= 0) ++live;
  check(live == 2, "one landed in the dead cell and was not resolved");
  check(s.orphan_completions().size() == 1u, "and it was reported, not dropped");

  for (const Completion& c : p->completions)
    if (c.cell >= 0)
      check(c.wi > 0, "a resolved completion got a positive well index");

  // The horizontal injector must get a different index than it would as a
  // vertical well in the same cell, because kz differs from kx here.
  const Well* i = s.step(0).find("I1");
  check(i && i->completions.size() == 1u, "the injector has one completion");
  if (i && !i->completions.empty() && i->completions[0].cell >= 0) {
    const int a = i->completions[0].cell;
    const double as_vert =
        peaceman_wi(PenetrationDir::Z, g.permx[a], g.permy[a], g.permz[a],
                    g.dx[a], g.dy[a], g.dz[a], 0.1, 0.0, Units::Metric);
    std::printf("  note  injector WI %.4g horizontal, %.4g if taken as vertical\n",
                i->completions[0].wi, as_vert);
    check(std::fabs(i->completions[0].wi - as_vert) > 1e-6 * as_vert,
          "the horizontal index is not the vertical one");
  }
}

// ---------------------------------------------------------------------------
void test_norne(const std::string& deck) {
  std::printf("\n[6] Norne's schedule\n");
  {
    std::ifstream t(deck);
    if (!t) { std::printf("  SKIP  deck not found\n"); return; }
  }

  Schedule s;
  s.read(deck);
  std::printf("%s", s.summary().c_str());

  check(s.n_steps() == 247, "247 report steps");
  check_near(s.start_day(), days_from_civil(1997, 11, 6), 1e-12,
             "START is 6 NOV 1997");
  check(s.well_names().size() >= 30, "at least 30 wells appear");

  if (s.n_steps() > 0) {
    const ScheduleStep& last = s.steps().back();
    check(last.date == "01 DEC 2006", "the history ends 1 DEC 2006");

    int prod = 0, inj = 0, horiz = 0, vert = 0, comps = 0;
    for (const Well& w : last.wells) {
      if (w.state.type == WellType::Injector) ++inj; else ++prod;
      bool h = false;
      for (const Completion& c : w.completions) {
        ++comps;
        if (c.dir != PenetrationDir::Z) h = true;
      }
      if (h) ++horiz; else ++vert;
    }
    check(prod > 0 && inj > 0, "both producers and injectors were read");
    check(horiz > 0, "some wells have horizontal sections");
    check(vert > 0, "and some are purely vertical");
    check(comps > 100, "the field has a substantial completion list");

    // C-4H is declared vertical ('Z') and B-2H horizontal ('X') in the deck;
    // these are the two the reader must not confuse.
    const Well* c4 = last.find("C-4H");
    const Well* b2 = last.find("B-2H");
    if (c4 && !c4->completions.empty())
      check(c4->completions[0].dir == PenetrationDir::Z, "C-4H is vertical");
    if (b2 && !b2->completions.empty()) {
      bool any_x = false;
      for (const Completion& c : b2->completions)
        if (c.dir == PenetrationDir::X) any_x = true;
      check(any_x, "B-2H has completions running along I");
    }
  }

  // Every step must be ordered in time and carry a date.
  bool ordered = true, dated = true;
  for (int n = 1; n < s.n_steps(); ++n) {
    if (!(s.step(n).day >= s.step(n - 1).day)) ordered = false;
    if (s.step(n).date.empty()) dated = false;
  }
  check(ordered, "report steps are monotone in time");
  check(dated, "every step carries a date");

  // After the day-202 snapshot the deck squeezes every D-4H completion with
  // explicit `0 0 0`; it then reperforates only seven oil-zone cells before
  // the WCONHIST state stored at day 229 reopens the well.
  // Retaining the old deep water-zone connections inverts its phase split.
  const Well* d4_229 = nullptr;
  for (const ScheduleStep& st : s.steps())
    if (std::fabs((st.day - s.start_day()) - 229.0) < 1e-9)
      d4_229 = st.find("D-4H");
  int d4_open = 0;
  bool d4_deep_open = false;
  if (d4_229)
    for (const Completion& c : d4_229->completions) {
      if (c.open) ++d4_open;
      if (c.open && c.k >= 14) d4_deep_open = true;  // Eclipse K >= 15
    }
  check(d4_229 && d4_229->state.open, "D-4H reopens on day 229");
  check(d4_open == 7, "D-4H reopens only seven reperforated cells");
  check(!d4_deep_open, "D-4H squeezed water-zone completions stay shut");

  // The integration that matters: the wells the schedule declares have to land
  // in cells the grid actually kept.  A completion index that is right in the
  // deck's numbering and wrong in the simulator's is the classic way a field
  // model runs and produces from the wrong place.
  {
    CornerPointGrid g;
    g.read_grdecl(deck);
    g.build();
    s.resolve(g);
    std::printf("%s", s.summary().c_str());

    const ScheduleStep& last = s.steps().back();
    int total = 0, live = 0;
    bool in_range = true, wi_ok = true;
    for (const Well& w : last.wells)
      for (const Completion& c : w.completions) {
        ++total;
        if (c.cell < 0) continue;
        ++live;
        if (c.cell >= g.n_active()) in_range = false;
        if (!(c.wi > 0)) wi_ok = false;
      }
    check(total > 0 && live > 0, "completions resolved against the Norne grid");
    check(in_range, "every resolved completion indexes a real active cell");
    check(wi_ok, "every resolved completion has a positive well index");
    check(live > (int)(0.9 * total),
          "the large majority of completions land in live cells");
    std::printf("  note  %d of %d completions resolved; %d lost to "
                "ACTNUM/MINPV\n", live, total, total - live);

    // Every well should keep at least one live completion, or it cannot flow.
    int dead_wells = 0;
    for (const Well& w : last.wells) {
      if (w.completions.empty()) continue;
      bool any = false;
      for (const Completion& c : w.completions) if (c.cell >= 0) any = true;
      if (!any) ++dead_wells;
    }
    check(dead_wells == 0, "no well lost all of its completions");
  }

  // Control modes actually vary across the history -- that is the capability
  // this module exists for, so it is worth asserting rather than assuming.
  std::set<std::string> modes;
  int switches = 0;
  for (int n = 1; n < s.n_steps(); ++n)
    for (const Well& w : s.step(n).wells) {
      modes.insert(to_string(w.state.mode));
      const Well* prev = s.step(n - 1).find(w.name);
      if (prev && prev->state.mode != w.state.mode) ++switches;
    }
  check(modes.size() >= 3, "at least three distinct control modes are used");

  // Norne first states DRSDT 0, then VAPPARS 0.5 0.0 before the first date.
  // Eclipse/OPM define these controls as mutually exclusive: the later
  // VAPPARS record cancels DRSDT and controls Rv through So/So_max.
  check(!s.have_drsdt(), "Norne's later VAPPARS cancels its earlier DRSDT");
  check(s.have_vappars(), "VAPPARS was read from the schedule include");
  check_near((double)s.vap1(), 0.5, 1e-12,
             "Norne Rv depletion exponent is 0.5");
  check_near((double)s.vap2(), 0.0, 1e-12,
             "Norne Rs depletion exponent is disabled");
  {
    // And it must no longer be announced as unhandled, or the run banner
    // keeps claiming a limitation that is not there any more.
    bool listed = false;
    for (const std::string& k : s.ignored_keywords())
      if (k == "DRSDT" || k == "VAPPARS") listed = true;
    check(!listed, "DRSDT/VAPPARS are no longer reported as not implemented");
  }
  check(switches > 0, "wells change control mode during the history");
  std::printf("  note  %d control changes across the history; modes seen:", switches);
  for (const std::string& m : modes) std::printf(" %s", m.c_str());
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string deck = (argc > 1) ? argv[1] : "";

  std::printf("NVRS schedule and well tests\n");
  std::printf("============================\n");
  try {
    test_dates();
    test_peaceman();
    test_control_switching();
    test_schedule_parsing();
    test_history_controls();
    test_shut_status_retains_control();
    test_completion_squeeze_and_reperforation();
    test_resolve_against_grid();
    if (!deck.empty()) test_norne(deck);
    else std::printf("\n[7] Norne: skipped (pass the .DATA deck to run)\n");
  } catch (const std::exception& e) {
    std::printf("\nEXCEPTION: %s\n", e.what());
    return 2;
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
