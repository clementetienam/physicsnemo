// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_equil.cpp -- datum initialisation, contacts, and regions.
//
//   g++ -O2 -std=c++17 -I../include test_equil.cpp ../src/equil.cpp
//       ../src/cpgrid.cpp -o t
//   ./t [path/to/NORNE_ATW2013.DATA]

#include "equil.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
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

// A layered box spanning a depth range wide enough to hold both contacts.
std::string write_column(const std::string& name, int nz, double top,
                         double dz, int nregions) {
  std::ostringstream f;
  const int nx = 2, ny = 2;
  const double dx = 50, dy = 50;
  f << "SPECGRID\n " << nx << " " << ny << " " << nz << " 1 F /\n\nCOORD\n";
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i)
      f << " " << i * dx << " " << j * dy << " " << (top - 100) << " "
        << i * dx << " " << j * dy << " " << (top + nz * dz + 100) << "\n";
  f << "/\n\nZCORN\n";
  for (int k = 0; k < nz; ++k)
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < ny; ++j)
        for (int d2 = 0; d2 < 2; ++d2)
          for (int i = 0; i < nx; ++i)
            for (int d = 0; d < 2; ++d) f << " " << (top + (k + h) * dz);
  f << "\n/\n\n";
  const int ng = nx * ny * nz;
  f << "PORO\n " << ng << "*0.25 /\n\nPERMX\n " << ng << "*300 /\n\n";
  f << "PERMY\n " << ng << "*300 /\n\nPERMZ\n " << ng << "*30 /\n\n";
  f << "NTG\n " << ng << "*1 /\n\n";
  if (nregions > 1) {
    // Split by I so that both regions span the full depth range.
    f << "EQLNUM\n";
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          f << " " << ((i < nx / 2) ? 1 : 2);
    f << "\n/\n\n";
  }
  return write_file(name, f.str());
}

// ---------------------------------------------------------------------------
void test_gradients() {
  std::printf("\n[1] gradients from density\n");
  const std::string p = write_file("nvrs_eq1.DATA", R"(
METRIC
EQUIL
  2500  250  2600  0.0  2400  0.0  1 0 0 /
DENSITY
  859.5  1033.0  0.854 /
)");
  Equilibration e;
  e.read(p);
  check(e.n_regions() == 1, "one EQUIL region");
  check(!e.have_gradients(), "no gradients until the caller supplies them");
  check_near(e.density().water, 1033.0, 1e-12, "water surface density");

  // With unit formation volume factors the gradient is just rho*g.
  e.set_fvf(1.0, 1.0, 1.0);
  check_near(e.water_gradient(), 1033.0 * 9.80665e-5, 1e-12,
             "water gradient at Bw = 1");

  // Norne's own numbers: the deck documents 0.1001 bar/m for water and
  // 0.0706 for oil.  Reproducing them is a check that the density-to-gradient
  // conversion, including the dissolved gas riding with the oil, is right.
  e.set_fvf(/*Bo*/ 1.336, /*Bw*/ 1.0113, /*Bg*/ 0.00456, /*Rs*/ 120.0);
  check_near(e.water_gradient(), 0.1001, 2e-3, "water gradient matches the deck");
  check_near(e.oil_gradient(), 0.0706, 2e-2, "oil gradient matches the deck");
  std::printf("  note  oil %.4f, water %.4f, gas %.4f bar/m\n",
              e.oil_gradient(), e.water_gradient(), e.gas_gradient());

  // A caller may also state them outright.
  e.set_gradients(0.0706, 0.1001, 0.0186);
  check_near(e.oil_gradient(), 0.0706, 1e-12, "gradients can be set directly");
}

void test_datum_and_contacts() {
  std::printf("\n[2] the datum is honoured, and the contacts are consistent\n");
  const std::string p = write_file("nvrs_eq2.DATA", R"(
METRIC
EQUIL
  2500  250  2600  0.0  2400  0.0  1 0 0 /
DENSITY
  859.5  1033.0  0.854 /
)");
  Equilibration e;
  e.read(p);
  e.set_gradients(0.0706, 0.1001, 0.0186);

  // The defining property: at the datum, the oil pressure is the datum
  // pressure exactly.  Everything else is measured from here.
  check_near(e.oil_pressure(0, 2500.0), 250.0, 1e-12,
             "oil pressure at the datum equals the datum pressure");

  // The column is linear at the stated gradient.
  check_near(e.oil_pressure(0, 2600.0) - e.oil_pressure(0, 2500.0),
             0.0706 * 100.0, 1e-12, "oil column rises at its gradient");

  // At each contact, the phase pressures must differ by exactly the stated
  // capillary pressure.  This is what makes the state an equilibrium, and it
  // is the check that catches anchoring the columns in the wrong place.
  check_near(e.water_pressure(0, 2600.0) - e.oil_pressure(0, 2600.0), 0.0,
             1e-12, "Pw = Po at the WOC when Pc is zero");
  check_near(e.gas_pressure(0, 2400.0) - e.oil_pressure(0, 2400.0), 0.0,
             1e-12, "Pg = Po at the GOC when Pc is zero");

  // With a non-zero capillary pressure the offset must appear at the contact
  // and nowhere else.
  const std::string p2 = write_file("nvrs_eq3.DATA", R"(
METRIC
EQUIL
  2500  250  2600  1.5  2400  0.8  1 0 0 /
DENSITY
  859.5  1033.0  0.854 /
)");
  Equilibration e2;
  e2.read(p2);
  e2.set_gradients(0.0706, 0.1001, 0.0186);
  check_near(e2.water_pressure(0, 2600.0) - e2.oil_pressure(0, 2600.0), 1.5,
             1e-12, "Pc at the WOC is exactly as stated");
  check_near(e2.gas_pressure(0, 2400.0) - e2.oil_pressure(0, 2400.0), 0.8,
             1e-12, "Pc at the GOC is exactly as stated");
  // Away from the contact the difference must change, at the difference of
  // the two gradients -- a constant offset everywhere would mean the water
  // column had been given the oil's gradient.
  const double d1 = e2.water_pressure(0, 2700.0) - e2.oil_pressure(0, 2700.0);
  check_near(d1 - 1.5, (0.1001 - 0.0706) * 100.0, 1e-12,
             "below the WOC the phases diverge at the gradient difference");
}

void test_initialise() {
  std::printf("\n[3] initialising a column\n");
  // 40 layers of 5 m from 2400 m: 2400..2600, with the GOC at 2450 and the
  // WOC at 2550, so all three phases are present.
  const std::string grd = write_column("nvrs_eq.grdecl", 40, 2400, 5, 1);
  CornerPointGrid g;
  g.read_grdecl(grd);
  g.build();

  const std::string p = write_file("nvrs_eq4.DATA", R"(
METRIC
EQUIL
  2500  250  2550  0.0  2450  0.0  1 0 0 /
DENSITY
  859.5  1033.0  0.854 /
)");
  Equilibration e;
  e.read(p);
  e.set_gradients(0.0706, 0.1001, 0.0186);

  const InitialState s = e.initialise(g);
  check((int)s.pressure.size() == g.n_active(), "one entry per active cell");

  int gas = 0, oil = 0, wat = 0;
  bool phases_ok = true, sum_ok = true, order_ok = true;
  for (int a = 0; a < g.n_active(); ++a) {
    const double z = g.depth[a];
    if (s.sg[a] > 0) { ++gas; if (z >= 2450) phases_ok = false; }
    else if (s.so[a] > 0) { ++oil; if (z < 2450 || z >= 2550) phases_ok = false; }
    else { ++wat; if (z < 2550) phases_ok = false; }
    if (std::fabs(s.sw[a] + s.so[a] + s.sg[a] - 1.0) > 1e-12) sum_ok = false;
    // Pressure must increase downwards everywhere.
    if (s.pressure[a] <= 0) order_ok = false;
  }
  check(gas > 0 && oil > 0 && wat > 0, "all three phase zones are populated");
  check(phases_ok, "every cell is in the zone its depth puts it in");
  check(sum_ok, "saturations sum to one");
  check(order_ok, "pressures are positive");
  std::printf("  note  %d gas cells, %d oil, %d water\n", gas, oil, wat);

  // The cell nearest the datum must sit on the datum's own column: its
  // pressure is the datum pressure offset by the gradient over the signed
  // depth difference.  The sign matters -- the nearest cell here is above the
  // datum, so the offset subtracts.
  double best = 1e30, pbest = 0, zbest = 0;
  for (int a = 0; a < g.n_active(); ++a)
    if (std::fabs(g.depth[a] - 2500.0) < best) {
      best = std::fabs(g.depth[a] - 2500.0);
      pbest = s.pressure[a];
      zbest = g.depth[a];
    }
  check_near(pbest, 250.0 + 0.0706 * (zbest - 2500.0), 1e-12,
             "the cell nearest the datum sits on the datum's column");
  std::printf("  note  nearest cell at %.2f m carries %.4f bar\n", zbest, pbest);

  // Refusing to guess is a feature: an un-gradiented Equilibration must throw
  // rather than silently initialise a field with a zero gradient.
  Equilibration e2;
  e2.read(p);
  bool threw = false;
  try { e2.initialise(g); } catch (const std::exception&) { threw = true; }
  check(threw, "initialise() refuses to run without gradients");
}

void test_regions() {
  std::printf("\n[4] two equilibration regions\n");
  const std::string grd = write_column("nvrs_eq2.grdecl", 40, 2400, 5, 2);
  CornerPointGrid g;
  g.read_grdecl(grd);
  g.build();
  check(g.has_array("EQLNUM"), "EQLNUM was read from the grid file");

  // Same contacts in both regions, different datum pressures.  Keeping the
  // contacts equal is deliberate: it isolates what this test is for, which is
  // that a cell takes its own region's datum. Different contacts would also
  // change which phase column a cell sits in, and the pressure difference
  // would then mix two effects.
  const std::string p = write_file("nvrs_eq5.DATA", R"(
METRIC
EQUIL
  2500  250  2550  0.0  2450  0.0  1 0 0 /
  2500  300  2550  0.0  2450  0.0  2 0 0 /
DENSITY
  859.5  1033.0  0.854 /
)");
  Equilibration e;
  e.read(p);
  e.set_gradients(0.0706, 0.1001, 0.0186);
  check(e.n_regions() == 2, "two EQUIL records");

  const InitialState s = e.initialise(g);

  // Cells must take the datum pressure of their own region: at equal depth,
  // region 2 sits 50 bar above region 1.  A reader that ignored EQLNUM would
  // give both the same pressure and look entirely plausible.
  double p1 = 0, p2 = 0;
  int n1 = 0, n2 = 0;
  for (int a = 0; a < g.n_active(); ++a) {
    if (std::fabs(g.depth[a] - 2502.5) > 2.6) continue;
    if (s.region[a] == 1) { p1 += s.pressure[a]; ++n1; }
    else                  { p2 += s.pressure[a]; ++n2; }
  }
  check(n1 > 0 && n2 > 0, "both regions have cells at the sample depth");
  if (n1 && n2) {
    p1 /= n1; p2 /= n2;
    check_near(p2 - p1, 50.0, 1e-9,
               "the two regions differ by their datum pressures");
    std::printf("  note  at 2502.5 m: region 1 %.3f bar, region 2 %.3f bar\n", p1, p2);
  }

  // The pressure profile must bend at each contact, because below the water
  // contact it is the water column that is hydrostatic and above the gas
  // contact it is the gas column. A single straight line through all three
  // legs is the mistake this checks for, and it is not visible from a plot of
  // the field -- it looks like a perfectly ordinary pressure gradient.
  {
    auto slope_between = [&](double za, double zb) {
      // Least-squares slope over the cells in a depth window of region 1.
      double n = 0, sx = 0, sy = 0, sxy = 0, sxx = 0;
      for (int a = 0; a < g.n_active(); ++a) {
        if (s.region[a] != 1) continue;
        const double z = g.depth[a];
        if (z < za || z > zb) continue;
        const double y = s.pressure[a];
        ++n; sx += z; sy += y; sxy += z * y; sxx += z * z;
      }
      return (n > 1) ? (n * sxy - sx * sy) / (n * sxx - sx * sx) : 0.0;
    };
    const double gas_leg = slope_between(2405, 2445);
    const double oil_leg = slope_between(2455, 2545);
    const double wat_leg = slope_between(2555, 2595);
    std::printf("  note  slopes: gas %.4f, oil %.4f, water %.4f bar/m\n",
                gas_leg, oil_leg, wat_leg);
    check_near(gas_leg, 0.0186, 1e-6, "the gas leg rises at the gas gradient");
    check_near(oil_leg, 0.0706, 1e-6, "the oil leg rises at the oil gradient");
    check_near(wat_leg, 0.1001, 1e-6, "the water leg rises at the water gradient");
  }
}

// ---------------------------------------------------------------------------
void test_norne(const std::string& deck) {
  std::printf("\n[5] Norne\n");
  {
    std::ifstream t(deck);
    if (!t) { std::printf("  SKIP  deck not found\n"); return; }
  }

  Equilibration e;
  e.read(deck);
  check(e.n_regions() == 5, "five equilibration regions");
  check(e.density().water > 0, "DENSITY was found in the PVT include");

  // The deck's own header states the gradients it was built with.
  e.set_gradients(0.0706, 0.1001, 0.0186);
  std::printf("%s", e.summary().c_str());

  if (e.n_regions() >= 5) {
    // Region 1: datum 2582.0 m at 268.56 bar, WOC 2692.0, GOC 2582.0.
    const EquilRecord& r1 = e.record(0);
    check_near(r1.datum_depth, 2582.0, 1e-12, "region 1 datum depth");
    check_near(r1.datum_pressure, 268.56, 1e-12, "region 1 datum pressure");
    check_near(r1.owc_depth, 2692.0, 1e-12, "region 1 WOC");
    check_near(r1.goc_depth, 2582.0, 1e-12, "region 1 GOC");

    // Region 4 is the shallow one: datum 2200.0 at 236.92 bar.
    const EquilRecord& r4 = e.record(3);
    check_near(r4.datum_depth, 2200.0, 1e-12, "region 4 datum depth");
    check_near(r4.datum_pressure, 236.92, 1e-12, "region 4 datum pressure");
    check_near(r4.owc_depth, 2400.0, 1e-12, "region 4 WOC");

    // The datums span nearly 400 m, which is why per-region initialisation
    // matters here rather than being a refinement.
    double zmin = 1e30, zmax = -1e30;
    for (const EquilRecord& r : e.records()) {
      zmin = std::min(zmin, (double)r.datum_depth);
      zmax = std::max(zmax, (double)r.datum_depth);
    }
    check(zmax - zmin > 300.0, "the region datums span more than 300 m");
    std::printf("  note  datums span %.1f m, from %.1f to %.1f\n",
                zmax - zmin, zmin, zmax);
  }

  // Initialise the real field.
  CornerPointGrid g;
  g.read_grdecl(deck);
  g.build();
  const InitialState s = e.initialise(g);

  check((int)s.pressure.size() == g.n_active(), "initialised every active cell");

  std::set<int> regions;
  int gas = 0, oil = 0, wat = 0;
  double pmin = 1e30, pmax = -1e30;
  bool sane = true;
  for (int a = 0; a < g.n_active(); ++a) {
    regions.insert(s.region[a]);
    if (s.sg[a] > 0) ++gas; else if (s.so[a] > 0) ++oil; else ++wat;
    pmin = std::min(pmin, (double)s.pressure[a]);
    pmax = std::max(pmax, (double)s.pressure[a]);
    if (!(s.pressure[a] > 100 && s.pressure[a] < 400)) sane = false;
  }
  check(regions.size() >= 2, "cells fall in more than one region");
  check(oil > 0 && wat > 0, "the field initialises with both oil and water");
  check(sane, "every initial pressure is in a plausible range for Norne");
  std::printf("  note  regions used %d, pressure %.2f .. %.2f bar\n",
              (int)regions.size(), pmin, pmax);
  std::printf("  note  %d gas cells, %d oil, %d water\n", gas, oil, wat);
}

}  // namespace

// ---------------------------------------------------------------------------
// RSVD -- dissolved gas against depth, per equilibration region
// ---------------------------------------------------------------------------
// Rs is not a primary unknown here, so the model needs an anchor for the gas an
// undersaturated oil retains.  A deck-level scalar puts one number everywhere;
// Norne states five depth tables that disagree with each other by 27%, and
// running the crest value over the whole field swells the oil in the leaner
// regions and costs stock-tank barrels.
//
// The parsing trap is the record structure, and it is worth a test on its own:
// each region's table is ONE record -- a flat run of depth/Rs pairs ended by a
// slash -- not one record per row.  Reading a pair per record yields a single
// table of the first two numbers, silently discards every other region and all
// depth dependence, and reproduces exactly the constant this replaces.  The
// first implementation did that and the numbers looked entirely plausible.
void test_rsvd(const std::string& deck) {
  std::printf("\n[6] Norne's RSVD\n");
  Equilibration e;
  e.read(deck);

  check(e.have_rsvd(), "RSVD was found in the deck");
  check((int)e.rsvd().size() == 5, "five tables, one per EQUIL record");
  if (e.rsvd().size() < 5) return;

  // sm3/sm3 -> scf/stb, the same factor the PVT tables use.
  const double k = (35.3146667 / 1000.0) / 6.2898107704 * 1000.0;

  // Regions 1, 3 and 5 are the gas-cap columns and run from about 120 sm3/sm3
  // at the crest down to 106.77 deeper; 2 and 4 sit near 94.5 throughout.
  for (int r : {0, 2, 4}) {
    const RsvdTable& t = e.rsvd()[(size_t)r];
    check(t.depth.size() == 4, "a crest table has its four rows");
    if (t.depth.size() == 4) {
      check_near((double)t.rs.front() * k, 675.0, 0.01,
                 "crest Rs is about 675 scf/stb");
      check_near((double)t.rs.back() * k, 599.5, 0.01,
                 "and falls to about 600 with depth");
    }
  }
  for (int r : {1, 3}) {
    const RsvdTable& t = e.rsvd()[(size_t)r];
    check(t.depth.size() == 3, "a lean table has its three rows");
    if (t.depth.size() == 3)
      check_near((double)t.rs.front() * k, 530.6, 0.01,
                 "lean Rs is about 531 scf/stb, not 675");
  }

  // Monotone and ascending in depth, which the lookup relies on.
  bool ascending = true;
  for (const RsvdTable& t : e.rsvd())
    for (size_t i = 1; i < t.depth.size(); ++i)
      if (t.depth[i] < t.depth[i - 1]) ascending = false;
  check(ascending, "every table is ascending in depth");

  // Interpolation: flat outside the range, linear inside.
  {
    const RsvdTable& t = e.rsvd()[0];
    check_near((double)t.at(t.depth.front() - 100), (double)t.rs.front(), 1e-12,
               "flat above the shallowest row");
    check_near((double)t.at(t.depth.back() + 100), (double)t.rs.back(), 1e-12,
               "flat below the deepest row");
    const Real mid = (t.depth[0] + t.depth[1]) / 2;
    check_near((double)t.at(mid), (double)(t.rs[0] + t.rs[1]) / 2, 1e-9,
               "linear halfway between two rows");
  }

  // And the per-cell anchor on the real grid.  The point of the whole exercise
  // is that this is a RANGE, not a number.
  CornerPointGrid g;
  g.read_grdecl(deck);
  g.build();
  const std::vector<Real> a = e.rs_anchor(g, (Real)k);
  check((int)a.size() == g.n_active(), "an anchor for every active cell");
  if (!a.empty()) {
    double lo = a[0], hi = a[0], sum = 0;
    for (Real v : a) { if (v < lo) lo = v; if (v > hi) hi = v; sum += v; }
    const double mean = sum / (double)a.size();
    std::printf("        anchor over %d cells: %.1f .. %.1f scf/stb, mean %.1f"
                "  (the deck scalar is 674 -- the CREST value)\n",
                (int)a.size(), lo, hi, mean);
    check(lo < 540.0, "the lean regions really are near 530, not 674");
    check(hi > 670.0, "and the crest really is near 675");
    check(mean < 640.0 && mean > 560.0,
          "so the field mean is well below the single number it replaces");
  }
}

int main(int argc, char** argv) {
  const std::string deck = (argc > 1) ? argv[1] : "";

  std::printf("NVRS equilibration tests\n");
  std::printf("========================\n");
  try {
    test_gradients();
    test_datum_and_contacts();
    test_initialise();
    test_regions();
    if (!deck.empty()) { test_norne(deck); test_rsvd(deck); }
    else std::printf("\n[5] Norne: skipped (pass the .DATA deck to run)\n");
  } catch (const std::exception& ex) {
    std::printf("\nEXCEPTION: %s\n", ex.what());
    return 2;
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
