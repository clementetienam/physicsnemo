// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_cpgrid.cpp -- checks on the corner-point reader and the connection list.
//
// The tests are arranged so that each one can only fail for a single reason.
// The first group builds grids whose answers are known analytically, because a
// geometry kernel that is wrong on a unit cube will be wrong on Norne in ways
// that are much harder to see.  The last group reads Norne itself and asserts
// the properties that must hold of any correct reading of it -- cell counts,
// pore volume, connection structure, and the fault behaviour that is the whole
// point of the exercise.
//
//   g++ -O2 -std=c++17 -I../include test_cpgrid.cpp ../src/cpgrid.cpp -o t
//   ./t [path/to/NORNE_ATW2013.DATA]

#include "cpgrid.hpp"

#include <algorithm>   // std::min, std::max: GCC supplies these through
                       // other headers, MSVC does not.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// Write a deck to a file beside the executable and return its path.  The
// block-centred tests build their decks inline, because the point of each is a
// number that can be checked by hand and an inline deck keeps that number next
// to the grid that produces it.
static std::string write_temp(const char* name, const char* text) {
  std::string path = std::string("nvrs_test_") + name;
  std::ofstream f(path);
  f << text;
  f.close();
  return path;
}

using namespace nvrs;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) { ++g_failures; std::printf("  FAIL  %s\n", what); }
  else      std::printf("  ok    %s\n", what);
}

void check_near(double got, double want, double tol, const char* what) {
  ++g_checks;
  const double err = std::fabs(got - want);
  const double rel = (std::fabs(want) > 0) ? err / std::fabs(want) : err;
  if (rel > tol) {
    ++g_failures;
    std::printf("  FAIL  %s: got %.10g, want %.10g (rel %.3g > %.3g)\n",
                what, got, want, rel, tol);
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

// Write a Cartesian box out as corner-point, which is the one case where the
// corner-point answer is known in closed form.
//
// `throw_at_i` displaces every COLUMN whose I index exceeds it downwards by
// `throw_z`.  The displacement is applied per cell, not per pillar corner, so
// the cells on the far side move as a rigid block and the two sides of the
// plane stop lining up -- which is what a throw fault is.  Corner-point
// geometry can express this because ZCORN stores the +I corners of cell i and
// the -I corners of cell i+1 as separate entries, free to differ.
std::string write_box(const std::string& name, int nx, int ny, int nz,
                      double dx, double dy, double dz, double top,
                      double perm, double poro,
                      int throw_at_i = -1, double throw_z = 0.0) {
  const std::string path = tmpdir() + "/" + name;
  std::ofstream f(path);
  f << "SPECGRID\n " << nx << " " << ny << " " << nz << " 1 F /\n\n";

  // Vertical pillars.  They are not displaced: the cells slide along them, and
  // the range is padded so every corner depth falls inside the pillar.
  const double zlo = top - 1000.0, zhi = top + nz * dz + 1000.0;
  f << "COORD\n";
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i) {
      const double x = i * dx, y = j * dy;
      f << " " << x << " " << y << " " << zlo << " "
        << x << " " << y << " " << zhi << "\n";
    }
  f << "/\n\n";

  // ZCORN runs i fastest (two entries per cell), then j, then k, with the top
  // face of every layer written before its bottom face.
  f << "ZCORN\n";
  for (int k = 0; k < nz; ++k)
    for (int half = 0; half < 2; ++half)            // top then bottom
      for (int j = 0; j < ny; ++j)
        for (int dup = 0; dup < 2; ++dup)           // -J then +J corners
          for (int i = 0; i < nx; ++i)
            for (int d = 0; d < 2; ++d) {           // -I then +I corners
              (void)d;
              const double s =
                  (throw_at_i >= 0 && i > throw_at_i) ? throw_z : 0.0;
              f << " " << (top + s + (k + half) * dz);
            }
  f << "\n/\n\n";

  auto field = [&](const char* kw, double v) {
    f << kw << "\n " << (nx * ny * nz) << "*" << v << " /\n\n";
  };
  field("PORO", poro);
  field("PERMX", perm);
  field("PERMY", perm);
  field("PERMZ", perm);
  field("NTG", 1.0);
  return path;
}

// ---------------------------------------------------------------------------
void test_single_cell() {
  std::printf("\n[1] one cell: volume, depth, and no connections\n");
  const std::string p = write_box("nvrs_t1.grdecl", 1, 1, 1,
                                  /*dx*/ 20, /*dy*/ 10, /*dz*/ 2,
                                  /*top*/ 1000, /*perm*/ 100, /*poro*/ 0.2);
  CornerPointGrid g;
  g.set_units(Units::Metric);
  g.read_grdecl(p);
  g.build();

  check(g.n_active() == 1, "one active cell");
  check_near(g.volume[0], 20.0 * 10.0 * 2.0, 1e-12, "bulk volume");
  check_near(g.porv[0], 20.0 * 10.0 * 2.0 * 0.2, 1e-12, "pore volume");
  check_near(g.depth[0], 1001.0, 1e-12, "centre depth");
  check_near(g.dz[0], 2.0, 1e-12, "thickness");
  check(g.conns.empty(), "an isolated cell has no connections");
}

void test_cartesian_transmissibility() {
  std::printf("\n[2] Cartesian box: transmissibility against k*A/d\n");
  const double dx = 20, dy = 10, dz = 2, K = 100, C = 0.0085267014;
  const std::string p = write_box("nvrs_t2.grdecl", 3, 2, 2, dx, dy, dz,
                                  2000, K, 0.25);
  CornerPointGrid g;
  g.set_units(Units::Metric);
  g.read_grdecl(p);
  g.build();

  check(g.n_active() == 12, "3 x 2 x 2 = 12 active cells");

  // In a uniform box the two half-transmissibilities are equal, so the series
  // combination is k*A/d with d the FULL cell-centre separation.
  const double tI = C * K * (dy * dz) / dx;
  const double tJ = C * K * (dx * dz) / dy;
  const double tK = C * K * (dx * dy) / dz;

  int nI = 0, nJ = 0, nK = 0;
  bool okI = true, okJ = true, okK = true;
  for (const Connection& c : g.conns) {
    const double want = (c.dir == 0) ? tI : (c.dir == 1) ? tJ : tK;
    const bool ok = std::fabs(c.trans - want) / want < 1e-10;
    if (c.dir == 0) { ++nI; okI &= ok; }
    else if (c.dir == 1) { ++nJ; okJ &= ok; }
    else if (c.dir == 2) { ++nK; okK &= ok; }
  }
  // 3x2x2: I faces 2*2*2 = 8, J faces 3*1*2 = 6, K faces 3*2*1 = 6.
  check(nI == 8 && nJ == 6 && nK == 6, "connection counts (8 I, 6 J, 6 K)");
  check(g.conns.size() == 20u, "no spurious non-neighbour connections");
  check_near(g.conns.empty() ? 0 : tI, tI, 1e-12, "expected T_I");
  check(okI, "every I connection equals k*A/dx");
  check(okJ, "every J connection equals k*A/dy");
  check(okK, "every K connection equals k*A/dz");
}

void test_eclipse_centroid_and_logical_permeability() {
  std::printf("\n[2b] wedge cells: Eclipse centre and logical K permeability\n");

  // Two 1x1 cells share the sloping face z=1+2y.  Eclipse defines the cell
  // centre as the mean of the eight input corners.  Both centres are exactly
  // one metre normally above/below the shared face centre, whose projected
  // area in K is 1 m2.  With KZ=30 mD, each half transmissibility is C*30 and
  // the series pair is C*30/2.  A volume centroid gives a different result;
  // applying KX or KY on this sloping face gives a different result as well.
  const std::string p = tmpdir() + "/nvrs_t2b.grdecl";
  {
    std::ofstream f(p);
    f << "SPECGRID\n 1 1 2 1 F /\n\n"
         "COORD\n"
         " 0 0 -10  0 0 10\n"
         " 1 0 -10  1 0 10\n"
         " 0 1 -10  0 1 10\n"
         " 1 1 -10  1 1 10 /\n\n"
         "ZCORN\n"
         " 0 0 0 0   1 1 3 3\n"
         " 1 1 3 3   4 4 4 4 /\n\n"
         "PORO\n 2*0.2 /\n"
         "PERMX\n 2*10 /\n"
         "PERMY\n 2*20 /\n"
         "PERMZ\n 2*30 /\n"
         "NTG\n 2*1 /\n";
  }
  CornerPointGrid g;
  g.set_units(Units::Metric);
  g.read_grdecl(p);
  g.build();
  check(g.conns.size() == 1u, "the two wedge cells share one K face");
  if (!g.conns.empty()) {
    check(g.conns[0].dir == 2, "the shared face is logical K");
    check_near(g.conns[0].trans, 0.0085267014 * 30.0 / 2.0, 1e-10,
               "Eclipse-centred K transmissibility");
  }
}

void test_fault_juxtaposition() {
  std::printf("\n[3] a throw fault: partial juxtaposition\n");
  // Four layers 2 m thick; the right half drops by exactly one layer.  Every
  // cell on the left should then face exactly one cell on the right, offset by
  // one in K -- a clean, checkable case of a non-neighbour connection.
  const std::string p = write_box("nvrs_t3.grdecl", 2, 1, 4, 20, 10, 2,
                                  1000, 100, 0.2,
                                  /*throw_at_i*/ 0, /*throw_z*/ 2.0);
  CornerPointGrid g;
  g.set_units(Units::Metric);
  g.read_grdecl(p);
  g.build();

  check(g.n_active() == 8, "8 active cells");

  int n_cross = 0, n_aligned = 0;
  for (const Connection& c : g.conns) {
    const int ga = g.global_of_active(c.a), gb = g.global_of_active(c.b);
    const int ia = ga % 2, ka = ga / 2;
    const int ib = gb % 2, kb = gb / 2;
    if (ia == ib) continue;                       // a vertical connection
    if (ka == kb) ++n_aligned; else ++n_cross;
  }
  check(n_aligned == 0, "a full-layer throw leaves no aligned I connections");
  check(n_cross == 3, "it creates 3 offset connections (layers 1-0, 2-1, 3-2)");

  // Half a layer of throw should instead split each face in two.
  const std::string p2 = write_box("nvrs_t4.grdecl", 2, 1, 4, 20, 10, 2,
                                   1000, 100, 0.2, 0, 1.0);
  CornerPointGrid g2;
  g2.set_units(Units::Metric);
  g2.read_grdecl(p2);
  g2.build();
  int n_lat = 0;
  double t_sum = 0;
  for (const Connection& c : g2.conns) {
    const int ga = g2.global_of_active(c.a), gb = g2.global_of_active(c.b);
    if ((ga % 2) != (gb % 2)) { ++n_lat; t_sum += c.trans; }
  }
  check(n_lat == 7, "half-layer throw splits the column into 7 lateral faces");

  // Whatever the throw, the total flow area across the fault plane cannot
  // exceed the unfaulted area, and here the overlap is complete except for the
  // half layer lost at each end.
  const double C = 0.0085267014;
  const double t_full = 4 * C * 100.0 * (10.0 * 2.0) / 20.0;
  check(t_sum < t_full * 1.0001 && t_sum > t_full * 0.7,
        "total cross-fault transmissibility is conserved up to the free ends");
}

void test_partial_width_fault_overlap() {
  std::printf("\n[3b] warped fault: overlap over only part of the face width\n");
  const std::string p = tmpdir() + "/nvrs_partial_fault.grdecl";
  {
    std::ofstream f(p);
    f << "SPECGRID\n 2 1 1 1 F /\n\nCOORD\n"
         " 0 0 -20 0 0 20\n 1 0 -20 1 0 20\n 2 0 -20 2 0 20\n"
         " 0 1 -20 0 1 20\n 1 1 -20 1 1 20\n 2 1 -20 2 1 20 /\n\n"
         "ZCORN\n 0 0 3 3  10 10 0 0  2 2 5 5  12 12 2 2 /\n\n"
         "PORO\n 2*0.2 /\nPERMX\n 2*100 /\nPERMY\n 2*100 /\n"
         "PERMZ\n 2*100 /\nNTG\n 2*1 /\n";
  }
  CornerPointGrid g;
  g.set_units(Units::Metric);
  g.read_grdecl(p);
  g.build();
  check(g.conns.size() == 1u,
        "a triangular partial-width overlap creates one connection");
  if (!g.conns.empty()) {
    check(g.conns[0].dir == 0, "the partial connection is in logical I");
    check(g.conns[0].trans > 0, "the partial connection is transmissive");
  }
}

void test_multfl_and_multz() {
  std::printf("\n[4] MULTFLT and MULTZ scale the right connections\n");
  const std::string base = write_box("nvrs_t5.grdecl", 2, 1, 2, 20, 10, 2,
                                     1000, 100, 0.2);
  // Reopen the file and append a fault on the I+ face of column 1, sealed to
  // one tenth, plus a MULTZ that halves the vertical connection in column 1.
  {
    std::ofstream f(base, std::ios::app);
    f << "FAULTS\n 'F1' 1 1 1 1 1 2 'X' /\n/\n\n";
    f << "MULTFLT\n 'F1' 0.1 /\n/\n\n";
    f << "MULTZ\n 0.5 1.0 0.5 1.0 /\n\n";
  }
  CornerPointGrid g;
  g.set_units(Units::Metric);
  g.read_grdecl(base);
  g.build();

  const double C = 0.0085267014;
  const double tI = C * 100.0 * (10.0 * 2.0) / 20.0;
  const double tK = C * 100.0 * (20.0 * 10.0) / 2.0;

  bool fault_ok = true, multz_ok = true;
  int n_fault = 0, n_vert = 0;
  for (const Connection& c : g.conns) {
    const int ga = g.global_of_active(c.a), gb = g.global_of_active(c.b);
    if ((ga % 2) != (gb % 2)) {
      ++n_fault;
      fault_ok &= std::fabs(c.trans - 0.1 * tI) / (0.1 * tI) < 1e-10;
    } else {
      ++n_vert;
      const double want = ((ga % 2) == 0) ? 0.5 * tK : tK;
      multz_ok &= std::fabs(c.trans - want) / want < 1e-10;
    }
  }
  check(n_fault == 2, "both fault faces found");
  check(fault_ok, "MULTFLT 0.1 applied to the cross-fault transmissibility");
  check(n_vert == 2, "both vertical connections found");
  check(multz_ok, "MULTZ applied per column, taken from the upper cell");
}

// FAULTS names a face by (I,J,K) range, not by connection: when the fault
// throws, that face's range still covers every cell on the low side, but each
// of them now faces a DIFFERENT cell across the fault -- test_fault_
// juxtaposition's "3 offset connections" for a full-layer throw.  Those are
// non-neighbour connections (dir 3), and MULTFLT has to reach every one of
// them or a deck's fault seal is only partly applied wherever the fault
// actually has relief, which is a fault's entire point.  Neither existing
// MULTFLT test exercises this together: test_multfl_and_multz has no throw,
// test_fault_juxtaposition has no MULTFLT.
void test_multflt_on_throw() {
  std::printf("\n[4b] MULTFLT reaches every juxtaposed connection a throw makes\n");
  const std::string p = write_box("nvrs_t5b.grdecl", 2, 1, 4, 20, 10, 2,
                                  1000, 100, 0.2,
                                  /*throw_at_i*/ 0, /*throw_z*/ 2.0);
  {
    std::ofstream f(p, std::ios::app);
    f << "FAULTS\n 'F1' 1 1 1 1 1 4 'X' /\n/\n\n";
    f << "MULTFLT\n 'F1' 0.1 /\n/\n\n";
  }
  CornerPointGrid g;
  g.set_units(Units::Metric);
  g.read_grdecl(p);
  g.build();

  const double C = 0.0085267014;
  const double tI = C * 100.0 * (10.0 * 2.0) / 20.0;   // one full lateral face

  int n_cross = 0;
  bool all_scaled = true;
  for (const Connection& c : g.conns) {
    const int ga = g.global_of_active(c.a), gb = g.global_of_active(c.b);
    if ((ga % 2) == (gb % 2)) continue;   // a vertical connection
    ++n_cross;
    all_scaled &= std::fabs(c.trans - 0.1 * tI) / (0.1 * tI) < 1e-10;
  }
  check(n_cross == 3, "the throw's 3 offset connections are all present");
  check(all_scaled, "MULTFLT 0.1 applied to every one of them, not just the aligned ones");
}

void test_inactive_and_pinch() {
  std::printf("\n[5] vertical bridging: MINPV pinches out, ACTNUM seals\n");

  // A plain ACTNUM=0 middle layer, MINPV never in force: rock the deck
  // itself excluded, not a thin streak its own pore volume trimmed.  Most of
  // a real corner-point grid's inactive cells are this -- Norne carries
  // 68,417 of them against 496 genuine MINPV cuts -- and the cells above and
  // below must NOT connect across one: nothing about ACTNUM says the gap is
  // thin, and bridging it with a half-transmissibility measured from each
  // surviving cell's own face turns however much rock sits between them into
  // a connection as strong as two cells that actually touch.
  {
    const std::string p = write_box("nvrs_t6.grdecl", 1, 1, 3, 20, 10, 2,
                                    1000, 100, 0.2);
    { std::ofstream f(p, std::ios::app); f << "ACTNUM\n 1 0 1 /\n\n"; }
    CornerPointGrid g;
    g.set_units(Units::Metric);
    g.read_grdecl(p);
    g.build();
    check(g.n_active() == 2, "the middle layer is inactive");
    check(g.conns.empty(), "ACTNUM seals: no connection across a real gap");
  }

  // PINCH explicitly makes an ACTNUM hole transparent when its thickness is
  // below the stated threshold.  This is distinct from MINPV/GAP and is the
  // path used by Norne's geometrically collapsed inactive layers.
  {
    const std::string p = write_box("nvrs_t6p.grdecl", 1, 1, 3, 20, 10, 2,
                                    1000, 100, 0.2);
    {
      std::ofstream f(p, std::ios::app);
      f << "PINCH\n 3 GAP 1* TOPBOT TOP /\n\n";
      f << "ACTNUM\n 1 0 1 /\n\n";
    }
    CornerPointGrid g;
    g.set_units(Units::Metric);
    g.read_grdecl(p);
    g.build();
    check(g.n_active() == 2, "PINCH leaves the middle ACTNUM cell inactive");
    check(g.conns.size() == 1u,
          "PINCH crosses an ACTNUM hole below its thickness threshold");
  }

  // The same three layers, ACTNUM 1 throughout, but the middle one's own
  // pore volume put below MINPV: that IS a pinch-out, in the ECLIPSE sense,
  // and the survivors must still connect the way the old (undifferentiated)
  // version of this test asserted for the ACTNUM case above.
  {
    const std::string p = write_box("nvrs_t6b.grdecl", 1, 1, 3, 20, 10, 2,
                                    1000, 100, 0.2);
    {
      std::ofstream f(p, std::ios::app);
      f << "EQUALS\n 'PORO' 0.001 1 1 1 1 2 2 /\n/\n\n";
      // Normal cells here carry 400 m^3 * 0.2 = 80 m^3 of pore volume; the
      // thinned middle one carries 400 * 0.001 = 0.4.  MINPV has to clear the
      // second and sit under the first, or every cell in the column reads as
      // a pinch-out and the test no longer isolates the middle one.
      f << "MINPV\n 10 /\n\n";
    }
    CornerPointGrid g;
    g.set_units(Units::Metric);
    g.read_grdecl(p);
    g.build();
    check(g.n_active() == 2, "the thin middle layer drops below MINPV");
    check(g.conns.size() == 1u, "MINPV pinch-out: the survivors still connect");

    // Each half-transmissibility is measured from a cell centre to that
    // cell's own face, so both are k*A/1 and the series pair is k*A/2 -- the
    // gap left by the pinched layer does not enter.  That is deliberate and
    // matches what ECLIPSE does across a genuine pinch-out: the two
    // half-transmissibilities are all the geometry there is to work with, so
    // a pinched connection is slightly more open than the true separation
    // would give.
    const double C = 0.0085267014, A = 20.0 * 10.0;
    const double half = C * 100.0 * A / 1.0;   // centre to its own face: 1 m
    const double want = half / 2.0;
    if (!g.conns.empty())
      check_near(g.conns[0].trans, want, 1e-9,
                 "transmissibility across the pinched layer");
  }
}

// ---------------------------------------------------------------------------
// The expected values below were derived from the deck itself by the NumPy
// reference implementation in scripts/cpgrid_ref.py, not copied from another
// simulator's output.  Two independent implementations of the same
// specification agreeing on them is the actual check; the constants are only
// how that agreement is recorded so it can be re-checked cheaply.
void test_norne(const std::string& deck) {
  std::printf("\n[6] Norne: %s\n", deck.c_str());
  {
    std::ifstream t(deck);
    if (!t) { std::printf("  SKIP  deck not found\n"); return; }
  }

  CornerPointGrid g;
  g.read_props_arrays = true;
  g.read_grdecl(deck);
  check(g.nx == 46 && g.ny == 112 && g.nz == 22, "dimensions 46 x 112 x 22");
  check(g.units() == Units::Metric, "the deck's METRIC unit system was picked up");
  check_near(g.min_pore_volume(), 500.0, 1e-12, "MINPV read from the deck");

  g.build();
  std::printf("%s", g.summary().c_str());

  // 44,927 is the number of cells ACTNUM leaves on, and it is the figure Norne
  // is documented with.  It can only come out right if SPECGRID, every
  // INCLUDE, and all four property lengths agree, so it checks the whole
  // reading path at once.  MINPV 500 then drops a further 496 sliver cells
  // holding 0.015% of the field's pore volume.
  check(g.n_active() == 44431,
        "44,431 active cells: 44,927 from ACTNUM, less 496 below MINPV");

  // PERMZ exists only as PERMX copied and then scaled layer by layer, so a
  // reader that ignores COPY and MULTIPLY gets vertical flow badly wrong while
  // looking entirely healthy.  These bracket the result of those operations.
  double kzmin = 1e300, kzmax = 0, kxmax = 0;
  for (int a = 0; a < g.n_active(); ++a) {
    kzmin = std::min(kzmin, (double)g.permz[a]);
    kzmax = std::max(kzmax, (double)g.permz[a]);
    kxmax = std::max(kxmax, (double)g.permx[a]);
  }
  check(kzmax < kxmax, "PERMZ was scaled below PERMX by the layer multipliers");
  check_near(kxmax, 3996.55, 1e-4, "max PERMX");
  check_near(kzmax, 2105.70, 1e-4, "max PERMZ");
  check(kzmin > 0, "no active cell was left with zero vertical permeability");

  double pv = 0;
  for (int a = 0; a < g.n_active(); ++a) pv += g.porv[a];
  check_near(pv, 6.73149e8, 1e-4, "total pore volume [m3]");

  // Carlson hysteresis uses the imbibition region and I-prefixed endpoints.
  // These used to be silently dropped even though the deck defines them.
  check(g.has_array("IMBNUM"), "IMBNUM retained for hysteresis");
  check(g.has_array("ISGCR") && g.has_array("ISGU") &&
            g.has_array("ISGL") && g.has_array("ISOGCR"),
        "gas imbibition endpoints retained");
  const std::vector<Real> isgcr = g.active_array("ISGCR");
  const std::vector<char> isgu_defined = g.active_array_defined("ISGU");
  const std::vector<Real> imbnum = g.active_array("IMBNUM");
  check((int)isgcr.size() == g.n_active(), "ISGCR has one value per active cell");
  check((int)imbnum.size() == g.n_active(), "IMBNUM has one value per active cell");
  if (!isgcr.empty()) check_near(isgcr.front(), 0.22, 1e-12, "Norne ISGCR");
  bool saw_isgu_missing = false, saw_isgu_defined = false;
  for (char v : isgu_defined) {
    saw_isgu_missing |= !v;
    saw_isgu_defined |= !!v;
  }
  check(saw_isgu_missing && saw_isgu_defined,
        "Norne ISGU retains partial-layer definedness for table fallback");
  if (!imbnum.empty()) check_near(imbnum.front(), 2.0, 1e-12, "Norne IMBNUM");

  // Structural properties that must hold whatever the numbers turn out to be.
  bool depth_ok = true, vol_ok = true, pv_ok = true;
  for (int a = 0; a < g.n_active(); ++a) {
    if (!(g.depth[a] > 2400 && g.depth[a] < 3100)) depth_ok = false;
    if (!(g.volume[a] > 0)) vol_ok = false;
    if (!(g.porv[a] >= 500.0)) pv_ok = false;
  }
  check(depth_ok, "every cell centre lies in the Norne depth range");
  check(vol_ok, "every active cell has positive volume");
  check(pv_ok, "every surviving cell clears MINPV");

  bool conn_ok = true;
  for (const Connection& c : g.conns)
    if (c.a < 0 || c.b < 0 || c.a >= g.n_active() || c.b >= g.n_active() ||
        c.a >= c.b || !(c.trans > 0))
      conn_ok = false;
  check(conn_ok, "connections are in range, ordered, and positive");

  int nI = 0, nJ = 0, nK = 0, nN = 0;
  for (const Connection& c : g.conns)
    switch (c.dir) { case 0: ++nI; break; case 1: ++nJ; break;
                     case 2: ++nK; break; default: ++nN; }
  // 744 fewer K connections than before the vertical pass stopped bridging
  // through a real ACTNUM=0 gap: those were wormholes through non-reservoir
  // rock, not pinch-outs, and see build_connections() for why they are gone.
  check(g.conns.size() == 125447u, "125,447 connections including PINCH NNCs");
  check(nI == 37611 && nJ == 42641 && nK == 36892, "I/J/K connection counts");
  check(nN == 8303, "8,303 non-neighbour connections from the fault throws");

  // The negligible-connection filter is what makes this matrix solvable.
  // Without it Norne spans fourteen orders of magnitude between its smallest
  // and largest transmissibility, and the algebraic multigrid inside CPR has
  // to coarsen that graph.
  double tmin = 1e300, tmax = 0;
  for (const Connection& c : g.conns) {
    tmin = std::min(tmin, (double)c.trans);
    tmax = std::max(tmax, (double)c.trans);
  }
  check(tmax / tmin < 1e8,
        "the surviving transmissibilities span under eight decades");
  std::printf("  note  transmissibility spans %.3g decades\n",
              std::log10(tmax / tmin));

  check(g.fault_names().size() == 61u, "61 named faults");

  // Norne's twenty flux regions are sealed against each other by 190 MULTREGT
  // records keyed on FLUXNUM.  They are worth asserting because the failure
  // mode is silent: a reader that mishandles the record leaves every barrier
  // wide open and still produces a healthy-looking grid.
  check(g.has_array("FLUXNUM"), "FLUXNUM was read for MULTREGT to key on");

  // Sealing every fault must remove exactly the cross-fault transmissibility
  // and leave the cell count untouched.  This is the sharpest available check
  // that fault faces are being matched to the right connections: if the face
  // lookup were wrong, sealing would either change nothing or take out
  // connections that no fault touches.
  CornerPointGrid gs;
  gs.read_grdecl(deck);
  for (const std::string& n : gs.fault_names()) gs.set_fault_multiplier(n, 0.0);
  gs.build();

  check(gs.n_active() == g.n_active(), "sealing faults changes no cell count");
  check(gs.conns.size() < g.conns.size(),
        "sealing removes the cross-fault connections");

  double t_open = 0, t_sealed = 0;
  for (const Connection& c : g.conns) t_open += c.trans;
  for (const Connection& c : gs.conns) t_sealed += c.trans;
  std::printf("  note  connections %d -> %d, transmissibility %.6g -> %.6g "
              "(%.1f%% removed)\n",
              (int)g.conns.size(), (int)gs.conns.size(), t_open, t_sealed,
              100.0 * (1.0 - t_sealed / t_open));
  check(t_sealed < t_open, "sealed total is strictly lower");

  // Every connection that survives sealing must be one no fault named, and
  // its transmissibility must be untouched.  That is the "nowhere else" half
  // of the claim, and it is what would catch a face-index off-by-one.
  size_t i = 0, j = 0, changed = 0, matched = 0;
  while (i < g.conns.size() && j < gs.conns.size()) {
    const Connection& x = g.conns[i];
    const Connection& y = gs.conns[j];
    if (x.a == y.a && x.b == y.b) {
      ++matched;
      if (std::fabs(x.trans - y.trans) > 1e-9 * x.trans) ++changed;
      ++i; ++j;
    } else if (x.a < y.a || (x.a == y.a && x.b < y.b)) {
      ++i;                     // a connection sealing removed entirely
    } else {
      ++j;                     // should not happen: sealing creates nothing
    }
  }
  check(j == gs.conns.size(), "sealing created no new connections");
  check(changed == 0,
        "every surviving connection kept its transmissibility exactly");
  std::printf("  note  %d connections survived sealing unchanged\n", (int)matched);
}

}  // namespace

// ---------------------------------------------------------------------------
// Block-centred geometry
// ---------------------------------------------------------------------------
// A deck may give cell sizes rather than corners, and most published decks do:
// SPE1, SPE3 and SPE9 are all block-centred.  The reader converts them into
// the corners a corner-point deck would have carried, so the whole of the rest
// of this file applies to them unchanged.  What has to be checked is the
// conversion itself, and every quantity below is known in closed form for a
// grid of bricks.
static void test_block_centred_dxv() {
  // 10 x 10 x 3, 1000 ft square cells, layers 20, 30 and 50 ft thick, top of
  // the first layer at 8325 ft.  FIELD units, so volumes come back in barrels.
  const char* deck =
      "RUNSPEC\n DIMENS\n 10 10 3 /\n FIELD\n"
      "GRID\n"
      " DXV\n 10*1000 /\n"
      " DYV\n 10*1000 /\n"
      " DZV\n 20 30 50 /\n"
      " TOPS\n 100*8325 /\n"
      " PORO\n 300*0.3 /\n"
      " PERMX\n 300*500 /\n PERMY\n 300*500 /\n PERMZ\n 300*50 /\n";
  const std::string path = write_temp("bc_dxv.DATA", deck);

  CornerPointGrid g;
  g.read_grdecl(path);
  g.build();

  check(g.nx == 10 && g.ny == 10 && g.nz == 3, "DXV/DYV/DZV: dimensions");
  check(g.n_active() == 300, "DXV/DYV/DZV: every cell is active");

  // 10^6 ft^2 of area per cell, 100 ft of column, 100 columns: 10^10 ft^3.
  // FIELD volumes are barrels, and there are 5.614583 ft^3 in one.
  const double ft3 = 1e6 * 100.0 * 100.0;
  const double want_bulk = ft3 / 5.614583;
  double bulk = 0;
  for (int a = 0; a < g.n_active(); ++a) bulk += g.volume[a];
  check(std::fabs(bulk - want_bulk) / want_bulk < 1e-9,
        "DXV/DYV/DZV: bulk volume matches the closed form");

  double pv = 0;
  for (int a = 0; a < g.n_active(); ++a) pv += g.porv[a];
  check(std::fabs(pv - 0.3 * want_bulk) / (0.3 * want_bulk) < 1e-9,
        "DXV/DYV/DZV: pore volume is porosity times bulk");

  // The thinnest and thickest layers, again in barrels.
  double vmin = 1e300, vmax = 0;
  for (int a = 0; a < g.n_active(); ++a) {
    vmin = std::min(vmin, (double)g.volume[a]);
    vmax = std::max(vmax, (double)g.volume[a]);
  }
  check(std::fabs(vmin - 1e6 * 20.0 / 5.614583) / vmin < 1e-9,
        "DXV/DYV/DZV: thinnest cell is the 20 ft layer");
  check(std::fabs(vmax - 1e6 * 50.0 / 5.614583) / vmax < 1e-9,
        "DXV/DYV/DZV: thickest cell is the 50 ft layer");

  // Cell centres: 8325 + 10 for the first layer, 8325 + 20 + 30 + 25 for the
  // third.  The layers stack, which is what makes the column continuous.
  double zmin = 1e300, zmax = -1e300;
  for (int a = 0; a < g.n_active(); ++a) {
    zmin = std::min(zmin, (double)g.depth[a]);
    zmax = std::max(zmax, (double)g.depth[a]);
  }
  check(std::fabs(zmin - 8335.0) < 1e-6,
        "DXV/DYV/DZV: first layer centres at 8335 ft");
  check(std::fabs(zmax - 8400.0) < 1e-6,
        "DXV/DYV/DZV: third layer centres at 8400 ft, the layers having stacked");

  // A brick grid has no non-neighbour connections and a count fixed by the
  // dimensions alone.
  const size_t want_conn = (size_t)9 * 10 * 3 + (size_t)10 * 9 * 3 +
                           (size_t)10 * 10 * 2;
  check(g.conns.size() == want_conn,
        "DXV/DYV/DZV: connection count is 9.10.3 + 10.9.3 + 10.10.2");
}

// The same grid written the other way, one value per cell rather than one per
// index, which is how SPE9 and most hand-written decks spell it.
static void test_block_centred_percell() {
  const char* deck =
      "RUNSPEC\n DIMENS\n 4 3 2 /\n FIELD\n"
      "GRID\n"
      " DX\n 24*300 /\n"
      " DY\n 24*300 /\n"
      " DZ\n 12*20 12*40 /\n"
      " TOPS\n 12*9000 /\n"
      " PORO\n 24*0.25 /\n"
      " PERMX\n 24*100 /\n PERMY\n 24*100 /\n PERMZ\n 24*10 /\n";
  const std::string path = write_temp("bc_percell.DATA", deck);

  CornerPointGrid g;
  g.read_grdecl(path);
  g.build();

  check(g.n_active() == 24, "DX/DY/DZ: every cell is active");

  const double want = (12.0 * 300 * 300 * 20 + 12.0 * 300 * 300 * 40) / 5.614583;
  double bulk = 0;
  for (int a = 0; a < g.n_active(); ++a) bulk += g.volume[a];
  check(std::fabs(bulk - want) / want < 1e-9,
        "DX/DY/DZ: bulk volume matches the closed form");

  double zmin = 1e300, zmax = -1e300;
  for (int a = 0; a < g.n_active(); ++a) {
    zmin = std::min(zmin, (double)g.depth[a]);
    zmax = std::max(zmax, (double)g.depth[a]);
  }
  check(std::fabs(zmin - 9010.0) < 1e-6, "DX/DY/DZ: first layer centres at 9010 ft");
  check(std::fabs(zmax - 9040.0) < 1e-6, "DX/DY/DZ: second layer centres at 9040 ft");
}

// A deck with neither form must say so, rather than producing a grid of
// nothing or reading past the end of an empty array.
static void test_no_geometry_is_refused() {
  const char* deck =
      "RUNSPEC\n DIMENS\n 5 5 2 /\n FIELD\n"
      "GRID\n PORO\n 50*0.2 /\n PERMX\n 50*100 /\n";
  const std::string path = write_temp("bc_none.DATA", deck);

  CornerPointGrid g;
  g.read_grdecl(path);
  bool threw = false;
  std::string what;
  try {
    g.build();
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  check(threw, "a deck with no geometry is refused rather than built");
  check(what.find("COORD") != std::string::npos &&
        what.find("DX") != std::string::npos,
        "the refusal names both the corner-point and block-centred forms");
}

int main(int argc, char** argv) {
  test_block_centred_dxv();
  test_block_centred_percell();
  test_no_geometry_is_refused();
  const std::string deck = (argc > 1) ? argv[1] : "";

  std::printf("NVRS corner-point grid tests\n");
  std::printf("============================\n");
  try {
    test_single_cell();
    test_cartesian_transmissibility();
    test_eclipse_centroid_and_logical_permeability();
    test_fault_juxtaposition();
    test_partial_width_fault_overlap();
    test_multfl_and_multz();
    test_multflt_on_throw();
    test_inactive_and_pinch();
    if (!deck.empty()) test_norne(deck);
    else std::printf("\n[6] Norne: skipped (pass the .DATA deck to run)\n");
  } catch (const std::exception& e) {
    std::printf("\nEXCEPTION: %s\n", e.what());
    return 2;
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
