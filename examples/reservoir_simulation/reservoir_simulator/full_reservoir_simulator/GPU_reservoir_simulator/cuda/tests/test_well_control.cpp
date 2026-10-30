// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_well_control.cpp -- the no-crossflow flowing-pressure solve.
//
// This is the arithmetic behind a producer's bottom-hole pressure, and it is
// exactly the kind of code that fails quietly: a well that solves the wrong
// equation still converges, still reports a rate, and still balances -- it just
// describes a different well.  So the solve is a pure function in
// include/well_control.hpp with no CUDA in it, and this exercises it directly.
//
// The property that makes the change safe to switch on is the first one below:
// where no completion would have flowed backwards, the constrained solve must
// return the SAME pressure as the plain division it replaces.  If that ever
// stops holding, every well in every deck moves, not just the broken ones.
//
//   g++ -O2 -std=c++17 -I../include test_well_control.cpp -o t && ./t

#include "well_control.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace nvrs;

namespace {

int g_checks = 0, g_failures = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) { ++g_failures; std::printf("  FAIL  %s\n", what); }
  else      std::printf("  ok    %s\n", what);
}

void near(double got, double want, double tol, const char* what) {
  ++g_checks;
  const double d = std::fabs(want) > 1e-30 ? std::fabs(got - want) / std::fabs(want)
                                           : std::fabs(got - want);
  const bool ok = d <= tol;
  if (!ok) ++g_failures;
  std::printf("  %-5s %-46s %14.7g  want %14.7g\n", ok ? "ok" : "FAIL", what,
              got, want);
}

std::vector<WellConn> make(const std::vector<std::pair<double, double>>& av) {
  std::vector<WellConn> c;
  for (size_t i = 0; i < av.size(); ++i)
    c.push_back(WellConn{av[i].first, av[i].second, (int)i});
  return c;
}

// The equation the solve claims to satisfy, evaluated independently.
double f(const std::vector<WellConn>& c, double p) {
  double s = 0;
  for (const WellConn& k : c) if (k.theta > p) s += k.a * (k.theta - p);
  return s;
}

// The unconstrained solve this replaces: one division over every completion.
double plain(const std::vector<WellConn>& c, double target) {
  double S = 0, T = 0;
  for (const WellConn& k : c) { S += k.a; T += k.a * k.theta; }
  return (T - target) / S;
}

}  // namespace

int main() {
  std::printf("test_well_control -- the no-crossflow flowing-pressure solve\n");

  // -------------------------------------------------------------------------
  std::printf("\n[1] no crossflow: identical to the division it replaces\n");
  //
  // Four completions at similar pressures and a modest target, so the root
  // sits below all of them.  This is the ordinary well, and it is the case
  // that must not move.
  {
    const std::vector<WellConn> c =
        make({{10.0, 4000.0}, {20.0, 3990.0}, {5.0, 4010.0}, {12.0, 3995.0}});
    const double target = 20000.0;
    const WellPwf r = solve_pwf_no_crossflow(c, target, 14.7);
    const double p0 = plain(c, target);
    near(r.pwf, p0, 1e-12, "same pressure as the unconstrained solve");
    check(r.n_active == 4, "every completion is producing");
    near(r.delivered, target, 1e-12, "and the target is met exactly");
    check(!r.limited, "nothing bound");
    // Belt and braces: the equation itself.
    near(f(c, r.pwf), target, 1e-12, "f(pwf) = target, evaluated separately");
  }

  // -------------------------------------------------------------------------
  std::printf("\n[2] crossflow: the plain solve is wrong, this one is not\n");
  //
  // One completion sits 2000 psi below the others -- a depleted or
  // water-swept layer, which is what Norne's long horizontal producers run
  // through.  The plain division puts p_wf above it and counts its reverse
  // flow against the target.
  {
    const std::vector<WellConn> c =
        make({{10.0, 4000.0}, {20.0, 4010.0}, {40.0, 2000.0}});
    const double target = 20000.0;
    const double p0 = plain(c, target);
    const WellPwf r = solve_pwf_no_crossflow(c, target, 14.7);

    check(p0 > 2000.0, "the plain solve sits above the low completion");
    // HIGHER, not lower.  The plain solve had to over-produce the two good
    // completions to net out at the target against the third one's reverse
    // flow; without that burden the well needs less drawdown, not more.
    check(r.pwf > p0, "the constrained solve needs LESS drawdown");
    check(r.n_active == 2, "the backward completion is excluded");
    near(f(c, r.pwf), target, 1e-12, "and the target is still met exactly");
    // The number that matters: how much reverse flow the plain solve was
    // counting.  It is not a rounding error.
    const double reverse = 40.0 * (p0 - 2000.0);
    std::printf("        plain pwf %.1f, constrained %.1f, reverse flow the "
                "plain solve counted: %.4g\n", p0, r.pwf, reverse);
    check(reverse > 0.25 * target,
          "the reverse flow was a large share of the target");
  }

  // -------------------------------------------------------------------------
  std::printf("\n[3] the well flows entirely forwards\n");
  {
    const std::vector<WellConn> c =
        make({{10.0, 4000.0}, {20.0, 4010.0}, {40.0, 2000.0}, {8.0, 1500.0}});
    const WellPwf r = solve_pwf_no_crossflow(c, 20000.0, 14.7);
    bool all_forward = true;
    for (const WellConn& k : c)
      if (conn_produces(k.theta, r.pwf) && k.theta - r.pwf < 0)
        all_forward = false;
    check(all_forward, "no admitted completion has negative drawdown");
    double back = 0;
    for (const WellConn& k : c)
      if (!conn_produces(k.theta, r.pwf)) back += k.a * (k.theta - r.pwf);
    check(back <= 0.0, "every excluded completion was below the pressure");
  }

  // -------------------------------------------------------------------------
  std::printf("\n[4] monotone in the target\n");
  //
  // A bigger ask must mean a lower pressure and never fewer completions.  This
  // is what makes the solve stable to put inside a Newton loop: the answer
  // moves smoothly with the state instead of latching.
  {
    const std::vector<WellConn> c =
        make({{10.0, 4000.0}, {20.0, 4010.0}, {40.0, 2000.0}, {8.0, 1500.0}});
    double prev_p = 1e30;
    int prev_n = 0;
    bool mono_p = true, mono_n = true;
    for (double t = 1000.0; t <= 400000.0; t *= 1.35) {
      const WellPwf r = solve_pwf_no_crossflow(c, t, -1e9);
      if (r.pwf > prev_p) mono_p = false;
      if (r.n_active < prev_n) mono_n = false;
      prev_p = r.pwf; prev_n = r.n_active;
      if (std::fabs(f(c, r.pwf) - t) / t > 1e-10) mono_p = false;
    }
    check(mono_p, "pressure falls as the target rises, and f(pwf) = target");
    check(mono_n, "the active set only ever grows");
  }

  // -------------------------------------------------------------------------
  std::printf("\n[5] the bottom-hole floor binds\n");
  {
    const std::vector<WellConn> c = make({{1.0, 3000.0}, {1.0, 2500.0}});
    // Far more than the well can make even pulled to the floor.
    const WellPwf r = solve_pwf_no_crossflow(c, 1.0e7, 500.0);
    near(r.pwf, 500.0, 1e-12, "the pressure stops at the floor");
    check(r.limited, "and the miss is reported");
    near(r.delivered, f(c, 500.0), 1e-12,
         "delivered is what the well makes there, not the request");
    check(r.delivered < 1.0e7, "which is less than was asked for");
    check(r.n_active == 2, "both completions produce at the floor");
  }

  // -------------------------------------------------------------------------
  std::printf("\n[6] degenerate inputs\n");
  {
    // A dead completion constrains nothing and must not enter the sort.
    const std::vector<WellConn> c =
        make({{0.0, 9999.0}, {10.0, 4000.0}, {-1.0, 8000.0}});
    const WellPwf r = solve_pwf_no_crossflow(c, 10000.0, 14.7);
    near(r.pwf, 4000.0 - 1000.0, 1e-12, "zero and negative A are ignored");
    check(r.n_active == 1, "only the live completion produces");

    // Every completion dead.
    const std::vector<WellConn> d = make({{0.0, 4000.0}});
    const WellPwf rd = solve_pwf_no_crossflow(d, 10000.0, 33.0);
    check(rd.limited && rd.n_active == 0, "a well with no mobility is limited");
    near(rd.pwf, 33.0, 1e-12, "and falls back to the floor");

    // A well asked for nothing.
    const std::vector<WellConn> e = make({{10.0, 4000.0}, {5.0, 3800.0}});
    const WellPwf re = solve_pwf_no_crossflow(e, 0.0, 14.7);
    check(re.n_active == 0, "a zero target produces nothing");
    near(re.pwf, 4000.0, 1e-12, "at the highest completion pressure");

    // Ties in theta must not lose a completion.
    const std::vector<WellConn> t =
        make({{10.0, 4000.0}, {10.0, 4000.0}, {10.0, 4000.0}});
    const WellPwf rt = solve_pwf_no_crossflow(t, 30000.0, 14.7);
    near(rt.pwf, 3000.0, 1e-12, "tied pressures share the target");
    check(rt.n_active == 3, "and all three stay open");
  }

  // -------------------------------------------------------------------------
  std::printf("\n[7] a Norne-shaped well\n");
  //
  // Twenty-one completions down a long horizontal, with the toe swept and
  // 2000 psi below the heel -- the B-4H geometry.  Under a reservoir-voidage
  // target the plain solve meets the voidage while a third of the well runs
  // backwards, which is how a producer comes to report negative oil.
  {
    std::vector<WellConn> c;
    for (int k = 0; k < 21; ++k) {
      WellConn w;
      w.a = 30.0 + 2.0 * k;
      w.theta = (k < 14) ? 4200.0 - 8.0 * k : 2150.0 - 12.0 * (k - 14);
      w.index = k;
      c.push_back(w);
    }
    const double target = 60000.0;
    const double p0 = plain(c, target);
    const WellPwf r = solve_pwf_no_crossflow(c, target, 14.7);
    double reverse = 0;
    int n_back = 0;
    for (const WellConn& k : c)
      if (k.theta < p0) { reverse += k.a * (p0 - k.theta); ++n_back; }
    std::printf("        plain pwf %.1f: %d of 21 completions backwards, "
                "%.4g re-injected against a %.4g target\n",
                p0, n_back, reverse, target);
    std::printf("        constrained pwf %.1f: %d completions, all forward\n",
                r.pwf, r.n_active);
    check(n_back > 0, "the plain solve does crossflow on this geometry");
    check(reverse > 0.5 * target,
          "and the reverse flow exceeds half the target");
    near(f(c, r.pwf), target, 1e-10, "the constrained solve meets the target");
    check(r.n_active == 14, "on the unswept completions alone");
    check(r.pwf > p0, "at a higher pressure, having dropped the reverse flow");
    // The gross throughput is what actually changes, and by a lot.  Under the
    // plain solve this well moves half a million units of fluid to report
    // sixty thousand; that is the number that turns into spurious gas
    // production and negative oil on B-4H.
    double gross_plain = 0;
    for (const WellConn& k : c) gross_plain += k.a * std::fabs(k.theta - p0);
    std::printf("        gross throughput: plain %.4g, constrained %.4g\n",
                gross_plain, r.delivered);
    check(gross_plain > 3.0 * r.delivered,
          "the plain solve moves several times the fluid it reports");
  }

  // -------------------------------------------------------------------------
  std::printf("\n[8] injector: one common BHP allocates by injectivity\n");
  {
    const std::vector<WellConn> c =
        make({{10.0, 3000.0}, {30.0, 3200.0}, {5.0, 4100.0}});
    const double target = 20000.0;
    const WellPwf r = solve_injector_pwf(c, target, 6000.0);
    double q = 0.0;
    for (const WellConn& k : c)
      q += k.a * std::max(0.0, r.pwf - k.theta);
    near(q, target, 1e-12, "the common injector BHP meets the target");
    check(r.n_active == 2, "the high-pressure completion stays inactive");
    check(!r.limited, "the pressure ceiling does not bind");
    near(10.0 * (r.pwf - 3000.0) / q, 0.325, 1e-12,
         "completion shares follow injectivity and threshold");
  }

  std::printf("\n[9] injector: the upper BHP limit binds once per well\n");
  {
    const std::vector<WellConn> c =
        make({{10.0, 3000.0}, {30.0, 3200.0}, {5.0, 4100.0}});
    const WellPwf r = solve_injector_pwf(c, 200000.0, 4500.0);
    near(r.pwf, 4500.0, 1e-12, "the injector stops at its BHP ceiling");
    check(r.limited, "the target miss is reported");
    near(r.delivered, 10.0 * 1500.0 + 30.0 * 1300.0 + 5.0 * 400.0,
         1e-12, "delivered rate sums every completion at the common BHP");
    check(r.delivered < 200000.0, "delivered rate is below the request");
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
