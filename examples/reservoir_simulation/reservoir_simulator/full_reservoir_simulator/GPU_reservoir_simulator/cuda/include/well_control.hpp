// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// well_control.hpp -- solving a rate-controlled well's flowing pressure when
// its completions are not allowed to flow backwards.
//
// The problem
// -----------
// A producer under a rate target has one flowing pressure and many
// completions.  At frozen mobility each completion delivers
//
//     q_k = A_k * (theta_k - p_wf)
//
// with A_k its productivity in whatever phase the target is written in and
// theta_k the cell pressure corrected to the well's datum.  Summing and
// solving for the target is one division:
//
//     p_wf = (SUM A_k theta_k - q_target) / SUM A_k                        (1)
//
// and that is what this simulator did.  It is right exactly when every
// completion ends up producing.  When one does not -- when theta_k < p_wf --
// its term is negative, and (1) happily counts that against the total: the
// well meets its target by producing hard at one end and pushing fluid back
// into the formation at the other.
//
// Why that is not a detail on Norne
// ---------------------------------
// Nearly every Norne producer is under WCONHIST RESV, so the target is a
// RESERVOIR voidage rate and (1) is solved on total mobility.  A well can
// therefore satisfy its voidage exactly while the oil part of it runs
// backwards.  B-4H and D-4H do exactly that in the c755f1e comparison: they
// report large NEGATIVE oil against Flow's positive, while their gas is large
// and positive.  One wellbore cannot produce gas and inject oil at the same
// pressure at the same moment.
//
// What OPM actually does
// ----------------------
// Worth checking rather than assuming.  StandardWell_impl.hpp's computePerfRate
// (opm-simulators, release/2024.10) splits on the sign of the drawdown, and the
// reverse-flow branch is:
//
//     } else {
//         // Do nothing if crossflow is not allowed
//         if (!allow_cf && this->isProducer()) {
//             return;
//         }
//         // Using total mobilities
//         Value total_mob_dense = mob[0];
//         for (int componentIdx = 1; ...) total_mob_dense += mob[componentIdx];
//         ...
//         for (int componentIdx = 0; ...) {
//             const Value cqt_i = - Tw[componentIdx] * (total_mob_dense * drawdown);
//             Value cqt_is = cqt_i / volumeRatio;
//             cq_s[componentIdx] = cmix_s[componentIdx] * cqt_is;
//         }
//     }
//
// So OPM has exactly two behaviours for a producer's reverse-flowing
// connection.  Either it contributes NOTHING (crossflow not allowed), or it
// takes TOTAL mobility distributed by cmix_s -- the composition the WELLBORE is
// already carrying.  In both cases the reverse flow has the well's own
// composition, which is why an OPM producer can never take oil in at one
// connection while pushing gas out of another: there is one mixture in the
// pipe.
//
// This simulator's forward branch is the same as OPM's, but its reverse branch
// is neither of OPM's: it applies each phase's own LOCAL FORMATION mobility, so
// the fluid going back in is whatever that particular cell happens to be
// mobile in.  That is what produces B-4H's negative oil beside its positive
// gas, and no setting of allow_cf in OPM can reproduce it.
//
// Which of OPM's two branches to reproduce
// ----------------------------------------
// Norne's WELSPECS defaults item 10, so crossflow is allowed and OPM takes the
// cmix_s branch.  Reproducing that exactly needs the wellbore mixture inside
// the assembly kernel -- three more numbers per WellCell and a sign branch in
// the well source term.  That is a larger and less testable change than this
// one, and it is the natural next step.
//
// What is done here is OPM's OTHER branch, the !allow_cf one: the connection
// contributes nothing.  It is a stricter approximation than Norne's deck asks
// for, and it is much closer than what is there now -- because under the
// cmix_s branch the reverse flow carries the well's composition, and for a
// gas-dominated well like B-4H that means very little oil goes back in either
// way.  The oil number, which is the one that is 62% of the deck's history,
// lands in nearly the same place under both of OPM's branches and nowhere near
// it under the current one.
//
// The constrained solve
// ---------------------
// What can be done exactly, and on the host, is to stop the reverse flow:
//
//     find p_wf such that   SUM_k A_k max(0, theta_k - p_wf) = q_target      (2)
//
// Write f(p) = SUM_k A_k max(0, theta_k - p).  f is continuous and piecewise
// linear with breakpoints at the theta_k, it is non-increasing everywhere and
// strictly decreasing wherever it is positive, f(max theta) = 0, and
// f(p) -> +inf as p -> -inf.  So for any q_target > 0 equation (2) has exactly
// one root, and sorting the completions by theta descending finds it in one
// pass: admit them one at a time, and stop at the first whose own theta
// already over-satisfies the target.
//
// Two properties make this safe to switch on:
//
//   - If the unconstrained root of (1) already lies below every theta_k, then
//     no completion was flowing backwards, the max() is inactive at the
//     solution, and (2) returns the SAME pressure.  A well that does not
//     crossflow cannot change.  tests/test_well_control.cpp pins this.
//   - Where they differ, (2) returns a HIGHER pressure -- less drawdown, not
//     more.  That direction is worth stating because the opposite is the
//     natural guess: surely dropping completions means pulling harder?  No.
//     Under (1) the reverse-flowing completions were SUBTRACTING from the
//     total, so the well had to over-produce the rest to net out at the
//     target.  Take the negative terms away and the remaining completions no
//     longer have to cover them.
//
//     The three-completion case in the test makes it concrete: A = 10, 20, 40
//     at theta = 4000, 4010, 2000, target 20000.  Equation (1) gives 2574 psi,
//     at which the well produces 42,971 forward and pushes 22,972 back in.
//     Equation (2) gives 3340 psi, at which it produces exactly 20,000 and the
//     low completion is shut.  Same voidage, half the gross throughput.
//
//     This is why the change is expected to raise oil on Norne rather than
//     lower it.  B-4H's reverse flow is oil-dominated while its forward flow
//     is gas-dominated -- it reports negative oil and 80,000 MSCF/day of gas
//     against Flow's positive oil and ~30,000 -- so removing the crossflow
//     takes the spurious gas production away and stops the oil being pushed
//     back into the formation, both toward Flow.
//
// The caller shuts the completions the solve excluded, so that the assembly
// sees the same active set the pressure was solved for.  That matters: the
// kernel has no sign test, so a completion left open below p_wf would put the
// reverse flow straight back.
//
// This header is deliberately free of CUDA and of every other simulator type,
// so that tests/test_well_control.cpp can exercise the algebra directly.  The
// solve is where a well control goes wrong quietly, and it is arithmetic.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace nvrs {

// One completion, as the solve sees it.
//
// `a` is the productivity SUM-term: rate per unit of drawdown, in the units
// the target is written in.  `theta` is the cell pressure already corrected to
// the well's bottom-hole datum, so that a single p_wf is comparable against
// all of them.  `index` is whatever the caller needs to find the completion
// again -- the solve only sorts and reads it back.
struct WellConn {
  double a = 0.0;
  double theta = 0.0;
  int index = -1;
};

// The result of the constrained solve.
struct WellPwf {
  double pwf = 0.0;        // the flowing pressure meeting the target
  double delivered = 0.0;  // what the active completions actually make at it
  int n_active = 0;        // completions producing at that pressure
  bool limited = false;    // the target could not be met; see below
};

// Solve (2).  `conns` may be in any order and is not modified.
//
// A completion with a_k <= 0 contributes nothing and is ignored -- a dead
// mobility is not a constraint, and including it would put a zero row in the
// sort for no reason.
//
// `target` <= 0 is not this function's business: a well asked for nothing is
// handled by the caller's shut-in path, and returning the highest theta here
// would be a pressure that produces exactly zero, which is a different
// statement.  It is reported as limited with n_active = 0.
//
// `floor_pwf` is the lowest pressure the well is allowed to pull to -- a
// bottom-hole limit, or atmospheric for a history well that has none.  If the
// target needs a pressure below it, the floor binds, the well delivers what it
// can, and `limited` says so.  That is the same switch OPM makes, and the
// caller reports the miss rather than the request.
inline WellPwf solve_pwf_no_crossflow(const std::vector<WellConn>& conns,
                                      double target, double floor_pwf) {
  WellPwf out;
  if (target <= 0.0) {
    out.limited = true;
    // The highest theta is the only pressure at which nothing flows either
    // way; anything higher would drive every completion backwards.
    double hi = 0.0;
    bool any = false;
    for (const WellConn& c : conns)
      if (c.a > 0.0 && (!any || c.theta > hi)) { hi = c.theta; any = true; }
    out.pwf = any ? hi : floor_pwf;
    return out;
  }

  // Live completions only, sorted by the pressure each one sees, highest
  // first: the best-connected, highest-pressure completions are the ones that
  // stay open as the well is pulled down.
  std::vector<WellConn> live;
  live.reserve(conns.size());
  for (const WellConn& c : conns)
    if (c.a > 0.0) live.push_back(c);
  if (live.empty()) {
    out.pwf = floor_pwf;
    out.limited = true;
    return out;
  }
  std::sort(live.begin(), live.end(),
            [](const WellConn& x, const WellConn& y) {
              return x.theta > y.theta;
            });

  // Admit completions one at a time.  With the first m admitted,
  //
  //     f(p) = T_m - S_m p        for p in [theta_m, theta_{m-1}]
  //
  // so the root lies in this interval exactly when f(theta_m) >= target, i.e.
  // when the m completions can meet the target without help from the next one
  // down.  `theta_m` for the last interval is -infinity, which is why the loop
  // falls through to the open-ended case below.
  double S = 0.0, T = 0.0;
  const size_t n = live.size();
  for (size_t m = 0; m < n; ++m) {
    S += live[m].a;
    T += live[m].a * live[m].theta;
    // The bottom of this interval: the next completion's pressure, or nothing
    // below the last one.
    const bool last = (m + 1 == n);
    const double lo = last ? 0.0 : live[m + 1].theta;
    const double f_lo = T - S * lo;
    if (last || f_lo >= target) {
      const double p = (T - target) / S;
      if (!last && p < lo) continue;   // guard against a degenerate interval
      out.pwf = p;
      out.n_active = (int)(m + 1);
      break;
    }
  }
  if (out.n_active == 0) {             // every completion admitted
    out.pwf = (T - target) / S;
    out.n_active = (int)n;
  }

  // The floor.  Below it the well is not allowed to pull, so it delivers what
  // the active set gives at the floor instead of the target.  Re-deriving the
  // active set AT the floor matters: dropping the pressure can only open more
  // completions, never close one.
  if (out.pwf < floor_pwf) {
    out.pwf = floor_pwf;
    out.limited = true;
    out.n_active = 0;
  }

  // What the well actually makes at the pressure it ended on, counting only
  // forward flow -- which is the whole point, and is also the number the
  // report should carry whether or not the target was reached.
  out.delivered = 0.0;
  int n_act = 0;
  for (const WellConn& c : live) {
    const double d = c.theta - out.pwf;
    if (d > 0.0) { out.delivered += c.a * d; ++n_act; }
  }
  out.n_active = n_act;
  return out;
}

// Would completion `theta` produce at this flowing pressure?
//
// The single place the open/shut test is written, so that the solve, the
// completions the caller shuts, and the rate it reports cannot disagree about
// where the boundary is.
inline bool conn_produces(double theta, double pwf) { return theta > pwf; }

// Injector counterpart of solve_pwf_no_crossflow(). At frozen mobility a
// completion takes q_k = A_k max(0, p_bhp - theta_k), where A is surface-rate
// injectivity and theta is cell pressure corrected to the well datum. The
// ceiling is an upper BHP limit on the whole well.
inline WellPwf solve_injector_pwf(const std::vector<WellConn>& conns,
                                  double target, double ceiling_pwf) {
  WellPwf out;
  std::vector<WellConn> live;
  live.reserve(conns.size());
  for (const WellConn& c : conns)
    if (c.a > 0.0) live.push_back(c);

  if (target <= 0.0 || live.empty()) {
    out.pwf = live.empty() ? ceiling_pwf : live.front().theta;
    out.limited = live.empty() && target > 0.0;
    return out;
  }

  // As BHP rises, connections open in ascending threshold order.
  std::sort(live.begin(), live.end(),
            [](const WellConn& x, const WellConn& y) {
              return x.theta < y.theta;
            });
  double S = 0.0, T = 0.0;
  const size_t n = live.size();
  for (size_t m = 0; m < n; ++m) {
    S += live[m].a;
    T += live[m].a * live[m].theta;
    const bool last = (m + 1 == n);
    const double hi = last ? 0.0 : live[m + 1].theta;
    const double f_hi = S * hi - T;
    if (last || f_hi >= target) {
      out.pwf = (target + T) / S;
      out.n_active = (int)(m + 1);
      break;
    }
  }

  if (ceiling_pwf > 0.0 && out.pwf > ceiling_pwf) {
    out.pwf = ceiling_pwf;
    out.limited = true;
  }
  out.delivered = 0.0;
  out.n_active = 0;
  for (const WellConn& c : live) {
    const double draw = out.pwf - c.theta;
    if (draw > 0.0) {
      out.delivered += c.a * draw;
      ++out.n_active;
    }
  }
  return out;
}

inline bool conn_injects(double theta, double pwf) { return pwf > theta; }

}  // namespace nvrs
