// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// pvt_tables.cpp -- reading tabulated black-oil PVT.
//
// The layout, and the trap in it
// ------------------------------
// PVTO and PVTG look like flat lists of numbers and are not.  Each is a set of
// curves, and the file marks where one curve ends with a slash:
//
//     PVTO
//        20.59    50.00  1.10615  1.180     <- Rs = 20.59, bubble point 50 bar
//                 75.00  1.10164  1.247     <- still Rs = 20.59, undersaturated
//                100.00  1.09744  1.315
//                150.00  1.08984  1.453 /   <- end of this Rs curve
//        28.19    70.00  1.12522  1.066     <- next curve
//        ...                              /
//     /                                     <- end of the keyword
//
// Read as one list, the pressures of different curves interleave and the oil
// appears to compress the wrong way.  The record terminator is the only thing
// distinguishing them, so it is what the reader keys on: each slash-terminated
// group is one curve, the first row of the group is its saturated point, and
// the rest is its undersaturated branch.
//
// PVTG is the mirror image, keyed on pressure with a vaporised-oil branch.
#include "pvt_tables.hpp"

#include "deck_token.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>

namespace nvrs {
namespace {

// METRIC -> FIELD, for the quantities a PVT table carries.
constexpr Real kBarToPsi = 14.503773773;
constexpr Real kM3ToBbl  = 6.2898107704;
constexpr Real kM3ToMscf = 35.3146667 / 1000.0;
// Rs is gas volume per oil volume: (sm3 -> Mscf) / (sm3 -> stb).
constexpr Real kRsMetricToField = kM3ToMscf / kM3ToBbl;
// Rv is oil per gas, the reciprocal pairing.
constexpr Real kRvMetricToField = kM3ToBbl / kM3ToMscf;
// Bg is reservoir volume per surface gas volume: (rm3 -> rb) / (sm3 -> scf).
constexpr Real kBgMetricToField = kM3ToBbl / (kM3ToMscf * 1000.0);

const PvtoRecord kNoPvto{};
const PvtgRecord kNoPvtg{};
const PvtwRecord kNoPvtw{};
const PvtDensity kNoDensity{};

}  // namespace

// ---------------------------------------------------------------------------
void PvtTables::read(const std::string& path) {
  DeckTokenizer tk(path);

  const std::string dir = [&] {
    const size_t s = path.find_last_of("/\\");
    return (s == std::string::npos) ? std::string(".") : path.substr(0, s);
  }();

  while (!tk.eof()) {
    std::string kw = tk.next();
    if (kw.empty()) break;
    if (!DeckTokenizer::is_keyword(kw)) continue;
    if (kw == "SCHEDULE" || kw == "END") break;

    // TITLE is free text terminated by a newline rather than by a slash.
    // Reading it as a record can consume DIMENS and desynchronise PROPS.
    if (kw == "TITLE") { tk.skip_title(); continue; }

    if (kw == "METRIC") { field_units_ = false; continue; }
    if (kw == "FIELD")  { field_units_ = true;  continue; }

    if (kw == "PVTO") {
      std::vector<PvtoRecord> region;
      for (;;) {
        // One slash-terminated group is one Rs curve.  An empty group is the
        // bare slash that closes the keyword.
        std::vector<double> v;
        tk.read_values(v);
        if (v.size() < 4) break;
        PvtoRecord r;
        r.rs = v[0];
        for (size_t i = 1; i + 2 < v.size(); i += 3) {
          r.p.push_back(v[i]);
          r.bo.push_back(v[i + 1]);
          r.mu.push_back(v[i + 2]);
        }
        if (!r.p.empty()) region.push_back(std::move(r));
      }
      if (!region.empty()) pvto_.push_back(std::move(region));

    } else if (kw == "ROCK") {
      // One record: the reference pressure and the compressibility.  A deck
      // may give one per PVT region; the first is taken, because `ct` is a
      // single scalar in the physics and a per-region rock compressibility
      // would need somewhere to put it that does not exist yet.
      // Same shape as PVTW: records until the bare slash that closes the
      // keyword.  read_record() consumes its own terminator, so there is no
      // skip afterwards; adding one swallows the keyword that follows.
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        const std::vector<std::string> rec = tk.read_record();
        if (rec.empty()) break;
        if (!have_rock_) {
          rock_pref_ = DeckTokenizer::num(rec, 0, 0.0);
          rock_cr_ = DeckTokenizer::num(rec, 1, 0.0);
          have_rock_ = true;
        }
      }

    } else if (kw == "PVDG") {
      // Dry gas: rows of pressure, Bg and viscosity, with no vaporised oil.
      // Stored as a PVTG whose every node carries the single ratio zero, so
      // everything downstream reads it as it reads a wet gas.
      std::vector<PvtgRecord> region;
      std::vector<double> v;
      tk.read_values(v);
      for (size_t i = 0; i + 2 < v.size(); i += 3) {
        PvtgRecord r;
        r.p = v[i];
        r.rv.push_back(0.0);
        r.bg.push_back(v[i + 1]);
        r.mu.push_back(v[i + 2]);
        region.push_back(std::move(r));
      }
      if (!region.empty()) pvtg_.push_back(std::move(region));

    } else if (kw == "PVDO") {
      // Dead oil: rows of pressure, Bo and viscosity, with no dissolved gas.
      // One saturated node at Rs = 0 whose undersaturated branch is the whole
      // table, which is what a dead oil is.
      std::vector<double> v;
      tk.read_values(v);
      PvtoRecord r;
      r.rs = 0.0;
      for (size_t i = 0; i + 2 < v.size(); i += 3) {
        r.p.push_back(v[i]);
        r.bo.push_back(v[i + 1]);
        r.mu.push_back(v[i + 2]);
      }
      if (!r.p.empty()) {
        std::vector<PvtoRecord> region;
        region.push_back(std::move(r));
        pvto_.push_back(std::move(region));
      }

    } else if (kw == "PVCDO") {
      // Compressible dead oil: p_ref, Bo_ref, co, mu_ref, c_mu.  The device
      // representation already supports an arbitrary-pressure dead-oil
      // branch, so sample the exact exponential Eclipse law.  A zero-
      // compressibility PVCDO (the SPE10 twin) remains exactly constant.
      if (!tk.at_records_end()) {
        const std::vector<std::string> rec = tk.read_record();
        if (rec.size() >= 2) {
          const Real pref = DeckTokenizer::num(rec, 0, 0.0);
          const Real bref = DeckTokenizer::num(rec, 1, 1.0);
          const Real co   = DeckTokenizer::num(rec, 2, 0.0);
          const Real mref = DeckTokenizer::num(rec, 3, 1.0);
          const Real cmu  = DeckTokenizer::num(rec, 4, 0.0);
          const Real span = std::max<Real>(std::abs(pref), 1.0);
          const Real plo = std::max<Real>(0.0, pref - span);
          const Real phi = pref + span;
          PvtoRecord r;
          r.rs = 0.0;
          constexpr int n = 65;
          for (int i = 0; i < n; ++i) {
            const Real p = plo + (phi - plo) * (Real)i / (Real)(n - 1);
            const Real dp = p - pref;
            r.p.push_back(p);
            r.bo.push_back(bref * std::exp(-co * dp));
            r.mu.push_back(mref * std::exp(cmu * dp));
          }
          std::vector<PvtoRecord> region;
          region.push_back(std::move(r));
          pvto_.push_back(std::move(region));
        }
      }
      tk.eat_slash();

    } else if (kw == "PVTG") {
      std::vector<PvtgRecord> region;
      for (;;) {
        std::vector<double> v;
        tk.read_values(v);
        if (v.size() < 4) break;
        PvtgRecord r;
        r.p = v[0];
        for (size_t i = 1; i + 2 < v.size(); i += 3) {
          r.rv.push_back(v[i]);
          r.bg.push_back(v[i + 1]);
          r.mu.push_back(v[i + 2]);
        }
        if (!r.rv.empty()) region.push_back(std::move(r));
      }
      if (!region.empty()) pvtg_.push_back(std::move(region));

    } else if (kw == "PVTW") {
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        const std::vector<std::string> rec = tk.read_record();
        if (rec.empty()) break;
        PvtwRecord w;
        w.p_ref = DeckTokenizer::num(rec, 0, 0.0);
        w.bw    = DeckTokenizer::num(rec, 1, 1.0);
        w.cw    = DeckTokenizer::num(rec, 2, 0.0);
        w.mu    = DeckTokenizer::num(rec, 3, 1.0);
        w.cmu   = DeckTokenizer::num(rec, 4, 0.0);
        pvtw_.push_back(w);
      }

    } else if (kw == "DENSITY") {
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        const std::vector<std::string> rec = tk.read_record();
        if (rec.empty()) break;
        PvtDensity d;
        d.oil   = DeckTokenizer::num(rec, 0, 0.0);
        d.water = DeckTokenizer::num(rec, 1, 0.0);
        d.gas   = DeckTokenizer::num(rec, 2, 0.0);
        density_.push_back(d);
      }

    } else if (kw == "SWOF" || kw == "SGOF") {
      // Equinor's own saturation functions, which until now were never read:
      // the run deck carried a hand-written placeholder instead, and OPM Flow
      // read this file, so the two simulators were comparing different rock.
      //
      // A saturation table is ONE slash-terminated block of numbers, not a
      // record per row.  Reading it with read_record() takes the first three
      // numbers of the whole block and stops, which produced a two-row table
      // from Norne's nineteen -- caught by the startup check, which reported
      // krw at maximum water as zero.
      //
      // Only the first region is kept; a second would need SATNUM, which
      // Norne does not carry.  read_values() stops at the first slash, which
      // is exactly the end of region one.
      std::vector<SatRow>& dst = (kw == "SWOF") ? swof_ : sgof_;
      std::vector<double> v;
      tk.read_values(v);
      if (!dst.empty() || v.size() < 6) continue;

      // Four columns (S, kr, kro, Pc) is the ECLIPSE form; three is accepted
      // for a table with the capillary column left off.  Where the count
      // divides both ways, the saturation column decides it: it is the one
      // that must increase strictly and stay within [0, 1].
      auto plausible = [&](int nc) {
        if (nc <= 0 || v.size() % (size_t)nc) return false;
        const size_t rows = v.size() / (size_t)nc;
        if (rows < 2) return false;
        for (size_t i = 0; i < rows; ++i) {
          const double x = v[i * nc];
          if (!(x >= -1e-9 && x <= 1.0 + 1e-9)) return false;
          if (i && !(x > v[(i - 1) * nc])) return false;
        }
        return true;
      };
      const int ncol = plausible(4) ? 4 : (plausible(3) ? 3 : 0);
      if (!ncol) {
        ignored_.push_back(kw + " (unrecognised column layout)");
        continue;
      }
      for (size_t i = 0; i + (size_t)ncol <= v.size(); i += (size_t)ncol) {
        SatRow r;
        r.s   = (Real)v[i];
        r.kr  = (Real)v[i + 1];
        r.kro = (Real)v[i + 2];
        r.pc  = (ncol == 4) ? (Real)v[i + 3] : (Real)0;
        dst.push_back(r);
      }

    } else if (kw == "EHYSTR") {
      const std::vector<std::string> rec = tk.read_record();
      have_ehystr_ = true;
      ehystr_kr_model_ = (int)DeckTokenizer::num(rec, 1, -1.0);

    } else if (kw == "INCLUDE") {
      std::string inc = tk.next();
      tk.skip_to_slash();
      if (inc.rfind("./", 0) == 0) inc = inc.substr(2);
      try {
        read(dir + "/" + inc);
      } catch (const std::exception&) {
        ignored_.push_back("missing include " + inc);
      }

    } else if (kw == "ECHO" || kw == "NOECHO" || kw == "RUNSPEC" ||
               kw == "GRID" || kw == "EDIT" || kw == "PROPS" ||
               kw == "REGIONS" || kw == "SOLUTION" || kw == "OIL" ||
               kw == "WATER" || kw == "GAS" || kw == "DISGAS" ||
               kw == "VAPOIL" || kw == "UNIFIN" || kw == "UNIFOUT" ||
               kw == "INIT") {
      // Section markers and flags.

    } else {
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        if (tk.eof()) break;
        const std::vector<std::string> rec = tk.read_record();
        if (rec.empty()) break;
      }
      ignored_.push_back(kw);
    }
  }
}

// ---------------------------------------------------------------------------
void PvtTables::to_field_units() {
  if (field_units_) return;

  for (auto& region : pvto_)
    for (PvtoRecord& r : region) {
      r.rs *= kRsMetricToField;
      for (Real& p : r.p) p *= kBarToPsi;
      // Bo is a volume ratio of the same kind in both systems, so unchanged.
    }
  for (auto& region : pvtg_)
    for (PvtgRecord& r : region) {
      r.p *= kBarToPsi;
      for (Real& v : r.rv) v *= kRvMetricToField;
      for (Real& b : r.bg) b *= kBgMetricToField;
    }
  if (have_rock_) {
    // A compressibility is per unit pressure, so it converts the other way
    // round from a pressure.
    rock_pref_ *= kBarToPsi;
    rock_cr_ /= kBarToPsi;
  }
  for (PvtwRecord& w : pvtw_) {
    w.p_ref *= kBarToPsi;
    w.cw /= kBarToPsi;
    w.cmu /= kBarToPsi;
  }
  // Densities stay as the deck gave them; the caller decides what to do with
  // them, and the simulator's own gravity term wants specific gravities.
  field_units_ = true;
}

// ---------------------------------------------------------------------------
int PvtTables::n_regions() const {
  return (int)std::max(std::max(pvto_.size(), pvtg_.size()),
                       std::max(pvtw_.size(), density_.size()));
}

const std::vector<PvtoRecord>& PvtTables::pvto(int region) const {
  static const std::vector<PvtoRecord> none;
  if (pvto_.empty()) return none;
  return pvto_[std::min<size_t>((size_t)std::max(region, 0), pvto_.size() - 1)];
}

const std::vector<PvtgRecord>& PvtTables::pvtg(int region) const {
  static const std::vector<PvtgRecord> none;
  if (pvtg_.empty()) return none;
  return pvtg_[std::min<size_t>((size_t)std::max(region, 0), pvtg_.size() - 1)];
}

const PvtwRecord& PvtTables::pvtw(int region) const {
  if (pvtw_.empty()) return kNoPvtw;
  return pvtw_[std::min<size_t>((size_t)std::max(region, 0), pvtw_.size() - 1)];
}

const PvtDensity& PvtTables::density(int region) const {
  if (density_.empty()) return kNoDensity;
  return density_[std::min<size_t>((size_t)std::max(region, 0),
                                   density_.size() - 1)];
}

// ---------------------------------------------------------------------------
// Host evaluation, mirroring the device functions in the header.
// ---------------------------------------------------------------------------
Real PvtTables::rs_sat(Real p, int region) const {
  const std::vector<PvtoRecord>& t = pvto(region);
  if (t.empty()) return 0;
  std::vector<Real> pb(t.size()), rs(t.size());
  for (size_t i = 0; i < t.size(); ++i) { pb[i] = t[i].p.front(); rs[i] = t[i].rs; }
  return pvt_lerp(pb.data(), rs.data(), (int)pb.size(), p);
}

void PvtTables::oil(Real p, Real rs_in, Real& bo, Real& mu, int region) const {
  bo = 1; mu = 1;
  const std::vector<PvtoRecord>& t = pvto(region);
  if (t.empty()) return;

  std::vector<Real> pb(t.size()), rs(t.size()), bs(t.size()), ms(t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    pb[i] = t[i].p.front(); rs[i] = t[i].rs;
    bs[i] = t[i].bo.front(); ms[i] = t[i].mu.front();
  }
  const Real rsb = pvt_lerp(pb.data(), rs.data(), (int)pb.size(), p);
  if (rs_in >= rsb - 1e-12) {
    bo = pvt_lerp(pb.data(), bs.data(), (int)pb.size(), p);
    mu = pvt_lerp(pb.data(), ms.data(), (int)pb.size(), p);
    return;
  }
  const int i = pvt_locate(rs.data(), (int)rs.size(), rs_in);
  const int j = std::min(i + 1, (int)rs.size() - 1);
  const Real drs = rs[j] - rs[i];
  const Real w = (drs != 0) ? (rs_in - rs[i]) / drs : 0;
  const Real bo_i = pvt_lerp(t[i].p.data(), t[i].bo.data(), (int)t[i].p.size(), p);
  const Real mu_i = pvt_lerp(t[i].p.data(), t[i].mu.data(), (int)t[i].p.size(), p);
  const Real bo_j = pvt_lerp(t[j].p.data(), t[j].bo.data(), (int)t[j].p.size(), p);
  const Real mu_j = pvt_lerp(t[j].p.data(), t[j].mu.data(), (int)t[j].p.size(), p);
  bo = bo_i + w * (bo_j - bo_i);
  mu = mu_i + w * (mu_j - mu_i);
}

void PvtTables::gas(Real p, Real rv, Real& bg, Real& mu, int region) const {
  const HostPvt h = flatten_pvt(*this, region);
  const DevicePvt d = h.view();
  if (rv < 0) dev_gas(d, p, bg, mu);
  else dev_gas(d, p, rv * (field_units_ ? 0.001 : 1.0), bg, mu);
}

Real PvtTables::rv_sat(Real p, int region) const {
  const std::vector<PvtgRecord>& t = pvtg(region);
  if (t.empty()) return 0;
  std::vector<Real> pp(t.size()), rr(t.size());
  for (size_t i = 0; i < t.size(); ++i) { pp[i] = t[i].p; rr[i] = t[i].rv.front(); }
  return pvt_lerp(pp.data(), rr.data(), (int)pp.size(), p);
}

void PvtTables::water(Real p, Real& bw, Real& mu, int region) const {
  const PvtwRecord& w = pvtw(region);
  const Real x = w.cw * (p - w.p_ref);
  bw = w.bw / (1 + x + 0.5 * x * x);
  const Real y = w.cmu * (p - w.p_ref);
  mu = w.mu / (1 + y + 0.5 * y * y);
}

// ---------------------------------------------------------------------------
// Flattening
// ---------------------------------------------------------------------------
// The curve-per-Rs layout the parser produces is right for reading and wrong
// for a kernel, which wants flat arrays it can binary-search.  Flattening is
// arithmetic, so it lives here rather than behind the CUDA guard: the upload
// below copies the result to the device, and HostPvt::view() hands the very
// same arrays to the evaluators on the host.  That is what lets the tests
// exercise the device code path without a device.
HostPvt flatten_pvt(const PvtTables& t, int region) {
  HostPvt h;
  h.rv_scale = t.rv_scale();
  const std::vector<PvtoRecord>& o = t.pvto(region);
  const std::vector<PvtgRecord>& g = t.pvtg(region);

  // Rs changes units here, and this is the only place it may.
  //
  // A FIELD deck states Rs in Mscf/stb, which is what to_field_units() gives
  // and what the tables above hold.  The solver does not work in those units:
  // its deck keyword RS is scf/stb, and the gas equation requires that, since
  //
  //     accumulation  = acc[rb/day] * bg[scf/rb] * Sg          -> scf/day
  //     dissolved gas = acc[rb/day] * bo[stb/rb] * So * Rs
  //
  // and the second is only scf/day if Rs is scf/stb.  Handing the kernel
  // Mscf/stb makes the dissolved-gas term a thousand times too small -- the
  // oil carries essentially no gas, the two equations decouple, and both the
  // physics and the conditioning of the pressure system go with it.
  //
  // The tables stay honest to the deck's convention; the conversion belongs at
  // the boundary between the deck's units and the solver's, which is here.
  const Real rs_to_solver = t.is_field() ? 1000.0 : 1.0;
  const Real rv_to_solver = t.is_field() ? 0.001 : 1.0;   // the reciprocal pairing

  h.us_off.assign(o.size() + 1, 0);
  for (size_t i = 0; i < o.size(); ++i) {
    h.sat_rs.push_back(o[i].rs * rs_to_solver);
    h.sat_p.push_back(o[i].p.front());
    h.sat_bo.push_back(o[i].bo.front());
    h.sat_mu.push_back(o[i].mu.front());
    // The whole curve goes into the undersaturated branch, saturated point
    // included, so interpolating upward from the bubble point is continuous.
    for (size_t k = 0; k < o[i].p.size(); ++k) {
      h.us_p.push_back(o[i].p[k]);
      h.us_bo.push_back(o[i].bo[k]);
      h.us_mu.push_back(o[i].mu[k]);
    }
    h.us_off[i + 1] = (int)h.us_p.size();
  }

  for (const PvtgRecord& r : g) {
    h.gsat_p.push_back(r.p);
    // VAPOIL uses the SATURATED end of each PVTG branch (the first row): its
    // Rv is the oil component carried in the gas phase, and Bg/mu must come
    // from that same composition.  The previous dry-end selection made Rv
    // identically zero and mixed OPM's wet-gas deck with dry-gas properties.
    // Norne's reservoir pressure stays below the high-pressure turn in the
    // saturated Bg curve, so its local compressibility remains positive.
    h.gsat_rv.push_back((r.rv.empty() ? 0 : r.rv.front()) * rv_to_solver);
    h.gsat_bg.push_back(r.bg.empty() ? 1 : r.bg.front());
    h.gsat_mu.push_back(r.mu.empty() ? 0.02 : r.mu.front());
  }

  h.gus_off.assign(g.size() + 1, 0);
  for (size_t i = 0; i < g.size(); ++i) {
    const PvtgRecord& r = g[i];
    // Eclipse writes each branch from saturated (largest Rv) to dry gas.
    // Store it ascending for binary search, dropping occasional repeated rows.
    for (size_t kk = r.rv.size(); kk > 0; --kk) {
      const size_t k = kk - 1;
      const Real rv = r.rv[k] * rv_to_solver;
      if (!h.gus_rv.empty() && (int)h.gus_rv.size() > h.gus_off[i] &&
          rv == h.gus_rv.back())
        continue;
      const Real bg = r.bg[k] > 0 ? r.bg[k] : (Real)1;
      const Real mu = r.mu[k] > 0 ? r.mu[k] : (Real)0.02;
      h.gus_rv.push_back(rv);
      h.gus_invbg.push_back((Real)1 / bg);
      h.gus_invbgmu.push_back((Real)1 / (bg * mu));
    }
    h.gus_off[i + 1] = (int)h.gus_rv.size();
  }

  const PvtwRecord& w = t.pvtw(region);
  h.w_pref = w.p_ref; h.w_bw = w.bw; h.w_cw = w.cw;
  h.w_mu = w.mu; h.w_cmu = w.cmu;
  h.valid = !o.empty() || !g.empty();
  return h;
}

DevicePvt HostPvt::view() const {
  DevicePvt d;
  // VAPOIL travels with the view whether or not the tables loaded, so a
  // deck that switches vaporised oil off still reads as off downstream.
  d.rv_scale = rv_scale;
  if (!valid) return d;
  if (!sat_rs.empty()) {
    d.sat_rs = sat_rs.data(); d.sat_p = sat_p.data();
    d.sat_bo = sat_bo.data(); d.sat_mu = sat_mu.data();
    d.n_sat = (int)sat_rs.size();
    d.us_off = us_off.data();
    d.us_p = us_p.data(); d.us_bo = us_bo.data(); d.us_mu = us_mu.data();
  }
  if (!gsat_p.empty()) {
    d.gsat_p = gsat_p.data(); d.gsat_rv = gsat_rv.data();
    d.gsat_bg = gsat_bg.data(); d.gsat_mu = gsat_mu.data();
    d.n_gsat = (int)gsat_p.size();
    d.gus_off = gus_off.data(); d.gus_rv = gus_rv.data();
    d.gus_invbg = gus_invbg.data(); d.gus_invbgmu = gus_invbgmu.data();
  }
  d.w_pref = w_pref; d.w_bw = w_bw; d.w_cw = w_cw;
  d.w_mu = w_mu; d.w_cmu = w_cmu;
  d.valid = true;
  return d;
}


// ---------------------------------------------------------------------------
std::string PvtTables::summary() const {
  std::ostringstream os;
  char b[256];
  const char* pu = field_units_ ? "psi" : "bar";

  os << "PVT tables\n";
  std::snprintf(b, sizeof b, "  units           : %s\n",
                field_units_ ? "FIELD" : "METRIC"); os << b;
  std::snprintf(b, sizeof b, "  regions         : %d\n", n_regions()); os << b;

  if (!pvto_.empty()) {
    const std::vector<PvtoRecord>& t = pvto(0);
    size_t nb = 0;
    for (const PvtoRecord& r : t) nb += r.p.size();
    std::snprintf(b, sizeof b,
                  "  PVTO            : %d Rs curves, %d points, "
                  "Rs %.4g .. %.4g, bubble point %.4g .. %.4g %s\n",
                  (int)t.size(), (int)nb, (double)t.front().rs,
                  (double)t.back().rs, (double)t.front().p.front(),
                  (double)t.back().p.front(), pu);
    os << b;
    std::snprintf(b, sizeof b, "                    Bo %.4f .. %.4f, "
                  "viscosity %.4f .. %.4f cP\n",
                  (double)t.back().bo.back(), (double)t.front().bo.front(),
                  (double)t.back().mu.front(), (double)t.front().mu.back());
    os << b;
  } else {
    os << "  PVTO            : absent (dead oil)\n";
  }

  if (!pvtg_.empty()) {
    const std::vector<PvtgRecord>& t = pvtg(0);
    std::snprintf(b, sizeof b,
                  "  PVTG            : %d pressure nodes, %.4g .. %.4g %s, "
                  "Bg %.5g .. %.5g, Rv %.4g .. %.4g\n",
                  (int)t.size(), (double)t.front().p, (double)t.back().p, pu,
                  (double)t.back().bg.front(), (double)t.front().bg.front(),
                  (double)t.front().rv.front(), (double)t.back().rv.front());
    os << b;
  } else {
    os << "  PVTG            : absent (dry gas)\n";
  }

  if (!pvtw_.empty()) {
    const PvtwRecord& w = pvtw(0);
    std::snprintf(b, sizeof b,
                  "  PVTW            : Bw %.5f at %.4g %s, cw %.4g, mu %.4f cP\n",
                  (double)w.bw, (double)w.p_ref, pu, (double)w.cw, (double)w.mu);
    os << b;
  }
  if (!density_.empty()) {
    const PvtDensity& dd = density(0);
    std::snprintf(b, sizeof b,
                  "  DENSITY         : oil %.2f, water %.2f, gas %.4f\n",
                  (double)dd.oil, (double)dd.water, (double)dd.gas);
    os << b;
  }
  return os.str();
}

}  // namespace nvrs
