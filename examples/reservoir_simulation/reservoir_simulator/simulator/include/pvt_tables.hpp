// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// pvt_tables.hpp -- tabulated black-oil PVT: live oil, wet gas, water.
//
// What a PVT table says, and why the shape is awkward
// ---------------------------------------------------
// The simulator has until now taken formation volume factors and viscosities
// as constants from the deck.  A field model states them as tables, and the
// tables are two-dimensional in a way that is not obvious from their layout.
//
// PVTO gives oil properties against dissolved gas Rs *and* pressure.  Each
// record begins with an Rs value and the bubble-point pressure that goes with
// it, then continues with further pressures at the *same* Rs -- the
// undersaturated branch, where the oil is above its bubble point and can
// dissolve no more gas:
//
//     20.59   50.00  1.10615  1.180     <- saturated: Rs = 20.59 at Pb = 50
//             75.00  1.10164  1.247     <- undersaturated, still Rs = 20.59
//            100.00  1.09744  1.315
//
// So the table is a set of curves, one per Rs, each starting on the saturated
// line.  Interpolating it means finding which two Rs curves bracket the state
// and then interpolating along each in pressure.  Treating it as a flat list
// of (P, Bo) triples -- which its layout invites -- silently mixes points from
// different Rs curves and produces an oil that compresses the wrong way.
//
// PVTG is the mirror image for wet gas: keyed on pressure, with a vaporised
// oil ratio Rv branch at each.
//
// Saturated and undersaturated
// ----------------------------
// Below the bubble point the oil is saturated: Rs is whatever the pressure
// supports, and both Rs and Bo are read off the saturated line.  Above it the
// oil holds the Rs it already has and Bo follows the undersaturated branch for
// that Rs.  Which regime a cell is in is a property of the state, not of the
// table, so the evaluation takes both pressure and Rs and decides.
//
// Device use
// ----------
// The tables are uploaded as flat arrays and evaluated by __device__ functions
// on a per-cell basis.  They are small -- Norne's PVTO is a few hundred points
// -- so they are read straight from global memory and rely on the cache rather
// than being staged in shared memory.
#pragma once

#include <cmath>
#include <string>
#include <vector>

// Reading and evaluating a PVT table is arithmetic, not CUDA.  Keeping this
// header free of cuda_runtime.h lets the tables and their tests build with a
// plain host compiler, which is what makes tests/test_pvt.cpp runnable
// anywhere; the evaluators below are then marked for the device only when
// compiled by nvcc.  The alias matches common.hpp exactly, so including both
// together is well formed.
#if defined(__CUDACC__)
#include "common.hpp"
#define NVRS_PVT_HD __host__ __device__
#else
#define NVRS_PVT_HD
#endif

namespace nvrs {

#if !defined(__CUDACC__)
using Real = double;
#endif

// Eclipse/OPM ROCK pore-volume law.  Keeping this tiny piece of arithmetic in
// the table header makes the residual, its Jacobian and host-side inventory
// use one definition, and lets the plain-C++ PVT test pin the convention.
NVRS_PVT_HD inline Real rock_pv_mult(Real p, Real pref, Real cr) {
  const Real x = cr * (p - pref);
  return (Real)1 + x + (Real)0.5 * x * x;
}

NVRS_PVT_HD inline Real rock_pv_mult_deriv(Real p, Real pref, Real cr) {
  return cr * ((Real)1 + cr * (p - pref));
}

// One saturated node of PVTO, with its undersaturated branch.
struct PvtoRecord {
  Real rs = 0;                    // dissolved gas [sm3/sm3 or Mscf/stb]
  std::vector<Real> p;            // pressures, first is the bubble point
  std::vector<Real> bo;           // oil formation volume factor
  std::vector<Real> mu;           // oil viscosity [cP]
};

// One pressure node of PVTG, with its vaporised-oil branch.
struct PvtgRecord {
  Real p = 0;                     // gas pressure
  std::vector<Real> rv;           // vaporised oil ratio, first is saturated
  std::vector<Real> bg;           // gas formation volume factor
  std::vector<Real> mu;           // gas viscosity [cP]
};

// PVTW: water is described by one reference point and two compressibilities.
struct PvtwRecord {
  Real p_ref = 0;
  Real bw = 1;
  Real cw = 0;                    // water compressibility [1/pressure]
  Real mu = 1;
  Real cmu = 0;                   // viscosibility [1/pressure]
};

struct PvtDensity {
  Real oil = 0, water = 0, gas = 0;   // surface densities
};

// ---------------------------------------------------------------------------
// Host-side tables, one set per PVT region.
// ---------------------------------------------------------------------------
class PvtTables {
 public:
  // Parse PVTO, PVTG, PVTW, DENSITY and ROCK from a deck.  INCLUDE is
  // followed; reading stops at SCHEDULE.
  void read(const std::string& path);

  bool have_pvto() const { return !pvto_.empty(); }
  bool have_pvtg() const { return !pvtg_.empty(); }
  bool have_pvtw() const { return !pvtw_.empty(); }
  bool have_density() const { return !density_.empty(); }

  // Saturation functions read from the FIELD deck, in ECLIPSE's own column
  // order: SWOF is Sw krw krow Pcow and SGOF is Sg krg krog Pcog.
  //
  // NVRS's own deck format puts the OIL curve second and the water or gas
  // curve third -- the reverse -- so these are kept separate and converted
  // once, where they are handed over.  Reading a real deck with the run
  // deck's convention swaps krw with krow, which is silent and ruinous.
  struct SatRow { Real s, kr, kro, pc; };
  const std::vector<SatRow>& swof() const { return swof_; }
  const std::vector<SatRow>& sgof() const { return sgof_; }
  bool have_satfunc() const { return !swof_.empty() && !sgof_.empty(); }

  // EHYSTR configuration.  Norne requests HYSTMOD=0 (Carlson) and assigns
  // IMBNUM=2 with I-prefixed endpoint properties.
  bool have_ehystr() const { return have_ehystr_; }
  int ehystr_kr_model() const { return ehystr_kr_model_; }

  // ROCK: the rock compressibility, and the pressure it is quoted at.
  // physics.hpp's `ct` means rock compressibility alone once PVT tables are
  // present, which is what this supplies.  The reference pressure is recorded
  // but unused: the accumulation term works in increments, so a shift in the
  // datum cancels.
  bool have_rock() const { return have_rock_; }
  Real rock_cr() const { return rock_cr_; }
  Real rock_pref() const { return rock_pref_; }

  const std::vector<PvtoRecord>& pvto(int region = 0) const;
  const std::vector<PvtgRecord>& pvtg(int region = 0) const;
  const PvtwRecord& pvtw(int region = 0) const;
  const PvtDensity& density(int region = 0) const;
  int n_regions() const;

  // Convert every table from METRIC to FIELD units.  Idempotent.
  void to_field_units();
  bool is_field() const { return field_units_; }
  void set_field_units(bool f) { field_units_ = f; }
  // VAPOIL: 0 removes vaporised oil, 1 (default) keeps PVTG's saturated Rv.
  Real rv_scale() const { return rv_scale_; }
  void set_rv_scale(Real s) { rv_scale_ = s; }

  // ---- host evaluation, also used to check the device path ---------------
  // Saturated dissolved gas at a pressure: the Rs the bubble-point line gives.
  Real rs_sat(Real p, int region = 0) const;
  // Oil formation volume factor and viscosity at (p, Rs).
  void oil(Real p, Real rs, Real& bo, Real& mu, int region = 0) const;
  // Gas, at (p, Rv). Passing a negative Rv asks for the saturated value.
  void gas(Real p, Real rv, Real& bg, Real& mu, int region = 0) const;
  void water(Real p, Real& bw, Real& mu, int region = 0) const;
  // Saturated vaporised oil at a pressure.
  Real rv_sat(Real p, int region = 0) const;

  std::string summary() const;
  const std::vector<std::string>& ignored_keywords() const { return ignored_; }

 private:
  std::vector<std::vector<PvtoRecord>> pvto_;   // [region][node]
  std::vector<std::vector<PvtgRecord>> pvtg_;
  std::vector<PvtwRecord> pvtw_;
  std::vector<PvtDensity> density_;
  std::vector<SatRow> swof_, sgof_;
  bool have_ehystr_ = false;
  bool have_rock_ = false;
  Real rock_cr_ = 0, rock_pref_ = 0;
  int ehystr_kr_model_ = -1;
  std::vector<std::string> ignored_;
  bool field_units_ = false;
  Real rv_scale_ = 1;
};

// ---------------------------------------------------------------------------
// Device tables
// ---------------------------------------------------------------------------
// A flattened form of one region's tables, suitable for a kernel.  The
// saturated lines are stored separately from the undersaturated branches
// because the saturated lookup is the common case and wants to be one search.
struct DevicePvt {
  // Saturated oil line: Rs and bubble-point pressure, ascending in both.
  const Real* sat_rs = nullptr;
  const Real* sat_p = nullptr;
  const Real* sat_bo = nullptr;
  const Real* sat_mu = nullptr;
  int n_sat = 0;

  // Undersaturated branches, concatenated. Branch i covers
  // [us_off[i], us_off[i+1]) and belongs to saturated node i.
  const int* us_off = nullptr;
  const Real* us_p = nullptr;
  const Real* us_bo = nullptr;
  const Real* us_mu = nullptr;

  // Saturated gas line.
  const Real* gsat_p = nullptr;
  const Real* gsat_rv = nullptr;
  const Real* gsat_bg = nullptr;
  const Real* gsat_mu = nullptr;
  int n_gsat = 0;

  // Wet-gas branches, concatenated by pressure node. OPM interpolates the
  // primary tabulated quantities 1/Bg and 1/(Bg*mu), not Bg and mu.
  const int* gus_off = nullptr;
  const Real* gus_rv = nullptr;
  const Real* gus_invbg = nullptr;
  const Real* gus_invbgmu = nullptr;

  // Water, as a reference point plus compressibilities.
  Real w_pref = 0, w_bw = 1, w_cw = 0, w_mu = 1, w_cmu = 0;

  // VAPOIL: 1 keeps the saturated Rv from PVTG, 0 removes vaporised oil
  // entirely.  It scales Rv rather than gating each consumer, so accumulation,
  // flux, wells and the in-place report cannot disagree about it.
  Real rv_scale = 1;

  bool valid = false;
};

// The same tables flattened, still in host memory and owning their arrays.
// view() returns a DevicePvt pointing into them, so every evaluator below can
// be run -- and tested -- on the host, against the identical data the kernel
// sees.  A PVT bug that only shows up on a GPU is a bug in the upload, not in
// the physics, and this keeps the two separable.
struct HostPvt {
  std::vector<Real> sat_rs, sat_p, sat_bo, sat_mu;
  std::vector<int> us_off;
  std::vector<Real> us_p, us_bo, us_mu;
  std::vector<Real> gsat_p, gsat_rv, gsat_bg, gsat_mu;
  std::vector<int> gus_off;
  std::vector<Real> gus_rv, gus_invbg, gus_invbgmu;
  Real w_pref = 0, w_bw = 1, w_cw = 0, w_mu = 1, w_cmu = 0;
  Real rv_scale = 1;                  // VAPOIL; see DevicePvt::rv_scale
  bool valid = false;

  DevicePvt view() const;
};

HostPvt flatten_pvt(const PvtTables& t, int region = 0);


// Upload one region's tables to the device.  The returned DevicePvt owns
// device memory that free_device_pvt() releases.
DevicePvt upload_pvt(const PvtTables& t, int region = 0);

// Every region at once, as a DEVICE array of n_regions DevicePvt.
//
// PVTNUM gives each cell a region and the deck can carry several --
// Norne has two.  Running region 1 everywhere gives the wrong oil
// compressibility, the wrong viscosity and the wrong dissolved gas in
// every cell of the other region.  Each entry is built by the same
// flatten-and-upload as upload_pvt(), so a one-region deck gets exactly
// what it got before, reached through one extra indirection.
DevicePvt* upload_pvt_regions(const PvtTables& t, int* n_regions_out);
void free_device_pvt(DevicePvt& d);

// ---------------------------------------------------------------------------
// Evaluation, callable from host and device
// ---------------------------------------------------------------------------

// Index of the last node whose x is <= v, clamped to [0, n-2]. Binary search:
// the tables are small but a linear scan inside a per-cell kernel is still the
// wrong shape, and a search is branch-predictable here.
NVRS_PVT_HD inline int pvt_locate(const Real* x, int n, Real v) {
  if (n < 2) return 0;
  int lo = 0, hi = n - 1;
  if (v <= x[0]) return 0;
  if (v >= x[n - 1]) return n - 2;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (x[mid] <= v) lo = mid; else hi = mid;
  }
  return lo;
}

NVRS_PVT_HD inline Real pvt_lerp(const Real* x, const Real* y, int n,
                                         Real v) {
  if (n <= 0) return 0;
  if (n == 1) return y[0];
  const int i = pvt_locate(x, n, v);
  const Real dx = x[i + 1] - x[i];
  const Real t = (dx != 0) ? (v - x[i]) / dx : 0;
  return y[i] + t * (y[i + 1] - y[i]);
}

// Saturated dissolved gas at a pressure.
NVRS_PVT_HD inline Real dev_rs_sat(const DevicePvt& d, Real p) {
  if (!d.valid || d.n_sat <= 0) return 0;
  return pvt_lerp(d.sat_p, d.sat_rs, d.n_sat, p);
}

// Oil properties at (p, rs).
//
// Below the bubble point for this Rs the oil is saturated and both properties
// come off the saturated line. Above it the oil keeps its Rs and follows that
// node's undersaturated branch. The branch is interpolated between the two
// saturated nodes that bracket Rs, which is what keeps the surface smooth
// across a bubble point rather than stepping at each tabulated Rs.
NVRS_PVT_HD inline void dev_oil(const DevicePvt& d, Real p, Real rs,
                                        Real& bo, Real& mu) {
  bo = 1; mu = 1;
  if (!d.valid || d.n_sat <= 0) return;

  // PVDO is flattened as one Rs=0 curve.  There is no neighbouring saturated
  // node to interpolate against; sample that sole pressure branch directly.
  // The former saturated path used j=i+1 and read past sat_bo/sat_mu.
  if (d.n_sat == 1) {
    const int a0 = d.us_off[0], b0 = d.us_off[1];
    const int n = b0 - a0;
    if (n <= 0) {
      bo = d.sat_bo[0]; mu = d.sat_mu[0];
      return;
    }
    if (n == 1) {
      bo = d.us_bo[a0]; mu = d.us_mu[a0];
      return;
    }
    const int i = pvt_locate(d.us_p + a0, n, p);
    const Real dp = d.us_p[a0 + i + 1] - d.us_p[a0 + i];
    const Real w = (dp != 0) ? (p - d.us_p[a0 + i]) / dp : 0;
    const Real bi = (Real)1 / d.us_bo[a0 + i];
    const Real bj = (Real)1 / d.us_bo[a0 + i + 1];
    const Real bmi = bi / d.us_mu[a0 + i];
    const Real bmj = bj / d.us_mu[a0 + i + 1];
    const Real b = bi + w * (bj - bi);
    const Real bm = bmi + w * (bmj - bmi);
    bo = (b > 0) ? (Real)1 / b : (Real)1;
    mu = (bm > 0) ? b / bm : (Real)1;
    return;
  }

  const Real rs_b = dev_rs_sat(d, p);
  if (rs >= rs_b - 1e-12) {
    // Saturated: OPM interpolates inverse B and inverse(B*mu), then recovers
    // viscosity from their ratio.  Interpolating B and mu directly is a
    // different PVT model even along the bubble-point line.
    const int i = pvt_locate(d.sat_p, d.n_sat, p);
    const int j = i + 1;
    const Real dp = d.sat_p[j] - d.sat_p[i];
    const Real a = (dp != 0) ? (p - d.sat_p[i]) / dp : 0;
    const Real bi = (Real)1 / d.sat_bo[i];
    const Real bj = (Real)1 / d.sat_bo[j];
    const Real bmi = bi / d.sat_mu[i];
    const Real bmj = bj / d.sat_mu[j];
    const Real b = bi + a * (bj - bi);
    const Real bm = bmi + a * (bmj - bmi);
    bo = (b > 0) ? (Real)1 / b : (Real)1;
    mu = (bm > 0) ? b / bm : (Real)1;
    return;
  }

  // Undersaturated: OPM's LiveOilPvt uses LeftExtreme guided interpolation.
  // Sample each Rs branch at a pressure shifted parallel to the saturated
  // (bubble-point) boundary, rather than vertically at the same pressure.
  const int i = pvt_locate(d.sat_rs, d.n_sat, rs);
  const int j = (i + 1 < d.n_sat) ? i + 1 : i;
  const Real drs = d.sat_rs[j] - d.sat_rs[i];
  const Real w = (drs != 0) ? (rs - d.sat_rs[i]) / drs : 0;
  const Real shift = d.sat_p[j] - d.sat_p[i];
  const Real p_i = p - w * shift;
  const Real p_j = p + ((Real)1 - w) * shift;

  Real bo_i, mu_i, bo_j, mu_j;
  {
    const int a = d.us_off[i], b = d.us_off[i + 1];
    const int n = b - a;
    bo_i = (n > 0) ? pvt_lerp(d.us_p + a, d.us_bo + a, n, p_i) : d.sat_bo[i];
    mu_i = (n > 0) ? pvt_lerp(d.us_p + a, d.us_mu + a, n, p_i) : d.sat_mu[i];
  }
  {
    const int a = d.us_off[j], b = d.us_off[j + 1];
    const int n = b - a;
    bo_j = (n > 0) ? pvt_lerp(d.us_p + a, d.us_bo + a, n, p_j) : d.sat_bo[j];
    mu_j = (n > 0) ? pvt_lerp(d.us_p + a, d.us_mu + a, n, p_j) : d.sat_mu[j];
  }
  const Real bi = (Real)1 / bo_i;
  const Real bj = (Real)1 / bo_j;
  const Real bmi = bi / mu_i;
  const Real bmj = bj / mu_j;
  const Real b = bi + w * (bj - bi);
  const Real bm = bmi + w * (bmj - bmi);
  bo = (b > 0) ? (Real)1 / b : (Real)1;
  mu = (bm > 0) ? b / bm : (Real)1;
}

// Saturated gas properties at a pressure.
NVRS_PVT_HD inline void dev_gas(const DevicePvt& d, Real p, Real& bg,
                                        Real& mu) {
  bg = 1; mu = 0.02;
  if (!d.valid || d.n_gsat <= 0) return;
  if (d.n_gsat == 1) {
    bg = d.gsat_bg[0]; mu = d.gsat_mu[0];
    return;
  }
  // OPM stores and interpolates b and b/mu. Interpolating B and mu directly
  // gives a subtly different compressibility even on the saturated line.
  const int i = pvt_locate(d.gsat_p, d.n_gsat, p);
  const int j = i + 1;
  const Real dp = d.gsat_p[j] - d.gsat_p[i];
  const Real a = (dp != 0) ? (p - d.gsat_p[i]) / dp : 0;
  const Real bi = (Real)1 / d.gsat_bg[i];
  const Real bj = (Real)1 / d.gsat_bg[j];
  const Real bmi = bi / d.gsat_mu[i];
  const Real bmj = bj / d.gsat_mu[j];
  const Real b = bi + a * (bj - bi);
  const Real bm = bmi + a * (bmj - bmi);
  bg = (b > 0) ? (Real)1 / b : (Real)1;
  mu = (bm > 0) ? b / bm : (Real)0.02;
}

// Wet-gas properties at (p,Rv), matching OPM WetGasPvt's RightExtreme
// interpolation. The two pressure branches are sampled at shifted Rv values
// so interpolation follows the saturated boundary rather than cutting across
// it. This matters when VAPPARS moves gas away from saturated composition.
NVRS_PVT_HD inline void dev_gas(const DevicePvt& d, Real p, Real rv,
                                Real& bg, Real& mu) {
  bg = 1; mu = (Real)0.02;
  if (!d.valid || d.n_gsat <= 0) return;
  if (!d.gus_off || !d.gus_rv || !d.gus_invbg || !d.gus_invbgmu ||
      d.n_gsat < 2) {
    dev_gas(d, p, bg, mu);
    return;
  }

  const int i = pvt_locate(d.gsat_p, d.n_gsat, p);
  const int j = i + 1;
  const Real dp = d.gsat_p[j] - d.gsat_p[i];
  const Real alpha = (dp != 0) ? (p - d.gsat_p[i]) / dp : 0;
  const Real shift = d.gsat_rv[j] - d.gsat_rv[i];
  const Real rv_i = rv - alpha * shift;
  const Real rv_j = rv + ((Real)1 - alpha) * shift;

  const int ai = d.gus_off[i], ni = d.gus_off[i + 1] - ai;
  const int aj = d.gus_off[j], nj = d.gus_off[j + 1] - aj;
  const Real bi = (ni > 0) ? pvt_lerp(d.gus_rv + ai, d.gus_invbg + ai,
                                      ni, rv_i) : (Real)1 / d.gsat_bg[i];
  const Real bj = (nj > 0) ? pvt_lerp(d.gus_rv + aj, d.gus_invbg + aj,
                                      nj, rv_j) : (Real)1 / d.gsat_bg[j];
  const Real bmi = (ni > 0) ? pvt_lerp(d.gus_rv + ai,
                                       d.gus_invbgmu + ai, ni, rv_i)
                            : bi / d.gsat_mu[i];
  const Real bmj = (nj > 0) ? pvt_lerp(d.gus_rv + aj,
                                       d.gus_invbgmu + aj, nj, rv_j)
                            : bj / d.gsat_mu[j];
  const Real invbg = bi + alpha * (bj - bi);
  const Real invbgmu = bmi + alpha * (bmj - bmi);
  bg = (invbg > 0) ? (Real)1 / invbg : (Real)1;
  mu = (invbgmu > 0) ? invbg / invbgmu : (Real)0.02;
}

NVRS_PVT_HD inline Real dev_rv_sat(const DevicePvt& d, Real p) {
  if (!d.valid || d.n_gsat <= 0) return 0;
  // rv_scale is 1 unless the deck set VAPOIL 0.  Every use of Rv in the
  // residual, the Jacobian, the well rates and field_oil_in_place() reaches
  // the tables through here, so one multiply switches all of them together.
  return d.rv_scale * pvt_lerp(d.gsat_p, d.gsat_rv, d.n_gsat, p);
}

// Water, from the reference point and its compressibilities.  The standard
// first-order expansion, which is what PVTW's four numbers describe.
NVRS_PVT_HD inline void dev_water(const DevicePvt& d, Real p, Real& bw,
                                          Real& mu) {
  if (!d.valid) { bw = 1; mu = 1; return; }
  const Real x = d.w_cw * (p - d.w_pref);
  bw = d.w_bw / (1 + x + 0.5 * x * x);
  const Real y = d.w_cmu * (p - d.w_pref);
  mu = d.w_mu / (1 + y + 0.5 * y * y);
}

// ---------------------------------------------------------------------------
// Everything one cell needs, in one evaluation
// ---------------------------------------------------------------------------
// The assembler works in shrinkage factors b = 1/B, because those are what
// multiply a reservoir volume to give a surface volume, and it needs their
// pressure derivatives for the Jacobian.  Those derivatives are the whole
// point of reading a table at all: a constant B has db/dp = 0, which says the
// fluid is incompressible.  For Norne's gas at 4000 psi the table gives a
// compressibility of 2.8e-4 /psi -- fourteen times the lumped rock-and-fluid
// constant the deck was standing in with.
struct PvtCell {
  Real bo = 1, bw = 1, bg = 1;        // shrinkage factors, 1/B
  Real dbo = 0, dbw = 0, dbg = 0;     // d(1/B)/dp
  Real rs = 0, drs = 0;               // dissolved gas and its pressure slope
  Real rv = 0, drv = 0;               // vaporised oil and its pressure slope
  Real drv_dso = 0;                    // VAPPARS dRv/dSo at fixed history
  Real dbg_dso = 0;                    // VAPPARS d(1/Bg)/dSo
  Real dmug_dso = 0;                   // VAPPARS d(mu_g)/dSo
  Real muo = 1, muw = 1, mug = 1;     // viscosities [cP]
  // Derivatives with respect to the switching primary u[2].  On the Sg
  // branch these contain the VAPPARS So response; on the Rs branch they are
  // the PVTO undersaturated-composition derivatives at fixed pressure.
  Real dbo_dx = 0, dbg_dx = 0, drs_dx = 0, drv_dx = 0;
  Real dmuo_dx = 0, dmug_dx = 0;
  Real dbo_dsw = 0, dbg_dsw = 0, drs_dsw = 0, drv_dsw = 0;
  Real dmuo_dsw = 0, dmug_dsw = 0;
};

// Dissolved gas at a cell: the saturated value, capped at what the oil was
// given to start with.
//
// Rs is not carried as an unknown here, so it has to be inferred from the
// state.  Above the bubble point the oil holds the gas it already has and can
// take no more; below it, Rs is whatever the bubble-point line supports.  That
// is the standard black-oil treatment without variable switching, and it is a
// function of pressure alone.
//
// It was briefly a function of Sg as well -- free gas means the oil must be
// saturated, so Rs should rise to the bubble-point value wherever gas appears
// -- ramped over a thin band in Sg to keep the residual continuous.  That was
// wrong, and wrong in a way worth recording.  The band was 1e-3 wide and the
// swing across it was 30 scf/stb, so dRs/dSg reached 30,000, and the Jacobian
// never carried that term: only dRs/dp was differenced.  Newton was handed a
// residual with a near-cliff in Sg and a derivative that could not see it,
// which is a good way to make a solver fail without making anything look
// wrong -- the pressure operator stayed a clean M-matrix throughout.
//
// The cost of dropping it is real and worth naming: gas injected into a cell
// cannot dissolve past rs_max, so it has nowhere to go but free saturation.
// Doing that properly needs Rs as a primary unknown with variable switching,
// not a ramp with a missing derivative.
NVRS_PVT_HD inline Real dev_rs_state(const DevicePvt& d, Real p, Real rs_max) {
  const Real rss = dev_rs_sat(d, p);
  return (rss < rs_max) ? rss : rs_max;
}

// Initial DRSDT reference state.  RSVD is an undersaturated-oil table and is
// not a ceiling on oil that already coexists with free gas.
NVRS_PVT_HD inline Real dev_rs_initial(const DevicePvt& d, Real p, Real sg,
                                      Real rs_anchor) {
  return (sg > (Real)1e-7) ? dev_rs_sat(d, p)
                            : dev_rs_state(d, p, rs_anchor);
}

// Evaluate the three phases and their pressure slopes at one cell.
//
// The slopes are taken as a centred difference of the same evaluators the
// residual uses, not as a separately derived formula.  That matters more than
// it might seem: the tables are piecewise linear, so within a segment the
// difference reproduces the segment slope exactly, and the Jacobian therefore
// describes precisely the function Newton is trying to zero.  A hand-derived
// slope that disagreed with the interpolation -- across the bubble point, say,
// where dev_oil switches branches -- would give a Newton step for a function
// the residual does not compute.
NVRS_PVT_HD inline PvtCell dev_pvt_cell(const DevicePvt& d, Real p,
                                        Real rs_max, Real so = (Real)1,
                                        Real so_max = (Real)1,
                                        Real vap1 = (Real)0) {
  PvtCell c;
  if (!d.valid) return c;

  Real h = 1.0e-4 * p;
  if (h < 1.0) h = 1.0;                      // never smaller than a psi
  const Real pm = (p - h > 0) ? p - h : p;
  const Real pp = p + h;
  const Real inv2h = 1.0 / (pp - pm);

  Real Bo, Bg, Bw, Bo_m, Bo_p, Bg_m, Bg_p, Bw_m, Bw_p, dummy;

  const Real rs   = dev_rs_state(d, p,  rs_max);
  const Real rs_m = dev_rs_state(d, pm, rs_max);
  const Real rs_p = dev_rs_state(d, pp, rs_max);
  c.rs  = rs;
  c.drs = (rs_p - rs_m) * inv2h;

  // Norne enables VAPOIL.  PVTG's saturated line supplies the stock-tank oil
  // carried by a unit of surface gas.  Rv is not a primary unknown in this
  // compact formulation, so use the saturated value implied by pressure, in
  // the same pressure-only approximation used for Rs above.  Keeping Rv in
  // PvtCell makes accumulation, fluxes, wells and reporting all use one
  // thermodynamic state instead of parsing VAPOIL and then silently dropping
  // it at the device boundary.
  // VAPPARS(1), matching OPM WetGasPvt: as oil is depleted below its
  // historical maximum, reduce the saturated Rv available to the gas.  The
  // factor is a saturation/history property, hence constant in the pressure
  // finite difference used for dRv/dp.
  Real vf = 1;
  const Real som = so_max < (Real)1 ? so_max : (Real)1;
  if (vap1 > 0 && som > (Real)0.01 && so < som) {
    const Real soe = so > (Real)0.001 ? so : (Real)0.001;
    vf = ::pow(soe / som, vap1);
    if (vf < (Real)0.001) vf = (Real)0.001;
    // The Eclipse floor makes the branch flat below So=0.001 (and if the
    // multiplicative floor itself binds).
    if (so > (Real)0.001 && vf > (Real)0.001)
      c.drv_dso = dev_rv_sat(d, p) * vap1 * vf / soe;
  }
  const Real rv   = vf * dev_rv_sat(d, p);
  const Real rv_m = vf * dev_rv_sat(d, pm);
  const Real rv_p = vf * dev_rv_sat(d, pp);
  c.rv  = rv;
  c.drv = (rv_p - rv_m) * inv2h;

  // Oil is evaluated at the Rs the state implies, and the difference moves Rs
  // with pressure too: below the bubble point the oil swells because it takes
  // up more gas, and that is the larger part of its compressibility.
  dev_oil(d, p,  rs,   Bo,   c.muo);
  dev_oil(d, pm, rs_m, Bo_m, dummy);
  dev_oil(d, pp, rs_p, Bo_p, dummy);

  dev_gas(d, p,  rv,   Bg,   c.mug);
  dev_gas(d, pm, rv_m, Bg_m, dummy);
  dev_gas(d, pp, rv_p, Bg_p, dummy);

  // VAPPARS also moves Bg and gas viscosity along the undersaturated PVTG
  // branch. Carry those saturation derivatives into Newton; omitting them
  // leaves the residual correct but its Jacobian frozen on the wrong curve.
  if (vap1 > 0 && som > (Real)0.01 && so < som) {
    const Real ds = (Real)1e-5;
    const Real slo = so > ds ? so - ds : (Real)0;
    const Real shi = so + ds < som ? so + ds : som;
    if (shi > slo) {
      Real flo = 1, fhi = 1;
      const Real elo = slo > (Real)0.001 ? slo : (Real)0.001;
      const Real ehi = shi > (Real)0.001 ? shi : (Real)0.001;
      flo = ::pow(elo / som, vap1);
      fhi = ::pow(ehi / som, vap1);
      if (flo < (Real)0.001) flo = (Real)0.001;
      if (fhi < (Real)0.001) fhi = (Real)0.001;
      Real Bg_lo, Bg_hi, mu_lo, mu_hi;
      dev_gas(d, p, flo * dev_rv_sat(d, p), Bg_lo, mu_lo);
      dev_gas(d, p, fhi * dev_rv_sat(d, p), Bg_hi, mu_hi);
      const Real ids = (Real)1 / (shi - slo);
      c.dbg_dso = ((Real)1 / Bg_hi - (Real)1 / Bg_lo) * ids;
      c.dmug_dso = (mu_hi - mu_lo) * ids;
    }
  }

  dev_water(d, p,  Bw,   c.muw);
  dev_water(d, pm, Bw_m, dummy);
  dev_water(d, pp, Bw_p, dummy);

  c.bo = (Bo > 0) ? 1.0 / Bo : 1.0;
  c.bg = (Bg > 0) ? 1.0 / Bg : 1.0;
  c.bw = (Bw > 0) ? 1.0 / Bw : 1.0;

  c.dbo = (Bo_p > 0 && Bo_m > 0) ? (1.0 / Bo_p - 1.0 / Bo_m) * inv2h : 0.0;
  c.dbg = (Bg_p > 0 && Bg_m > 0) ? (1.0 / Bg_p - 1.0 / Bg_m) * inv2h : 0.0;
  c.dbw = (Bw_p > 0 && Bw_m > 0) ? (1.0 / Bw_p - 1.0 / Bw_m) * inv2h : 0.0;
  return c;
}

// Evaluate an oil-only cell whose third primary is Rs.  Passing Rs as the
// cap to dev_pvt_cell gives the correct fixed-composition pressure derivative
// while Rs is below the saturated line.  The extra centred difference is the
// missing PVTO column derivative d(bo,muo)/dRs used by the reservoir and well
// Jacobians.  The appearance test switches branch before a material overshoot
// can make the min() choose RsSat for a converged state.
NVRS_PVT_HD inline PvtCell dev_pvt_cell_rs(const DevicePvt& d, Real p, Real rs,
                                           Real so = (Real)1,
                                           Real so_max = (Real)1,
                                           Real vap1 = (Real)0) {
  PvtCell c = dev_pvt_cell(d, p, rs, so, so_max, vap1);
  c.rs = rs;
  c.drs = 0;
  c.drs_dx = 1;
  Real h = fmax((Real)1e-3, (Real)1e-5 * fmax(fabs(rs), (Real)1));
  const Real rm = fmax((Real)0, rs - h), rp = rs + h;
  Real Bom, Bop, mum, mup;
  dev_oil(d, p, rm, Bom, mum);
  dev_oil(d, p, rp, Bop, mup);
  const Real inv = rp > rm ? (Real)1 / (rp - rm) : (Real)0;
  c.dbo_dx = (Bom > 0 && Bop > 0) ? ((Real)1 / Bop - (Real)1 / Bom) * inv
                                   : (Real)0;
  c.dmuo_dx = (mup - mum) * inv;
  return c;
}

// Oil-absent wet-gas state: u[2] is Rv and pressure is Pg.  Gas PVT must be
// differentiated at fixed Rv in pressure and at fixed pressure in Rv; using
// the saturated pressure curve here would silently put dRv/dp back into a
// column where Rv is an independent unknown.
NVRS_PVT_HD inline PvtCell dev_pvt_cell_rv(const DevicePvt& d, Real pg, Real rv,
                                           Real so_max = (Real)1,
                                           Real vap1 = (Real)0) {
  PvtCell c = dev_pvt_cell(d, pg, (Real)1e30, (Real)0, so_max, vap1);
  c.rv = rv;
  c.drv = 0;
  c.drv_dx = 1;
  c.drv_dso = c.dbg_dso = c.dmug_dso = 0;
  Real h = fmax((Real)1, (Real)1e-4 * pg);
  const Real pm = fmax((Real)1, pg - h), pp = pg + h;
  Real Bgm, Bgp, mum, mup;
  dev_gas(d, pm, rv, Bgm, mum);
  dev_gas(d, pp, rv, Bgp, mup);
  const Real ip = (Real)1 / (pp - pm);
  c.dbg = (Bgm > 0 && Bgp > 0) ? ((Real)1 / Bgp - (Real)1 / Bgm) * ip
                                : (Real)0;
  Real hr = fmax((Real)1e-10, (Real)1e-4 * fmax(fabs(rv), (Real)1e-8));
  const Real rm = fmax((Real)0, rv - hr), rp = rv + hr;
  Real Bglo, Bghi, mulo, muhi;
  dev_gas(d, pg, rm, Bglo, mulo);
  dev_gas(d, pg, rp, Bghi, muhi);
  const Real ir = rp > rm ? (Real)1 / (rp - rm) : (Real)0;
  c.dbg_dx = (Bglo > 0 && Bghi > 0)
                 ? ((Real)1 / Bghi - (Real)1 / Bglo) * ir : (Real)0;
  c.dmug_dx = (muhi - mulo) * ir;
  Real Bg, mu;
  dev_gas(d, pg, rv, Bg, mu);
  c.bg = Bg > 0 ? (Real)1 / Bg : (Real)1;
  c.mug = mu;
  return c;
}

// The same thing for a run with no tables: the deck's constants, with zero
// slopes.  Having one type for both keeps the assembler free of a second code
// path -- the constant-PVT run is then just a table that happens to be flat,
// and the ct term in BOParams remains its only source of compressibility.
NVRS_PVT_HD inline PvtCell const_pvt_cell(Real Bo, Real Bw, Real Bg, Real rs,
                                          Real muo, Real muw, Real mug) {
  PvtCell c;
  c.bo = 1.0 / Bo; c.bw = 1.0 / Bw; c.bg = 1.0 / Bg;
  c.rs = rs;
  c.muo = muo; c.muw = muw; c.mug = mug;
  return c;
}

// Evaluate the uploaded tables ON THE DEVICE and bring one cell's worth back.
//
// Uploading a table and using a table are different claims, and the summary
// printed at startup only supports the first.  This closes that gap: it runs
// the same dev_pvt_cell() the kernel runs, on the same DevicePvt the kernel is
// handed, and returns what it got.  If that disagrees with the host evaluator
// the fault is in the upload; if it comes back as the default 1/1/1 the tables
// never reached the kernel at all, whatever the banner said.
PvtCell probe_device_pvt(const DevicePvt& d, Real p, Real rs_max);

}  // namespace nvrs
