// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// fluid.hpp — relative permeability and phase mobilities with analytic
// derivatives, ported from RelPerm2 / RelPerm3 in NVR.py.  Two-phase uses the
// closed-form Brooks-Corey quadratics; three-phase uses tabulated SWOF/SGOF
// with the ECLIPSE/OPM default saturation-weighted three-phase KRO model.
// Tables live in constant/global memory and are interpolated on device.
#pragma once

#include "common.hpp"
#include <cmath>

namespace nvrs {

// One cell's saturation endpoints, from ENDSCALE.
//
// Norne carries SWL, SWCR, SWU, SGL, SGCR, SGU, SOWCR, SOGCR and the maximum
// relative permeabilities KRW, KRO and KRG as grid arrays.  They move the
// water front directly -- where a cell's water starts to flow and where its
// oil stops -- which is what the table alone cannot say cell by cell.
struct CellEnd {
  Real swl = 0, swcr = 0, swu = 1;
  Real sgl = 0, sgcr = 0, sgu = 1;
  Real sowcr = 0, sogcr = 0;
  Real krw = -1, kro = -1, krg = -1;   // negative: not given, keep the table's
};

// Per-cell endpoint arrays on the device, any of which may be null.
struct EndScale {
  const Real* swl = nullptr;   const Real* swcr = nullptr;
  const Real* swu = nullptr;   const Real* sgl = nullptr;
  const Real* sgcr = nullptr;  const Real* sgu = nullptr;
  const Real* sowcr = nullptr; const Real* sogcr = nullptr;
  const Real* krw = nullptr;   const Real* kro = nullptr;
  const Real* krg = nullptr;
  // Imbibition endpoints (I-prefixed ECLIPSE properties).  Carlson
  // hysteresis evaluates the same tabulated non-wetting curves through these endpoints
  // after reversal. A missing item falls back to the IMBNUM table endpoint;
  // explicitly supplied per-cell values override it.
  const Real* iswl = nullptr;   const Real* iswcr = nullptr;
  const Real* iswu = nullptr;   const Real* isgl = nullptr;
  const Real* isgcr = nullptr;  const Real* isgu = nullptr;
  const Real* isowcr = nullptr; const Real* isogcr = nullptr;
  const Real* ikrw = nullptr;   const Real* ikro = nullptr;
  const Real* ikrg = nullptr;
  int on = 0;                  // 0: no ENDSCALE, use the tables as they are
  // Deck keyword SCALECRS.  NO (the default) scales each curve between two
  // endpoints; YES pins a third node at the saturation where the displaced
  // phase stops moving, so the displacing phase's curve is mapped in two
  // pieces.  Norne says YES, and the two give visibly different fronts.
  int three_point = 0;
};

// Device-side view of the rel-perm tables (SoA, uploaded once).
struct SatTables {
  const Real* sw = nullptr;    // SWOF Sw nodes
  const Real* krow = nullptr;  // oil kr vs Sw
  const Real* krw = nullptr;   // water kr vs Sw
  int nsw = 0;
  const Real* sg = nullptr;    // SGOF Sg nodes
  const Real* krog = nullptr;  // oil kr vs Sg
  const Real* krg = nullptr;   // gas kr vs Sg
  int nsg = 0;

  // Capillary pressure, column four of SWOF and SGOF.  Null, or all zero,
  // means the phases share one pressure -- which is what this simulator did
  // before these existed.
  const Real* pcow = nullptr;  // p_o - p_w  vs Sw  [psi]
  const Real* pcog = nullptr;  // p_g - p_o  vs Sg  [psi]
  // Per-cell maximum-Pcow multiplier established from SWATINIT.
  const Real* pcow_scale = nullptr;
  int have_pc = 0;             // set only when a table carries a non-zero

  // The table's OWN endpoints, found once when it is uploaded.
  //
  // Endpoint scaling maps a cell's endpoints onto these, so they have to be
  // known: the connate and maximum saturations are the first and last nodes,
  // and the critical ones are where each curve leaves zero.
  Real sw_lo = 0, sw_up = 1, sw_cr = 0;   // SWOF: first node, last, krw > 0
  Real so_wcr = 0;                        // 1 - (last Sw where krow > 0)
  Real sg_lo = 0, sg_up = 1, sg_cr = 0;   // SGOF likewise
  Real so_gcr = 0;
  Real krw_max = 1, kro_max = 1, krg_max = 1;

  // ENDSCALE, when the deck carries it.  Held here rather than passed
  // separately so that every kernel already taking the tables gets the
  // endpoints too, with no change to a single signature.
  EndScale es;

  // EHYSTR: Carlson non-wetting relative-permeability hysteresis
  // (HYSTMOD=0).  OPM applies it to both oil/water and gas/oil: `sw_ow_hy`
  // is the smallest historical Sw+Sg (largest So), while `sg_hy` is the
  // largest historical Sg.
  // `sg_hy` is one Real per cell, the largest Sg that cell has reached since
  // the last accepted step -- frozen for the duration of a Newton solve the
  // same way DRSDT's rs_cap is, and for the same reason: moving it inside the
  // Newton loop would make the residual chase a target shifting under it.
  //
  // Norne assigns IMBNUM=2.  Its second SGOF has the same values as the first,
  // but I-prefixed endpoint properties make the effective imbibition curve
  // different (notably ISGCR=0.22 versus SGCR=0.03).
  int hyst_on = 0;
  const Real* sg_hy = nullptr;
  const Real* sw_ow_hy = nullptr;
};



// Find a saturation table's own endpoints.  Host only, called once per run.
//
// Endpoint scaling is a map from the cell's endpoints to the table's, so the
// table's have to be known before a single cell can be scaled.  The connate
// and maximum saturations are the first and last nodes; the critical ones are
// where a curve leaves zero, which is read off the tabulated kr rather than
// assumed -- Norne's SWOF and SGOF do not put them at the table ends.
inline void set_table_endpoints(SatTables& T, const Real* sw, const Real* krw,
                                const Real* krow, int nsw, const Real* sg,
                                const Real* krg, const Real* krog, int nsg) {
  const Real tol = (Real)1e-12;
  if (sw && nsw > 1) {
    T.sw_lo = sw[0];
    T.sw_up = sw[nsw - 1];
    T.sw_cr = sw[0];
    if (krw)
      for (int i = 0; i < nsw; ++i) {
        if (krw[i] > tol) break;
        T.sw_cr = sw[i];           // last node still dry
      }
    T.so_wcr = 0;
    if (krow)
      for (int i = 0; i < nsw; ++i)
        if (krow[i] <= tol) { T.so_wcr = 1 - sw[i]; break; }
    if (krw) T.krw_max = krw[nsw - 1];
    if (krow) T.kro_max = krow[0];
  }
  if (sg && nsg > 1) {
    T.sg_lo = sg[0];
    T.sg_up = sg[nsg - 1];
    T.sg_cr = sg[0];
    if (krg)
      for (int i = 0; i < nsg; ++i) {
        if (krg[i] > tol) break;
        T.sg_cr = sg[i];
      }
    T.so_gcr = 0;
    if (krog)
      for (int i = 0; i < nsg; ++i)
        if (krog[i] <= tol) { T.so_gcr = 1 - sg[i]; break; }
    if (krg) T.krg_max = krg[nsg - 1];
  }
  // A degenerate pair would divide by zero in map_sat, which returns the low
  // endpoint and a zero slope -- survivable, but it means the curve is flat,
  // so keep the table honest instead.
  if (!(T.sw_up > T.sw_cr)) T.sw_cr = T.sw_lo;
  if (!(T.sg_up > T.sg_cr)) T.sg_cr = T.sg_lo;
}

// Gather one cell's endpoints, falling back to the table's where an array is
// absent -- a deck may scale only some of them.
__host__ __device__ inline CellEnd cell_end(const EndScale& e,
                                            const SatTables& T, int c) {
  CellEnd o;
  o.swl = e.swl ? e.swl[c] : T.sw_lo;
  o.swcr = e.swcr ? e.swcr[c] : T.sw_cr;
  o.swu = e.swu ? e.swu[c] : T.sw_up;
  o.sgl = e.sgl ? e.sgl[c] : T.sg_lo;
  o.sgcr = e.sgcr ? e.sgcr[c] : T.sg_cr;
  o.sgu = e.sgu ? e.sgu[c] : T.sg_up;
  o.sowcr = e.sowcr ? e.sowcr[c] : T.so_wcr;
  o.sogcr = e.sogcr ? e.sogcr[c] : T.so_gcr;
  o.krw = e.krw ? e.krw[c] : (Real)-1;
  o.kro = e.kro ? e.kro[c] : (Real)-1;
  o.krg = e.krg ? e.krg[c] : (Real)-1;
  return o;
}

// Imbibition endpoints for Carlson hysteresis. Eclipse/OPM initializes these
// from the IMBNUM table's unscaled endpoints, then overwrites the items that
// an I-prefixed grid property explicitly supplies. They therefore do not
// inherit a cell's drainage-scaled endpoint when a partial grid edit omitted
// that cell.
__host__ __device__ inline CellEnd imb_cell_end(const EndScale& e,
                                                const SatTables& T, int c) {
  CellEnd o;
  o.swl = T.sw_lo; o.swcr = T.sw_cr; o.swu = T.sw_up;
  o.sgl = T.sg_lo; o.sgcr = T.sg_cr; o.sgu = T.sg_up;
  o.sowcr = T.so_wcr; o.sogcr = T.so_gcr;
  o.krw = o.kro = o.krg = (Real)-1;
  auto given = [](const Real* p, int q, Real fallback) {
    return p && isfinite((double)p[q]) ? p[q] : fallback;
  };
  o.swl = given(e.iswl, c, o.swl);
  o.swcr = given(e.iswcr, c, o.swcr);
  o.swu = given(e.iswu, c, o.swu);
  o.sgl = given(e.isgl, c, o.sgl);
  o.sgcr = given(e.isgcr, c, o.sgcr);
  o.sgu = given(e.isgu, c, o.sgu);
  o.sowcr = given(e.isowcr, c, o.sowcr);
  o.sogcr = given(e.isogcr, c, o.sogcr);
  o.krw = given(e.ikrw, c, o.krw);
  o.kro = given(e.ikro, c, o.kro);
  o.krg = given(e.ikrg, c, o.krg);
  return o;
}

// Three-point scaling: the same idea with a node pinned in the middle.
//
// SCALECRS=YES anchors each relative-permeability curve at three saturations
// rather than two and maps each interval separately.  Each curve has its own
// points: for example water uses SWCR, 1-SOWCR-SGL, SWU, while oil in the
// oil/water system uses SWL+SGL, SWCR+SGL, 1-SOWCR.  The middle node is what
// makes it worth doing: two-point and three-point scaling put the same cell's
// mobility at noticeably different places over the range a front sweeps.
//
// Degenerate CELL intervals fall back to the two-point map rather than divide
// by zero.  Degenerate TABLE intervals do not: equal target anchors are a
// valid plateau.  Norne's unscaled water curve has its residual-oil and
// maximum-water anchors at the same table saturation, so SCALECRS maps the
// cell's [1-SOWCR-SGL, SWU] interval to a constant maximum krw.  Treating
// t2==t3 as invalid stretched the rising limb all the way to SWU and reduced
// water kr by as much as 0.21 against OPM.
__host__ __device__ inline Real map_sat(Real S, Real lo, Real hi, Real tlo,
                                        Real thi, Real* slope);

__host__ __device__ inline Real map_sat3(Real S, Real x1, Real x2, Real x3,
                                         Real t1, Real t2, Real t3,
                                         Real* slope) {
  if (!(x2 > x1 + (Real)1e-12) || !(x3 > x2 + (Real)1e-12) ||
      t2 < t1 || t3 < t2)
    return map_sat(S, x1, x3, t1, t3, slope);
  if (S <= x2) {
    *slope = (t2 - t1) / (x2 - x1);
    return t1 + (S - x1) * (*slope);
  }
  *slope = (t3 - t2) / (x3 - x2);
  return t2 + (S - x2) * (*slope);
}

// Map a cell saturation onto the table's axis between two endpoint pairs.
//
// Two-point scaling: the cell's [lo, hi] is stretched onto the table's
// [tlo, thi], so a cell whose water becomes mobile at 0.3 reads the table
// where the table's water becomes mobile, whatever value that is.  The slope
// goes back to the caller because the Jacobian needs d(table S)/d(cell S).
__host__ __device__ inline Real map_sat(Real S, Real lo, Real hi, Real tlo,
                                        Real thi, Real* slope) {
  const Real d = hi - lo;
  if (!(d > 1e-12)) { *slope = 0; return tlo; }
  const Real k = (thi - tlo) / d;
  *slope = k;
  return tlo + (S - lo) * k;
}

// Capillary pressures at a cell, with the slopes the Jacobian needs.
struct Cap {
  Real pcow = 0, dpcow = 0;    // and d/dSw
  Real pcog = 0, dpcog = 0;    // and d/dSg
};

// Linear table interpolation with slope (value + d/dx), monotone x nodes.
__host__ __device__ inline void interp1(const Real* xp, const Real* fp, int m,
                                        Real x, Real& f, Real& df) {
  if (x <= xp[0]) { f = fp[0]; df = 0; return; }
  if (x >= xp[m - 1]) { f = fp[m - 1]; df = 0; return; }
  int lo = 0, hi = m - 1;
  while (hi - lo > 1) { int mid = (lo + hi) >> 1; (x < xp[mid] ? hi : lo) = mid; }
  Real dx = xp[hi] - xp[lo];
  Real slope = (fp[hi] - fp[lo]) / (dx + 1e-30);
  f = fp[lo] + slope * (x - xp[lo]);
  df = slope;
}

// --- Two-phase Brooks-Corey (analytic) -------------------------------------
// Returns water/oil mobilities and dM/dSw.  Matches RelPerm2 exactly.
struct Mob2 { Real Mw, Mo, dMw, dMo; };
__host__ __device__ inline Mob2 relperm2(Real Sw, Real uw, Real uo, Real bw,
                                         Real bo, Real swi, Real swr) {
  Real denom = 1.0 - swi - swr;
  const Real Sraw = (Sw - swi) / denom;
  const Real S = fmin(fmax(Sraw, (Real)0), (Real)1);
  Real oneS = 1.0 - S;
  Mob2 m;
  m.Mw = S * S / (uw * bw);
  m.Mo = oneS * oneS / (uo * bo);
  // Eclipse tables are constant outside their first/last nodes.  In
  // particular SPE10's SWOF has krw=0,kro=1 throughout Sw=[0,Swc].  Squaring
  // an extrapolated negative normalized saturation made krw positive again
  // below connate water and supplied Newton with a derivative of the wrong
  // sign exactly after physical saturation projection became active.
  if (Sraw <= 0 || Sraw >= 1) {
    m.dMw = 0;
    m.dMo = 0;
  } else {
    m.dMw = 2.0 * S / (uw * bw * denom);
    m.dMo = -2.0 * oneS / (uo * bo * denom);
  }
  return m;
}


// Two-phase field decks define their physics with SWOF, whose piecewise-linear
// interpolation is not identical to an analytic Corey curve between nodes.
// The distinction is largest just above connate water: a coarse tabulation of
// krw=S^2 can have several times the analytic mobility in its first interval.
// Use this helper everywhere a two-phase residual/control/report has a table;
// compact synthetic decks without one retain the analytic fallback above.
__host__ __device__ inline Mob2 relperm2_table(
    Real Sw, const SatTables& T, Real uw, Real uo, Real bw, Real bo,
    Real swi, Real swr) {
  if (T.nsw < 2 || !T.sw || !T.krw || !T.krow)
    return relperm2(Sw, uw, uo, bw, bo, swi, swr);
  Real krw = 0, krow = 0, dkrw = 0, dkrow = 0;
  interp1(T.sw, T.krw, T.nsw, Sw, krw, dkrw);
  interp1(T.sw, T.krow, T.nsw, Sw, krow, dkrow);
  Mob2 m;
  m.Mw = krw / (uw * bw);
  m.Mo = krow / (uo * bo);
  m.dMw = dkrw / (uw * bw);
  m.dMo = dkrow / (uo * bo);
  return m;
}
// --- Three-phase (tabulated + ECLIPSE/OPM default) -------------------------
// Returns phase mobilities and their partials wrt (Sw, Sg).  Matches RelPerm3.
struct Mob3 {
  Real Mw, Mo, Mg;
  Real dMw_dSw, dMo_dSw, dMo_dSg, dMg_dSw, dMg_dSg;
};
// Capillary pressure at one cell.
//
// SWOF column four is Pcow = p_o - p_w and SGOF column four is Pcog =
// p_g - p_o, both as functions of the wetting-phase saturation of their pair.
// Interpolated on the same nodes as the relative permeabilities, so a cell
// whose kr comes from between two rows gets its Pc from between the same two.
//
// Returns zeros when the deck gave no fourth column, which makes every phase
// pressure equal and reproduces the behaviour that predates this.
__host__ __device__ inline Cap capillary(Real Sw, Real Sg, const SatTables& T,
                                         int cell = -1) {
  Cap c;
  if (!T.have_pc) return c;
  if (T.pcow && T.nsw > 1) {
    Real sw_tab = Sw, jac = 1;
    if (T.es.on && cell >= 0) {
      const CellEnd e = cell_end(T.es, T, cell);
      // Pc saturation scaling is always two-point, even when SCALECRS asks
      // for three-point relative-permeability scaling.
      sw_tab = map_sat(Sw, e.swl, e.swu, T.sw_lo, T.sw_up, &jac);
    }
    interp1(T.sw, T.pcow, T.nsw, sw_tab, c.pcow, c.dpcow);
    const Real scale = (T.pcow_scale && cell >= 0) ? T.pcow_scale[cell]
                                                   : (Real)1;
    c.pcow *= scale;
    c.dpcow *= jac * scale;
  }
  if (T.pcog && T.nsg > 1) interp1(T.sg, T.pcog, T.nsg, Sg, c.pcog, c.dpcog);
  return c;
}

// Inverse of the endpoint maps above.  Carlson aligns the imbibition curve to
// the drainage curve at the reversal point, so it needs to map the drainage
// table coordinate back onto the imbibition cell coordinate.
__host__ __device__ inline Real inv_map_sat(Real y, Real xlo, Real xhi,
                                            Real tlo, Real thi) {
  const Real d = thi - tlo;
  if (!(d > (Real)1e-12)) return xlo;
  return xlo + (y - tlo) * (xhi - xlo) / d;
}

__host__ __device__ inline Real inv_map_sat3(Real y, Real x1, Real x2,
                                             Real x3, Real t1, Real t2,
                                             Real t3) {
  if (!(x2 > x1 + (Real)1e-12) || !(x3 > x2 + (Real)1e-12) ||
      t2 < t1 || t3 < t2)
    return inv_map_sat(y, x1, x3, t1, t3);
  // At a lower plateau t1==t2, the shared value belongs to the start of the
  // second interval; at an upper plateau t2==t3 it belongs to the end of the
  // first.  Either choice is continuous, but these choices preserve the
  // branch whose inverse is unique.
  const bool first = y < t2 || (t2 > t1 && y <= t2);
  return first ? inv_map_sat(y, x1, x2, t1, t2)
               : inv_map_sat(y, x2, x3, t2, t3);
}

// Carlson HYSTMOD=0 for the gas/oil non-wetting curve, matching OPM 2024.10:
// use drainage until Sg reaches its historical maximum, then evaluate the
// imbibition curve at a constant horizontal shift chosen to make krg
// continuous at that reversal.  Because Norne's two SGOF value tables are
// identical, matching krg is equivalent to matching their table coordinate.
// The shift is frozen with sg_hy during Newton, hence d(Sg_shifted)/dSg = 1.
__host__ __device__ inline Real carlson_sg(Real Sg, Real sg_hy,
                                           const CellEnd& d,
                                           const CellEnd& i,
                                           const SatTables& T) {
  if (!(sg_hy > d.sgcr) || !(Sg < sg_hy)) return Sg;

  Real jd = 1;
  Real table_at_reversal;
  if (T.es.three_point) {
    table_at_reversal =
        map_sat3(sg_hy, d.sgcr, (Real)1 - d.sogcr - d.swl, d.sgu,
                 T.sg_cr, (Real)1 - T.so_gcr - T.sw_lo, T.sg_up, &jd);
  } else {
    table_at_reversal =
        map_sat(sg_hy, d.sgcr, d.sgu, T.sg_cr, T.sg_up, &jd);
  }

  Real imb_at_reversal;
  if (T.es.three_point) {
    imb_at_reversal =
        inv_map_sat3(table_at_reversal, i.sgcr,
                     (Real)1 - i.sogcr - i.swl, i.sgu, T.sg_cr,
                     (Real)1 - T.so_gcr - T.sw_lo, T.sg_up);
  } else {
    imb_at_reversal =
        inv_map_sat(table_at_reversal, i.sgcr, i.sgu, T.sg_cr, T.sg_up);
  }
  return Sg + (imb_at_reversal - sg_hy);
}

// Carlson for the oil/water system.  OPM passes 1-So (= Sw+Sg) as the
// wetting-coordinate used to evaluate the non-wetting oil curve, and stores
// its historical minimum.  On subsequent water imbibition the I-endpoint
// curve is shifted horizontally so krow is continuous at the reversal.
__host__ __device__ inline Real carlson_sw_ow(Real Sw_ow, Real sw_ow_hy,
                                              const CellEnd& d,
                                              const CellEnd& i,
                                              const SatTables& T) {
  if (!(sw_ow_hy < (Real)2) || !(Sw_ow > sw_ow_hy)) return Sw_ow;

  Real jd = 1;
  Real table_at_reversal;
  if (T.es.three_point) {
    table_at_reversal =
        map_sat3(sw_ow_hy, d.swl + d.sgl, d.swcr + d.sgl,
                 (Real)1 - d.sowcr, T.sw_lo + T.sg_lo,
                 T.sw_cr + T.sg_lo, (Real)1 - T.so_wcr, &jd);
  } else {
    table_at_reversal =
        map_sat(sw_ow_hy, d.swl + d.sgl, (Real)1 - d.sowcr,
                T.sw_lo + T.sg_lo, (Real)1 - T.so_wcr, &jd);
  }

  Real imb_at_reversal;
  if (T.es.three_point) {
    imb_at_reversal =
        inv_map_sat3(table_at_reversal, i.swl + i.sgl,
                     i.swcr + i.sgl, (Real)1 - i.sowcr,
                     T.sw_lo + T.sg_lo, T.sw_cr + T.sg_lo,
                     (Real)1 - T.so_wcr);
  } else {
    imb_at_reversal =
        inv_map_sat(table_at_reversal, i.swl + i.sgl,
                    (Real)1 - i.sowcr, T.sw_lo + T.sg_lo,
                    (Real)1 - T.so_wcr);
  }
  return Sw_ow + (imb_at_reversal - sw_ow_hy);
}

__host__ __device__ inline Mob3 relperm3(Real Sw, Real Sg, const SatTables& T,
                                         Real uw, Real uo, Real ug, Real bw,
                                         Real bo, Real bg, Real swi, Real swr,
                                         int cell = -1) {
  (void)swi;
  (void)swr;
  Real krow, dkrow, krw, dkrw, krog, dkrog, krg, dkrg;
  Real swco = T.sw_lo;
  Real sw = fmax(Sw, swco);
  Real dsw_dSw = Sw > swco ? (Real)1 : (Real)0;
  Real Sw_ow = sw + Sg;
  if (T.es.on && cell >= 0) {
    // ENDSCALE: read the table where the CELL's own endpoints say to.
    //
    // Two-point scaling.  Each of the four curves has its own pair of
    // endpoints and therefore its own map, which is why this cannot be done
    // by transforming Sw once before the call: krw is scaled between the
    // critical and maximum water, while krow is scaled between connate water
    // and the water at which oil stops moving.  The same Sw reads the table
    // at two different places.
    //
    // The chain rule is the point.  A mapped lookup returns d(kr)/d(table S),
    // and the Jacobian needs d(kr)/d(cell S), so every slope is multiplied by
    // the map's own -- without that the derivatives quietly disagree with the
    // residual and Newton stops converging.
    const CellEnd e = cell_end(T.es, T, cell);
    swco = e.swl;
    sw = fmax(Sw, swco);
    dsw_dSw = Sw > swco ? (Real)1 : (Real)0;
    Sw_ow = sw + Sg;
    Real jw, jow, jg, jog;
    // Carlson changes each subsystem's non-wetting curve.  Water remains on
    // drainage; krow and krg use their own accepted-step reversal histories.
    CellEnd ge = e;
    CellEnd oe = e;
    Real Sg_curve = Sg;
    Real Sw_ow_curve = Sw_ow;
    if (T.hyst_on && T.sg_hy && cell >= 0 && Sg < T.sg_hy[cell]) {
      ge = imb_cell_end(T.es, T, cell);
      Sg_curve = carlson_sg(Sg, T.sg_hy[cell], e, ge, T);
    }
    if (T.hyst_on && T.sw_ow_hy && cell >= 0 &&
        Sw_ow > T.sw_ow_hy[cell]) {
      oe = imb_cell_end(T.es, T, cell);
      Sw_ow_curve = carlson_sw_ow(Sw_ow, T.sw_ow_hy[cell], e, oe, T);
    }
    // SCALECRS applies the same three-point saturation map to all four
    // two-phase curves.  The middle point differs by curve; see OPM's
    // EclEpsScalingPoints::init().
    Real Sw_w, Sg_g, Sw_o, Sg_o;
    const Real Sg_go = Sg + sw - swco;
    if (T.es.three_point) {
      Sw_w = map_sat3(Sw, e.swcr, 1 - e.sowcr - e.sgl, e.swu,
                      T.sw_cr, 1 - T.so_wcr - T.sg_lo, T.sw_up, &jw);
      Sg_g = map_sat3(Sg_curve, ge.sgcr, 1 - ge.sogcr - ge.swl, ge.sgu,
                      T.sg_cr, 1 - T.so_gcr - T.sw_lo, T.sg_up, &jg);
      // Oil in the oil/water system is tabulated against Sw+Sg.  Oil in the
      // gas/oil system is tabulated against the equivalent gas saturation
      // Sg + max(Sw,Swl) - Swl (OPM supplies So=1-Sg-max(Sw,Swl)).
      Sw_o = map_sat3(Sw_ow_curve, oe.swl + oe.sgl, oe.swcr + oe.sgl,
                      1 - oe.sowcr, T.sw_lo + T.sg_lo,
                      T.sw_cr + T.sg_lo, 1 - T.so_wcr, &jow);
      Sg_o = map_sat3(Sg_go, e.sgl, e.sgcr, 1 - e.swl - e.sogcr,
                      T.sg_lo, T.sg_cr, 1 - T.sw_lo - T.so_gcr, &jog);
    } else {
      Sw_w = map_sat(Sw, e.swcr, e.swu, T.sw_cr, T.sw_up, &jw);
      Sg_g = map_sat(Sg_curve, ge.sgcr, ge.sgu, T.sg_cr, T.sg_up, &jg);
      Sw_o = map_sat(Sw_ow_curve, oe.swl + oe.sgl, 1 - oe.sowcr,
                     T.sw_lo + T.sg_lo, 1 - T.so_wcr, &jow);
      Sg_o = map_sat(Sg_go, e.sgl, 1 - e.swl - e.sogcr, T.sg_lo,
                     1 - T.sw_lo - T.so_gcr, &jog);
    }
    interp1(T.sw, T.krow, T.nsw, Sw_o, krow, dkrow); dkrow *= jow;
    interp1(T.sw, T.krw, T.nsw, Sw_w, krw, dkrw);    dkrw *= jw;
    interp1(T.sg, T.krog, T.nsg, Sg_o, krog, dkrog); dkrog *= jog;
    interp1(T.sg, T.krg, T.nsg, Sg_g, krg, dkrg);    dkrg *= jg;
    // Vertical scaling: KRW/KRO/KRG set each phase curve's maximum.  The OPM
    // default is a weighted average of the two oil curves, so the same oil
    // factor is applied to both; the resulting three-phase KRO is scaled once.
    if (e.krw >= 0 && T.krw_max > 1e-12) {
      const Real f = e.krw / T.krw_max; krw *= f; dkrw *= f;
    }
    if (oe.kro >= 0 && T.kro_max > 1e-12) {
      const Real f = oe.kro / T.kro_max; krow *= f; dkrow *= f;
    }
    if (ge.kro >= 0 && T.kro_max > 1e-12) {
      const Real f = ge.kro / T.kro_max; krog *= f; dkrog *= f;
    }
    if (ge.krg >= 0 && T.krg_max > 1e-12) {
      const Real f = ge.krg / T.krg_max; krg *= f; dkrg *= f;
    }
  } else {
    interp1(T.sw, T.krow, T.nsw, Sw_ow, krow, dkrow);
    interp1(T.sw, T.krw, T.nsw, Sw, krw, dkrw);
    interp1(T.sg, T.krog, T.nsg, Sg + sw - swco, krog, dkrog);
    // Without endpoint scaling both effective curves are the same table.
    interp1(T.sg, T.krg, T.nsg, Sg, krg, dkrg);
  }

  // ECLIPSE/OPM default three-phase oil relative permeability:
  //
  //   kro = [ Sg*kro_go + (max(Sw,Swco)-Swco)*kro_ow ]
  //         / [ Sg + max(Sw,Swco)-Swco ]
  //
  // OPM regularizes the vanishing denominator below 1e-5 by blending to the
  // mean of the two oil curves.  Keep the exact branch and its derivatives so
  // the Newton Jacobian differentiates the residual that was actually used.
  const Real w = sw - swco;
  const Real d = w + Sg;
  const Real A = Sg * krog + w * krow;
  const Real eps = (Real)1e-5;

  Real KRO, dKRO_dSw, dKRO_dSg;
  const Real K2 = (krow + krog) * (Real)0.5;
  const Real dK2_dSw = (Real)0.5 * (dkrow + dkrog) * dsw_dSw;
  const Real dK2_dSg = (Real)0.5 * (dkrow + dkrog);
  if (d <= eps * (Real)0.5) {
    KRO = K2;
    dKRO_dSw = dK2_dSw;
    dKRO_dSg = dK2_dSg;
  } else {
    const Real invd = (Real)1 / d;
    const Real K1 = A * invd;
    const Real dA_dSw = dsw_dSw * (Sg * dkrog + krow + w * dkrow);
    const Real dA_dSg = krog + Sg * dkrog + w * dkrow;
    const Real dK1_dSw = (dA_dSw - K1 * dsw_dSw) * invd;
    const Real dK1_dSg = (dA_dSg - K1) * invd;
    if (d < eps) {
      const Real alpha = (eps - d) / (eps * (Real)0.5);
      const Real da_dSw = -(Real)2 * dsw_dSw / eps;
      const Real da_dSg = -(Real)2 / eps;
      KRO = alpha * K2 + ((Real)1 - alpha) * K1;
      dKRO_dSw = alpha * dK2_dSw + ((Real)1 - alpha) * dK1_dSw +
                  da_dSw * (K2 - K1);
      dKRO_dSg = alpha * dK2_dSg + ((Real)1 - alpha) * dK1_dSg +
                  da_dSg * (K2 - K1);
    } else {
      KRO = K1;
      dKRO_dSw = dK1_dSw;
      dKRO_dSg = dK1_dSg;
    }
  }
  Real invW = 1.0 / (uw * bw), invO = 1.0 / (uo * bo), invG = 1.0 / (ug * bg);

  Mob3 m;
  m.Mw = krw * invW;
  m.Mo = KRO * invO;
  m.Mg = krg * invG;
  m.dMw_dSw = dkrw * invW;
  m.dMo_dSw = dKRO_dSw * invO;
  m.dMo_dSg = dKRO_dSg * invO;
  m.dMg_dSw = 0;
  m.dMg_dSg = dkrg * invG;
  return m;
}

}  // namespace nvrs
