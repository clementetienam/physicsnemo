// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_endscale.cu -- ENDSCALE: per-cell relative permeability endpoints.
//
// Norne's deck carries SWL, SWCR, SWU, SGL, SGCR, SGU, SOWCR, SOGCR and the
// maximum relative permeabilities as grid arrays, so each cell has its own
// water saturation at which water starts to flow and its own at which oil
// stops.  A single field-wide table cannot reproduce a front whose speed
// varies cell by cell, which is what left NVRS's water arriving late against
// Flow between roughly 500 and 3000 days.
//
// Three properties are worth asserting, and the third is the one that bites:
//
//   1. IDENTITY.  A cell whose endpoints are the table's own must reproduce
//      the unscaled curve to rounding.  If the map is wrong in any way, this
//      is where it shows, before any question of whether the physics improved.
//   2. The shift actually happens -- water below a raised SWCR is immobile.
//   3. DERIVATIVES.  Scaling multiplies every slope by the map's own; miss
//      that chain rule and the Jacobian silently disagrees with the residual,
//      Newton stalls, and nothing says why.  Checked against central
//      differences.
//
// Sampling avoids the table nodes deliberately.  kr is piecewise linear, so a
// central difference straddling a node averages two different one-sided
// slopes and disagrees with the analytic value by O(1) however correct the
// code is -- the first run of this test "failed" for exactly that reason.
//
//   nvcc -std=c++17 -arch=sm_70 -I../include test_endscale.cu -o t && ./t
#include <cstdio>
#include <cmath>
#include <vector>
#include "fluid.hpp"
using namespace nvrs;
int main() {
  // A SWOF/SGOF pair whose critical points are NOT at the table ends.
  std::vector<Real> sw{0.10,0.20,0.30,0.40,0.60,0.80,0.90};
  std::vector<Real> krw{0.00,0.00,0.02,0.08,0.30,0.70,1.00};
  std::vector<Real> kro{0.90,0.60,0.35,0.18,0.04,0.00,0.00};
  std::vector<Real> sg{0.00,0.05,0.10,0.30,0.50,0.70};
  std::vector<Real> krg{0.00,0.00,0.03,0.25,0.60,0.95};
  std::vector<Real> kog{0.85,0.70,0.55,0.20,0.02,0.00};
  SatTables T; T.sw=sw.data(); T.krw=krw.data(); T.krow=kro.data(); T.nsw=7;
  T.sg=sg.data(); T.krg=krg.data(); T.krog=kog.data(); T.nsg=6;
  set_table_endpoints(T, sw.data(), krw.data(), kro.data(), 7,
                      sg.data(), krg.data(), kog.data(), 6);
  std::vector<Real> pcow{50,40,30,20,10,2,0};
  T.pcow=pcow.data(); T.have_pc=1;
  std::printf("table: swl %.2f swcr %.2f swu %.2f sowcr %.2f | "
              "sgl %.2f sgcr %.2f sgu %.2f sogcr %.2f | krwmax %.2f kromax %.2f\n",
    (double)T.sw_lo,(double)T.sw_cr,(double)T.sw_up,(double)T.so_wcr,
    (double)T.sg_lo,(double)T.sg_cr,(double)T.sg_up,(double)T.so_gcr,
    (double)T.krw_max,(double)T.kro_max);

  // OPM's ECLIPSE-default three-phase oil law evaluates krow at Sw+Sg and
  // krog at Sg+Sw-Swco, then combines them using saturation weights.  At
  // Sw=.35, Sg=.10: krow(.45)=.145 and krog(.35)=.155.
  const double opm_expect = (0.25*0.145 + 0.10*0.155) / 0.35;
  const double opm_got = relperm3(.35,.10,T,1,1,1,1,1,1,.10,.10,-1).Mo;
  std::printf("TEST OPM default KRO  : got %.9f want %.9f  %s\n",
              opm_got, opm_expect,
              fabs(opm_got-opm_expect) < 1e-12 ? "PASS" : "FAIL");
  // A two-phase FIELD run must use SWOF's piecewise-linear interpolation,
  // including its slope, rather than silently substituting analytic Corey.
  {
    const Mob2 m = relperm2_table((Real)0.25, T, 1, 1, 1, 1, 0.1, 0.1);
    const bool pass = fabs(m.Mw - 0.01) < 1e-12 &&
                      fabs(m.dMw - 0.2) < 1e-12;
    std::printf("TEST two-phase SWOF    : Mw %.6f slope %.6f  %s\n",
                (double)m.Mw, (double)m.dMw, pass ? "PASS" : "FAIL");
  }


  // TEST 1 -- identity: a cell whose endpoints ARE the table's must reproduce
  // the unscaled curve exactly.  If this fails the map is wrong, full stop.
  Real a_swl[1]{T.sw_lo}, a_swcr[1]{T.sw_cr}, a_swu[1]{T.sw_up};
  Real a_sowcr[1]{T.so_wcr}, a_sgl[1]{T.sg_lo}, a_sgcr[1]{T.sg_cr};
  Real a_sgu[1]{T.sg_up}, a_sogcr[1]{T.so_gcr};
  SatTables I = T;
  I.es.on=1; I.es.swl=a_swl; I.es.swcr=a_swcr; I.es.swu=a_swu;
  I.es.sowcr=a_sowcr; I.es.sgl=a_sgl; I.es.sgcr=a_sgcr; I.es.sgu=a_sgu;
  I.es.sogcr=a_sogcr;
  double worst=0;
  for (double S=0.12; S<0.88; S+=0.01)
    for (double G=0.0; G<0.5; G+=0.05) {
      Mob3 u=relperm3(S,G,T,1,1,1,1,1,1,0.1,0.1,-1);
      Mob3 v=relperm3(S,G,I,1,1,1,1,1,1,0.1,0.1, 0);
      worst=fmax(worst,fabs(u.Mw-v.Mw)+fabs(u.Mo-v.Mo)+fabs(u.Mg-v.Mg));
    }
  std::printf("TEST identity map     : worst |dMob| = %.3e  %s\n", worst,
              worst<1e-12?"PASS":"FAIL");

  // TEST 2 -- a genuinely shifted cell: water mobile later, oil dying earlier.
  Real b_swcr[1]{0.45}, b_swl[1]{0.25}, b_swu[1]{0.85}, b_sowcr[1]{0.25};
  SatTables C = T;
  C.es.on=1; C.es.swl=b_swl; C.es.swcr=b_swcr; C.es.swu=b_swu;
  C.es.sowcr=b_sowcr;
  Real pc_scale[1]{1.75}; C.pcow_scale=pc_scale;
  {
    const Real S = 0.55;
    const Cap p = capillary(S, 0, C, 0);
    const Real st = T.sw_lo + (S-b_swl[0])*(T.sw_up-T.sw_lo)
                                  /(b_swu[0]-b_swl[0]);
    Real raw=0, draw=0;
    interp1(sw.data(), pcow.data(), (int)sw.size(), st, raw, draw);
    const Real jac=(T.sw_up-T.sw_lo)/(b_swu[0]-b_swl[0]);
    std::printf("TEST SWATINIT Pc scale: Pc %.9f want %.9f, dPc %.9f want %.9f  %s\n",
                (double)p.pcow, (double)(raw*pc_scale[0]),
                (double)p.dpcow, (double)(draw*jac*pc_scale[0]),
                fabs(p.pcow-raw*pc_scale[0]) < 1e-11 &&
                fabs(p.dpcow-draw*jac*pc_scale[0]) < 1e-11 ? "PASS" : "FAIL");
  }
  std::printf("TEST shifted endpoints: krw at Sw=0.40 -> unscaled %.4f, "
              "scaled %.4f (want 0 below Swcr=0.45)\n",
    (double)relperm3(0.40,0.0,T,1,1,1,1,1,1,0.1,0.1,-1).Mw,
    (double)relperm3(0.40,0.0,C,1,1,1,1,1,1,0.1,0.1, 0).Mw);

  // TEST 3 -- derivatives.  Endpoint scaling multiplies every slope by the
  // map's own; if that chain rule is missing the Jacobian silently disagrees
  // with the residual and Newton stalls.  Compare against finite differences.
  double dmax=0; const double h=1e-6;
  for (double S=0.30; S<0.80; S+=0.02) {
    Mob3 m=relperm3(S,0.10,C,1,1,1,1,1,1,0.1,0.1,0);
    Mob3 p=relperm3(S+h,0.10,C,1,1,1,1,1,1,0.1,0.1,0);
    Mob3 q=relperm3(S-h,0.10,C,1,1,1,1,1,1,0.1,0.1,0);
    double fd_w=(p.Mw-q.Mw)/(2*h), fd_o=(p.Mo-q.Mo)/(2*h);
    double sc=fmax(1.0,fabs(fd_w)+fabs(fd_o));
    dmax=fmax(dmax,(fabs(fd_w-m.dMw_dSw)+fabs(fd_o-m.dMo_dSw))/sc);
  }
  std::printf("TEST dMw/dSw, dMo/dSw : worst relative FD error = %.3e  %s\n",
              dmax, dmax<2e-4?"PASS":"FAIL");
  // Give the cell shifted GAS endpoints too, and report which node each
  // sample sits on: a central difference straddling a table kink compares a
  // two-sided average against a one-sided slope and will differ by O(1)
  // however correct the code is.
  Real c_sgcr[1]{0.12}, c_sgu[1]{0.60}, c_sgl[1]{0.02}, c_sogcr[1]{0.22};
  C.es.sgcr=c_sgcr; C.es.sgu=c_sgu; C.es.sgl=c_sgl; C.es.sogcr=c_sogcr;
  double gmax=0, gmax_node=0;
  for (double G=0.023; G<0.45; G+=0.017) {
    Mob3 m=relperm3(0.35,G,C,1,1,1,1,1,1,0.1,0.1,0);
    Mob3 p=relperm3(0.35,G+h,C,1,1,1,1,1,1,0.1,0.1,0);
    Mob3 q=relperm3(0.35,G-h,C,1,1,1,1,1,1,0.1,0.1,0);
    double fd_g=(p.Mg-q.Mg)/(2*h), fd_o=(p.Mo-q.Mo)/(2*h);
    double sc=fmax(1.0,fabs(fd_g)+fabs(fd_o));
    double err=(fabs(fd_g-m.dMg_dSg)+fabs(fd_o-m.dMo_dSg))/sc;
    // Does this sample map onto a table node, for either gas curve?
    bool onnode=false;
    for (int i=0;i<6;i++) {
      Real jg,jo;
      Real a=map_sat((Real)G,c_sgcr[0],c_sgu[0],T.sg_cr,T.sg_up,&jg);
      Real b=map_sat((Real)G,c_sgl[0],1-c_sogcr[0],T.sg_lo,1-T.so_gcr,&jo);
      if (fabs(a-sg[i])<2e-3 || fabs(b-sg[i])<2e-3) onnode=true;
    }
    if (onnode) gmax_node=fmax(gmax_node,err); else gmax=fmax(gmax,err);
    if (err>1e-3) std::printf("      Sg=%.4f err=%.2e %s\n", G, err,
                              onnode?"(on a table node)":"(NOT on a node)");
  }
  std::printf("TEST dMg/dSg, dMo/dSg : worst relative FD error = %.3e  %s\n",
              gmax, gmax<2e-4?"PASS":"FAIL");
  std::printf("   (samples landing on a table kink, excluded: worst %.3e)\n",
              gmax_node);
  // TEST 4 -- SCALECRS=YES.  Norne sets it, so this is the path that actually
  // runs.  Same two properties: a cell whose endpoints are the table's must
  // still reproduce the table (all three anchors coincide, so both pieces of
  // the map are the identity), and the derivatives must still match.
  SatTables I3 = I; I3.es.three_point = 1;
  double w3 = 0;
  for (double S = 0.12; S < 0.88; S += 0.01)
    for (double G = 0.0; G < 0.5; G += 0.05) {
      Mob3 u = relperm3(S, G, T, 1,1,1,1,1,1, 0.1, 0.1, -1);
      Mob3 v = relperm3(S, G, I3, 1,1,1,1,1,1, 0.1, 0.1, 0);
      w3 = fmax(w3, fabs(u.Mw-v.Mw)+fabs(u.Mo-v.Mo)+fabs(u.Mg-v.Mg));
    }
  std::printf("TEST 3pt identity     : worst |dMob| = %.3e  %s\n", w3,
              w3 < 1e-12 ? "PASS" : "FAIL");

  SatTables C3 = C; C3.es.three_point = 1;
  double d3 = 0;
  for (double S = 0.313; S < 0.80; S += 0.017) {
    Mob3 m = relperm3(S, 0.103, C3, 1,1,1,1,1,1, 0.1, 0.1, 0);
    Mob3 a = relperm3(S+h, 0.103, C3, 1,1,1,1,1,1, 0.1, 0.1, 0);
    Mob3 b = relperm3(S-h, 0.103, C3, 1,1,1,1,1,1, 0.1, 0.1, 0);
    double fw = (a.Mw-b.Mw)/(2*h), fo = (a.Mo-b.Mo)/(2*h);
    double sc = fmax(1.0, fabs(fw)+fabs(fo));
    d3 = fmax(d3, (fabs(fw-m.dMw_dSw)+fabs(fo-m.dMo_dSw))/sc);
  }
  std::printf("TEST 3pt derivatives  : worst relative FD error = %.3e  %s\n",
              d3, d3 < 2e-4 ? "PASS" : "FAIL");

  // TEST 5 -- and the two methods must actually DIFFER, or SCALECRS is
  // being read and ignored.  The middle anchor sits at residual oil, so the
  // gap should open up over the range the waterflood sweeps.
  double gap = 0; double at = 0;
  for (double S = 0.30; S < 0.85; S += 0.01) {
    double a = (double)relperm3(S, 0.05, C,  1,1,1,1,1,1, 0.1, 0.1, 0).Mw;
    double b = (double)relperm3(S, 0.05, C3, 1,1,1,1,1,1, 0.1, 0.1, 0).Mw;
    if (fabs(a-b) > gap) { gap = fabs(a-b); at = S; }
  }
  std::printf("TEST 2pt vs 3pt differ: worst |krw| gap = %.4f at Sw = %.2f  "
              "%s\n", gap, at, gap > 1e-3 ? "PASS" : "FAIL (SCALECRS inert)");

  // Norne-shaped SCALECRS plateau.  The unscaled SWOF reaches its residual-oil
  // anchor at the same saturation as maximum water (t2=t3=1), while the
  // scaled cell has 1-SOWCR-SGL=0.87 below SWU=1.  OPM maps the rising limb to
  // 1 by 0.87 and leaves it there.  Falling back to a two-point map instead
  // sends Sw=0.86 only to about 0.82 and suppresses krw severely.
  {
    Real jm = 0, jp = 0;
    const Real m = map_sat3((Real)0.86, (Real)0.24, (Real)0.87, (Real)1,
                            (Real)0.0001, (Real)1, (Real)1, &jm);
    const Real p = map_sat3((Real)0.90, (Real)0.24, (Real)0.87, (Real)1,
                            (Real)0.0001, (Real)1, (Real)1, &jp);
    const Real want = (Real)0.0001 + ((Real)0.86 - (Real)0.24) *
                      ((Real)1 - (Real)0.0001) /
                      ((Real)0.87 - (Real)0.24);
    const bool pass = fabs(m-want) < 1e-12 && p == (Real)1 && jp == (Real)0;
    std::printf("TEST 3pt table plateau: map(0.86) %.8f want %.8f, "
                "map(0.90) %.1f slope %.1f  %s\n",
                (double)m, (double)want, (double)p, (double)jp,
                pass ? "PASS" : "FAIL");
  }
  // TEST 6 -- Norne's EHYSTR HYSTMOD=0 is Carlson, not Land/Killough.  Gas
  // follows drainage at the historical maximum and switches continuously to
  // the I-endpoint-scaled curve below it.  ISGCR is intentionally much larger
  // than SGCR, as it is in Norne (0.22 versus 0.03).
  Real i_swl[1]{b_swl[0]};
  Real i_sgcr[1]{0.25}, i_sgu[1]{c_sgu[0]}, i_sgl[1]{c_sgl[0]};
  Real i_sogcr[1]{c_sogcr[0]}, sg_hy[1]{0.40};
  SatTables H = C3;
  H.es.iswl = i_swl; H.es.isgcr = i_sgcr; H.es.isgu = i_sgu;
  H.es.isgl = i_sgl; H.es.isogcr = i_sogcr;
  H.hyst_on = 1; H.sg_hy = sg_hy;
  const double kg_rev = relperm3(.35, sg_hy[0], H, 1,1,1,1,1,1,.1,.1,0).Mg;
  const double kg_left = relperm3(.35, sg_hy[0] - 1e-8, H,
                                  1,1,1,1,1,1,.1,.1,0).Mg;
  const double kg_scan = relperm3(.35, .25, H, 1,1,1,1,1,1,.1,.1,0).Mg;
  SatTables D = C3; D.hyst_on = 0;
  const double kg_drain = relperm3(.35, .25, D, 1,1,1,1,1,1,.1,.1,0).Mg;
  const bool hpass = fabs(kg_rev - kg_left) < 1e-6 && kg_scan < kg_drain;
  std::printf("TEST Carlson hysteresis: reversal jump %.3e, scan %.5f < "
              "drainage %.5f  %s\n", fabs(kg_rev - kg_left), kg_scan,
              kg_drain, hpass ? "PASS" : "FAIL");

  // OPM applies the same Carlson model to the non-wetting oil curve in the
  // oil/water subsystem.  Its history coordinate is min(Sw+Sg)=1-max(So).
  // Use a displaced imbibition residual-oil endpoint so the scan is visibly
  // below drainage while remaining continuous at reversal.
  Real i_swcr[1]{b_swcr[0]}, i_swu[1]{b_swu[0]};
  Real i_sowcr[1]{0.30}, sw_ow_hy[1]{0.35};
  H.es.iswcr = i_swcr; H.es.iswu = i_swu;
  H.es.isowcr = i_sowcr; H.sw_ow_hy = sw_ow_hy;
  const double ko_rev = relperm3(sw_ow_hy[0] - .05, .05, H,
                                  1,1,1,1,1,1,.1,.1,0).Mo;
  const double ko_right = relperm3(sw_ow_hy[0] - .05 + 1e-8, .05, H,
                                    1,1,1,1,1,1,.1,.1,0).Mo;
  const double ko_scan = relperm3(.50, .05, H, 1,1,1,1,1,1,.1,.1,0).Mo;
  const double ko_drain = relperm3(.50, .05, D, 1,1,1,1,1,1,.1,.1,0).Mo;
  const bool owpass = fabs(ko_rev - ko_right) < 1e-6 && ko_scan < ko_drain;
  std::printf("TEST Carlson oil/water: reversal jump %.3e, scan %.5f < "
              "drainage %.5f  %s\n", fabs(ko_rev - ko_right), ko_scan,
              ko_drain, owpass ? "PASS" : "FAIL");

  // Norne partially COPYs I-endpoints. Untouched cells arrive as NaN and
  // must inherit imbibition-table values rather than acquire a zero-width gas
  // curve.
  Real missing[1]{NAN};
  SatTables P = H;
  P.es.isgu = missing; P.es.iswl = missing;
  P.es.isgl = missing; P.es.isogcr = missing;
  const CellEnd pi = imb_cell_end(P.es, P, 0);
  const double pm_rev = relperm3(.35, sg_hy[0], P,
                                  1,1,1,1,1,1,.1,.1,0).Mg;
  const double pm_left = relperm3(.35, sg_hy[0] - 1e-8, P,
                                   1,1,1,1,1,1,.1,.1,0).Mg;
  const bool ppass = isfinite(pm_left) && fabs(pm_rev - pm_left) < 1e-6 &&
                     pi.sgu == P.sg_up && pi.swl == P.sw_lo &&
                     pi.sgcr == i_sgcr[0];
  std::printf("TEST partial I-endpoints: reversal jump %.3e, "
              "ISGU %.2f=table %.2f  %s\n",
              fabs(pm_rev - pm_left), pi.sgu, P.sg_up,
              ppass ? "PASS" : "FAIL");

  return 0;
}
