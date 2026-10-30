// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_units.cpp -- METRIC run decks.
//
// The property that matters is round-tripping: a deck written in metric and
// the same deck written in field units must describe the SAME reservoir.  A
// conversion that is merely self-consistent is not enough, because every
// wrong factor is also self-consistent -- it just simulates a different
// field, quietly, and the run converges and reports success.
//
// Rates are the case worth guarding.  12811 sm3/day read as 12811 stb/day is
// a sixth of the intended injection, and nothing downstream can tell.
//
//   g++ -O2 -std=c++17 -I../include test_units.cpp ../src/deck.cpp -o t && ./t
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include "deck.hpp"

using namespace nvrs;

static int fails = 0;

static void near(const char* what, double got, double want, double tol) {
  const double e = std::fabs(got - want) / (std::fabs(want) > 1e-30 ? std::fabs(want) : 1.0);
  const bool ok = e <= tol;
  if (!ok) ++fails;
  std::printf("  %-26s %14.6g  want %14.6g   %s\n", what, got, want,
              ok ? "ok" : "FAIL");
}

static std::string write_deck(const char* name, bool metric) {
  const std::string path = std::string(name);
  std::ofstream f(path);
  f << (metric ? "METRIC\n" : "FIELD\n");
  f << "LINESEARCH 7\n";
  f << "WELLSCHUR 1\n";
  // 30 m == 98.4252 ft;  250 bar == 3625.94 psi;  1000 kg/m3 == 1.0 g/cc
  if (metric) {
    f << "DX 30\nDY 30\nDZ 6\n"
      << "PINIT 250\nPBUB 200\nPATM 1.01325\n"
      << "PRODBHP 100\nINJBHP 400\nDPMAX 34.4738\n"
      << "CT 1.45038e-4\n"
      << "RHOW 1000\nRHOO 800\nRHOG 10\n"
      << "RS 100\n"
      << "WELLS\n"
      << "1 1  1  1590.0  0.1524 0 250 all INJ1\n"
      << "5 5 -1    -1    0.1524 0 100 all PRO1\n"
      << "/\n";
  } else {
    f << "DX 98.4252\nDY 98.4252\nDZ 19.68504\n"
      << "PINIT 3625.943\nPBUB 2900.755\nPATM 14.69595\n"
      << "PRODBHP 1450.377\nINJBHP 5801.509\nDPMAX 500.0\n"
      << "CT 1.0e-5\n"
      << "RHOW 1.0\nRHOO 0.8\nRHOG 0.01\n"
      << "RS 561.458\n"
      << "WELLS\n"
      << "1 1  1  10000.8  0.5 0 3625.943 all INJ1\n"
      << "5 5 -1    -1     0.5 0 1450.377 all PRO1\n"
      << "/\n";
  }
  return path;
}

int main() {
  const Deck m = parse_deck(write_deck("_u_metric.DATA", true));
  const Deck g = parse_deck(write_deck("_u_field.DATA", false));

  std::printf("METRIC deck, converted:\n");
  near("DX [ft]",          m.dx,        g.dx,        1e-4);
  near("DZ [ft]",          m.dz,        g.dz,        1e-4);
  near("PINIT [psi]",      m.p_init,    g.p_init,    1e-4);
  near("PATM [psi]",       m.p_atm,     g.p_atm,     1e-4);
  near("PRODBHP [psi]",    m.prod_bhp,  g.prod_bhp,  1e-4);
  near("INJBHP [psi]",     m.inj_bhp,   g.inj_bhp,   1e-4);
  near("DPMAX [psi]",      m.dp_max,    g.dp_max,    1e-4);
  near("CT [1/psi]",       m.ct,        g.ct,        1e-4);
  near("RHOW [g/cc]",      m.rhow,      g.rhow,      1e-9);
  near("RHOG [g/cc]",      m.rhog,      g.rhog,      1e-9);
  near("RS [scf/stb]",     m.rs_const,  g.rs_const,  1e-4);
  near("LINESEARCH",        m.line_search, 7.0,       0.0);
  near("WELLSCHUR",         m.well_schur, 1.0,        0.0);
  near("CNVRELAXED default", m.cnv_relaxed, 1.0,      0.0);

  if (m.wells.size() == 2 && g.wells.size() == 2) {
    near("well rw [ft]",     m.wells[0].rw,   g.wells[0].rw,   1e-3);
    near("well pwf [psi]",   m.wells[0].pwf,  g.wells[0].pwf,  1e-4);
    near("inj rate [stb/d]", m.wells[0].rate, g.wells[0].rate, 1e-4);
    // The -1 sentinel means "no rate target"; scaling it would turn a
    // producer into an injector of -6.29 stb/day.
    near("producer sentinel", m.wells[1].rate, -1.0, 1e-12);
  } else {
    std::printf("  wells not parsed (%zu / %zu)  FAIL\n",
                m.wells.size(), g.wells.size());
    ++fails;
  }

  // Idempotent: a converted deck must not convert twice.
  Deck again = m;
  to_field_units(again);
  near("idempotent PINIT", again.p_init, m.p_init, 1e-12);

  std::printf("\n%s\n", fails ? "FAILED" : "all unit conversions agree");
  std::remove("_u_metric.DATA");
  std::remove("_u_field.DATA");
  return fails != 0;
}
