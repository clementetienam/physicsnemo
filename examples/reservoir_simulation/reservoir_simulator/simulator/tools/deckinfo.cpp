// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// deckinfo.cpp -- read a field deck and report exactly what was understood.
//
// This is the tool to run before trying to simulate an unfamiliar deck.  It
// loads the grid, the schedule and the equilibration, and prints what each
// reader made of them, including everything it did *not* understand.  A
// simulator that silently ignores half a deck will still produce a plausible
// looking answer, so the value here is in the negative space: the keywords
// listed as not implemented, the completions that landed in dead cells, the
// includes that were not found.
//
//   deckinfo NORNE_ATW2013.DATA
//   deckinfo NORNE_ATW2013.DATA --gradients 0.0706,0.1001,0.0186
//   deckinfo NORNE_ATW2013.DATA --fvf 1.336,1.0113,0.00456 --rs 120
//   deckinfo NORNE_ATW2013.DATA --csv out_prefix
//
// The exit status is non-zero if the deck could not be read at all, so this
// can be used as a gate in a script.
#include "cpgrid.hpp"
#include "equil.hpp"
#include "schedule.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace nvrs;

namespace {

void rule(const char* title) {
  std::printf("\n"
              "================================================================\n"
              " %s\n"
              "================================================================\n",
              title);
}

bool parse_triple(const char* s, Real& a, Real& b, Real& c) {
  return std::sscanf(s, "%lf,%lf,%lf", &a, &b, &c) == 3;
}

// Per-well summary at a chosen step: what it is, how it is controlled, and
// where it is completed.  This is the table that answers "did the deck's wells
// come through intact".
void report_wells(const Schedule& sched, int step, const CornerPointGrid& grid) {
  if (step < 0 || step >= sched.n_steps()) return;
  const ScheduleStep& s = sched.step(step);

  std::printf("\n  wells at step %d (%s):\n", step, s.date.c_str());
  std::printf("    %-8s %-9s %-6s %-6s %10s  %5s %5s %5s  %s\n",
              "well", "type", "mode", "status", "target", "comp", "live",
              "horiz", "depth range");
  std::printf("    %s\n", std::string(78, '-').c_str());

  for (const Well& w : s.wells) {
    int live = 0, horiz = 0;
    Real zmin = 1e30, zmax = -1e30;
    for (const Completion& c : w.completions) {
      if (c.dir != PenetrationDir::Z) ++horiz;
      if (c.cell < 0) continue;
      ++live;
      zmin = std::min(zmin, grid.depth[c.cell]);
      zmax = std::max(zmax, grid.depth[c.cell]);
    }
    char depth[48] = "-";
    if (live) std::snprintf(depth, sizeof depth, "%.0f - %.0f", zmin, zmax);

    const Real t = w.state.target();
    char target[32];
    if (t > 0 && std::isfinite((double)t)) std::snprintf(target, sizeof target, "%.1f", t);
    else std::snprintf(target, sizeof target, "-");

    std::printf("    %-8s %-9s %-6s %-6s %10s  %5d %5d %5d  %s\n",
                w.name.c_str(), to_string(w.state.type),
                to_string(w.state.mode),
                w.state.open ? "open" : "shut", target,
                (int)w.completions.size(), live, horiz, depth);
  }
}

void write_csv(const std::string& prefix, const CornerPointGrid& g,
               const InitialState* init) {
  {
    std::ofstream f(prefix + "_cells.csv");
    f << "active,i,j,k,x,y,depth,dx,dy,dz,volume,porv,poro,ntg,permx,permy,permz";
    if (init) f << ",pressure,sw,so,sg,eqlnum";
    f << "\n";
    for (int a = 0; a < g.n_active(); ++a) {
      const int gi = g.global_of_active(a);
      const int i = gi % g.nx, j = (gi / g.nx) % g.ny, k = gi / (g.nx * g.ny);
      f << a << ',' << i + 1 << ',' << j + 1 << ',' << k + 1 << ','
        << g.centroid_x[a] << ',' << g.centroid_y[a] << ',' << g.depth[a] << ','
        << g.dx[a] << ',' << g.dy[a] << ',' << g.dz[a] << ','
        << g.volume[a] << ',' << g.porv[a] << ',' << g.poro[a] << ','
        << g.ntg[a] << ',' << g.permx[a] << ',' << g.permy[a] << ','
        << g.permz[a];
      if (init)
        f << ',' << init->pressure[a] << ',' << init->sw[a] << ','
          << init->so[a] << ',' << init->sg[a] << ',' << init->region[a];
      f << "\n";
    }
  }
  {
    std::ofstream f(prefix + "_conns.csv");
    f << "a,b,trans,dir\n";
    for (const Connection& c : g.conns)
      f << c.a << ',' << c.b << ',' << c.trans << ',' << c.dir << "\n";
  }
  std::printf("\n  wrote %s_cells.csv and %s_conns.csv\n",
              prefix.c_str(), prefix.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: deckinfo <deck.DATA> [--gradients o,w,g] "
                 "[--fvf bo,bw,bg] [--rs value] [--step n] [--csv prefix]\n");
    return 2;
  }

  const std::string deck = argv[1];
  Real go = 0, gw = 0, gg = 0;
  Real bo = 0, bw = 0, bg = 0, rs = 0;
  bool have_grad = false, have_fvf = false;
  int step = -1;
  std::string csv;

  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--gradients") && i + 1 < argc)
      have_grad = parse_triple(argv[++i], go, gw, gg);
    else if (!std::strcmp(argv[i], "--fvf") && i + 1 < argc)
      have_fvf = parse_triple(argv[++i], bo, bw, bg);
    else if (!std::strcmp(argv[i], "--rs") && i + 1 < argc)
      rs = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--step") && i + 1 < argc)
      step = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc)
      csv = argv[++i];
  }

  std::printf("NVRS deck reader\n  deck: %s\n", deck.c_str());

  // ---- grid --------------------------------------------------------------
  CornerPointGrid grid;
  rule("GRID");
  try {
    grid.read_grdecl(deck);
    grid.build();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\nERROR reading the grid: %s\n", e.what());
    return 1;
  }
  std::printf("%s", grid.summary().c_str());

  // ---- schedule ----------------------------------------------------------
  rule("SCHEDULE");
  Schedule sched;
  try {
    sched.read(deck);
    sched.resolve(grid);
    std::printf("%s", sched.summary().c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR reading the schedule: %s\n", e.what());
  }

  if (sched.n_steps() > 0) {
    if (step < 0) step = sched.n_steps() - 1;
    if (step >= sched.n_steps()) step = sched.n_steps() - 1;
    report_wells(sched, step, grid);

    if (!sched.orphan_completions().empty()) {
      std::printf("\n  completions in dead cells (%d):\n",
                  (int)sched.orphan_completions().size());
      int n = 0;
      for (const std::string& o : sched.orphan_completions()) {
        std::printf("    %s\n", o.c_str());
        if (++n == 20) {
          std::printf("    ... and %d more\n",
                      (int)sched.orphan_completions().size() - 20);
          break;
        }
      }
    }
  }

  // ---- equilibration -----------------------------------------------------
  rule("SOLUTION");
  Equilibration eq;
  const InitialState* init_ptr = nullptr;
  InitialState init;
  try {
    eq.read(deck);
    if (have_grad) eq.set_gradients(go, gw, gg);
    else if (have_fvf) eq.set_fvf(bo, bw, bg, rs);

    std::printf("%s", eq.summary().c_str());

    if (eq.have_gradients() && eq.n_regions() > 0) {
      init = eq.initialise(grid);
      init_ptr = &init;

      double pmin = 1e30, pmax = -1e30, pv_o = 0, pv_w = 0, pv_g = 0;
      for (int a = 0; a < grid.n_active(); ++a) {
        pmin = std::min(pmin, (double)init.pressure[a]);
        pmax = std::max(pmax, (double)init.pressure[a]);
        pv_o += init.so[a] * grid.porv[a];
        pv_w += init.sw[a] * grid.porv[a];
        pv_g += init.sg[a] * grid.porv[a];
      }
      std::printf("  initial state   : pressure %.2f .. %.2f\n", pmin, pmax);
      std::printf("  fluid in place  : oil %.4g, water %.4g, gas %.4g "
                  "(reservoir volume)\n", pv_o, pv_w, pv_g);
    } else if (!eq.have_gradients()) {
      std::printf("\n  No initial state computed: phase gradients were not\n"
                  "  supplied.  Pass --gradients o,w,g or --fvf bo,bw,bg,\n"
                  "  because they cannot be derived without the PVT tables\n"
                  "  and guessing them would corrupt the initial pressure\n"
                  "  invisibly.\n");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR in equilibration: %s\n", e.what());
  }

  if (!csv.empty()) write_csv(csv, grid, init_ptr);

  // ---- what is still missing ---------------------------------------------
  rule("READINESS");
  const bool grid_ok = grid.n_active() > 0 && !grid.conns.empty();
  const bool wells_ok = sched.n_steps() > 0;
  const bool init_ok = init_ptr != nullptr;
  std::printf("  grid and connections   : %s\n", grid_ok ? "yes" : "NO");
  std::printf("  wells and schedule     : %s\n", wells_ok ? "yes" : "NO");
  std::printf("  initial state          : %s\n",
              init_ok ? "yes" : "no (gradients not supplied)");
  std::printf("\n  Not modelled by these readers, and therefore not applied\n"
              "  even where the deck states them: PVT tables (live oil and\n"
              "  wet gas), saturation-function endpoint scaling, hysteresis,\n"
              "  VFP tables and tubing-head control, group control, and\n"
              "  tracers.  Each is listed above as it was encountered.\n");

  return grid_ok ? 0 : 1;
}
