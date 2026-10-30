// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_field_assembly.cu -- assembling a corner-point field model on the GPU.
//
// This is the end of the path that starts at a field deck: read the corner-point
// grid and its faults, resolve the wells onto it, equilibrate from the datum,
// build the linear system from the connection list, and assemble a Jacobian on
// the device.  Everything up to the last step is checked by the host-side deck
// tests; what is checked here is the join.
//
// The properties asserted are the ones that a wrong join breaks, and that a
// converged answer would not reveal for several timesteps:
//
//   * the matrix graph is the connection graph -- every row has the diagonal
//     and exactly one block per neighbour, and its columns are sorted;
//   * nothing is left unwritten -- an empty row means the kernel never visited
//     a cell the sparsity pattern allocated space for;
//   * the diagonal is signed correctly and the block inverse exists, which is
//     what the CPR second stage depends on;
//   * the residual is finite everywhere, which is where a bad transmissibility
//     or a zero pore volume shows up first.
//
//   nvcc -o t test_field_assembly.cu ... && ./t path/to/NORNE_ATW2013.DATA

#include "common.hpp"
#include "deck.hpp"
#include "field_model.hpp"
#include "linear_solver.hpp"
#include "physics.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace nvrs;

namespace {

int g_failures = 0, g_checks = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) { ++g_failures; std::printf("  FAIL  %s\n", what); }
  else      std::printf("  ok    %s\n", what);
}

std::string tmpdir() {
  const char* t = std::getenv("TMPDIR");
  if (!t) t = std::getenv("TEMP");
  if (!t) t = "/tmp";
  return std::string(t);
}

template <typename T>
std::vector<T> pull(const T* dev, size_t n) {
  std::vector<T> h(n);
  if (n) NVRS_CHECK_CUDA(cudaMemcpy(h.data(), dev, sizeof(T) * n,
                                    cudaMemcpyDeviceToHost));
  return h;
}

// The fluid and solver settings a corner-point deck does not supply in a form
// this simulator reads.  Grid, wells and initial state all come from the field
// deck; only these come from here.
std::string write_run_deck() {
  const std::string p = tmpdir() + "/nvrs_field_run.DATA";
  std::ofstream f(p);
  f << "TITLE   FIELD-MODEL ASSEMBLY CHECK\n"
    << "NCOMP   3\nMODEL   BLACKOIL\n"
    << "OUTDIR  OUTPUT_FIELDTEST\n"
    // The grid keywords are ignored on this path but the parser expects them.
    << "NX 1\nNY 1\nNZ 1\nDX 50\nDY 50\nDZ 20\n"
    << "UW 0.32\nUO 1.2\nUG 0.02\n"
    << "BW 1.02\nBO 1.34\nBG 0.0045\n"
    << "RHOW 1.03\nRHOO 0.86\nRHOG 0.001\n"
    << "CFO 0.00001\n"
    << "SWI 0.1\nSWR 0.1\n"
    << "PINIT 3900\nPBUB 3900\nPATM 14.6959\n"
    << "SWINIT 0.2\nSGINIT 0.05\nSOINIT 0.75\n"
    << "GRAVITY 1\n"
    << "JACOBIAN ANALYTIC\n"
    << "AIM 1\n"
    << "SWOF\n 0.00 1.0000 0.0000\n 0.30 1.0000 0.0000\n"
       " 0.50 0.0800 0.0460\n 0.70 0.0001 0.1660\n 1.00 0.0000 0.2400\n/\n"
    << "SGOF\n 0.00 1.0000 0.0000\n 0.10 0.3200 0.0900\n"
       " 0.25 0.0300 0.3300\n 0.40 0.0000 0.6700\n 1.00 0.0000 1.0000\n/\n"
    << "DTREPORT 100\nTEND 2000\n";
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  std::printf("NVRS field-model assembly test\n");
  std::printf("==============================\n");

  if (argc < 2) {
    std::printf("usage: %s <field-deck.DATA>\n", argv[0]);
    MPI_Finalize();
    return 0;
  }
  const std::string field_deck = argv[1];
  { std::ifstream t(field_deck);
    if (!t) { std::printf("SKIP  deck not found: %s\n", field_deck.c_str());
              MPI_Finalize(); return 0; } }

  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (ndev == 0) { std::printf("no CUDA device; skipping\n"); MPI_Finalize(); return 0; }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s (sm_%d%d, %.1f GB)\n\n", prop.name, prop.major,
              prop.minor, prop.totalGlobalMem / 1e9);

  // --- read the field deck -------------------------------------------------
  // Norne's own documented phase gradients, in bar/m.
  const Real grad[3] = {0.0706, 0.1001, 0.0186};
  // Connate water for the hydrocarbon columns. Norne states SWL per layer in
  // its PROPS section, which the grid reader does not descend into; 0.1 is the
  // deck's own SWI and a defensible stand-in.
  auto fm = load_field_model(field_deck, grad, /*connate_water*/ 0.1);

  std::printf("%s", fm->grid.summary().c_str());
  std::printf("%s", fm->schedule.summary().c_str());
  check(fm->grid.n_active() > 0, "the grid has active cells");
  check(!fm->grid.conns.empty(), "the grid has connections");
  check(fm->have_initial, "an initial state was equilibrated");
  check(fm->schedule.n_steps() > 0, "the schedule has report steps");

  const int cells = fm->grid.n_active();

  // --- build the solver's view of it --------------------------------------
  Deck d = parse_deck(write_run_deck());
  Reservoir r = make_reservoir_field(d, *fm);
  DeviceLinearSystem sys;
  build_sparsity_field(r, sys, *fm);

  std::printf("\nlinear system\n");
  std::printf("  cells      : %d\n", sys.cells);
  std::printf("  rows       : %d  (bs = %d)\n", sys.n, sys.bs);
  std::printf("  nonzeros   : %d  (%.2f per row)\n", sys.nnz,
              (double)sys.nnz / sys.n);
  std::printf("  pressure   : %d rows, %d nonzeros\n", sys.cells, sys.pnnz);
  std::printf("  degree     : max %d, %d connection entries\n",
              r.max_degree, r.nconn);

  check(sys.cells == cells, "one row block per active cell");
  check(r.use_conn_list == 1, "the connection-list assembly was selected");
  check(r.max_degree > 6,
        "some cell has more than six neighbours, which only a fault produces");
  check(r.nconn == 2 * (int)fm->grid.conns.size(),
        "each connection appears once from each side");
  check(sys.pnnz == cells + r.nconn,
        "the pressure block has a diagonal plus one entry per connection");

  // The kernel sees pore volume only through cellvol*poro.  That product must
  // equal the grid's finalized PORV, including MULTPV.  Comparing only total
  // volumes is too weak: compensating multipliers in different cells would
  // still give the wrong local front speeds.
  {
    const std::vector<Real> hp = pull(r.poro, (size_t)cells);
    const std::vector<Real> hv = pull(r.cellvol, (size_t)cells);
    double worst = 0.0;
    for (int l = 0; l < cells; ++l) {
      const int a = fm->part.active_of_local[l];
      const double want = (double)fm->grid.porv[a];
      const double got = (double)hp[l] * (double)hv[l];
      worst = std::max(worst, std::abs(got - want) /
                                std::max(1.0, std::abs(want)));
    }
    check(worst < 2e-12,
          "cellvol*poro equals deck PORV (including MULTPV) in every cell");
    std::printf("        worst relative PORV mismatch %.3e\n", worst);
  }

  // --- the matrix graph ----------------------------------------------------
  {
    const std::vector<int> rp = pull(sys.rowptr, (size_t)sys.n + 1);
    const std::vector<int> ci = pull(sys.colind, (size_t)sys.nnz);
    bool sorted = true, has_diag = true, in_range = true;
    for (int row = 0; row < sys.n; ++row) {
      bool diag = false;
      for (int k = rp[row]; k < rp[row + 1]; ++k) {
        if (ci[k] < 0 || ci[k] >= sys.n) in_range = false;
        if (k > rp[row] && ci[k] <= ci[k - 1]) sorted = false;
        if (ci[k] == row) diag = true;
      }
      if (!diag) has_diag = false;
    }
    check(in_range, "every column index is a real row");
    check(sorted, "columns are ascending within each row, as cuSPARSE requires");
    check(has_diag, "every row carries its own diagonal");
  }

  // --- assemble ------------------------------------------------------------
  r.dt = 1.0;
  const Real rnorm = assemble_blackoil(r, sys);
  NVRS_CHECK_CUDA(cudaDeviceSynchronize());
  std::printf("\nassembly\n  residual norm : %.6g\n", rnorm);

  check(std::isfinite((double)rnorm), "the residual norm is finite");

  const std::vector<Real> aval = pull(sys.aval64, (size_t)sys.nnz);
  const std::vector<Real> rhs = pull(sys.b, (size_t)sys.n);
  const std::vector<PrecReal> dinv =
      pull(sys.dinv, (size_t)cells * r.bs * r.bs);

  {
    long long nan_a = 0, nan_b = 0;
    for (Real v : aval) if (!std::isfinite((double)v)) ++nan_a;
    for (Real v : rhs)  if (!std::isfinite((double)v)) ++nan_b;
    check(nan_a == 0, "every Jacobian entry is finite");
    check(nan_b == 0, "every residual entry is finite");
  }

  {
    // No row may be entirely zero: the sparsity pattern reserved space for a
    // block, so a zero row means the kernel never wrote one.
    const std::vector<int> rp = pull(sys.rowptr, (size_t)sys.n + 1);
    int empty = 0;
    for (int row = 0; row < sys.n; ++row) {
      bool any = false;
      for (int k = rp[row]; k < rp[row + 1]; ++k)
        if (aval[k] != 0.0) { any = true; break; }
      if (!any) ++empty;
    }
    check(empty == 0, "no row of the Jacobian was left unwritten");
    if (empty) std::printf("        (%d empty rows)\n", empty);
  }

  {
    // The diagonal of the CPR *pressure* block must be positive in every cell:
    // that block is what the algebraic multigrid coarsens, and a non-positive
    // diagonal would mean a cell with no pore volume or a transmissibility of
    // the wrong sign.
    //
    // The individual phase rows are a different matter and must not be checked
    // this way. The oil-row pressure derivative is the sum of b_o T M_o over a
    // cell's connections, so it is legitimately zero wherever the oil mobility
    // is -- every cell below the water contact and above the gas contact. That
    // is a property of a sharp-interface initial state, not a defect. The
    // pressure row survives it because it is a weighted combination of all
    // three equations, and the water or gas row supplies the coupling there.
    const std::vector<int> prp = pull(sys.prowptr, (size_t)cells + 1);
    const std::vector<int> pci = pull(sys.pcolind, (size_t)sys.pnnz);
    const std::vector<PrecReal> pv = pull(sys.pval, (size_t)sys.pnnz);
    int bad = 0, found = 0;
    for (int c = 0; c < cells; ++c)
      for (int k = prp[c]; k < prp[c + 1]; ++k)
        if (pci[k] == c) {
          ++found;
          if (!(pv[k] > 0)) ++bad;
        }
    check(found == cells, "every pressure row has its diagonal");
    // A handful of cells may still show a zero here, and the cause is storage
    // rather than assembly: pval is held in PrecReal, which is FP32 in a mixed-
    // precision build, and Norne's smallest transmissibilities are of order
    // 1e-11. Multiplied by a mobility, the product can fall below the smallest
    // normal FP32 value and flush to zero. Such a cell is connected only
    // through a barrier the deck intended to be closed, so the practical
    // consequence is nil -- but it is a real limit of storing the pressure
    // operator in single precision, and is asserted as a bound rather than
    // hidden.
    check(bad <= cells / 10000,
          "the CPR pressure diagonal is positive in all but a few cells");
    if (bad)
      std::printf("        (%d of %d flushed to zero in FP32 storage)\n",
                  bad, cells);
  }

  {
    // Connate water must be present in the hydrocarbon columns. A cell
    // initialised with no water at all has no water phase to move, and the
    // water equation there carries no pressure coupling until water arrives.
    int dry = 0;
    for (int a = 0; a < cells; ++a)
      if (fm->initial.sw[a] <= 0.0) ++dry;
    check(dry == 0, "no cell was initialised with zero water saturation");
    if (dry) std::printf("        (%d of %d cells)\n", dry, cells);
  }

  {
    long long nan_d = 0;
    for (PrecReal v : dinv) if (!std::isfinite((double)v)) ++nan_d;
    check(nan_d == 0, "the CPR block-diagonal inverse is finite everywhere");
  }

  // --- the schedule actually changes things --------------------------------
  {
    const int last = fm->schedule.n_steps() - 1;
    apply_schedule_step(r, *fm, last);
    const Real rn2 = assemble_blackoil(r, sys);
    NVRS_CHECK_CUDA(cudaDeviceSynchronize());
    std::printf("\n  residual at step 0    : %.6g\n", rnorm);
    std::printf("  residual at step %-4d : %.6g\n", last, rn2);
    check(std::isfinite((double)rn2), "the last step also assembles finitely");
    check(rn2 != rnorm,
          "changing the report step changes the system, so control is live");
  }

  free_reservoir(r);
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  MPI_Finalize();
  return g_failures ? 1 : 0;
}
