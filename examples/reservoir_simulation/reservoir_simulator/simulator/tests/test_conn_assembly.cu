// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_conn_assembly.cu -- the connection-list assembly against the 7-point one.
//
// The claim being tested
// ----------------------
// Generalising the assembly from a fixed six-face stencil to an arbitrary
// connection list is only safe if, on a grid where both descriptions apply,
// they produce the same linear system.  A Cartesian box is such a grid: its
// connection list holds exactly the six faces the fixed stencil holds.  So the
// two kernels are run on the same state and every array they fill is compared
// -- the residual, the whole coupled Jacobian, the CPR pressure block, the
// block-diagonal inverse and the quasi-IMPES weights.
//
// This is a stronger check than comparing a converged answer.  A wrong
// Jacobian usually still converges, just more slowly and to a slightly
// different place; comparing the matrix entry by entry catches the error where
// it happens rather than several timesteps downstream.
//
// The two are not expected to agree bit for bit.  The connection-list kernel
// forms the gravity head as (p_j - p_i) - g*rho*(z_j - z_i) where the old one
// forms (p_j - g*rho*z_j) - (p_i - g*rho*z_i).  Those are equal in exact
// arithmetic and differ in the last bits in floating point -- the new form
// being the better-conditioned of the two, since it never subtracts two large
// g*rho*z terms.  So the comparison is to a relative tolerance.
//
//   nvcc -o t test_conn_assembly.cu ../src/*.cu ../src/deck.cpp -I../include
//   ./t

#include "common.hpp"
#include "deck.hpp"
#include "grid.hpp"
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

// Largest relative difference between two arrays, scaled by the larger
// magnitude present so that a big entry and a tiny one are judged alike.
template <typename T>
double max_rel_diff(const std::vector<T>& a, const std::vector<T>& b,
                    size_t* where = nullptr) {
  double scale = 0;
  for (size_t i = 0; i < a.size(); ++i) scale = std::max(scale, std::fabs((double)a[i]));
  if (scale == 0) scale = 1;
  double worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs((double)a[i] - (double)b[i]) / scale;
    if (d > worst) { worst = d; if (where) *where = i; }
  }
  return worst;
}

// A small three-phase deck with gravity on, a producer and an injector, so the
// comparison exercises the well terms and the upstream switching rather than
// just the interior flux.
std::string write_deck(int nx, int ny, int nz) {
  const std::string p = tmpdir() + "/nvrs_conn_test.DATA";
  std::ofstream f(p);
  f << "TITLE   CONNECTION-LIST ASSEMBLY CROSS-CHECK\n"
    // Three phases, so the kernel under test is the three-phase analytic one.
    << "NCOMP   3\nMODEL   BLACKOIL\n"
    << "OUTDIR  OUTPUT_CONNTEST\n"
    << "NX      " << nx << "\nNY      " << ny << "\nNZ      " << nz << "\n"
    << "DX      50\nDY      50\nDZ      20\n"
    << "KZMULT  0.2\n"
    // A heterogeneous field: uniform permeability would make many Jacobian
    // entries coincide, which would weaken the comparison.
    << "PERMMIN 10\nPERMMAX 800\nPOROMIN 0.08\nPOROMAX 0.28\n"
    << "UW 1.0\nUO 2.5\nUG 0.02\n"
    << "BW 1.0\nBO 1.2\nBG 0.005\n"
    << "RHOW 1.0\nRHOO 0.8\nRHOG 0.01\n"
    << "CFO 0.00001\n"
    // Compressibility off, so the two kernels are comparing the same physics.
    // The connection-list kernel carries a total-compressibility term in the
    // accumulation and the fixed-stencil one does not: that term is what keeps
    // the pressure operator non-singular on a field model under depletion, and
    // it was added only to the connection-list path so that every existing
    // structured benchmark stays bit-identical.  With CT 0 it vanishes and the
    // equivalence this test exists to check is exact again.
    << "CT 0\n"
    << "SWI 0.1\nSWR 0.1\n"
    << "PINIT 3000\nPBUB 3000\nPATM 14.6959\n"
    << "SWINIT 0.2\nSGINIT 0.05\nSOINIT 0.75\n"
    // Gravity on: the depth term is exactly what the two kernels compute
    // differently, so leaving it off would skip the interesting comparison.
    << "GRAVITY 1\n"
    << "JACOBIAN ANALYTIC\n"
    << "AIM 1\n"
    << "SWOF\n 0.00 1.0000 0.0000\n 0.30 1.0000 0.0000\n"
       " 0.50 0.0800 0.0460\n 0.70 0.0001 0.1660\n 1.00 0.0000 0.2400\n/\n"
    << "SGOF\n 0.00 1.0000 0.0000\n 0.10 0.3200 0.0900\n"
       " 0.25 0.0300 0.3300\n 0.40 0.0000 0.6700\n 1.00 0.0000 1.0000\n/\n"
    << "DTREPORT 50\nTEND 200\n"
    // One injector and one producer, so the well terms are exercised too.
    << "WELLS\n"
       "   2   2   1    500   0.5   0.0  2850   all I1\n"
    << "   " << (nx - 1) << "   " << (ny - 1)
    << "  -1     -1   0.5   0.0  2850   all P1\n/\n";
  return p;
}

// Run one assembly and take a copy of everything it wrote.
struct Snapshot {
  Real rnorm = 0;
  std::vector<Real> aval, b;
  std::vector<PrecReal> pval, dinv, pweight;
};

Snapshot run_assembly(Reservoir& r, DeviceLinearSystem& sys, int conn) {
  r.use_conn_list = conn;
  // Poison the outputs first, so an entry the kernel fails to write shows up
  // as a difference instead of quietly inheriting the other run's value.
  NVRS_CHECK_CUDA(cudaMemset(sys.aval64, 0, sizeof(Real) * sys.nnz));
  NVRS_CHECK_CUDA(cudaMemset(sys.pval, 0, sizeof(PrecReal) * sys.pnnz));
  NVRS_CHECK_CUDA(cudaMemset(sys.b, 0, sizeof(Real) * sys.n));

  Snapshot s;
  s.rnorm = assemble_blackoil(r, sys);
  NVRS_CHECK_CUDA(cudaDeviceSynchronize());

  const int cells = r.grid.n_local();
  s.aval = pull(sys.aval64, (size_t)sys.nnz);
  s.b = pull(sys.b, (size_t)sys.n);
  s.pval = pull(sys.pval, (size_t)sys.pnnz);
  s.dinv = pull(sys.dinv, (size_t)cells * r.bs * r.bs);
  s.pweight = pull(sys.pweight, (size_t)cells * r.bs);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  std::printf("NVRS connection-list assembly test\n");
  std::printf("==================================\n");

  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (ndev == 0) { std::printf("no CUDA device; skipping\n"); return 0; }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

  const int nx = 12, ny = 10, nz = 6;
  const std::string deck_path = write_deck(nx, ny, nz);

  Deck d = parse_deck(deck_path);
  Grid g = Grid::make(d.nx, d.ny, d.nz, d.dx, d.dy, d.dz, 0, 1);
  Reservoir r = make_reservoir(d, g);
  DeviceLinearSystem sys;
  build_sparsity(r, sys);

  std::printf("grid   : %d x %d x %d = %d cells, bs = %d\n",
              nx, ny, nz, g.n_local(), r.bs);
  std::printf("matrix : %d rows, %d nonzeros; pressure block %d\n",
              sys.n, sys.nnz, sys.pnnz);
  std::printf("conns  : %d entries, max degree %d\n\n", r.nconn, r.max_degree);

  check(r.bs == 3, "three-phase black oil");
  check(r.conn_ptr != nullptr, "the connection list was built");
  check(r.max_degree == 6, "a Cartesian interior cell has six neighbours");

  // The connection list must describe the same graph as the 7-point arrays.
  {
    const int cells = g.n_local();
    const std::vector<int> cptr = pull(r.conn_ptr, (size_t)cells + 1);
    const std::vector<int> cnbr = pull(r.conn_nbr, (size_t)r.nconn);
    const std::vector<int> nbr7 = pull(r.nbr, (size_t)cells * 7);
    const std::vector<Real> tf = pull(r.tface, (size_t)cells * 6);

    bool same = true;
    long long total = 0;
    for (int c = 0; c < cells && same; ++c) {
      // every live face of the fixed stencil appears exactly once in the list
      for (int f = 0; f < 6; ++f) {
        const int nc = nbr7[7 * c + 1 + f];
        if (nc < 0 || !(tf[6 * c + f] > 0)) continue;
        ++total;
        int found = 0;
        for (int m = cptr[c]; m < cptr[c + 1]; ++m)
          if (cnbr[m] == nc) ++found;
        if (found != 1) same = false;
      }
    }
    check(same, "every stencil face appears exactly once in the connection list");
    check(total == (long long)r.nconn,
          "and the list holds no connections the stencil does not");
  }

  // --- the comparison ------------------------------------------------------
  const Snapshot stencil = run_assembly(r, sys, 0);
  const Snapshot conn    = run_assembly(r, sys, 1);

  std::printf("\nresidual norm: stencil %.17g\n               conn    %.17g\n",
              stencil.rnorm, conn.rnorm);

  const double tol = 1e-12;
  size_t at = 0;

  const double d_b = max_rel_diff(stencil.b, conn.b, &at);
  std::printf("\n  rhs        max rel diff %.3e (row %zu)\n", d_b, at);
  check(d_b < tol, "the residual matches");

  const double d_a = max_rel_diff(stencil.aval, conn.aval, &at);
  std::printf("  jacobian   max rel diff %.3e (entry %zu of %d)\n",
              d_a, at, sys.nnz);
  check(d_a < tol, "every Jacobian entry matches");

  const double d_p = max_rel_diff(stencil.pval, conn.pval, &at);
  std::printf("  pressure   max rel diff %.3e (entry %zu of %d)\n",
              d_p, at, sys.pnnz);
  check(d_p < 1e-6, "the CPR pressure block matches (FP32 storage)");

  const double d_d = max_rel_diff(stencil.dinv, conn.dinv, &at);
  std::printf("  dinv       max rel diff %.3e\n", d_d);
  check(d_d < 1e-6, "the block-diagonal inverse matches");

  const double d_w = max_rel_diff(stencil.pweight, conn.pweight, &at);
  std::printf("  pweight    max rel diff %.3e\n", d_w);
  check(d_w < 1e-6, "the quasi-IMPES weights match");

  const double rn = std::fabs(stencil.rnorm - conn.rnorm) /
                    (std::fabs(stencil.rnorm) + 1e-300);
  check(rn < tol, "the residual norms match");

  // Nothing may be left unwritten: a zero row would mean the connection-list
  // kernel skipped a block the stencil filled.
  {
    int zero_rows = 0;
    const std::vector<int> rp = pull(sys.rowptr, (size_t)sys.n + 1);
    for (int row = 0; row < sys.n; ++row) {
      bool any = false;
      for (int k = rp[row]; k < rp[row + 1]; ++k)
        if (conn.aval[k] != 0.0) { any = true; break; }
      if (!any) ++zero_rows;
    }
    check(zero_rows == 0, "no row of the connection-list Jacobian is empty");
  }

  free_reservoir(r);
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  MPI_Finalize();
  return g_failures ? 1 : 0;
}
