// Reproduce build_sparsity's neighbour/column logic on the host and check the
// two properties the linear algebra depends on:
//
//   1. the pressure CSR column ids are the SAME global rows a serial assembly
//      would produce, so HYPRE is handed one consistent matrix however the
//      domain was cut;
//   2. every interior cell's face neighbours are found, and a face is treated
//      as no-flow only at the true reservoir boundary -- never at a rank
//      boundary, which would silently disconnect the domain.
#include "grid.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

using nvrs::Grid;
using nvrs::kFaceXlo; using nvrs::kFaceXhi;
using nvrs::kFaceYlo; using nvrs::kFaceYhi;
using nvrs::kFaceZlo; using nvrs::kFaceZhi;

static int failures = 0;
static void check(bool ok, const char* what) {
  if (!ok) { std::printf("  FAIL  %s\n", what); ++failures; }
}

// Mirror of the lambda in build_sparsity.
static int neigh(const Grid& g, int il, int jl, int kl, int f) {
  int ni = il, nj = jl, nk = kl;
  if (f == kFaceXlo) ni--; else if (f == kFaceXhi) ni++;
  else if (f == kFaceYlo) nj--; else if (f == kFaceYhi) nj++;
  else if (f == kFaceZlo) nk--; else nk++;
  if (ni < 0)      return g.nbr[kFaceXlo] >= 0 ? g.gid_face(kFaceXlo, nj, nk) : -1;
  if (ni >= g.nxl) return g.nbr[kFaceXhi] >= 0 ? g.gid_face(kFaceXhi, nj, nk) : -1;
  if (nj < 0)      return g.nbr[kFaceYlo] >= 0 ? g.gid_face(kFaceYlo, ni, nk) : -1;
  if (nj >= g.nyl) return g.nbr[kFaceYhi] >= 0 ? g.gid_face(kFaceYhi, ni, nk) : -1;
  if (nk < 0)      return g.nbr[kFaceZlo] >= 0 ? g.gid_face(kFaceZlo, ni, nj) : -1;
  if (nk >= g.nzl) return g.nbr[kFaceZhi] >= 0 ? g.gid_face(kFaceZhi, ni, nj) : -1;
  return g.lid(ni, nj, nk);
}

static void run(int nx, int ny, int nz, int nranks) {
  std::vector<Grid> gs;
  for (int r = 0; r < nranks; ++r)
    gs.push_back(Grid::make(nx, ny, nz, 1, 1, 1, r, nranks));
  const Grid& g0 = gs[0];
  std::printf("%3dx%3dx%3d on %2d ranks (%dx%dx%d)", nx, ny, nz, nranks,
              g0.px, g0.py, g0.pz);

  // The serial reference: for every global cell, the set of global rows its
  // pressure row should couple to.
  auto serial_row = [&](int ig, int jg, int kg) {
    return gs[0].global_row(ig, jg, kg);
  };

  std::map<long long, std::set<long long>> got;   // assembled distributed
  long long rows_seen = 0;

  for (int r = 0; r < nranks; ++r) {
    const Grid& g = gs[r];
    for (int kl = 0; kl < g.nzl; ++kl)
      for (int jl = 0; jl < g.nyl; ++jl)
        for (int il = 0; il < g.nxl; ++il) {
          const int c = g.lid(il, jl, kl);
          std::vector<int> members = {c};
          for (int f = 0; f < 6; ++f) {
            const int nc = neigh(g, il, jl, kl, f);
            if (nc >= 0) members.push_back(nc);
          }
          std::sort(members.begin(), members.end());
          // Ghost indices must sort after every interior index, which is what
          // lets the CSR come out in ascending column order.
          for (size_t m = 1; m < members.size(); ++m)
            check(members[m] > members[m - 1], "duplicate column in a row");
          bool seen_ghost = false;
          for (int mc : members) {
            if (mc >= g.n_local()) seen_ghost = true;
            else check(!seen_ghost, "interior index sorted after a ghost");
          }
          const long long row = g.row0() + c;
          ++rows_seen;
          for (int mc : members) got[row].insert(g.global_cell(mc));
        }
  }
  check(rows_seen == (long long)nx * ny * nz, "not every row assembled");

  // Reference: build the same coupling from global coordinates alone.
  long long mismatches = 0;
  for (int kg = 0; kg < nz; ++kg)
    for (int jg = 0; jg < ny; ++jg)
      for (int ig = 0; ig < nx; ++ig) {
        std::set<long long> want{serial_row(ig, jg, kg)};
        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};
        for (int f = 0; f < 6; ++f) {
          const int a = ig + di[f], b = jg + dj[f], c = kg + dk[f];
          if (a < 0 || a >= nx || b < 0 || b >= ny || c < 0 || c >= nz) continue;
          want.insert(serial_row(a, b, c));
        }
        if (got[serial_row(ig, jg, kg)] != want) ++mismatches;
      }
  check(mismatches == 0, "distributed coupling differs from serial");
  if (mismatches) std::printf("   (%lld rows differ)", mismatches);

  // The matrix must be symmetric in structure: if row A couples to B, then B
  // couples to A.  An asymmetry means a rank boundary swallowed a connection.
  long long asym = 0;
  for (const auto& kv : got)
    for (long long col : kv.second)
      if (!got.count(col) || !got[col].count(kv.first)) ++asym;
  check(asym == 0, "structurally asymmetric: a connection was dropped");

  std::printf("   ok\n");
}

int main() {
  std::printf("=== sparsity under 3-D decomposition ===\n");
  for (int p : {1, 2, 3, 4, 6, 8}) {
    run(20, 16, 10, p);
    run(17, 13, 11, p);
    run(12, 12, 4, p);
  }
  std::printf("\n%s (%d failure%s)\n",
              failures ? "FAILED" : "all checks passed", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
