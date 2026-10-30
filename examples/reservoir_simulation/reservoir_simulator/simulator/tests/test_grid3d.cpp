// Host-only checks on the 3-D decomposition in grid.hpp.
//
// The renumbering is the part that can silently corrupt everything: HYPRE needs
// each rank to own a contiguous global row range, every cell must be owned by
// exactly one rank, and every rank's view of a shared cell's global row must
// agree.  None of that is visible in a simulation result until the answer is
// quietly wrong, so check it directly.
#include "grid.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

using nvrs::Grid;

static int failures = 0;
static void check(bool ok, const char* what, const char* detail = "") {
  if (!ok) {
    std::printf("  FAIL  %s %s\n", what, detail);
    ++failures;
  }
}

static void run(int nx, int ny, int nz, int nranks) {
  std::vector<Grid> gs;
  for (int r = 0; r < nranks; ++r)
    gs.push_back(Grid::make(nx, ny, nz, 1, 1, 1, r, nranks));

  const Grid& g0 = gs[0];
  std::printf("%3dx%3dx%3d on %2d ranks -> %d x %d x %d", nx, ny, nz, nranks,
              g0.px, g0.py, g0.pz);

  // 1. every cell owned exactly once, and the owner agrees it owns it
  std::vector<int> owner((size_t)nx * ny * nz, -1);
  long long total = 0;
  for (int r = 0; r < nranks; ++r) {
    const Grid& g = gs[r];
    total += g.n_local();
    for (int kl = 0; kl < g.nzl; ++kl)
      for (int jl = 0; jl < g.nyl; ++jl)
        for (int il = 0; il < g.nxl; ++il) {
          const int ig = g.i0 + il, jg = g.j0 + jl, kg = g.k0 + kl;
          const size_t nat = ((size_t)kg * ny + jg) * nx + ig;
          check(owner[nat] < 0, "cell owned twice");
          owner[nat] = r;
        }
  }
  check(total == (long long)nx * ny * nz, "cell count mismatch");
  for (size_t c = 0; c < owner.size(); ++c) check(owner[c] >= 0, "cell unowned");

  // 2. rows are a contiguous partition of [0, N): every row hit exactly once
  std::vector<char> seen((size_t)nx * ny * nz, 0);
  for (int r = 0; r < nranks; ++r) {
    const Grid& g = gs[r];
    const long long lo = g.row0();
    for (long long q = lo; q < lo + g.n_local(); ++q) {
      check(q >= 0 && q < (long long)seen.size(), "row out of range");
      check(!seen[(size_t)q], "row claimed by two ranks");
      seen[(size_t)q] = 1;
    }
  }
  for (size_t q = 0; q < seen.size(); ++q) check(seen[q], "row unclaimed");

  // 3. global_row agrees with the owner's own local numbering, and every rank
  //    computes the same value for the same cell
  for (int kg = 0; kg < nz; ++kg)
    for (int jg = 0; jg < ny; ++jg)
      for (int ig = 0; ig < nx; ++ig) {
        const long long ref = gs[0].global_row(ig, jg, kg);
        for (int r = 1; r < nranks; ++r)
          check(gs[r].global_row(ig, jg, kg) == ref, "ranks disagree on row");
        const int o = owner[((size_t)kg * ny + jg) * nx + ig];
        const Grid& g = gs[o];
        const long long expect =
            g.row0() + g.lid(ig - g.i0, jg - g.j0, kg - g.k0);
        check(ref == expect, "row != owner row0 + local index");
      }

  // 4. ghost slabs: each ghost cell maps to a real global cell owned by the
  //    neighbour on that face, and ghost indices exceed every interior index
  for (int r = 0; r < nranks; ++r) {
    const Grid& g = gs[r];
    std::set<int> ghosts;
    for (int f = 0; f < 6; ++f) {
      const int cnt = g.gcount(f), off = g.goff(f);
      if (!cnt) continue;
      check(off >= g.n_local(), "ghost overlaps interior");
      for (int m = 0; m < cnt; ++m) {
        const int c = off + m;
        check(ghosts.insert(c).second, "ghost index reused across faces");
        int ig, jg, kg;
        g.global_ijk(c, &ig, &jg, &kg);
        check(ig >= 0 && ig < nx && jg >= 0 && jg < ny && kg >= 0 && kg < nz,
              "ghost outside the domain");
        const int o = owner[((size_t)kg * ny + jg) * nx + ig];
        check(o == g.nbr[f], "ghost not owned by the face neighbour");
      }
    }
    check((int)ghosts.size() == g.nghost(), "nghost disagrees with the slabs");
  }
  std::printf("   ok\n");
}

int main() {
  std::printf("=== 3-D decomposition ===\n");
  for (int p : {1, 2, 3, 4, 6, 8}) {
    run(200, 200, 25, p);
    run(46, 112, 22, p);
    run(40, 40, 3, p);
    run(17, 13, 11, p);          // all prime, nothing divides evenly
  }

  // The old code cut only along K.  Forcing that factorisation must reproduce
  // the old numbering exactly, or results from before this change stop being
  // comparable with results from after it.
  std::printf("\n=== K-slab compatibility (NVRS_DECOMP=1 1 P) ===\n");
  for (int p : {1, 2, 4}) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "1 1 %d", p);
#if defined(_WIN32)
    _putenv_s("NVRS_DECOMP", buf);
#else
    setenv("NVRS_DECOMP", buf, 1);
#endif
    for (int r = 0; r < p; ++r) {
      Grid g = Grid::make(200, 200, 25, 1, 1, 1, r, p);
      check(g.px == 1 && g.py == 1 && g.pz == p, "override ignored");
      check(g.row0() == (long long)g.k0 * 200 * 200,
            "K-slab row0 differs from the old formula");
      check(g.nxl == 200 && g.nyl == 200, "K-slab should own full i,j");
    }
    std::printf("  %d rank(s): row0 == k0*nx*ny   ok\n", p);
  }
#if defined(_WIN32)
  _putenv_s("NVRS_DECOMP", "");
#else
  unsetenv("NVRS_DECOMP");
#endif

  // What the change is for.
  std::printf("\n=== halo surface, 200x200x25 ===\n");
  std::printf("  ranks   layout    owned/rank   ghosts/rank   surface\n");
  for (int p : {1, 2, 4, 8}) {
    Grid g = Grid::make(200, 200, 25, 1, 1, 1, 0, p);
    Grid mid = Grid::make(200, 200, 25, 1, 1, 1, p / 2, p);
    std::printf("  %5d   %dx%dx%d      %8d      %8d      %5.1f%%\n", p, g.px,
                g.py, g.pz, mid.n_local(), mid.nghost(),
                100.0 * mid.nghost() / mid.n_local());
  }

  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all checks passed",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
