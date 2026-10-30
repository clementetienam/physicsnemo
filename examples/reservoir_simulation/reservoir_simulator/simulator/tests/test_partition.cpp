// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_partition.cpp -- distributing an unstructured grid.
//
// The partition is computed independently on every rank and the communication
// lists are never exchanged, so the whole scheme rests on one claim: that every
// rank, given the same grid, derives lists that agree. That is exactly what can
// be checked without MPI -- build every rank's partition in one process and
// compare them against each other.
//
// What must hold:
//
//   * every active cell is owned by exactly one rank;
//   * the union of the owned sets is the whole grid, and the global row
//     numbering is a permutation of [0, n_active);
//   * my send list to q equals q's receive list from me, entry for entry, in
//     the same order -- this is the property that makes the exchange correct,
//     and the one that a negotiation round would normally establish;
//   * every connection is assembled by at least one rank, and a connection
//     between two owned cells by exactly one;
//   * every cell a rank's connections reach is either owned by it or present
//     as a ghost.
//
//   g++ -O2 -std=c++17 -I../include test_partition.cpp ../src/field_partition.cpp
//       ../src/cpgrid.cpp -o t && ./t [deck.DATA]

#include "field_partition.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
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

// A box with a fault throw, so the connection graph is not a plain 7-point
// stencil and the partition has genuinely irregular cuts to deal with.
std::string write_faulted_box(int nx, int ny, int nz) {
  const std::string p = tmpdir() + "/nvrs_part.grdecl";
  std::ofstream f(p);
  const double dx = 50, dy = 50, dz = 10, top = 2000;
  f << "SPECGRID\n " << nx << " " << ny << " " << nz << " 1 F /\n\nCOORD\n";
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i)
      f << " " << i * dx << " " << j * dy << " " << (top - 500) << " "
        << i * dx << " " << j * dy << " " << (top + nz * dz + 500) << "\n";
  f << "/\n\nZCORN\n";
  // Displace the half of the grid beyond the middle I column by half a layer,
  // which throws every lateral face there out of alignment.
  const int fault_i = nx / 2;
  for (int k = 0; k < nz; ++k)
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < ny; ++j)
        for (int d2 = 0; d2 < 2; ++d2)
          for (int i = 0; i < nx; ++i)
            for (int d = 0; d < 2; ++d) {
              (void)d;
              const double s = (i > fault_i) ? 0.5 * dz : 0.0;
              f << " " << (top + s + (k + h) * dz);
            }
  f << "\n/\n\n";
  const int ng = nx * ny * nz;
  f << "PORO\n " << ng << "*0.22 /\n\nPERMX\n " << ng << "*250 /\n\n";
  f << "PERMY\n " << ng << "*250 /\n\nPERMZ\n " << ng << "*25 /\n\n";
  f << "NTG\n " << ng << "*1 /\n\n";
  // Punch a hole, so the active set is not a clean box.
  f << "ACTNUM\n";
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        f << " " << ((i > nx / 3 && i < nx / 3 + 2 && j > ny / 3 &&
                      j < ny / 3 + 2) ? 0 : 1);
  f << "\n/\n";
  return p;
}

void check_partition_set(const CornerPointGrid& g, int nranks) {
  std::printf("\n-- %d ranks --\n", nranks);

  std::vector<FieldPartition> parts;
  parts.reserve(nranks);
  for (int r = 0; r < nranks; ++r) parts.push_back(partition_field(g, r, nranks));

  const int na = g.n_active();

  // Every cell owned exactly once, and the owner maps agree across ranks.
  {
    std::vector<int> owned(na, 0);
    bool owners_agree = true;
    for (int r = 0; r < nranks; ++r) {
      for (int a = 0; a < na; ++a) {
        if (parts[r].owner_of_active[a] != parts[0].owner_of_active[a])
          owners_agree = false;
      }
      for (int l = 0; l < parts[r].n_owned; ++l)
        ++owned[parts[r].active_of_local[l]];
    }
    check(owners_agree, "every rank agrees on who owns what");
    bool once = true;
    for (int a = 0; a < na; ++a) if (owned[a] != 1) once = false;
    check(once, "every active cell is owned by exactly one rank");
  }

  // Global rows form a permutation of [0, na).
  {
    long long total = 0;
    for (int r = 0; r < nranks; ++r) total += parts[r].n_owned;
    check(total == na, "the owned counts sum to the active count");
    std::vector<long long> starts;
    for (int r = 0; r < nranks; ++r) starts.push_back(parts[r].row0);
    bool ordered = std::is_sorted(starts.begin(), starts.end());
    check(ordered, "row offsets increase with rank");
    bool contiguous = true;
    for (int r = 0; r + 1 < nranks; ++r)
      if (parts[r].row0 + parts[r].n_owned != parts[r + 1].row0)
        contiguous = false;
    check(contiguous, "each rank's global rows are contiguous and abut");
    check(parts[nranks - 1].row0 + parts[nranks - 1].n_owned == na,
          "the last rank ends at the global count");
  }

  // The claim the whole design rests on: send and receive lists match without
  // ever having been exchanged.
  {
    bool matched = true;
    int pairs = 0, entries = 0;
    for (int r = 0; r < nranks; ++r)
      for (const HaloPeer& p : parts[r].peers) {
        const int q = p.rank;
        // What q says it receives from r.
        const HaloPeer* back = nullptr;
        for (const HaloPeer& o : parts[q].peers)
          if (o.rank == r) { back = &o; break; }
        if (!back) { matched = false; continue; }
        ++pairs;
        if (p.send.size() != back->recv.size()) { matched = false; continue; }
        for (size_t i = 0; i < p.send.size(); ++i) {
          // Compare by ACTIVE index: the local numbering differs per rank, so
          // the lists agree on which cells, not on which slots.
          const int sent = parts[r].active_of_local[p.send[i]];
          const int recvd = parts[q].active_of_local[back->recv[i]];
          if (sent != recvd) matched = false;
          ++entries;
        }
      }
    check(matched, "every send list matches the peer's receive list, in order");
    std::printf("  note  %d directed peer pairs, %d cell entries\n", pairs, entries);
  }

  // Every connection is assembled, and an interior one only once.
  {
    std::map<std::pair<int, int>, int> seen;
    for (int r = 0; r < nranks; ++r)
      for (const Connection& c : parts[r].conns) {
        const int ga = parts[r].active_of_local[c.a];
        const int gb = parts[r].active_of_local[c.b];
        ++seen[{std::min(ga, gb), std::max(ga, gb)}];
      }
    check(seen.size() == g.conns.size(), "every connection is assembled somewhere");
    int interior_once = 0, cut_twice = 0, wrong = 0;
    for (const auto& kv : seen) {
      const int oa = parts[0].owner_of_active[kv.first.first];
      const int ob = parts[0].owner_of_active[kv.first.second];
      if (oa == ob) { if (kv.second == 1) ++interior_once; else ++wrong; }
      else          { if (kv.second == 2) ++cut_twice;     else ++wrong; }
    }
    check(wrong == 0,
          "interior connections appear once, cut connections once per side");
    std::printf("  note  %d interior, %d cut\n", interior_once, cut_twice);
  }

  // Everything a rank's connections reach is present locally.
  {
    bool reachable = true;
    for (int r = 0; r < nranks; ++r) {
      const int ext = parts[r].n_local_ext();
      for (const Connection& c : parts[r].conns)
        if (c.a < 0 || c.a >= ext || c.b < 0 || c.b >= ext) reachable = false;
    }
    check(reachable, "every connection endpoint is an owned cell or a ghost");
  }

  // Balance and halo overhead, reported rather than asserted: an index-space
  // cut of an irregular active set cannot be perfectly even, and pretending
  // otherwise would just make the test brittle.
  {
    int lo = parts[0].n_owned, hi = parts[0].n_owned, ghosts = 0;
    for (int r = 0; r < nranks; ++r) {
      lo = std::min(lo, parts[r].n_owned);
      hi = std::max(hi, parts[r].n_owned);
      ghosts += parts[r].n_ghost;
    }
    std::printf("  note  owned %d..%d (%.2fx), ghosts %d total (%.1f%%)\n",
                lo, hi, lo ? (double)hi / lo : 0.0, ghosts,
                na ? 100.0 * ghosts / na : 0.0);
    check(lo > 0, "no rank was left with nothing to do");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::printf("NVRS field partition tests\n");
  std::printf("==========================\n");

  try {
    // A synthetic faulted grid first: small enough to reason about, and
    // irregular enough to be a real test.
    const std::string grd = write_faulted_box(20, 16, 8);
    CornerPointGrid g;
    g.read_grdecl(grd);
    g.build();
    std::printf("\nsynthetic: %d active cells, %d connections\n",
                g.n_active(), (int)g.conns.size());
    for (int n : {1, 2, 3, 4, 6, 8}) check_partition_set(g, n);

    // Then the real thing, if a deck was given.
    if (argc > 1) {
      std::ifstream t(argv[1]);
      if (t) {
        CornerPointGrid f;
        f.read_grdecl(argv[1]);
        f.build();
        std::printf("\n\n=== %s ===\n", argv[1]);
        std::printf("%d active cells, %d connections\n", f.n_active(),
                    (int)f.conns.size());
        for (int n : {1, 2, 4, 8}) check_partition_set(f, n);
      } else {
        std::printf("\n(deck %s not found; skipped)\n", argv[1]);
      }
    }
  } catch (const std::exception& e) {
    std::printf("\nEXCEPTION: %s\n", e.what());
    return 2;
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
