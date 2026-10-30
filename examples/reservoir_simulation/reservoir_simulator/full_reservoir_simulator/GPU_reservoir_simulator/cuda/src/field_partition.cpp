// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// field_partition.cpp -- distributing an unstructured grid across ranks.
//
// See field_partition.hpp for why this needs no communication to set up.
#include "field_partition.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>

namespace nvrs {

// ---------------------------------------------------------------------------
void choose_factorisation(int nranks, int nx, int ny, int nz,
                          int& px, int& py, int& pz) {
  px = py = pz = 1;
  if (nranks <= 1) return;

  // Minimise the total area of the cut planes.  Cutting along a direction p
  // times introduces (p-1) planes of the other two extents, so the cost is
  //   (px-1)*ny*nz + (py-1)*nx*nz + (pz-1)*nx*ny
  // which is what a halo actually costs, rather than a proxy like aspect
  // ratio.
  double best = -1;
  for (int a = 1; a <= nranks; ++a) {
    if (nranks % a) continue;
    const int rem = nranks / a;
    for (int b = 1; b <= rem; ++b) {
      if (rem % b) continue;
      const int c = rem / b;
      // A direction cannot be cut more times than it has cells.
      if (a > nx || b > ny || c > nz) continue;
      const double cost = (double)(a - 1) * ny * nz +
                          (double)(b - 1) * nx * nz +
                          (double)(c - 1) * nx * ny;
      if (best < 0 || cost < best) { best = cost; px = a; py = b; pz = c; }
    }
  }
  // A grid too small to cut in three dimensions falls back to cutting the
  // longest direction as far as it will go.
  if (best < 0) { px = std::min(nranks, nx); py = pz = 1; }
}

namespace {

// Start and count of block b when n items are cut into p blocks, with the
// remainder spread over the first blocks rather than piled on the last.
inline void block_range(int n, int p, int b, int& start, int& count) {
  const int base = n / p, rem = n % p;
  count = base + (b < rem ? 1 : 0);
  start = b * base + std::min(b, rem);
}

// Cut an axis into p blocks carrying roughly equal WEIGHT, and return the
// block each coordinate falls in.
//
// Cutting the index space evenly is the obvious thing and it is wrong here.
// Norne has 68,417 inactive cells out of 113,344, and they are not spread
// evenly: an even cut of the indices gave 5,010 active cells to one rank of
// four and 15,934 to another, a factor of 3.2. Eight ranks reached 4.7, at
// which point most of the machine is waiting.
//
// Weighting by active-cell count instead puts the cuts where the reservoir
// actually is. The blocks then differ in index extent, which costs nothing:
// the halo is built from the connection graph, not from slab arithmetic, so
// unequal blocks are no harder to exchange than equal ones.
void weighted_cuts(const std::vector<long long>& w, int p,
                   std::vector<int>& block_of) {
  const int n = (int)w.size();
  block_of.assign(n, 0);
  if (p <= 1 || n == 0) return;

  long long total = 0;
  for (long long x : w) total += x;
  if (total <= 0) {                      // nothing active: fall back to even
    for (int b = 0; b < p; ++b) {
      int s, c; block_range(n, p, b, s, c);
      for (int i = s; i < s + c; ++i) block_of[i] = b;
    }
    return;
  }

  // Walk the axis, closing a block once it holds its share. Every block is
  // given at least one coordinate, so a rank is never left with nothing even
  // when the weight is concentrated in a few slices.
  long long acc = 0;
  int b = 0, placed = 0;
  for (int i = 0; i < n; ++i) {
    block_of[i] = b;
    acc += w[i];
    ++placed;
    const int blocks_left = p - b - 1;
    const int coords_left = n - i - 1;
    const bool must_close = coords_left <= blocks_left;   // reserve one each
    const long long target = total * (b + 1) / p;
    if (b + 1 < p && placed > 0 && (acc >= target || must_close)) {
      ++b;
      placed = 0;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
FieldPartition partition_field(const CornerPointGrid& g, int rank, int nranks) {
  FieldPartition P;
  P.rank = rank;
  P.nranks = nranks;

  const int na = g.n_active();
  const int nx = g.nx, ny = g.ny, nz = g.nz;

  int px, py, pz;
  choose_factorisation(nranks, nx, ny, nz, px, py, pz);

  // ---- ownership ---------------------------------------------------------
  // Cut each axis where the ACTIVE cells are, not where the indices are.  The
  // weight of a coordinate is the number of active cells in its slice.
  //
  // The three axes are cut independently rather than recursively.  A recursive
  // bisection would balance better still, but it makes the block a rank owns
  // depend on the order the cuts were taken, and the whole scheme rests on
  // every rank deriving the same answer from the same data.  Independent cuts
  // are trivially reproducible and recover most of the benefit.
  std::vector<long long> wi(nx, 0), wj(ny, 0), wk(nz, 0);
  for (int a = 0; a < na; ++a) {
    const int gi = g.global_of_active(a);
    ++wi[gi % nx];
    ++wj[(gi / nx) % ny];
    ++wk[gi / (nx * ny)];
  }

  std::vector<int> bi, bj, bk;
  weighted_cuts(wi, px, bi);
  weighted_cuts(wj, py, bj);
  weighted_cuts(wk, pz, bk);

  P.owner_of_active.assign(na, 0);
  std::vector<int> owned_count(nranks, 0);
  for (int a = 0; a < na; ++a) {
    const int gi = g.global_of_active(a);
    const int i = gi % nx, j = (gi / nx) % ny, k = gi / (nx * ny);
    // Rank order (rz*py + ry)*px + rx, matching the Cartesian decomposition,
    // so that rank r's cells form a contiguous run in this ordering.
    const int r = (bk[k] * py + bj[j]) * px + bi[i];
    P.owner_of_active[a] = r;
    ++owned_count[r];
  }

  P.min_owned = *std::min_element(owned_count.begin(), owned_count.end());
  P.max_owned = *std::max_element(owned_count.begin(), owned_count.end());

  // Global row offsets: every rank computes the same prefix sum, so no
  // Exscan is needed and every rank knows every other rank's range.
  std::vector<long long> row0(nranks + 1, 0);
  for (int r = 0; r < nranks; ++r) row0[r + 1] = row0[r] + owned_count[r];
  P.row0 = row0[rank];
  P.n_global = row0[nranks];

  // ---- local numbering ---------------------------------------------------
  // Owned cells first, in ascending active order, which makes this rank's
  // global rows contiguous and ascending.
  P.local_of_active.assign(na, -1);
  P.active_of_local.clear();
  P.active_of_local.reserve(owned_count[rank]);
  for (int a = 0; a < na; ++a)
    if (P.owner_of_active[a] == rank) {
      P.local_of_active[a] = (int)P.active_of_local.size();
      P.active_of_local.push_back(a);
    }
  P.n_owned = (int)P.active_of_local.size();

  // ---- ghosts ------------------------------------------------------------
  // Any cell this rank does not own but a connection reaches.  Collected in
  // ascending active order so the numbering is deterministic.
  std::set<int> ghost_set;
  for (const Connection& c : g.conns) {
    const int oa = P.owner_of_active[c.a], ob = P.owner_of_active[c.b];
    if (oa == rank && ob != rank) ghost_set.insert(c.b);
    else if (ob == rank && oa != rank) ghost_set.insert(c.a);
  }
  for (int a : ghost_set) {
    P.local_of_active[a] = (int)P.active_of_local.size();
    P.active_of_local.push_back(a);
  }
  P.n_ghost = (int)ghost_set.size();

  // ---- connections this rank assembles -----------------------------------
  P.n_cut_conns = 0;
  for (const Connection& c : g.conns) {
    const int oa = P.owner_of_active[c.a], ob = P.owner_of_active[c.b];
    if (oa != rank && ob != rank) continue;
    if (oa != ob) ++P.n_cut_conns;
    Connection lc = c;
    lc.a = P.local_of_active[c.a];
    lc.b = P.local_of_active[c.b];
    P.conns.push_back(lc);
  }

  // ---- communication lists ------------------------------------------------
  // For each other rank q: the owned cells q needs from me, and the ghosts I
  // need from q.  Both are derived from the same global connection list, and
  // both are sorted by the active index of the cell being sent -- a rule that
  // does not depend on which rank is evaluating it.  So my send list to q is
  // exactly q's receive list from me, entry for entry, with no exchange.
  std::map<int, std::set<int>> to_send;   // q -> active indices I send
  std::map<int, std::set<int>> to_recv;   // q -> active indices I receive
  for (const Connection& c : g.conns) {
    const int oa = P.owner_of_active[c.a], ob = P.owner_of_active[c.b];
    if (oa == ob) continue;
    if (oa == rank) { to_send[ob].insert(c.a); to_recv[ob].insert(c.b); }
    else if (ob == rank) { to_send[oa].insert(c.b); to_recv[oa].insert(c.a); }
  }

  for (const auto& kv : to_send) {
    HaloPeer p;
    p.rank = kv.first;
    for (int a : kv.second) p.send.push_back(P.local_of_active[a]);
    auto it = to_recv.find(kv.first);
    if (it != to_recv.end())
      for (int a : it->second) p.recv.push_back(P.local_of_active[a]);
    P.peers.push_back(std::move(p));
  }
  // A peer that only sends to me would be missed by the loop above; in a
  // symmetric connection graph that cannot happen, but the graph is built from
  // a deck and is not worth assuming symmetric.
  for (const auto& kv : to_recv) {
    if (to_send.count(kv.first)) continue;
    HaloPeer p;
    p.rank = kv.first;
    for (int a : kv.second) p.recv.push_back(P.local_of_active[a]);
    P.peers.push_back(std::move(p));
  }
  std::sort(P.peers.begin(), P.peers.end(),
            [](const HaloPeer& a, const HaloPeer& b) { return a.rank < b.rank; });

  return P;
}

// ---------------------------------------------------------------------------
std::string FieldPartition::summary() const {
  std::ostringstream os;
  char b[256];
  os << "Partition\n";
  std::snprintf(b, sizeof b, "  ranks           : %d  (this is %d)\n",
                nranks, rank); os << b;
  std::snprintf(b, sizeof b, "  owned           : %d cells, ghosts %d "
                "(%.1f%% overhead)\n", n_owned, n_ghost,
                n_owned ? 100.0 * n_ghost / n_owned : 0.0); os << b;
  std::snprintf(b, sizeof b, "  global rows     : %lld, this rank starts at %lld\n",
                n_global, row0); os << b;
  std::snprintf(b, sizeof b, "  balance         : %d .. %d cells per rank "
                "(%.2fx)\n", min_owned, max_owned,
                min_owned ? (double)max_owned / min_owned : 0.0); os << b;
  std::snprintf(b, sizeof b, "  connections     : %d assembled here, %d cross a "
                "rank boundary\n", (int)conns.size(), n_cut_conns); os << b;
  std::snprintf(b, sizeof b, "  peers           : %d", (int)peers.size());
  os << b;
  if (!peers.empty()) {
    os << "  (";
    for (size_t i = 0; i < peers.size(); ++i) {
      if (i) os << ", ";
      std::snprintf(b, sizeof b, "%d: %d/%d", peers[i].rank,
                    (int)peers[i].send.size(), (int)peers[i].recv.size());
      os << b;
    }
    os << " send/recv)";
  }
  os << "\n";
  return os.str();
}

}  // namespace nvrs
