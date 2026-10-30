// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// grid.hpp — structured Cartesian grid with a 3-D MPI domain decomposition that
// scales from a single GPU to multi-GPU nodes to multi-node clusters.
//
// Decomposition strategy
// ----------------------
// The global grid (nx,ny,nz) is cut into a px x py x pz array of rectangular
// blocks, one per MPI rank, with the factorisation chosen to minimise the total
// halo surface.
//
// This used to be a 1-D cut along K, which is simpler but scales badly.  On
// 200 x 200 x 25 across four ranks, K-slabs give each rank 200 x 200 x 6 with
// two 200 x 200 ghost faces: 80,000 ghost cells against 250,000 owned, a 32%
// surface-to-volume ratio.  A 2 x 2 x 1 cut gives each rank 100 x 100 x 25 with
// four 100 x 25 faces: 10,000 ghosts, 4%.  Eight times less halo traffic, and
// the gap widens on every algebraic-multigrid coarse level, because a 1-D cut
// runs out of layers to divide while a 3-D cut keeps shrinking in all three
// directions.
//
// Global row numbering
// --------------------
// HYPRE's IJ interface requires each rank to own a *contiguous* global row
// range [row0, row0+n_local).  Natural ordering g = (k*ny + j)*nx + i has that
// property for K-slabs and loses it for 3-D blocks, so the rows are renumbered:
// rank r owns rows [row0(r), row0(r) + n_local(r)), ordered within the block as
// i fastest, then j, then k.  With ranks ordered (rz*py + ry)*px + rx, the
// offset closes analytically -- no communication and no lookup table:
//
//     row0(rx,ry,rz) = k0*nx*ny  +  j0*nx*nzl  +  i0*nyl*nzl
//
// because the ranks preceding this one in linear order are exactly: every rank
// on a lower K band (a full nx*ny slab per layer), then every rank on a lower J
// band within this K band (a full nx*nzl slab per row), then the ranks to the
// left within this row.  For px = py = 1 this collapses to k0*nx*ny, which is
// the old K-slab numbering, so the two schemes agree wherever both apply.
//
// Halo layout
// -----------
// The TPFA stencil reaches one cell across each face, so a rank stores up to
// six one-cell ghost slabs immediately after its interior cells:
//
//   local index   0 .. cells-1                interior (owned), i fastest
//                 cells + goff(f) ..          ghost slab for face f, when a
//                                             neighbour exists on that face
//
// Interior cells stay contiguous, so `row0` still marks a contiguous global row
// range for HYPRE; the ghosts are a purely local extension used by the on-rank
// CSR columns.  Ghost indices all exceed every interior index, which is what
// lets build_sparsity emit CSR columns in ascending order after a plain sort.
#pragma once

#include "common.hpp"

#include <algorithm>

namespace nvrs {

// Face ordering used everywhere: matches the neighbour loop in build_sparsity.
enum Face { kFaceXlo = 0, kFaceXhi, kFaceYlo, kFaceYhi, kFaceZlo, kFaceZhi };

struct Grid {
  int nx = 1, ny = 1, nz = 1;      // global dimensions
  Real dx = 1, dy = 1, dz = 1;     // cell sizes [ft]

  // --- MPI decomposition (3-D blocks) ---
  int rank = 0, nranks = 1;
  int px = 1, py = 1, pz = 1;      // process grid
  int rx = 0, ry = 0, rz = 0;      // this rank's position in it
  int i0 = 0, j0 = 0, k0 = 0;      // first global index owned, per axis
  int nxl = 1, nyl = 1, nzl = 1;   // owned extent, per axis
  int nbr[6] = {-1, -1, -1, -1, -1, -1};   // neighbour ranks, -1 = domain edge

  // ---- sizes -------------------------------------------------------------
  __host__ __device__ int n_global() const { return nx * ny * nz; }
  __host__ __device__ int n_local() const { return nxl * nyl * nzl; }

  // Cells in the ghost slab for one face; 0 when that face is a domain edge.
  __host__ __device__ int gcount(int f) const {
    if (nbr[f] < 0) return 0;
    if (f == kFaceXlo || f == kFaceXhi) return nyl * nzl;
    if (f == kFaceYlo || f == kFaceYhi) return nxl * nzl;
    return nxl * nyl;
  }
  // Local index of the first cell of face f's ghost slab.
  __host__ __device__ int goff(int f) const {
    int o = n_local();
    for (int q = 0; q < f; ++q) o += gcount(q);
    return o;
  }
  __host__ __device__ int nghost() const {
    int s = 0;
    for (int f = 0; f < 6; ++f) s += gcount(f);
    return s;
  }
  // Total addressable local cells = interior + ghosts.  CSR *columns* run over
  // this range; CSR *rows* only over the interior [0, n_local()).
  __host__ __device__ int n_local_ext() const { return n_local() + nghost(); }

  // ---- local indexing ----------------------------------------------------
  // Local (il,jl,kl) -> local interior linear index, i fastest.
  __host__ __device__ int lid(int il, int jl, int kl) const {
    return (kl * nyl + jl) * nxl + il;
  }
  // Index within face f's ghost slab, from the two in-face local coordinates.
  __host__ __device__ int gid_face(int f, int a, int b) const {
    // (a,b) = (jl,kl) for an X face, (il,kl) for a Y face, (il,jl) for a Z face.
    if (f == kFaceXlo || f == kFaceXhi) return goff(f) + b * nyl + a;
    if (f == kFaceYlo || f == kFaceYhi) return goff(f) + b * nxl + a;
    return goff(f) + b * nxl + a;
  }

  // ---- per-axis block distribution ---------------------------------------
  // Axis n split over p ranks: the first (n % p) ranks get one extra cell.
  __host__ __device__ static int axis_start(int n, int p, int r) {
    int base = n / p, rem = n % p;
    return r * base + (r < rem ? r : rem);
  }
  __host__ __device__ static int axis_count(int n, int p, int r) {
    return n / p + (r < n % p ? 1 : 0);
  }
  // Which rank along this axis owns global index gi.
  __host__ __device__ static int axis_owner(int n, int p, int gi) {
    int base = n / p, rem = n % p;
    if (base == 0) return gi < p ? gi : p - 1;      // more ranks than cells
    int big = rem * (base + 1);
    return (gi < big) ? gi / (base + 1) : rem + (gi - big) / base;
  }

  // ---- global row numbering (see the header comment) ---------------------
  __host__ __device__ long long row0_of(int qx, int qy, int qz) const {
    const long long a0 = axis_start(nx, px, qx);
    const long long b0 = axis_start(ny, py, qy);
    const long long c0 = axis_start(nz, pz, qz);
    const long long cy = axis_count(ny, py, qy);
    const long long cz = axis_count(nz, pz, qz);
    return c0 * (long long)nx * ny + b0 * (long long)nx * cz + a0 * cy * cz;
  }
  __host__ __device__ long long row0() const { return row0_of(rx, ry, rz); }

  // Renumbered global row of a global cell (ig,jg,kg), on whichever rank owns
  // it.  This is what the pressure CSR hands to HYPRE.
  __host__ __device__ long long global_row(int ig, int jg, int kg) const {
    const int qx = axis_owner(nx, px, ig);
    const int qy = axis_owner(ny, py, jg);
    const int qz = axis_owner(nz, pz, kg);
    const int a0 = axis_start(nx, px, qx), ca = axis_count(nx, px, qx);
    const int b0 = axis_start(ny, py, qy), cb = axis_count(ny, py, qy);
    const int c0 = axis_start(nz, pz, qz);
    const long long within =
        ((long long)(kg - c0) * cb + (jg - b0)) * ca + (ig - a0);
    return row0_of(qx, qy, qz) + within;
  }

  // ---- global coordinates of a local cell (interior or ghost) ------------
  __host__ __device__ void global_ijk(int c, int* ig, int* jg, int* kg) const {
    if (c < n_local()) {
      const int il = c % nxl, jl = (c / nxl) % nyl, kl = c / (nxl * nyl);
      *ig = i0 + il; *jg = j0 + jl; *kg = k0 + kl;
      return;
    }
    for (int f = 0; f < 6; ++f) {
      const int cnt = gcount(f);
      if (cnt == 0) continue;
      const int off = goff(f);
      if (c < off || c >= off + cnt) continue;
      const int m = c - off;
      switch (f) {
        case kFaceXlo: *ig = i0 - 1;   *jg = j0 + m % nyl; *kg = k0 + m / nyl; break;
        case kFaceXhi: *ig = i0 + nxl; *jg = j0 + m % nyl; *kg = k0 + m / nyl; break;
        case kFaceYlo: *ig = i0 + m % nxl; *jg = j0 - 1;   *kg = k0 + m / nxl; break;
        case kFaceYhi: *ig = i0 + m % nxl; *jg = j0 + nyl; *kg = k0 + m / nxl; break;
        case kFaceZlo: *ig = i0 + m % nxl; *jg = j0 + m / nxl; *kg = k0 - 1;   break;
        default:       *ig = i0 + m % nxl; *jg = j0 + m / nxl; *kg = k0 + nzl; break;
      }
      return;
    }
    *ig = i0; *jg = j0; *kg = k0;      // unreachable for a valid index
  }
  // Global K index of a local cell: the depth datum needs it for every cell,
  // ghosts included, so gravity is measured from the global top and not the
  // rank's own first layer.
  __host__ __device__ int global_k(int c) const {
    int a, b, k;
    global_ijk(c, &a, &b, &k);
    return k;
  }
  // Renumbered global row of a local cell (interior or ghost).
  __host__ __device__ long long global_cell(int c) const {
    int a, b, k;
    global_ijk(c, &a, &b, &k);
    return global_row(a, b, k);
  }
  // Does this rank own global column (ig,jg)?  A well lives on one areal
  // column, which under a 3-D cut is present on only some ranks.
  __host__ __device__ bool owns_column(int ig, int jg) const {
    return ig >= i0 && ig < i0 + nxl && jg >= j0 && jg < j0 + nyl;
  }

  // ---- construction ------------------------------------------------------
  // Pick px*py*pz = nranks minimising the halo surface, subject to no axis
  // being cut finer than it has cells.
  static void factor(int nx, int ny, int nz, int nranks, int* bx, int* by,
                     int* bz) {
    double best = -1.0;
    *bx = *by = *bz = 1;
    for (int p = 1; p <= nranks; ++p) {
      if (nranks % p || p > nx) continue;
      for (int q = 1; q * p <= nranks; ++q) {
        if (nranks % (p * q) || q > ny) continue;
        const int s = nranks / (p * q);
        if (s > nz) continue;
        // Ghost cells per rank, which is what the exchange actually moves.
        const double cx = (double)nx / p, cy = (double)ny / q, cz = (double)nz / s;
        const double surf = (p > 1 ? 2 * cy * cz : 0) +
                            (q > 1 ? 2 * cx * cz : 0) +
                            (s > 1 ? 2 * cx * cy : 0);
        if (best < 0 || surf < best) { best = surf; *bx = p; *by = q; *bz = s; }
      }
    }
  }

  static Grid make(int nx, int ny, int nz, Real dx, Real dy, Real dz, int rank,
                   int nranks) {
    Grid g;
    g.nx = nx; g.ny = ny; g.nz = nz;
    g.dx = dx; g.dy = dy; g.dz = dz;
    g.rank = rank; g.nranks = nranks;

    // NVRS_DECOMP="px py pz" overrides the automatic choice, so a decomposition
    // can be swept without a rebuild.  Ignored unless the product matches.
    int fx = 1, fy = 1, fz = 1;
    bool forced = false;
    if (const char* e = std::getenv("NVRS_DECOMP")) {
      int a = 0, b = 0, c = 0;
      if (std::sscanf(e, "%d %d %d", &a, &b, &c) == 3 && a > 0 && b > 0 &&
          c > 0 && a * b * c == nranks && a <= nx && b <= ny && c <= nz) {
        fx = a; fy = b; fz = c; forced = true;
      }
    }
    if (!forced) factor(nx, ny, nz, nranks, &fx, &fy, &fz);
    g.px = fx; g.py = fy; g.pz = fz;

    // Rank ordering (rz*py + ry)*px + rx, which is what row0_of assumes.
    g.rx = rank % fx;
    g.ry = (rank / fx) % fy;
    g.rz = rank / (fx * fy);

    g.i0 = axis_start(nx, fx, g.rx); g.nxl = axis_count(nx, fx, g.rx);
    g.j0 = axis_start(ny, fy, g.ry); g.nyl = axis_count(ny, fy, g.ry);
    g.k0 = axis_start(nz, fz, g.rz); g.nzl = axis_count(nz, fz, g.rz);

    auto lin = [&](int a, int b, int c) { return (c * fy + b) * fx + a; };
    g.nbr[kFaceXlo] = (g.rx > 0)      ? lin(g.rx - 1, g.ry, g.rz) : -1;
    g.nbr[kFaceXhi] = (g.rx < fx - 1) ? lin(g.rx + 1, g.ry, g.rz) : -1;
    g.nbr[kFaceYlo] = (g.ry > 0)      ? lin(g.rx, g.ry - 1, g.rz) : -1;
    g.nbr[kFaceYhi] = (g.ry < fy - 1) ? lin(g.rx, g.ry + 1, g.rz) : -1;
    g.nbr[kFaceZlo] = (g.rz > 0)      ? lin(g.rx, g.ry, g.rz - 1) : -1;
    g.nbr[kFaceZhi] = (g.rz < fz - 1) ? lin(g.rx, g.ry, g.rz + 1) : -1;
    return g;
  }
};

}  // namespace nvrs
