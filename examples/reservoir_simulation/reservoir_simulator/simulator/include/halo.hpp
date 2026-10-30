// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// halo.hpp — the one-layer K halo exchange that makes the MPI domain
// decomposition physically correct.
//
// Why this exists
// ---------------
// The TPFA flux stencil couples a cell to its six face neighbours.  Under the
// K-slab decomposition (grid.hpp) the two K faces on a slab boundary point at
// cells owned by another rank.  Without an exchange those faces have to be
// treated as no-flow, which silently turns an N-rank run into N *independent*
// reservoirs stacked on top of each other — the fields and well rates then
// depend on the rank count, which is exactly the bug this module removes.
//
// Contract
// --------
// A "cell-blocked" device vector v has `bs` consecutive components per cell and
// is laid out over the extended local index space of grid.hpp:
//
//     v[bs*c + a],  c in [0, n_local())            interior, owned by this rank
//     v[bs*c + a],  c in [n_local(), n_local_ext()) ghost, owned by a neighbour
//
// `halo_exchange` overwrites *only* the ghost part, copying the neighbouring
// ranks' boundary layers into it.  Call it before any operation that reads
// neighbour values: residual/Jacobian assembly, and every distributed SpMV.
//
// Transport
// ---------
// By default the layers are staged through pinned host buffers, which works
// with any MPI.  Export NVRS_CUDA_AWARE_MPI=1 to pass device pointers straight
// to MPI (UCX/HPC-X, MPICH+GDRCopy, …) and skip the staging copies.
#pragma once

#include "common.hpp"
#include "grid.hpp"

namespace nvrs {

// Allocate the staging buffers.  `bs_max` is the largest block size that will
// ever be exchanged (block size for state vectors, 1 for scalar rock fields).
// Safe to call more than once; buffers only ever grow.
void halo_init(const Grid& g, int bs_max);

// Release the staging buffers.
void halo_finalize();

// Decide once, collectively, whether MPI can be handed device pointers.
// MUST be called by every rank at a point all of them reach: the decision is
// made by actually exchanging bytes, which is itself collective.
void halo_probe(const Grid& g);

// Fill the ghost region of a cell-blocked device vector.  No-op for a single
// rank or for a rank with no K neighbours.
void halo_exchange(Real* v, int bs, const Grid& g);

// The same exchange enqueued on a stream through NCCL (nccl_comm.hpp): packs,
// sends and receives with no host synchronisation, so it can be recorded in a
// CUDA graph.  Requires nccl_ready() and a prior halo_init(g, bs).
void halo_exchange_nccl(Real* v, int bs, const Grid& g, cudaStream_t st);

// What the halo exchange actually cost.  Multi-rank runs have been slower than
// single-rank ones on this code, and wall clock alone cannot say whether that
// is the exchange, the Krylov reductions or BoomerAMG's own communication.
// These are host-side timings around synchronous MPI calls, so std::chrono is
// the right instrument and no device synchronisation is added.
struct HaloStats {
  double seconds = 0.0;    // wall time inside halo_exchange
  double stage_seconds = 0.0;  // of which cudaMemcpy staging (host path only)
  long   calls = 0;
  double bytes = 0.0;      // moved in one direction, summed over calls
};
const HaloStats& halo_stats();
void halo_reset_stats();

// Record one exchange performed elsewhere.
//
// The connection-list halo (halo_conn.cu) is a different implementation in a
// different translation unit, but it is the same cost in the same run.  Until
// it reported here, a field run's summary said
//
//     Halo : 0 exchanges, 0.00 s (0.0% of wall), 0.00 GB moved
//
// on exactly the runs the counters exist to describe -- and its cost was
// folded invisibly into the operator+halo line instead, which is the one
// number that could have said how much of a multi-GPU slowdown is
// communication.  `bytes` is one direction, matching halo_exchange.
void halo_note_exchange(double seconds, double stage_seconds, double bytes);

// True when the device-pointer (CUDA-aware MPI) path is active.
bool halo_cuda_aware();

}  // namespace nvrs
