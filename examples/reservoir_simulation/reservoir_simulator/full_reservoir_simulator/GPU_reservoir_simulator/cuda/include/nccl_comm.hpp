// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// nccl_comm.hpp -- stream-ordered communication for the Krylov tail.
//
// MPI is called from the host, so every halo exchange and every reduction in
// the Krylov iteration forces the host to wait for the device, call MPI, and
// launch again.  That is also what keeps a multi-rank Arnoldi step out of a
// CUDA graph: a graph can only hold work that is enqueued on a stream.  NCCL's
// all-reduce and point-to-point send/receive are enqueued on a stream and can
// be captured, so with NCCL the whole post-multigrid tail of an Arnoldi step --
// halo exchanges and reductions included -- replays as one graph on every rank.
//
// Only the Krylov tail uses it.  Assembly, the multigrid libraries and every
// host-side reduction keep MPI.  Built only when nccl.h is found
// (NVRS_HAVE_NCCL); otherwise every function below is a stub, nccl_init()
// returns false and multi-rank runs keep the MPI path.
#pragma once

#include <cstddef>

#include "common.hpp"
#include "grid.hpp"

namespace nvrs {

// Compiled with NCCL?
bool nccl_built();
// Communicator up?
bool nccl_ready();
// Create the communicator over MPI_COMM_WORLD.  Collective: every rank must
// call it at the same point, and every rank gets the same answer (false on all
// if any rank failed).  Idempotent.
bool nccl_init(const Grid& g);
void nccl_finalize();

// In place, sum over ranks, on stream st.
void nccl_allreduce_sum(Real* d, int count, cudaStream_t st);

// Point-to-point, grouped: call between nccl_group_start/end so that the sends
// and receives of one exchange are posted together and cannot deadlock.
void nccl_group_start();
void nccl_group_end();
void nccl_send(const Real* buf, size_t count, int peer, cudaStream_t st);
void nccl_recv(Real* buf, size_t count, int peer, cudaStream_t st);

}  // namespace nvrs
