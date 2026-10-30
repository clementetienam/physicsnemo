// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// nccl_comm.cu -- see nccl_comm.hpp.
#include "nccl_comm.hpp"

#include <mpi.h>

#include <cstdio>

#if defined(NVRS_HAVE_NCCL)
#include <nccl.h>
#endif

namespace nvrs {

#if defined(NVRS_HAVE_NCCL)

namespace {
ncclComm_t g_comm = nullptr;
bool g_tried = false;
bool g_ready = false;

bool ok(ncclResult_t r, const char* what, int rank) {
  if (r == ncclSuccess) return true;
  if (rank == 0)
    std::printf("  [nccl] %s failed: %s\n", what, ncclGetErrorString(r));
  return false;
}
}  // namespace

bool nccl_built() { return true; }
bool nccl_ready() { return g_ready; }

bool nccl_init(const Grid& g) {
  if (g_tried) return g_ready;
  g_tried = true;
  if (g.nranks <= 1) return false;
  ncclUniqueId id;
  int good = 1;
  if (g.rank == 0 && !ok(ncclGetUniqueId(&id), "ncclGetUniqueId", g.rank)) good = 0;
  MPI_Bcast(&good, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!good) return false;
  MPI_Bcast(&id, (int)sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  int mine = ok(ncclCommInitRank(&g_comm, g.nranks, id, g.rank),
                "ncclCommInitRank", g.rank) ? 1 : 0;
  int all = 0;
  MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all) {
    if (mine && g_comm) ncclCommDestroy(g_comm);
    g_comm = nullptr;
    return false;
  }
  g_ready = true;
  if (g.rank == 0) {
    int v = 0;
    ncclGetVersion(&v);
    std::printf("  [nccl] communicator over %d ranks (NCCL %d): the Krylov tail's "
                "halo exchanges and reductions are stream-ordered\n",
                g.nranks, v);
  }
  return true;
}

void nccl_finalize() {
  if (g_comm) ncclCommDestroy(g_comm);
  g_comm = nullptr;
  g_ready = false;
}

void nccl_allreduce_sum(Real* d, int count, cudaStream_t st) {
  if (count <= 0) return;
  ncclAllReduce(d, d, (size_t)count, ncclDouble, ncclSum, g_comm, st);
}

void nccl_group_start() { ncclGroupStart(); }
void nccl_group_end() { ncclGroupEnd(); }

void nccl_send(const Real* buf, size_t count, int peer, cudaStream_t st) {
  if (count) ncclSend(buf, count, ncclDouble, peer, g_comm, st);
}
void nccl_recv(Real* buf, size_t count, int peer, cudaStream_t st) {
  if (count) ncclRecv(buf, count, ncclDouble, peer, g_comm, st);
}

#else   // built without NCCL: multi-rank keeps the MPI path

bool nccl_built() { return false; }
bool nccl_ready() { return false; }
bool nccl_init(const Grid&) { return false; }
void nccl_finalize() {}
void nccl_allreduce_sum(Real*, int, cudaStream_t) {}
void nccl_group_start() {}
void nccl_group_end() {}
void nccl_send(const Real*, size_t, int, cudaStream_t) {}
void nccl_recv(Real*, size_t, int, cudaStream_t) {}

#endif

}  // namespace nvrs
