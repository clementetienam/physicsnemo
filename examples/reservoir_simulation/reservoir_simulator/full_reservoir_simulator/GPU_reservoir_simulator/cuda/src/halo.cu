// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// halo.cu — six-face ghost exchange for the 3-D decomposition (see halo.hpp).
//
// One Sendrecv per face was replaced by a single non-blocking round over all
// six: the faces are independent, and posting them together costs one latency
// instead of six.  Each face's ghost slab is contiguous in the local vector, so
// MPI can receive straight into it; only the *send* side needs a gather, since
// the interior boundary layer of an X or Y face is strided.
#include "halo.hpp"
#include "nccl_comm.hpp"

// mpi.h FIRST, always.  mpi-ext.h is not self-contained: Open MPI's copy uses
// OMPI_DECLSPEC, MPI_Comm and MPI_Request without declaring them, so including
// it first produces a wall of "expected a ;" from every prototype in every
// extension header.  It is also not part of the MPI standard and is missing on
// some builds, hence the __has_include guard.
//
// NVRS_NO_MPI_EXT=1 skips it entirely, for a toolchain whose copy will not
// parse.  The cost is only that CUDA-aware MPI stops being auto-detected;
// NVRS_CUDA_AWARE_MPI=1 still turns it on explicitly.
#include <mpi.h>

#if !defined(NVRS_NO_MPI_EXT) && defined(__has_include)
#  if __has_include(<mpi-ext.h>)
#    include <mpi-ext.h>
#    define NVRS_HAVE_MPI_EXT 1
#  endif
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace nvrs {
namespace {

struct HaloCtx {
  size_t cap = 0;             // capacity in Real elements of each staging area
  Real* d_send = nullptr;     // device gather buffer, all faces concatenated
  Real* h_send = nullptr;     // pinned host mirrors, used only when MPI cannot
  Real* h_recv = nullptr;     // take device pointers
  bool cuda_aware = false;
  bool probed = false;
};

HaloCtx g_halo;
HaloStats g_stats;

inline double wall_now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// How long to let the device-pointer probe run before declaring the transport
// unusable.  Generous: it happens once, at start-up.
constexpr double kProbeTimeout = 20.0;   // seconds

// One tag per face.  A send on face f is matched by the neighbour's receive on
// the opposite face f^1, so the sender uses tag f and the receiver tag f^1.
constexpr int kTagBase = 7100;

void free_buffers() {
  if (g_halo.d_send) cudaFree(g_halo.d_send);
  if (g_halo.h_send) cudaFreeHost(g_halo.h_send);
  if (g_halo.h_recv) cudaFreeHost(g_halo.h_recv);
  g_halo.d_send = nullptr;
  g_halo.h_send = g_halo.h_recv = nullptr;
  g_halo.cap = 0;
}

// Gather the interior boundary layer of one face into a contiguous buffer.
// The in-face index matches Grid::gid_face, so the neighbour's ghost slab is
// filled in the order it expects with no unpack step.
__global__ void k_pack_face(const Real* __restrict__ v, Real* __restrict__ out,
                            int face, int nxl, int nyl, int nzl, int bs,
                            int count) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= count) return;
  int il, jl, kl;
  switch (face) {
    case kFaceXlo: il = 0;       jl = m % nyl; kl = m / nyl; break;
    case kFaceXhi: il = nxl - 1; jl = m % nyl; kl = m / nyl; break;
    case kFaceYlo: il = m % nxl; jl = 0;       kl = m / nxl; break;
    case kFaceYhi: il = m % nxl; jl = nyl - 1; kl = m / nxl; break;
    case kFaceZlo: il = m % nxl; jl = m / nxl; kl = 0;       break;
    default:       il = m % nxl; jl = m / nxl; kl = nzl - 1; break;
  }
  const long long src = (long long)bs * ((kl * (long long)nyl + jl) * nxl + il);
  Real* dst = out + (long long)bs * m;
  for (int a = 0; a < bs; ++a) dst[a] = v[src + a];
}

}  // namespace

bool halo_cuda_aware() { return g_halo.cuda_aware; }

const HaloStats& halo_stats() { return g_stats; }
void halo_reset_stats() { g_stats = HaloStats(); }

void halo_note_exchange(double seconds, double stage_seconds, double bytes) {
  g_stats.seconds += seconds;
  g_stats.stage_seconds += stage_seconds;
  g_stats.bytes += bytes;
  ++g_stats.calls;
}

namespace {

// Move a few bytes between device buffers with the calls this file depends on,
// and believe the transport only if the bytes arrive.
//
// MPIX_Query_cuda_support() reports what the library was *built* to support,
// which is not the same as what works: a build can advertise CUDA support and
// still hang or fault on a device pointer if the transport was never
// initialised for it.  Trusting the query alone turned a slow multi-rank run
// into one that made no progress at all.
bool probe_device_mpi(const Grid& g) {
  const int rank = g.rank, nranks = g.nranks;
  bool ok = true;
  double *sbuf = nullptr, *rbuf = nullptr;
  if (cudaMalloc(&sbuf, sizeof(double) * 2) != cudaSuccess) return false;
  if (cudaMalloc(&rbuf, sizeof(double) * 2) != cudaSuccess) {
    cudaFree(sbuf);
    return false;
  }
  const double seed[2] = {(double)rank + 1.0, 1.0};
  double got[2] = {0.0, 0.0};
  if (cudaMemcpy(sbuf, seed, sizeof(seed), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    ok = false;
  if (cudaMemset(rbuf, 0, sizeof(double) * 2) != cudaSuccess) ok = false;

  if (ok) {
    // Ring exchange, non-blocking with a deadline.  The failure this probe
    // exists to catch is an MPI that accepts a device pointer and then never
    // completes; a blocking call would simply hang here instead of later.
    const int up = (rank + 1) % nranks;
    const int down = (rank + nranks - 1) % nranks;
    if (cudaDeviceSynchronize() != cudaSuccess) ok = false;
    MPI_Request req[2];
    if (ok &&
        (MPI_Irecv(rbuf, 2, MPI_DOUBLE, down, 9101, MPI_COMM_WORLD, &req[0]) !=
             MPI_SUCCESS ||
         MPI_Isend(sbuf, 2, MPI_DOUBLE, up, 9101, MPI_COMM_WORLD, &req[1]) !=
             MPI_SUCCESS))
      ok = false;
    if (ok) {
      const double deadline = wall_now() + kProbeTimeout;
      int done = 0;
      while (!done && wall_now() < deadline)
        MPI_Testall(2, req, &done, MPI_STATUSES_IGNORE);
      if (!done) {
        for (int i = 0; i < 2; ++i) {
          MPI_Cancel(&req[i]);
          MPI_Request_free(&req[i]);
        }
        ok = false;
      }
    }
    if (ok && cudaMemcpy(got, rbuf, sizeof(got), cudaMemcpyDeviceToHost) !=
                  cudaSuccess)
      ok = false;
    if (ok && got[0] != (double)down + 1.0) ok = false;
  }

  if (ok) {
    // Distinct send and receive buffers: MPI_IN_PLACE on device memory is the
    // least reliably implemented corner of CUDA-aware MPI, so nothing here
    // depends on it.
    if (cudaDeviceSynchronize() != cudaSuccess) ok = false;
    if (MPI_Allreduce(sbuf, rbuf, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) !=
        MPI_SUCCESS)
      ok = false;
    if (ok && cudaMemcpy(got, rbuf, sizeof(got), cudaMemcpyDeviceToHost) !=
                  cudaSuccess)
      ok = false;
    if (ok && got[1] != (double)nranks) ok = false;
  }

  cudaFree(sbuf);
  cudaFree(rbuf);
  cudaGetLastError();          // clear anything the failure latched

  int mine = ok ? 1 : 0, all = 0;
  MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return all == 1;
}

bool detect_cuda_aware_mpi(const Grid& g) {
  if (const char* e = std::getenv("NVRS_CUDA_AWARE_MPI"))
    return e[0] == '1';
  if (g.nranks <= 1) return false;
#if defined(NVRS_HAVE_MPI_EXT) && defined(MPIX_CUDA_AWARE_SUPPORT) && \
    MPIX_CUDA_AWARE_SUPPORT
  if (MPIX_Query_cuda_support() != 1) return false;
  return probe_device_mpi(g);
#else
  return false;
#endif
}

}  // namespace

void halo_probe(const Grid& g) {
  if (g_halo.probed) return;
  g_halo.probed = true;               // set first: the probe must run once
  g_halo.cuda_aware = detect_cuda_aware_mpi(g);
  if (g.nranks > 1 && g.rank == 0) {
    std::printf("  Decomp   : %d x %d x %d ranks, %d x %d x %d cells each, "
                "%d ghost (%.1f%%)\n",
                g.px, g.py, g.pz, g.nxl, g.nyl, g.nzl, g.nghost(),
                100.0 * g.nghost() / (g.n_local() > 0 ? g.n_local() : 1));
    std::printf("  Halo     : %s\n",
                g_halo.cuda_aware
                    ? "device pointers passed straight to MPI (verified)"
                    : "staged through pinned host memory");
    std::fflush(stdout);
  }
}

void halo_init(const Grid& g, int bs_max) {
  // The probe is collective, and halo_exchange returns early on a rank with no
  // neighbours -- probing lazily from there would leave those ranks outside the
  // collective.  main.cu calls halo_probe() where every rank arrives; this is
  // only a backstop.
  if (!g_halo.probed) halo_probe(g);
  if (g.nranks <= 1) return;
  const size_t need = (size_t)bs_max * (size_t)g.nghost();
  if (need == 0 || need <= g_halo.cap) return;
  free_buffers();
  NVRS_CHECK_CUDA(cudaMalloc(&g_halo.d_send, sizeof(Real) * need));
  if (!g_halo.cuda_aware) {
    NVRS_CHECK_CUDA(cudaMallocHost(&g_halo.h_send, sizeof(Real) * need));
    NVRS_CHECK_CUDA(cudaMallocHost(&g_halo.h_recv, sizeof(Real) * need));
  }
  g_halo.cap = need;
}

void halo_finalize() { free_buffers(); }

void halo_exchange_nccl(Real* v, int bs, const Grid& g, cudaStream_t st) {
  if (g.nranks <= 1 || g.n_local() <= 0) return;
  const int nghost = g.nghost();
  if (nghost == 0) return;
  // The send buffer mirrors the ghost layout, exactly as in halo_exchange.
  for (int f = 0; f < 6; ++f) {
    const int cnt = g.gcount(f);
    if (cnt == 0) continue;
    const int off = g.goff(f) - g.n_local();
    const int threads = 256, blocks = (cnt + threads - 1) / threads;
    k_pack_face<<<blocks, threads, 0, st>>>(v, g_halo.d_send + (size_t)bs * off,
                                            f, g.nxl, g.nyl, g.nzl, bs, cnt);
  }
  // Receive straight into the ghost region; a face's neighbour sends from its
  // opposite face.  One peer per face, so sends and receives pair uniquely.
  Real* rbuf = v + (size_t)bs * g.n_local();
  nccl_group_start();
  for (int f = 0; f < 6; ++f) {
    const int cnt = g.gcount(f);
    if (cnt == 0) continue;
    const size_t off = (size_t)bs * (g.goff(f) - g.n_local());
    const size_t n = (size_t)bs * cnt;
    nccl_recv(rbuf + off, n, g.nbr[f], st);
    nccl_send(g_halo.d_send + off, n, g.nbr[f], st);
    g_stats.bytes += (double)sizeof(Real) * n;
  }
  nccl_group_end();
  ++g_stats.calls;       // no host time to record: the exchange is on the stream
}

void halo_exchange(Real* v, int bs, const Grid& g) {
  if (g.nranks <= 1) return;
  if (g.n_local() <= 0) return;
  const int nghost = g.nghost();
  if (nghost == 0) return;              // no neighbours on any face

  halo_init(g, bs);

  const double t_enter = wall_now();
  ++g_stats.calls;

  // --- gather every outgoing face ----------------------------------------
  // The send buffer mirrors the ghost layout, so face f's payload sits at
  // bs*(goff(f) - n_local()) in it and lines up with the receive side.
  const double t_pack0 = wall_now();
  for (int f = 0; f < 6; ++f) {
    const int cnt = g.gcount(f);
    if (cnt == 0) continue;
    const int off = g.goff(f) - g.n_local();
    const int threads = 256, blocks = (cnt + threads - 1) / threads;
    k_pack_face<<<blocks, threads>>>(v, g_halo.d_send + (size_t)bs * off, f,
                                     g.nxl, g.nyl, g.nzl, bs, cnt);
  }
  NVRS_CHECK_CUDA(cudaGetLastError());
  NVRS_CHECK_CUDA(cudaDeviceSynchronize());   // the gather must retire first
  g_stats.stage_seconds += wall_now() - t_pack0;

  Real* sbuf = g_halo.d_send;
  Real* rbuf = v + (size_t)bs * g.n_local();   // ghost region, contiguous
  if (!g_halo.cuda_aware) {
    const double t0 = wall_now();
    NVRS_CHECK_CUDA(cudaMemcpy(g_halo.h_send, g_halo.d_send,
                               sizeof(Real) * (size_t)bs * nghost,
                               cudaMemcpyDeviceToHost));
    g_stats.stage_seconds += wall_now() - t0;
    sbuf = g_halo.h_send;
    rbuf = g_halo.h_recv;
  }

  // --- one non-blocking round over all six faces -------------------------
  MPI_Request req[12];
  int nreq = 0;
  for (int f = 0; f < 6; ++f) {
    const int cnt = g.gcount(f);
    if (cnt == 0) continue;
    const int off = g.goff(f) - g.n_local();
    const int n = bs * cnt;
    g_stats.bytes += (double)sizeof(Real) * n;
    // Receive into face f's ghost slab; the neighbour sends it from its
    // opposite face, hence the f^1 tag.
    MPI_Irecv(rbuf + (size_t)bs * off, n, MPI_DOUBLE, g.nbr[f],
              kTagBase + (f ^ 1), MPI_COMM_WORLD, &req[nreq++]);
    MPI_Isend(sbuf + (size_t)bs * off, n, MPI_DOUBLE, g.nbr[f], kTagBase + f,
              MPI_COMM_WORLD, &req[nreq++]);
  }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

  if (!g_halo.cuda_aware) {
    const double t0 = wall_now();
    NVRS_CHECK_CUDA(cudaMemcpy(v + (size_t)bs * g.n_local(), g_halo.h_recv,
                               sizeof(Real) * (size_t)bs * nghost,
                               cudaMemcpyHostToDevice));
    g_stats.stage_seconds += wall_now() - t0;
  }
  g_stats.seconds += wall_now() - t_enter;
}

}  // namespace nvrs
