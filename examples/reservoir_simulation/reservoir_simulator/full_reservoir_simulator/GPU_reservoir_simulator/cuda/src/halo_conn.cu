// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// halo_conn.cu -- the unstructured halo exchange.
#include <chrono>

#include "halo.hpp"
#include "halo_conn.hpp"
#include "nccl_comm.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace nvrs {
namespace {

// Gather the bs-vectors of the listed cells into a contiguous buffer.
__global__ void k_gather(const Real* __restrict__ u, const int* __restrict__ idx,
                         int n, int bs, Real* __restrict__ buf) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * bs) return;
  const int cell = t / bs, v = t % bs;
  buf[t] = u[bs * idx[cell] + v];
}

// Scatter a received buffer into the ghost slots it belongs to.
__global__ void k_scatter(const Real* __restrict__ buf,
                          const int* __restrict__ idx, int n, int bs,
                          Real* __restrict__ u) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * bs) return;
  const int cell = t / bs, v = t % bs;
  u[bs * idx[cell] + v] = buf[t];
}

inline int blocks(int n, int threads = 256) {
  return (n + threads - 1) / threads;
}

}  // namespace

// ---------------------------------------------------------------------------
bool halo_conn_cuda_aware() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  return MPIX_Query_cuda_support() == 1;
#elif defined(OMPI_MAJOR_VERSION)
  // Open MPI without the query macro: assume support, which has been the
  // default in every release that matters for this simulator.
  return true;
#else
  // Unknown implementation.  Staging through pinned host memory is correct
  // everywhere; guessing wrong the other way is a segmentation fault inside
  // MPI, which is a bad way to find out.
  return false;
#endif
}

// ---------------------------------------------------------------------------
void HaloConn::init(const FieldPartition& part, int bs_max) {
  destroy();
  bs_max_ = bs_max;
  cuda_aware_ = halo_conn_cuda_aware();

  for (const HaloPeer& p : part.peers) {
    if (p.send.empty() && p.recv.empty()) continue;
    Peer q;
    q.rank = p.rank;
    q.n_send = (int)p.send.size();
    q.n_recv = (int)p.recv.size();

    if (q.n_send) {
      NVRS_CHECK_CUDA(cudaMalloc(&q.d_send_idx, sizeof(int) * q.n_send));
      NVRS_CHECK_CUDA(cudaMemcpy(q.d_send_idx, p.send.data(),
                                 sizeof(int) * q.n_send, cudaMemcpyHostToDevice));
      NVRS_CHECK_CUDA(cudaMalloc(&q.d_send_buf,
                                 sizeof(Real) * q.n_send * bs_max));
      if (!cuda_aware_)
        NVRS_CHECK_CUDA(cudaMallocHost(&q.h_send_buf,
                                       sizeof(Real) * q.n_send * bs_max));
    }
    if (q.n_recv) {
      NVRS_CHECK_CUDA(cudaMalloc(&q.d_recv_idx, sizeof(int) * q.n_recv));
      NVRS_CHECK_CUDA(cudaMemcpy(q.d_recv_idx, p.recv.data(),
                                 sizeof(int) * q.n_recv, cudaMemcpyHostToDevice));
      NVRS_CHECK_CUDA(cudaMalloc(&q.d_recv_buf,
                                 sizeof(Real) * q.n_recv * bs_max));
      if (!cuda_aware_)
        NVRS_CHECK_CUDA(cudaMallocHost(&q.h_recv_buf,
                                       sizeof(Real) * q.n_recv * bs_max));
    }
    peers_.push_back(q);
  }
}

void HaloConn::destroy() {
  for (Peer& q : peers_) {
    if (q.d_send_idx) cudaFree(q.d_send_idx);
    if (q.d_recv_idx) cudaFree(q.d_recv_idx);
    if (q.d_send_buf) cudaFree(q.d_send_buf);
    if (q.d_recv_buf) cudaFree(q.d_recv_buf);
    if (q.h_send_buf) cudaFreeHost(q.h_send_buf);
    if (q.h_recv_buf) cudaFreeHost(q.h_recv_buf);
  }
  peers_.clear();
}

long long HaloConn::bytes_per_exchange(int bs) const {
  long long n = 0;
  for (const Peer& q : peers_)
    n += (long long)(q.n_send + q.n_recv) * bs * (long long)sizeof(Real);
  return n;
}

// ---------------------------------------------------------------------------
void HaloConn::exchange_nccl(Real* u, int bs, cudaStream_t st) {
  if (peers_.empty()) return;
  for (Peer& q : peers_) {
    if (!q.n_send) continue;
    const int n = q.n_send * bs;
    k_gather<<<blocks(n), 256, 0, st>>>(u, q.d_send_idx, q.n_send, bs,
                                        q.d_send_buf);
  }
  // One peer entry per rank, so each pair has one send and one receive.
  nccl_group_start();
  for (Peer& q : peers_) {
    if (q.n_recv) nccl_recv(q.d_recv_buf, (size_t)q.n_recv * bs, q.rank, st);
    if (q.n_send) nccl_send(q.d_send_buf, (size_t)q.n_send * bs, q.rank, st);
  }
  nccl_group_end();
  double sent = 0;
  for (Peer& q : peers_) {
    sent += (double)q.n_send * bs * (double)sizeof(Real);
    if (!q.n_recv) continue;
    const int n = q.n_recv * bs;
    k_scatter<<<blocks(n), 256, 0, st>>>(q.d_recv_buf, q.d_recv_idx, q.n_recv,
                                         bs, u);
  }
  halo_note_exchange(0.0, 0.0, sent);
}

void HaloConn::exchange(Real* u, int bs) {
  if (peers_.empty()) return;
  const auto t_begin = std::chrono::high_resolution_clock::now();
  double stage = 0.0;
  auto now = [] { return std::chrono::high_resolution_clock::now(); };
  auto secs = [](std::chrono::high_resolution_clock::time_point a,
                 std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };

  const int np = (int)peers_.size();
  std::vector<MPI_Request> reqs;
  reqs.reserve(2 * np);

  // Post every receive first, so a sender never waits on an unposted receive.
  for (int i = 0; i < np; ++i) {
    Peer& q = peers_[i];
    if (!q.n_recv) continue;
    Real* dst = cuda_aware_ ? q.d_recv_buf : q.h_recv_buf;
    MPI_Request r;
    MPI_Irecv(dst, q.n_recv * bs, MPI_DOUBLE, q.rank, 77, MPI_COMM_WORLD, &r);
    reqs.push_back(r);
  }

  // Gather everything before any send, so the packing kernels queue back to
  // back and the device is not asked to alternate with MPI calls.
  for (int i = 0; i < np; ++i) {
    Peer& q = peers_[i];
    if (!q.n_send) continue;
    const int n = q.n_send * bs;
    k_gather<<<blocks(n), 256>>>(u, q.d_send_idx, q.n_send, bs, q.d_send_buf);
  }
  NVRS_CHECK_CUDA(cudaGetLastError());
  if (!cuda_aware_) {
    const auto t0 = now();
    for (int i = 0; i < np; ++i) {
      Peer& q = peers_[i];
      if (!q.n_send) continue;
      NVRS_CHECK_CUDA(cudaMemcpy(q.h_send_buf, q.d_send_buf,
                                 sizeof(Real) * q.n_send * bs,
                                 cudaMemcpyDeviceToHost));
    }
    stage += secs(t0, now());
  } else {
    // The sends read device memory that the gather kernels just wrote.
    NVRS_CHECK_CUDA(cudaDeviceSynchronize());
  }

  for (int i = 0; i < np; ++i) {
    Peer& q = peers_[i];
    if (!q.n_send) continue;
    const Real* src = cuda_aware_ ? q.d_send_buf : q.h_send_buf;
    MPI_Request r;
    MPI_Isend(src, q.n_send * bs, MPI_DOUBLE, q.rank, 77, MPI_COMM_WORLD, &r);
    reqs.push_back(r);
  }

  MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

  for (int i = 0; i < np; ++i) {
    Peer& q = peers_[i];
    if (!q.n_recv) continue;
    if (!cuda_aware_) {
      const auto t0 = now();
      NVRS_CHECK_CUDA(cudaMemcpy(q.d_recv_buf, q.h_recv_buf,
                                 sizeof(Real) * q.n_recv * bs,
                                 cudaMemcpyHostToDevice));
      stage += secs(t0, now());
    }
    const int n = q.n_recv * bs;
    k_scatter<<<blocks(n), 256>>>(q.d_recv_buf, q.d_recv_idx, q.n_recv, bs, u);
  }
  NVRS_CHECK_CUDA(cudaGetLastError());

  // One direction, matching halo_exchange's convention: what this rank sent.
  // Counted rather than halving the two-way figure -- the two differ (rank 1
  // sends 561 to rank 2 and receives 558 back), so halving would report a
  // number that is not either of them.
  double sent = 0;
  for (const Peer& q : peers_)
    sent += (double)q.n_send * bs * (double)sizeof(Real);
  halo_note_exchange(secs(t_begin, now()), stage, sent);
}

}  // namespace nvrs
