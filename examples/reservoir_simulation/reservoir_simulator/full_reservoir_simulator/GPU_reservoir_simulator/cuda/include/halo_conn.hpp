// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// halo_conn.hpp -- the unstructured halo exchange.
//
// The structured halo copies contiguous slabs, so it can hand MPI a pointer
// into the state vector and be done.  An unstructured halo cannot: the cells a
// rank must send are an arbitrary list, so they have to be gathered into a
// buffer first and scattered on arrival.
//
// The gather and scatter run on the device, and the buffers stay in device
// memory, so a CUDA-aware MPI never moves the data through the host.  Where
// MPI is not CUDA-aware the buffers are staged through pinned host memory,
// which is still one copy each way rather than one per cell.
//
// Both directions are posted before either is waited on, so the exchange
// overlaps across peers instead of serialising into a ring.
#pragma once

#include <vector>

// The exchange itself is CUDA; its DECLARATION is not.  field_model.hpp holds
// a HaloConn by value, so every host translation unit that describes a field
// model -- the unit conversion, the deck load, and the tests that exercise them
// -- would otherwise need cuda_runtime.h to compile.  Following the convention
// cpgrid.hpp and pvt_tables.hpp already set: pull in common.hpp under nvcc, and
// otherwise declare the one alias this header needs.  The alias is identical to
// common.hpp's, so including both together is well formed.
#if defined(__CUDACC__)
#include "common.hpp"
#endif
#include "field_partition.hpp"

namespace nvrs {

#if !defined(__CUDACC__)
using Real = double;
#endif

class HaloConn {
 public:
  // Prepare the exchange for a partition, sized for the widest vector that
  // will be sent (bs values per cell).
  void init(const FieldPartition& part, int bs_max);
  void destroy();
  ~HaloConn() { destroy(); }

  // Fill the ghost entries of `u` from the ranks that own them.  `u` has
  // bs * n_local_ext() entries: owned first, ghosts after.
  void exchange(Real* u, int bs);
#if defined(__CUDACC__)
  // The same exchange through NCCL on a stream (nccl_comm.hpp): gather,
  // send/receive and scatter with no host synchronisation, capturable in a
  // CUDA graph.  Requires nccl_ready() and bs <= the init() width.
  void exchange_nccl(Real* u, int bs, cudaStream_t st);
#endif

  bool active() const { return !peers_.empty(); }
  int n_peers() const { return (int)peers_.size(); }
  // Bytes moved per exchange, both directions, for the run's timing report.
  long long bytes_per_exchange(int bs) const;

 private:
  struct Peer {
    int rank = -1;
    int n_send = 0, n_recv = 0;
    int* d_send_idx = nullptr;   // device: local indices to gather
    int* d_recv_idx = nullptr;   // device: local indices to scatter into
    Real* d_send_buf = nullptr;
    Real* d_recv_buf = nullptr;
    Real* h_send_buf = nullptr;  // pinned staging, when MPI is not CUDA-aware
    Real* h_recv_buf = nullptr;
  };

  std::vector<Peer> peers_;
  int bs_max_ = 0;
  bool cuda_aware_ = false;
};

// Whether this build should hand device pointers straight to MPI.  Decided
// once: it depends on the MPI implementation, not on the data.
bool halo_conn_cuda_aware();

}  // namespace nvrs
