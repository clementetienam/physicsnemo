// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// stub_halo_conn.cpp -- the one CUDA symbol a host-only deck test still needs.
//
// FieldModel holds a HaloConn by value, so ~FieldModel calls ~HaloConn, which
// calls HaloConn::destroy() -- and that lives in halo_conn.cu, which needs
// nvcc.  Nothing else about loading a deck does: the grid, the schedule,
// equilibration, the unit conversion and the partition are all host code, and
// keeping them testable on a laptop is the whole reason units_field.cpp exists
// as a .cpp.
//
// The stub is safe precisely because no host test ever calls init(): the peer
// list is empty and there are no device buffers to free, so the real destroy()
// would have nothing to do either.  Stubbing the unreachable rather than
// weakening the header keeps ~HaloConn honest in the build that does exchange.
// scripts/syntax_check.sh stubs MPI and HYPRE the same way and for the same
// reason.
//
// This file is for the tests only.  It is not in CMakeLists.txt or the
// Makefile, and linking it into the simulator would be a bug -- the real
// definition would collide with it at link time, which is the check.

#include "halo_conn.hpp"

#include <cstdio>
#include <cstdlib>

namespace nvrs {

void HaloConn::destroy() {
  // Reached only through ~HaloConn on a model that was never given a
  // partition to exchange over.  If a test ever does set one up, the exchange
  // it wants is the CUDA one, and this says so rather than quietly freeing
  // nothing.
  if (!peers_.empty()) {
    std::fprintf(stderr,
                 "stub_halo_conn: a host-only test built a halo with %zu "
                 "peer(s); link halo_conn.cu instead.\n",
                 peers_.size());
    std::abort();
  }
}

}  // namespace nvrs
