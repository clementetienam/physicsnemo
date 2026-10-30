// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// field_partition.hpp -- distributing an unstructured grid across ranks.
//
// The problem
// -----------
// The structured halo exchanges rectangular slabs: a rank knows that its
// neighbour across the +I face needs exactly the plane i = nxl-1, and both
// sides can compute that plane's extent from the decomposition alone.  None of
// that survives corner-point geometry.  A fault connects cells that are not
// structurally adjacent, an inactive region punches holes in every slab, and
// the set of cells a rank must send is simply "the ones somebody else's
// connections reach" -- an arbitrary list, not a face.
//
// So the exchange becomes explicit: for each neighbouring rank, a list of local
// cells to send and a list of ghost slots to receive into.
//
// Agreeing on the lists without talking
// -------------------------------------
// Communication lists normally require a negotiation round: each rank works out
// what it needs and tells the owners, who then know what to send.  That is not
// necessary here, and avoiding it removes a whole class of ordering bug.
//
// Every rank reads the whole deck already -- the grid is the same file on every
// process -- so every rank can compute the entire partition and both sides of
// every list from the same data.  If both apply the same deterministic rule,
// the lists agree by construction: my send list to rank q is, entry for entry,
// q's receive list from me.  Nothing is exchanged to set that up, so nothing
// can be exchanged in the wrong order.
//
// The ordering rule is: within a pair of ranks, entries are sorted by the
// global (active) index of the cell being sent.  Both sides can evaluate that,
// and it does not depend on which side is asking.
//
// Ownership
// ---------
// Cells are assigned by cutting the structured (i,j,k) index space into a
// px x py x pz array of blocks, exactly as the Cartesian decomposition does,
// and giving each rank the active cells that fall in its block.  A fault does
// not move a cell in index space, so this keeps the vast majority of
// connections on-rank while reusing a decomposition that is already known to
// balance surface against volume.  It does not balance *active* cells
// perfectly, because inactive regions are not spread evenly; the imbalance is
// reported so it is visible rather than assumed away.
#pragma once

#include <string>
#include <vector>

#include "cpgrid.hpp"

namespace nvrs {

// Which cells this rank exchanges with one neighbour.
struct HaloPeer {
  int rank = -1;
  std::vector<int> send;   // local indices of owned cells to send
  std::vector<int> recv;   // local indices of ghost slots to receive into
};

// The distribution of one corner-point grid over the ranks.
struct FieldPartition {
  int rank = 0, nranks = 1;

  // Local numbering: owned cells occupy [0, n_owned), ghosts follow in
  // [n_owned, n_owned + n_ghost).  Ghost indices exceeding every owned index
  // is what lets the CSR columns be emitted in ascending order.
  int n_owned = 0;
  int n_ghost = 0;
  int n_local_ext() const { return n_owned + n_ghost; }

  // Global row range this rank owns, for the distributed solvers.  Contiguous
  // by construction: ranks are numbered in block order and each owns a
  // consecutive run.
  long long row0 = 0;
  long long n_global = 0;

  // Maps between the grid's active numbering and this rank's local numbering.
  std::vector<int> local_of_active;    // n_active, -1 if not present here
  std::vector<int> active_of_local;    // n_local_ext()
  std::vector<int> owner_of_active;    // n_active, which rank owns each cell

  // Connections whose flux this rank is responsible for: those with at least
  // one owned endpoint, in local indices.  A connection between two owned
  // cells appears once; a connection to a ghost appears on both ranks, and
  // each computes the half that lands in its own cell.
  std::vector<Connection> conns;

  std::vector<HaloPeer> peers;

  // ---- diagnostics -------------------------------------------------------
  int n_cut_conns = 0;      // connections crossing a rank boundary
  int min_owned = 0, max_owned = 0;
  std::string summary() const;
};

// Choose a px x py x pz factorisation of `nranks` that minimises the total
// halo surface for the given index-space extents.  Shared with the Cartesian
// decomposition's intent, restated here because it applies to the index space
// rather than to a cell count.
void choose_factorisation(int nranks, int nx, int ny, int nz,
                          int& px, int& py, int& pz);

// Build the partition.  Every rank calls this with the same grid and gets a
// consistent view; no communication takes place.
FieldPartition partition_field(const CornerPointGrid& g, int rank, int nranks);

}  // namespace nvrs
