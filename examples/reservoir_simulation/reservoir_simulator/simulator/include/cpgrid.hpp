// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// cpgrid.hpp -- corner-point geometry, faults, and the connection list.
//
// Why this exists
// ---------------
// The simulator's structured grid (grid.hpp) describes a box of uniform cells
// whose neighbours are known by arithmetic: cell (i,j,k) touches (i+1,j,k) and
// so on, six faces, one transmissibility each.  Every field model departs from
// that in the same three ways:
//
//   * the geometry is corner-point.  Cells are hexahedra defined by eight
//     corners lying on pillars, so volumes, depths and face areas have to be
//     computed rather than assumed;
//   * cells are inactive.  ACTNUM removes the parts of the box that are not
//     reservoir, and MINPV removes what is left that is too thin to matter.
//     The survivors must be numbered contiguously or the linear algebra wastes
//     rows on nothing;
//   * faults displace layers.  A cell may face part of one neighbour and part
//     of another, or a cell that is not its structured neighbour at all.  The
//     connection graph stops being 7-point.
//
// All three reduce to one abstraction: a list of connections, each naming two
// active cells and the transmissibility between them.  That is what this file
// produces.  The solver does not care where a connection came from -- a
// structured face, a fault juxtaposition, or an explicit NNC -- so expressing
// the grid this way is what allows a field deck to run through the same
// assembly, preconditioner and Krylov solver as the synthetic cases.
//
// Deck arrays
// -----------
// A field deck does not state its properties outright.  It reads PERMX from a
// file, COPYs it to PERMZ, and then MULTIPLYs PERMZ by a different factor in
// each of twenty-two layers.  The reader therefore keeps a named array store
// rather than a fixed set of members, and implements the operators that decks
// use to build one array from another: COPY, EQUALS, MULTIPLY, ADD, each over
// an optional box, with BOX/ENDBOX setting the default.  Anything the store
// holds can be asked for by name, which is also how region arrays such as
// FLUXNUM and SATNUM arrive.
//
// Units
// -----
// A corner-point deck is ordinarily METRIC: metres, mD, m^3, bar, cP, with a
// transmissibility constant of 0.0085267014.  set_units() selects the
// convention, and a METRIC or FIELD keyword in the deck sets it directly; the
// geometry itself is unit agnostic.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace nvrs {

// Grid construction is host-side and has no CUDA in it, so this header
// deliberately does not include common.hpp: it lets the reader, the geometry
// and the tests build with a plain host compiler, which is what makes the
// checks in tests/test_cpgrid.cpp runnable anywhere.  The alias is identical to
// the one in common.hpp, so including both headers together is well formed.
using Real = double;

// One connection between two active cells.  `trans` already carries every
// multiplier that applies to it: NTG on the lateral faces, MULTX/Y/Z, MULTFLT
// for a fault face, and MULTREGT across a region boundary.
struct Connection {
  int a = -1;          // active cell index, a < b
  int b = -1;
  Real trans = 0;      // transmissibility [cP.m^3/(day.bar)] or field equiv.
  int dir = 0;         // 0 = I, 1 = J, 2 = K, 3 = non-neighbour (fault/NNC)
  Real threshold = 0;  // THPRES applied to every phase potential [bar or psi]
};

struct RegionThreshold {
  int r1 = -1, r2 = -1;  // one-based EQLNUM regions
  Real pressure = 0;
};

// A named fault, as FAULTS declares it: a set of cell faces.  The multiplier
// arrives separately through MULTFLT, which is why the two are kept apart.
struct FaultFace {
  int i1 = 0, i2 = 0, j1 = 0, j2 = 0, k1 = 0, k2 = 0;   // 1-based, inclusive
  int face = 0;        // 0 = I-, 1 = I+, 2 = J-, 3 = J+, 4 = K-, 5 = K+
};

struct Fault {
  std::string name;
  std::vector<FaultFace> faces;
  Real mult = 1.0;     // from MULTFLT; 1.0 until one is read
};

// A MULTREGT record: scale the transmissibility of every connection whose two
// cells fall in the named pair of regions.  Region -1 means "any".
struct RegionMult {
  int r1 = -1, r2 = -1;
  Real mult = 1.0;
  int dir = 7;              // bitmask 1=X, 2=Y, 4=Z
  std::string region_array = "MULTNUM";
};

// The unit system a deck declares.  Only the constants that enter the
// transmissibility and the well index differ.
enum class Units { Field, Metric };

// ---------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------
class CornerPointGrid {
 public:
  // ---- dimensions -------------------------------------------------------
  int nx = 0, ny = 0, nz = 0;
  int n_global() const { return nx * ny * nz; }
  int n_active() const { return (int)active_to_global_.size(); }

  // ---- geometry, per ACTIVE cell ----------------------------------------
  std::vector<Real> volume;     // bulk volume [m^3] (or ft^3 in field units)
  std::vector<Real> depth;      // cell centre depth, positive downwards
  // Cell extents along each index direction.  A corner-point cell has no
  // single "dx", so these are the mean edge lengths: the I-direction extent is
  // the mean of the four edges that run from the cell's I- face to its I+
  // face, and likewise for J and K.  The Peaceman well index needs all three,
  // and which two it uses depends on the direction the well penetrates.
  std::vector<Real> dx, dy, dz;
  std::vector<Real> centroid_x, centroid_y;

  // ---- rock, per ACTIVE cell --------------------------------------------
  std::vector<Real> poro;
  std::vector<Real> ntg;
  std::vector<Real> permx, permy, permz;
  std::vector<Real> porv;       // pore volume, after NTG, MULTPV and MINPV

  // Read per-cell arrays defined in PROPS, REGIONS and SOLUTION as well as
  // GRID.  Deck keyword PROPSARRAYS, and off by default.
  //
  // It is not only ENDSCALE that becomes visible when this is on.  EQLNUM is
  // a REGIONS keyword, and it selects which EQUIL record each cell
  // equilibrates against -- Norne has five, with different contacts -- so
  // turning this on changes the initial state of the whole field, not just
  // the relative permeability.  That is more faithful to the deck, and it is
  // also a different model from the one every previous run used, so it is a
  // decision the deck has to make explicitly.
  bool read_props_arrays = false;

  // Deck keyword SCALECRS.  YES selects three-point endpoint scaling for the
  // displacing phase; anything else (and its absence) leaves it two-point.
  bool scalecrs = false;

  // ---- connectivity ------------------------------------------------------
  std::vector<Connection> conns;

  // ---- reading -----------------------------------------------------------
  // Parse a GRDECL file or a deck.  Array keywords are stored by name; the
  // deck operators (COPY, EQUALS, MULTIPLY, ADD, BOX) are applied as they are
  // read, in order, which is the only way to get a deck's intent right.
  // INCLUDE is followed, resolved relative to the including file.
  //
  // Parsing stops at PROPS, REGIONS, SOLUTION, SUMMARY or SCHEDULE: those
  // sections hold record-structured keywords this reader does not model, and
  // stopping is honest where guessing would silently corrupt the grid.
  void read_grdecl(const std::string& path);

  // Same parser; the name reads better at a call site that is loading one
  // property include rather than a grid.
  void read_property(const std::string& path) { read_grdecl(path); }

  // ---- construction ------------------------------------------------------
  void set_units(Units u) { units_ = u; }
  Units units() const { return units_; }

  // Cells whose pore volume falls below this are deactivated (MINPV).  A deck
  // that states MINPV sets this itself.
  void set_min_pore_volume(Real v) { min_pv_ = v; }
  Real min_pore_volume() const { return min_pv_; }

  // Connections below this fraction of the largest transmissibility are
  // dropped, unless dropping one would leave a cell with no connections at
  // all.  They carry no fluid and a great deal of ill conditioning; see
  // build_connections().  Zero disables the filter.
  void set_drop_threshold(Real rel) { drop_rel_ = rel; }
  Real drop_threshold() const { return drop_rel_; }

  // Compute geometry, number the active cells, and build the connection list
  // with transmissibilities.  Call after the grid and properties are read.
  void build();

  // ---- the array store ---------------------------------------------------
  // Any keyword the deck defined, on the GLOBAL numbering.  Returns nullptr if
  // the deck never mentioned it.
  const std::vector<Real>* array(const std::string& name) const;
  // The same array compacted onto the active numbering; empty if absent.
  std::vector<Real> active_array(const std::string& name) const;
  // Whether each active-cell value was explicitly supplied by an array or a
  // grid edit. Grid-edit destinations are born with ECLIPSE defaults, so a
  // numeric zero cannot distinguish an explicit zero from an untouched cell.
  std::vector<char> active_array_defined(const std::string& name) const;
  bool has_array(const std::string& name) const { return array(name) != nullptr; }

  // ---- indexing ----------------------------------------------------------
  int global_index(int i, int j, int k) const { return (k * ny + j) * nx + i; }
  int active_of_global(int g) const { return global_to_active_[g]; }
  int global_of_active(int a) const { return active_to_global_[a]; }

  // ---- diagnostics -------------------------------------------------------
  // A short account of what was read and built, for the run log.  A field deck
  // is large enough that silent misreadings are expensive; this is the record
  // that says what the simulator believes it was given.
  std::string summary() const;

  // Keywords the reader encountered but does not implement.  Reported so that
  // a deck relying on one of them fails loudly in review rather than quietly
  // at run time.
  const std::vector<std::string>& ignored_keywords() const { return ignored_; }

  // The faults the deck declared, with the multipliers in force.  Exposed for
  // reporting and for sensitivity work: sealing or opening a named fault is
  // one of the first things anyone asks of a faulted model.
  const std::vector<Fault>& faults() const { return faults_; }
  std::vector<std::string> fault_names() const;
  // Override one fault's multiplier before build().  Returns false if no
  // fault of that name was declared.
  bool set_fault_multiplier(const std::string& name, Real m);

  // The eight corners of a global cell, ordered
  //   0..3 = top    face (i,j), (i+1,j), (i,j+1), (i+1,j+1)
  //   4..7 = bottom face, same order.
  void cell_corners(int i, int j, int k, Real xyz[8][3]) const;

 private:
  Units units_ = Units::Metric;
  Real min_pv_ = 0.0;
  // PINCH controls vertical NNC generation through inactive cells.  GAP is
  // the Eclipse default: cells removed by MINPV may be crossed regardless of
  // thickness, while cells already inactive in ACTNUM may be crossed only
  // when their geometric thickness is at most pinch_thickness_.
  bool pinch_active_ = false;
  bool pinch_gap_ = true;
  Real pinch_thickness_ = 0.001;
  // Default chosen from the measured distribution on Norne: it removes
  // about 3% of connections carrying under a millionth of the total
  // transmissibility, and isolates no cell.
  Real drop_rel_ = 1e-6;

  std::vector<Real> coord_;     // 6 * (nx+1) * (ny+1)
  std::vector<Real> zcorn_;     // 8 * nx * ny * nz
  std::vector<RegionThreshold> region_thresholds_;

  // Block-centred geometry, when the deck gives cell sizes instead of corners.
  // DXV/DYV/DZV are one value per index in that direction; DX/DY/DZ are one
  // per cell and are held in the ordinary array store.  Empty unless the deck
  // used them.
  std::vector<Real> dxv_, dyv_, dzv_;
  // True when the corners below were built from those sizes rather than
  // read from the deck.  Reported, because the two are not the same input
  // and a reader should say which it had.
  bool block_centred_ = false;

  // True once the reader has passed PROPS/REGIONS/SOLUTION.  A member rather
  // than a local because INCLUDE recurses into read_grdecl, and an include
  // opened from PROPS is still in PROPS.
  bool in_tail_ = false;

  // Every array keyword the deck defined, on the global numbering.
  std::map<std::string, std::vector<Real>> arrays_;
  std::map<std::string, std::vector<char>> array_defined_;

  std::vector<Fault> faults_;
  std::vector<RegionMult> regmults_;
  std::vector<std::string> ignored_;

  // The current BOX, 0-based and inclusive; the whole grid unless set.
  int bi1_ = 0, bi2_ = -1, bj1_ = 0, bj2_ = -1, bk1_ = 0, bk2_ = -1;

  std::vector<int> actnum_;
  std::vector<int> global_to_active_;
  std::vector<int> active_to_global_;

  // Which inactive cells were removed by MINPV rather than ACTNUM.  PINCH GAP
  // treats these specially; ordinary ACTNUM holes remain barriers unless an
  // explicit PINCH thickness says their geometry has collapsed.
  std::vector<char> minpv_cut_;

  // Geometry on the GLOBAL numbering.  The connection pass walks the
  // structured index space, including cells that are about to be dropped, so
  // it needs the geometry before number_active() compacts it.
  std::vector<Real> geom_vol_, geom_cx_, geom_cy_, geom_cz_;
  std::vector<Real> geom_dx_, geom_dy_, geom_dz_;

  // Statistics gathered during build(), reported by summary().
  int n_inactive_ = 0, n_minpv_ = 0, n_fault_conns_ = 0, n_nnc_ = 0,
      n_regt_conns_ = 0, n_sealed_ = 0, n_dropped_small_ = 0;

  // Fetch an array for writing, creating it at its documented default if the
  // deck has not mentioned it yet.
  std::vector<Real>& array_for_write(const std::string& name);
  static Real default_value(const std::string& name);
  static bool is_array_keyword(const std::string& name);

  void set_dims(int nx_, int ny_, int nz_);
  void reset_box() { bi1_ = bj1_ = bk1_ = 0; bi2_ = nx - 1; bj2_ = ny - 1; bk2_ = nz - 1; }

  void compute_geometry();
  void number_active();
  // Build COORD and ZCORN from DX/DY/DZ (or DXV/DYV/DZV) and TOPS, so that a
  // block-centred deck is read through the same geometry as a corner-point
  // one.  Returns false, changing nothing, when the deck did not give enough
  // to do it.  Called by build() only when COORD and ZCORN are absent.
  bool synthesise_block_centred();

  void build_connections();
  Real trans_constant() const;
  Real half_trans(int g, int face, const Real perm[3]) const;
  void apply_fault_multipliers();
  void apply_region_multipliers();
};

}  // namespace nvrs
