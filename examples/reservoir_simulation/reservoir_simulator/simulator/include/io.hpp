// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// io.hpp — output: field snapshots (binary + JSON meta consumed by the Python
// post-processor), well-rate summary (RSM CSV), and the written insight summary.
#pragma once

#include "deck.hpp"
#include "ecl_io.hpp"
#include "linear_solver.hpp"
#include "physics.hpp"
#include "well_report.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace nvrs {

class Writer {
 public:
  Writer(const Deck& d, const Grid& g);

  // Describe the corner-point grid a field deck brought with it.
  //
  // Without this the sidecar reports the RUN deck's DIMENS, which for a field
  // model is 1 x 1 x 1 -- the run deck does not declare a grid, it names one --
  // and the well list is the run deck's, which is empty.  Every plot then draws
  // a single cell and no wells, from a run that solved 44,431 of them.
  //
  // The map matters as much as the dimensions.  State vectors are indexed by
  // ACTIVE cell, and a field grid's active cells are a subset scattered through
  // the box: Norne has 44,431 of 113,344.  Reshaping the state to nx.ny.nz
  // without scattering it first lays the field out in the wrong cells entirely.
  // A well marker.  Three kinds, not two: a water injector and a gas injector
  // do opposite things to a reservoir and belong in different colours, and on
  // Norne the same well is both at different times -- C-4H alternates all the
  // way through the history.  "winj", "ginj" or "prod".
  struct WellXY { std::string name; int i, j; std::string kind; };

  // Everything a field grid knows about where its cells actually are.
  //
  // The index box is enough to draw a map and not enough to draw Norne.  A
  // corner-point cell is a hexahedron on two pillars: it is not a cuboid, the
  // layers are not flat, and a fault throw offsets one column against its
  // neighbour so that cells with adjacent (i,j,k) are nowhere near each other
  // in space.  Drawing such a grid as a regular array shows the topology and
  // hides the geometry -- which on a faulted field is most of what there is to
  // see.  So the corners go out as well, eight per cell, and anything that
  // wants the true shape, the faults or a depth contour has them.
  struct FieldGeometry {
    int nx = 0, ny = 0, nz = 0;
    std::vector<int> active_to_global;   // global box index of each active cell
    // Six float32 columns per active cell: i, j, k (1-based) then the x, y, z
    // of the cell centre.  The layout a corner-point plot wants: reshape the
    // columns onto the index box and plot_surface draws the real dipping,
    // faulted layers instead of a cuboid.
    std::vector<float> cgrid;
    std::vector<WellXY> wells;

    // Global solver row -> active cell, length n_active.  Empty on one rank.
    //
    // A multi-rank gather arrives in GLOBAL ROW order: rank 0's owned cells,
    // then rank 1's, each ascending in active index.  That is NOT active
    // order, because the partition cuts index-space blocks and so each rank
    // owns a scattered subset -- rank 0 does not own active cells 0..11591.
    // Writing the gathered vector out as though it were active-ordered lays
    // every value in the wrong cell.  This permutation is what makes the two
    // agree, and without it the gather has to stay switched off.
    std::vector<int> active_of_row;
  };
  void set_field_grid(const FieldGeometry& g);

  // Append the current state as report step `step` at time `t_days`.
  void snapshot(Reservoir& r, int step, Real t_days);

  // After the run: write meta.json, RSM_FVM.csv (well rates) and summary.txt.
  void finalize(const Deck& d, Reservoir& r,
                const std::vector<Real>& step_times,
                const std::vector<int>& newton_iters,
                const std::vector<int>& linear_iters, double wall_seconds,
                double assembly_seconds = 0.0,
                double linear_seconds = 0.0,
                const CprTimings& cpr = CprTimings{});

  // Well rates for one report step, computed by the caller where the schedule
  // and the device state are both in scope.
  //
  // The alternative -- reconstructing them in finalize() from the snapshot
  // files -- is what the Peaceman post-process below does, and it works only
  // when the deck's own NX/NY/NZ describe the grid.  A field deck's do not:
  // they are placeholders, so every well fails the bounds test, every data row
  // comes out holding just a time, and the reader drops the lot.  When these
  // are recorded they are used instead, because they came from the same
  // WellCell array the kernels read.
  void record_well_rates(Real t_days, const std::vector<WellRateRow>& rows);
  void record_well_cumulative(double oil_produced, double water_produced);

  // Which rank is actually rank 0.
  //
  // The writer decided this from its own grid, and on a field run that grid
  // says "rank 0 of 1" on every rank -- so all four wrote meta.json, and the
  // reader found four JSON documents concatenated:
  //   ! cannot read meta.json: Extra data: line 31 column 1
  // The state files and the well CSV were being written four times over too.
  void set_mpi(int rank, int nranks) { mpi_rank_ = rank; mpi_nranks_ = nranks; }
  // Run-level figures written to meta.json by finalize(): the I/O inside the
  // wall clock, the time-integration work, and the arithmetic of the arms.
  void set_run_stats(double io_s, long long substeps, long long newton,
                     long long newton_lost, long long linear, double days,
                     double wall_s, int operator_mode, int cpr_decouple) {
    rs_io_ = io_s; rs_sub_ = substeps; rs_newton_ = newton;
    rs_newton_lost_ = newton_lost; rs_linear_ = linear; rs_days_ = days;
    rs_wall_ = wall_s; rs_opmode_ = operator_mode; rs_decouple_ = cpr_decouple;
    rs_set_ = true;
  }

  // The grid's own pore volume [rb] and centre depth [ft] per ACTIVE
  // cell, forwarded to the ECLIPSE INIT file.  Call before the first
  // snapshot, which is when that file is written.
  void set_cell_geometry(std::vector<float> porv, std::vector<float> depth);

  // The DECK's porosity per active cell, for the INIT file's PORO.
  // The solver's own array has NTG folded in, which PORV already
  // carries, so writing that would count it twice.
  void set_cell_porosity(std::vector<float> poro);

 private:
  Deck deck_;
  Grid grid_;
  std::string dir_;
  int nsnap_ = 0;
  bool rs_set_ = false;
  double rs_io_ = 0, rs_days_ = 0, rs_wall_ = 0;
  long long rs_sub_ = 0, rs_newton_ = 0, rs_newton_lost_ = 0, rs_linear_ = 0;
  int rs_opmode_ = 1, rs_decouple_ = 0;
  std::vector<Real> times_;
  // Global permeability/porosity fields, gathered once on rank 0 at the first
  // snapshot.  The well-rate post-process needs the *actual* cell permeability
  // to reproduce the same Peaceman well index the solver used, and the ECLIPSE
  // INIT file needs both.
  std::vector<float> perm_global_, permz_global_, poro_global_;
  // Corner-point geometry from a field deck; zero when the grid is the run
  // deck's own uniform box.
  FieldGeometry fgeom_;
  // Whether the static rock has been gathered.  This MUST be a plain flag kept
  // in step on every rank: gather_field only resizes its output on rank 0, so
  // testing perm_global_.empty() is true forever on the other ranks and they
  // would keep entering a collective that rank 0 has stopped calling.
  // True MPI identity, independent of grid_ (see set_mpi).
  int mpi_rank_ = 0, mpi_nranks_ = 1;
  bool rock_gathered_ = false;
  // Per-report-step well rates, when the driver supplied them.  Empty for a
  // structured deck, which still takes the Peaceman path in finalize().
  std::vector<Real> wr_times_;
  std::vector<std::vector<WellRateRow>> wr_rows_;
  std::vector<std::array<double, 2>> wr_cumulative_;
  // ECLIPSE / OPM output (rank 0 only; null when ECLOUT is 0).
  std::unique_ptr<ecl::EclipseWriter> ecl_;
  // rank-0 gathers global fields; other ranks send their slabs.
  void gather_field(const Real* dev, int stride, int offset, std::vector<float>& out);
};

}  // namespace nvrs
