// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// field_model.hpp -- running a corner-point field deck.
//
// This is the join between the deck readers (cpgrid, schedule, equil) and the
// solver.  The readers produce a description of a field; this turns that
// description into the arrays the assembly kernels consume.
//
// Two things have to be reconciled.
//
// Geometry.  The solver's Reservoir was written around a uniform Cartesian
// block: one cell volume, one set of face areas, six neighbours.  A field model
// has none of those.  What it does have is a list of active cells with
// individual volumes and depths, and a list of connections with individual
// transmissibilities -- which is exactly what the connection-list assembly
// added by CONNLIST consumes.  So the conversion is mostly a matter of filling
// those arrays instead of deriving them.
//
// Units.  A corner-point deck is ordinarily METRIC and the solver works in
// field units, because every Darcy constant, well index and accumulation term
// in it was written that way.  Rather than make the kernels unit-aware -- which
// would put a conversion factor in the innermost loop and a chance of getting
// one wrong in every term -- the conversion is done once here, when the grid is
// loaded.  After that the solver sees field units and cannot tell where the
// model came from.
#pragma once

#include <memory>
#include <string>

#include "cpgrid.hpp"
#include "equil.hpp"
#include "field_partition.hpp"
#include "halo_conn.hpp"
#include "schedule.hpp"

namespace nvrs {

struct Deck;
struct Reservoir;
struct DeviceLinearSystem;

// Converting results back to the units the deck was written in.
//
// The solver works in field units throughout, because every Darcy constant,
// well index and accumulation term in it was written that way. A METRIC deck
// is converted on the way in. Reporting the answer in field units would then
// hand back a pressure in psi to someone who wrote bar, which is a needless
// way to make results hard to check against the deck they came from.
//
// So the unit system the deck declared is remembered, and results are
// converted back through it on the way out. A FIELD deck converts neither way
// and every function here is the identity.
struct UnitConverter {
  Units input = Units::Field;

  bool is_metric() const { return input == Units::Metric; }

  // Solver units -> the deck's units.
  Real pressure(Real psi) const { return is_metric() ? psi / 14.503773773 : psi; }
  Real length(Real ft) const { return is_metric() ? ft / 3.280839895 : ft; }
  Real volume(Real bbl) const { return is_metric() ? bbl / 6.2898107704 : bbl; }
  Real liquid_rate(Real bbl_day) const { return volume(bbl_day); }
  // Field gas is Mscf/day, metric gas is sm^3/day.
  Real gas_rate(Real mscf_day) const {
    return is_metric() ? mscf_day * 28.316846592 : mscf_day;
  }
  Real gas_volume(Real mscf) const { return gas_rate(mscf); }
  Real transmissibility(Real t_field) const {
    return is_metric() ? t_field * 14.503773773 / 6.2898107704 : t_field;
  }

  // The deck's units -> solver units, for anything supplied after loading.
  Real to_solver_pressure(Real p) const {
    return is_metric() ? p * 14.503773773 : p;
  }
  Real to_solver_rate(Real q) const { return is_metric() ? q * 6.2898107704 : q; }

  const char* pressure_unit() const { return is_metric() ? "bar" : "psi"; }
  const char* length_unit() const { return is_metric() ? "m" : "ft"; }
  const char* volume_unit() const { return is_metric() ? "m3" : "bbl"; }
  const char* liquid_rate_unit() const { return is_metric() ? "sm3/day" : "stb/day"; }
  const char* gas_rate_unit() const { return is_metric() ? "sm3/day" : "Mscf/day"; }
  const char* name() const { return is_metric() ? "METRIC" : "FIELD"; }
};

// Everything read from a field deck, in the deck's own units.
struct FieldModel {
  CornerPointGrid grid;
  Schedule schedule;
  Equilibration equil;
  InitialState initial;
  bool have_initial = false;

  // RSVD, resolved to one value per ACTIVE cell, in the solver's scf/stb.
  // Empty when the deck states no RSVD, in which case the run keeps the deck's
  // scalar RS.  Built during load_field_model(), while the grid's depths and
  // the RSVD tables are still in the same length unit.
  std::vector<Real> rs_anchor;

  // DRSDT from the SCHEDULE section, converted to the solver's scf/stb per
  // day.  Negative means the deck never stated it and Rs stays a pure function
  // of pressure, which is what every run did before this.
  Real drsdt = -1;
  // VAPPARS exponents are dimensionless.  vap1 acts on saturated Rv using
  // historical maximum oil saturation; vap2 acts on Rs (not used by Norne).
  Real vap1 = 0, vap2 = 0;
  std::string deck_path;

  // The unit system the deck declared, kept so that results can be reported
  // in it. Set when the model is read, and unaffected by to_field_units().
  Units input_units = Units::Field;
  UnitConverter units() const { return UnitConverter{input_units}; }

  // How the grid is split across ranks, and the exchange that follows from it.
  // On one rank the partition owns everything and the halo has no peers, so
  // the same code path serves both cases.
  FieldPartition part;
  HaloConn halo;

  int n_active() const { return grid.n_active(); }
};

// Read a corner-point deck: grid, schedule and equilibration.
//
// `gradients` supplies the phase pressure gradients that equilibration needs
// and cannot derive without PVT tables (see equil.hpp).  If it is null, the
// initial state is not computed and `have_initial` stays false; the caller
// must then supply one.  Nothing is guessed.
// `connate_water` is the water saturation placed in the oil and gas columns
// where the grid carries no SWL array. It is a required physical input, not a
// detail: a hydrocarbon column initialised with no water has no water phase to
// move, and its water equation carries no pressure coupling at all.
std::unique_ptr<FieldModel> load_field_model(const std::string& deck_path,
                                             const Real* gradients = nullptr,
                                             Real connate_water = 0.0,
                                             int rank = 0, int nranks = 1,
                                             int props_arrays = 0,
                                             Real trans_drop_rel = 1e-6);

// Convert the model in place from its own unit system to field units, which is
// what the solver expects.  A deck already in field units is left alone.
// Idempotent: calling it twice does not convert twice.
void to_field_units(FieldModel& fm);

// Build a Reservoir from a field model.  The Deck supplies the fluid and
// solver settings that a corner-point deck does not carry in a form this
// simulator reads -- viscosities, formation volume factors, saturation tables,
// timestep and solver control.
Reservoir make_reservoir_field(const Deck& d, FieldModel& fm);

// Build the linear system and the connection list from the field model's
// connections, rather than from a structured six-face stencil.
void build_sparsity_field(const Reservoir& r, DeviceLinearSystem& sys,
                          const FieldModel& fm);

// Apply the well state of one report step to the device well array.  This is
// what makes controls time-varying: it is called at every report boundary, not
// once at startup.
void apply_schedule_step(Reservoir& r, const FieldModel& fm, int step);

// As above, but supplying a flowing bottom-hole pressure for producers whose
// schedule record states none, and counting the wells left shut for want of
// one.  See Deck::prod_bhp for why this is needed.
void apply_schedule_step(Reservoir& r, const FieldModel& fm, int step,
                         Real default_prod_bhp, int* n_shut_no_bhp,
                         Real default_inj_bhp = 0,
                         int* n_inj_no_bhp = nullptr);

// Which report step covers a given time, in days from the schedule's start.
int schedule_step_at(const FieldModel& fm, double day);

}  // namespace nvrs
