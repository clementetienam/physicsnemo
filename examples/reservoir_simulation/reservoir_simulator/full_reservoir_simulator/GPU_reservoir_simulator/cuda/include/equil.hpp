// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// equil.hpp -- equilibrium initialisation from a datum, by region.
//
// What EQUIL says
// ---------------
// A field deck does not give initial pressure cell by cell.  It gives, for
// each equilibration region, a datum depth and the pressure at that datum,
// the depths of the water-oil and gas-oil contacts, and the capillary pressure
// at each contact.  The initial state is whatever is in hydrostatic
// equilibrium with that: pressure rising down each phase column at that
// phase's own gradient, and the phases stacked in the order the contacts
// dictate.
//
// Norne has five such regions, and their datums differ by nearly 400 m, so the
// difference between doing this properly and simply setting a uniform pressure
// is not subtle -- it is the difference between a field that starts in
// equilibrium and one that spends the first months of simulated time
// redistributing fluid it should never have had.
//
// Densities, and an honest dependency
// -----------------------------------
// The gradient of each phase column is rho_reservoir * g.  A deck states
// *surface* densities in DENSITY; the reservoir density is that divided by the
// phase's formation volume factor, which comes from the PVT tables and varies
// with pressure and dissolved gas.
//
// This module does not read PVT tables.  Rather than hide that behind a
// plausible-looking default, the dependency is explicit: a caller either
// supplies the formation volume factors, or supplies the phase gradients
// outright, and summary() reports which was used and what gradients resulted.
// A wrong gradient is invisible in the output -- the pressure field still looks
// like a pressure field -- so the one thing this interface must not do is let
// a caller get gradients without choosing them.
#pragma once

#include <string>
#include <vector>

#include "cpgrid.hpp"

namespace nvrs {

// One EQUIL record: everything the deck says about one region.
struct EquilRecord {
  Real datum_depth = 0;
  Real datum_pressure = 0;
  Real owc_depth = 0;          // water-oil contact
  Real pc_owc = 0;             // capillary pressure at the WOC
  Real goc_depth = 0;          // gas-oil contact
  Real pc_goc = 0;             // capillary pressure at the GOC
  int rs_method = 0;           // live-oil initialisation option
  int rv_method = 0;           // wet-gas initialisation option
  int accuracy = 0;
};

// RSVD -- dissolved gas against depth, one table per equilibration region.
//
// This is the undersaturated anchor the black-oil model needs, and it is a
// FUNCTION OF DEPTH, not a constant.  Norne states five of them, and they do
// not agree with one another: regions 1, 3 and 5 run 120.25 sm3/sm3 at the
// crest down to 106.77 deeper, while regions 2 and 4 sit at 94.5 throughout.
// In field units that is 675 scf/stb against 531 -- a 27% spread across the
// field.
//
// Running one scalar for the whole model, which is what a deck-level RS
// keyword gives, puts the crest value everywhere.  Every cell in regions 2 and
// 4 then holds 27% more dissolved gas than the deck says, its oil is swollen
// by that gas, and a reservoir barrel of it yields fewer stock-tank barrels --
// which is the direction of NVRS's oil deficit against Flow and of its
// gas-oil ratios.
struct RsvdTable {
  std::vector<Real> depth;   // ascending
  std::vector<Real> rs;      // dissolved gas at that depth

  bool empty() const { return depth.empty(); }

  // Linear in depth, flat outside the tabulated range -- the ECLIPSE rule.
  Real at(Real z) const {
    if (depth.empty()) return 0;
    if (z <= depth.front()) return rs.front();
    if (z >= depth.back()) return rs.back();
    size_t i = 0;
    while (i + 2 < depth.size() && depth[i + 1] < z) ++i;
    const Real d = depth[i + 1] - depth[i];
    if (!(d > 0)) return rs[i];
    const Real w = (z - depth[i]) / d;
    return rs[i] + w * (rs[i + 1] - rs[i]);
  }
};

// Surface densities, as DENSITY gives them, per PVT region.
struct SurfaceDensity {
  Real oil = 0, water = 0, gas = 0;
};

// The state this module produces, one entry per active cell.
struct InitialState {
  std::vector<Real> pressure;   // oil-phase pressure, deck units
  // Hydrostatic oil-minus-water pressure at each cell centre. When SWATINIT
  // is present, Eclipse/Flow scales that cell's Pcow curve so the stated Sw
  // gives this pressure difference.
  std::vector<Real> pcow_target;
  std::vector<Real> sw;
  std::vector<Real> sg;
  std::vector<Real> so;
  std::vector<int> region;      // EQLNUM, 1-based
  // How many cells took their water saturation from SWATINIT rather than
  // from the contacts, and its range.  Reported rather than assumed: with
  // capillary pressure in the flux, whether the initial state came from the
  // deck or from a sharp contact decides whether step 1 has a discontinuity
  // to relax.
  int n_swatinit = 0;
  Real swatinit_lo = 0, swatinit_hi = 0;
};

class Equilibration {
 public:
  // Parse EQUIL, DENSITY and the region arrays from a deck.  INCLUDE is
  // followed.  Reading stops at SCHEDULE.
  void read(const std::string& path);

  // Supply the phase gradients directly, in pressure units per length.
  // Overrides anything derived from densities.
  void set_gradients(Real oil, Real water, Real gas);

  // Supply formation volume factors, so reservoir densities can be formed
  // from the deck's surface densities.  Rs is the dissolved gas that rides
  // with the oil and contributes to its reservoir density.
  void set_fvf(Real bo, Real bw, Real bg, Real rs = 0.0);

  // Compute the initial state on a grid.  Throws if no gradients are
  // available, rather than inventing them.
  //
  // `default_swl` is the connate water saturation to place in the oil and gas
  // columns where the grid carries no SWL array. Water is present everywhere
  // in a real reservoir, held immobile by capillarity; initialising the oil
  // column to zero water leaves those cells with no water phase at all, which
  // is both unphysical and numerically awkward -- the water equation there has
  // no pressure coupling until water arrives from somewhere.
  InitialState initialise(const CornerPointGrid& grid,
                          Real default_swl = 0.0) const;

  int n_regions() const { return (int)records_.size(); }

  // RSVD, if the deck stated it.  One table per EQUIL record, in the order
  // they were read; empty if the deck has none.
  const std::vector<RsvdTable>& rsvd() const { return rsvd_; }
  bool have_rsvd() const {
    for (const RsvdTable& t : rsvd_) if (!t.empty()) return true;
    return false;
  }

  // The undersaturated Rs anchor for every active cell, from RSVD and the
  // cell's own EQLNUM and depth.  Empty if the deck stated no RSVD, in which
  // case the caller keeps its scalar.
  //
  // `to_field` scales the table's dissolved-gas values into the solver's
  // scf/stb; the depths are compared in whatever unit the grid is currently
  // holding, so call this BEFORE or AFTER conversion consistently -- it reads
  // grid.depth directly and the tables are in the deck's own length unit.
  std::vector<Real> rs_anchor(const CornerPointGrid& grid,
                              Real to_field_rs) const;
  const EquilRecord& record(int r) const { return records_[r]; }
  const std::vector<EquilRecord>& records() const { return records_; }
  const SurfaceDensity& density(int pvt_region = 0) const;
  bool have_gradients() const { return have_grad_; }

  Real oil_gradient() const { return grad_oil_; }
  Real water_gradient() const { return grad_water_; }
  Real gas_gradient() const { return grad_gas_; }

  void set_units(Units u) { units_ = u; }
  Units units() const { return units_; }

  std::string summary() const;
  const std::vector<std::string>& ignored_keywords() const { return ignored_; }

  // Pressure of each phase at a depth, in one region.  Exposed because it is
  // the whole content of the model and is far easier to check directly than
  // through a whole initialised field.
  Real oil_pressure(int region, Real depth) const;
  Real water_pressure(int region, Real depth) const;
  Real gas_pressure(int region, Real depth) const;

 private:
  std::vector<EquilRecord> records_;
  std::vector<RsvdTable> rsvd_;       // one per EQUIL record, in deck order
  std::vector<SurfaceDensity> densities_;
  std::vector<Real> eqlnum_;          // global numbering, if the deck gave one
  std::vector<std::string> ignored_;
  Units units_ = Units::Metric;

  Real grad_oil_ = 0, grad_water_ = 0, grad_gas_ = 0;
  bool have_grad_ = false;
  bool grad_from_fvf_ = false;
  Real bo_ = 1, bw_ = 1, bg_ = 1, rs_ = 0;

  friend class EquilReader;
};

// Acceleration due to gravity, in the units that turn a density into a
// pressure gradient for each system:
//   METRIC: kg/m^3 -> bar/m
//   FIELD : lb/ft^3 -> psi/ft
Real gravity_constant(Units u);

}  // namespace nvrs
