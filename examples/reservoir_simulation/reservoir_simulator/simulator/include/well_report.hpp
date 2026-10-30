// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// well_report.hpp -- per-well surface rates for a field (corner-point) run.
//
// Why this exists at all
// ----------------------
// io.cu already reconstructs well rates after the run, by walking the index
// box (k*ny + j)*nx + i and applying Peaceman to each perforated cell.  That
// works for a deck whose NX/NY/NZ describe the grid.  It cannot work for a
// field deck, where those numbers are placeholders the parser wants and the
// real grid arrives from FIELDDECK: on Norne the deck says nz = 1 while the
// grid is 46 x 112 x 22, so every well fails the `i >= d.nx` bounds test, is
// skipped, and RSM_FVM.csv comes out with a full header and rows holding only
// a time.  The reader then finds no row whose length matches the header, and
// the whole well section of the report silently disappears.
//
// A corner-point well cannot be found by index arithmetic in any case.  Its
// completions are resolved against ACTNUM and MINPV, land in active-cell
// numbering, and carry a connection factor that came from the deck (COMPDAT
// states it for most of Norne's) rather than from a formula.  The information
// is there -- it is just not reachable from an (i, j, k) triple.
//
// So this computes the rates where the answer is known: from the same
// WellCell array the assembly kernels read, copied back off the device, keyed
// by the same completion -> active-cell map the schedule resolved.  Nothing
// here re-derives a well index, a bottom-hole pressure, or a shut-in rule.
// If the kernel used it, this reports it; if the kernel skipped a completion,
// so does this.
#pragma once
#include <string>
#include <vector>

#include "common.hpp"

namespace nvrs {

struct Deck;
struct FieldModel;
struct Reservoir;
class PvtTables;

// One well at one report step, in surface volumes, sign conventions as
// ECLIPSE reports them: production positive, injection positive on its own
// keyword.  Oil and water in STB/day, gas in MSCF/day.
struct WellRateRow {
  std::string name;
  int  type = 0;              // -1 producer, +1 injector, 0 shut
  Real qo = 0, qw = 0, qg = 0;
  // Net reservoir-volume rate [RB/day].  Positive follows the well's own
  // direction: production for producers, injection for injectors.  Unlike
  // qo/qw/qg this contains no surface shrinkage or solution-gas coupling and
  // is therefore the quantity needed for a field voidage audit.
  Real qres = 0;
  Real bhp = 0;               // psia
  int  nperf = 0;             // live completions this well contributed

  // Crossflow: the part of this well's completion flow that runs BACKWARDS.
  //
  // The rates above are net sums over the completions, and the assembly has no
  // sign test, so a completion whose cell has fallen below the flowing pressure
  // puts fluid back into the formation and subtracts from the total.  That is
  // deliberate -- dropping those terms would report production the reservoir
  // never lost -- but it makes a net rate ambiguous: a well reporting 2,000
  // STB/day of oil might be making 2,000, or making 40,000 at one end while
  // 38,000 goes back in at the other.  Those are different wells, and only the
  // second is a modelling problem.
  //
  // On the Norne comparison this is the difference that matters.  B-4H and D-4H
  // report large NEGATIVE oil against Flow's positive, while their gas is large
  // and positive -- the signature of one well flowing both ways at once, which
  // is what happens when each completion is given the local mobility of every
  // phase instead of the wellbore mixture OPM's StandardWell carries.  Without
  // a number for it there is no way to tell that from a rate target that is
  // simply too small, and the two want opposite fixes.
  //
  // Reverse flow only, as a positive magnitude, in the same units as the rates
  // beside it.  A well with no crossflow reports zero, which is the common
  // case and costs nothing to carry.
  Real xflow_o = 0, xflow_w = 0, xflow_g = 0;
  int  nperf_reverse = 0;     // completions flowing the wrong way
};

// Rates for every well live at schedule step `step`.
//
// `pvt` may be null, in which case the reservoir's constant B factors are
// used -- the same fallback the kernel takes when the deck supplies no
// PVTO/PVTG.
std::vector<WellRateRow> field_well_rates(const Deck& d, const FieldModel& fm,
                                          const Reservoir& r, int step,
                                          const PvtTables* pvt);

// Sum a step's rates over the ranks.
//
// A well's completions can land on more than one rank -- Norne's horizontal
// producers run across the index-space blocks the partition cuts on -- and
// each rank can only see the ones it owns.  Without this, rank 0 writes its
// own share to RSM_FVM.csv and calls it the field's production.
//
// Every rank walks the same schedule, so the rows are in the same order with
// the same names, and a flat reduction is enough.  Collective: all ranks.
void reduce_well_rates(std::vector<WellRateRow>& rows, int nranks);

// Give every rate-controlled producer the flowing pressure that meets its
// target, and push it back to the device.
//
// Without this a producer is a pure BHP well: apply_schedule_step() reads
// bhp_limit (or DEFAULTBHP) and uses it as the OPERATING pressure, while the
// deck's ORAT/LRAT target is parsed and dropped.  A well that opens into a
// region its neighbours have been injecting gas into then produces whatever
// the drawdown gives, which on Norne is the step-49 failure: C-3H converts
// from injector to ORAT producer on 20 MAY 1999, the gas residual arrives at
// 6.6e9 against 7.2e5 for oil, and Newton oscillates in a two-cycle that no
// timestep chop can settle.
//
// Frozen-mobility linearisation, which is exact for the pressure dependence:
//
//     q_ph(pwf) = SUM_k  b_ph WI_k lambda_ph,k (p_k - pwf)
//
// is linear in pwf, so the pressure meeting a target follows in closed form.
// The BHP limit remains a limit: where the target would need to pull below it,
// the well is BHP-constrained and the limit is what it gets -- which is the
// same switch OPM makes, evaluated once per report step rather than inside the
// Newton loop.  Collective: all ranks.
void apply_producer_rate_targets(Reservoir& r, const FieldModel& fm, int step,
                                 const Deck& d, const PvtTables* pvt,
                                 bool freeze_pressure_drop = false);

// Solve each rate-controlled injector at one common datum BHP and distribute
// its requested surface rate by current completion injectivity. Collective.
void apply_injector_rate_targets(Reservoir& r, const FieldModel& fm, int step,
                                 const Deck& d, const PvtTables* pvt,
                                 bool handoff_to_producer = false);

// Oil in place, in STOCK TANK volumes.
//
// The report has been comparing the change in SUM(PORV * So) -- a reservoir
// volume -- against cumulative production in stock-tank barrels, and calling
// the ratio an implied formation volume factor.  For a live oil that is not a
// material balance at all: below the bubble point solution gas evolves and the
// oil SHRINKS, so SUM(PORV * So) falls with no oil produced whatever.  A ratio
// of 1.786 against OPM's 1.304 can therefore mean the rates are too low, or
// that more gas came out of solution, and one number cannot separate them.
//
//     N = SUM_c  PORV_c * So_c * b_o(p_c)
//
// carries the shrinkage, so the difference between two report steps IS the
// stock-tank oil that left, and comparing it against cumulative production is
// a balance that closes.  Collective: all ranks.
// `water_stb`, if given, receives SUM(PORV * Sw * b_w) by the same rule.
// `clamp_stb`, if given, receives SUM(PORV * clamp_dso * b_o): the stock-tank
// oil the Newton limiter has created or destroyed since the run began, which
// is a sink for oil that no well rate can account for.  Negative means oil
// destroyed.  See Reservoir::clamp_dso.
double field_oil_in_place(const FieldModel& fm, const Reservoir& r,
                          const PvtTables* pvt, double* water_stb = nullptr,
                          double* clamp_stb = nullptr);

// Where the residual actually lives, and what is in those cells.
//
// A step that will not converge is usually a handful of cells, and every
// question worth asking is about them: is the residual concentrated or
// spread, is the worst cell perforated, what is its pore volume, is it a
// well being asked for a rate the rock cannot deliver.  Reasoning about it
// from the norm alone has produced two wrong diagnoses of Norne's step 49.
//
// `b` is the assembled right-hand side, -R, laid out bs per cell.
void report_worst_cells(const Reservoir& r, const Real* b_dev,
                        const FieldModel* fm, int topn = 3);

}  // namespace nvrs
