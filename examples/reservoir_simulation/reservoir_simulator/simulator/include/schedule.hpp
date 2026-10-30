// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// schedule.hpp -- wells, completions, and time-varying control.
//
// What a field schedule actually is
// ---------------------------------
// The synthetic cases this simulator grew up on declare a well once and run it
// unchanged to the end.  A field deck does not work that way.  Norne's history
// is 247 dated steps over nine years, and across them wells are drilled, opened
// and shut, re-perforated, switched between rate and pressure control, and
// handed new targets every few weeks.  The schedule is not configuration; it is
// a sequence of events.
//
// This module turns that sequence into something a simulator can index: a list
// of steps, each holding the complete state of every well as of that date.
// Reading the deck is a fold over events -- WELSPECS defines, COMPDAT
// perforates, WCONPROD/WCONINJE/WCONHIST set targets, WELOPEN opens and shuts
// -- and each DATES or TSTEP record freezes the running state into a step.
// The simulator then never parses anything: it asks for the state at step n.
//
// Control, and why a mode alone is not enough
// -------------------------------------------
// A well is not simply "on rate" or "on pressure".  It carries a target mode
// and a set of limits, and it moves between them.  In practice injectors are
// given a rate and a bottom-hole pressure ceiling: they inject at the rate
// until the formation will not take it, and thereafter hold the pressure.
// Producers are commonly the mirror image -- held at a bottom-hole pressure,
// producing whatever the reservoir delivers -- with rate ceilings that cap them
// if the reservoir delivers too much.
//
// So `WellState` stores the requested mode *and* every limit the deck gave,
// and `effective_mode()` decides which one binds given the rate and pressure
// the well actually achieved.  A simulator that hard-codes one mode per well
// will run, and will quietly diverge from the deck the first time a well hits
// a limit -- which for an injector is usually within the first few steps.
//
// Units
// -----
// Rates and pressures are held in the deck's own unit system: METRIC gives
// sm^3/day and bar, FIELD gives stb/day, Mscf/day and psi.  Conversion belongs
// at the point of use, not here, so that what this module reports can be
// compared directly against the deck it came from.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "cpgrid.hpp"

namespace nvrs {

// ---------------------------------------------------------------------------
// Completions
// ---------------------------------------------------------------------------

// The direction the wellbore penetrates a cell.  This is what distinguishes a
// horizontal completion from a vertical one, and it changes the well index:
// the Peaceman radius is built from the two permeabilities and two cell
// extents *perpendicular* to the wellbore, and the exposed length is the
// extent *along* it.  Norne uses 'Z' for its vertical wells and 'X' for the
// horizontal ones, which run along I at fixed J.
enum class PenetrationDir { X = 0, Y = 1, Z = 2 };

struct Completion {
  int i = 0, j = 0, k = 0;              // 0-based global cell indices
  bool open = true;
  PenetrationDir dir = PenetrationDir::Z;

  // The deck may state the connection transmissibility factor outright, or
  // default it (`1*`) and expect the simulator to compute one.  Norne states
  // it for most completions and defaults it for none, but a deck that defaults
  // it is ordinary, so both paths must exist.  Negative means "not given".
  Real cf = -1.0;
  Real diameter = 0.0;                  // wellbore diameter
  Real kh = -1.0;                       // permeability-thickness, if given
  Real skin = 0.0;
  Real r0 = -1.0;                       // Peaceman radius, if given
  int sat_table = -1;

  // Filled in by resolve(): the active-cell index this completion lands in,
  // and the well index actually used.  A completion in an inactive cell -- a
  // routine occurrence once MINPV and ACTNUM have run -- resolves to -1 and is
  // reported rather than silently dropped.
  int cell = -1;
  Real wi = 0.0;
};

// ---------------------------------------------------------------------------
// Wells
// ---------------------------------------------------------------------------
enum class WellType { Producer, Injector };
enum class InjectedPhase { Water, Gas, Oil };

// Control modes, spanning what WCONPROD, WCONINJE and WCONHIST can request.
enum class CtrlMode {
  Shut,
  ORAT,    // oil rate
  WRAT,    // water rate
  GRAT,    // gas rate
  LRAT,    // liquid rate
  RESV,    // reservoir volume rate
  BHP,     // bottom-hole pressure
  THP,     // tubing-head pressure
  RATE,    // injector surface rate
  GRUP,    // under group control
};

const char* to_string(CtrlMode m);
const char* to_string(WellType t);
const char* to_string(InjectedPhase p);

// The complete control state of one well at one moment.
//
// Every limit the deck gave is kept, whether or not it is the active target,
// because which one binds is a run-time question, not a parsing one.  A limit
// that was defaulted stays at its "absent" sentinel: producers get +infinity
// for rate ceilings and 0 for the BHP floor, injectors the reverse, so an
// absent limit can never be the binding one.
struct WellState {
  bool defined = false;         // has WELSPECS been seen
  bool open = false;
  WellType type = WellType::Producer;
  InjectedPhase inj_phase = InjectedPhase::Water;
  CtrlMode mode = CtrlMode::Shut;

  // Targets and limits, in deck units.
  Real orat = 0, wrat = 0, grat = 0, lrat = 0, resv = 0;
  Real bhp_limit = 0;           // producer: lower bound; injector: upper bound
  Real thp_limit = 0;
  Real surface_rate = 0;        // injector target
  Real resv_rate = 0;           // injector reservoir-volume target
  Real efficiency = 1.0;        // WEFAC
  int vfp_table = -1;

  // Observed rates from WCONHIST, kept separate from prediction targets.
  // History-control assembly selects the requested CMODE and, for RESV,
  // converts all three observations at the current PVT state.
  bool historical = false;
  Real hist_orat = 0, hist_wrat = 0, hist_grat = 0, hist_bhp = 0, hist_thp = 0;

  // Which control binds, given what the well achieved over the last step.
  //
  // `rate` is the well's rate in the units of its requested mode and `bhp` its
  // bottom-hole pressure.  An injector asked for a rate that drives it past its
  // pressure ceiling switches to BHP; a producer held on rate that falls below
  // its pressure floor does the same.  Returns the requested mode when nothing
  // binds, which is the common case.
  CtrlMode effective_mode(Real rate, Real bhp) const;

  // The target value that goes with `mode`, in deck units.
  Real target() const;
};

struct Well {
  std::string name;
  std::string group = "FIELD";
  int head_i = 0, head_j = 0;           // 0-based
  Real ref_depth = -1.0;                // datum for the BHP; -1 = top completion
  std::string preferred_phase = "OIL";
  std::vector<Completion> completions;
  WellState state;
};

// ---------------------------------------------------------------------------
// One report step
// ---------------------------------------------------------------------------
// A group's production target, from GCONPROD.
//
// Norne's producers spend much of the history on WCONPROD records that name
// RESV and default the value: they are not asking for a reservoir rate, they
// are deferring to the group.  Without this the wells had no constraint at
// all and ran on bottom-hole pressure, producing ten to twenty-five times
// what the field actually made.
struct GroupControl {
  std::string name;
  CtrlMode mode = CtrlMode::Shut;   // ORAT, WRAT, GRAT, LRAT or RESV
  Real target = 0;                  // in deck units, converted with the rest
  bool defined = false;
};

// GCONINJE: a group's ceiling on INJECTION, one record per phase.
//
// Without this each injector delivers whatever rate its own WCONINJE states,
// and the group's ceiling never applies.  On Norne that is not a detail: the
// wells' individual rates sum well above what the group is allowed, and the
// field takes in 1147 M reservoir bbl of water against Flow's 648.
//
// Keyed by (group, phase) because a group injects water and gas under
// separate targets, and a well is bound only by the record for the phase it
// injects.
struct GroupInject {
  std::string name;
  int phase = 0;                    // 1 water, 2 gas, 3 oil
  // RATE: a surface rate.  RESV: a reservoir rate.  REIN and VREP are
  // fractions of what the group produced, and are not implemented -- a group
  // under one of those keeps its wells' own rates and says so.
  enum class Ctl { None, Rate, Resv, Rein, Vrep } ctl = Ctl::None;
  Real target = 0;
  bool defined = false;
};

struct ScheduleStep {
  double day = 0.0;                     // days since the deck's START
  std::string date;                     // "15 JAN 1998", for reporting
  std::vector<Well> wells;              // complete state as of this date

  // Group targets in force at this date, by group name, and the GRUPTREE
  // child -> parent map.  A well names its group in WELSPECS; the tree is
  // what lets a well under a subgroup find a target set on FIELD.
  std::map<std::string, GroupControl> groups;
  // GCONINJE, keyed "GROUP|phase" -- a group injects water and gas under
  // separate targets and a well is bound only by its own phase's record.
  std::map<std::string, GroupInject> ginject;
  std::map<std::string, std::string> parent;

  const Well* find(const std::string& name) const;

  // The nearest ancestor of `group` carrying a rate target, or null.
  // Walks child -> parent, so a target on FIELD governs everything beneath
  // it unless a subgroup states its own.
  const GroupControl* group_target(const std::string& group) const;

  // The nearest ancestor of `group` carrying an injection target for this
  // phase (1 water, 2 gas), or null.  Same walk as group_target().
  const GroupInject* group_inject(const std::string& group, int phase) const;
};

// Convert GCONINJE targets from a metric deck to the solver's field units.
// RATE is phase-dependent (gas uses scf/day; liquids use stb/day), RESV is a
// reservoir volume rate, and REIN/VREP are dimensionless fractions.
void convert_group_inject_targets(ScheduleStep& step,
                                  Real liquid_rate_factor,
                                  Real gas_rate_factor);

// What changed about the wells between two report steps.
//
// A field schedule is a sequence of events, and the events are the whole
// content of it: a well is drilled, opened, given a new target, converted from
// production to injection, shut in.  Running one without saying what those
// events are means the first sign of a well doing something unexpected is a
// timestep that will not converge, with nothing to connect it to.
//
// Norne converts C-4H from a producer to a gas injector three months in.  That
// is the kind of thing a run should announce when it happens.
//
// Events are graded, because they are not equally interesting.  A well taking
// a new rate target is routine.  A well changing what it *is* -- producer to
// injector, or water injector to gas injector -- changes which equation its
// source term enters and what the fluid around it will do, and is by far the
// most likely reason a step that converged yesterday will not converge today.
// Those are flagged major and the run banners them; the rest are logged.
struct WellEvent {
  std::string text;
  bool major = false;
};

std::vector<WellEvent> well_changes(const ScheduleStep& from,
                                      const ScheduleStep& to);

// ---------------------------------------------------------------------------
// The schedule
// ---------------------------------------------------------------------------
class Schedule {
 public:
  // Parse the SCHEDULE section of a deck (INCLUDE is followed).  Reading may
  // start from the deck itself: everything before SCHEDULE is skipped.
  void read(const std::string& path);

  // Attach the schedule to a grid: map every completion to an active cell and
  // compute the well index where the deck did not supply one.  Must be called
  // before the completions' `cell` and `wi` mean anything.
  void resolve(const CornerPointGrid& grid);

  int n_steps() const { return (int)steps_.size(); }
  const ScheduleStep& step(int n) const { return steps_[n]; }
  const std::vector<ScheduleStep>& steps() const { return steps_; }
  // Writable access, for converting the well indices into another unit system
  // after resolve() has computed them.  Not for editing the schedule itself.
  std::vector<ScheduleStep>& steps_mutable() { return steps_; }

  // Every well name that appears anywhere in the schedule.
  std::vector<std::string> well_names() const;

  double start_day() const { return start_day_; }
  const std::string& start_date() const { return start_date_; }
  Units units() const { return units_; }
  void set_units(Units u) { units_ = u; }

  // Completions that fell in inactive cells, as "WELL i j k" strings.  Never
  // silently discarded: a well losing half its perforations to MINPV is a
  // material fact about the run.
  const std::vector<std::string>& orphan_completions() const { return orphans_; }
  // DRSDT: the largest rate at which dissolved gas may increase, in the deck's
  // own units per day.  Negative means the deck never stated it and Rs is
  // unconstrained -- which is what this simulator did for every deck until
  // now.  Norne states 0, meaning free gas may never redissolve.
  Real drsdt() const { return drsdt_given_ ? drsdt_ : (Real)-1; }
  bool have_drsdt() const { return drsdt_given_; }
  // VAPPARS is mutually exclusive with DRSDT/DRVDT.  Its first exponent
  // reduces saturated Rv as oil saturation falls below its historical high;
  // the second does the analogous Rs treatment (zero in Norne).
  bool have_vappars() const { return vappars_given_; }
  Real vap1() const { return vap1_; }
  Real vap2() const { return vap2_; }
  // First two items of TUNING's first record: initial and maximum internal
  // timestep in days.  They are solver controls, but live in the field deck
  // and therefore must cross the Schedule reader into the driver.
  Real tuning_initial_step() const { return tuning_dt_init_; }
  Real tuning_max_step() const { return tuning_dt_max_; }

  const std::vector<std::string>& ignored_keywords() const { return ignored_; }

  // INCLUDE files that could not be opened.  Not fatal -- a deck ships tables
  // this simulator does not model -- but always reported, so an include that
  // did matter surfaces instead of disappearing.
  const std::vector<std::string>& missing_includes() const { return missing_; }

  std::string summary() const;

 private:
  std::vector<ScheduleStep> steps_;
  std::map<std::string, Well> current_;   // running state as the deck is read
  // Running group state, carried between records the same way the wells
  // are: GCONPROD and GRUPTREE state changes, not a complete picture.
  std::map<std::string, GroupControl> cur_groups_;
  std::map<std::string, GroupInject> cur_ginject_;
  std::map<std::string, std::string> cur_parent_;
  std::vector<std::string> orphans_, ignored_, missing_;
  Units units_ = Units::Metric;
  // DRSDT, in the deck's own gas-per-oil unit per day.  Dimensionless as a
  // RATIO in metric (sm3/sm3) so it converts like Rs itself; the caller does
  // that, because only it knows the deck's system.
  Real drsdt_ = -1;
  bool drsdt_given_ = false;
  Real vap1_ = 0, vap2_ = 0;
  bool vappars_given_ = false;
  Real tuning_dt_init_ = 0, tuning_dt_max_ = 0;
  double start_day_ = 0.0;
  std::string start_date_;
  bool resolved_ = false;

  // WHISTCTL overrides the CMODE on subsequent WCONHIST records.  SHUT is
  // the sentinel for OPM's NONE/undefined state (i.e. use the record CMODE).
  CtrlMode history_override_ = CtrlMode::Shut;

  // Whether the reader has passed the SCHEDULE marker.  This is deliberately
  // a member and not a local: the marker sits in the parent deck while the
  // events sit in an included file, so a flag rebuilt per file would treat
  // the entire well history as pre-SCHEDULE content and read none of it.
  bool in_schedule_ = false;

  void snapshot(double day, const std::string& date);
};

// The Peaceman well index for one completion.
//
//   WI = 2 pi C K_h L / ( ln(r0/rw) + skin )
//
// with the equivalent radius r0 formed from the two directions perpendicular
// to the wellbore.  `dir` selects which two those are, which is the whole
// reason a horizontal well cannot reuse the vertical formula.
Real peaceman_wi(PenetrationDir dir, Real kx, Real ky, Real kz,
                 Real dx, Real dy, Real dz, Real rw, Real skin,
                 Units units, Real* r0_out = nullptr);

// Days between two civil dates, used to place DATES records on a timeline.
double days_from_civil(int y, int m, int d);

}  // namespace nvrs
