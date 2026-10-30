// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// schedule.cpp -- reading the SCHEDULE section, and the well index.
//
// Reading is a fold.  The deck is a stream of events applied to a running set
// of wells; DATES and TSTEP freeze that set into a report step.  Nothing is
// interpreted twice and nothing is looked up by searching backwards, which is
// what keeps a 247-step, 462 kB schedule cheap to load.
//
// Two things here are easy to get quietly wrong.
//
// The first is the well index for a horizontal completion.  Peaceman's formula
// is usually written for a vertical well, and the two permeabilities and two
// cell extents in it are the ones *perpendicular* to the wellbore.  For a well
// running along I those are (ky, kz) and (dy, dz), not (kx, ky) and (dx, dy),
// and the exposed length is dx.  Using the vertical form on a horizontal well
// does not fail; it just returns a productivity that can be wrong by a large
// factor, in whichever direction the anisotropy happens to point.
//
// The second is control switching.  A well carries a requested mode and a set
// of limits, and which one binds is decided each step from what the well
// actually achieved.  See effective_mode() below.
#include "schedule.hpp"

#include "deck_token.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <sstream>

namespace nvrs {
namespace {

constexpr Real kInf = std::numeric_limits<Real>::infinity();

const char* kMonths[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};

int month_number(const std::string& m) {
  for (int i = 0; i < 12; ++i)
    if (m.size() >= 3 && std::toupper((unsigned char)m[0]) == kMonths[i][0] &&
        std::toupper((unsigned char)m[1]) == kMonths[i][1] &&
        std::toupper((unsigned char)m[2]) == kMonths[i][2])
      return i + 1;
  return 0;
}

PenetrationDir parse_dir(const std::string& s) {
  if (s.empty()) return PenetrationDir::Z;
  switch (std::toupper((unsigned char)s[0])) {
    case 'X': case 'I': return PenetrationDir::X;
    case 'Y': case 'J': return PenetrationDir::Y;
    default:            return PenetrationDir::Z;
  }
}

// Keywords that can legitimately begin a SCHEDULE item.  Used to recognise
// where an unmodelled keyword's data ends: the next keyword starting is as
// good a terminator as an empty record, and unlike a per-keyword record count
// it cannot be got wrong one keyword at a time.
//
// Record fields really do collide with these names: RPTSCHED's mnemonic list
// contains a literal 'WELSPECS'.  So callers must ask next_is_quoted() first
// and treat any quoted token as data -- this test alone is not sufficient.
bool is_schedule_keyword(const std::string& t) {
  static const std::set<std::string> k = {
      "DATES", "TSTEP", "WELSPECS", "COMPDAT", "COMPDATL", "WCONPROD",
      "WCONHIST", "WCONINJE", "WCONINJH", "WELOPEN", "WEFAC", "WELTARG",
      "GRUPTREE", "GCONPROD", "GCONINJE", "GECON", "WECON", "GRUPNET",
      // (GRUPTREE and GCONPROD are parsed below; they stay in this set
      //  because it is the list of keywords the tokenizer recognises, not
      //  the list it ignores.)
      "RPTSCHED", "RPTRST", "WRFTPLT", "WTRACER", "WTEST", "WPAVE",
      "WPAVEDEP", "WPIMULT", "WPOLYMER", "WSALT", "WLIST", "WGRUPCON",
      "COMPLUMP", "WELSEGS", "COMPSEGS", "WSEGVALV", "NODEPROP", "BRANPROP",
      "GUIDERAT", "LIFTOPT", "VFPPROD", "VFPINJ", "TUNING", "ZIPPY2",
      "DRVDT", "NEXTSTEP", "WSEGITER", "NETBALAN", "VAPPARS",
      "INCLUDE", "END", "ENDSCALE", "SCHEDULE", "NOECHO", "ECHO", "MESSAGES",
      "SAVE", "RESTART", "SKIPREST", "WHISTCTL", "ACTIONX", "ENDACTIO",
  };
  return k.count(t) > 0;
}

CtrlMode parse_mode(const std::string& s) {
  if (s == "ORAT") return CtrlMode::ORAT;
  if (s == "WRAT") return CtrlMode::WRAT;
  if (s == "GRAT") return CtrlMode::GRAT;
  if (s == "LRAT") return CtrlMode::LRAT;
  if (s == "RESV") return CtrlMode::RESV;
  if (s == "BHP")  return CtrlMode::BHP;
  if (s == "THP")  return CtrlMode::THP;
  if (s == "RATE") return CtrlMode::RATE;
  if (s == "GRUP") return CtrlMode::GRUP;
  return CtrlMode::Shut;
}

}  // namespace

// ---------------------------------------------------------------------------
const char* to_string(CtrlMode m) {
  switch (m) {
    case CtrlMode::Shut: return "SHUT";
    case CtrlMode::ORAT: return "ORAT";
    case CtrlMode::WRAT: return "WRAT";
    case CtrlMode::GRAT: return "GRAT";
    case CtrlMode::LRAT: return "LRAT";
    case CtrlMode::RESV: return "RESV";
    case CtrlMode::BHP:  return "BHP";
    case CtrlMode::THP:  return "THP";
    case CtrlMode::RATE: return "RATE";
    case CtrlMode::GRUP: return "GRUP";
  }
  return "?";
}

const char* to_string(WellType t) {
  return t == WellType::Producer ? "PRODUCER" : "INJECTOR";
}

const char* to_string(InjectedPhase p) {
  switch (p) {
    case InjectedPhase::Water: return "WATER";
    case InjectedPhase::Gas:   return "GAS";
    case InjectedPhase::Oil:   return "OIL";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Civil date arithmetic (Howard Hinnant's days_from_civil).
// ---------------------------------------------------------------------------
double days_from_civil(int y, int m, int d) {
  y -= m <= 2;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153u * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return (double)(era * 146097LL + (long long)doe - 719468LL);
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------
Real WellState::target() const {
  switch (mode) {
    case CtrlMode::ORAT: return orat;
    case CtrlMode::WRAT: return wrat;
    case CtrlMode::GRAT: return grat;
    case CtrlMode::LRAT: return lrat;
    case CtrlMode::RESV: return (type == WellType::Injector) ? resv_rate : resv;
    case CtrlMode::RATE: return surface_rate;
    case CtrlMode::BHP:  return bhp_limit;
    case CtrlMode::THP:  return thp_limit;
    default:             return 0.0;
  }
}

CtrlMode WellState::effective_mode(Real rate, Real bhp) const {
  if (!open || mode == CtrlMode::Shut) return CtrlMode::Shut;

  if (type == WellType::Injector) {
    // An injector is given a rate and a pressure ceiling.  It injects at the
    // rate until the formation will not take it; from then on the ceiling
    // binds and the rate is whatever the reservoir accepts.
    if (mode == CtrlMode::RATE || mode == CtrlMode::RESV) {
      if (bhp_limit > 0 && bhp > bhp_limit) return CtrlMode::BHP;
      return mode;
    }
    // Held on pressure, but not permitted to exceed a rate the deck stated.
    if (mode == CtrlMode::BHP && surface_rate > 0 && rate > surface_rate)
      return CtrlMode::RATE;
    return mode;
  }

  // A producer held on a rate target drops to its pressure floor when it can
  // no longer sustain the rate.  This is the switch that matters most: it is
  // what stops a declining well from being asked for a rate the reservoir
  // stopped being able to deliver years earlier.
  if (mode == CtrlMode::ORAT || mode == CtrlMode::WRAT ||
      mode == CtrlMode::GRAT || mode == CtrlMode::LRAT ||
      mode == CtrlMode::RESV) {
    if (bhp_limit > 0 && bhp < bhp_limit) return CtrlMode::BHP;
    return mode;
  }

  // Held on pressure, capped by whichever rate ceiling the deck gave.  Checked
  // in the order a deck lists them; an absent ceiling sits at infinity and can
  // never bind.
  if (mode == CtrlMode::BHP) {
    if (orat > 0 && orat < kInf && rate > orat) return CtrlMode::ORAT;
    if (lrat > 0 && lrat < kInf && rate > lrat) return CtrlMode::LRAT;
  }
  return mode;
}

const Well* ScheduleStep::find(const std::string& name) const {
  for (const Well& w : wells)
    if (w.name == name) return &w;
  return nullptr;
}

const GroupControl* ScheduleStep::group_target(const std::string& group) const {
  // Nearest ancestor with a usable rate, so a target on FIELD governs a well
  // in a subgroup that states none of its own.  Bounded: a malformed
  // GRUPTREE can name a cycle, and a schedule is not worth hanging over.
  std::string g = group;
  for (int hop = 0; hop < 32 && !g.empty(); ++hop) {
    auto it = groups.find(g);
    if (it != groups.end() && it->second.defined &&
        it->second.mode != CtrlMode::Shut && it->second.target > 0)
      return &it->second;
    auto pit = parent.find(g);
    if (pit == parent.end() || pit->second == g) break;
    g = pit->second;
  }
  return nullptr;
}

const GroupInject* ScheduleStep::group_inject(const std::string& group,
                                              int phase) const {
  std::string g = group;
  for (int hop = 0; hop < 32 && !g.empty(); ++hop) {
    auto it = ginject.find(g + "|" + std::to_string(phase));
    if (it != ginject.end() && it->second.defined &&
        it->second.ctl != GroupInject::Ctl::None && it->second.target > 0)
      return &it->second;
    auto pit = parent.find(g);
    if (pit == parent.end() || pit->second == g) break;
    g = pit->second;
  }
  return nullptr;
}

void convert_group_inject_targets(ScheduleStep& step,
                                  Real liquid_rate_factor,
                                  Real gas_rate_factor) {
  for (auto& kv : step.ginject) {
    GroupInject& g = kv.second;
    if (!std::isfinite((double)g.target)) continue;
    if (g.ctl == GroupInject::Ctl::Rate) {
      g.target *= (g.phase == 2) ? gas_rate_factor : liquid_rate_factor;
    } else if (g.ctl == GroupInject::Ctl::Resv) {
      g.target *= liquid_rate_factor;
    }
  }
}

// ---------------------------------------------------------------------------
std::vector<WellEvent> well_changes(const ScheduleStep& from,
                                    const ScheduleStep& to) {
  std::vector<WellEvent> out;
  char b[256];
  auto major = [&out](const char* s) { out.push_back({s, true}); };
  auto minor = [&out](const char* s) { out.push_back({s, false}); };

  auto rate_of = [](const WellState& s) {
    return (s.type == WellType::Injector) ? s.surface_rate : s.target();
  };
  auto fmt_rate = [](Real q) {
    static char t[48];
    if (q > 0 && std::isfinite((double)q)) std::snprintf(t, sizeof t, "%.4g", (double)q);
    else std::snprintf(t, sizeof t, "-");
    return std::string(t);
  };

  for (const Well& w : to.wells) {
    const Well* p = from.find(w.name);
    const WellState& n = w.state;

    if (!p) {
      if (n.open) {
        std::snprintf(b, sizeof b, "%s: appears, %s on %s", w.name.c_str(),
                      to_string(n.type), to_string(n.mode));
        minor(b);
      }
      continue;
    }
    const WellState& o = p->state;

    // A well changing what it is, which is the event most worth hearing about.
    if (o.type != n.type) {
      if (n.type == WellType::Injector)
        std::snprintf(b, sizeof b, "%s: PRODUCER -> %s INJECTOR, rate %s",
                      w.name.c_str(), to_string(n.inj_phase),
                      fmt_rate(n.surface_rate).c_str());
      else
        std::snprintf(b, sizeof b, "%s: INJECTOR -> PRODUCER on %s",
                      w.name.c_str(), to_string(n.mode));
      major(b);
    } else if (n.type == WellType::Injector && o.inj_phase != n.inj_phase) {
      std::snprintf(b, sizeof b, "%s: %s INJECTOR -> %s INJECTOR, rate %s",
                    w.name.c_str(), to_string(o.inj_phase),
                    to_string(n.inj_phase), fmt_rate(n.surface_rate).c_str());
      major(b);
    }

    if (o.open != n.open) {
      std::snprintf(b, sizeof b, "%s: %s", w.name.c_str(),
                    n.open ? "opened" : "shut in");
      minor(b);
    } else if (o.mode != n.mode && o.type == n.type) {
      std::snprintf(b, sizeof b, "%s: control %s -> %s, target %s",
                    w.name.c_str(), to_string(o.mode), to_string(n.mode),
                    fmt_rate(rate_of(n)).c_str());
      minor(b);
    }

    // A large rate change is worth reporting even when nothing else moved: it
    // is the other way a schedule surprises a solver.
    const Real qo = rate_of(o), qn = rate_of(n);
    if (o.type == n.type && o.mode == n.mode && o.open && n.open &&
        std::isfinite((double)qo) && std::isfinite((double)qn) && qo > 0 &&
        (qn > 4 * qo || qn < 0.25 * qo)) {
      std::snprintf(b, sizeof b, "%s: rate %s -> %s", w.name.c_str(),
                    fmt_rate(qo).c_str(), fmt_rate(qn).c_str());
      minor(b);
    }

    if (p->completions.size() != w.completions.size()) {
      std::snprintf(b, sizeof b, "%s: %d completions -> %d", w.name.c_str(),
                    (int)p->completions.size(), (int)w.completions.size());
      minor(b);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The well index
// ---------------------------------------------------------------------------
Real peaceman_wi(PenetrationDir dir, Real kx, Real ky, Real kz,
                 Real dx, Real dy, Real dz, Real rw, Real skin,
                 Units units, Real* r0_out) {
  // Pick the two directions perpendicular to the wellbore, and the length of
  // cell the well is exposed to along it.  This is the whole difference
  // between a vertical and a horizontal completion.
  Real k1, k2, d1, d2, L;
  switch (dir) {
    case PenetrationDir::X: k1 = ky; k2 = kz; d1 = dy; d2 = dz; L = dx; break;
    case PenetrationDir::Y: k1 = kx; k2 = kz; d1 = dx; d2 = dz; L = dy; break;
    default:                k1 = kx; k2 = ky; d1 = dx; d2 = dy; L = dz; break;
  }
  if (k1 <= 0 || k2 <= 0 || L <= 0 || rw <= 0) return 0.0;

  // Peaceman's equivalent radius for an anisotropic, non-square cell.
  const Real r12 = std::sqrt(k1 / k2), r21 = std::sqrt(k2 / k1);
  const Real num = std::sqrt(r21 * d1 * d1 + r12 * d2 * d2);
  const Real den = std::sqrt(std::sqrt(k2 / k1)) + std::sqrt(std::sqrt(k1 / k2));
  const Real r0 = 0.28 * num / den;
  if (r0_out) *r0_out = r0;
  if (r0 <= rw) return 0.0;

  const Real kh = std::sqrt(k1 * k2) * L;
  const Real denom = std::log(r0 / rw) + skin;
  if (denom <= 0) return 0.0;

  // 2 pi times the Darcy constant of the deck's unit system.
  const Real C = (units == Units::Metric) ? 0.0085267014 : 0.00112712;
  return 2.0 * 3.14159265358979323846 * C * kh / denom;
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------
void Schedule::snapshot(double day, const std::string& date) {
  ScheduleStep s;
  s.day = day;
  s.date = date;
  s.wells.reserve(current_.size());
  for (const auto& kv : current_) s.wells.push_back(kv.second);
  s.groups = cur_groups_;
  s.ginject = cur_ginject_;
  s.parent = cur_parent_;
  steps_.push_back(std::move(s));
}

void Schedule::read(const std::string& path) {
  DeckTokenizer tk(path);

  const std::string dir = [&] {
    const size_t s = path.find_last_of("/\\");
    return (s == std::string::npos) ? std::string(".") : path.substr(0, s);
  }();

  // The running clock.  START sets the origin; DATES place absolute dates on
  // it and TSTEP advances it by a number of days.
  double now = start_day_;
  std::string now_date = start_date_;

  while (!tk.eof()) {
    std::string kw = tk.next();
    if (kw.empty()) break;
    if (!DeckTokenizer::is_keyword(kw)) continue;

    if (kw == "START") {
      const std::vector<std::string> r = tk.read_record();
      if (r.size() >= 3) {
        const int d = std::atoi(r[0].c_str());
        const int m = month_number(r[1]);
        const int y = std::atoi(r[2].c_str());
        if (m) {
          start_day_ = days_from_civil(y, m, d);
          now = start_day_;
          char b[64];
          std::snprintf(b, sizeof b, "%02d %s %04d", d, kMonths[m - 1], y);
          start_date_ = b;
          now_date = start_date_;
        }
      }
      continue;
    }
    if (kw == "METRIC") { units_ = Units::Metric; continue; }
    if (kw == "FIELD")  { units_ = Units::Field;  continue; }

    if (kw == "TUNING") {
      // Eclipse TUNING is three records.  TSINIT and TSMAXZ are the first two
      // items of record one; the remaining nonlinear/linear tolerances are
      // intentionally left to the NVRS run deck.  Read every record so the
      // second and third records cannot be mistaken for new keywords.
      int record = 0;
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        if (record++ == 0) {
          tuning_dt_init_ = DeckTokenizer::num(r, 0, tuning_dt_init_);
          tuning_dt_max_ = DeckTokenizer::num(r, 1, tuning_dt_max_);
        }
      }
      continue;
    }

    if (kw == "SCHEDULE") { in_schedule_ = true; continue; }
    if (kw == "INCLUDE") {
      std::string inc = tk.next();
      tk.skip_to_slash();
      if (inc.rfind("./", 0) == 0) inc = inc.substr(2);
      // Only SCHEDULE-section includes are followed.  The grid and fluid
      // includes hold nothing this reader wants, and chasing them would make
      // loading a schedule depend on files -- an 11 MB corner-point grid, a
      // shelf of PVT tables -- that have no bearing on it.
      if (!in_schedule_) continue;
      // Carry the clock into the include and back out again.
      const double saved_start = start_day_;
      start_day_ = now;
      const std::string saved_date = start_date_;
      start_date_ = now_date;
      try {
        read(dir + "/" + inc);
      } catch (const std::exception&) {
        // An include that will not open is recorded rather than fatal.  A
        // field deck routinely ships tables this simulator does not model --
        // Norne carries three dozen VFP files -- and refusing to load the
        // well schedule because a tubing table is absent would be the wrong
        // trade.  It is never silent: missing_includes() reports every one,
        // so a file that actually mattered shows up rather than vanishing.
        missing_.push_back(inc);
      }
      start_day_ = saved_start;
      start_date_ = saved_date;
      if (!steps_.empty()) { now = steps_.back().day; now_date = steps_.back().date; }
      continue;
    }
    if (kw == "END") break;

    // Everything below is only meaningful inside SCHEDULE.  Before it, the
    // deck is describing the grid and the fluids, which this reader ignores.
    if (!in_schedule_) {
      if (kw == "GRID" || kw == "PROPS" || kw == "REGIONS" ||
          kw == "SOLUTION" || kw == "SUMMARY" || kw == "RUNSPEC" ||
          kw == "EDIT" || kw == "ECHO" || kw == "NOECHO")
        continue;
      if (kw == "TITLE") { tk.skip_title(); continue; }
      // Before SCHEDULE, consume uninteresting input token by token.  Blindly
      // seeking '/' is unsafe: phase/output flags and SUMMARY mnemonics have
      // no terminator and would swallow SCHEDULE plus its first well record.
      continue;
    }

    if (kw == "GRUPTREE") {
      // child parent /  -- the hierarchy a group target propagates down.
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        const std::string child = DeckTokenizer::text(r, 0, "");
        const std::string par = DeckTokenizer::text(r, 1, "FIELD");
        if (!child.empty()) cur_parent_[child] = par;
      }
      continue;
    }

    if (kw == "GCONPROD") {
      // group  mode  orat  wrat  grat  lrat  ...
      //
      // The mode names which of the four rates is the target, and the others
      // are ceilings.  NONE and FLD mean the group states no target of its
      // own -- FLD defers to its parent, which the tree walk handles by
      // simply not finding a target here.
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        const std::string gname = DeckTokenizer::text(r, 0, "");
        if (gname.empty()) continue;
        const std::string mode = DeckTokenizer::text(r, 1, "NONE");
        GroupControl g;
        g.name = gname;
        g.defined = true;
        const Real orat = DeckTokenizer::num(r, 2, 0.0);
        const Real wrat = DeckTokenizer::num(r, 3, 0.0);
        const Real grat = DeckTokenizer::num(r, 4, 0.0);
        const Real lrat = DeckTokenizer::num(r, 5, 0.0);
        if (mode == "ORAT")      { g.mode = CtrlMode::ORAT; g.target = orat; }
        else if (mode == "WRAT") { g.mode = CtrlMode::WRAT; g.target = wrat; }
        else if (mode == "GRAT") { g.mode = CtrlMode::GRAT; g.target = grat; }
        else if (mode == "LRAT") { g.mode = CtrlMode::LRAT; g.target = lrat; }
        else if (mode == "RESV") { g.mode = CtrlMode::RESV;
                                   g.target = DeckTokenizer::num(r, 6, 0.0); }
        else                     { g.mode = CtrlMode::Shut; g.target = 0; }
        if (!std::isfinite((double)g.target) || g.target <= 0)
          g.mode = CtrlMode::Shut;      // stated no usable rate
        cur_groups_[gname] = g;
      }
      continue;
    }

    if (kw == "GCONINJE") {
      // group  phase  control  surface-rate  reservoir-rate  reinj-frac
      //        voidage-frac  ...
      //
      // The control names which of the rates is the target.  RATE is a
      // surface rate and RESV a reservoir one; REIN and VREP are fractions of
      // what the group PRODUCED, which needs the previous step's production
      // and is not implemented -- such a group keeps its wells' own rates.
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        const std::string gname = DeckTokenizer::text(r, 0, "");
        if (gname.empty()) continue;
        const std::string ph = DeckTokenizer::text(r, 1, "");
        GroupInject g;
        g.name = gname;
        g.defined = true;
        if (ph == "WATER" || ph == "WAT") g.phase = 1;
        else if (ph == "GAS")             g.phase = 2;
        else if (ph == "OIL")             g.phase = 3;
        else                              continue;   // no phase, no record
        const std::string ctl = DeckTokenizer::text(r, 2, "NONE");
        if (ctl == "RATE") {
          g.ctl = GroupInject::Ctl::Rate;
          g.target = DeckTokenizer::num(r, 3, 0.0);
        } else if (ctl == "RESV") {
          g.ctl = GroupInject::Ctl::Resv;
          g.target = DeckTokenizer::num(r, 4, 0.0);
        } else if (ctl == "REIN") {
          g.ctl = GroupInject::Ctl::Rein;
          g.target = DeckTokenizer::num(r, 5, 0.0);
        } else if (ctl == "VREP") {
          g.ctl = GroupInject::Ctl::Vrep;
          g.target = DeckTokenizer::num(r, 6, 0.0);
        } else {
          g.ctl = GroupInject::Ctl::None;
        }
        if (!std::isfinite((double)g.target) || g.target <= 0)
          g.ctl = GroupInject::Ctl::None;
        cur_ginject_[gname + "|" + std::to_string(g.phase)] = g;
      }
      continue;
    }

    if (kw == "WELSPECS") {
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        const std::string name = r[0];
        Well& w = current_[name];
        w.name = name;
        w.group = DeckTokenizer::text(r, 1, w.group);
        w.head_i = DeckTokenizer::integer(r, 2, w.head_i + 1) - 1;
        w.head_j = DeckTokenizer::integer(r, 3, w.head_j + 1) - 1;
        w.ref_depth = DeckTokenizer::num(r, 4, -1.0);
        w.preferred_phase = DeckTokenizer::text(r, 5, "OIL");
        w.state.defined = true;
        if (w.state.mode == CtrlMode::Shut && !w.state.open) {
          // Newly declared and not yet given a control: closed until one
          // arrives, which is what a deck means by declaring it early.
          w.state.type = (w.preferred_phase == "WATER" ||
                          w.preferred_phase == "GAS")
                             ? WellType::Injector
                             : WellType::Producer;
        }
      }

    } else if (kw == "COMPDAT") {
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        const std::string name = r[0];
        auto it = current_.find(name);
        if (it == current_.end()) {
          ignored_.push_back("COMPDAT for undeclared well " + name);
          continue;
        }
        Well& w = it->second;

        // A defaulted I or J means "the well head", which is how a deck writes
        // a vertical completion under the platform slot.
        const int ci = DeckTokenizer::integer(r, 1, w.head_i + 1) - 1;
        const int cj = DeckTokenizer::integer(r, 2, w.head_j + 1) - 1;
        const int k1 = DeckTokenizer::integer(r, 3, 1) - 1;
        const int k2 = DeckTokenizer::integer(r, 4, k1 + 1) - 1;
        const std::string status = DeckTokenizer::text(r, 5, "OPEN");
        const int satnum = DeckTokenizer::integer(r, 6, -1);
        const Real cf = DeckTokenizer::num(r, 7, -1.0);
        const Real diam = DeckTokenizer::num(r, 8, 0.0);
        const Real kh = DeckTokenizer::num(r, 9, -1.0);
        const Real skin = DeckTokenizer::num(r, 10, 0.0);
        // r[11] is the D-factor, which this simulator does not model.
        const PenetrationDir pdir = parse_dir(DeckTokenizer::text(r, 12, "Z"));
        const Real r0 = DeckTokenizer::num(r, 13, -1.0);

        for (int k = k1; k <= k2; ++k) {
          // A repeated COMPDAT for the same cell replaces the earlier one --
          // that is how a deck re-perforates or changes a completion's status
          // partway through a history.
          Completion c;
          c.i = ci; c.j = cj; c.k = k;
          c.open = (status == "OPEN");
          c.dir = pdir;
          c.cf = cf;
          c.diameter = diam;
          c.kh = kh;
          c.skin = skin;
          c.r0 = r0;
          c.sat_table = satnum;

          auto ex = std::find_if(
              w.completions.begin(), w.completions.end(),
              [&](const Completion& o) {
                return o.i == c.i && o.j == c.j && o.k == c.k;
              });
          if (ex == w.completions.end()) w.completions.push_back(c);
          else *ex = c;
        }
      }

    } else if (kw == "DRSDT") {
      // The maximum rate at which dissolved gas may INCREASE, in the deck's
      // gas-per-oil unit per day.  Norne states 0: free gas may never go back
      // into solution.  That is not a detail on a field that reinjects 294 G
      // scf into its own gas cap -- every pressure recovery would otherwise
      // redissolve gas the deck says must stay free.
      //
      // One record, one number; a defaulted or absent value means no limit,
      // which is the behaviour of a deck that never states it.
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        drsdt_ = DeckTokenizer::num(r, 0, -1.0);
        drsdt_given_ = true;
        vappars_given_ = false; // mixing-rate controls are mutually exclusive
      }

    } else if (kw == "VAPPARS") {
      // Eclipse/OPM: VAPPARS replaces any earlier DRSDT/DRVDT control.  Norne
      // states DRSDT 0 in the parent deck, then VAPPARS 0.5 0.0 in the first
      // schedule include, before the first date; retaining both is wrong.
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        vap1_ = DeckTokenizer::num(r, 0, 0.0);
        vap2_ = DeckTokenizer::num(r, 1, 0.0);
        vappars_given_ = true;
        drsdt_given_ = false;
        drsdt_ = -1;
      }

    } else if (kw == "WCONPROD") {
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        auto it = current_.find(r[0]);
        if (it == current_.end()) { ignored_.push_back("WCONPROD " + r[0]); continue; }
        WellState& s = it->second.state;
        s.type = WellType::Producer;
        s.historical = false;
        s.open = (DeckTokenizer::text(r, 1, "OPEN") == "OPEN");
        s.mode = parse_mode(DeckTokenizer::text(r, 2, "BHP"));
        // Absent rate ceilings sit at infinity so they can never bind.
        s.orat = DeckTokenizer::num(r, 3, kInf);
        s.wrat = DeckTokenizer::num(r, 4, kInf);
        s.grat = DeckTokenizer::num(r, 5, kInf);
        s.lrat = DeckTokenizer::num(r, 6, kInf);
        s.resv = DeckTokenizer::num(r, 7, kInf);
        s.bhp_limit = DeckTokenizer::num(r, 8, 0.0);
        s.thp_limit = DeckTokenizer::num(r, 9, 0.0);
        s.vfp_table = DeckTokenizer::integer(r, 10, -1);
        // Status and control mode are independent in Eclipse/OPM.  Keep the
        // requested mode even while the well is shut so a later WELOPEN OPEN
        // can resume it without waiting for another WCON record.
      }

    } else if (kw == "WCONHIST") {
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        auto it = current_.find(r[0]);
        if (it == current_.end()) { ignored_.push_back("WCONHIST " + r[0]); continue; }
        WellState& s = it->second.state;
        // A well arriving here may have been an INJECTOR until this record,
        // and an injector's bhp_limit is a CEILING -- 600 bar on Norne's gas
        // injectors.  WCONPROD and WCONINJE both rewrite the field on every
        // record, so they cannot carry one convention into the other; this
        // branch assigned it only when the record states an observed BHP, so
        // a converted well kept its old ceiling and the producer branch of
        // apply_schedule_step then used 8702 psi as its FLOWING pressure --
        // against 3191 psi of reservoir.  Drawdown came out at -5511 and
        // every such "producer" ran as a large injector: the step-49 gas
        // residual of -3.66e7 per cell is exactly Rs.b_o.WI.lambda_o.dp at
        // that drawdown.  Clear it, so a producer starts with no floor and
        // takes one only from this record or from PRODBHP.
        const bool was_injector = (s.type == WellType::Injector);
        const bool entering_history = !s.historical;
        // OPM keeps WCONHIST's observed BHP (BHPH) separate from the producer
        // BHP constraint.  On entry to history mode the constraint is reset
        // to FBHPDEF; NVRS represents that default by zero here and applies
        // the deck-wide PRODBHP fallback when controls are assembled.
        if (was_injector || entering_history || s.mode == CtrlMode::BHP)
          s.bhp_limit = 0;
        s.type = WellType::Producer;
        s.historical = true;
        s.open = (DeckTokenizer::text(r, 1, "OPEN") == "OPEN");
        const CtrlMode record_mode =
            parse_mode(DeckTokenizer::text(r, 2, "ORAT"));
        s.mode = (history_override_ != CtrlMode::Shut)
                     ? history_override_
                     : record_mode;
        // WCONHIST states what the well *did*, not what it is asked to do.
        // Keeping these apart from the targets is what stops an observed rate
        // being fed back in as a control request.
        s.hist_orat = DeckTokenizer::num(r, 3, 0.0);
        s.hist_wrat = DeckTokenizer::num(r, 4, 0.0);
        s.hist_grat = DeckTokenizer::num(r, 5, 0.0);
        s.vfp_table = DeckTokenizer::integer(r, 6, s.vfp_table);
        s.hist_thp  = DeckTokenizer::num(r, 8, 0.0);
        s.hist_bhp  = DeckTokenizer::num(r, 9, 0.0);
        // The observed rate is also the target when running in history mode.
        s.orat = s.hist_orat;
        s.wrat = s.hist_wrat;
        s.grat = s.hist_grat;
        s.lrat = s.hist_orat + s.hist_wrat;
        // OPM has no fixed RESV value for a history producer; it derives the
        // target from these surface rates at the current reservoir state.
        s.resv = 0;
        // BHPH is normally only an observation (WBHPH).  OPM uses it as the
        // limit only when BHP itself is the selected history control.
        if (s.mode == CtrlMode::BHP && s.hist_bhp > 0)
          s.bhp_limit = s.hist_bhp;
        // Preserve CMODE while shut; WELOPEN may reopen this history control.
      }

    } else if (kw == "WCONINJE" || kw == "WCONINJH") {
      const bool hist = (kw == "WCONINJH");
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        auto it = current_.find(r[0]);
        if (it == current_.end()) { ignored_.push_back(kw + " " + r[0]); continue; }
        WellState& s = it->second.state;
        s.type = WellType::Injector;
        s.historical = hist;
        const std::string phase = DeckTokenizer::text(r, 1, "WATER");
        s.inj_phase = (phase == "GAS")   ? InjectedPhase::Gas
                    : (phase == "OIL")   ? InjectedPhase::Oil
                                         : InjectedPhase::Water;
        s.open = (DeckTokenizer::text(r, 2, "OPEN") == "OPEN");
        if (hist) {
          // WCONINJH gives the observed rate; control is by rate.
          s.mode = CtrlMode::RATE;
          s.surface_rate = DeckTokenizer::num(r, 3, 0.0);
          s.bhp_limit = DeckTokenizer::num(r, 4, kInf);
        } else {
          s.mode = parse_mode(DeckTokenizer::text(r, 3, "RATE"));
          s.surface_rate = DeckTokenizer::num(r, 4, kInf);
          s.resv_rate = DeckTokenizer::num(r, 5, kInf);
          // An injector's BHP limit is a ceiling, so an absent one is infinite.
          s.bhp_limit = DeckTokenizer::num(r, 6, kInf);
          s.thp_limit = DeckTokenizer::num(r, 7, 0.0);
          s.vfp_table = DeckTokenizer::integer(r, 8, -1);
        }
        // Preserve the injector control while shut.  Its next state change may
        // be WELOPEN OPEN rather than another WCONINJE record.
      }

    } else if (kw == "WELOPEN") {
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        auto it = current_.find(r[0]);
        if (it == current_.end()) { ignored_.push_back("WELOPEN " + r[0]); continue; }
        Well& w = it->second;
        const std::string status = DeckTokenizer::text(r, 1, "OPEN");
        const bool selectors_defaulted =
            (r.size() <= 2 || DeckTokenizer::is_default(r[2])) &&
            (r.size() <= 3 || DeckTokenizer::is_default(r[3])) &&
            (r.size() <= 4 || DeckTokenizer::is_default(r[4]));
        const int ci = DeckTokenizer::integer(r, 2, 0);
        const int cj = DeckTokenizer::integer(r, 3, 0);
        const int ck = DeckTokenizer::integer(r, 4, 0);
        const bool opening = (status == "OPEN");

        if (selectors_defaulted) {
          // Whole-well form.  Shutting is unambiguous; re-opening restores the
          // well to whatever control it last held, which is why the mode is
          // not cleared when it shuts.  `open` alone gates assembly/reporting.
          w.state.open = opening;
        } else {
          // Explicit zero is a wildcard completion selector, not a defaulted
          // whole-well item.  Norne uses `SHUT 0 0 0 2*` for a squeeze, then
          // reintroduces only selected cells with later COMPDAT records.  If
          // zero were folded into the whole-well form, those old water-zone
          // completions would silently reopen with the next WCONHIST record.
          for (Completion& c : w.completions) {
            if (ci > 0 && c.i != ci - 1) continue;
            if (cj > 0 && c.j != cj - 1) continue;
            if (ck > 0 && c.k != ck - 1) continue;
            c.open = opening;
          }
        }
      }

    } else if (kw == "WEFAC") {
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        auto it = current_.find(r[0]);
        if (it == current_.end()) continue;
        it->second.state.efficiency = DeckTokenizer::num(r, 1, 1.0);
      }

    } else if (kw == "WHISTCTL") {
      // OPM accepts ORAT/WRAT/GRAT/LRAT/RESV/BHP as effective history
      // controls.  NONE cancels the override.  The second record item is the
      // optional terminate-on-BHP flag and does not alter the control mode.
      const std::vector<std::string> r = tk.read_record();
      const CtrlMode m = parse_mode(DeckTokenizer::text(r, 0, "NONE"));
      switch (m) {
        case CtrlMode::ORAT: case CtrlMode::WRAT: case CtrlMode::GRAT:
        case CtrlMode::LRAT: case CtrlMode::RESV: case CtrlMode::BHP:
          history_override_ = m;
          break;
        default:
          history_override_ = CtrlMode::Shut;
          break;
      }

    } else if (kw == "DATES") {
      for (;;) {
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        if (r.size() < 3) continue;
        const int d = std::atoi(r[0].c_str());
        const int m = month_number(r[1]);
        const int y = std::atoi(r[2].c_str());
        if (!m) continue;
        now = days_from_civil(y, m, d);
        char b[64];
        std::snprintf(b, sizeof b, "%02d %s %04d", d, kMonths[m - 1], y);
        now_date = b;
        snapshot(now, now_date);
      }

    } else if (kw == "TSTEP") {
      std::vector<double> v;
      tk.read_values(v);
      for (double dt : v) {
        now += dt;
        // Render the reached date so a TSTEP-driven deck reports like a
        // DATES-driven one.
        const long long z = (long long)std::floor(now) + 719468;
        const long long era = (z >= 0 ? z : z - 146096) / 146097;
        const unsigned doe = (unsigned)(z - era * 146097);
        const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        const int yy = (int)(yoe) + (int)(era * 400);
        const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const unsigned mp = (5 * doy + 2) / 153;
        const unsigned dd = doy - (153 * mp + 2) / 5 + 1;
        const unsigned mm = mp + (mp < 10 ? 3 : -9);
        char b[64];
        std::snprintf(b, sizeof b, "%02u %s %04d", dd, kMonths[mm - 1],
                      yy + (mm <= 2));
        now_date = b;
        snapshot(now, now_date);
      }

    } else {
      // A keyword this reader does not model.  Its data still has to be
      // consumed, or the stream falls out of step and the *next* keyword is
      // read as though it were data.
      //
      // The number of records is not something that can be assumed.  Some of
      // these keywords take exactly one record and no terminator (RPTSCHED is
      // the common one, and Norne has seventy-two of them); others take a
      // slash-terminated list.  Skipping a single-record keyword as though it
      // were a list swallows whatever follows it -- which here meant eating
      // eighty DATES records and silently shortening a nine-year history to
      // six, with no error anywhere.
      //
      // So rather than keep a table of record counts per keyword and get one
      // of them wrong, stop on whichever comes first: the empty terminating
      // record, or the point where the next token is itself a keyword.
      for (;;) {
        if (tk.at_slash()) { tk.eat_slash(); break; }
        if (tk.eof()) break;
        // A quoted token is a data field, never a keyword -- RPTSCHED's
        // mnemonic list literally contains 'WELSPECS'.
        if (!tk.next_is_quoted() && is_schedule_keyword(tk.peek())) break;
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
      }
      ignored_.push_back(kw);
    }
  }

  // A deck that never wrote a DATES record still describes one state; without
  // this a TSTEP-free schedule would load as zero steps and run nothing.
  if (steps_.empty() && !current_.empty()) snapshot(now, now_date);
}

// ---------------------------------------------------------------------------
// Resolving completions against the grid
// ---------------------------------------------------------------------------
void Schedule::resolve(const CornerPointGrid& grid) {
  orphans_.clear();
  std::set<std::string> seen_orphans;

  for (ScheduleStep& s : steps_)
    for (Well& w : s.wells)
      for (Completion& c : w.completions) {
        c.cell = -1;
        c.wi = 0.0;
        if (c.i < 0 || c.i >= grid.nx || c.j < 0 || c.j >= grid.ny ||
            c.k < 0 || c.k >= grid.nz)
          continue;
        const int g = grid.global_index(c.i, c.j, c.k);
        const int a = grid.active_of_global(g);
        if (a < 0) {
          // The cell was removed by ACTNUM or MINPV.  Common, and worth
          // reporting: a well losing perforations changes what it can produce.
          char b[128];
          std::snprintf(b, sizeof b, "%s %d %d %d", w.name.c_str(),
                        c.i + 1, c.j + 1, c.k + 1);
          if (seen_orphans.insert(b).second) orphans_.push_back(b);
          continue;
        }
        c.cell = a;

        if (c.cf >= 0) {
          // The deck stated the connection factor; use it rather than
          // recomputing, because it usually encodes a completion model this
          // simulator does not have.
          c.wi = c.cf;
        } else {
          const Real rw = (c.diameter > 0) ? 0.5 * c.diameter : 0.1;
          c.wi = peaceman_wi(c.dir, grid.permx[a], grid.permy[a], grid.permz[a],
                             grid.dx[a], grid.dy[a], grid.dz[a], rw, c.skin,
                             grid.units());
        }
      }
  resolved_ = true;
}

std::vector<std::string> Schedule::well_names() const {
  std::set<std::string> n;
  for (const ScheduleStep& s : steps_)
    for (const Well& w : s.wells) n.insert(w.name);
  return {n.begin(), n.end()};
}

std::string Schedule::summary() const {
  std::ostringstream os;
  char b[256];

  const std::vector<std::string> names = well_names();

  // Count wells and completions as of the last step, and classify by the
  // direction their completions run: a deck's mix of vertical and horizontal
  // wells is the first thing to check after reading one.
  int n_prod = 0, n_inj = 0, n_comp = 0, n_horiz = 0, n_vert = 0, n_open = 0;
  if (!steps_.empty()) {
    for (const Well& w : steps_.back().wells) {
      if (w.state.type == WellType::Injector) ++n_inj; else ++n_prod;
      if (w.state.open) ++n_open;
      bool horizontal = false;
      for (const Completion& c : w.completions) {
        ++n_comp;
        if (c.dir != PenetrationDir::Z) horizontal = true;
      }
      if (horizontal) ++n_horiz; else ++n_vert;
    }
  }

  os << "Schedule\n";
  std::snprintf(b, sizeof b, "  start           : %s\n",
                start_date_.empty() ? "(none)" : start_date_.c_str()); os << b;
  if (!steps_.empty()) {
    std::snprintf(b, sizeof b, "  report steps    : %d, %s .. %s (%.0f days)\n",
                  (int)steps_.size(), steps_.front().date.c_str(),
                  steps_.back().date.c_str(),
                  steps_.back().day - steps_.front().day); os << b;
  }
  std::snprintf(b, sizeof b, "  wells           : %d  (%d producers, %d injectors, "
                "%d open at the end)\n",
                (int)names.size(), n_prod, n_inj, n_open); os << b;
  std::snprintf(b, sizeof b, "  trajectories    : %d with horizontal sections, "
                "%d purely vertical\n", n_horiz, n_vert); os << b;
  std::snprintf(b, sizeof b, "  completions     : %d at the final step\n", n_comp);
  os << b;

  if (resolved_) {
    int resolved = 0, deck_cf = 0, computed = 0;
    if (!steps_.empty())
      for (const Well& w : steps_.back().wells)
        for (const Completion& c : w.completions) {
          if (c.cell < 0) continue;
          ++resolved;
          if (c.cf >= 0) ++deck_cf; else ++computed;
        }
    std::snprintf(b, sizeof b, "  resolved        : %d in active cells "
                  "(%d used the deck's factor, %d computed by Peaceman)\n",
                  resolved, deck_cf, computed); os << b;
    if (!orphans_.empty()) {
      std::snprintf(b, sizeof b, "  in dead cells   : %d completions "
                    "(ACTNUM or MINPV removed the cell)\n",
                    (int)orphans_.size()); os << b;
    }
  }

  if (!missing_.empty()) {
    std::snprintf(b, sizeof b, "  missing includes: %d not found "
                  "(tables this simulator does not model)\n",
                  (int)missing_.size()); os << b;
  }

  if (!ignored_.empty()) {
    std::set<std::string> uniq(ignored_.begin(), ignored_.end());
    os << "  not implemented : ";
    int n = 0;
    for (const std::string& s : uniq) {
      if (n++) os << ", ";
      os << s;
      if (n == 12) { os << ", ... (" << (int)uniq.size() - 12 << " more)"; break; }
    }
    os << "\n";
  }
  return os.str();
}

}  // namespace nvrs
