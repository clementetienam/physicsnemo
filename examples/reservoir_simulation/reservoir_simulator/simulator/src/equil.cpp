// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// equil.cpp -- equilibrium initialisation from a datum, by region.
//
// The model
// ---------
// Within one equilibration region, each phase has its own hydrostatic column.
// The oil column is anchored at the datum, where the deck states the pressure:
//
//     Po(z) = Pdatum + grad_oil * (z - z_datum)
//
// The other two are anchored at the contacts, where the deck states the
// capillary pressure that separates them from the oil:
//
//     Pw(z) = Po(z_owc) + Pc_owc + grad_water * (z - z_owc)
//     Pg(z) = Po(z_goc) + Pc_goc + grad_gas   * (z - z_goc)
//
// Anchoring at the contacts rather than at the datum is what makes the state
// an equilibrium: it guarantees the phase pressures agree at the contact to
// within exactly the stated capillary pressure, whatever the gradients are and
// wherever the datum happens to sit.  Anchoring all three at the datum would
// look almost identical on a plot and would put a pressure discontinuity at
// every contact, which the first timestep would then try to flatten.
//
// Saturations follow the contacts: gas above the GOC, oil between the
// contacts, water below the WOC.  This is the sharp-interface initialisation.
// A deck with capillary-pressure saturation functions would instead spread a
// transition zone across each contact; doing that needs the saturation tables,
// which this module deliberately does not read.  The sharp version is a
// defensible starting state and, importantly, an honest one: what it omits is
// visible rather than approximated.
#include "equil.hpp"

#include "deck_token.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nvrs {

Real gravity_constant(Units u) {
  // METRIC: rho [kg/m^3] * 9.80665 [m/s^2] = Pa/m, and 1 bar = 1e5 Pa.
  // FIELD : rho [lb/ft^3] / 144 [in^2/ft^2] = psi/ft.
  return (u == Units::Metric) ? 9.80665e-5 : (1.0 / 144.0);
}

// ---------------------------------------------------------------------------
void Equilibration::set_gradients(Real oil, Real water, Real gas) {
  grad_oil_ = oil;
  grad_water_ = water;
  grad_gas_ = gas;
  have_grad_ = true;
  grad_from_fvf_ = false;
}

void Equilibration::set_fvf(Real bo, Real bw, Real bg, Real rs) {
  bo_ = bo; bw_ = bw; bg_ = bg; rs_ = rs;
  if (densities_.empty())
    throw std::runtime_error(
        "set_fvf: the deck gave no DENSITY to convert");
  const SurfaceDensity& d = densities_.front();
  const Real g = gravity_constant(units_);
  if (bo <= 0 || bw <= 0 || bg <= 0)
    throw std::runtime_error("set_fvf: formation volume factors must be positive");

  // Reservoir oil carries its dissolved gas, which is a material part of its
  // density: at Norne's Rs the gas contributes about a tenth of it.
  grad_oil_ = ((d.oil + rs * d.gas) / bo) * g;
  grad_water_ = (d.water / bw) * g;
  grad_gas_ = (d.gas / bg) * g;
  have_grad_ = true;
  grad_from_fvf_ = true;
}

const SurfaceDensity& Equilibration::density(int pvt_region) const {
  static const SurfaceDensity kNone;
  if (densities_.empty()) return kNone;
  if (pvt_region < 0 || pvt_region >= (int)densities_.size())
    return densities_.front();
  return densities_[pvt_region];
}

// ---------------------------------------------------------------------------
Real Equilibration::oil_pressure(int region, Real depth) const {
  const EquilRecord& e = records_[region];
  return e.datum_pressure + grad_oil_ * (depth - e.datum_depth);
}

Real Equilibration::water_pressure(int region, Real depth) const {
  const EquilRecord& e = records_[region];
  const Real po_at_owc = oil_pressure(region, e.owc_depth);
  return po_at_owc + e.pc_owc + grad_water_ * (depth - e.owc_depth);
}

Real Equilibration::gas_pressure(int region, Real depth) const {
  const EquilRecord& e = records_[region];
  const Real po_at_goc = oil_pressure(region, e.goc_depth);
  return po_at_goc + e.pc_goc + grad_gas_ * (depth - e.goc_depth);
}

// ---------------------------------------------------------------------------
void Equilibration::read(const std::string& path) {
  DeckTokenizer tk(path);

  const std::string dir = [&] {
    const size_t s = path.find_last_of("/\\");
    return (s == std::string::npos) ? std::string(".") : path.substr(0, s);
  }();

  while (!tk.eof()) {
    std::string kw = tk.next();
    if (kw.empty()) break;
    if (!DeckTokenizer::is_keyword(kw)) continue;
    if (kw == "SCHEDULE" || kw == "END") break;

    if (kw == "METRIC") { units_ = Units::Metric; continue; }
    if (kw == "FIELD")  { units_ = Units::Field;  continue; }

    if (kw == "EQUIL") {
      // EQUIL takes NTEQUL records and stops -- no empty terminating record --
      // so the list ends where the next keyword begins.  See at_records_end().
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        EquilRecord e;
        e.datum_depth    = DeckTokenizer::num(r, 0, 0.0);
        e.datum_pressure = DeckTokenizer::num(r, 1, 0.0);
        e.owc_depth      = DeckTokenizer::num(r, 2, 0.0);
        e.pc_owc         = DeckTokenizer::num(r, 3, 0.0);
        e.goc_depth      = DeckTokenizer::num(r, 4, 0.0);
        e.pc_goc         = DeckTokenizer::num(r, 5, 0.0);
        e.rs_method      = DeckTokenizer::integer(r, 6, 0);
        e.rv_method      = DeckTokenizer::integer(r, 7, 0);
        e.accuracy       = DeckTokenizer::integer(r, 8, 0);
        records_.push_back(e);
      }

    } else if (kw == "RSVD") {
      // One table per equilibration region.  Each table is ONE record -- a
      // flat run of depth/Rs pairs terminated by a slash -- not one record per
      // row, which is the trap:
      //
      //     RSVD
      //     2582.0  120.25
      //     2597.0  110.00
      //     2660.7  106.77
      //     2697.0  106.77 /      <- all four rows are this region's record
      //     2500    94.5000
      //     2510.0  94.3
      //     2590.0  94.1 /        <- the next region
      //
      // Reading a pair per record gives one table of the first two numbers and
      // silently discards the depth dependence and every other region, which
      // is how this first came out as a single flat 675 scf/stb over the whole
      // of Norne -- exactly the constant it was written to replace.
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        if (tk.eof()) break;
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        RsvdTable t;
        for (size_t i2 = 0; i2 + 1 < r.size(); i2 += 2) {
          const Real z = DeckTokenizer::num(r, (int)i2, 0.0);
          const Real rs = DeckTokenizer::num(r, (int)i2 + 1, 0.0);
          if (!t.depth.empty() && z < t.depth.back()) continue;  // keep ascending
          t.depth.push_back(z);
          t.rs.push_back(rs);
        }
        if (!t.depth.empty()) rsvd_.push_back(std::move(t));
      }

    } else if (kw == "DENSITY") {
      // DENSITY takes NTPVT records, likewise unterminated.
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
        SurfaceDensity d;
        d.oil   = DeckTokenizer::num(r, 0, 0.0);
        d.water = DeckTokenizer::num(r, 1, 0.0);
        d.gas   = DeckTokenizer::num(r, 2, 0.0);
        densities_.push_back(d);
      }

    } else if (kw == "EQLNUM") {
      std::vector<double> v;
      tk.read_values(v);
      eqlnum_.assign(v.begin(), v.end());

    } else if (kw == "INCLUDE") {
      std::string inc = tk.next();
      tk.skip_to_slash();
      if (inc.rfind("./", 0) == 0) inc = inc.substr(2);
      try {
        read(dir + "/" + inc);
      } catch (const std::exception&) {
        ignored_.push_back("missing include " + inc);
      }

    } else if (kw == "ECHO" || kw == "NOECHO" || kw == "RUNSPEC" ||
               kw == "GRID" || kw == "EDIT" || kw == "PROPS" ||
               kw == "REGIONS" || kw == "SOLUTION" || kw == "OIL" ||
               kw == "WATER" || kw == "GAS" || kw == "DISGAS" ||
               kw == "VAPOIL") {
      // Section markers and flags: no data, no terminator.

    } else {
      // Skip this keyword's data.  Same rule as the schedule reader: stop at
      // the empty terminating record or where the next keyword begins, so a
      // single-record keyword does not swallow what follows it.
      for (;;) {
        if (tk.at_records_end()) { tk.eat_slash(); break; }
        const std::vector<std::string> r = tk.read_record();
        if (r.empty()) break;
      }
      ignored_.push_back(kw);
    }
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The undersaturated Rs anchor, per active cell
// ---------------------------------------------------------------------------
// Rs is not a primary unknown in this formulation, so the model needs a value
// for the dissolved gas an oil RETAINS where there is no free gas to saturate
// it.  A deck-level scalar puts one number over the whole field; RSVD is what
// the deck actually says, and on Norne the five tables span 675 to 531 scf/stb.
//
// Depth is read from the grid in whatever length unit the grid currently
// holds, and the RSVD depths are in the deck's own unit, so this must be
// called while the two still agree -- before to_field_units() moves the grid
// to feet.  `to_field_rs` scales the tabulated dissolved gas into the solver's
// scf/stb and is the caller's business, because only it knows whether the deck
// was metric.
std::vector<Real> Equilibration::rs_anchor(const CornerPointGrid& grid,
                                           Real to_field_rs) const {
  std::vector<Real> out;
  if (!have_rsvd()) return out;                 // caller keeps its scalar

  const int na = grid.n_active();
  out.assign((size_t)na, (Real)0);

  // Same region resolution as initialise(): EQLNUM if the deck gave one,
  // otherwise every cell is in region 1.
  const std::vector<Real>* eq =
      eqlnum_.empty() ? grid.array("EQLNUM") : &eqlnum_;

  for (int a = 0; a < na; ++a) {
    const int g = grid.global_of_active(a);
    int r = 0;
    if (eq && (size_t)g < eq->size()) r = (int)(*eq)[(size_t)g] - 1;
    if (r < 0 || r >= (int)rsvd_.size()) r = 0;
    // A region whose own table is empty falls back to the first non-empty one
    // rather than to zero: no dissolved gas at all is a far worse answer than
    // a neighbouring region's.
    const RsvdTable* t = &rsvd_[(size_t)r];
    if (t->empty())
      for (const RsvdTable& c : rsvd_) if (!c.empty()) { t = &c; break; }
    out[(size_t)a] = t->at(grid.depth[a]) * to_field_rs;
  }
  return out;
}

InitialState Equilibration::initialise(const CornerPointGrid& grid,
                                       Real default_swl) const {
  if (records_.empty())
    throw std::runtime_error("no EQUIL records were read");
  if (!have_grad_)
    throw std::runtime_error(
        "no phase gradients: call set_gradients() or set_fvf() first");

  // Connate water, per cell where the deck gave an SWL array and from the
  // caller's default otherwise.  It varies by layer in most field models --
  // Norne's ranges from 0.04 to 0.16 across its twenty-two layers.
  const std::vector<Real>* swl = grid.array("SWL");

  // SWATINIT: the deck's own initial water saturation, cell by cell.
  //
  // This matters as soon as the saturation tables carry a capillary pressure,
  // which Norne's do -- Pcow runs to 54.5 psi.  The contacts built below are
  // SHARP: connate water above the water-oil contact and Sw = 1 below it,
  // with no transition zone, because spreading one requires inverting Pcow
  // against the capillary head.  The flux, meanwhile, does carry capillary
  // pressure: p_w = p_o - Pcow(Sw).  So across the contact the water
  // potential jumps by the full Pcow -- about 54 psi on Norne -- and the
  // initial state is not in equilibrium with the equations that then have to
  // solve it.  Newton spends report step 1 trying to relax a discontinuity
  // that the initialisation created, and on Norne it does not manage: the
  // worst cells come in equal and opposite pairs straddling the contact,
  // (i7 j53 k16) at Sw = 0.06 against (i7 j53 k17) at Sw = 1.0.
  //
  // SWATINIT is the deck's answer to exactly this. It is the saturation
  // distribution the model was built with, transition zone included, so
  // taking it removes the discontinuity rather than smoothing it. Present in
  // Norne as INCLUDE/PETRO/SWINITIAL.INC, and readable once PROPSARRAYS is
  // set; absent, the sharp contacts below still apply.
  const std::vector<Real>* swatinit = grid.array("SWATINIT");

  const int na = grid.n_active();
  InitialState s;
  s.pressure.resize(na);
  s.pcow_target.resize(na);
  s.sw.resize(na);
  s.sg.resize(na);
  s.so.resize(na);
  s.region.resize(na);

  // EQLNUM selects the region per cell.  A deck without one has a single
  // region, which is what a one-record EQUIL means.
  const std::vector<Real>* eq =
      eqlnum_.empty() ? grid.array("EQLNUM") : &eqlnum_;

  for (int a = 0; a < na; ++a) {
    const int g = grid.global_of_active(a);
    int r = 0;
    if (eq && (size_t)g < eq->size()) r = (int)(*eq)[(size_t)g] - 1;
    if (r < 0 || r >= (int)records_.size()) r = 0;
    s.region[a] = r + 1;

    const EquilRecord& e = records_[r];
    const Real z = grid.depth[a];
    s.pcow_target[a] = oil_pressure(r, z) - water_pressure(r, z);

    // Connate water is present in every column; only the hydrocarbon phase
    // changes across a contact.
    Real swc = default_swl;
    if (swl && (size_t)g < swl->size()) swc = (*swl)[(size_t)g];
    if (!(swc >= 0)) swc = 0;
    if (swc > 0.9) swc = 0.9;

    // Phases stack by depth between the contacts.  A deck that puts the GOC at
    // or above the reservoir top -- as four of Norne's five regions do, by
    // setting it equal to the datum -- simply has no gas cap, and this falls
    // out of the comparison rather than needing a special case.
    // The deck's own water saturation where it gives one; the contacts still
    // decide which hydrocarbon occupies the rest.
    Real sw0 = -1;
    if (swatinit && (size_t)g < swatinit->size()) {
      const Real v = (*swatinit)[(size_t)g];
      if (std::isfinite((double)v) && v >= 0 && v <= 1) {
        sw0 = v;
        if (sw0 < swc) sw0 = swc;      // never below the cell's connate water
      }
    }

    if (sw0 >= 0) {
      if (s.n_swatinit == 0) { s.swatinit_lo = s.swatinit_hi = sw0; }
      if (sw0 < s.swatinit_lo) s.swatinit_lo = sw0;
      if (sw0 > s.swatinit_hi) s.swatinit_hi = sw0;
      ++s.n_swatinit;
    }

    if (z >= e.owc_depth) {
      // Below the water contact the column is water, whatever SWATINIT holds
      // for a cell the deck may never have initialised.
      s.sw[a] = (sw0 >= 0) ? sw0 : (Real)1.0;
      s.so[a] = 1.0 - s.sw[a]; s.sg[a] = 0.0;
    } else if (z >= e.goc_depth) {
      const Real w = (sw0 >= 0) ? sw0 : swc;
      s.sw[a] = w; s.so[a] = 1.0 - w; s.sg[a] = 0.0;
    } else {
      const Real w = (sw0 >= 0) ? sw0 : swc;
      s.sw[a] = w; s.so[a] = 0.0; s.sg[a] = 1.0 - w;
    }

    // The pressure follows the column of the phase that is actually there.
    //
    // The primary variable is the oil-phase pressure, and with zero capillary
    // pressure all three phase pressures are equal -- so the profile is
    // continuous across a contact, but its SLOPE changes, because below the
    // water contact it is the water column that is in hydrostatic equilibrium
    // and above the gas contact it is the gas column.
    //
    // Using the oil gradient everywhere gives a single straight line that is
    // right only in the oil leg. On Norne that leaves the aquifer out of
    // equilibrium by (0.1001 - 0.0706) bar/m over a water column of several
    // hundred metres -- of order ten bar, in half the cells in the model. The
    // first timestep then has to absorb a pressure transient that should never
    // have existed, and the linear solve stalls trying.
    if (z >= e.owc_depth)      s.pressure[a] = water_pressure(r, z);
    else if (z >= e.goc_depth) s.pressure[a] = oil_pressure(r, z);
    else                       s.pressure[a] = gas_pressure(r, z);
  }
  return s;
}

// ---------------------------------------------------------------------------
std::string Equilibration::summary() const {
  std::ostringstream os;
  char b[256];

  os << "Equilibration\n";
  std::snprintf(b, sizeof b, "  regions         : %d\n", (int)records_.size());
  os << b;
  const char* pu = (units_ == Units::Metric) ? "bar" : "psi";
  const char* lu = (units_ == Units::Metric) ? "m" : "ft";

  for (size_t r = 0; r < records_.size(); ++r) {
    const EquilRecord& e = records_[r];
    std::snprintf(b, sizeof b,
                  "    %zu: datum %.1f %s at %.2f %s, WOC %.1f, GOC %.1f\n",
                  r + 1, e.datum_depth, lu, e.datum_pressure, pu,
                  e.owc_depth, e.goc_depth);
    os << b;
  }

  if (!densities_.empty()) {
    const SurfaceDensity& d = densities_.front();
    std::snprintf(b, sizeof b,
                  "  surface density : oil %.2f, water %.2f, gas %.3f\n",
                  d.oil, d.water, d.gas);
    os << b;
  }

  if (have_grad_) {
    std::snprintf(b, sizeof b,
                  "  gradients       : oil %.4f, water %.4f, gas %.4f %s/%s "
                  "(%s)\n",
                  grad_oil_, grad_water_, grad_gas_, pu, lu,
                  grad_from_fvf_ ? "from DENSITY and the supplied FVFs"
                                 : "supplied directly");
    os << b;
    if (grad_from_fvf_) {
      std::snprintf(b, sizeof b,
                    "                    Bo %.4f, Bw %.4f, Bg %.5f, Rs %.1f\n",
                    bo_, bw_, bg_, rs_);
      os << b;
    }
  } else {
    os << "  gradients       : NOT SET -- initialise() will refuse to guess\n";
  }

  if (!eqlnum_.empty()) {
    std::set<int> regions;
    for (Real v : eqlnum_) regions.insert((int)v);
    std::snprintf(b, sizeof b, "  EQLNUM          : %d cells, %d distinct regions\n",
                  (int)eqlnum_.size(), (int)regions.size());
    os << b;
  }

  if (!ignored_.empty()) {
    std::set<std::string> uniq(ignored_.begin(), ignored_.end());
    os << "  not implemented : ";
    int n = 0;
    for (const std::string& s : uniq) {
      if (n++) os << ", ";
      os << s;
      if (n == 10) { os << ", ... (" << (int)uniq.size() - 10 << " more)"; break; }
    }
    os << "\n";
  }
  return os.str();
}

}  // namespace nvrs
