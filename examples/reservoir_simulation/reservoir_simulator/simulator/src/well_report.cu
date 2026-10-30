// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// well_report.cu -- surface rates per well for a corner-point field run.
//
// The whole point is that this duplicates no physics.  The three things a
// well rate needs -- the connection factor, the flowing bottom-hole pressure,
// and whether the completion is live at all -- are exactly the three things
// apply_schedule_step() already resolved and wrote into the per-cell WellCell
// array that the assembly kernels read.  So that array is copied back and
// used verbatim.  A rate reported here is the rate the solver applied, not a
// second opinion about it.
//
// The one thing not in WellCell is which well owns a cell, and that comes
// from the schedule's completion list, whose `cell` field is the active-cell
// index the same array is keyed by.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <cstring>
#include <utility>
#include <vector>

#include "deck.hpp"
#include "field_model.hpp"
#include "fluid.hpp"
#include "physics.hpp"
#include "pvt_tables.hpp"
#include "well_report.hpp"

#include "well_control.hpp"
#include "wellcell.hpp"


namespace nvrs {

namespace {

// Eclipse's default COMPORD is TRACK, not geometric depth order and not raw
// COMPDAT declaration order.  OPM WellConnections::orderTRACK starts at the
// connection closest to the well head in I/J (depth breaks ties), then walks
// to the closest remaining I/J connection.  The order is topological: a
// deviated well can become shallower farther along the bore.  It controls both
// the defaulted WELSPECS BHP datum and the segment-by-segment hydrostatic
// column, so replacing it with min(depth) creates a nearly constant but
// well-specific BHP offset.
std::vector<size_t> well_track_order(const Well& w, const FieldModel& fm) {
  std::vector<size_t> order;
  order.reserve(w.completions.size());
  for (size_t ic = 0; ic < w.completions.size(); ++ic) {
    const Completion& c = w.completions[ic];
    if (c.cell >= 0 && c.cell < (int)fm.grid.depth.size())
      order.push_back(ic);
  }
  for (size_t pos = 0; pos < order.size(); ++pos) {
    const int oi = pos ? w.completions[order[pos - 1]].i : w.head_i;
    const int oj = pos ? w.completions[order[pos - 1]].j : w.head_j;
    const double oz = pos
        ? (double)fm.grid.depth[w.completions[order[pos - 1]].cell] : 0.0;
    size_t best = pos;
    long long best_ij = std::numeric_limits<long long>::max();
    double best_dz = std::numeric_limits<double>::infinity();
    for (size_t q = pos; q < order.size(); ++q) {
      const Completion& c = w.completions[order[q]];
      const long long di = (long long)c.i - oi;
      const long long dj = (long long)c.j - oj;
      const long long ij = di * di + dj * dj;
      const double dz = std::fabs((double)fm.grid.depth[c.cell] - oz);
      if (ij < best_ij || (ij == best_ij && dz < best_dz)) {
        best = q; best_ij = ij; best_dz = dz;
      }
    }
    if (best != pos) std::swap(order[pos], order[best]);
  }
  return order;
}

double well_reference_depth(const Well& w, const FieldModel& fm,
                            const std::vector<size_t>& order) {
  const double stated = (double)w.ref_depth;
  if (stated >= 0.0 && std::isfinite(stated)) return stated;
  if (!order.empty())
    return (double)fm.grid.depth[w.completions[order.front()].cell];
  return 0.0;
}

size_t well_track_signature(const std::vector<Well>& wells) {
  size_t h = (size_t)1469598103934665603ULL;
  auto add = [&](size_t v) {
    h ^= v + (size_t)0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  add(wells.size());
  for (const Well& w : wells) {
    add((size_t)(unsigned)w.head_i); add((size_t)(unsigned)w.head_j);
    add(w.completions.size());
    for (const Completion& c : w.completions) {
      add((size_t)(unsigned)c.i); add((size_t)(unsigned)c.j);
      add((size_t)(unsigned)c.k); add((size_t)(unsigned)c.cell);
    }
  }
  return h;
}

// Give a host SatTables the same endpoint scaling the device one has.
//
// Without this the reported rates and the flow solve use different relative
// permeability curves: the kernel reads each cell's own SWCR/SOWCR while this
// file reads the field-wide table, so a well's productivity here would not be
// the productivity the reservoir actually delivers -- and the bottom-hole
// pressure solved from it would miss its target.
//
// The arrays are on the ACTIVE numbering, which is what a Completion's `cell`
// already is, so no permutation is needed.  Built once and kept: the grid
// does not change during a run.
void host_endscale(const FieldModel& fm, SatTables& T, int enabled) {
  if (!enabled) return;
  static bool done = false;
  static std::vector<Real> a[22];
  static const char* nm[22] = {
      "SWL", "SWCR", "SWU", "SGL", "SGCR", "SGU", "SOWCR", "SOGCR", "KRW", "KRO", "KRG",
      "ISWL", "ISWCR", "ISWU", "ISGL", "ISGCR", "ISGU", "ISOWCR", "ISOGCR", "IKRW", "IKRO", "IKRG"};
  static int nfound = 0;
  if (!done) {
    done = true;
    for (int i = 0; i < 22; ++i)
      if (fm.grid.has_array(nm[i])) {
        a[i] = fm.grid.active_array(nm[i]);
        if ((int)a[i].size() == fm.grid.n_active()) ++nfound;
        else a[i].clear();
      }
  }
  if (!nfound) return;
  const Real* q[22];
  for (int i = 0; i < 22; ++i) q[i] = a[i].empty() ? nullptr : a[i].data();
  T.es.swl = q[0];   T.es.swcr = q[1];  T.es.swu = q[2];
  T.es.sgl = q[3];   T.es.sgcr = q[4];  T.es.sgu = q[5];
  T.es.sowcr = q[6]; T.es.sogcr = q[7];
  T.es.krw = q[8];   T.es.kro = q[9];   T.es.krg = q[10];
  T.es.iswl = q[11]; T.es.iswcr = q[12]; T.es.iswu = q[13];
  T.es.isgl = q[14]; T.es.isgcr = q[15]; T.es.isgu = q[16];
  T.es.isowcr = q[17]; T.es.isogcr = q[18];
  T.es.ikrw = q[19]; T.es.ikro = q[20]; T.es.ikrg = q[21];
  T.es.on = 1;
  // The reservoir kernels copy this flag in field_model.cu.  The host-side
  // well-control tables must use the same saturation map: otherwise Norne's
  // SCALECRS YES advances the reservoir with three-point relperms while the
  // BHP solve evaluates every completion with two-point relperms.
  T.es.three_point = fm.grid.scalecrs ? 1 : 0;
}

// Host-side well controls and summaries must use the same PVTNUM region as
// the reservoir kernel.  Keeping the flattened tables here also avoids
// rebuilding all PVTO/PVTG branches on every Newton iteration -- the tables
// are immutable for the duration of a run.
const std::vector<HostPvt>& host_pvt_regions(const PvtTables* pvt) {
  static const PvtTables* cached = nullptr;
  static std::vector<HostPvt> regions;
  if (pvt != cached) {
    cached = pvt;
    regions.clear();
    if (pvt && (pvt->have_pvto() || pvt->have_pvtg())) {
      const int n = std::max(1, pvt->n_regions());
      regions.reserve((size_t)n);
      for (int r = 0; r < n; ++r) regions.push_back(flatten_pvt(*pvt, r));
    }
  }
  return regions;
}

int pvt_region_of(const FieldModel& fm, int active, int nregions) {
  if (nregions <= 1 || active < 0 || active >= fm.grid.n_active()) return 0;
  const std::vector<Real>* pn = fm.grid.array("PVTNUM");
  if (!pn) return 0;
  const int global = fm.grid.global_of_active(active);
  if (global < 0 || global >= (int)pn->size()) return 0;
  const int region = (int)std::lround((*pn)[global]) - 1;
  return (region >= 0 && region < nregions) ? region : 0;
}

DevicePvt pvt_for_active(const std::vector<HostPvt>& regions,
                         const FieldModel& fm, int active) {
  if (regions.empty()) return DevicePvt{};
  return regions[(size_t)pvt_region_of(fm, active, (int)regions.size())].view();
}

// The undersaturated Rs anchor for one ACTIVE cell, exactly as the kernel sees
// it.  RSVD makes this a per-cell quantity; reading the scalar here while the
// assembly reads the array would make every reported rate, water cut and
// gas-oil ratio describe a slightly different fluid from the one the residual
// was built on.
// The DRSDT ceiling, mirrored on the host.
//
// A DRSDT run moves this array every accepted step, so the host copy has to be
// refreshed -- and refreshed ONCE per entry point, not once per lookup.
// rs_anchor_of() below is called inside loops over completions and, in
// field_oil_in_place(), over every active cell; a device copy inside it would
// mean 44,431 copies of a 44,431-element array to evaluate one field total.
//
// So the copy is explicit: each entry point that evaluates PVT calls
// refresh_drsdt_cap() once, and the lookup is then a plain array read.
std::vector<Real>& drsdt_cap_host() {
  static std::vector<Real> cap;
  return cap;
}

void refresh_drsdt_cap(const FieldModel& fm, const Reservoir& r) {
  std::vector<Real>& cap = drsdt_cap_host();
  cap.clear();
  const int cells = fm.part.n_owned;
  if (!r.rs_cap || cells <= 0) return;
  cap.assign((size_t)cells, (Real)0);
  if (cudaMemcpy(cap.data(), r.rs_cap, sizeof(Real) * (size_t)cells,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    cap.clear();
}

std::vector<Real>& gas_primary_host() {
  static std::vector<Real> meaning;
  return meaning;
}

void refresh_gas_primary(const FieldModel& fm, const Reservoir& r) {
  std::vector<Real>& meaning = gas_primary_host();
  meaning.clear();
  if (!r.gas_primary || fm.part.n_owned <= 0) return;
  meaning.resize((size_t)fm.part.n_owned);
  if (cudaMemcpy(meaning.data(), r.gas_primary,
                 sizeof(Real) * meaning.size(), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    meaning.clear();
}

bool rs_primary_at(const FieldModel& fm, const Reservoir& r, int active) {
  if (!r.gas_primary || active < 0 ||
      active >= (int)fm.part.local_of_active.size()) return false;
  const int loc = fm.part.local_of_active[(size_t)active];
  const auto& meaning = gas_primary_host();
  return loc >= 0 && loc < (int)meaning.size() &&
         meaning[(size_t)loc] > 0.5 && meaning[(size_t)loc] < 1.5;
}

Real gas_meaning_at(const FieldModel& fm, const Reservoir& r, int active) {
  if (!r.gas_primary || active < 0 ||
      active >= (int)fm.part.local_of_active.size()) return 0;
  const int loc = fm.part.local_of_active[(size_t)active];
  const auto& meaning = gas_primary_host();
  return loc >= 0 && loc < (int)meaning.size() ? meaning[(size_t)loc] : 0;
}

Real physical_sg_at(const FieldModel& fm, const Reservoir& r, int active,
                    Real sw, Real x) {
  const Real meaning = gas_meaning_at(fm, r, active);
  return meaning > (Real)1.5 ? fmax((Real)0, (Real)1 - sw)
       : meaning > (Real)0.5 ? (Real)0 : x;
}

Real oil_pressure_at(const FieldModel& fm, const Reservoir& r, int active,
                     Real p, Real sw, Real sg, const SatTables& tables) {
  if (gas_meaning_at(fm, r, active) <= (Real)1.5) return p;
  return p - capillary(sw, sg, tables, active).pcog;
}

std::vector<Real>& so_max_host() {
  static std::vector<Real> v;
  return v;
}

void refresh_vappars_host(const FieldModel& fm, const Reservoir& r) {
  std::vector<Real>& out = so_max_host();
  out.clear();
  if (!r.so_max) return;
  const int cells = fm.part.n_owned;
  if (cells <= 0) return;
  std::vector<Real> loc((size_t)cells, (Real)0);
  if (cudaMemcpy(loc.data(), r.so_max, sizeof(Real) * (size_t)cells,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return;
  out.assign((size_t)fm.grid.n_active(), (Real)0);
  for (int a = 0; a < fm.grid.n_active(); ++a) {
    if ((size_t)a >= fm.part.local_of_active.size()) continue;
    const int l = fm.part.local_of_active[(size_t)a];
    if (l >= 0 && l < cells) out[(size_t)a] = loc[(size_t)l];
  }
}

// The dissolved-gas ceiling for one ACTIVE cell, exactly as the kernel sees it:
// the DRSDT cap where a deck states DRSDT, otherwise the RSVD anchor,
// otherwise the deck's scalar.  Reading the static anchor during a DRSDT run
// would evaluate every reported rate, water cut and gas-oil ratio on more
// dissolved gas than the residual used, and the report would then describe a
// different fluid from the run.
Real rs_anchor_of(const FieldModel& fm, const Reservoir& r, int active) {
  if (active >= 0) {
    const std::vector<Real>& cap = drsdt_cap_host();
    if (!cap.empty() && active < (int)fm.part.local_of_active.size()) {
      const int loc = fm.part.local_of_active[(size_t)active];
      if (loc >= 0 && loc < (int)cap.size()) return cap[(size_t)loc];
    }
    if ((size_t)active < fm.rs_anchor.size())
      return fm.rs_anchor[(size_t)active];
  }
  return r.rs_max;
}

PvtCell pvt_cell_at(const DevicePvt& pv, Real p, Real sw, Real x,
                    int active, const FieldModel& fm, const Reservoir& r) {
  const bool rs_primary = rs_primary_at(fm, r, active);
  const Real meaning = gas_meaning_at(fm, r, active);
  const bool rv_primary = meaning > (Real)1.5;
  const Real sg = physical_sg_at(fm, r, active, sw, x);
  const Real so = std::max((Real)0, (Real)1 - sw - sg);
  const std::vector<Real>& hist = so_max_host();
  const Real som = active >= 0 && active < (int)hist.size()
                       ? hist[(size_t)active]
                       : so;
  // RSVD is an oil-only charge, not a ceiling in a free-gas cell.  Preserve
  // an explicit DRSDT cap when present; otherwise select saturated Rs.
  const Real rslim = (!r.rs_cap && sg > (Real)1e-7)
                         ? (Real)1e30 : rs_anchor_of(fm, r, active);
  PvtCell f = rs_primary ? dev_pvt_cell_rs(pv, p, x, so, som, r.vap1)
              : rv_primary ? dev_pvt_cell_rv(pv, p, x, som, r.vap1)
                           : dev_pvt_cell(pv, p, rslim, so, som, r.vap1);
  if (!rs_primary && !rv_primary) {
    f.drv_dx = -f.drv_dso;
    f.dbg_dx = -f.dbg_dso;
    f.dmug_dx = -f.dmug_dso;
  }
  if (!rv_primary) {
    f.drv_dsw = -f.drv_dso;
    f.dbg_dsw = -f.dbg_dso;
    f.dmug_dsw = -f.dmug_dso;
  }
  return f;
}

// The EHYSTR historical-high Sg, mirrored on the host in ACTIVE order --
// unlike drsdt_cap_host() above, because relperm3's `cell` argument on this
// side of the code is the completion's ACTIVE index (c.cell), not the local
// one the kernel uses, and T.sg_hy is read by plain `T.sg_hy[cell]` inside
// relperm3 with no translation of its own.  See host_endscale() for the same
// active-order convention already used for ENDSCALE's per-cell arrays.
std::vector<Real>& sg_hy_host() {
  static std::vector<Real> v;
  return v;
}

std::vector<Real>& sw_ow_hy_host() {
  static std::vector<Real> v;
  return v;
}

void copy_hyst_active(const FieldModel& fm, const Real* device, Real fill,
                      std::vector<Real>& out) {
  out.clear();
  if (!device) return;
  const int cells = fm.part.n_owned;
  if (cells <= 0) return;
  std::vector<Real> loc((size_t)cells, fill);
  if (cudaMemcpy(loc.data(), device, sizeof(Real) * (size_t)cells,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return;
  const int na = fm.grid.n_active();
  if (na <= 0) return;
  out.assign((size_t)na, fill);
  for (int a = 0; a < na; ++a) {
    if ((size_t)a >= fm.part.local_of_active.size()) continue;
    const int lo = fm.part.local_of_active[(size_t)a];
    if (lo >= 0 && lo < cells) out[(size_t)a] = loc[(size_t)lo];
  }
}

void refresh_hyst_host(const FieldModel& fm, const Reservoir& r) {
  sg_hy_host().clear();
  sw_ow_hy_host().clear();
  if (!r.use_hysteresis) return;
  copy_hyst_active(fm, r.sg_hy, (Real)0, sg_hy_host());
  copy_hyst_active(fm, r.sw_ow_hy, (Real)2, sw_ow_hy_host());
}

// SWOF and SGOF do not change during a run.  The control path used to create
// six vectors, grow them row by row, and then discard them in both the
// injector and producer routine on every Newton/line-search evaluation.
// Keep the owning vectors and only rebuild the lightweight SatTables view.
SatTables host_sat_tables(const Deck& d, const FieldModel& fm,
                          const Reservoir& r) {
  struct Cache {
    const Deck* deck = nullptr;
    std::vector<Real> sw, krow, krw, sg, krog, krg;
  };
  static Cache c;
  if (c.deck != &d) {
    c.deck = &d;
    c.sw.clear(); c.krow.clear(); c.krw.clear();
    c.sg.clear(); c.krog.clear(); c.krg.clear();
    c.sw.reserve(d.swof.size());
    c.krow.reserve(d.swof.size());
    c.krw.reserve(d.swof.size());
    for (const auto& row : d.swof) {
      c.sw.push_back(row[0]);
      c.krow.push_back(row[1]);
      c.krw.push_back(row[2]);
    }
    c.sg.reserve(d.sgof.size());
    c.krog.reserve(d.sgof.size());
    c.krg.reserve(d.sgof.size());
    for (const auto& row : d.sgof) {
      c.sg.push_back(row[0]);
      c.krog.push_back(row[1]);
      c.krg.push_back(row[2]);
    }
  }
  SatTables T{};
  T.sw = c.sw.data(); T.krow = c.krow.data(); T.krw = c.krw.data();
  T.nsw = (int)c.sw.size();
  T.sg = c.sg.data(); T.krog = c.krog.data(); T.krg = c.krg.data();
  T.nsg = (int)c.sg.size();
  set_table_endpoints(T, c.sw.data(), c.krw.data(), c.krow.data(), T.nsw,
                      c.sg.data(), c.krg.data(), c.krog.data(), T.nsg);
  host_endscale(fm, T, d.use_endscale);
  T.hyst_on = r.use_hysteresis;
  T.sg_hy = sg_hy_host().empty() ? nullptr : sg_hy_host().data();
  T.sw_ow_hy = sw_ow_hy_host().empty() ? nullptr : sw_ow_hy_host().data();
  return T;
}
// Keep host-side well controls and reporting on the same phase model as the
// reservoir residual. A two-phase run has no SGOF table by construction, but
// it can have SWOF; use its piecewise-linear curves when present and retain
// the analytic Corey fallback for compact decks without a table.
Mob3 well_relperm(int bs, Real sw, Real sg, const SatTables& tables,
                  Real uw, Real uo, Real ug, Real bw, Real bo, Real bg,
                  Real swi, Real swr, int cell) {
  if (bs == 2) {
    const Mob2 q = relperm2_table(sw, tables, uw, uo, bw, bo, swi, swr);
    Mob3 m{};
    m.Mw = q.Mw;
    m.Mo = q.Mo;
    m.dMw_dSw = q.dMw;
    m.dMo_dSw = q.dMo;
    return m;
  }
  return relperm3(sw, sg, tables, uw, uo, ug, bw, bo, bg, swi, swr,
                  cell);
}


// Injector and producer controls are evaluated consecutively in the hot
// Newton path.  The injector has already copied the identical reservoir state
// to the host and its updated WellCell array is exactly the producer input.
// Move those buffers across the call boundary instead of synchronously
// downloading them again.  The caller opts in explicitly, so schedule/report
// calls in the opposite order cannot consume a stale snapshot.
struct ControlHandoff {
  const Reservoir* reservoir = nullptr;
  const FieldModel* field = nullptr;
  int step = -1;
  long long accepted_state_id = -1;
  bool history_valid = false;
  bool ready = false;
  std::vector<Real> u;
};

ControlHandoff& control_handoff() {
  static ControlHandoff h;
  return h;
}

struct ControlStateIndexCache {
  const Reservoir* reservoir = nullptr;
  const ScheduleStep* schedule = nullptr;
  const int* local_map = nullptr;
  int cells = 0;
  int* device_cells = nullptr;
  std::vector<int> loc;
  std::vector<int> packed_of_local;
};

ControlStateIndexCache& control_state_indices(
    Reservoir& r, const ScheduleStep& sc,
    const std::vector<int>& local_of_active, int cells, int bs) {
  static thread_local ControlStateIndexCache cache;
  bool dirty = cache.reservoir != &r || cache.schedule != &sc ||
               cache.local_map != local_of_active.data() ||
               cache.cells != cells;
  if (dirty) {
    cache.reservoir = &r;
    cache.schedule = &sc;
    cache.local_map = local_of_active.data();
    cache.cells = cells;
    cache.loc.clear();
    for (const Well& w : sc.wells)
      for (const Completion& c : w.completions) {
        if (c.cell < 0 || c.cell >= (int)local_of_active.size()) continue;
        const int l = local_of_active[(size_t)c.cell];
        if (l >= 0 && l < cells) cache.loc.push_back(l);
      }
    std::sort(cache.loc.begin(), cache.loc.end());
    cache.loc.erase(std::unique(cache.loc.begin(), cache.loc.end()),
                    cache.loc.end());
    cache.packed_of_local.assign((size_t)cells, -1);
    for (int i = 0; i < (int)cache.loc.size(); ++i)
      cache.packed_of_local[(size_t)cache.loc[(size_t)i]] = i;
  }
  const int n = (int)cache.loc.size();
  if (n > r.well_state_capacity) {
    if (r.well_state_cells) cudaFree(r.well_state_cells);
    if (r.well_state_values) cudaFree(r.well_state_values);
    NVRS_CHECK_CUDA(cudaMalloc(&r.well_state_cells,
                               sizeof(int) * (size_t)n));
    NVRS_CHECK_CUDA(cudaMalloc(&r.well_state_values,
                               sizeof(Real) * (size_t)bs * (size_t)n));
    r.well_state_capacity = n;
    dirty = true;
  }
  if (cache.device_cells != r.well_state_cells) dirty = true;
  if (dirty && n) {
    NVRS_CHECK_CUDA(cudaMemcpy(r.well_state_cells, cache.loc.data(),
                               sizeof(int) * (size_t)n,
                               cudaMemcpyHostToDevice));
    cache.device_cells = r.well_state_cells;
  }
  return cache;
}

__global__ void k_gather_well_state(const Real* state, int bs,
                                    const int* cells, int n, Real* packed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int cell = cells[i];
  for (int v = 0; v < bs; ++v)
    packed[(size_t)bs * i + v] = state[(size_t)bs * cell + v];
}

void gather_control_state(Reservoir& r, int bs,
                          const ControlStateIndexCache& index,
                          std::vector<Real>& host) {
  const int n = (int)index.loc.size();
  host.resize((size_t)bs * n);
  if (!n) return;
  constexpr int block = 128;
  k_gather_well_state<<<(n + block - 1) / block, block>>>(
      r.u, bs, r.well_state_cells, n, r.well_state_values);
  NVRS_CHECK_CUDA(cudaGetLastError());
  NVRS_CHECK_CUDA(cudaMemcpy(host.data(), r.well_state_values,
                             sizeof(Real) * host.size(),
                             cudaMemcpyDeviceToHost));
}

__global__ void k_scatter_wellcells(WellCell* dst, const int* cells,
                                    const WellCell* values, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[cells[i]] = values[i];
}

// A control refresh changes only scheduled completion cells.  Uploading the
// entire 44k-cell WellCell array made this small host calculation pay a 3.9 MB
// synchronous H2D copy thousands of times.  Compact the live schedule cells
// and scatter them on-device; apply_schedule_step() remains the authoritative
// full-array refresh when the schedule itself changes.
void upload_schedule_wellcells(Reservoir& r, const ScheduleStep& sc,
                               const std::vector<int>& local_of_active,
                               int cells, const WellCell* host) {
  struct Cache {
    const Reservoir* reservoir = nullptr;
    const ScheduleStep* schedule = nullptr;
    const int* local_map = nullptr;
    int cells = 0;
    int* device_cells = nullptr;
    std::vector<int> loc;
    std::vector<WellCell> value;
  };
  static thread_local Cache cache;
  bool indices_dirty =
      cache.reservoir != &r || cache.schedule != &sc ||
      cache.local_map != local_of_active.data() || cache.cells != cells;
  if (indices_dirty) {
    cache.reservoir = &r;
    cache.schedule = &sc;
    cache.local_map = local_of_active.data();
    cache.cells = cells;
    cache.loc.clear();
    for (const Well& w : sc.wells)
      for (const Completion& c : w.completions) {
        if (c.cell < 0 || c.cell >= (int)local_of_active.size()) continue;
        const int l = local_of_active[(size_t)c.cell];
        if (l >= 0 && l < cells) cache.loc.push_back(l);
      }
    std::sort(cache.loc.begin(), cache.loc.end());
    cache.loc.erase(std::unique(cache.loc.begin(), cache.loc.end()),
                    cache.loc.end());
  }
  const int n = (int)cache.loc.size();
  if (!n) return;

  if (n > r.well_update_capacity) {
    if (r.well_update_cells) cudaFree(r.well_update_cells);
    if (r.well_update_values) cudaFree(r.well_update_values);
    NVRS_CHECK_CUDA(cudaMalloc(&r.well_update_cells, sizeof(int) * (size_t)n));
    NVRS_CHECK_CUDA(cudaMalloc(&r.well_update_values,
                               sizeof(WellCell) * (size_t)n));
    r.well_update_capacity = n;
    indices_dirty = true;
  }
  if (cache.device_cells != r.well_update_cells) indices_dirty = true;
  if (indices_dirty) {
    NVRS_CHECK_CUDA(cudaMemcpy(r.well_update_cells, cache.loc.data(),
                               sizeof(int) * (size_t)n,
                               cudaMemcpyHostToDevice));
    cache.device_cells = r.well_update_cells;
  }
  cache.value.resize((size_t)n);
  for (int i = 0; i < n; ++i)
    cache.value[(size_t)i] = host[(size_t)cache.loc[(size_t)i]];
  NVRS_CHECK_CUDA(cudaMemcpy(r.well_update_values, cache.value.data(),
                             sizeof(WellCell) * (size_t)n,
                             cudaMemcpyHostToDevice));
  constexpr int block = 128;
  k_scatter_wellcells<<<(n + block - 1) / block, block>>>(
      static_cast<WellCell*>(r.wellcells), r.well_update_cells,
      static_cast<const WellCell*>(r.well_update_values), n);
  NVRS_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

namespace {
// NVRS_DEBUG_WELL=<name>: trace apply_producer_rate_targets()'s per-completion
// theta/pwf and open/shut decision for one well, every call.  Temporary
// instrumentation for tracking down the D-4H WWCT chatter under WELLXFLOW 0;
// not wired into any deck keyword, so it costs nothing when unset.
const char* debug_well() {
  static const char* w = std::getenv("NVRS_DEBUG_WELL");
  return w;
}
}  // namespace

namespace {
// The last rate target each well was given, by name: (kind, value).
//
// Function-local rather than a file static so its initialisation order is
// not a question, and keyed by name because that is what survives a well
// changing type, control mode or completion count between records.
std::map<std::string, std::pair<int, double>>& last_target() {
  static std::map<std::string, std::pair<int, double>> m;
  return m;
}

// What the controls actually IMPOSED this step, by well name: (kind, rate).
//
// A rate-controlled well produces its target -- that is what "controlled"
// means, and apply_producer_rate_targets() solves the flowing pressure that
// delivers it.  Reporting it by recomputing b.WI.lambda.(p - pwf) from the
// saved state is a second, independent calculation of the same thing, and the
// two drifted: the report claimed 297 M STB of cumulative oil while the
// saturation fields said 75.8 M had actually left the reservoir, a factor of
// 3.9 between the well curves and the material balance of the same run.
//
// So report what was imposed.  The recomputation still sets the PHASE RATIOS
// -- water cut and gas-oil ratio come from the state, as they must -- and the
// controlled phase is scaled to its target, carrying the others with it.
std::map<std::string, std::pair<int, double>>& applied_rate() {
  static std::map<std::string, std::pair<int, double>> m;
  return m;
}
}  // namespace


std::vector<WellRateRow> field_well_rates(const Deck& d, const FieldModel& fm,
                                          const Reservoir& r, int step,
                                          const PvtTables* pvt) {
  refresh_drsdt_cap(fm, r);   // once; see the note on rs_anchor_of
  refresh_gas_primary(fm, r);
  refresh_vappars_host(fm, r);
  std::vector<WellRateRow> out;
  const int cells = fm.part.n_owned;
  const int bs = r.bs;

  // Whether this rank can contribute any rates at all.
  //
  // It must NOT return early when it cannot.  The rows are reduced across the
  // ranks afterwards, and that reduction is a collective sized by the row
  // count: a rank that bailed out with an empty vector would either skip the
  // collective the others are waiting in, or enter it with a different length.
  // Either is a hang, and it would be a hang in the rare case -- a rank owning
  // no cells -- which is the one nobody tests.  So always walk the schedule,
  // which every rank holds identically, and contribute nothing if there is
  // nothing to contribute.
  bool have_state = (cells > 0 && r.u != nullptr && r.wellcells != nullptr);

  std::vector<Real> u;
  std::vector<WellCell> wc;
  if (have_state) {
    u.resize((size_t)bs * cells);
    wc.resize(cells);
    if (cudaMemcpy(u.data(), r.u, sizeof(Real) * u.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(wc.data(), r.wellcells, sizeof(WellCell) * cells,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      have_state = false;
  }

  // --- the same rel-perm tables the kernel was given -----------------------
  refresh_hyst_host(fm, r);
  SatTables T = host_sat_tables(d, fm, r);

  // --- and the same PVT ----------------------------------------------------
  // flatten_pvt() gives a host-resident copy whose view() every evaluator in
  // pvt_tables.hpp accepts, so dev_pvt_cell() here is the identical function
  // the kernel calls on identical data.
  const std::vector<HostPvt>& hp = host_pvt_regions(pvt);

  const ScheduleStep& s = fm.schedule.step(step);
  const std::vector<int>& l2a = fm.part.local_of_active;

  for (const Well& w : s.wells) {
    if (!w.state.defined) continue;
    WellRateRow row;
    row.name = w.name;
    double Ao = 0, Aw = 0, Ag = 0;   // phase productivities, no drawdown
    const bool injector = (w.state.type == WellType::Injector);
    row.type = (!w.state.open || w.state.mode == CtrlMode::Shut)
                   ? 0
                   : (injector ? 1 : -1);

    for (const Completion& c : w.completions) {
      if (!have_state) break;
      if (c.cell < 0 || !c.open) continue;
      if (c.cell >= (int)l2a.size()) continue;
      const int loc = l2a[c.cell];
      if (loc < 0 || loc >= cells) continue;      // another rank owns it
      const WellCell& k = wc[loc];

      // A completion carrying no FLOW still has a PRESSURE, and the two are
      // reported separately.
      //
      // This used to skip the cell outright whenever the kernel had skipped
      // it, which is right for the rates and wrong for the bottom-hole
      // pressure.  A producer whose rate target is zero -- Norne is full of
      // them, every `WCONHIST ... 'ORAT' 0.000 0.000 0.000` record and every
      // interval where the derived RESV target comes out zero -- has its
      // completions cleared by apply_producer_rate_targets(), so every one of
      // them failed this test, `nperf` stayed 0, and the well reported a
      // bottom-hole pressure of exactly ZERO.
      //
      // That is what puts the comb of vertical drops to 0 psi through NVRS's
      // WBHP curves where Flow's run smoothly across the same interval, and
      // WBHP is the worst-scoring quantity in the comparison: 22 of 27
      // producers below the threshold with NRMSE routinely over 100%.  Almost
      // none of that is the pressure being wrong.  It is the pressure being
      // absent and read as zero.
      //
      // A shut-in well's bottom-hole pressure is the reservoir pressure at its
      // perforations, which is exactly what the zero-target path already
      // computed and stored in `bhp` before clearing the flags -- it was just
      // never read back.  So take the pressure from any cell this well is
      // completed in, and let the rate terms below keep their own test.
      // Completion flags are device state from the most recent control solve.
      // A schedule SHUT can therefore encounter a stale live flag while we
      // still inspect the completion to retain its pressure.  Never let that
      // stale flag manufacture phase rates: OPM keeps the solved BHP state for
      // a shut well, but all of WOPR/WWPR/WGPR/WWIR/WGIR are exactly zero.
      const bool live = row.type != 0 && (k.is_prod || k.is_inj);
      if (!live && !(k.wi > 0)) continue;   // not this well's cell at all

      ++row.nperf;
      // pwf is corrected to this completion's depth; BHP is defined and
      // reported at the WELSPECS reference depth.
      row.bhp = (k.bhp > 0) ? k.bhp : k.pwf;
      if (!live) continue;                  // pressure only; it carries no flow

      if (k.is_inj) {
        // Report what the reservoir ACCEPTED, not merely the schedule request.
        // The assembly clips a rate-controlled injector at the completion's
        // BHP ceiling.  Summing k.q*_inj here used to ignore that clip, making
        // it impossible to tell a schedule-allocation error from an
        // injectivity/BHP limitation in the comparison report.
        Real qw = k.qw_inj, qo = k.qo_inj, qg = k.qg_inj;
        const Real p = u[(size_t)bs * loc];
        const Real Sw = u[(size_t)bs * loc + 1];
        const Real X = (bs == 3) ? u[(size_t)bs * loc + 2] : (Real)0;
        const Real Sg = physical_sg_at(fm, r, c.cell, Sw, X);
        const DevicePvt pv = pvt_for_active(hp, fm, c.cell);
        const PvtCell F = pv.valid
                              ? pvt_cell_at(pv, p, Sw, X, c.cell, fm, r)
                              : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                                               r.uo, r.uw, r.ug);
        const Mob3 m = well_relperm(bs, Sw, Sg, T, F.muw, F.muo, F.mug,
                                (Real)1, (Real)1, (Real)1,
                                r.swi, r.swr, c.cell);
        const Real po = oil_pressure_at(fm, r, c.cell, p, Sw, Sg, T);
        // Legacy injectors store the completion's BHP ceiling in pwf, so the
        // report must reproduce the assembly-side capacity clip.  Mode 9 sets
        // pwf to zero because its host solve has already evaluated both the
        // injecting and reverse-producing branches at the common operating
        // BHP. Reapplying the legacy clip with pwf=0 erased every positive
        // injection leg and left only negative internal-crossflow rates.
        if (k.pwf > 0) {
          const Real draw = std::max((Real)0, k.pwf - po);
          const Real qres = k.wi * (m.Mw + m.Mo + m.Mg) * draw;
          if (qw > 0) qw = std::min(qw, qres * F.bw);
          if (qo > 0) qo = std::min(qo, qres * F.bo);
          if (qg > 0) qg = std::min(qg, qres * F.bg);
        }
        if (fm.part.rank == 0 && debug_well() && w.name == debug_well())
          std::printf("  [dbg %s injector] step %d (i%d j%d k%d) "
                      "qw=%.8g qo=%.8g qg=%.8g p=%.8g bhp=%.8g\n",
                      w.name.c_str(), step, c.i + 1, c.j + 1, c.k + 1,
                      (double)qw, (double)qo, (double)qg, (double)p,
                      (double)k.bhp);
        row.qw += qw;
        row.qo += qo;
        row.qg += qg;
        // F.* are inverse formation-volume factors (surface/reservoir).
        if (F.bw > 0) row.qres += qw / F.bw;
        if (F.bo > 0) row.qres += qo / F.bo;
        if (F.bg > 0) row.qres += qg / F.bg;
        continue;
      }

      const Real p = u[(size_t)bs * loc];
      const Real Sw = u[(size_t)bs * loc + 1];
      const Real X = (bs == 3) ? u[(size_t)bs * loc + 2] : (Real)0;
      const Real Sg = physical_sg_at(fm, r, c.cell, Sw, X);
      const DevicePvt pv = pvt_for_active(hp, fm, c.cell);

      // Drawdown, sign and all.
      //
      // This has to be the SAME expression the assembly kernel applies, or
      // the report is not measuring the run.  The kernel has no sign test:
      //
      //     R_oil += bo * wi * Mo * (p_i - pwf)
      //
      // so a completion whose cell has fallen below the flowing pressure puts
      // fluid BACK into the reservoir, and dropping those terms here counts
      // production the reservoir never lost.  Skipping them is part of why
      // the stock-tank oil balance reads 113 to 132% where a conserved
      // component must read 100.
      const Real po = oil_pressure_at(fm, r, c.cell, p, Sw, Sg, T);
      const Real draw = po - k.pwf;
      if (!std::isfinite((double)draw)) continue;

      const PvtCell F = pv.valid
                            ? pvt_cell_at(pv, p, Sw, X, c.cell, fm, r)
                            : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                                             r.uo, r.uw, r.ug);
      // B factors are passed as 1 so relperm3 returns kr/mu: the shrinkage is
      // applied explicitly below, exactly as the assembly kernel does it.
      // Multiplying by b in both places is the 1/B double count that made the
      // gas flux three orders of magnitude too large once already.
      const Mob3 m = well_relperm(bs, Sw, Sg, T, F.muw, F.muo, F.mug,
                              (Real)1, (Real)1, (Real)1, r.swi, r.swr, c.cell);

      // Surface oil has an oil-phase contribution and, with VAPOIL, the oil
      // vaporised into produced gas.  OPM reports their sum as WOPR.
      //
      // A reverse-flowing completion under the crossflow model (xflow_model)
      // does not carry local formation mobility per phase -- see
      // assemble_blackoil.cu's producer well block, which this mirrors term
      // for term so the report measures what the kernel actually assembled.
      Real qo_k, qw_k, qg_k = 0, qres_k = 0;
      if (draw < 0 && k.xflow_model) {
        const Real lam = m.Mo + m.Mw + m.Mg;
        const Real q_res = k.wi * lam * draw;
        qres_k = q_res;
        const Real q_res_o = k.mix_o * q_res, q_res_w = k.mix_w * q_res,
                  q_res_g = k.mix_g * q_res;
        qo_k = F.bo * q_res_o + F.rv * F.bg * q_res_g;
        qw_k = F.bw * q_res_w;
        qg_k = F.bg * q_res_g + F.rs * F.bo * q_res_o;
      } else {
        qres_k = k.wi * (m.Mo + m.Mw + m.Mg) * draw;
        qo_k = (F.bo * k.wi * m.Mo + F.rv * F.bg * k.wi * m.Mg) * draw;
        qw_k = F.bw * k.wi * m.Mw * draw;
        qg_k = (F.bg * k.wi * m.Mg + F.rs * F.bo * k.wi * m.Mo) * draw;
      }
      // Measured before the sum, because after it the sign is gone.  A negative
      // completion rate is fluid going back into the formation; see
      // WellRateRow's note on why the net alone cannot be read.
      if (draw < 0) {
        ++row.nperf_reverse;
        row.xflow_o += -qo_k;
        row.xflow_w += -qw_k;
        if (bs == 3) row.xflow_g += -qg_k;
      }
      row.qo += qo_k;
      row.qw += qw_k;
      row.qres += qres_k;
      // Free gas plus the gas dissolved in the oil this completion makes,
      // which is what ECLIPSE reports as WGPR.  F.bg is scf/rb and F.rs is
      // scf/stb, so this accumulation is deliberately in SCF/day, like the
      // gas residual.  It is converted to the summary's MSCF/day below.
      if (bs == 3) row.qg += qg_k;
      // The same sums WITHOUT the drawdown.  These are the well's phase
      // productivities, and their ratios are the water cut and gas-oil ratio
      // -- properties of the fluid at the perforations, independent of how
      // hard the well is being pulled.  Which is what makes them safe to
      // divide by when the drawdown is not.
      Ao += (double)(F.bo * k.wi * m.Mo + F.rv * F.bg * k.wi * m.Mg);
      Aw += (double)(F.bw * k.wi * m.Mw);
      Ag += (double)(F.bg * k.wi * m.Mg + F.rs * F.bo * k.wi * m.Mo);
    }

    // What the well DELIVERED, not what it was asked for.
    //
    // This block used to replace the rates computed above with a
    // reconstruction from applied_rate() -- the rate the frozen-mobility
    // target solve says the well can make -- split across the phases by their
    // productivity ratios.  That is the rate the well was HELD TO, evaluated
    // when the controls were set, and it is not what the reservoir gave up by
    // the end of the step: the mobilities move, the bottom-hole limit binds
    // on some completions and not others, and crossflow appears.
    //
    // The measured version above is the same expression the assembly kernel
    // integrates, evaluated on the same converged state, so it IS the run.
    // The reconstruction is a model of the run, and the two disagreed by 13%
    // on oil -- and far more on water, because the water rate came out as the
    // oil rate times Aw/Ao, a water cut implied by mobility ratios rather
    // than the one the well actually made.
    //
    // applied_rate() keeps its real job, which is setting the flowing
    // pressure in apply_producer_rate_targets().  A well that meets its
    // target still reports it, because at the pressure that meets the target
    // the measured rate IS the target.  A well that misses reports the miss,
    // which is the point.
    // The black-oil gas equation and schedule controls use SCF/day internally.
    // Eclipse FIELD summary WGPR is MSCF/day.  Writing this value unchanged
    // made every NVRS gas rate and GOR exactly 1000x too large even when the
    // underlying component flux was right.
    row.qg *= (Real)0.001;
    row.xflow_g *= (Real)0.001;   // the same scf/day -> Mscf/day
    // Match the Flow summary convention used by the comparison cases: WBHP
    // is zero when the schedule state is SHUT. This is reporting only; the
    // last converged well pressure in the reservoir/control state is neither
    // cleared nor fed back into the solve.
    if (row.type == 0) row.bhp = (Real)0;
    out.push_back(row);
  }
  return out;
}

void reduce_well_rates(std::vector<WellRateRow>& rows, int nranks) {
  // No `rows.empty()` guard: the row list comes from the schedule, so it is
  // the same length on every rank, and skipping the collective on whichever
  // rank happened to produce none would hang the rest.
  if (nranks <= 1) return;
  const size_t n = rows.size();
  // Eight summed values per well: the three surface rates, reservoir voidage,
  // the three crossflow
  // magnitudes, and the count of completions running backwards.  Crossflow adds
  // across ranks exactly as the rates do -- a horizontal Norne producer can be
  // cut by the partition, and a well that flows both ways is most likely to be
  // one that is -- and it rides in the same collective rather than a second
  // one, because the cost here is latency and not bytes.
  std::vector<double> sum(8 * n), mx(2 * n);
  for (size_t i = 0; i < n; ++i) {
    sum[8 * i + 0] = (double)rows[i].qo;
    sum[8 * i + 1] = (double)rows[i].qw;
    sum[8 * i + 2] = (double)rows[i].qg;
    sum[8 * i + 3] = (double)rows[i].qres;
    sum[8 * i + 4] = (double)rows[i].xflow_o;
    sum[8 * i + 5] = (double)rows[i].xflow_w;
    sum[8 * i + 6] = (double)rows[i].xflow_g;
    sum[8 * i + 7] = (double)rows[i].nperf_reverse;
    // BHP is a property of the well, not a quantity to add up; a rank holding
    // none of its completions reports zero, so the maximum picks the rank that
    // does hold them.  nperf rides along as a max only to stay diagnostic.
    mx[2 * i + 0] = (double)rows[i].bhp;
    mx[2 * i + 1] = (double)rows[i].nperf;
  }
  MPI_Allreduce(MPI_IN_PLACE, sum.data(), (int)sum.size(), MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, mx.data(), (int)mx.size(), MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  for (size_t i = 0; i < n; ++i) {
    rows[i].qo = (Real)sum[8 * i + 0];
    rows[i].qw = (Real)sum[8 * i + 1];
    rows[i].qg = (Real)sum[8 * i + 2];
    rows[i].qres = (Real)sum[8 * i + 3];
    rows[i].xflow_o = (Real)sum[8 * i + 4];
    rows[i].xflow_w = (Real)sum[8 * i + 5];
    rows[i].xflow_g = (Real)sum[8 * i + 6];
    rows[i].nperf_reverse = (int)sum[8 * i + 7];
    rows[i].bhp = (Real)mx[2 * i + 0];
    rows[i].nperf = (int)mx[2 * i + 1];
  }
}


void apply_injector_rate_targets(Reservoir& r, const FieldModel& fm, int step,
                                 const Deck& d, const PvtTables* pvt,
                                 bool handoff_to_producer) {
  ControlHandoff& handoff = control_handoff();
  handoff.ready = false;
  const bool reuse_history =
      handoff_to_producer && handoff.history_valid &&
      handoff.reservoir == &r && handoff.field == &fm &&
      handoff.accepted_state_id == r.accepted_state_id;
  // Primary-variable meaning is Newton state, not accepted-timestep history.
  // blackoil_pvswitch_adapt() may change Sg/Rs/Rv after every update, so the
  // host well equations must see the current device meanings even while the
  // slower DRSDT/VAPPARS history arrays remain cached for this accepted state.
  refresh_gas_primary(fm, r);
  if (!reuse_history) {
    refresh_drsdt_cap(fm, r);
    refresh_vappars_host(fm, r);
  }
  const int cells = fm.part.n_owned;
  const int bs = r.bs;
  if (step < 0 || step >= fm.schedule.n_steps() || cells <= 0 || !r.u ||
      !r.wellcells)
    return;
  const ScheduleStep& sc = fm.schedule.step(step);
  const std::vector<int>& l2a = fm.part.local_of_active;
  ControlStateIndexCache& state_index =
      control_state_indices(r, sc, l2a, cells, bs);

  std::vector<Real> u;
  if (handoff_to_producer) u = std::move(handoff.u);
  gather_control_state(r, bs, state_index, u);
  auto U = [&](int loc, int v) -> Real {
    const int packed = state_index.packed_of_local[(size_t)loc];
    return u[(size_t)bs * packed + v];
  };
  auto* wc = static_cast<WellCell*>(r.wellcells_host);
  if (!wc) return;
  // Only scheduled completion cells can carry injector coupling.  Clearing
  // the entire owned grid here made every Newton refresh O(number of cells):
  // 1.12 million writes for SPE10 to reset at most 425 live entries.
  if (r.well_schur_enabled == 8 || r.well_schur_enabled == 9) {
    for (const int loc : state_index.loc) {
      if (loc < 0 || loc >= cells) continue;
      wc[(size_t)loc].inj_dqdp = 0;
      wc[(size_t)loc].inj_dqdu[0] = 0;
      wc[(size_t)loc].inj_dqdu[1] = 0;
      wc[(size_t)loc].inj_dqdu[2] = 0;
      wc[(size_t)loc].inj_rate_coupled = 0;
      for (int j = 0; j < 9; ++j) {
        wc[(size_t)loc].inj_direct[j] = 0;
        wc[(size_t)loc].inj_dgdu[j] = 0;
        wc[(size_t)loc].inj_drdg[j] = 0;
      }
      wc[(size_t)loc].inj_xflow_coupled = 0;
    }
  }

  if (!reuse_history) {
    refresh_hyst_host(fm, r);
    if (handoff_to_producer) {
      handoff.reservoir = &r;
      handoff.field = &fm;
      handoff.accepted_state_id = r.accepted_state_id;
      handoff.history_valid = true;
    }
  }
  SatTables T = host_sat_tables(d, fm, r);
  const std::vector<HostPvt>& hp = host_pvt_regions(pvt);

  const size_t nw = sc.wells.size();

  // apply_schedule_step has already applied WEFAC and GCONINJE and placed the
  // requested rate on the connections. Preserve that well total if this
  // report step is retried; otherwise a pressure-limited call could silently
  // turn its delivered rate into the retry's target.
  struct TargetCache {
    const FieldModel* model = nullptr;
    const void* wellcells = nullptr;
    int step = -1;
    std::vector<double> q;
  };
  static TargetCache cache;
  if (cache.model != &fm || cache.wellcells != (const void*)r.wellcells ||
      cache.step != step || cache.q.size() != nw) {
    cache.model = &fm;
    cache.wellcells = (const void*)r.wellcells;
    cache.step = step;
    cache.q.assign(nw, 0.0);
    for (size_t iw = 0; iw < nw; ++iw) {
      const Well& w = sc.wells[iw];
      if (!w.state.open || w.state.type != WellType::Injector) continue;
      for (const Completion& c : w.completions) {
        if (!c.open || c.cell < 0 || c.cell >= (int)l2a.size()) continue;
        const int loc = l2a[c.cell];
        if (loc < 0 || loc >= cells || !wc[(size_t)loc].is_inj) continue;
        if (w.state.inj_phase == InjectedPhase::Water)
          cache.q[iw] += wc[(size_t)loc].qw_inj;
        else if (w.state.inj_phase == InjectedPhase::Oil)
          cache.q[iw] += wc[(size_t)loc].qo_inj;
        else
          cache.q[iw] += wc[(size_t)loc].qg_inj;
      }
    }
    if (fm.part.nranks > 1 && !cache.q.empty())
      MPI_Allreduce(MPI_IN_PLACE, cache.q.data(), (int)cache.q.size(),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  struct InjectorScratch {
    std::vector<size_t> off;
    std::vector<double> ca, dp_local;
    // Mode-9 injector crossflow coefficients per completion:
    // total reservoir mobility, datum-corrected pressure, three surface
    // component coefficients on a common STB-equivalent basis, then B/Rs/Rv.
    std::vector<double> xcoef, xderiv, xrate;
    std::vector<WellConn> conns;
    size_t track_signature = 0;
    std::vector<std::vector<size_t>> track;
  };
  static thread_local InjectorScratch scratch;
  const size_t track_signature = well_track_signature(sc.wells);
  if (scratch.track_signature != track_signature ||
      scratch.track.size() != nw) {
    scratch.track_signature = track_signature;
    scratch.track.resize(nw);
    for (size_t iw = 0; iw < nw; ++iw)
      scratch.track[iw] = well_track_order(sc.wells[iw], fm);
  }
  auto& off = scratch.off;
  off.assign(nw + 1, 0);
  for (size_t iw = 0; iw < nw; ++iw)
    off[iw + 1] = off[iw] + sc.wells[iw].completions.size();
  auto& ca = scratch.ca;  // A, A*theta per completion
  ca.assign(2 * off[nw], 0.0);
  // Hydrostatic correction belongs to a completion, not to every reservoir
  // cell.  Keeping a cells-sized zeroed array here wrote 355 kB on every
  // control refresh to use at most a few hundred entries.
  auto& dp_local = scratch.dp_local;
  dp_local.assign(off[nw], 0.0);
  auto& xcoef = scratch.xcoef;
  auto& xderiv = scratch.xderiv;
  if (r.well_schur_enabled == 9)
    xcoef.assign(10 * off[nw], 0.0);
  else
    xcoef.clear();
  if (r.well_schur_enabled == 9)
    xderiv.assign(30 * off[nw], 0.0);
  else
    xderiv.clear();

  for (size_t iw = 0; iw < nw; ++iw) {
    const Well& w = sc.wells[iw];
    if (!w.state.open || w.state.type != WellType::Injector ||
        !(cache.q[iw] > 0.0))
      continue;
    const std::vector<size_t>& track = scratch.track[iw];
    const double ref_depth = well_reference_depth(w, fm, track);
    for (size_t ic = 0; ic < w.completions.size(); ++ic) {
      const Completion& c = w.completions[ic];
      if (!c.open || c.cell < 0 || c.cell >= (int)l2a.size()) continue;
      const int loc = l2a[c.cell];
      if (loc < 0 || loc >= cells) continue;
      const WellCell& k = wc[(size_t)loc];
      if (!k.is_inj || !(k.wi > 0)) continue;
      const Real pressure = U(loc, 0);
      const Real sw = U(loc, 1);
      const Real x = (bs == 3) ? U(loc, 2) : (Real)0;
      const Real sg = physical_sg_at(fm, r, c.cell, sw, x);
      const Real po = oil_pressure_at(fm, r, c.cell, pressure, sw, sg, T);
      const DevicePvt pv = pvt_for_active(hp, fm, c.cell);
      const PvtCell F = pv.valid
                            ? pvt_cell_at(pv, pressure, sw, x, c.cell, fm, r)
                            : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                                             r.uo, r.uw, r.ug);
      const Mob3 m = well_relperm(bs, sw, sg, T, F.muw, F.muo, F.mug,
                              (Real)1, (Real)1, (Real)1, r.swi, r.swr,
                              c.cell);
      const double lam = std::max(0.0, (double)(m.Mw + m.Mo + m.Mg));
      double b = (double)F.bg;
      double rho = (double)r.rhog * b / (double)units::kFt3PerBbl;
      if (w.state.inj_phase == InjectedPhase::Water) {
        b = (double)F.bw;
        rho = (double)r.rhow * b;
      } else if (w.state.inj_phase == InjectedPhase::Oil) {
        b = (double)F.bo;
        rho = (double)r.rhoo * b;
      }
      const double dp = r.gravity
                            ? (double)r.grav_grad * rho *
                                  ((double)fm.grid.depth[c.cell] - ref_depth)
                            : 0.0;
      const double A = b * (double)k.wi * lam;
      const size_t ci = off[iw] + ic;
      ca[2 * ci] += A;
      ca[2 * ci + 1] += A * ((double)po - dp);
      dp_local[ci] = dp;
      if (r.well_schur_enabled == 9) {
        const double kvol = (double)units::kFt3PerBbl;
        xcoef[10 * ci + 0] = (double)k.wi * lam;
        xcoef[10 * ci + 1] = (double)po - dp;
        xcoef[10 * ci + 2] =
            (double)F.bo * (double)k.wi * (double)m.Mo +
            (double)F.rv * (double)F.bg * (double)k.wi * (double)m.Mg;
        xcoef[10 * ci + 3] =
            (double)F.bw * (double)k.wi * (double)m.Mw;
        xcoef[10 * ci + 4] =
            ((double)F.bg * (double)k.wi * (double)m.Mg +
             (double)F.rs * (double)F.bo * (double)k.wi * (double)m.Mo) /
            kvol;
        xcoef[10 * ci + 5] = (double)F.bo;
        xcoef[10 * ci + 6] = (double)F.bw;
        xcoef[10 * ci + 7] = (double)F.bg;
        xcoef[10 * ci + 8] = (double)F.rs;
        xcoef[10 * ci + 9] = (double)F.rv;

        auto crossflow_coeff = [&](Real pe, Real swe, Real xe, double* z) {
          const Real sge = physical_sg_at(fm, r, c.cell, swe, xe);
          const Real poe = oil_pressure_at(fm, r, c.cell, pe, swe, sge, T);
          const PvtCell fe = pv.valid
              ? pvt_cell_at(pv, pe, swe, xe, c.cell, fm, r)
              : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                               r.uo, r.uw, r.ug);
          const Mob3 me = well_relperm(bs, swe, sge, T, fe.muw, fe.muo, fe.mug,
                                   (Real)1, (Real)1, (Real)1, r.swi, r.swr,
                                   c.cell);
          const double le = (double)k.wi *
              std::max(0.0, (double)(me.Mw + me.Mo + me.Mg));
          double be = (double)fe.bg;
          double rhoe =
              (double)r.rhog * be / (double)units::kFt3PerBbl;
          if (w.state.inj_phase == InjectedPhase::Water) {
            be = (double)fe.bw;
            rhoe = (double)r.rhow * be;
          } else if (w.state.inj_phase == InjectedPhase::Oil) {
            be = (double)fe.bo;
            rhoe = (double)r.rhoo * be;
          }
          const double dpe = r.gravity
              ? (double)r.grav_grad * rhoe *
                    ((double)fm.grid.depth[c.cell] - ref_depth)
              : 0.0;
          z[0] = le;
          z[1] = (double)poe - dpe;
          z[2] = (double)fe.bo * (double)k.wi * (double)me.Mo +
                 (double)fe.rv * (double)fe.bg *
                     (double)k.wi * (double)me.Mg;
          z[3] = (double)fe.bw * (double)k.wi * (double)me.Mw;
          z[4] = ((double)fe.bg * (double)k.wi * (double)me.Mg +
                  (double)fe.rs * (double)fe.bo *
                      (double)k.wi * (double)me.Mo) /
                 (double)units::kFt3PerBbl;
          z[5] = (double)fe.bo; z[6] = (double)fe.bw;
          z[7] = (double)fe.bg; z[8] = (double)fe.rs;
          z[9] = (double)fe.rv;
        };
        const Real uv[3] = {pressure, sw, x};
        const Real h[3] = {
            (Real)std::max(1.0e-3,
                           1.0e-4 * std::fabs((double)pressure)),
            (Real)1.0e-6,
            (Real)std::max(1.0e-7,
                           1.0e-5 * std::max(std::fabs((double)x), 1.0e-3))};
        for (int v = 0; v < 3; ++v) {
          Real up[3] = {uv[0], uv[1], uv[2]};
          Real um[3] = {uv[0], uv[1], uv[2]};
          up[v] += h[v]; um[v] -= h[v];
          if (v == 0 && um[v] <= 0) um[v] = uv[v];
          double zp[10], zm[10];
          crossflow_coeff(up[0], up[1], up[2], zp);
          crossflow_coeff(um[0], um[1], um[2], zm);
          const double den = (double)(up[v] - um[v]);
          if (den == 0.0) continue;
          for (int q = 0; q < 10; ++q)
            xderiv[30 * ci + 3 * q + v] = (zp[q] - zm[q]) / den;
        }
      }
    }
  }
  if (fm.part.nranks > 1 && !ca.empty())
    MPI_Allreduce(MPI_IN_PLACE, ca.data(), (int)ca.size(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
  if (fm.part.nranks > 1 && !xcoef.empty())
    MPI_Allreduce(MPI_IN_PLACE, xcoef.data(), (int)xcoef.size(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
  if (fm.part.nranks > 1 && !xderiv.empty())
    MPI_Allreduce(MPI_IN_PLACE, xderiv.data(), (int)xderiv.size(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

  double asked = 0.0, delivered = 0.0;
  int controlled = 0, limited = 0;
  for (size_t iw = 0; iw < nw; ++iw) {
    const Well& w = sc.wells[iw];
    if (!w.state.open || w.state.type != WellType::Injector ||
        !(cache.q[iw] > 0.0))
      continue;
    double ceiling = (std::isfinite((double)w.state.bhp_limit) &&
                      w.state.bhp_limit > 0)
                         ? (double)w.state.bhp_limit
                         : (double)d.inj_bhp;
    if (!(ceiling > 0.0)) continue;
    auto& conns = scratch.conns;
    conns.clear();
    conns.reserve(w.completions.size());
    for (size_t ic = 0; ic < w.completions.size(); ++ic) {
      const size_t ci = off[iw] + ic;
      const double A = ca[2 * ci];
      if (A > 0.0)
        conns.push_back(WellConn{A, ca[2 * ci + 1] / A, (int)ic});
    }
    const WellPwf sol = solve_injector_pwf(conns, cache.q[iw], ceiling);
    if (conns.empty() || !(sol.pwf > 0.0) || !std::isfinite(sol.pwf))
      continue;

    if (r.well_schur_enabled == 9) {
      const int well_modes = bs == 2 ? 2 : 3;
      const double kvol = (double)units::kFt3PerBbl;
      double target_common[3] = {0.0, 0.0, 0.0};
      int target_eq = 2;
      double target_total = cache.q[iw] / kvol;
      if (w.state.inj_phase == InjectedPhase::Water) {
        target_eq = 1;
        target_total = cache.q[iw];
      } else if (w.state.inj_phase == InjectedPhase::Oil) {
        target_eq = 0;
        target_total = cache.q[iw];
      }
      target_common[target_eq] = target_total;
      auto& xr = scratch.xrate;
      xr.assign(3 * off[nw], 0.0);
      auto evaluate_crossflow = [&](double trial_bhp, Real mix[3],
                                    bool save_rates) {
        double produced[3] = {0.0, 0.0, 0.0};
        for (size_t ic = 0; ic < w.completions.size(); ++ic) {
          const size_t ci = off[iw] + ic;
          const double theta = xcoef[10 * ci + 1];
          if (!(theta > trial_bhp)) continue;
          const double draw = theta - trial_bhp;
          for (int ph = 0; ph < 3; ++ph)
            produced[ph] += xcoef[10 * ci + 2 + ph] * draw;
        }
        const double stream_total =
            target_total + produced[0] + produced[1] + produced[2];
        if (!(stream_total > 0.0)) return -target_total;
        for (int ph = 0; ph < 3; ++ph)
          mix[ph] = (Real)((produced[ph] + target_common[ph]) /
                           stream_total);

        double injected_total = 0.0;
        if (save_rates)
          for (size_t ic = 0; ic < w.completions.size(); ++ic) {
            const size_t ci = off[iw] + ic;
            xr[3 * ci] = xr[3 * ci + 1] = xr[3 * ci + 2] = 0.0;
          }
        for (size_t ic = 0; ic < w.completions.size(); ++ic) {
          const size_t ci = off[iw] + ic;
          const double l = xcoef[10 * ci];
          const double theta = xcoef[10 * ci + 1];
          if (!(l > 0.0)) continue;
          if (theta > trial_bhp) {
            if (save_rates) {
              const double draw = theta - trial_bhp;
              for (int ph = 0; ph < 3; ++ph)
                xr[3 * ci + ph] =
                    -xcoef[10 * ci + 2 + ph] * draw;
            }
            continue;
          }
          const double bo = xcoef[10 * ci + 5];
          const double bw = xcoef[10 * ci + 6];
          const double bg = xcoef[10 * ci + 7];
          const double rs = xcoef[10 * ci + 8];
          const double rv = xcoef[10 * ci + 9];
          const double bgc = bg / kvol;
          const double rsc = rs / kvol;
          const double rvc = rv * kvol;
          const double dis = std::max(1.0e-12, 1.0 - rs * rv);
          const double vr =
              (double)mix[1] / bw +
              ((double)mix[0] - rvc * (double)mix[2]) / (bo * dis) +
              ((double)mix[2] - rsc * (double)mix[0]) / (bgc * dis);
          if (!(vr > 1.0e-20)) continue;
          const double qs = l * (trial_bhp - theta) / vr;
          injected_total += qs;
          if (save_rates)
            for (int ph = 0; ph < 3; ++ph)
              xr[3 * ci + ph] = (double)mix[ph] * qs;
        }
        return injected_total - stream_total;
      };

      Real mix[3] = {0, 0, 0};
      double lo = 0.0, hi = ceiling;
      const double fhi = evaluate_crossflow(hi, mix, false);
      bool limited_x = fhi < 0.0;
      if (!limited_x) {
        for (int it = 0; it < 40; ++it) {
          const double mid = 0.5 * (lo + hi);
          const double fmix = evaluate_crossflow(mid, mix, false);
          if (fmix < 0.0) lo = mid;
          else hi = mid;
        }
      }
      const double xbhp = limited_x ? ceiling : 0.5 * (lo + hi);
      evaluate_crossflow(xbhp, mix, true);

      double well_D[9] = {};
      const double g0[3] = {xbhp, (double)mix[0], (double)mix[1]};
      auto rate_from_coeff = [&](const double* z, const double g[3],
                                 double q[3]) {
        q[0] = q[1] = q[2] = 0.0;
        // Oil-water has one independent mixture fraction.  Preserve the
        // three-phase [BHP, oil, water] coordinates, but derive water from
        // oil when the gas equation is absent.
        const double co = g[1];
        const double cw = bs == 2 ? 1.0 - co : g[2];
        const double cg = bs == 2 ? 0.0 : 1.0 - co - cw;
        if (z[1] > g[0]) {
          const double draw = z[1] - g[0];
          q[0] = -z[2] * draw;
          q[1] = -z[3] * draw;
          q[2] = -z[4] * draw;
          return;
        }
        if (!(z[0] > 0.0)) return;
        const double bgc = z[7] / kvol;
        const double rsc = z[8] / kvol;
        const double rvc = z[9] * kvol;
        const double dis = std::max(1.0e-12, 1.0 - z[8] * z[9]);
        const double vr =
            cw / z[6] +
            (co - rvc * cg) / (z[5] * dis) +
            (cg - rsc * co) / (bgc * dis);
        if (!(vr > 1.0e-20)) return;
        const double qs = z[0] * (g[0] - z[1]) / vr;
        q[0] = co * qs;
        q[1] = cw * qs;
        q[2] = cg * qs;
      };

      if (!limited_x) {
        const double gh[3] = {
            std::max(1.0e-5, 1.0e-6 * std::fabs(g0[0])),
            1.0e-6, 1.0e-6};
        for (int mode = 0; mode < well_modes; ++mode) {
          double gp[3] = {g0[0], g0[1], g0[2]};
          double gm[3] = {g0[0], g0[1], g0[2]};
          gp[mode] += gh[mode];
          gm[mode] -= gh[mode];
          const double den = gp[mode] - gm[mode];
          for (size_t ic = 0; ic < w.completions.size(); ++ic) {
            const size_t ci = off[iw] + ic;
            double qp[3], qm[3];
            rate_from_coeff(&xcoef[10 * ci], gp, qp);
            rate_from_coeff(&xcoef[10 * ci], gm, qm);
            for (int eq = 0; eq < bs; ++eq)
              well_D[3 * eq + mode] += (qp[eq] - qm[eq]) / den;
          }
        }
      }
      auto solve_D = [&](const double rhs[3], double ans[3]) {
        double a[3][4] = {};
        for (int i = 0; i < well_modes; ++i) {
          for (int j = 0; j < well_modes; ++j) a[i][j] = well_D[3 * i + j];
          a[i][well_modes] = rhs[i];
        }
        for (int col = 0; col < well_modes; ++col) {
          int piv = col;
          for (int row = col + 1; row < well_modes; ++row)
            if (std::fabs(a[row][col]) > std::fabs(a[piv][col])) piv = row;
          if (!(std::fabs(a[piv][col]) > 1.0e-20)) {
            ans[0] = ans[1] = ans[2] = 0.0;
            return false;
          }
          if (piv != col)
            for (int j = col; j <= well_modes; ++j)
              std::swap(a[col][j], a[piv][j]);
          const double inv = 1.0 / a[col][col];
          for (int j = col; j <= well_modes; ++j) a[col][j] *= inv;
          for (int row = 0; row < well_modes; ++row) {
            if (row == col) continue;
            const double f = a[row][col];
            for (int j = col; j <= well_modes; ++j)
              a[row][j] -= f * a[col][j];
          }
        }
        for (int i = 0; i < well_modes; ++i) ans[i] = a[i][well_modes];
        return true;
      };

      ++controlled;
      if (limited_x) ++limited;
      asked += cache.q[iw];
      double delivered_target = 0.0;
      double net_component[3] = {};
      for (size_t ic = 0; ic < w.completions.size(); ++ic) {
        const Completion& c = w.completions[ic];
        if (!c.open || c.cell < 0 || c.cell >= (int)l2a.size()) continue;
        const int loc = l2a[c.cell];
        if (loc < 0 || loc >= cells || !wc[(size_t)loc].is_inj) continue;
        const size_t ci = off[iw] + ic;
        WellCell& k = wc[(size_t)loc];
        k.bhp = (Real)xbhp;
        k.pwf = 0; // host crossflow residual already applies the BHP limit
        k.qo_inj = (Real)xr[3 * ci];
        k.qw_inj = (Real)xr[3 * ci + 1];
        k.qg_inj = (Real)(kvol * xr[3 * ci + 2]);
        net_component[0] += (double)k.qo_inj;
        net_component[1] += (double)k.qw_inj;
        net_component[2] += (double)k.qg_inj;
        k.mix_o = mix[0]; k.mix_w = mix[1]; k.mix_g = mix[2];
        if (!limited_x) {
          double dq_dg[3][3] = {};
          const double gh[3] = {
              std::max(1.0e-5, 1.0e-6 * std::fabs(g0[0])),
              1.0e-6, 1.0e-6};
          for (int mode = 0; mode < well_modes; ++mode) {
            double gp[3] = {g0[0], g0[1], g0[2]};
            double gm[3] = {g0[0], g0[1], g0[2]};
            gp[mode] += gh[mode]; gm[mode] -= gh[mode];
            double qp[3], qm[3];
            rate_from_coeff(&xcoef[10 * ci], gp, qp);
            rate_from_coeff(&xcoef[10 * ci], gm, qm);
            for (int eq = 0; eq < bs; ++eq)
              dq_dg[eq][mode] = (qp[eq] - qm[eq]) /
                                (gp[mode] - gm[mode]);
          }
          double dq_du[3][3] = {};
          for (int zq = 0; zq < 10; ++zq) {
            const double base = xcoef[10 * ci + zq];
            const double hz =
                std::max(1.0e-9, 1.0e-6 * std::max(1.0, std::fabs(base)));
            double zp[10], zm[10];
            for (int q = 0; q < 10; ++q)
              zp[q] = zm[q] = xcoef[10 * ci + q];
            zp[zq] += hz; zm[zq] -= hz;
            double qp[3], qm[3];
            rate_from_coeff(zp, g0, qp);
            rate_from_coeff(zm, g0, qm);
            for (int eq = 0; eq < bs; ++eq) {
              const double dq_dz = (qp[eq] - qm[eq]) / (2.0 * hz);
              for (int v = 0; v < bs; ++v)
                dq_du[eq][v] +=
                    dq_dz * xderiv[30 * ci + 3 * zq + v];
            }
          }
          for (int v = 0; v < bs; ++v) {
            const double rhs[3] =
                {-dq_du[0][v], -dq_du[1][v], -dq_du[2][v]};
            double dg[3] = {};
            solve_D(rhs, dg);
            for (int mode = 0; mode < well_modes; ++mode)
              k.inj_dgdu[3 * mode + v] = (Real)dg[mode];
          }
          for (int eq = 0; eq < bs; ++eq) {
            const double scale = eq == 2 ? kvol : 1.0;
            for (int v = 0; v < bs; ++v)
              k.inj_direct[3 * eq + v] =
                  (Real)(scale * dq_du[eq][v]);
            for (int mode = 0; mode < well_modes; ++mode)
              k.inj_drdg[3 * eq + mode] =
                  (Real)(-scale * dq_dg[eq][mode]);
          }
          k.inj_xflow_coupled = 1;
        }
        if (target_eq == 0) delivered_target += xr[3 * ci];
        else if (target_eq == 1) delivered_target += xr[3 * ci + 1];
        else delivered_target += kvol * xr[3 * ci + 2];
      }
      delivered += std::max(0.0, delivered_target);
      if (fm.part.rank == 0 && std::getenv("NVRS_DEBUG_WELL_BALANCE"))
        std::printf("  [injector-balance] %s net oil/water/gas "
                    "[%+.9g %+.9g %+.9g], BHP %.9g\n", w.name.c_str(),
                    net_component[0], net_component[1], net_component[2],
                    xbhp);
      continue;
    }

    ++controlled;
    if (sol.limited) ++limited;
    asked += cache.q[iw];
    delivered += sol.delivered;
    for (size_t ic = 0; ic < w.completions.size(); ++ic) {
      const Completion& c = w.completions[ic];
      if (!c.open || c.cell < 0 || c.cell >= (int)l2a.size()) continue;
      const int loc = l2a[c.cell];
      if (loc < 0 || loc >= cells || !wc[(size_t)loc].is_inj) continue;
      const size_t ci = off[iw] + ic;
      const double A = ca[2 * ci];
      const double theta = A > 0.0 ? ca[2 * ci + 1] / A : sol.pwf;
      const Real q = (Real)(A * std::max(0.0, sol.pwf - theta));
      WellCell& k = wc[(size_t)loc];
      // `bhp` is the operating datum pressure reported for the well. `pwf`
      // is also the assembly's injectivity ceiling. A rate-controlled well
      // must retain the deck's upper limit there: using the solved operating
      // pressure makes any pressure rise during Newton throttle the requested
      // rate immediately, even though the well can raise its BHP and remain
      // on rate control. When the ceiling truly binds, sol.pwf == ceiling and
      // the two values naturally coincide.
      k.bhp = (Real)sol.pwf;
      k.pwf = (Real)(ceiling + dp_local[ci]);
      k.qw_inj = k.qo_inj = k.qg_inj = 0;
      if (w.state.inj_phase == InjectedPhase::Water) k.qw_inj = q;
      else if (w.state.inj_phase == InjectedPhase::Oil) k.qo_inj = q;
      else k.qg_inj = q;
      if (r.well_schur_enabled == 8 && !sol.limited &&
          A > 0.0 && sol.pwf > theta) {
        k.inj_dqdp = (Real)A;
        k.inj_rate_coupled = 1;
        const Real p0 = U(loc, 0);
        const Real sw0 = U(loc, 1);
        const Real x0 = bs == 3 ? U(loc, 2) : (Real)0;
        const DevicePvt pv = pvt_for_active(hp, fm, c.cell);
        auto coeff = [&](Real pe, Real swe, Real xe, Real& ae,
                         Real& thetae) {
          const Real sge = physical_sg_at(fm, r, c.cell, swe, xe);
          thetae = oil_pressure_at(fm, r, c.cell, pe, swe, sge, T);
          const PvtCell fe = pv.valid
              ? pvt_cell_at(pv, pe, swe, xe, c.cell, fm, r)
              : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                               r.uo, r.uw, r.ug);
          const Mob3 me = well_relperm(bs, swe, sge, T, fe.muw, fe.muo, fe.mug,
                                   (Real)1, (Real)1, (Real)1, r.swi, r.swr,
                                   c.cell);
          Real be = fe.bg;
          if (w.state.inj_phase == InjectedPhase::Water) be = fe.bw;
          else if (w.state.inj_phase == InjectedPhase::Oil) be = fe.bo;
          ae = be * k.wi * (me.Mw + me.Mo + me.Mg);
        };
        const Real uv[3] = {p0, sw0, x0};
        const Real h[3] = {
            (Real)std::max(1.0e-3, 1.0e-4 * std::fabs((double)p0)),
            (Real)1.0e-6,
            (Real)std::max(1.0e-7,
                           1.0e-5 * std::max(std::fabs((double)x0), 1.0e-3))};
        for (int v = 0; v < 3; ++v) {
          Real up[3] = {uv[0], uv[1], uv[2]};
          Real um[3] = {uv[0], uv[1], uv[2]};
          up[v] += h[v];
          um[v] -= h[v];
          if (v == 0 && um[v] <= 0) um[v] = uv[v];
          Real ap = 0, am = 0, tp = 0, tm = 0;
          coeff(up[0], up[1], up[2], ap, tp);
          coeff(um[0], um[1], um[2], am, tm);
          const Real den = up[v] - um[v];
          if (den != (Real)0) {
            const Real da = (ap - am) / den;
            const Real dt = (tp - tm) / den;
            k.inj_dqdu[v] =
                da * (Real)(sol.pwf - theta) - (Real)A * dt;
          }
        }
      }
    }
  }
  // In a paired Newton refresh the producer consumes this host array
  // immediately and performs the one final upload.  No kernel can observe the
  // injector-only intermediate state between these two calls.
  if (!handoff_to_producer)
    upload_schedule_wellcells(r, sc, l2a, cells, wc);
  if (handoff_to_producer) {
    handoff.reservoir = &r;
    handoff.field = &fm;
    handoff.step = step;
    handoff.accepted_state_id = r.accepted_state_id;
    handoff.u = std::move(u);
    handoff.ready = true;
  }
  if (fm.part.rank == 0 && controlled > 0) {
    static int calls = 0;
    if ((calls++ % 40) == 0)
      std::printf("  [injector] %d common-BHP rate well(s): asked %.4g, "
                  "deliver %.4g (%.1f%%); %d at pressure limit\n",
                  controlled, asked, delivered,
                  asked > 0.0 ? 100.0 * delivered / asked : 0.0, limited);
  }
}


void apply_producer_rate_targets(Reservoir& r, const FieldModel& fm, int step,
                                 const Deck& d, const PvtTables* pvt,
                                 bool freeze_pressure_drop) {
  const int cells = fm.part.n_owned;
  const int bs = r.bs;
  if (step < 0 || step >= fm.schedule.n_steps()) return;
  const ScheduleStep& sc = fm.schedule.step(step);
  const std::vector<int>& l2a = fm.part.local_of_active;
  const size_t nw = sc.wells.size();
  ControlStateIndexCache& state_index =
      control_state_indices(r, sc, l2a, cells, bs);

  bool have = (cells > 0 && r.u != nullptr && r.wellcells != nullptr &&
               r.wellcells_host != nullptr);
  std::vector<Real> u;
  auto* wc = static_cast<WellCell*>(r.wellcells_host);
  ControlHandoff& handoff = control_handoff();
  const bool use_handoff =
      have && handoff.ready && handoff.reservoir == &r &&
      handoff.field == &fm && handoff.step == step &&
      handoff.accepted_state_id == r.accepted_state_id &&
      handoff.u.size() == (size_t)bs * state_index.loc.size();
  if (use_handoff) {
    u = std::move(handoff.u);
  } else if (have) {
    gather_control_state(r, bs, state_index, u);
  }
  auto U = [&](int loc, int v) -> Real {
    const int packed = state_index.packed_of_local[(size_t)loc];
    return u[(size_t)bs * packed + v];
  };
  handoff.ready = false;

  if (!use_handoff) {
    refresh_drsdt_cap(fm, r);
    refresh_gas_primary(fm, r);
    refresh_vappars_host(fm, r);
  }
  if (!use_handoff) refresh_hyst_host(fm, r);
  SatTables T = host_sat_tables(d, fm, r);
  const std::vector<HostPvt>& hp = host_pvt_regions(pvt);

  // OPM's SurfaceToReservoirVoidage::defineState() evaluates history RESV
  // controls at the hydrocarbon-pore-volume average of the FIP region.  The
  // old implementation averaged only the well's completions and weighted
  // them by mobility, which makes the target depend on where that particular
  // well happens to be perforated.  Norne's history controls are field-region
  // controls, so reproduce OPM's average here.  Keep a full-PV fallback for a
  // water-only region, exactly as OPM does.
  // This routine is called after every Newton update.  OPM's regional state
  // belongs to the report-step control state, so evaluate it once when the
  // step changes instead of scanning the entire reservoir on every Newton
  // iteration.  That keeps the correction out of the hot nonlinear path.
  struct RegionAverageCache {
    const FieldModel* model = nullptr;
    int step = -1;
    long long accepted_state_id = -1;
    double p = 0, rs = 0, rv = 0;
  };
  static RegionAverageCache region_average;
  if (region_average.model != &fm || region_average.step != step ||
      region_average.accepted_state_id != r.accepted_state_id) {
    double ravg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const bool gpu_average = have && blackoil_region_average(r, ravg);
    if (have && !gpu_average) {
      static thread_local std::vector<Real> region_u;
      region_u.resize((size_t)bs * cells);
      NVRS_CHECK_CUDA(cudaMemcpy(region_u.data(), r.u,
                                 sizeof(Real) * region_u.size(),
                                 cudaMemcpyDeviceToHost));
      const std::vector<int>& a2l = fm.part.active_of_local;
      for (int c = 0; c < cells; ++c) {
        const int a = (c < (int)a2l.size()) ? a2l[c] : -1;
        if (a < 0 || a >= (int)fm.grid.porv.size()) continue;
        const double pv_cell = (double)fm.grid.porv[a];
        if (!(pv_cell > 0)) continue;
        const Real p = region_u[(size_t)bs * c];
        const Real sw = (bs >= 2) ? region_u[(size_t)bs * c + 1] : (Real)0;
        const Real x = (bs >= 3) ? region_u[(size_t)bs * c + 2] : (Real)0;
        const Real sg = physical_sg_at(fm, r, a, sw, x);
        const DevicePvt cell_pvt = pvt_for_active(hp, fm, a);
        const PvtCell F = cell_pvt.valid
                              ? pvt_cell_at(cell_pvt, p, sw, x, a, fm, r)
                              : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                                               r.uo, r.uw, r.ug);
        const double hpv = pv_cell * std::max(0.0, 1.0 - (double)sw);
        if (hpv > 0) {
          ravg[0] += hpv;
          ravg[1] += hpv * (double)p;
          ravg[2] += hpv * (double)F.rs;
          ravg[3] += hpv * (double)F.rv;
        }
        ravg[4] += pv_cell;
        ravg[5] += pv_cell * (double)p;
        ravg[6] += pv_cell * (double)F.rs;
        ravg[7] += pv_cell * (double)F.rv;
      }
    }
    if (fm.part.nranks > 1)
      MPI_Allreduce(MPI_IN_PLACE, ravg, 8, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    const int roff = (ravg[0] > 0) ? 0 : 4;
    const double rweight = ravg[roff];
    region_average.model = &fm;
    region_average.step = step;
    region_average.accepted_state_id = r.accepted_state_id;
    region_average.p = rweight > 0 ? ravg[roff + 1] / rweight : 0.0;
    region_average.rs = rweight > 0 ? ravg[roff + 2] / rweight : 0.0;
    region_average.rv = rweight > 0 ? ravg[roff + 3] / rweight : 0.0;
  }
  const double region_p = region_average.p;
  const double region_rs = region_average.rs;
  const double region_rv = region_average.rv;

  // Two sums per well: SUM A_k and SUM A_k (p_k - dp_k), with A_k the phase
  // mobility the target is written in and dp_k the hydrostatic pressure from
  // the well's BHP reference depth to completion k.  OPM's StandardWell uses
  // well_pressure = bhp + connections.pressure_diff(perf); solving against a
  // single pressure at every perforation loses that completion correction.
  // Every rank walks the same well list, so the reduction below is the same
  // length everywhere.
  struct ProducerScratch {
    std::vector<double> acc, conn_acc, conn_dp;
    std::vector<double> direct_acc, direct_conn_acc, direct_conn_dp;
    std::vector<double> report_bhp_shift;
    std::vector<size_t> conn_off;
    std::vector<Real> conn_drdpw, conn_rev, conn_mob, conn_surface, conn_po;
    std::vector<Real> conn_scoef, conn_dA_du, conn_dtheta_du, conn_dmob_du;
    std::vector<Real> conn_dscoef_du, conn_dpvt_du;
    std::vector<double> conn_density, newton_conn_dp;
    std::vector<Real> target;
    std::vector<char> wants, grp_kind;
    std::vector<int> unconstrained;
    std::vector<std::string> grp_of;
    std::vector<WellSchurEntry> schur;
    std::vector<WellConn> conns;
    size_t track_signature = 0;
    std::vector<std::vector<size_t>> track;
  };
  static thread_local ProducerScratch scratch;
  const size_t track_signature = well_track_signature(sc.wells);
  if (scratch.track_signature != track_signature ||
      scratch.track.size() != nw) {
    scratch.track_signature = track_signature;
    scratch.track.resize(nw);
    for (size_t iw = 0; iw < nw; ++iw)
      scratch.track[iw] = well_track_order(sc.wells[iw], fm);
  }
  auto& acc = scratch.acc;
  acc.assign(2 * nw, 0.0);

  // Per-COMPLETION productivity and datum-corrected pressure, for the
  // no-crossflow solve in include/well_control.hpp.  The two well sums above
  // are not enough for it: it needs each completion's own threshold, because
  // which ones flow forward is the answer rather than an input.
  //
  // Indexed by a flat offset over the schedule's completion lists, which every
  // rank walks identically, so it reduces in the same collective as `acc` and
  // no rank has to know which completions another one owns.  Norne has 504
  // completions across 36 wells, so this is a few kilobytes.
  auto& conn_off = scratch.conn_off;
  conn_off.assign(nw + 1, 0);
  for (size_t iw = 0; iw < nw; ++iw)
    conn_off[iw + 1] = conn_off[iw] + sc.wells[iw].completions.size();
  const size_t n_conn = conn_off[nw];
  auto& conn_acc = scratch.conn_acc;   // A_k, A_k * theta_k
  conn_acc.assign(2 * n_conn, 0.0);
  // Frozen-state coefficients for the pressure-only eliminated-well
  // Jacobian.  Reuse the PVT/relperm evaluation already needed by the control
  // solve instead of repeating that expensive host work in the assignment
  // pass.
  auto& conn_drdpw = scratch.conn_drdpw;
  auto& conn_rev = scratch.conn_rev; // WI*lambda, bo,bw,bg,rs,rv
  auto& conn_mob = scratch.conn_mob; // Mo,Mw,Mg from the coefficient pass
  auto& conn_surface = scratch.conn_surface; // produced O,W,G components
  auto& conn_scoef = scratch.conn_scoef;     // surface rate / drawdown
  auto& conn_po = scratch.conn_po;
  auto& conn_density = scratch.conn_density;
  conn_drdpw.assign(3 * n_conn, 0);
  conn_rev.assign(6 * n_conn, 0);
  conn_mob.assign(3 * n_conn, 0);
  conn_surface.assign(3 * n_conn, 0);
  conn_scoef.assign(3 * n_conn, 0);
  conn_po.assign(n_conn, 0);
  auto& conn_dA_du = scratch.conn_dA_du;
  auto& conn_dtheta_du = scratch.conn_dtheta_du;
  auto& conn_dmob_du = scratch.conn_dmob_du;
  auto& conn_dscoef_du = scratch.conn_dscoef_du;
  auto& conn_dpvt_du = scratch.conn_dpvt_du; // bo,bw,bg,rs,rv by primary
  if (r.well_schur_enabled > 1) {
    conn_dA_du.assign(3 * n_conn, 0);
    conn_dtheta_du.assign(3 * n_conn, 0);
    conn_dmob_du.assign(9 * n_conn, 0);
    conn_dscoef_du.assign(9 * n_conn, 0);
    conn_dpvt_du.assign(15 * n_conn, 0);
  }
  conn_density.assign(n_conn, 0.0);
  // Like conn_acc, the hydrostatic correction is indexed by completion.  A
  // cells-sized scratch array needlessly zeroed 355 kB on every Newton and
  // line-search evaluation.
  auto& conn_dp = scratch.conn_dp;
  auto& target = scratch.target;
  auto& wants = scratch.wants;
  auto& unconstrained = scratch.unconstrained;
  conn_dp.assign(n_conn, 0.0);
  target.assign(nw, 0);
  wants.assign(nw, 0);
  unconstrained.clear();
  int carried = 0;
  // Wells deferring to a group: which mobility their group's target is
  // written in, and which group it came from.
  auto& grp_kind = scratch.grp_kind;
  auto& grp_of = scratch.grp_of;
  grp_kind.assign(nw, 0);
  grp_of.resize(nw);
  for (std::string& name : grp_of) name.clear();

  for (size_t iw = 0; iw < nw; ++iw) {
    const Well& w = sc.wells[iw];
    if (!w.state.defined || !w.state.open) continue;
    if (w.state.type != WellType::Producer) continue;

    // Which target, if any.  A history record (WCONHIST) states what the well
    // DID; a control record states what it is asked for.  Either is a rate to
    // be met, and neither is a bottom-hole pressure.
    // Which phase combination the target constrains.  Norne is a history
    // deck, so most records are WCONHIST and the CONTROL target is absent --
    // the log prints "control ORAT -> RESV, target -".  Reading only
    // state.orat therefore left almost every producer unconstrained, which is
    // why C-3H still ran away after the first version of this.
    Real tgt = 0;
    int kind = 0;   // 1 oil, 2 liquid, 3 gas, 4 water, 5 reservoir volume

    // In history mode OPM controls the phase combination named by CMODE (or
    // WHISTCTL), including a zero target.  Choosing the first nonzero history
    // column is wrong for Norne: nearly every producer says RESV while also
    // carrying nonzero oil and gas observations.
    if (w.state.historical) {
      switch (w.state.mode) {
        case CtrlMode::ORAT: tgt = w.state.hist_orat; kind = 1; break;
        case CtrlMode::LRAT: tgt = w.state.hist_orat + w.state.hist_wrat;
                             kind = 2; break;
        case CtrlMode::GRAT: tgt = w.state.hist_grat; kind = 3; break;
        case CtrlMode::WRAT: tgt = w.state.hist_wrat; kind = 4; break;
        case CtrlMode::RESV: kind = 5; break; // target after PVT averaging
        default: break; // BHP/THP/SHUT are not rate controls
      }
    } else {
      const Real t = w.state.target();
      if (t >= 0 && std::isfinite((double)t)) {
        switch (w.state.mode) {
          case CtrlMode::ORAT: tgt = t; kind = 1; break;
          case CtrlMode::LRAT: tgt = t; kind = 2; break;
          case CtrlMode::GRAT: tgt = t; kind = 3; break;
          case CtrlMode::WRAT: tgt = t; kind = 4; break;
          case CtrlMode::RESV: tgt = t; kind = 5; break;
          default: break;      // BHP, THP, GRUP: a pressure, not a rate
        }
      }
    }
    // WEFAC does not scale an individual well's control equation in OPM.  It
    // scales well contributions to reservoir/group accounting.  Norne has no
    // active non-unit WEFAC records, so leave the control target unchanged.

    const bool history_bhp =
        w.state.historical && w.state.mode == CtrlMode::BHP;
    if (!kind && !history_bhp) {
      // Nothing stated for the WELL.  Its group may state one for all of
      // them: that is what a WCONPROD record naming RESV with a defaulted
      // value means -- not "produce no reservoir volume" but "I am under\n// group control".  Marked here and allocated below, once every well's
      // productivity is known, since a share cannot be computed one well at
      // a time.
      const GroupControl* g = sc.group_target(w.group);
      if (g) {
        switch (g->mode) {
          case CtrlMode::ORAT: grp_kind[iw] = 1; break;
          case CtrlMode::LRAT: grp_kind[iw] = 2; break;
          case CtrlMode::GRAT: grp_kind[iw] = 3; break;
          case CtrlMode::WRAT: grp_kind[iw] = 4; break;
          case CtrlMode::RESV: grp_kind[iw] = 5; break;
          default: break;
        }
        if (grp_kind[iw]) {
          grp_of[iw] = g->name;
          wants[iw] = grp_kind[iw];   // accumulate the right mobility
        }
      }
    }
    if (!kind && !history_bhp && !grp_kind[iw]) {
      // Nothing stated THIS step.  Before falling back to pressure control,
      // try the last rate the deck gave this well.
      //
      // Norne's producers alternate between ORAT records that state a rate
      // and RESV records that default it, because in between they are under
      // GROUP control -- GCONPROD and GRUPTREE, which this simulator does not
      // implement yet.  Running them on PRODBHP for those stretches let them
      // produce 10 to 25 times the observed rate: B-2H peaks at 130,000
      // STB/day against Flow's 5,000, which is most of Norne's whole field
      // production from one well.
      //
      // Holding the last stated rate is an approximation, and it is a far
      // better one than a bottom-hole pressure nobody asked for.  It is also
      // what the well was doing a few days earlier.
      auto prev = last_target().find(w.name);
      if (prev != last_target().end() && prev->second.second > 0) {
        kind = prev->second.first;
        tgt = prev->second.second;
        ++carried;
      }
    }
    if (!kind) {
      // Still accumulate a mobility-weighted pressure for this well, so the
      // drawdown cap below has something to measure against.
      wants[iw] = 9;   // "no target, accumulate reservoir mobility only"
      // No rate constraint at all.  The well will produce whatever the
      // drawdown from its BHP gives, and if it has just been converted from
      // injection that is a well opening into rock its neighbours filled with
      // gas.  Say so: it is the single most likely reason a step will not
      // converge, and it is invisible otherwise.
      if (!history_bhp) unconstrained.push_back((int)iw);
    } else {
      wants[iw] = (char)kind;
      target[iw] = tgt;
      if (!(w.state.historical && kind == 5))
        last_target()[w.name] = std::make_pair(kind, (double)tgt);
    }

    if (!have) continue;

    // WELSPECS states BHP at a reference depth.  If it is defaulted, OPM uses
    // the top connection.  `depth` is already in feet here, including for a
    // METRIC Norne deck (to_field_units converted it once at model load).
    const std::vector<size_t>& track = scratch.track[iw];
    const double ref_depth = well_reference_depth(w, fm, track);
    for (size_t ic = 0; ic < w.completions.size(); ++ic) {
      const Completion& c = w.completions[ic];
      if (c.cell < 0 || !c.open) continue;
      if (c.cell >= (int)l2a.size()) continue;
      const int loc = l2a[c.cell];
      if (loc < 0 || loc >= cells) continue;
      const WellCell& k = wc[loc];
      // A producer's completion, whether or not it is CURRENTLY flagged as
      // producing.
      //
      // The flag is an output of this routine as well as an input: the
      // no-crossflow solve below shuts the completions it excluded, and this
      // runs again on the next Newton iteration.  Testing `k.is_prod` here
      // would mean a completion shut once could never be reconsidered, so the
      // active set could only ever shrink -- a ratchet, and exactly the
      // unstable outer fixed point that made the injector version of this
      // unusable before.  Deciding membership from the SCHEDULE instead makes
      // each call a pure function of the current state.
      //
      // `k.is_inj` is still respected: a cell an injector owns is not this
      // producer's to claim, and wi <= 0 means the schedule never resolved it.
      if (k.is_inj || !(k.wi > 0)) continue;

      const Real p = U(loc, 0);
      const Real Sw = U(loc, 1);
      const Real X = (bs == 3) ? U(loc, 2) : (Real)0;
      const Real Sg = physical_sg_at(fm, r, c.cell, Sw, X);
      const Real po = oil_pressure_at(fm, r, c.cell, p, Sw, Sg, T);
      const DevicePvt cell_pvt = pvt_for_active(hp, fm, c.cell);
      const PvtCell F = cell_pvt.valid
                            ? pvt_cell_at(cell_pvt, p, Sw, X, c.cell, fm, r)
                            : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                                             r.uo, r.uw, r.ug);
      const Mob3 m = well_relperm(bs, Sw, Sg, T, F.muw, F.muo, F.mug,
                              (Real)1, (Real)1, (Real)1, r.swi, r.swr,
                              c.cell);
      // The mobility the target is written in.  Gas carries the solution gas
      // as well, matching what ECLIPSE calls WGPR; reservoir volume drops the
      // shrinkage factors, being a reservoir rate rather than a surface one.
      double A = 0;
      switch (wants[iw]) {
        case 1: A = (double)(F.bo * k.wi * m.Mo +
                             F.rv * F.bg * k.wi * m.Mg); break;
        case 2: A = (double)(F.bo * k.wi * m.Mo +
                             F.rv * F.bg * k.wi * m.Mg +
                             F.bw * k.wi * m.Mw); break;
        case 3: A = (double)(F.bg * k.wi * m.Mg + F.rs * F.bo * k.wi * m.Mo);
                break;
        case 4: A = (double)(F.bw * k.wi * m.Mw); break;
        default: A = (double)(k.wi * (m.Mo + m.Mw + m.Mg)); break;
      }

      // Approximate OPM's flowing connection density with the local
      // reservoir-volume mobility mixture.  The deck densities are surface
      // specific gravities; multiplying by shrinkage factors gives reservoir
      // densities.  Oil includes the mass of its dissolved gas, while gas's
      // scf/bbl conversion contributes the ft3-per-bbl divisor.  This is
      // deliberately host-side and adds no CUDA kernel or linear-solver work.
      double dp = 0.0;
      if (r.gravity && c.cell < (int)fm.grid.depth.size()) {
        const double rho_w = (double)r.rhow * (double)F.bw;
        const double rho_o =
            ((double)r.rhoo + (double)F.rs * (double)r.rhog /
                                  (double)units::kFt3PerBbl) *
            (double)F.bo;
        const double rho_g =
            ((double)r.rhog / (double)units::kFt3PerBbl +
             (double)F.rv * (double)r.rhoo) * (double)F.bg;
        const double lw = std::max(0.0, (double)m.Mw);
        const double lo = std::max(0.0, (double)m.Mo);
        const double lg = std::max(0.0, (double)m.Mg);
        const double lsum = lw + lo + lg;
        const double rho_mix =
            lsum > 0.0 ? (lw * rho_w + lo * rho_o + lg * rho_g) / lsum
                       : rho_o;
        dp = (double)r.grav_grad * rho_mix *
             ((double)fm.grid.depth[c.cell] - ref_depth);
      }
      const size_t ci = conn_off[iw] + ic;
      conn_po[ci] = po;
      // Lagged perforation component rates for the wellbore-density update.
      // OPM likewise builds connection pressure drops from the current well
      // state before solving the next coupled well equations. Positive here
      // means production into the wellbore.
      const Real draw0 = po - k.pwf;
      const Real qo0 = k.wi * m.Mo * draw0;
      const Real qw0 = k.wi * m.Mw * draw0;
      const Real qg0 = k.wi * m.Mg * draw0;
      if (draw0 < 0 && k.xflow_model) {
        // The hydrostatic column must consume the same reverse-completion
        // rates as the reservoir residual.  Using local phase mobilities here
        // gave OPM's segment-density algorithm the wrong fluid even after the
        // source term itself had switched to the wellbore cmix_s branch.
        const Real kvol = (Real)units::kFt3PerBbl;
        const Real bgc = F.bg / kvol;
        const Real rsc = F.rs / kvol;
        const Real rvc = F.rv * kvol;
        const Real dis = std::max((Real)1e-12,
                                  (Real)1 - F.rs * F.rv);
        const Real vr =
            k.mix_w / F.bw +
            (k.mix_o - rvc * k.mix_g) / (F.bo * dis) +
            (k.mix_g - rsc * k.mix_o) / (bgc * dis);
        const Real qres = k.wi * (m.Mo + m.Mw + m.Mg) * draw0;
        const Real qs = vr > (Real)1e-20 ? qres / vr : (Real)0;
        conn_surface[3 * ci + 0] = k.mix_o * qs;
        conn_surface[3 * ci + 1] = k.mix_w * qs;
        conn_surface[3 * ci + 2] = kvol * k.mix_g * qs;
      } else {
        conn_surface[3 * ci + 0] = F.bo * qo0 + F.rv * F.bg * qg0;
        conn_surface[3 * ci + 1] = F.bw * qw0;
        conn_surface[3 * ci + 2] = F.bg * qg0 + F.rs * F.bo * qo0;
      }
      conn_scoef[3 * ci + 0] =
          F.bo * k.wi * m.Mo + F.rv * F.bg * k.wi * m.Mg;
      conn_scoef[3 * ci + 1] = F.bw * k.wi * m.Mw;
      conn_scoef[3 * ci + 2] =
          (F.bg * k.wi * m.Mg + F.rs * F.bo * k.wi * m.Mo) /
          (Real)units::kFt3PerBbl;

      conn_drdpw[3 * ci + 0] =
          -(F.bo * k.wi * m.Mo + F.rv * F.bg * k.wi * m.Mg);
      conn_drdpw[3 * ci + 1] = -(F.bw * k.wi * m.Mw);
      conn_drdpw[3 * ci + 2] =
          -(F.bg * k.wi * m.Mg + F.rs * F.bo * k.wi * m.Mo);
      conn_rev[6 * ci + 0] = k.wi * (m.Mo + m.Mw + m.Mg);
      conn_rev[6 * ci + 1] = F.bo;
      conn_rev[6 * ci + 2] = F.bw;
      conn_rev[6 * ci + 3] = F.bg;
      conn_rev[6 * ci + 4] = F.rs;
      conn_rev[6 * ci + 5] = F.rv;
      conn_mob[3 * ci + 0] = m.Mo;
      conn_mob[3 * ci + 1] = m.Mw;
      conn_mob[3 * ci + 2] = m.Mg;
      conn_dp[ci] = dp;

      // WELLSCHUR 2+ differentiates the host-eliminated well state,
      // not just its frozen-mobility pressure average. Keep this experimental
      // path separate from mode 1 so the validated rank-one operator remains
      // bit-for-bit reproducible. Numerical differentiation is appropriate
      // here: these are host-only well coefficients (at most 504 on Norne),
      // and reusing the exact PVT/relperm/primary-switch helpers avoids a
      // second, inevitably drifting analytic implementation.
      // Fixed-BHP wells have no eliminated control unknown: their datum BHP
      // is prescribed, and the local reservoir-well Jacobian already carries
      // the mobility derivatives.  Computing the numerical Schur derivatives
      // for every such completion is dead work (340 completions in SPE10).
      // The cumulative hydrostatic column remains refreshed above.
      if (r.well_schur_enabled > 1 && wants[iw] != 9) {
        struct LocalWellCoeff {
          Real A, theta, mob[3], scoef[3], pvt[5];
        };
        auto coefficient = [&](Real pe, Real swe, Real xe) {
          LocalWellCoeff z{};
          const Real sge = physical_sg_at(fm, r, c.cell, swe, xe);
          const Real poe = oil_pressure_at(fm, r, c.cell, pe, swe, sge, T);
          const PvtCell Fe = cell_pvt.valid
                                 ? pvt_cell_at(cell_pvt, pe, swe, xe, c.cell,
                                               fm, r)
                                 : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                                                  r.uo, r.uw, r.ug);
          const Mob3 me = well_relperm(bs, swe, sge, T, Fe.muw, Fe.muo, Fe.mug,
                                   (Real)1, (Real)1, (Real)1, r.swi, r.swr,
                                   c.cell);
          switch (wants[iw]) {
            case 1: z.A = Fe.bo * k.wi * me.Mo +
                          Fe.rv * Fe.bg * k.wi * me.Mg; break;
            case 2: z.A = Fe.bo * k.wi * me.Mo +
                          Fe.rv * Fe.bg * k.wi * me.Mg +
                          Fe.bw * k.wi * me.Mw; break;
            case 3: z.A = Fe.bg * k.wi * me.Mg +
                          Fe.rs * Fe.bo * k.wi * me.Mo; break;
            case 4: z.A = Fe.bw * k.wi * me.Mw; break;
            default: z.A = k.wi * (me.Mo + me.Mw + me.Mg); break;
          }
          double dpe = 0.0;
          if (r.gravity && c.cell < (int)fm.grid.depth.size()) {
            const double rw = (double)r.rhow * (double)Fe.bw;
            const double ro =
                ((double)r.rhoo + (double)Fe.rs * (double)r.rhog /
                                      (double)units::kFt3PerBbl) *
                (double)Fe.bo;
            const double rg =
                ((double)r.rhog / (double)units::kFt3PerBbl +
                 (double)Fe.rv * (double)r.rhoo) * (double)Fe.bg;
            const double lw = std::max(0.0, (double)me.Mw);
            const double lo = std::max(0.0, (double)me.Mo);
            const double lg = std::max(0.0, (double)me.Mg);
            const double ls = lw + lo + lg;
            const double rho = ls > 0.0 ? (lw * rw + lo * ro + lg * rg) / ls
                                        : ro;
            dpe = (double)r.grav_grad * rho *
                  ((double)fm.grid.depth[c.cell] - ref_depth);
          }
          // Flow's StandardWellConnections stores connection pressure drops
          // as Scalar explicit quantities, not AD evaluations.  Mode 2+
          // follows that complete-well path below, so its linearization must
          // hold the cumulative wellbore column fixed.  Subtracting a frozen
          // scalar does not affect this numerical derivative; using the old
          // local-density dpe derivative here would differentiate a pressure
          // model that is no longer in the residual.
          z.theta = (r.well_schur_enabled > 1)
                        ? poe : (Real)((double)poe - dpe);
          z.mob[0] = me.Mo; z.mob[1] = me.Mw; z.mob[2] = me.Mg;
          z.scoef[0] = Fe.bo * k.wi * me.Mo +
                       Fe.rv * Fe.bg * k.wi * me.Mg;
          z.scoef[1] = Fe.bw * k.wi * me.Mw;
          z.scoef[2] = Fe.bg * k.wi * me.Mg +
                       Fe.rs * Fe.bo * k.wi * me.Mo;
          z.scoef[2] /= (Real)units::kFt3PerBbl;
          z.pvt[0] = Fe.bo; z.pvt[1] = Fe.bw; z.pvt[2] = Fe.bg;
          z.pvt[3] = Fe.rs; z.pvt[4] = Fe.rv;
          return z;
        };
        Real uv[3] = {p, Sw, X};
        Real h[3] = {
            (Real)std::max(1.0e-3, 1.0e-4 * std::fabs((double)p)),
            (Real)1.0e-6,
            (Real)std::max(1.0e-7,
                           1.0e-5 * std::max(std::fabs((double)X), 1.0e-3))};
        const int nv = bs == 3 ? 3 : 2;
        for (int v = 0; v < nv; ++v) {
          Real up[3] = {uv[0], uv[1], uv[2]};
          Real um[3] = {uv[0], uv[1], uv[2]};
          up[v] += h[v]; um[v] -= h[v];
          if (v == 0 && um[v] <= 0) um[v] = uv[v];
          const LocalWellCoeff zp = coefficient(up[0], up[1], up[2]);
          const LocalWellCoeff zm = coefficient(um[0], um[1], um[2]);
          const Real den = up[v] - um[v];
          if (!(den != (Real)0)) continue;
          conn_dA_du[3 * ci + v] = (zp.A - zm.A) / den;
          conn_dtheta_du[3 * ci + v] = (zp.theta - zm.theta) / den;
          for (int ph = 0; ph < 3; ++ph)
            conn_dmob_du[9 * ci + 3 * ph + v] =
                (zp.mob[ph] - zm.mob[ph]) / den;
          for (int ph = 0; ph < 3; ++ph)
            conn_dscoef_du[9 * ci + 3 * ph + v] =
                (zp.scoef[ph] - zm.scoef[ph]) / den;
          for (int q = 0; q < 5; ++q)
            conn_dpvt_du[15 * ci + 3 * q + v] =
                (zp.pvt[q] - zm.pvt[q]) / den;
        }
      }
      acc[2 * iw + 0] += A;
      acc[2 * iw + 1] += A * ((double)po - dp);
      // The same two numbers kept per completion, so the constrained solve can
      // ask which of them are above the flowing pressure.  theta = p - dp is
      // the cell pressure corrected to the well's datum, which is what makes a
      // single bottom-hole pressure comparable across the whole well.
      conn_acc[2 * ci + 0] += A;
      conn_acc[2 * ci + 1] += A * ((double)po - dp);
      if (fm.part.rank == 0 && debug_well() && w.name == debug_well())
        std::printf("  [dbg %s] step %d ic=%zu (i%d j%d k%d) wi=%.6g A=%.6g "
                    "theta=%.4g Sw=%.4g Sg=%.4g p=%.4g dp=%.4g\n",
                    w.name.c_str(), step, ic, c.i, c.j, c.k, (double)k.wi, A,
                    (double)po - dp, (double)Sw, (double)Sg, (double)p, dp);
    }
  }

  if (fm.part.nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, acc.data(), (int)acc.size(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    // Every rank walks the same schedule, so conn_acc has the same length and
    // the same meaning on all of them; a completion contributes on the one
    // rank that owns its cell and zero everywhere else.
    if (!conn_acc.empty())
      MPI_Allreduce(MPI_IN_PLACE, conn_acc.data(), (int)conn_acc.size(),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (!conn_surface.empty()) {
      MPI_Allreduce(MPI_IN_PLACE, conn_surface.data(),
                    (int)conn_surface.size(), MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, conn_po.data(), (int)conn_po.size(),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, conn_rev.data(), (int)conn_rev.size(),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, conn_mob.data(), (int)conn_mob.size(),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, conn_scoef.data(), (int)conn_scoef.size(),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if (r.well_schur_enabled > 1) {
        MPI_Allreduce(MPI_IN_PLACE, conn_dA_du.data(),
                      (int)conn_dA_du.size(), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, conn_dtheta_du.data(),
                      (int)conn_dtheta_du.size(), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, conn_dmob_du.data(),
                      (int)conn_dmob_du.size(), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, conn_dscoef_du.data(),
                      (int)conn_dscoef_du.size(), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, conn_dpvt_du.data(),
                      (int)conn_dpvt_du.size(), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
      }
    }
  }

  // Preserve the established local-density control equation.  The cumulative
  // OPM-style column below is used to translate the solved completion
  // pressures back to a physically comparable datum BHP, not to perturb the
  // reservoir residual before its full well Jacobian is available.
  scratch.direct_acc = acc;
  scratch.direct_conn_acc = conn_acc;
  scratch.direct_conn_dp = conn_dp;
  scratch.report_bhp_shift.assign(nw, 0.0);

  // OPM StandardWellConnections forms the hydrostatic column segment by
  // segment.  The density of a segment is the mixture flowing upward through
  // it, i.e. the sum of this perforation and all deeper perforations.  The old
  // NVRS approximation applied each cell's local mobility mixture over the
  // complete datum-to-cell distance; when the producing mixture changes with
  // depth that creates hundreds of psi of spurious BHP shift.
  const bool use_frozen_pressure_drop =
      freeze_pressure_drop && scratch.newton_conn_dp.size() == n_conn;
  if (!use_frozen_pressure_drop) for (size_t iw = 0; iw < nw; ++iw) {
    const Well& w = sc.wells[iw];
    if (!wants[iw] || w.state.type != WellType::Producer) continue;

    double running[3] = {0.0, 0.0, 0.0};
    const std::vector<size_t>& track = scratch.track[iw];
    for (size_t pos = track.size(); pos-- > 0;) {
      const size_t ic = track[pos];
      const size_t ci = conn_off[iw] + ic;
      const double a = conn_acc[2 * ci];
      if (!(a > 0)) continue;
      for (int q = 0; q < 3; ++q)
        running[q] += (double)conn_surface[3 * ci + q];

      const double bo = std::max(1.0e-30, (double)conn_rev[6 * ci + 1]);
      const double bw = std::max(1.0e-30, (double)conn_rev[6 * ci + 2]);
      const double bg = std::max(1.0e-30, (double)conn_rev[6 * ci + 3]);
      const double rs = (double)conn_rev[6 * ci + 4];
      const double rv = (double)conn_rev[6 * ci + 5];
      const double os = std::abs(running[0]);
      const double ws = std::abs(running[1]);
      const double gs = std::abs(running[2]);
      const double det = std::max(1.0e-12, 1.0 - rs * rv);
      const double qo = std::max(0.0, (os - rv * gs) / det) / bo;
      const double qg = std::max(0.0, (gs - rs * os) / det) / bg;
      const double qw = ws / bw;
      const double rho_w = (double)r.rhow * bw;
      const double rho_o =
          ((double)r.rhoo + rs * (double)r.rhog /
                                (double)units::kFt3PerBbl) * bo;
      const double rho_g =
          ((double)r.rhog / (double)units::kFt3PerBbl +
           rv * (double)r.rhoo) * bg;
      const double qt = qo + qw + qg;
      if (qt > 0.0) {
        conn_density[ci] = (qo * rho_o + qw * rho_w + qg * rho_g) / qt;
      } else {
        const double lo = std::max(0.0, (double)conn_mob[3 * ci + 0]);
        const double lw = std::max(0.0, (double)conn_mob[3 * ci + 1]);
        const double lg = std::max(0.0, (double)conn_mob[3 * ci + 2]);
        const double lt = lo + lw + lg;
        conn_density[ci] = lt > 0.0
                               ? (lo * rho_o + lw * rho_w + lg * rho_g) / lt
                               : rho_o;
      }
    }

    const double ref_depth = well_reference_depth(w, fm, track);
    double z_above = ref_depth;
    double pressure_drop = 0.0;
    acc[2 * iw + 1] = 0.0;
    for (const size_t ic : track) {
      const Completion& c = w.completions[ic];
      const size_t ci = conn_off[iw] + ic;
      const double a = conn_acc[2 * ci];
      if (!(a > 0) || c.cell < 0 || !c.open ||
          c.cell >= (int)fm.grid.depth.size())
        continue;
      const double z = (double)fm.grid.depth[c.cell];
      if (r.gravity)
        pressure_drop += (double)r.grav_grad * conn_density[ci] *
                         (z - z_above);
      conn_dp[ci] = pressure_drop;
      z_above = z;
      const double theta = (double)conn_po[ci] - pressure_drop;
      conn_acc[2 * ci + 1] = a * theta;
      acc[2 * iw + 1] += a * theta;
    }
  }
  if (!use_frozen_pressure_drop) {
    scratch.newton_conn_dp = conn_dp;
  } else {
    conn_dp = scratch.newton_conn_dp;
    for (size_t iw = 0; iw < nw; ++iw) {
      const Well& w = sc.wells[iw];
      if (!wants[iw] || w.state.type != WellType::Producer) continue;
      acc[2 * iw + 1] = 0.0;
      for (size_t ic = 0; ic < w.completions.size(); ++ic) {
        const size_t ci = conn_off[iw] + ic;
        const double a = conn_acc[2 * ci];
        if (!(a > 0)) continue;
        const double theta = (double)conn_po[ci] - conn_dp[ci];
        conn_acc[2 * ci + 1] = a * theta;
        acc[2 * iw + 1] += a * theta;
      }
    }
  }
  if (r.well_schur_enabled <= 1) {
    // Retain the validated production trajectory for mode 1: its reservoir
    // equation still uses the historical local-density column, and the
    // segment calculation only translates that pressure to a comparable
    // reported datum BHP.  Experimental complete modes use Flow's cumulative
    // connection column in both the control equation and reservoir source.
    for (size_t iw = 0; iw < nw; ++iw) {
      const double sumA = scratch.direct_acc[2 * iw];
      if (sumA > 0.0)
        scratch.report_bhp_shift[iw] =
            (acc[2 * iw + 1] - scratch.direct_acc[2 * iw + 1]) / sumA;
    }
    acc = scratch.direct_acc;
    conn_acc = scratch.direct_conn_acc;
    conn_dp = scratch.direct_conn_dp;
  }

  // History RESV has no fixed RESV column.  As in OPM, derive its target at
  // the current reservoir condition from all observed surface phase rates.
  for (size_t iw = 0; iw < nw; ++iw) {
    const Well& w = sc.wells[iw];
    if (!w.state.historical || w.state.mode != CtrlMode::RESV ||
        wants[iw] != 5)
      continue;
    int first_active = -1;
    for (const Completion& c : w.completions)
      if (c.open && c.cell >= 0) { first_active = c.cell; break; }
    const DevicePvt well_pvt = pvt_for_active(hp, fm, first_active);

    // RateConverter::dissolvedVaporisedRatio(): infer the mixture carried by
    // the observed surface rates, then cap it at the region-average state.
    double rs_eff = 0.0;
    if (w.state.hist_orat > 0)
      rs_eff = std::min(region_rs,
                        std::max(0.0, (double)w.state.hist_grat /
                                          (double)w.state.hist_orat));
    double rv_eff = 0.0;
    if (w.state.hist_grat > 0)
      rv_eff = std::min(region_rv,
                        std::max(0.0, (double)w.state.hist_orat /
                                          (double)w.state.hist_grat));

    double Bo = (double)r.bo, Bw = (double)r.bw, Bg = (double)r.bg;
    if (well_pvt.valid && region_p > 0) {
      Real bo = 1, bw = 1, bg = 1, mu = 1;
      dev_oil(well_pvt, (Real)region_p, (Real)rs_eff, bo, mu);
      dev_water(well_pvt, (Real)region_p, bw, mu);
      dev_gas(well_pvt, (Real)region_p, (Real)rv_eff, bg, mu);
      if (bo > 0) Bo = (double)bo;
      if (bw > 0) Bw = (double)bw;
      if (bg > 0) Bg = (double)bg;
    }

    // Sum the three reservoir-phase voidages after inverting the coupled
    // oil/gas component matrix.  Bo/Bw/Bg here are formation-volume factors,
    // so these coefficients are exactly OPM RateConverter's algebra.
    const double det = std::max(1.0e-12, 1.0 - rs_eff * rv_eff);
    const double co = (Bo - rs_eff * Bg) / det;
    const double cw = Bw;
    const double cg = (Bg - rv_eff * Bo) / det;
    const double q = co * (double)w.state.hist_orat +
                     cw * (double)w.state.hist_wrat +
                     cg * (double)w.state.hist_grat;
    target[iw] = (Real)std::max(0.0, q);
    last_target()[w.name] = std::make_pair(5, (double)target[iw]);
  }

  // --- share each group's target among the wells that defer to it ---------
  //
  // In proportion to productivity: A_w = SUM_k b WI lambda over the well's
  // live completions, which is the rate that well would make per unit of
  // drawdown.  It is the standard default -- a guide rate of "potential" --
  // and it is already computed above for the pwf solve, in the very mobility
  // the group's target is written in.
  //
  // Done after the reduction because on several ranks no single rank holds
  // all of a group's wells, let alone all of one well's completions.
  int allocated = 0;
  {
    std::map<std::string, double> gsum;
    for (size_t iw = 0; iw < nw; ++iw)
      if (grp_kind[iw]) gsum[grp_of[iw]] += acc[2 * iw + 0];
    for (size_t iw = 0; iw < nw; ++iw) {
      if (!grp_kind[iw]) continue;
      const double tot = gsum[grp_of[iw]];
      const double mine = acc[2 * iw + 0];
      const GroupControl* g = sc.group_target(sc.wells[iw].group);
      if (!g || !(tot > 0) || !(mine > 0)) {
        // No productivity to share on, so this well gets nothing rather than
        // a share of a division by zero.  It keeps whatever
        // apply_schedule_step gave it.
        wants[iw] = 0;
        continue;
      }
      target[iw] = (Real)((double)g->target * mine / tot);
      ++allocated;
    }
  }
  if (fm.part.rank == 0 && allocated > 0) {
    static int said_group = 0;
    if (!said_group) {
      said_group = 1;
      std::printf("  [well] %d producer(s) under group control; the group's "
                  "target is shared by productivity\n", allocated);
    }
  }

  if (!have) return;

  // How much crossflow the constrained solve removed this call, for the
  // banner below.  Counted here rather than inside the loop's branch so
  // that a run with WELLXFLOW 1 reports zero rather than nothing.
  int xflow_shut = 0, xflow_wells = 0;

  auto& schur = scratch.schur;
  schur.clear();
  if (r.well_schur_enabled) schur.reserve(n_conn);
  r.well_schur_count = 0;
  r.well_schur_groups = 0;

  // Mode 8 adds the pressure block for a rate-controlled injector.  At
  // frozen mobility q_k=A_k(p_bhp-theta_k) and sum(q_k)=target, hence
  //
  //   d p_bhp / d p_j = A_j/sum(A),
  //   d R_k / d p_j = A_k delta_kj - A_k A_j/sum(A).
  //
  // The first term is assembled locally in assemble_blackoil.cu; these
  // entries supply the dense second term without changing the sparse graph.
  if (r.well_schur_enabled == 8 && bs >= 2) {
    std::vector<double> inj_sumA(nw, 0.0);
    for (size_t iw = 0; iw < nw; ++iw) {
      const Well& w = sc.wells[iw];
      if (!w.state.open || w.state.type != WellType::Injector) continue;
      for (const Completion& c : w.completions) {
        if (!c.open || c.cell < 0 || c.cell >= (int)l2a.size()) continue;
        const int loc = l2a[c.cell];
        if (loc < 0 || loc >= cells) continue;
        const WellCell& k = wc[(size_t)loc];
        if (k.inj_rate_coupled && k.inj_dqdp > 0)
          inj_sumA[iw] += (double)k.inj_dqdp;
      }
    }
    if (fm.part.nranks > 1 && !inj_sumA.empty())
      MPI_Allreduce(MPI_IN_PLACE, inj_sumA.data(), (int)inj_sumA.size(),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (size_t iw = 0; iw < nw; ++iw) {
      const Well& w = sc.wells[iw];
      if (!(inj_sumA[iw] > 0)) continue;
      int eq = 2;
      if (w.state.inj_phase == InjectedPhase::Water) eq = 1;
      else if (w.state.inj_phase == InjectedPhase::Oil) eq = 0;
      for (const Completion& c : w.completions) {
        if (!c.open || c.cell < 0 || c.cell >= (int)l2a.size()) continue;
        const int loc = l2a[c.cell];
        if (loc < 0 || loc >= cells) continue;
        const WellCell& k = wc[(size_t)loc];
        if (!k.inj_rate_coupled || !(k.inj_dqdp > 0)) continue;
        WellSchurEntry e;
        e.cell = loc;
        e.group = (int)iw;
        for (int v = 0; v < 3; ++v)
          e.dg_du[v] = -k.inj_dqdu[v] / (Real)inj_sumA[iw];
        e.dr_dg[3 * eq] = -k.inj_dqdp;
        schur.push_back(e);
      }
    }
  }
  if (r.well_schur_enabled == 9 && bs >= 2) {
    for (size_t iw = 0; iw < nw; ++iw) {
      const Well& w = sc.wells[iw];
      if (!w.state.open || w.state.type != WellType::Injector) continue;
      for (const Completion& c : w.completions) {
        if (!c.open || c.cell < 0 || c.cell >= (int)l2a.size()) continue;
        const int loc = l2a[c.cell];
        if (loc < 0 || loc >= cells) continue;
        const WellCell& k = wc[(size_t)loc];
        if (!k.inj_xflow_coupled) continue;
        WellSchurEntry e;
        e.cell = loc;
        e.group = (int)iw;
        for (int j = 0; j < 9; ++j) {
          e.dg_du[j] = k.inj_dgdu[j];
          e.dr_dg[j] = k.inj_drdg[j];
        }
        schur.push_back(e);
      }
    }
  }
  applied_rate().clear();
  for (size_t iw = 0; iw < nw; ++iw) {
    if (!wants[iw]) continue;
    const double sumA = acc[2 * iw + 0], sumAp = acc[2 * iw + 1];
    // Filled in below, once BHP is final:
    // q = SUM A (p - dp) - BHP SUM A.  This is the target exactly when it was
    // reachable and the smaller rate the well can actually make when the
    // bottom-hole limit bound instead.
    const Well& w0 = sc.wells[iw];

    // Test a zero target BEFORE testing the controlled phase's productivity.
    // A stopped well can quite legitimately have no mobility in the phase
    // named by its control.  F-4H, for example, is water-filled while its
    // WCONHIST record says ORAT=0.  The old ordering returned on sumA==0 and
    // left apply_schedule_step()'s producer flags live, so the nominally
    // stopped well produced 38--53 kSTB/day of water at the fallback BHP.
    // Besides corrupting WWPR this was a real source in the water residual.
    //
    // Preserve a useful datum pressure for reporting.  With no controlled-
    // phase mobility sumAp/sumA is undefined, so average the already computed
    // hydrostatically translated connection pressures instead.
    if (wants[iw] != 9 && target[iw] == (Real)0) {
      double bhp0 = 0.0;
      int n_bhp = 0;
      for (size_t ic = 0; ic < w0.completions.size(); ++ic) {
        const Completion& c = w0.completions[ic];
        if (c.cell < 0 || !c.open || c.cell >= (int)l2a.size()) continue;
        const int loc = l2a[c.cell];
        if (loc < 0 || loc >= cells || !wc[loc].is_prod) continue;
        const size_t ci = conn_off[iw] + ic;
        const double theta = (double)conn_po[ci] - (double)conn_dp[ci];
        if (std::isfinite(theta)) { bhp0 += theta; ++n_bhp; }
      }
      if (n_bhp > 0) bhp0 /= (double)n_bhp;
      applied_rate()[w0.name] = std::make_pair((int)wants[iw], 0.0);
      for (const Completion& c : w0.completions) {
        if (c.cell < 0 || !c.open || c.cell >= (int)l2a.size()) continue;
        const int loc = l2a[c.cell];
        if (loc < 0 || loc >= cells || !wc[loc].is_prod) continue;
        if (n_bhp > 0) wc[loc].bhp = (Real)bhp0;
        wc[loc].pwf = U(loc, 0);
        wc[loc].is_prod = 0;
      }
      continue;
    }

    if (!(sumA > 0)) continue;
    // A HISTORY well has no bottom-hole limit unless it is under BHP control.
    //
    // This is OPM's rule, in WellProductionProperties::handleWCONHIST:
    //
    //     if (this->predictionMode || this->controlMode == ProducerCMode::BHP)
    //         this->setBHPLimit(bhp_def);
    //
    // so a WCONHIST producer on ORAT, LRAT, RESV, WRAT or GRAT gets no limit
    // at all -- it made the rate the deck records, and whatever pressure that
    // took is a fact rather than a constraint to be checked.  Imposing one
    // throttles wells Flow never throttles, which is how every Norne producer
    // came out rate-controlled and still delivered 54% of Flow's oil.
    const bool history_free =
        (w0.state.historical && w0.state.mode != CtrlMode::BHP);
    const bool deck_gave_bhp =
        (!history_free && std::isfinite((double)w0.state.bhp_limit) &&
         w0.state.bhp_limit > 0);

    // OPM does not interpret an active zero ORAT (or WRAT/GRAT/LRAT) as
    // "make that phase zero but let the others crossflow".  Its
    // stoppedOrZeroRateTarget path sets WQTotal to zero, so the whole well is
    // inactive.  Solving only SUM(q_o)=0 here gave a common BHP between the
    // completion pressures: oil cancelled across the well while enormous
    // water and gas rates flowed in opposite directions.  Those are the
    // B-4H/D-4H spikes and the 0/1 WWCT steps in the Norne comparison.
    //
    // A zero source needs no well Jacobian either.  Clearing is_prod removes
    // both from the existing assembly path; apply_schedule_step restores the
    // flag at the next schedule interval if the well receives a nonzero rate.
    double pwf;
    // Set by the no-crossflow branch: the pressure it returned is already
    // final, and the clamps written for the plain division must not be
    // re-applied to it.
    bool constrained = false;
    // True whenever the assembly should use the wellbore-mixture crossflow
    // model (assemble_blackoil.cu) rather than the legacy per-phase LOCAL
    // formation mobility for this well's reverse-flowing completions.
    bool xflow_composition = false;
    double delivered = 0.0;
    bool rate_reached = false;
    if (wants[iw] == 9) {
      // No rate target.  If the deck stated a bottom-hole pressure, that IS
      // the control and it stands.  If it did not, the pressure in use is our
      // own PRODBHP, so bound the drawdown it implies against the mobility-
      // weighted pressure the well actually sees.
      if (deck_gave_bhp) {
        // Fixed BHP is stated at the WELSPECS datum too.  Keep solving the
        // completion pressures below so every perforation receives the
        // segment hydrostatic correction; returning here applied the datum
        // pressure unchanged to all depths and biased SPE10 pressure by
        // roughly one well-column head (40--60 psi).
        pwf = (double)w0.state.bhp_limit;
        xflow_composition = true;
      } else {
        if (!(d.prod_draw_max > 0)) continue;
        const double p_avg = sumAp / sumA;
        const double floor_draw = p_avg - (double)d.prod_draw_max;
        pwf = (double)d.prod_bhp;
        if (pwf < floor_draw) pwf = floor_draw;
      }
    } else if (d.well_crossflow == 2) {
      // WELSPECS permits crossflow (its default on NORNE).  Keep every
      // completion in the common-pressure control and use the wellbore
      // mixture for any reverse-flow residual.  Mode 1 has the same pressure
      // solve but the legacy local-phase reverse composition.
      pwf = (sumAp - (double)target[iw]) / sumA;
      xflow_composition = true;
    } else if (!d.well_crossflow && wants[iw] == 5) {
      // RESV under the crossflow-composition model: total RESERVOIR VOIDAGE
      // is composition-independent, so the plain division already meets the
      // target exactly, whichever way each completion's flow eventually
      // splits into phases.  The no-crossflow solve below exists to stop a
      // completion's reverse flow being counted with the WRONG composition
      // (a phase this cell happens to be locally mobile in, rather than the
      // wellbore's own mixture); now that the kernel corrects that directly
      // (see assemble_blackoil.cu), a RESV well no longer needs its reverse
      // completions excluded from the control equation at all -- excluding
      // them only under-delivers the target by however much they carry.
      // Nearly every Norne producer is RESV, so this is not a corner case.
      pwf = (sumAp - (double)target[iw]) / sumA;
      xflow_composition = true;
    } else if (!d.well_crossflow) {
      // The pressure meeting the target with NO completion flowing backwards.
      //
      //     SUM_k A_k max(0, theta_k - pwf) = target
      //
      // rather than the plain division below, which counts a completion's
      // reverse flow against the total and lets one wellbore produce at one
      // end while injecting at the other.  See include/well_control.hpp for
      // why that is not a rounding detail on Norne, and note the direction:
      // dropping the negative terms means the well needs LESS drawdown, not
      // more, because it is no longer over-producing the good completions to
      // net out against its own re-injection.
      //
      // Identical to the plain division whenever nothing was flowing
      // backwards, which is what makes this safe for every other deck.
      auto& conns = scratch.conns;
      conns.clear();
      conns.reserve(w0.completions.size());
      for (size_t ic = 0; ic < w0.completions.size(); ++ic) {
        const size_t ci = conn_off[iw] + ic;
        const double a = conn_acc[2 * ci + 0];
        if (!(a > 0)) continue;
        WellConn wcn;
        wcn.a = a;
        wcn.theta = conn_acc[2 * ci + 1] / a;   // A*theta summed, then divided
        wcn.index = (int)ic;
        conns.push_back(wcn);
      }
      const double floor_now =
          history_free ? (double)d.p_atm
                       : (deck_gave_bhp ? (double)w0.state.bhp_limit
                                        : (double)d.prod_bhp);
      const WellPwf sol =
          solve_pwf_no_crossflow(conns, (double)target[iw], floor_now);
      pwf = sol.pwf;
      delivered = sol.delivered;
      constrained = true;
      xflow_composition = true;
      if (fm.part.rank == 0 && debug_well() && w0.name == debug_well())
        std::printf("  [dbg %s] step %d target=%.6g floor=%.6g conns=%zu "
                    "-> pwf=%.6g delivered=%.6g n_active=%d\n",
                    w0.name.c_str(), step, (double)target[iw], floor_now,
                    conns.size(), sol.pwf, sol.delivered, sol.n_active);
      xflow_shut += (int)conns.size() - sol.n_active;
      if ((int)conns.size() > sol.n_active) ++xflow_wells;
    } else {
      // WELLXFLOW 1: the original unconstrained division, kept so a run can be
      // put back exactly as it was.
      // q(pwf) = sumAp - pwf * sumA, so the pressure meeting the target is:
      pwf = (sumAp - (double)target[iw]) / sumA;
    }

    // The BHP limit is a FLOOR for a producer.  Where the target needs a
    // lower pressure than the well is allowed to pull, the limit binds and
    // the well simply produces less -- the switch OPM makes too.
    // A producer's flowing pressure must lie BELOW the pressure it sees.
    // Above it the well is injecting, whatever the deck calls it, and the
    // source term changes sign: on Norne that turned seven producers into
    // large gas injectors and froze the step.  Equality means no flow, which
    // is the right answer for a well asked to produce nothing.
    //
    // NOT applied to the constrained solve, and this is the one place the two
    // paths must not share a clamp.  p_avg is the mobility-weighted mean over
    // EVERY completion, the low ones included, so it sits below the pressure
    // the constrained solve returns whenever any completion was going to
    // crossflow -- which is precisely the case this is meant to fix.  Forcing
    // pwf down to p_avg there would put the reverse flow straight back and
    // make the whole solve a no-op.
    //
    // The constrained solve enforces the same intent more strictly anyway: its
    // root always lies below the highest theta, and every completion left open
    // has theta > pwf by construction, so no open completion can be injecting.
    if (!constrained) {
      const double p_avg = sumAp / sumA;
      if (pwf > p_avg) pwf = p_avg;
    }

    const Well& w = w0;
    if (wants[iw] != 9) {
      // No floor for a history well; otherwise the deck's, or ours.  A
      // nominal atmospheric bound remains, because a negative flowing
      // pressure is not a well.
      const double floor_bhp = history_free
                                   ? (double)d.p_atm
                                   : (deck_gave_bhp
                                          ? (double)w.state.bhp_limit
                                          : (double)d.prod_bhp);
      if (floor_bhp > 0 && pwf < floor_bhp) pwf = floor_bhp;
      // What the well will ACTUALLY make at the pressure it ends up with.
      // Equal to the target where the target was reachable; less where the
      // bottom-hole limit bound first, which is the switch OPM makes and the
      // number the report should carry either way.
      //
      // The constrained solve already returns this, counting forward flow
      // only.  `sumAp - pwf * sumA` would count the shut completions' reverse
      // flow back in and report a rate the well is not making.
      const double q = constrained ? delivered : (sumAp - pwf * sumA);
      applied_rate()[w.name] =
          std::make_pair((int)wants[iw], q > 0 ? q : 0.0);
      rate_reached = target[iw] > 0 &&
                     q >= (1.0 - 1.0e-9) * (double)target[iw];
    }
    if (!std::isfinite(pwf) || pwf <= 0) continue;

    // Crossflow composition (WELLXFLOW 0 only): the wellbore's own surface-
    // component mixture, from the completions forward-flowing at the final
    // pwf. This is OPM's cmix_s. The receiving completion converts cmix_s
    // back to reservoir volume with its local B, Rs and Rv; carrying reservoir
    // phase fractions here is incorrect when source and receiving PVT differ.
    // One pass gathers it, then the assignment pass below
    // copies it onto every completion of the well -- including the
    // reverse-flowing ones that are the only ones that read it.  See
    // include/well_control.hpp and wellcell.hpp's mix_o/mix_w/mix_g.
    Real mix_o = 0, mix_w = 0, mix_g = 0;
    double Q_surf[3] = {0.0, 0.0, 0.0};
    double L_surf[3] = {0.0, 0.0, 0.0};
    double Qtot_surf = 0.0;
    auto evaluate_well_stream = [&](double trial_pwf, Real& co_out,
                                    Real& cw_out, Real& cg_out,
                                    double q_out[3]) {
      q_out[0] = q_out[1] = q_out[2] = 0.0;
      double gross[3] = {0.0, 0.0, 0.0};
      for (size_t ic = 0; ic < w.completions.size(); ++ic) {
        const Completion& c = w.completions[ic];
        if (c.cell < 0 || !c.open || c.cell >= (int)l2a.size()) continue;
        const int loc = l2a[c.cell];
        if (loc < 0 || loc >= cells) continue;
        const WellCell& k = wc[loc];
        if (k.is_inj || !(k.wi > 0)) continue;
        const size_t ci = conn_off[iw] + ic;
        const double a = conn_acc[2 * ci + 0];
        if (!(a > 0)) continue;
        const double theta = conn_acc[2 * ci + 1] / a;
        if (!conn_produces(theta, trial_pwf)) continue;
        const double draw = theta - trial_pwf;
        for (int ph = 0; ph < 3; ++ph) {
          const double sc = (double)conn_scoef[3 * ci + ph];
          gross[ph] += sc * draw;
        }
      }
      const double gross_total = gross[0] + gross[1] + gross[2];
      if (!(gross_total > 0.0)) return false;
      Real co = (Real)(gross[0] / gross_total);
      Real cw = (Real)(gross[1] / gross_total);
      Real cg = (Real)(gross[2] / gross_total);

        // Complete the wellbore component balances.  Gross forward
        // production is only the initial guess: reverse completions remove
        // surface components from the same well stream, and their reservoir
        // rate depends on that stream's composition through B, Rs and Rv.
        // OPM carries these fractions as well unknowns and solves the same
        // conservation equations.  The two-fraction fixed point is tiny and
        // host-only; damping keeps it inside the physical simplex at wells
        // whose net rate is momentarily close to zero.
        for (int it = 0; it < 40; ++it) {
          double net[3] = {gross[0], gross[1], gross[2]};
          for (size_t ic = 0; ic < w.completions.size(); ++ic) {
            const Completion& c = w.completions[ic];
            if (c.cell < 0 || !c.open) continue;
            const size_t ci = conn_off[iw] + ic;
            const double a = conn_acc[2 * ci];
            if (!(a > 0)) continue;
            const double theta = conn_acc[2 * ci + 1] / a;
            if (conn_produces(theta, trial_pwf)) continue;
            const double q = (double)conn_rev[6 * ci];
            const double bo = (double)conn_rev[6 * ci + 1];
            const double bw = (double)conn_rev[6 * ci + 2];
            const double bg = (double)conn_rev[6 * ci + 3];
            const double rs = (double)conn_rev[6 * ci + 4];
            const double rv = (double)conn_rev[6 * ci + 5];
            const double bgc = bg / (double)units::kFt3PerBbl;
            const double rsc = rs / (double)units::kFt3PerBbl;
            const double rvc = rv * (double)units::kFt3PerBbl;
            const double dis = std::max(1.0e-12, 1.0 - rs * rv);
            const double vr =
                (double)cw / bw +
                ((double)co - rvc * (double)cg) / (bo * dis) +
                ((double)cg - rsc * (double)co) / (bgc * dis);
            if (!(vr > 1.0e-20)) continue;
            const double qs = q * (theta - trial_pwf) / vr;
            net[0] += (double)co * qs;
            net[1] += (double)cw * qs;
            net[2] += (double)cg * qs;
          }
          const double no = std::max(0.0, net[0]);
          const double nw = std::max(0.0, net[1]);
          const double ng = std::max(0.0, net[2]);
          const double nt = no + nw + ng;
          if (!(nt > 0)) break;
          const double co_new = no / nt, cw_new = nw / nt, cg_new = ng / nt;
          const double err = std::max({std::fabs(co_new - (double)co),
                                       std::fabs(cw_new - (double)cw),
                                       std::fabs(cg_new - (double)cg)});
          co = (Real)(0.5 * ((double)co + co_new));
          cw = (Real)(0.5 * ((double)cw + cw_new));
          cg = (Real)(1.0 - (double)co - (double)cw);
          if (err < 1.0e-10) break;
        }
      q_out[0] = gross[0]; q_out[1] = gross[1]; q_out[2] = gross[2];
      for (size_t ic = 0; ic < w.completions.size(); ++ic) {
        const size_t ci = conn_off[iw] + ic;
        const double a = conn_acc[2 * ci];
        if (!(a > 0)) continue;
        const double theta = conn_acc[2 * ci + 1] / a;
        if (conn_produces(theta, trial_pwf)) continue;
        const double l = (double)conn_rev[6 * ci];
        const double bo = (double)conn_rev[6 * ci + 1];
        const double bw = (double)conn_rev[6 * ci + 2];
        const double bg = (double)conn_rev[6 * ci + 3];
        const double rs = (double)conn_rev[6 * ci + 4];
        const double rv = (double)conn_rev[6 * ci + 5];
        const double bgc = bg / (double)units::kFt3PerBbl;
        const double rsc = rs / (double)units::kFt3PerBbl;
        const double rvc = rv * (double)units::kFt3PerBbl;
        const double dis = std::max(1.0e-12, 1.0 - rs * rv);
        const double vr = (double)cw / bw +
            ((double)co - rvc * (double)cg) / (bo * dis) +
            ((double)cg - rsc * (double)co) / (bgc * dis);
        if (!(vr > 1.0e-20)) continue;
        const double qs = l * (theta - trial_pwf) / vr;
        q_out[0] += (double)co * qs;
        q_out[1] += (double)cw * qs;
        q_out[2] += (double)cg * qs;
      }
      co_out = co; cw_out = cw; cg_out = cg;
      return true;
    };

    if (xflow_composition) {
      auto controlled_common_rate = [&](const double q[3]) {
        switch (wants[iw]) {
          case 1: return q[0];
          case 2: return q[0] + q[1];
          case 3: return q[2];
          case 4: return q[1];
          default: return sumAp - pwf * sumA;
        }
      };
      if (wants[iw] >= 1 && wants[iw] <= 4) {
        const double p_hi = sumAp / sumA;
        const double floor_bhp = history_free
            ? (double)d.p_atm
            : (deck_gave_bhp ? (double)w.state.bhp_limit
                             : (double)d.prod_bhp);
        const double p_lo = std::min(p_hi, std::max(0.0, floor_bhp));
        const double tgt = wants[iw] == 3
            ? (double)target[iw] / (double)units::kFt3PerBbl
            : (double)target[iw];
        Real lo_o = 0, lo_w = 0, lo_g = 0, hi_o = 0, hi_w = 0, hi_g = 0;
        double qlo[3], qhi[3];
        const bool ok_lo = evaluate_well_stream(p_lo, lo_o, lo_w, lo_g, qlo);
        const bool ok_hi = evaluate_well_stream(p_hi, hi_o, hi_w, hi_g, qhi);
        if (ok_lo && controlled_common_rate(qlo) >= tgt &&
            (!ok_hi || controlled_common_rate(qhi) <= tgt)) {
          double lo = p_lo, hi = p_hi;
          for (int it = 0; it < 32; ++it) {
            const double mid = 0.5 * (lo + hi);
            Real mo = 0, mw = 0, mg = 0;
            double qm[3];
            if (evaluate_well_stream(mid, mo, mw, mg, qm) &&
                controlled_common_rate(qm) > tgt)
              lo = mid;
            else
              hi = mid;
          }
          pwf = 0.5 * (lo + hi);
        } else if (ok_lo && controlled_common_rate(qlo) < tgt) {
          pwf = p_lo;
        }
      }
      if (evaluate_well_stream(pwf, mix_o, mix_w, mix_g, Q_surf)) {
        Qtot_surf = Q_surf[0] + Q_surf[1] + Q_surf[2];
        for (size_t ic = 0; ic < w.completions.size(); ++ic) {
          const size_t ci = conn_off[iw] + ic;
          const double a = conn_acc[2 * ci];
          if (!(a > 0)) continue;
          const double theta = conn_acc[2 * ci + 1] / a;
          if (!conn_produces(theta, pwf)) continue;
          for (int ph = 0; ph < 3; ++ph)
            L_surf[ph] += (double)conn_scoef[3 * ci + ph];
        }
      }
      if (wants[iw] >= 1 && wants[iw] <= 4) {
        double actual = controlled_common_rate(Q_surf);
        if (wants[iw] == 3) actual *= (double)units::kFt3PerBbl;
        applied_rate()[w.name] =
            std::make_pair((int)wants[iw], std::max(0.0, actual));
        rate_reached = target[iw] > 0 &&
            actual >= (1.0 - 1.0e-8) * (double)target[iw];
      }
    }

    // For a rate-controlled well that is not on its BHP bound,
    // d(pwf)/d(p_k) is A_k/sum(A).  A no-crossflow control excludes reverse
    // completions from that denominator; RESV's composition-aware solve uses
    // all completions.  Group targets are excluded for now because their
    // productivity allocation adds a second, inter-well derivative.
    double schur_sumA = 0.0;
    const bool schur_well = r.well_schur_enabled && rate_reached &&
                            !grp_kind[iw] && wants[iw] > 0 && wants[iw] <= 5;
    if (schur_well) {
      for (size_t ic = 0; ic < w.completions.size(); ++ic) {
        const size_t ci = conn_off[iw] + ic;
        const double a = conn_acc[2 * ci];
        if (!(a > 0)) continue;
        const double theta = conn_acc[2 * ci + 1] / a;
        const bool forward = conn_produces(theta, pwf);
        if (!constrained || forward) schur_sumA += a;
      }
      // Keep the pressure part of the eliminated-well column through
      // crossflow.  At frozen mobility this is the exact derivative of the
      // common-BHP rate equation; only the wellbore-mixture derivative is
      // absent.  Dropping the whole column whenever one completion reversed
      // left the hardest wells with only local Jacobian terms and caused 22
      // extra timestep chops on NORNE.
    }

    // Exact local well block for [pwf, surface-oil fraction,
    // surface-water fraction].  Reservoir rows already contain C=dR/du and
    // dr_dg below is B=dR/dg.  The eliminated mode derivative must therefore
    // be dg/du=-D^-1 Cw, where D is the three well equations: active control
    // plus two component-fraction balances.  Mode 6 enables this path.
    double well_D[9] = {};
    double well_Q[3] = {};
    bool has_reverse_completion = false;
    if (xflow_composition) {
      for (size_t jc = 0; jc < w.completions.size(); ++jc) {
        const size_t cj = conn_off[iw] + jc;
        const double a = conn_acc[2 * cj];
        if (!(a > 0)) continue;
        const double theta = conn_acc[2 * cj + 1] / a;
        if (!conn_produces(theta, pwf)) {
          has_reverse_completion = true;
          break;
        }
      }
    }
    // Mode 7 is a semismooth hybrid: the complete three-unknown block is
    // indispensable when reverse flow makes the wellbore mixture an active
    // reservoir unknown. For an all-forward well those two mixture columns
    // have no reservoir-side B block, so use the empirically more robust
    // frozen-mobility pressure column until crossflow actually appears.
    const bool exact_well_block =
        (r.well_schur_enabled == 6 ||
         ((r.well_schur_enabled == 7 || r.well_schur_enabled == 8 ||
           r.well_schur_enabled == 9) &&
          has_reverse_completion)) &&
        schur_sumA > 0 &&
        xflow_composition && Qtot_surf > 0;
    if (exact_well_block) {
      double dQdg[3][3] = {};
      for (size_t jc = 0; jc < w.completions.size(); ++jc) {
        const size_t cj = conn_off[iw] + jc;
        const double a = conn_acc[2 * cj];
        if (!(a > 0)) continue;
        const double theta = conn_acc[2 * cj + 1] / a;
        const double draw = theta - pwf;
        if (conn_produces(theta, pwf)) {
          for (int ph = 0; ph < 3; ++ph) {
            const double sc = (double)conn_scoef[3 * cj + ph];
            well_Q[ph] += sc * draw;
            dQdg[ph][0] -= sc;
          }
        } else {
          const double l = (double)conn_rev[6 * cj];
          const double bo = (double)conn_rev[6 * cj + 1];
          const double bw = (double)conn_rev[6 * cj + 2];
          const double bg = (double)conn_rev[6 * cj + 3];
          const double rs = (double)conn_rev[6 * cj + 4];
          const double rv = (double)conn_rev[6 * cj + 5];
          const double bgc = bg / (double)units::kFt3PerBbl;
          const double rsc = rs / (double)units::kFt3PerBbl;
          const double rvc = rv * (double)units::kFt3PerBbl;
          const double dis = std::max(1.0e-12, 1.0 - rs * rv);
          const double vr =
              (double)mix_w / bw +
              ((double)mix_o - rvc * (double)mix_g) / (bo * dis) +
              ((double)mix_g - rsc * (double)mix_o) / (bgc * dis);
          if (!(vr > 1.0e-20)) continue;
          const double qres = l * draw;
          const double qs = qres / vr;
          const double cfrac[3] = {(double)mix_o, (double)mix_w,
                                   (double)mix_g};
          const double dvr_dco =
              (1.0 + rvc) / (bo * dis) -
              (1.0 + rsc) / (bgc * dis);
          const double dvr_dcw =
              1.0 / bw + rvc / (bo * dis) - 1.0 / (bgc * dis);
          for (int ph = 0; ph < 3; ++ph) {
            well_Q[ph] += cfrac[ph] * qs;
            dQdg[ph][0] += cfrac[ph] * (-l / vr);
            dQdg[ph][1] +=
                ((ph == 0 ? 1.0 : (ph == 2 ? -1.0 : 0.0)) -
                 cfrac[ph] * dvr_dco / vr) * qs;
            dQdg[ph][2] +=
                ((ph == 1 ? 1.0 : (ph == 2 ? -1.0 : 0.0)) -
                 cfrac[ph] * dvr_dcw / vr) * qs;
          }
        }
      }
      const double qt = well_Q[0] + well_Q[1] + well_Q[2];
      // The first row is the actual active control equation.  RESV controls
      // total reservoir voidage and retains the mobility-sum derivative.
      // Surface controls act on the net component stream after reverse-flow
      // conversion, so their pwf and mixture columns must come from dQdg.
      double ctrl_weight[3] = {};
      if (wants[iw] == 1) ctrl_weight[0] = 1.0;
      else if (wants[iw] == 2)
        ctrl_weight[0] = ctrl_weight[1] = 1.0;
      else if (wants[iw] == 3) ctrl_weight[2] = 1.0;
      else if (wants[iw] == 4) ctrl_weight[1] = 1.0;
      if (wants[iw] >= 1 && wants[iw] <= 4) {
        for (int g = 0; g < 3; ++g)
          for (int ph = 0; ph < 3; ++ph)
            well_D[g] += ctrl_weight[ph] * dQdg[ph][g];
      } else {
        well_D[0] = -schur_sumA;
      }
      for (int g = 0; g < 3; ++g) {
        const double dqt = dQdg[0][g] + dQdg[1][g] + dQdg[2][g];
        well_D[3 + g] = dQdg[0][g] -
                        (g == 1 ? qt : 0.0) - (double)mix_o * dqt;
        well_D[6 + g] = dQdg[1][g] -
                        (g == 2 ? qt : 0.0) - (double)mix_w * dqt;
      }
    }

    auto solve_well_D = [&](const double rhs[3], double x[3]) {
      double M[3][4];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = well_D[3 * i + j];
        M[i][3] = rhs[i];
      }
      for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int row = col + 1; row < 3; ++row)
          if (std::fabs(M[row][col]) > std::fabs(M[piv][col])) piv = row;
        if (!(std::fabs(M[piv][col]) > 1.0e-24)) {
          x[0] = x[1] = x[2] = 0.0;
          return;
        }
        if (piv != col)
          for (int j = col; j < 4; ++j) std::swap(M[col][j], M[piv][j]);
        const double inv = 1.0 / M[col][col];
        for (int j = col; j < 4; ++j) M[col][j] *= inv;
        for (int row = 0; row < 3; ++row) {
          if (row == col) continue;
          const double f = M[row][col];
          for (int j = col; j < 4; ++j) M[row][j] -= f * M[col][j];
        }
      }
      for (int i = 0; i < 3; ++i) x[i] = M[i][3];
    };

    for (size_t ic = 0; ic < w.completions.size(); ++ic) {
      const Completion& c = w.completions[ic];
      if (c.cell < 0 || !c.open) continue;
      if (c.cell >= (int)l2a.size()) continue;
      const int loc = l2a[c.cell];
      if (loc < 0 || loc >= cells) continue;
      if (wc[loc].is_inj || !(wc[loc].wi > 0)) continue;
      const size_t ci = conn_off[iw] + ic;

      if (xflow_composition) {
        // Keep every completion with live mobility in the assembly -- the
        // sign of (cell pressure - pwf) is now what the kernel branches on,
        // not whether this routine shut the completion.  A completion the
        // no-crossflow target solve excluded still flows: it takes the
        // wellbore's own composition (mix_o/mix_w/mix_g) instead of nothing,
        // which is OPM's cmix_s branch rather than its !allow_cf one.  See
        // include/well_control.hpp for why Norne's WELSPECS asks for the
        // former.
        const double a = conn_acc[2 * ci + 0];
        const double theta = (a > 0) ? conn_acc[2 * ci + 1] / a : 0.0;
        const bool producing = (a > 0) && conn_produces(theta, pwf);
        wc[loc].is_prod = (a > 0) ? 1 : 0;
        wc[loc].bhp = (Real)(pwf + (rate_reached
                                        ? scratch.report_bhp_shift[iw] : 0.0));
        wc[loc].pwf = (a > 0) ? (Real)(pwf + conn_dp[ci])
                              : U(loc, 0);   // no mobility, no flow
        wc[loc].xflow_model = (a > 0) ? 1 : 0;
        wc[loc].mix_o = mix_o; wc[loc].mix_w = mix_w; wc[loc].mix_g = mix_g;
        if (fm.part.rank == 0 && debug_well() && w.name == debug_well())
          std::printf("  [dbg %s] step %d ic=%zu (i%d j%d k%d) theta=%.6g "
                      "pwf=%.6g %s\n",
                      w.name.c_str(), step, ic, c.i, c.j, c.k, theta, pwf,
                      producing ? "OPEN" : "xflow");
      } else if (wc[loc].is_prod) {
        wc[loc].bhp = (Real)(pwf + (rate_reached
                                        ? scratch.report_bhp_shift[iw] : 0.0));
        wc[loc].pwf = (Real)(pwf + conn_dp[ci]);
      }

      if (schur_sumA > 0) {
        const double a = conn_acc[2 * ci];
        if (!(a > 0)) continue;
        const double theta = conn_acc[2 * ci + 1] / a;
        const bool forward = conn_produces(theta, pwf);
        WellSchurEntry e;
        e.cell = loc;
        e.group = (int)iw;
        if (r.well_schur_enabled == 1) {
          e.dg_du[0] = (!constrained || forward)
                           ? (Real)(a / schur_sumA) : (Real)0;
        } else {
          // Implicit derivative of the active rate-control equation
          //   sum A_k (theta_k - pwf) - target = 0.
          // Modes 1 and 2 are two independent wellbore fractions; gas is
          // 1-mix_o-mix_w. They matter only for reverse-flow residuals.
          for (int v = 0; v < bs && v < 3; ++v) {
            const bool included = !constrained || forward;
            double num = a * (double)conn_dtheta_du[3 * ci + v];
            // Mode 2 is the complete control-pressure derivative. Mode 3
            // deliberately freezes mobility to isolate the nonsmooth term;
            // mode 4 adds mixture coupling to the complete derivative.
            if (r.well_schur_enabled == 2 || r.well_schur_enabled >= 4)
              num += (double)conn_dA_du[3 * ci + v] * (theta - pwf);
            // Mode 5 combines the validated frozen-mobility pressure column
            // with the complete surface-mixture columns.  This isolates the
            // missing wellbore-composition derivative from dA/du in the
            // nonsmooth active-control equation.
            const double dpwf =
                (r.well_schur_enabled == 5 ||
                 ((r.well_schur_enabled == 7 ||
                   r.well_schur_enabled == 8 ||
                   r.well_schur_enabled == 9) && !exact_well_block))
                    ? ((included && v == 0) ? a / schur_sumA : 0.0)
                    : (included ? num / schur_sumA : 0.0);
            e.dg_du[v] = (Real)dpwf;
            if (exact_well_block) {
              double dQ[3] = {};
              if (forward) {
                for (int ph = 0; ph < 3; ++ph)
                  dQ[ph] =
                      (double)conn_dscoef_du[9 * ci + 3 * ph + v] *
                          (theta - pwf) +
                      (double)conn_scoef[3 * ci + ph] *
                          (double)conn_dtheta_du[3 * ci + v];
              } else {
                const double l = (double)conn_rev[6 * ci];
                const double bo = (double)conn_rev[6 * ci + 1];
                const double bw = (double)conn_rev[6 * ci + 2];
                const double bg = (double)conn_rev[6 * ci + 3];
                const double rs = (double)conn_rev[6 * ci + 4];
                const double rv = (double)conn_rev[6 * ci + 5];
                const double bgc = bg / (double)units::kFt3PerBbl;
                const double rsc = rs / (double)units::kFt3PerBbl;
                const double rvc = rv * (double)units::kFt3PerBbl;
                const double dis = std::max(1.0e-12, 1.0 - rs * rv);
                const double vro =
                    ((double)mix_o - rvc * (double)mix_g) / (bo * dis);
                const double vrw = (double)mix_w / bw;
                const double vrg =
                    ((double)mix_g - rsc * (double)mix_o) / (bgc * dis);
                const double vr = std::max(1.0e-20, vro + vrw + vrg);
                const double dbo = (double)conn_dpvt_du[15 * ci + v];
                const double dbw =
                    (double)conn_dpvt_du[15 * ci + 3 + v];
                const double dbg =
                    (double)conn_dpvt_du[15 * ci + 6 + v] /
                    (double)units::kFt3PerBbl;
                const double drs =
                    (double)conn_dpvt_du[15 * ci + 9 + v] /
                    (double)units::kFt3PerBbl;
                const double drv =
                    (double)conn_dpvt_du[15 * ci + 12 + v] *
                    (double)units::kFt3PerBbl;
                const double ddis = -(drs * rvc + rsc * drv);
                const double dvro = -drv * (double)mix_g / (bo * dis) -
                    vro * (dbo / bo + ddis / dis);
                const double dvrw = -(double)mix_w * dbw / (bw * bw);
                const double dvrg = -drs * (double)mix_o / (bgc * dis) -
                    vrg * (dbg / bgc + ddis / dis);
                double dl = 0.0;
                for (int ph = 0; ph < 3; ++ph)
                  dl += (double)wc[loc].wi *
                        (double)conn_dmob_du[9 * ci + 3 * ph + v];
                const double qres = l * (theta - pwf);
                const double dqres =
                    dl * (theta - pwf) +
                    l * (double)conn_dtheta_du[3 * ci + v];
                const double dqs =
                    (dqres * vr - qres * (dvro + dvrw + dvrg)) /
                    (vr * vr);
                dQ[0] = (double)mix_o * dqs;
                dQ[1] = (double)mix_w * dqs;
                dQ[2] = (double)mix_g * dqs;
              }
              const double dqt = dQ[0] + dQ[1] + dQ[2];
              double control_C = included ? num : 0.0;
              if (wants[iw] >= 1 && wants[iw] <= 4) {
                control_C = 0.0;
                if (wants[iw] == 1) control_C = dQ[0];
                else if (wants[iw] == 2) control_C = dQ[0] + dQ[1];
                else if (wants[iw] == 3) control_C = dQ[2];
                else control_C = dQ[1];
              }
              const double Cw[3] = {
                  control_C,
                  dQ[0] - (double)mix_o * dqt,
                  dQ[1] - (double)mix_w * dqt};
              const double rhs[3] = {-Cw[0], -Cw[1], -Cw[2]};
              double dg[3];
              solve_well_D(rhs, dg);
              e.dg_du[v] = (Real)dg[0];
              e.dg_du[3 + v] = (Real)dg[1];
              e.dg_du[6 + v] = (Real)dg[2];
            } else if (r.well_schur_enabled >= 4 && xflow_composition &&
                Qtot_surf > 0) {
              double dQ[3];
              for (int ph = 0; ph < 3; ++ph) {
                const double direct = forward
                    ? (double)conn_dscoef_du[9 * ci + 3 * ph + v] *
                              (theta - pwf) +
                          (double)conn_scoef[3 * ci + ph] *
                              (double)conn_dtheta_du[3 * ci + v]
                    : 0.0;
                dQ[ph] = direct - L_surf[ph] * dpwf;
              }
              const double dQt = dQ[0] + dQ[1] + dQ[2];
              const double den = Qtot_surf * Qtot_surf;
              e.dg_du[3 + v] =
                  (Real)((dQ[0] * Qtot_surf - Q_surf[0] * dQt) / den);
              e.dg_du[6 + v] =
                  (Real)((dQ[1] * Qtot_surf - Q_surf[1] * dQt) / den);
            }
          }
        }
        if (forward || !xflow_composition) {
          for (int eq = 0; eq < 3; ++eq)
            e.dr_dg[3 * eq] = conn_drdpw[3 * ci + eq];
        } else {
          const Real q = conn_rev[6 * ci];
          const Real bo = conn_rev[6 * ci + 1];
          const Real bw = conn_rev[6 * ci + 2];
          const Real bg = conn_rev[6 * ci + 3];
          const Real rs = conn_rev[6 * ci + 4];
          const Real rv = conn_rev[6 * ci + 5];
          const Real kvol = (Real)units::kFt3PerBbl;
          const Real bgc = bg / kvol, rsc = rs / kvol, rvc = rv * kvol;
          const Real dis = std::max((Real)1e-12, (Real)1 - rs * rv);
          const Real vro = (mix_o - rvc * mix_g) / (bo * dis);
          const Real vrw = mix_w / bw;
          const Real vrg = (mix_g - rsc * mix_o) / (bgc * dis);
          const Real vr = std::max((Real)1e-20, vro + vrw + vrg);
          // q_res = q*(theta-pwf), so d(q_surface)/d(pwf)=-q/vr.
          e.dr_dg[0] = -mix_o * q / vr;
          e.dr_dg[3] = -mix_w * q / vr;
          e.dr_dg[6] = -kvol * mix_g * q / vr;
          if (r.well_schur_enabled >= 4) {
            // The independent mixture modes are co and cw, with
            // cg=1-co-cw. Differentiate both the component fraction and the
            // black-oil surface-to-reservoir volume ratio.
            const Real q_res = q * (Real)(theta - pwf);
            const Real q_s = q_res / vr;
            const Real dvr_dco =
                ((Real)1 + rvc) / (bo * dis) -
                ((Real)1 + rsc) / (bgc * dis);
            const Real dvr_dcw =
                (Real)1 / bw + rvc / (bo * dis) - (Real)1 / (bgc * dis);
            const Real scale_o = q_s * dvr_dco / vr;
            const Real scale_w = q_s * dvr_dcw / vr;
            e.dr_dg[1] = q_s - mix_o * scale_o;
            e.dr_dg[2] = -mix_o * scale_w;
            e.dr_dg[4] = -mix_w * scale_o;
            e.dr_dg[5] = q_s - mix_w * scale_w;
            e.dr_dg[7] = kvol * (-q_s - mix_g * scale_o);
            e.dr_dg[8] = kvol * (-q_s - mix_g * scale_w);
          }
        }
        schur.push_back(e);
      }
    }
  }

  upload_schedule_wellcells(r, sc, l2a, cells, wc);
  if (r.well_schur_enabled && !schur.empty()) {
    // The device reduction locates each well's contiguous range by binary
    // search.  Producer-only assembly is naturally in schedule order, but
    // mode 8 contributes injector entries before the producer loop.  Restore
    // the ordering invariant explicitly before upload.
    std::stable_sort(schur.begin(), schur.end(),
                     [](const WellSchurEntry& a, const WellSchurEntry& b) {
                       return a.group < b.group;
                     });
    if ((int)schur.size() > r.well_schur_capacity)
      throw std::runtime_error("well Schur completion list exceeds capacity");
    NVRS_CHECK_CUDA(cudaMemcpy(r.well_schur, schur.data(),
                               sizeof(WellSchurEntry) * schur.size(),
                               cudaMemcpyHostToDevice));
    r.well_schur_count = (int)schur.size();
    r.well_schur_groups = (int)nw;
  }
  if (use_handoff) {
    // Retain state-buffer capacity for the next paired solve; `ready`
    // remains false, so it cannot be mistaken for fresh data.
    handoff.u = std::move(u);
  }

  // Asked against delivered, summed over the rate-controlled producers.
  //
  // Every producer on Norne is rate-controlled and Flow runs the same targets
  // and meets them, yet NVRS recovers 227 M STB against Flow's 422.  Only two
  // things can be true: the wells are not reaching their targets, or the
  // targets are not the ones Flow is using.  They need opposite fixes -- more
  // drawdown or a parsing correction -- and this is the one number that says
  // which.  q here is what the well will make at the pressure it ends up
  // with, after the bottom-hole limit has had its say.
  if (fm.part.rank == 0) {
    // EVERY rate kind, not just oil.
    //
    // This block existed to answer exactly the question above and could not,
    // because it filtered on `wants[iw] != 1` -- oil-rate control only.  Norne
    // is a history deck: 1,970 of its 2,109 WCONHIST records name RESV and
    // only 139 name ORAT, and those 139 are the zero-rate ones.  So for the
    // whole of Norne this printed "asked 0 STB/day, will deliver 0 (0%)",
    // 1,188 times, about the handful of wells that were switched off, and said
    // nothing whatever about the 94% of producers actually carrying the field.
    //
    // The one number that separates "the wells are not reaching their targets"
    // from "the targets are not the ones Flow is using" was reporting on the
    // wrong wells.
    //
    // Reported per kind, because the units differ -- oil, liquid and water
    // targets are stb/day, gas is scf/day, and RESV is a reservoir rate -- and
    // summing them would be meaningless.
    static const char* kind_name[6] = {"", "oil", "liquid", "gas", "water",
                                       "reservoir-volume"};
    static const char* kind_unit[6] = {"", "STB/day", "STB/day", "scf/day",
                                       "STB/day", "rb/day"};
    double want[6] = {0, 0, 0, 0, 0, 0}, got[6] = {0, 0, 0, 0, 0, 0};
    int n_short[6] = {0, 0, 0, 0, 0, 0}, n_rate[6] = {0, 0, 0, 0, 0, 0};
    for (size_t iw = 0; iw < nw; ++iw) {
      const int k = (int)wants[iw];
      if (k <= 0 || k > 5) continue;             // 9 is "no target at all"
      auto it = applied_rate().find(sc.wells[iw].name);
      if (it == applied_rate().end()) continue;
      ++n_rate[k];
      want[k] += (double)target[iw];
      got[k] += it->second.second;
      if (it->second.second < 0.99 * (double)target[iw]) ++n_short[k];
    }
    static int said_ask = 0;
    const bool due = ((said_ask++ % 40) == 0);
    for (int k = 1; k <= 5 && due; ++k) {
      if (n_rate[k] == 0) continue;
      std::printf("  [ask] %d %s-rate producer(s): asked %.4g %s, will deliver "
                  "%.4g (%.0f%%); %d short of target\n",
                  n_rate[k], kind_name[k], want[k], kind_unit[k], got[k],
                  want[k] > 0 ? 100.0 * got[k] / want[k] : 0.0, n_short[k]);
    }
  }

  // Two separate one-shots, and the distinction matters more than it looks.
  //
  // The first version latched on the FIRST call, which is the initial report
  // step, where no well has yet had time to deplete anything and nothing
  // crossflows.  It therefore printed "this changes nothing" once and stayed
  // silent for the rest of a run in which it went on to shut completions on
  // four wells -- a log line that was not merely useless but actively
  // misleading, and it cost a full Norne cycle to see through.
  //
  // So: say the mode is on when the controls are first solved, and say what it
  // did the first time it actually does something.
  if (fm.part.rank == 0 && d.well_crossflow == 2) {
    static int said_mixture_mode = 0;
    if (!said_mixture_mode) {
      said_mixture_mode = 1;
      std::printf("  [well] WELSPECS crossflow active with common-wellbore "
                  "mixture (WELLXFLOW 2)\n");
    }
  } else if (fm.part.rank == 0 && !d.well_crossflow) {
    static int said_mode = 0;
    if (!said_mode) {
      said_mode = 1;
      std::printf("  [well] no-crossflow controls active: a producer's rate is "
                  "carried only by completions above its flowing pressure "
                  "(WELLXFLOW 1 for the old behaviour)\n");
    }
    static int said_xf = 0;
    if (!said_xf && xflow_wells > 0) {
      said_xf = 1;
      std::printf("  [well] first crossflow shut-in: %d producer(s), %d "
                  "completion(s) below their flowing pressure\n",
                  xflow_wells, xflow_shut);
    }
  }

  if (fm.part.rank == 0 && carried > 0) {
    static int said_carry = 0;
    if (!said_carry) {
      said_carry = 1;
      std::printf("  [well] %d producer(s) held at their last stated rate "
                  "(the deck defers them to group control, which is not "
                  "implemented)\n", carried);
    }
  }
  static int reported_bhp_step = -1;
  if (fm.part.rank == 0 && !unconstrained.empty() &&
      reported_bhp_step != step) {
    reported_bhp_step = step;
    std::printf("  [well] no rate constraint, running on BHP alone:");
    int shown = 0;
    for (int iw : unconstrained) {
      if (shown++ == 6) { std::printf(" ..."); break; }
      std::printf(" %s(%s)", sc.wells[iw].name.c_str(),
                  to_string(sc.wells[iw].state.mode));
    }
    std::printf("\n");
    std::fflush(stdout);
  }
}


double field_oil_in_place(const FieldModel& fm, const Reservoir& r,
                          const PvtTables* pvt, double* water_stb,
                          double* clamp_stb) {
  refresh_drsdt_cap(fm, r);   // once; see the note on rs_anchor_of
  refresh_gas_primary(fm, r);
  refresh_vappars_host(fm, r);
  const int cells = fm.part.n_owned;
  const int bs = r.bs;
  if (cells <= 0 || !r.u) return 0.0;

  std::vector<Real> u((size_t)bs * cells);
  if (cudaMemcpy(u.data(), r.u, sizeof(Real) * u.size(),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return 0.0;

  const std::vector<HostPvt>& hp = host_pvt_regions(pvt);

  // The Newton limiter's running effect on oil, over the same cells and with
  // the same shrinkage the in-place sum uses, so the two are comparable STB.
  std::vector<Real> cdso;
  if (clamp_stb && r.clamp_dso) {
    cdso.resize((size_t)cells);
    if (cudaMemcpy(cdso.data(), r.clamp_dso, sizeof(Real) * cdso.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      cdso.clear();
  }
  double clamp = 0.0;

  // The grid's own pore volume, which already carries NTG, MULTPV and MINPV.
  const std::vector<int>& a2l = fm.part.active_of_local;
  double n = 0.0, w = 0.0;
  for (int c = 0; c < cells; ++c) {
    const int a = (c < (int)a2l.size()) ? a2l[c] : -1;
    if (a < 0 || a >= (int)fm.grid.porv.size()) continue;
    const Real p = u[(size_t)bs * c];
    const Real sw = (bs >= 2) ? u[(size_t)bs * c + 1] : (Real)0;
    const Real x = (bs >= 3) ? u[(size_t)bs * c + 2] : (Real)0;
    const Real sg = physical_sg_at(fm, r, a, sw, x);
    const double so = 1.0 - (double)sw - (double)sg;
    const DevicePvt pv = pvt_for_active(hp, fm, a);
    const PvtCell F = pv.valid ? pvt_cell_at(pv, p, sw, x, a, fm, r)
                               : const_pvt_cell(r.bo, r.bw, r.bg, r.rs,
                                                r.uo, r.uw, r.ug);
    const double pvfac = r.have_rock
                             ? (double)rock_pv_mult(p, r.rock_pref, r.ct)
                             : 1.0;

    // Water first, and NOT behind the oil guard.
    //
    // `if (!(so > 0)) continue;` used to sit above this, so every cell with
    // no oil was skipped -- which is the whole aquifer, where Sw = 1 and most
    // of Norne's water is.  Water in place came out at 498 M STB against the
    // 2624 M reservoir bbl the same field gives through compare.py, and the
    // balance then reported the field gaining water the wells never injected.
    // The impossibility was the meter, not the reservoir.
    if (F.bw > 0 && std::isfinite((double)F.bw) && sw > 0)
      w += (double)fm.grid.porv[a] * pvfac * (double)sw * (double)F.bw;

    const double oil_phase = so > 0 ? so * (double)F.bo : 0.0;
    const double vap_oil = sg > 0 ? (double)sg * (double)F.rv * (double)F.bg
                                  : 0.0;
    if (std::isfinite(oil_phase) && std::isfinite(vap_oil))
      n += (double)fm.grid.porv[a] * pvfac * (oil_phase + vap_oil);

    if (!cdso.empty() && F.bo > 0 && std::isfinite((double)cdso[c]))
      clamp += (double)fm.grid.porv[a] * pvfac * (double)cdso[c] * (double)F.bo;
  }

  if (fm.part.nranks > 1) {
    double acc[3] = {n, w, clamp};
    MPI_Allreduce(MPI_IN_PLACE, acc, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    n = acc[0]; w = acc[1]; clamp = acc[2];
  }
  if (water_stb) *water_stb = w;
  if (clamp_stb) *clamp_stb = clamp;
  return n;
}

void report_worst_cells(const Reservoir& r, const Real* b_dev,
                        const FieldModel* fm, int topn) {
  const int cells = fm ? fm->part.n_owned : 0;
  const int bs = r.bs;
  if (cells <= 0 || !b_dev || bs < 1) return;

  std::vector<Real> b((size_t)bs * cells), u((size_t)bs * cells),
      poro((size_t)cells);
  std::vector<Real> meaning;
  std::vector<WellCell> wc((size_t)cells);
  if (cudaMemcpy(b.data(), b_dev, sizeof(Real) * b.size(),
                 cudaMemcpyDeviceToHost) != cudaSuccess) return;
  cudaMemcpy(u.data(), r.u, sizeof(Real) * u.size(), cudaMemcpyDeviceToHost);
  if (r.gas_primary) {
    meaning.resize((size_t)cells);
    cudaMemcpy(meaning.data(), r.gas_primary, sizeof(Real) * meaning.size(),
               cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(poro.data(), r.poro, sizeof(Real) * cells, cudaMemcpyDeviceToHost);
  if (r.wellcells)
    cudaMemcpy(wc.data(), r.wellcells, sizeof(WellCell) * cells,
               cudaMemcpyDeviceToHost);

  static const char* eqname[3] = {"oil", "water", "gas"};
  for (int a = 0; a < bs && a < 3; ++a) {
    // Sort this equation's cells by |residual|, biggest first.
    std::vector<std::pair<double, int>> v;
    v.reserve(cells);
    double total = 0;
    for (int c = 0; c < cells; ++c) {
      const double x = std::fabs((double)b[(size_t)bs * c + a]);
      v.push_back({x, c});
      total += x * x;
    }
    std::partial_sort(v.begin(), v.begin() + std::min<int>(topn, cells),
                      v.end(), std::greater<std::pair<double, int>>());
    total = std::sqrt(total);
    if (!(total > 0)) continue;

    // How concentrated is it?  One cell means a well; thousands means the
    // equation itself is off.
    const double cut = 0.01 * v[0].first;
    int nhot = 0;
    for (const auto& e : v) { if (e.first >= cut) ++nhot; else break; }

    std::printf("  [worst] %-5s |R|=%.4e over %d cell(s) within 1%% of the "
                "peak\n", eqname[a], total, nhot);
    for (int k = 0; k < topn && k < (int)v.size(); ++k) {
      const int c = v[k].second;
      const int act = (c < (int)fm->part.active_of_local.size())
                          ? fm->part.active_of_local[c] : -1;
      int gi = -1, gj = -1, gk = -1;
      if (act >= 0 && act < fm->grid.n_active()) {
        const int g = fm->grid.global_of_active(act);
        gi = g % fm->grid.nx;
        gj = (g / fm->grid.nx) % fm->grid.ny;
        gk = g / (fm->grid.nx * fm->grid.ny);
      }
      const Real pc = u[(size_t)bs * c];
      const Real sw = u[(size_t)bs * c + 1];
      const Real kind = meaning.empty() ? (Real)0 : meaning[(size_t)c];
      const Real sg = bs != 3 ? (Real)0
                      : kind > (Real)1.5 ? (Real)1 - sw
                      : kind > (Real)0.5 ? (Real)0
                                         : u[(size_t)bs * c + 2];
      const WellCell& k2 = wc[c];
      std::printf("          #%d cell %d (i%d j%d k%d) R=%.4e  p=%.1f "
                  "Sw=%.4f Sg=%.4f poro=%.4f\n",
                  k + 1, c, gi + 1, gj + 1, gk + 1, (double)b[(size_t)bs * c + a],
                  (double)pc, (double)sw, (double)sg, (double)poro[c]);
      if (k2.is_prod || k2.is_inj)
        std::printf("             well: %s wi=%.4e pwf=%.1f "
                    "qw=%.4e qo=%.4e qg=%.4e\n",
                    k2.is_inj ? "INJECTOR" : "producer", (double)k2.wi,
                    (double)k2.pwf, (double)k2.qw_inj, (double)k2.qo_inj,
                    (double)k2.qg_inj);
      else
        std::printf("             not perforated\n");
    }
  }
  std::fflush(stdout);
}

}  // namespace nvrs
