// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// ecl_io.hpp — ECLIPSE / OPM Flow compatible output.
//
// Why
// ---
// Native `state_*.bin` snapshots are convenient for this repository's own
// post-processor and useless to everyone else.  Writing the ECLIPSE family of
// files instead means the results open directly in ResInsight, OPM's `summary`
// and `compare_ECL_*` tools, Petrel, and anything else in the standard
// reservoir-engineering toolchain — and it lets us diff against an OPM Flow run
// of the same deck, which is the strongest verification available.
//
// Files produced (matching what OPM Flow emits for a run named CASE)
// ------------------------------------------------------------------
//   CASE.EGRID    grid geometry: COORD / ZCORN corner-point pillars + ACTNUM
//   CASE.INIT     static properties: PORO, PERMX/Y/Z, DX/DY/DZ, TOPS, PORV
//   CASE.UNRST    unified restart: PRESSURE, SWAT, SGAS, SOIL per report step
//   CASE.SMSPEC   summary vector specification (names, units, well names)
//   CASE.UNSMRY   unified summary data (the time series themselves)
//   CASE.RSM      the human-readable ASCII summary ECLIPSE prints
//
// Binary format
// -------------
// All binary files use the ECLIPSE "unformatted" convention: Fortran sequential
// records, big-endian, each record bracketed by a 4-byte byte-count marker.
// A named array is a header record
//
//     [int32 16]["KEYWORD "][int32 count]["TYPE"][int32 16]
//
// followed by the data split into blocks of at most 1000 elements (105 for
// CHAR), each block bracketed by its own byte-count markers.  Types are INTE
// (int32), REAL (float32), DOUB (float64), LOGI (int32 0/1), CHAR (8 bytes,
// blank padded) and MESS (header only, no data).
#pragma once

#include "common.hpp"
#include "deck.hpp"
#include "grid.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace nvrs {
namespace ecl {

// ---------------------------------------------------------------------------
// Low-level writer for ECLIPSE unformatted files.
// ---------------------------------------------------------------------------
class UnformattedFile {
 public:
  explicit UnformattedFile(const std::string& path, bool append = false);
  ~UnformattedFile();

  bool good() const { return f_.good(); }

  void inte(const std::string& kw, const std::vector<int>& v);
  void real(const std::string& kw, const std::vector<float>& v);
  void doub(const std::string& kw, const std::vector<double>& v);
  void logi(const std::string& kw, const std::vector<int>& v);   // 0 / 1
  void chars(const std::string& kw, const std::vector<std::string>& v);
  void mess(const std::string& kw);

 private:
  std::ofstream f_;

  void put_i32(int32_t v);                 // big-endian
  void put_f32(float v);                   // big-endian
  void put_f64(double v);                  // big-endian
  void header(const std::string& kw, int count, const char* type);
  template <typename T, typename PutFn>
  void blocked(const std::vector<T>& v, int block, int elem_bytes, PutFn put);
};

// ---------------------------------------------------------------------------
// A summary vector: one time series (well or field level).
// ---------------------------------------------------------------------------
struct SummaryVector {
  std::string keyword;   // e.g. "WOPR", "FOPR", "TIME"
  std::string wgname;    // well name, or ":+:+:+:+" for field/time entries
  std::string unit;      // e.g. "STB/DAY"
  int num = 0;           // well/region number (0 = none)
  std::vector<double> values;   // one per report step
};

// ---------------------------------------------------------------------------
// The high-level ECLIPSE output writer.
//
// Usage mirrors the run itself: construct once, call grid()/init() at start-up,
// restart_step() at each report step, and summary()/rsm() at the end.
// ---------------------------------------------------------------------------
class EclipseWriter {
 public:
  // `case_name` is the ECLIPSE base name (no extension); files land in `dir`.
  EclipseWriter(const std::string& dir, const std::string& case_name,
                const Deck& d);

  // Override the grid dimensions taken from the deck.
  //
  // A field run's deck does not declare a grid, it names one, so DIMENS there
  // is a placeholder -- 1 x 1 x 1 on Norne.  The EGRID went out saying so
  // while the restart arrays carried 44,431 values, and any reader refuses
  // the pair:
  //
  //   array has 44431 values; grid is 1x1x1 = 1 cells
  //
  // Call before the first snapshot, which is when the EGRID is written.
  void set_dims(int nx, int ny, int nz) {
    nx_ = nx; ny_ = ny; nz_ = nz; ncell_ = nx * ny * nz;
  }

  // Which cells of the box are active, one flag per nx*ny*nz cell.
  //
  // ECLIPSE stores a restart array with one value per ACTIVE cell and lets
  // ACTNUM say where they go.  Writing ACTNUM as all ones therefore promises
  // nx*ny*nz values and delivers 44,431:
  //
  //   array has 44431 values; grid is 46x112x22 = 113344 cells
  //   and ACTNUM accounts for 113344
  //
  // Empty leaves every cell active, which is right for a Cartesian deck.
  void set_actnum(std::vector<int> a) { actnum_ = std::move(a); }

  // True per-ACTIVE-cell pore volume [rb] and centre depth [ft].
  //
  // write_init() computed both from the run deck's DX/DY/DZ, which a
  // field deck does not set: every one of Norne's cells was given the
  // same 50 x 50 x 20 ft box against real volumes spanning 1.07e4 to
  // 4.5e6 ft3, and every depth was k*dz against a reservoir sitting at
  // 8,093 to 10,045 ft.  Anything reading the INIT file then had the
  // wrong pore volume -- benchmark/compare.py put NVRS's water in place
  // at 19.86 million rb against Flow's 417.2 on the same field.
  void set_cell_geometry(std::vector<float> porv, std::vector<float> depth) {
    porv_ = std::move(porv); depth_ = std::move(depth);
  }

  // The deck's porosity, for PORO.  Empty uses whatever write_init is
  // handed, which is the solver's NTG-weighted array.
  void set_cell_porosity(std::vector<float> poro) { poro_ = std::move(poro); }

  // Geometry — CASE.EGRID.  A uniform Cartesian block is emitted as a
  // degenerate corner-point grid, which every reader understands.
  void write_egrid();

  // Static properties — CASE.INIT.  `perm` and `poro` are global fields in
  // natural order (length nx*ny*nz).
  void write_init(const std::vector<float>& perm,
                  const std::vector<float>& permz,
                  const std::vector<float>& poro);

  // One report step appended to CASE.UNRST.  Fields are global natural order.
  void write_restart_step(int seqnum, double t_days,
                          const std::vector<float>& pressure,
                          const std::vector<float>& swat,
                          const std::vector<float>& sgas,
                          const std::vector<float>& soil);

  // Time series — CASE.SMSPEC + CASE.UNSMRY + CASE.RSM.
  void write_summary(const std::vector<double>& times,
                     const std::vector<SummaryVector>& vectors);

 private:
  std::string dir_, case_;
  Deck deck_;
  int nx_, ny_, nz_, ncell_;
  // Per-cell active flags; empty means every cell is active.
  std::vector<int> actnum_;
  // Per-active-cell pore volume and depth; empty falls back to the
  // deck's uniform box.
  std::vector<float> porv_, depth_, poro_;
  bool unrst_started_ = false;

  std::string path(const char* ext) const;
  // INTEHEAD/LOGIHEAD/DOUBHEAD blocks shared by INIT and UNRST.
  std::vector<int> intehead(int nactive) const;
  std::vector<int> logihead() const;
  std::vector<double> doubhead(double t_days) const;
  void write_rsm(const std::vector<double>& times,
                 const std::vector<SummaryVector>& vectors);
};

}  // namespace ecl
}  // namespace nvrs
