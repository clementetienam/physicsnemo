// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// ecl_io.cpp — implementation of the ECLIPSE / OPM Flow compatible writers.
// See ecl_io.hpp for the file format description.
#include "ecl_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace nvrs {
namespace ecl {

namespace {

// ECLIPSE block sizes: numeric arrays are chunked at 1000 elements per Fortran
// record, character arrays at 105.
constexpr int kNumBlock = 1000;
constexpr int kCharBlock = 105;
constexpr int kCharLen = 8;

// Pad or truncate to the fixed 8-character ECLIPSE field width.
std::string pad8(const std::string& s) {
  std::string o = s.substr(0, kCharLen);
  o.resize(kCharLen, ' ');
  return o;
}

}  // namespace

// ===========================================================================
// UnformattedFile
// ===========================================================================
UnformattedFile::UnformattedFile(const std::string& path, bool append) {
  auto mode = std::ios::binary | std::ios::out;
  if (append) mode |= std::ios::app;
  f_.open(path, mode);
}

UnformattedFile::~UnformattedFile() {
  if (f_.is_open()) f_.close();
}

// ECLIPSE files are big-endian regardless of host byte order, so every scalar
// is written byte by byte from the most significant end.
void UnformattedFile::put_i32(int32_t v) {
  uint32_t u = static_cast<uint32_t>(v);
  char b[4] = {char((u >> 24) & 0xFF), char((u >> 16) & 0xFF),
               char((u >> 8) & 0xFF), char(u & 0xFF)};
  f_.write(b, 4);
}

void UnformattedFile::put_f32(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  put_i32(static_cast<int32_t>(u));
}

void UnformattedFile::put_f64(double v) {
  uint64_t u;
  std::memcpy(&u, &v, 8);
  char b[8];
  for (int i = 0; i < 8; ++i) b[i] = char((u >> (56 - 8 * i)) & 0xFF);
  f_.write(b, 8);
}

void UnformattedFile::header(const std::string& kw, int count, const char* type) {
  put_i32(16);
  std::string name = pad8(kw);
  f_.write(name.data(), kCharLen);
  put_i32(count);
  f_.write(type, 4);
  put_i32(16);
}

// Write `v` as a sequence of Fortran records of at most `block` elements each.
template <typename T, typename PutFn>
void UnformattedFile::blocked(const std::vector<T>& v, int block, int elem_bytes,
                              PutFn put) {
  size_t i = 0;
  const size_t n = v.size();
  // A zero-length array still needs no data records at all — ECLIPSE readers
  // rely on `count` in the header, not on a trailing empty record.
  while (i < n) {
    const int m = static_cast<int>(std::min<size_t>(block, n - i));
    put_i32(m * elem_bytes);
    for (int k = 0; k < m; ++k) put(v[i + k]);
    put_i32(m * elem_bytes);
    i += m;
  }
}

void UnformattedFile::inte(const std::string& kw, const std::vector<int>& v) {
  header(kw, (int)v.size(), "INTE");
  blocked(v, kNumBlock, 4, [this](int x) { put_i32(x); });
}

void UnformattedFile::real(const std::string& kw, const std::vector<float>& v) {
  header(kw, (int)v.size(), "REAL");
  blocked(v, kNumBlock, 4, [this](float x) { put_f32(x); });
}

void UnformattedFile::doub(const std::string& kw, const std::vector<double>& v) {
  header(kw, (int)v.size(), "DOUB");
  blocked(v, kNumBlock, 8, [this](double x) { put_f64(x); });
}

void UnformattedFile::logi(const std::string& kw, const std::vector<int>& v) {
  header(kw, (int)v.size(), "LOGI");
  // ECLIPSE logicals are 4-byte: 0 = false, -1 (all bits set) = true.
  blocked(v, kNumBlock, 4, [this](int x) { put_i32(x ? -1 : 0); });
}

void UnformattedFile::chars(const std::string& kw,
                            const std::vector<std::string>& v) {
  header(kw, (int)v.size(), "CHAR");
  blocked(v, kCharBlock, kCharLen, [this](const std::string& s) {
    std::string p = pad8(s);
    f_.write(p.data(), kCharLen);
  });
}

void UnformattedFile::mess(const std::string& kw) {
  header(kw, 0, "MESS");
}

// ===========================================================================
// EclipseWriter
// ===========================================================================
EclipseWriter::EclipseWriter(const std::string& dir, const std::string& case_name,
                             const Deck& d)
    : dir_(dir), case_(case_name), deck_(d),
      nx_(d.nx), ny_(d.ny), nz_(d.nz), ncell_(d.nx * d.ny * d.nz) {
  std::error_code ec;
  std::filesystem::create_directories(dir_, ec);
}

std::string EclipseWriter::path(const char* ext) const {
  return dir_ + "/" + case_ + ext;
}

// ---------------------------------------------------------------------------
// INTEHEAD — the 411-word integer header OPM writes into INIT and UNRST.
// Only the fields readers actually consult are populated; the rest stay zero.
// ---------------------------------------------------------------------------
std::vector<int> EclipseWriter::intehead(int nactive) const {
  std::vector<int> ih(411, 0);
  ih[2] = 2;                       // unit system: 1 METRIC, 2 FIELD, 3 LAB
  ih[8] = nx_;
  ih[9] = ny_;
  ih[10] = nz_;
  ih[11] = nactive;                // number of active cells
  // Phase indicator: 1 oil, 2 water, 3 oil/water, 4 gas, 5 oil/gas,
  // 6 gas/water, 7 oil/water/gas.
  ih[14] = (deck_.model == Model::BlackOil2P) ? 3 : 7;
  ih[64] = 1;                      // start day
  ih[65] = 1;                      // start month
  ih[66] = 2026;                   // start year
  ih[94] = 100;                    // simulator identifier (100 = ECLIPSE 100)
  return ih;
}

std::vector<int> EclipseWriter::logihead() const {
  std::vector<int> lh(121, 0);
  lh[0] = 0;                       // not a radial grid
  lh[3] = (deck_.model != Model::BlackOil2P) ? 1 : 0;   // dissolved gas present
  return lh;
}

std::vector<double> EclipseWriter::doubhead(double t_days) const {
  std::vector<double> dh(229, 0.0);
  dh[0] = t_days;                  // simulation time [days]
  return dh;
}

// ---------------------------------------------------------------------------
// CASE.EGRID
// ---------------------------------------------------------------------------
void EclipseWriter::write_egrid() {
  UnformattedFile f(path(".EGRID"));
  if (!f.good()) return;

  std::vector<int> filehead(100, 0);
  filehead[0] = 3;                 // file format version
  filehead[1] = 2007;              // release year
  filehead[4] = 0;                 // grid type: 0 = corner point
  f.inte("FILEHEAD", filehead);

  // Field units throughout (the decks are psia / ft / mD / STB).
  f.chars("GRIDUNIT", {"FEET", ""});

  std::vector<int> gridhead(100, 0);
  gridhead[0] = 1;                 // corner-point grid
  gridhead[1] = nx_;
  gridhead[2] = ny_;
  gridhead[3] = nz_;
  gridhead[24] = 1;                // number of reservoirs
  f.inte("GRIDHEAD", gridhead);

  const double dx = deck_.dx, dy = deck_.dy, dz = deck_.dz;
  const double top = 0.0;          // datum depth of the top surface [ft]
  const double bot = top + nz_ * dz;

  // COORD: one pillar per (i,j) node of the (nx+1) x (ny+1) lattice, each
  // pillar a straight line given by its top and bottom (x,y,z).
  std::vector<float> coord;
  coord.reserve(6 * (nx_ + 1) * (ny_ + 1));
  for (int j = 0; j <= ny_; ++j)
    for (int i = 0; i <= nx_; ++i) {
      const float x = (float)(i * dx), y = (float)(j * dy);
      coord.push_back(x); coord.push_back(y); coord.push_back((float)top);
      coord.push_back(x); coord.push_back(y); coord.push_back((float)bot);
    }
  f.real("COORD", coord);

  // ZCORN: eight corner depths per cell, ordered
  //   k -> {top, bottom} -> j -> {front, back} -> i -> {left, right}
  // giving 2nx * 2ny * 2nz entries.
  std::vector<float> zcorn((size_t)8 * ncell_, 0.0f);
  auto zidx = [&](int i, int xs, int j, int ys, int k, int zs) -> size_t {
    return (((size_t)(2 * k + zs) * (2 * ny_) + (2 * j + ys)) * (2 * nx_)) +
           (2 * i + xs);
  };
  for (int k = 0; k < nz_; ++k) {
    const float ztop = (float)(top + k * dz), zbot = (float)(top + (k + 1) * dz);
    for (int j = 0; j < ny_; ++j)
      for (int i = 0; i < nx_; ++i)
        for (int ys = 0; ys < 2; ++ys)
          for (int xs = 0; xs < 2; ++xs) {
            zcorn[zidx(i, xs, j, ys, k, 0)] = ztop;
            zcorn[zidx(i, xs, j, ys, k, 1)] = zbot;
          }
  }
  f.real("ZCORN", zcorn);

  // Every cell is active in this simulator (no inactive/pinched-out cells).
  f.inte("ACTNUM", actnum_.size() == (size_t)ncell_
                       ? actnum_
                       : std::vector<int>(ncell_, 1));
  f.inte("ENDGRID", {});
}

// ---------------------------------------------------------------------------
// CASE.INIT
// ---------------------------------------------------------------------------
void EclipseWriter::write_init(const std::vector<float>& perm,
                               const std::vector<float>& permz,
                               const std::vector<float>& poro) {
  UnformattedFile f(path(".INIT"));
  if (!f.good()) return;

  f.inte("INTEHEAD", intehead(ncell_));
  f.logi("LOGIHEAD", logihead());
  f.doub("DOUBHEAD", doubhead(0.0));

  const double dx = deck_.dx, dy = deck_.dy, dz = deck_.dz;
  const double bulk = dx * dy * dz;

  // How many values each array carries.
  //
  // ECLIPSE stores one per ACTIVE cell and lets ACTNUM place them; the arrays
  // handed in are active-ordered too.  Writing ncell_ of them while `perm`
  // holds 44,431 meant every cell past the 44,431st silently took perm_min
  // and poro_min -- two thirds of the field, on the file every consumer reads
  // the rock from.
  const int nout = actnum_.empty()
                       ? ncell_
                       : (int)std::count(actnum_.begin(), actnum_.end(), 1);
  const bool geom = ((int)porv_.size() == nout && (int)depth_.size() == nout);

  std::vector<float> porv(nout, 0.0f), px(nout), py(nout), pz(nout);
  std::vector<float> vdx(nout, (float)dx), vdy(nout, (float)dy),
      vdz(nout, (float)dz), tops(nout), depth(nout);
  for (int c = 0; c < nout; ++c) {
    const float k = (c < (int)perm.size()) ? perm[c] : (float)deck_.perm_min;
    const float p = (c < (int)poro.size()) ? poro[c] : (float)deck_.poro_min;
    // ECLIPSE stores PORV in reservoir barrels in FIELD units, while
    // bulk is ft^3.  Writing ft^3 here made every consumer of the INIT
    // file read a pore volume 5.6146x too large -- benchmark/compare.py
    // reported 7.842 Mbbl of water in place at t = 0 against Flow's
    // 1.397 Mbbl on an identical grid, which is that factor exactly.
    // The grid's own pore volume and depth where the caller supplied them,
    // and the deck's uniform box only where it did not.
    porv[c] = geom ? porv_[c] : (float)(bulk * p / units::kFt3PerBbl);
    px[c] = k;
    py[c] = k;                                    // Ky = Kx in this model
    // A corner-point deck can define PERMZ independently and edit it by layer
    // with COPY/MULTIPLY.  The solver uses that resolved array, so INIT must
    // report the same rock rather than reconstructing Kz from a scalar.
    pz[c] = (c < (int)permz.size()) ? permz[c]
                                    : (float)(deck_.kz_mult * k);
    if (geom) {
      depth[c] = depth_[c];
      tops[c] = depth_[c];        // no layer thickness to subtract here
    } else {
      const int kk = c / (nx_ * ny_);
      tops[c] = (float)(kk * dz);
      depth[c] = (float)((kk + 0.5) * dz);
    }
  }

  f.real("PORV", porv);
  f.real("PORO", ((int)poro_.size() == nout)
                     ? poro_
                     : (((int)poro.size() == nout)
                            ? poro
                            : std::vector<float>(nout,
                                                 (float)deck_.poro_min)));
  f.real("PERMX", px);
  f.real("PERMY", py);
  f.real("PERMZ", pz);
  f.real("DX", vdx);
  f.real("DY", vdy);
  f.real("DZ", vdz);
  f.real("TOPS", tops);
  f.real("DEPTH", depth);
  f.inte("ACTNUM", actnum_.size() == (size_t)ncell_
                       ? actnum_
                       : std::vector<int>(ncell_, 1));
  // Region arrays: a single region each, which is what a deck without
  // FIPNUM/SATNUM/PVTNUM/EQLNUM implies.
  f.inte("FIPNUM", std::vector<int>(nout, 1));
  f.inte("SATNUM", std::vector<int>(nout, 1));
  f.inte("PVTNUM", std::vector<int>(nout, 1));
  f.inte("EQLNUM", std::vector<int>(nout, 1));
}

// ---------------------------------------------------------------------------
// CASE.UNRST — unified restart, one solution section per report step.
// ---------------------------------------------------------------------------
void EclipseWriter::write_restart_step(int seqnum, double t_days,
                                       const std::vector<float>& pressure,
                                       const std::vector<float>& swat,
                                       const std::vector<float>& sgas,
                                       const std::vector<float>& soil) {
  // The first step truncates, subsequent steps append — that is what makes it
  // a *unified* restart rather than one file per step.
  UnformattedFile f(path(".UNRST"), unrst_started_);
  if (!f.good()) return;
  unrst_started_ = true;

  f.inte("SEQNUM", {seqnum});
  f.inte("INTEHEAD", intehead(ncell_));
  f.logi("LOGIHEAD", logihead());
  f.doub("DOUBHEAD", doubhead(t_days));
  f.mess("STARTSOL");
  f.real("PRESSURE", pressure);
  f.real("SWAT", swat);
  if (deck_.model != Model::BlackOil2P) f.real("SGAS", sgas);
  f.real("SOIL", soil);
  f.mess("ENDSOL");
}

// ---------------------------------------------------------------------------
// CASE.SMSPEC + CASE.UNSMRY + CASE.RSM
// ---------------------------------------------------------------------------
void EclipseWriter::write_summary(const std::vector<double>& times,
                                  const std::vector<SummaryVector>& vectors) {
  // --- SMSPEC: what each column of the summary means -----------------------
  {
    UnformattedFile f(path(".SMSPEC"));
    if (!f.good()) return;

    const int nlist = (int)vectors.size();
    f.inte("INTEHEAD", {2, 100});          // FIELD units, ECLIPSE 100 flavour
    f.chars("RESTART", std::vector<std::string>(9, ""));
    f.inte("DIMENS", {nlist, nx_, ny_, nz_, 0, -1});

    std::vector<std::string> kws, wgn, uns;
    std::vector<int> nums;
    kws.reserve(nlist); wgn.reserve(nlist); uns.reserve(nlist); nums.reserve(nlist);
    for (const auto& v : vectors) {
      kws.push_back(v.keyword);
      wgn.push_back(v.wgname);
      uns.push_back(v.unit);
      nums.push_back(v.num);
    }
    f.chars("KEYWORDS", kws);
    f.chars("WGNAMES", wgn);
    f.inte("NUMS", nums);
    f.chars("UNITS", uns);
    // Start date: day, month, year, hour, minute, microsecond.
    f.inte("STARTDAT", {1, 1, 2026, 0, 0, 0});
  }

  // --- UNSMRY: the values themselves ---------------------------------------
  {
    UnformattedFile f(path(".UNSMRY"));
    if (!f.good()) return;
    f.inte("SEQHDR", {0});
    const size_t nsteps = times.size();
    for (size_t s = 0; s < nsteps; ++s) {
      f.inte("MINISTEP", {(int)s});
      std::vector<float> params;
      params.reserve(vectors.size());
      for (const auto& v : vectors)
        params.push_back(s < v.values.size() ? (float)v.values[s] : 0.0f);
      f.real("PARAMS", params);
    }
  }

  write_rsm(times, vectors);
}

// ---------------------------------------------------------------------------
// CASE.RSM — the ASCII summary ECLIPSE prints.
//
// Layout: pages of at most 10 data columns.  Each page opens with the ASCII
// form-feed page marker '1', a run-identification block, then four header lines
// (keyword / unit / blank / well name) followed by the rows.
// ---------------------------------------------------------------------------
void EclipseWriter::write_rsm(const std::vector<double>& times,
                              const std::vector<SummaryVector>& vectors) {
  std::ofstream o(path(".RSM"));
  if (!o) return;

  // Column 0 is always TIME; the remainder are paginated 10 at a time.
  std::vector<const SummaryVector*> data;
  const SummaryVector* timevec = nullptr;
  for (const auto& v : vectors) {
    if (v.keyword == "TIME" && !timevec) timevec = &v;
    else data.push_back(&v);
  }

  const size_t nsteps = times.size();
  const size_t kPerPage = 10;
  const int w = 14;                       // column width

  for (size_t page = 0; page < data.size(); page += kPerPage) {
    const size_t last = std::min(page + kPerPage, data.size());

    o << "1\n";                            // ECLIPSE page break
    o << " " << std::string(131, '-') << "\n";
    o << " SUMMARY OF RUN " << case_ << "\n";
    o << " " << deck_.title << "\n";
    o << " " << std::string(131, '-') << "\n";

    // Line 1: mnemonic.  Line 2: unit.  Line 3: (blank / number).
    // Line 4: well or group name.
    o << " " << std::setw(w) << std::left << "TIME";
    for (size_t k = page; k < last; ++k)
      o << std::setw(w) << std::left << data[k]->keyword;
    o << "\n";

    o << " " << std::setw(w) << std::left << "DAYS";
    for (size_t k = page; k < last; ++k)
      o << std::setw(w) << std::left << data[k]->unit;
    o << "\n";

    o << " " << std::setw(w) << std::left << "";
    for (size_t k = page; k < last; ++k) {
      std::string n = data[k]->num ? std::to_string(data[k]->num) : "";
      o << std::setw(w) << std::left << n;
    }
    o << "\n";

    o << " " << std::setw(w) << std::left << "";
    for (size_t k = page; k < last; ++k) {
      std::string wg = data[k]->wgname;
      if (wg == ":+:+:+:+") wg = "";      // field-level vectors have no name
      o << std::setw(w) << std::left << wg;
    }
    o << "\n";

    o << " " << std::string(131, '-') << "\n";

    o << std::scientific << std::setprecision(5);
    for (size_t s = 0; s < nsteps; ++s) {
      o << " " << std::setw(w) << std::left << times[s];
      for (size_t k = page; k < last; ++k) {
        const double val =
            (s < data[k]->values.size()) ? data[k]->values[s] : 0.0;
        o << std::setw(w) << std::left << val;
      }
      o << "\n";
    }
    o << "\n";
  }
}

}  // namespace ecl
}  // namespace nvrs
