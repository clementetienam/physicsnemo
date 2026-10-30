// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// cpgrid.cpp -- corner-point geometry, fault juxtaposition, transmissibility.
//
// The algorithm, in the order the code runs it
// --------------------------------------------
// 1. Parse.  GRDECL is a flat token stream: keywords, numbers, `N*value`
//    repeat counts, `--` comments and `/` terminators.  Array keywords go into
//    a named store; the deck operators (COPY, EQUALS, MULTIPLY, ADD, over an
//    optional BOX) are applied in the order they appear, because that order is
//    the deck's meaning -- Norne's vertical permeability, for instance, exists
//    only as PERMX copied and then scaled differently in each of 22 layers.
//
// 2. Corners.  Each cell is a hexahedron whose eight corners lie on four
//    pillars.  COORD gives the pillars as a top and a bottom point; ZCORN
//    gives, for every cell, the depth of each of its eight corners.  A corner's
//    (x,y) follows from linear interpolation along its pillar at that depth.
//
// 3. Volume and centroid.  The six faces of a corner-point cell are in general
//    not planar, so the cell is decomposed into tetrahedra: an interior point,
//    a face centre, and each edge of that face.  Twenty-four tetrahedra per
//    cell give the volume and the centroid exactly for that decomposition,
//    which is the same one the face areas use, so the two stay consistent.
//
//    Eclipse transmissibilities use a second, deliberately simpler centre:
//    the arithmetic mean of the eight input corners.  This distinction is
//    important for wedge cells, so the TPFA path computes that centre locally.
//
// 4. Half-transmissibilities.  For a cell c and one of its faces,
//
//        T_c = C * (A . K . D) / (D . D)
//
//    with A the outward area vector of the face, D the vector from the Eclipse
//    input-cell centre to the face centre, and K the diagonal permeability
//    component belonging to that logical face.  This is the Eclipse/OPM
//    two-point form; it degenerates to k*A/d on a Cartesian cell.
//
// 5. Juxtaposition.  This is where faults are handled, and the reason no
//    special case for them appears anywhere else.  A lateral connection is not
//    assumed to join equal layers.  For each pair of adjacent columns the code
//    walks both columns and connects every pair of cells whose faces overlap
//    in depth, weighting each half-transmissibility by the fraction of its own
//    face that the overlap covers.  On an unfaulted grid every cell overlaps
//    exactly its own layer and the result is the ordinary 7-point stencil.
//    Where a fault throws one side down, the same loop produces the several
//    partial connections that the throw actually creates -- which is what a
//    fault *is*, geometrically.  MULTFLT then scales those faces.
//
// 6. Combination.  The two half-transmissibilities are combined in series, the
//    face multipliers applied, and finally MULTREGT for connections that cross
//    a region boundary.
#include "cpgrid.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nvrs {
namespace {

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
// GRDECL files reach hundreds of megabytes, so the whole file is read once and
// scanned in place rather than streamed through iostreams.
struct Tokenizer {
  std::string buf;
  size_t p = 0;

  explicit Tokenizer(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    buf.resize((size_t)n);
    f.read(&buf[0], n);
    buf.resize((size_t)f.gcount());
  }

  void skip() {
    while (p < buf.size()) {
      char c = buf[p];
      if (c == '-' && p + 1 < buf.size() && buf[p + 1] == '-') {
        while (p < buf.size() && buf[p] != '\n') ++p;
      } else if (std::isspace((unsigned char)c)) {
        ++p;
      } else {
        return;
      }
    }
  }

  bool eof() { skip(); return p >= buf.size(); }
  bool at_slash() { skip(); return p < buf.size() && buf[p] == '/'; }

  // Step over a record terminator and the free text that may follow it.
  //
  // A deck is allowed to annotate a record after its closing slash --
  //   'PERMZ' 0.64  1 46 1 112 13 13 /  Tofte 2.1.3
  // is one of Norne's -- and everything from the slash to the end of that line
  // is a comment.  Treating it as data is how a reader ends up trying to parse
  // a formation name as a number, or worse, silently swallowing the next
  // record while looking for a terminator it already passed.
  void consume_slash() {
    ++p;
    while (p < buf.size() && buf[p] != '\n') ++p;
  }

  void eat_slash() { if (at_slash()) consume_slash(); }

  // One whitespace-delimited token.  Quoted strings come back without quotes.
  std::string next() {
    skip();
    if (p >= buf.size()) return {};
    if (buf[p] == '\'') {
      size_t s = ++p;
      while (p < buf.size() && buf[p] != '\'') ++p;
      std::string t = buf.substr(s, p - s);
      if (p < buf.size()) ++p;
      return t;
    }
    if (buf[p] == '/') { consume_slash(); return "/"; }
    size_t s = p;
    while (p < buf.size() && !std::isspace((unsigned char)buf[p]) && buf[p] != '/')
      ++p;
    if (p == s) { ++p; return next(); }
    return buf.substr(s, p - s);
  }

  // Read values until the terminating `/`, expanding `N*value` repeats.
  void read_values(std::vector<Real>& out, size_t expect = 0) {
    if (expect) out.reserve(expect);
    for (;;) {
      skip();
      if (p >= buf.size()) return;
      if (buf[p] == '/') { consume_slash(); return; }
      size_t s = p;
      while (p < buf.size() && !std::isspace((unsigned char)buf[p]) && buf[p] != '/')
        ++p;
      const char* tok = buf.c_str() + s;
      size_t len = p - s;
      if (!len) continue;
      // `N*value`, and the bare `N*` form meaning N defaulted entries.
      const char* star = (const char*)std::memchr(tok, '*', len);
      if (star) {
        long rep = std::strtol(tok, nullptr, 10);
        Real v = (star + 1 < tok + len) ? std::strtod(star + 1, nullptr) : 0.0;
        if (rep > 0) out.insert(out.end(), (size_t)rep, v);
      } else {
        // A trailing flag such as SPECGRID's 'F' is not a number; skip it.
        if (!(std::isdigit((unsigned char)tok[0]) || tok[0] == '-' ||
              tok[0] == '+' || tok[0] == '.'))
          continue;
        out.push_back(std::strtod(tok, nullptr));
      }
    }
  }

  void skip_to_slash() {
    for (;;) {
      skip();
      if (p >= buf.size()) return;
      if (buf[p] == '/') { consume_slash(); return; }
      next();
    }
  }

  // TITLE is the sole common Eclipse free-text record terminated by a line
  // ending rather than by '/'.  Its text may follow TITLE on the same line or
  // occupy the next line.  Skipping it as an ordinary unknown keyword would
  // consume every following keyword through the next slash.
  void skip_title() {
    bool text_on_keyword_line = false;
    while (p < buf.size() && buf[p] != '\n') {
      if (!std::isspace((unsigned char)buf[p]))
        text_on_keyword_line = true;
      ++p;
    }
    if (!text_on_keyword_line && p < buf.size()) {
      ++p;
      while (p < buf.size() && buf[p] != '\n') ++p;
    }
  }
};

inline bool is_keyword(const std::string& t) {
  if (t.empty() || t.size() > 8) return false;
  if (!std::isalpha((unsigned char)t[0])) return false;
  for (char c : t)
    if (!std::isalnum((unsigned char)c) && c != '_' && c != '-' && c != '+')
      return false;
  return true;
}

// ---------------------------------------------------------------------------
// Small vector helpers
// ---------------------------------------------------------------------------
struct V3 { Real x = 0, y = 0, z = 0; };
inline V3 operator-(const V3& a, const V3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator+(const V3& a, const V3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator*(const V3& a, Real s) { return {a.x * s, a.y * s, a.z * s}; }
inline Real dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// The four corners of each face, in the corner numbering documented in the
// header (0-3 top, 4-7 bottom; within a level i fastest, then j).
// Order within a face is a closed loop, so consecutive pairs are edges.
const int kFaceCorner[6][4] = {
    {0, 2, 6, 4},   // I-
    {1, 5, 7, 3},   // I+
    {0, 4, 5, 1},   // J-
    {2, 3, 7, 6},   // J+
    {0, 1, 3, 2},   // K- (top, shallower)
    {4, 6, 7, 5},   // K+ (bottom)
};

// Sections after which the grid is fully described.  Their keywords are
// record-structured in ways this reader does not model, so it stops rather
// than risk misreading them as grid data.
bool is_terminating_section(const std::string& kw) {
  return kw == "PROPS" || kw == "REGIONS" || kw == "SOLUTION" ||
         kw == "SUMMARY" || kw == "SCHEDULE";
}

}  // namespace

// ---------------------------------------------------------------------------
// The array store
// ---------------------------------------------------------------------------
bool CornerPointGrid::is_array_keyword(const std::string& n) {
  static const std::set<std::string> k = {
      "PORO",   "PERMX",  "PERMY",  "PERMZ",  "NTG",    "ACTNUM",
      "MULTX",  "MULTY",  "MULTZ",  "MULTPV", "MULTX-", "MULTY-", "MULTZ-",
      "TOPS",   "DX",     "DY",     "DZ",
      "SATNUM", "EQLNUM", "FIPNUM", "PVTNUM", "FLUXNUM", "MULTNUM",
      "ROCKNUM", "IMBNUM", "ENDNUM",
      "SWL",    "SWU",    "SWCR",   "SGL",    "SGU",    "SGCR",
      "SOWCR",  "SOGCR",  "ISGCR",  "ISWCR",  "ISGU",   "ISWL",
      "ISWU",   "ISGL",   "ISOWCR", "ISOGCR", "IKRW",   "IKRO",
      "IKRG",
      "KRW",    "KRO",    "KRG",    "PCW",    "PCG",    "SWATINIT",
  };
  return k.count(n) > 0;
}

Real CornerPointGrid::default_value(const std::string& n) {
  if (n == "NTG" || n == "ACTNUM" || n.rfind("MULT", 0) == 0) return 1.0;
  if (n == "SATNUM" || n == "EQLNUM" || n == "FIPNUM" || n == "PVTNUM" ||
      n == "FLUXNUM" || n == "MULTNUM" || n == "ROCKNUM" || n == "IMBNUM" ||
      n == "ENDNUM")
    return 1.0;
  return 0.0;
}

std::vector<Real>& CornerPointGrid::array_for_write(const std::string& name) {
  auto it = arrays_.find(name);
  if (it != arrays_.end()) return it->second;
  if (nx <= 0)
    throw std::runtime_error(
        "grid dimensions (SPECGRID/DIMENS) must appear before " + name);
  array_defined_[name] = std::vector<char>((size_t)n_global(), 0);
  return arrays_
      .emplace(name, std::vector<Real>((size_t)n_global(), default_value(name)))
      .first->second;
}

const std::vector<Real>* CornerPointGrid::array(const std::string& name) const {
  auto it = arrays_.find(name);
  return (it == arrays_.end()) ? nullptr : &it->second;
}

std::vector<Real> CornerPointGrid::active_array(const std::string& name) const {
  const std::vector<Real>* g = array(name);
  if (!g) return {};
  std::vector<Real> out(active_to_global_.size());
  for (size_t a = 0; a < active_to_global_.size(); ++a)
    out[a] = (*g)[(size_t)active_to_global_[a]];
  return out;
}

std::vector<char> CornerPointGrid::active_array_defined(
    const std::string& name) const {
  const auto it = array_defined_.find(name);
  if (it == array_defined_.end()) return {};
  const std::vector<char>& g = it->second;
  std::vector<char> out(active_to_global_.size(), 0);
  for (size_t a = 0; a < active_to_global_.size(); ++a) {
    const size_t gi = (size_t)active_to_global_[a];
    if (gi < g.size()) out[a] = g[gi];
  }
  return out;
}

void CornerPointGrid::set_dims(int x, int y, int z) {
  nx = x; ny = y; nz = z;
  reset_box();
}

// ---------------------------------------------------------------------------
// Corners of one cell
// ---------------------------------------------------------------------------
void CornerPointGrid::cell_corners(int i, int j, int k, Real xyz[8][3]) const {
  const int nxc = 2 * nx, nyc = 2 * ny;
  for (int c = 0; c < 8; ++c) {
    const int a = c & 1;                 // along I
    const int b = (c >> 1) & 1;          // along J
    const int cc = (c >> 2) & 1;         // along K

    const size_t zi = ((size_t)(2 * k + cc) * nyc + (2 * j + b)) * nxc + (2 * i + a);
    const Real z = zcorn_[zi];

    const size_t pi = (size_t)((j + b) * (nx + 1) + (i + a)) * 6;
    const Real xt = coord_[pi + 0], yt = coord_[pi + 1], zt = coord_[pi + 2];
    const Real xb = coord_[pi + 3], yb = coord_[pi + 4], zb = coord_[pi + 5];

    // Interpolate along the pillar.  A degenerate pillar (zt == zb, which some
    // decks emit for collapsed columns) falls back to the top point.
    const Real dzp = zb - zt;
    const Real t = (std::fabs(dzp) > 1e-12) ? (z - zt) / dzp : 0.0;
    xyz[c][0] = xt + t * (xb - xt);
    xyz[c][1] = yt + t * (yb - yt);
    xyz[c][2] = z;
  }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------
void CornerPointGrid::read_grdecl(const std::string& path) {
  Tokenizer tk(path);

  const std::string dir = [&] {
    size_t s = path.find_last_of("/\\");
    return (s == std::string::npos) ? std::string(".") : path.substr(0, s);
  }();

  auto ensure_sized = [&](const std::string& kw) {
    if (nx <= 0)
      throw std::runtime_error(
          "grid dimensions (SPECGRID/DIMENS) must appear before " + kw);
  };

  // A box specification: six integers, each possibly defaulted with `1*`.
  // Returns false if the record held no box, in which case the current BOX
  // applies -- which is how a bare `MULTIPLY 'PERMZ' 0.2 /` is meant to read.
  auto read_box = [&](int& i1, int& i2, int& j1, int& j2, int& k1, int& k2) {
    std::vector<Real> v;
    tk.read_values(v);
    if (v.size() >= 6) {
      i1 = (int)v[0] - 1; i2 = (int)v[1] - 1;
      j1 = (int)v[2] - 1; j2 = (int)v[3] - 1;
      k1 = (int)v[4] - 1; k2 = (int)v[5] - 1;
      return true;
    }
    i1 = bi1_; i2 = bi2_; j1 = bj1_; j2 = bj2_; k1 = bk1_; k2 = bk2_;
    return false;
  };

  // Apply `op` over a box of an array.
  auto over_box = [&](auto& dst, int i1, int i2, int j1, int j2,
                      int k1, int k2, const auto& op) {
    i1 = std::max(i1, 0); j1 = std::max(j1, 0); k1 = std::max(k1, 0);
    i2 = std::min(i2, nx - 1); j2 = std::min(j2, ny - 1); k2 = std::min(k2, nz - 1);
    for (int k = k1; k <= k2; ++k)
      for (int j = j1; j <= j2; ++j)
        for (int i = i1; i <= i2; ++i)
          op(dst[(size_t)global_index(i, j, k)]);
  };

  while (!tk.eof()) {
    std::string kw = tk.next();
    if (kw.empty()) break;
    if (!is_keyword(kw)) continue;
    if (is_terminating_section(kw)) {
      // SUMMARY and SCHEDULE hold nothing the grid wants.
      if (kw == "SUMMARY" || kw == "SCHEDULE") break;
      if (!read_props_arrays) break;
      // PROPS, REGIONS and SOLUTION do.  This reader used to stop here, and
      // the cost was silent: every per-cell array those sections define was
      // dropped without a word.  ENDSCALE's SWL/SWCR/SWU/SOWCR and the rest
      // are PROPS keywords, and PVTNUM/SATNUM/EQLNUM/FIPNUM are REGIONS
      // keywords, so none of them ever reached the grid -- which is why
      // PVTNUM reported two regions with every cell in region one.
      //
      // Carry on, but in a restricted mode: read the arrays and the box
      // operators, and touch nothing else.  These sections are full of
      // record-structured keywords this reader does not model, which is what
      // it was avoiding, so the rule below is that anything unrecognised is
      // stepped over rather than interpreted.
      in_tail_ = true;
      continue;
    }
    // Restricted mode.  Deliberately NOT skip_to_slash for the unknown case:
    // a saturation table's rows are numbers, is_keyword() rejects them, and
    // letting them fall through is safer than guessing where a multi-record
    // keyword ends.
    if (in_tail_ && !(is_array_keyword(kw) || kw == "BOX" || kw == "ENDBOX" ||
                      kw == "COPY" || kw == "EQUALS" || kw == "MULTIPLY" ||
                      kw == "ADD" || kw == "INCLUDE" || kw == "SCALECRS" ||
                      kw == "THPRES"))
      continue;

    if (kw == "DXV" || kw == "DYV" || kw == "DZV") {
      // One value per index along that axis, so nx, ny or nz of them.  A
      // block-centred deck may give these instead of the per-cell DX/DY/DZ.
      ensure_sized(kw);
      std::vector<Real>& v = (kw == "DXV") ? dxv_ : (kw == "DYV") ? dyv_ : dzv_;
      const size_t want = (kw == "DXV") ? (size_t)nx
                        : (kw == "DYV") ? (size_t)ny : (size_t)nz;
      v.clear();
      tk.read_values(v, want);
      if (v.size() != want)
        throw std::runtime_error(kw + ": expected " + std::to_string(want) +
                                 " values, got " + std::to_string(v.size()));

    } else if (kw == "SPECGRID" || kw == "DIMENS") {
      std::vector<Real> v;
      tk.read_values(v);
      if (v.size() < 3) throw std::runtime_error(kw + ": need three dimensions");
      set_dims((int)v[0], (int)v[1], (int)v[2]);

    } else if (kw == "COORD") {
      ensure_sized(kw);
      coord_.clear();
      tk.read_values(coord_, (size_t)6 * (nx + 1) * (ny + 1));
      if (coord_.size() != (size_t)6 * (nx + 1) * (ny + 1))
        throw std::runtime_error("COORD: wrong number of values");

    } else if (kw == "ZCORN") {
      ensure_sized(kw);
      zcorn_.clear();
      tk.read_values(zcorn_, (size_t)8 * nx * ny * nz);
      if (zcorn_.size() != (size_t)8 * nx * ny * nz)
        throw std::runtime_error("ZCORN: wrong number of values");

    } else if (is_array_keyword(kw)) {
      ensure_sized(kw);
      std::vector<Real> v;
      tk.read_values(v, (size_t)n_global());
      // TOPS is the one array a deck routinely gives for the top layer alone.
      // That is not a short array by mistake: a block-centred deck states the
      // depth of the first layer and lets the rest follow from the
      // thicknesses, so nx*ny values is the normal spelling and nx*ny*nz the
      // exception.  Both are kept as given; synthesise_block_centred() reads
      // whichever it finds.
      if (kw == "TOPS" && v.size() == (size_t)nx * ny && nz > 1) {
        arrays_[kw] = std::move(v);
        array_defined_[kw] = std::vector<char>(arrays_[kw].size(), 1);
        continue;
      }
      if (v.size() != (size_t)n_global()) {
        // In GRID a short array is a corrupt deck and must stop the run.
        // Past PROPS it far more likely means the name was an ITEM inside
        // another keyword's record -- RPTPROPS and RPTSOL list array names
        // as arguments -- so drop what was read and carry on rather than end
        // the run over a word that was never an array.
        if (in_tail_) { ignored_.push_back(kw); continue; }
        std::ostringstream os;
        os << kw << ": expected " << n_global() << " values, read " << v.size();
        throw std::runtime_error(os.str());
      }
      arrays_[kw] = std::move(v);
      array_defined_[kw] = std::vector<char>((size_t)n_global(), 1);

    } else if (kw == "SCALECRS") {
      // One item, YES or NO.  It decides whether the displacing phase's
      // relative permeability is scaled between two endpoints or three, and
      // the two put the front in different places -- so read it rather than
      // assume the default.
      const std::string v = tk.next();
      scalecrs = (!v.empty() && (v[0] == 'Y' || v[0] == 'y'));
      tk.skip_to_slash();

    } else if (kw == "BOX") {
      ensure_sized(kw);
      std::vector<Real> v;
      tk.read_values(v);
      if (v.size() >= 6) {
        bi1_ = (int)v[0] - 1; bi2_ = (int)v[1] - 1;
        bj1_ = (int)v[2] - 1; bj2_ = (int)v[3] - 1;
        bk1_ = (int)v[4] - 1; bk2_ = (int)v[5] - 1;
      }

    } else if (kw == "ENDBOX") {
      reset_box();

    } else if (kw == "COPY") {
      ensure_sized(kw);
      while (!tk.at_slash() && !tk.eof()) {
        const std::string src = tk.next();
        if (src.empty() || src == "/") break;
        const std::string dst = tk.next();
        int i1, i2, j1, j2, k1, k2;
        read_box(i1, i2, j1, j2, k1, k2);
        const std::vector<Real>* s = array(src);
        if (!s) { ignored_.push_back("COPY from undefined " + src); continue; }
        const std::vector<Real> scopy = *s;   // src and dst may be the same
        const auto sm_it = array_defined_.find(src);
        std::vector<Real>& d = array_for_write(dst);
        std::vector<char>& dm = array_defined_[dst];
        size_t idx = 0;
        (void)idx;
        for (int k = std::max(k1, 0); k <= std::min(k2, nz - 1); ++k)
          for (int j = std::max(j1, 0); j <= std::min(j2, ny - 1); ++j)
            for (int i = std::max(i1, 0); i <= std::min(i2, nx - 1); ++i) {
              const size_t g = (size_t)global_index(i, j, k);
              d[g] = scopy[g];
              dm[g] = (sm_it == array_defined_.end() ||
                       g >= sm_it->second.size()) ? 1 : sm_it->second[g];
            }
      }
      tk.eat_slash();

    } else if (kw == "EQUALS" || kw == "MULTIPLY" || kw == "ADD") {
      ensure_sized(kw);
      while (!tk.at_slash() && !tk.eof()) {
        const std::string name = tk.next();
        if (name.empty() || name == "/") break;
        std::vector<Real> v;
        tk.read_values(v);
        if (v.empty()) continue;
        const Real val = v[0];
        int i1 = bi1_, i2 = bi2_, j1 = bj1_, j2 = bj2_, k1 = bk1_, k2 = bk2_;
        if (v.size() >= 7) {
          i1 = (int)v[1] - 1; i2 = (int)v[2] - 1;
          j1 = (int)v[3] - 1; j2 = (int)v[4] - 1;
          k1 = (int)v[5] - 1; k2 = (int)v[6] - 1;
        }
        std::vector<Real>& d = array_for_write(name);
        std::vector<char>& dm = array_defined_[name];
        if (kw == "EQUALS")
          over_box(d, i1, i2, j1, j2, k1, k2, [&](Real& x) { x = val; });
        else if (kw == "MULTIPLY")
          over_box(d, i1, i2, j1, j2, k1, k2, [&](Real& x) { x *= val; });
        else
          over_box(d, i1, i2, j1, j2, k1, k2, [&](Real& x) { x += val; });
        over_box(dm, i1, i2, j1, j2, k1, k2, [](char& x) { x = 1; });
      }
      tk.eat_slash();

    } else if (kw == "MINPV" || kw == "MINPVV") {
      std::vector<Real> v;
      tk.read_values(v);
      if (!v.empty()) min_pv_ = v[0];

    } else if (kw == "PINCH") {
      // threshold, GAP/NOGAP, empty-gap, TOPBOT/ALL, TOP/ALL.  The present
      // TPFA builder uses the TOPBOT calculation and the active bounding-cell
      // multipliers, which are the defaults and the options used by Norne.
      // Parse all items nevertheless so that the two decisions governing NNC
      // creation are not silently replaced by guesses.
      pinch_active_ = true;
      const std::string thick = tk.next();
      if (!thick.empty() && thick != "/" && thick.find('*') == std::string::npos)
        pinch_thickness_ = std::strtod(thick.c_str(), nullptr);
      if (!tk.at_slash()) {
        const std::string opt = tk.next();
        if (!opt.empty() && opt.find('*') == std::string::npos)
          pinch_gap_ = (opt != "NOGAP");
      }
      tk.skip_to_slash();

    } else if (kw == "FAULTS") {
      // Records of  'name' i1 i2 j1 j2 k1 k2 'face' /  until a bare `/`.
      for (;;) {
        if (tk.at_slash()) { tk.eat_slash(); break; }
        if (tk.eof()) break;
        std::string name = tk.next();
        if (name.empty() || name == "/") break;
        FaultFace ff;
        ff.i1 = std::atoi(tk.next().c_str());
        ff.i2 = std::atoi(tk.next().c_str());
        ff.j1 = std::atoi(tk.next().c_str());
        ff.j2 = std::atoi(tk.next().c_str());
        ff.k1 = std::atoi(tk.next().c_str());
        ff.k2 = std::atoi(tk.next().c_str());
        const std::string face = tk.next();
        tk.skip_to_slash();

        // 'X' and 'X+' mean the I+ face; 'X-' the I- face; likewise Y and Z.
        int f = -1;
        if (!face.empty()) {
          const char d = (char)std::toupper((unsigned char)face[0]);
          const bool minus = face.size() > 1 && face[1] == '-';
          if (d == 'X' || d == 'I') f = minus ? 0 : 1;
          else if (d == 'Y' || d == 'J') f = minus ? 2 : 3;
          else if (d == 'Z' || d == 'K') f = minus ? 4 : 5;
        }
        if (f < 0) continue;
        ff.face = f;

        auto it = std::find_if(faults_.begin(), faults_.end(),
                               [&](const Fault& x) { return x.name == name; });
        if (it == faults_.end()) {
          Fault nf; nf.name = name; nf.faces.push_back(ff);
          faults_.push_back(std::move(nf));
        } else {
          it->faces.push_back(ff);
        }
      }

    } else if (kw == "MULTFLT") {
      for (;;) {
        if (tk.at_slash()) { tk.eat_slash(); break; }
        if (tk.eof()) break;
        const std::string name = tk.next();
        if (name.empty() || name == "/") break;
        const Real m = std::strtod(tk.next().c_str(), nullptr);
        tk.skip_to_slash();
        bool found = false;
        for (auto& fl : faults_)
          if (fl.name == name) { fl.mult = m; found = true; }
        if (!found) ignored_.push_back("MULTFLT for unknown fault " + name);
      }

    } else if (kw == "MULTREGT") {
      // reg1 reg2 mult [dir] [nnc-flag] [region-array] /
      for (;;) {
        if (tk.at_slash()) { tk.eat_slash(); break; }
        if (tk.eof()) break;
        const std::string t1 = tk.next();
        if (t1.empty() || t1 == "/") break;
        RegionMult rm;
        rm.r1 = (t1 == "*" ) ? -1 : std::atoi(t1.c_str());
        const std::string t2 = tk.next();
        rm.r2 = (t2 == "*") ? -1 : std::atoi(t2.c_str());
        rm.mult = std::strtod(tk.next().c_str(), nullptr);
        // The remaining three items are optional and routinely defaulted with
        // `N*`, as in Norne's  `2  1  1.0  2* F/`.
        std::string dirs, nnc, reg;
        for (;;) {
          if (tk.at_slash()) { tk.eat_slash(); break; }
          if (tk.eof()) break;
          const std::string t = tk.next();
          if (t.empty() || t == "/") break;
          // An `N*` run of defaulted items carries no information here.
          if (t.find('*') != std::string::npos) continue;
          if (dirs.empty() && (t == "X" || t == "Y" || t == "Z" || t == "XY" ||
                               t == "XZ" || t == "YZ" || t == "XYZ"))
            dirs = t;
          else if (nnc.empty() && (t == "ALL" || t == "NNC" || t == "NONNC"))
            nnc = t;
          else
            reg = t;
        }
        if (!dirs.empty()) {
          rm.dir = 0;
          if (dirs.find('X') != std::string::npos) rm.dir |= 1;
          if (dirs.find('Y') != std::string::npos) rm.dir |= 2;
          if (dirs.find('Z') != std::string::npos) rm.dir |= 4;
        }
        // The last item names which region array to read, and it is a single
        // letter code, not an array name: 'F' selects FLUXNUM and 'M' selects
        // MULTNUM.  Taking it literally is silent: the lookup finds no array
        // called "F", every record is skipped, and a deck that seals twenty
        // flux regions against each other simulates as if fully open.
        if (!reg.empty()) {
          if (reg == "F") rm.region_array = "FLUXNUM";
          else if (reg == "M") rm.region_array = "MULTNUM";
          else rm.region_array = reg;
        }
        regmults_.push_back(rm);
      }

    } else if (kw == "THPRES") {
      // Explicit pressure barriers between EQLNUM regions.  The value is in
      // the deck's pressure unit and is converted with the rest of the grid
      // before device upload.  Directional records are retained; Norne gives
      // both directions with identical values.
      for (;;) {
        if (tk.at_slash()) { tk.eat_slash(); break; }
        if (tk.eof()) break;
        const std::string a = tk.next();
        if (a.empty() || a == "/") break;
        const std::string b = tk.next();
        const std::string p = tk.next();
        RegionThreshold t;
        t.r1 = std::atoi(a.c_str());
        t.r2 = std::atoi(b.c_str());
        t.pressure = std::strtod(p.c_str(), nullptr);
        tk.skip_to_slash();
        if (t.r1 > 0 && t.r2 > 0 && t.pressure >= 0)
          region_thresholds_.push_back(t);
      }

    } else if (kw == "INCLUDE") {
      std::string inc = tk.next();
      tk.skip_to_slash();
      // Deck paths are written with forward slashes and a leading './'.
      if (inc.rfind("./", 0) == 0) inc = inc.substr(2);
      read_grdecl(dir + "/" + inc);

    } else if (kw == "TITLE") {
      tk.skip_title();
    } else if (kw == "METRIC") { units_ = Units::Metric;
    } else if (kw == "FIELD")  { units_ = Units::Field;
    } else if (kw == "ECHO" || kw == "NOECHO" || kw == "GRID" ||
               kw == "EDIT" || kw == "RUNSPEC" || kw == "OIL" ||
               kw == "WATER" || kw == "GAS" || kw == "DISGAS" ||
               kw == "VAPOIL" || kw == "NEWTRAN" || kw == "GRIDFILE" ||
               kw == "INIT" || kw == "NOGGF" || kw == "UNIFIN" ||
               kw == "UNIFOUT") {
      // Section markers and flags: no data, no terminator.

    } else {
      // A keyword this reader does not implement.  Skip its data so the rest
      // of the file still parses, and record it so nothing is silently lost.
      ignored_.push_back(kw);
      tk.skip_to_slash();
    }
  }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
void CornerPointGrid::compute_geometry() {
  const size_t ng = (size_t)n_global();
  geom_vol_.assign(ng, 0.0);
  geom_cx_.assign(ng, 0.0);
  geom_cy_.assign(ng, 0.0);
  geom_cz_.assign(ng, 0.0);
  geom_dx_.assign(ng, 0.0);
  geom_dy_.assign(ng, 0.0);
  geom_dz_.assign(ng, 0.0);

  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const size_t g = (size_t)global_index(i, j, k);
        Real c[8][3];
        cell_corners(i, j, k, c);

        V3 v[8];
        for (int q = 0; q < 8; ++q) v[q] = {c[q][0], c[q][1], c[q][2]};

        // An interior point to fan the tetrahedra from.
        V3 cc{};
        for (int q = 0; q < 8; ++q) cc = cc + v[q];
        cc = cc * (1.0 / 8.0);

        Real vol = 0.0;
        V3 cen{};
        for (int f = 0; f < 6; ++f) {
          V3 fc{};
          for (int e = 0; e < 4; ++e) fc = fc + v[kFaceCorner[f][e]];
          fc = fc * 0.25;
          for (int e = 0; e < 4; ++e) {
            const V3& p0 = v[kFaceCorner[f][e]];
            const V3& p1 = v[kFaceCorner[f][(e + 1) & 3]];
            const Real tv =
                std::fabs(dot(p0 - cc, cross(p1 - cc, fc - cc))) / 6.0;
            vol += tv;
            cen = cen + (cc + p0 + p1 + fc) * (0.25 * tv);
          }
        }
        geom_vol_[g] = vol;
        if (vol > 0) cen = cen * (1.0 / vol); else cen = cc;
        geom_cx_[g] = cen.x; geom_cy_[g] = cen.y; geom_cz_[g] = cen.z;

        // Extents along the three index directions, as mean edge lengths.
        // The Peaceman well index needs all three, and a corner-point cell has
        // no single dx, so each is averaged over the four parallel edges.
        // Corner c = (a,b,cc) with a along I, b along J, cc along K, so the
        // edges are the corner pairs that differ in exactly one bit.
        Real ex = 0, ey = 0, ez = 0;
        for (int c = 0; c < 8; ++c) {
          if (!(c & 1)) {                       // pair with the +I corner
            const V3 d = v[c | 1] - v[c];
            ex += std::sqrt(dot(d, d));
          }
          if (!(c & 2)) {                       // pair with the +J corner
            const V3 d = v[c | 2] - v[c];
            ey += std::sqrt(dot(d, d));
          }
          if (!(c & 4)) {                       // pair with the +K corner
            const V3 d = v[c | 4] - v[c];
            ez += std::sqrt(dot(d, d));
          }
        }
        geom_dx_[g] = ex * 0.25;
        geom_dy_[g] = ey * 0.25;

        // Thickness uses the face-to-face drop rather than the edge length:
        // it is the vertical extent the well index and the pore volume want,
        // not the slant length of a deviated pillar.
        Real zt = 0, zb = 0;
        for (int e = 0; e < 4; ++e) {
          zt += v[kFaceCorner[4][e]].z;
          zb += v[kFaceCorner[5][e]].z;
        }
        geom_dz_[g] = std::fabs(zb - zt) * 0.25;
        (void)ez;
      }
}

// ---------------------------------------------------------------------------
// Active numbering
// ---------------------------------------------------------------------------
void CornerPointGrid::number_active() {
  const size_t ng = (size_t)n_global();

  // Pull the rock out of the array store, defaulting anything the deck did not
  // give.  PERMY and PERMZ default to PERMX, which is the deck convention.
  const std::vector<Real>* a_poro = array("PORO");
  const std::vector<Real>* a_ntg  = array("NTG");
  const std::vector<Real>* a_px   = array("PERMX");
  const std::vector<Real>* a_py   = array("PERMY");
  const std::vector<Real>* a_pz   = array("PERMZ");
  const std::vector<Real>* a_mpv  = array("MULTPV");
  const std::vector<Real>* a_act  = array("ACTNUM");

  std::vector<Real> g_poro = a_poro ? *a_poro : std::vector<Real>(ng, 0.0);
  std::vector<Real> g_ntg  = a_ntg  ? *a_ntg  : std::vector<Real>(ng, 1.0);
  std::vector<Real> g_px   = a_px   ? *a_px   : std::vector<Real>(ng, 0.0);
  std::vector<Real> g_py   = a_py   ? *a_py   : g_px;
  std::vector<Real> g_pz   = a_pz   ? *a_pz   : g_px;
  std::vector<Real> g_mpv  = a_mpv  ? *a_mpv  : std::vector<Real>(ng, 1.0);

  actnum_.assign(ng, 1);
  if (a_act)
    for (size_t g = 0; g < ng; ++g) actnum_[g] = ((*a_act)[g] != 0.0) ? 1 : 0;

  global_to_active_.assign(ng, -1);
  active_to_global_.clear();

  // Geometry is computed in the deck's LENGTH unit cubed -- m^3 for METRIC,
  // ft^3 for FIELD -- but a cell volume and a pore volume are RESERVOIR
  // volumes, and ECLIPSE states those in m^3 and in reservoir BARRELS.  MINPV
  // is quoted in the same unit, and everything downstream of here treats
  // volume and porv as reservoir volume: to_field_units() takes a metric m^3
  // to barrels, and the accumulation term is Vc * poro / dt against shrinkage
  // factors in stb/rb.
  //
  // So a FIELD deck's ft^3 has to become barrels here, at the one place the
  // geometry is turned into a volume.  Left in ft^3 it is 5.61x too large: the
  // accumulation term, the MINPV cut and the pore-volume ledger are all out by
  // that factor, and every one of them is silent -- the run converges and
  // reports a reservoir 5.61 times too big.  A METRIC deck is untouched, and
  // is converted to barrels later with the rest of the model.
  const Real vol_unit =
      (units_ == Units::Field) ? (Real)(1.0 / 5.614583) : (Real)1;  // ft^3/bbl

  minpv_cut_.assign(ng, 0);
  n_inactive_ = 0;
  n_minpv_ = 0;
  for (size_t g = 0; g < ng; ++g) {
    if (!actnum_[g]) { ++n_inactive_; continue; }
    const Real pv = geom_vol_[g] * vol_unit * g_poro[g] * g_ntg[g] * g_mpv[g];
    if (min_pv_ > 0 && pv < min_pv_) {
      // MINPV: too little pore volume to be worth a degree of freedom.  The
      // cell is dropped, and the vertical pass connects across it -- but only
      // it, not a neighbour ACTNUM already excluded; see minpv_cut_.
      actnum_[g] = 0;
      minpv_cut_[g] = 1;
      ++n_minpv_;
      continue;
    }
    global_to_active_[g] = (int)active_to_global_.size();
    active_to_global_.push_back((int)g);
  }

  // Record the post-MINPV ACTNUM so callers see what was actually simulated.
  {
    std::vector<Real>& a = array_for_write("ACTNUM");
    for (size_t g = 0; g < ng; ++g) a[g] = actnum_[g] ? 1.0 : 0.0;
  }

  const size_t na = active_to_global_.size();
  volume.resize(na); depth.resize(na);
  dx.resize(na); dy.resize(na); dz.resize(na);
  centroid_x.resize(na); centroid_y.resize(na); porv.resize(na);
  poro.resize(na); ntg.resize(na);
  permx.resize(na); permy.resize(na); permz.resize(na);
  for (size_t a = 0; a < na; ++a) {
    const size_t g = (size_t)active_to_global_[a];
    volume[a] = geom_vol_[g] * vol_unit;
    depth[a] = geom_cz_[g];
    dx[a] = geom_dx_[g];
    dy[a] = geom_dy_[g];
    dz[a] = geom_dz_[g];
    centroid_x[a] = geom_cx_[g];
    centroid_y[a] = geom_cy_[g];
    porv[a] = geom_vol_[g] * vol_unit * g_poro[g] * g_ntg[g] * g_mpv[g];
    poro[a] = g_poro[g]; ntg[a] = g_ntg[g];
    permx[a] = g_px[g]; permy[a] = g_py[g]; permz[a] = g_pz[g];
  }

  // Keep the resolved rock in the store on the global numbering, so the
  // connection pass and any later consumer read the same values.
  arrays_["PERMX"] = std::move(g_px);
  arrays_["PERMY"] = std::move(g_py);
  arrays_["PERMZ"] = std::move(g_pz);
  arrays_["PORO"]  = std::move(g_poro);
  arrays_["NTG"]   = std::move(g_ntg);
}

// ---------------------------------------------------------------------------
// Transmissibility
// ---------------------------------------------------------------------------
Real CornerPointGrid::trans_constant() const {
  // T has units of (reservoir volume rate) x viscosity / pressure, so the
  // constant absorbs the Darcy conversion for the deck's unit system.
  //   METRIC: mD, m, cP, bar, m^3/day
  //   FIELD : mD, ft, cP, psi, rb/day
  return (units_ == Units::Metric) ? 0.0085267014 : 0.00112712;
}

Real CornerPointGrid::half_trans(int g, int face, const Real perm[3]) const {
  const int k = g / (nx * ny);
  const int j = (g / nx) % ny;
  const int i = g % nx;

  Real c[8][3];
  cell_corners(i, j, k, c);
  V3 v[8];
  for (int q = 0; q < 8; ++q) v[q] = {c[q][0], c[q][1], c[q][2]};

  // Area vector of the face, as the sum of four triangles about the face
  // centre.  Summing triangles rather than taking one diagonal cross product
  // is what keeps a warped (non-planar) face correct.
  V3 fc{};
  for (int e = 0; e < 4; ++e) fc = fc + v[kFaceCorner[face][e]];
  fc = fc * 0.25;

  V3 area{};
  for (int e = 0; e < 4; ++e) {
    const V3& p0 = v[kFaceCorner[face][e]];
    const V3& p1 = v[kFaceCorner[face][(e + 1) & 3]];
    area = area + cross(p0 - fc, p1 - fc) * 0.5;
  }

  // Flow's CpGrid transmissibility path intentionally uses EclipseGrid's
  // input-grid centre, not the volume centroid exposed by the Dune cell
  // geometry.  EclipseGrid defines that centre as the arithmetic mean of the
  // eight corners.  Keep geom_c* as the physical, volume-weighted centroid for
  // depth and equilibration, but use the Eclipse centre for TPFA distances.
  V3 cen{};
  for (int q = 0; q < 8; ++q) cen = cen + v[q];
  cen = cen * (1.0 / 8.0);
  const V3 d = fc - cen;
  const Real d2 = dot(d, d);
  if (d2 <= 0) return 0.0;

  if (dot(area, d) < 0) area = area * -1.0;   // orient outward

  // Eclipse/OPM TPFA associates each logical face with the corresponding
  // diagonal permeability component (I -> Kx, J -> Ky, K -> Kz), then uses
  // the geometric projection |A.d|/|d|^2.  Projecting K*A component by
  // component is a different discretisation: on Norne's strongly sloping,
  // pinched K faces it leaks the much larger Kx/Ky into a nominally vertical
  // connection and produces individual TRANZ errors of several orders of
  // magnitude.  Keep the corner-point geometry in A and d, but select the
  // permeability by the face's topological direction exactly as Flow does.
  const int dim = face / 2;
  const Real akd = perm[dim] * std::fabs(dot(area, d));
  if (akd <= 0) return 0.0;

  return trans_constant() * akd / d2;
}

void CornerPointGrid::build_connections() {
  conns.clear();
  n_fault_conns_ = 0;
  n_nnc_ = 0;
  n_regt_conns_ = 0;

  const std::vector<Real>& px = arrays_["PERMX"];
  const std::vector<Real>& py = arrays_["PERMY"];
  const std::vector<Real>& pz = arrays_["PERMZ"];
  const std::vector<Real>& tg = arrays_["NTG"];

  auto mult_of = [&](const char* name, size_t g) -> Real {
    const std::vector<Real>* a = array(name);
    return a ? (*a)[g] : 1.0;
  };

  // Top and bottom depths at the two ends of a lateral face.  The endpoints
  // are ordered in increasing J on I faces and increasing I on J faces, so
  // the same parameter s in [0,1] refers to the same point on both sides of a
  // fault.  Averaging these four depths before intersecting loses a real
  // overlap whenever two warped faces meet only over part of their width.
  auto face_z_edges = [&](int i, int j, int k, int face,
                          Real top[2], Real bot[2]) {
    Real c[8][3];
    cell_corners(i, j, k, c);
    static const int top_corner[4][2] = {{0,2}, {1,3}, {0,1}, {2,3}};
    static const int bot_corner[4][2] = {{4,6}, {5,7}, {4,5}, {6,7}};
    for (int e = 0; e < 2; ++e) {
      top[e] = c[top_corner[face][e]][2];
      bot[e] = c[bot_corner[face][e]][2];
      if (bot[e] < top[e]) std::swap(bot[e], top[e]);
    }
  };

  // Integral over face width of max(0, min(bottoms) - max(tops)).  All edge
  // depths are linear in s; splitting at their crossings makes this exact.
  auto face_overlap = [](const Real t1[2], const Real b1[2],
                         const Real t2[2], const Real b2[2]) -> Real {
    std::vector<Real> cuts{0.0, 1.0};
    auto add_crossing = [&](const Real a[2], const Real c[2]) {
      const Real d0 = a[0] - c[0];
      const Real dd = (a[1] - a[0]) - (c[1] - c[0]);
      if (std::fabs(dd) <= (Real)1e-14) return;
      const Real s = -d0 / dd;
      if (s > 0 && s < 1) cuts.push_back(s);
    };
    add_crossing(t1, t2);
    add_crossing(b1, b2);
    add_crossing(b1, t1); add_crossing(b1, t2);
    add_crossing(b2, t1); add_crossing(b2, t2);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(), [](Real a, Real b) {
                 return std::fabs(a - b) <= (Real)1e-12;
               }), cuts.end());
    auto z = [](const Real a[2], Real s) { return a[0] + s * (a[1] - a[0]); };
    auto ov = [&](Real s) {
      return std::max((Real)0,
                      std::min(z(b1, s), z(b2, s)) -
                      std::max(z(t1, s), z(t2, s)));
    };
    Real area = 0;
    for (size_t q = 1; q < cuts.size(); ++q)
      area += (Real)0.5 * (ov(cuts[q - 1]) + ov(cuts[q])) *
              (cuts[q] - cuts[q - 1]);
    return area;
  };

  // ---- lateral connections, with juxtaposition ---------------------------
  for (int dir = 0; dir < 2; ++dir) {
    const int di = (dir == 0) ? 1 : 0;
    const int dj = (dir == 0) ? 0 : 1;
    const int face_pos = (dir == 0) ? 1 : 3;   // I+ / J+
    const int face_neg = (dir == 0) ? 0 : 2;   // I- / J-

    for (int j = 0; j + dj < ny; ++j)
      for (int i = 0; i + di < nx; ++i) {
        const int i2 = i + di, j2 = j + dj;

        // The broad min/max corner ranges are monotone down each column and
        // provide a safe sweep bound; the exact overlap below still uses all
        // four edge depths and therefore does not lose a partial-width face.
        int k2lo = 0;
        for (int k1 = 0; k1 < nz; ++k1) {
          const size_t g1 = (size_t)global_index(i, j, k1);
          if (!actnum_[g1]) continue;
          Real te1[2], be1[2];
          face_z_edges(i, j, k1, face_pos, te1, be1);
          const Real t1 = std::min(te1[0], te1[1]);
          const Real b1 = std::max(be1[0], be1[1]);
          const Real h1 = (Real)0.5 * ((be1[0] - te1[0]) +
                                      (be1[1] - te1[1]));
          if (h1 <= 0) continue;

          while (k2lo < nz) {
            Real tt_edge[2], bb_edge[2];
            face_z_edges(i2, j2, k2lo, face_neg, tt_edge, bb_edge);
            if (std::max(bb_edge[0], bb_edge[1]) > t1) break;
            ++k2lo;
          }

          for (int k2 = k2lo; k2 < nz; ++k2) {
            const size_t g2 = (size_t)global_index(i2, j2, k2);
            Real te2[2], be2[2];
            face_z_edges(i2, j2, k2, face_neg, te2, be2);
            const Real t2 = std::min(te2[0], te2[1]);
            const Real b2 = std::max(be2[0], be2[1]);
            if (t2 >= b1) break;
            if (b2 <= t1) continue;
            if (!actnum_[g2]) continue;
            const Real h2 = (Real)0.5 * ((be2[0] - te2[0]) +
                                        (be2[1] - te2[1]));
            if (h2 <= 0) continue;

            const Real ov = face_overlap(te1, be1, te2, be2);
            if (ov <= 1e-9) continue;

            // NTG thins the flowing part of a lateral face.
            const Real p1[3] = {px[g1] * tg[g1], py[g1] * tg[g1], pz[g1]};
            const Real p2[3] = {px[g2] * tg[g2], py[g2] * tg[g2], pz[g2]};

            const Real T1 = half_trans((int)g1, face_pos, p1) * (ov / h1);
            const Real T2 = half_trans((int)g2, face_neg, p2) * (ov / h2);
            if (T1 <= 0 || T2 <= 0) continue;

            Real T = 1.0 / (1.0 / T1 + 1.0 / T2);
            T *= (dir == 0) ? mult_of("MULTX", g1) : mult_of("MULTY", g1);
            T *= (dir == 0) ? mult_of("MULTX-", g2) : mult_of("MULTY-", g2);
            if (T <= 0) continue;

            Connection cn;
            cn.a = global_to_active_[g1];
            cn.b = global_to_active_[g2];
            if (cn.a > cn.b) std::swap(cn.a, cn.b);
            cn.trans = T;
            // A connection between different layers cannot arise on an
            // unfaulted grid, so it is reported as a non-neighbour one.
            cn.dir = (k1 == k2) ? dir : 3;
            if (k1 != k2) ++n_nnc_;
            conns.push_back(cn);
          }
        }
      }
  }

  // ---- vertical connections ----------------------------------------------
  // No throw in K, so this is the straightforward pass: connect each active
  // cell to the next active cell below it in its column.  With PINCH GAP, a
  // MINPV-cut cell may be crossed regardless of thickness.  An ACTNUM hole is
  // crossed only when PINCH is active and its actual vertical thickness is no
  // larger than the deck threshold.  This distinction matters in Norne: most
  // inactive cells are real non-reservoir barriers, but several collapsed
  // inactive layers are explicitly made vertically transparent by PINCH.
  //
  // Skipping over a cell ACTNUM already excluded is not the same thing, and
  // kprev must NOT survive one: that cell was never a thin streak the deck's
  // own pore volume trimmed, it is rock the deck never called reservoir --
  // shale, an eroded interval, out-of-zone -- and on Norne there are 138
  // times as many of those (68,417) as there are genuine MINPV pinch-outs
  // (496).  Bridging through it anyway connects the active cell above to the
  // active cell below with a half-transmissibility sized from each cell's
  // OWN face-to-centroid distance -- the same number two directly-touching
  // cells would get -- however many feet or layers of a real seal sit
  // between them.  That is a wormhole, not a pinch-out.
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      int kprev = -1;
      for (int k = 0; k < nz; ++k) {
        const size_t g = (size_t)global_index(i, j, k);
        if (!actnum_[g]) {
          const bool minpv_gap = minpv_cut_[g] && pinch_gap_;
          const bool collapsed = pinch_active_ &&
              geom_dz_[g] <= pinch_thickness_ + (Real)1e-12;
          if (!minpv_gap && !collapsed) kprev = -1;
          continue;
        }
        if (kprev >= 0) {
          const size_t gp = (size_t)global_index(i, j, kprev);
          const Real pp[3] = {px[gp], py[gp], pz[gp]};
          const Real pc[3] = {px[g], py[g], pz[g]};
          const Real T1 = half_trans((int)gp, 5, pp);   // K+ of the upper cell
          const Real T2 = half_trans((int)g, 4, pc);    // K- of the lower cell
          if (T1 > 0 && T2 > 0) {
            Real T = 1.0 / (1.0 / T1 + 1.0 / T2);
            T *= mult_of("MULTZ", gp) * mult_of("MULTZ-", g);
            if (T > 0) {
              Connection cn;
              cn.a = global_to_active_[gp];
              cn.b = global_to_active_[g];
              if (cn.a > cn.b) std::swap(cn.a, cn.b);
              cn.trans = T;
              cn.dir = 2;
              conns.push_back(cn);
            }
          }
        }
        kprev = k;
      }
    }

  apply_fault_multipliers();
  apply_region_multipliers();

  // Two cells can be joined by more than one connection once juxtaposition
  // splits a face, and the solver wants one entry per pair, so merge them.
  std::sort(conns.begin(), conns.end(), [](const Connection& x, const Connection& y) {
    return (x.a != y.a) ? (x.a < y.a) : (x.b < y.b);
  });
  size_t w = 0;
  for (size_t r = 0; r < conns.size(); ++r) {
    if (w > 0 && conns[w - 1].a == conns[r].a && conns[w - 1].b == conns[r].b) {
      conns[w - 1].trans += conns[r].trans;
    } else {
      conns[w++] = conns[r];
    }
  }
  conns.resize(w);

  // A multiplier of zero -- a sealing MULTFLT or MULTREGT -- means there is no
  // connection, not a connection that carries nothing.  Keeping it would put a
  // structurally zero entry in every matrix the solver builds and hand the AMG
  // hierarchy a graph edge across a barrier it is meant to respect, which is
  // exactly the coupling a sealed fault is there to prevent.  Dropped after the
  // merge, so partial faces that sum to something real are kept.
  //
  // The same argument extends past exactly zero.  A multiplier of 1e-4 applied
  // twice, or a sliver face on a fault, leaves a transmissibility many orders
  // of magnitude below the field's typical value: it carries no fluid on any
  // timescale the simulation resolves, and it costs a great deal of
  // conditioning.  Norne spans fourteen orders of magnitude between its
  // smallest and largest connection, and the algebraic multigrid in CPR has to
  // coarsen that graph.
  //
  // So connections below `drop_rel_` of the largest are removed too -- but
  // only where doing so leaves both cells still connected to something. A cell
  // cut loose entirely would have no pressure coupling at all, and its row in
  // the pressure operator would be identically zero, which is a worse problem
  // than the one being solved. At the default threshold on Norne this removes
  // about three per cent of connections carrying under a millionth of the
  // total transmissibility, and isolates none.
  Real tmax = 0;
  for (const Connection& c : conns) tmax = std::max(tmax, c.trans);
  const Real floor_t = (drop_rel_ > 0) ? drop_rel_ * tmax : 0.0;

  std::vector<int> degree((size_t)n_active(), 0);
  for (const Connection& c : conns)
    if (c.trans > floor_t) { ++degree[c.a]; ++degree[c.b]; }

  size_t k = 0;
  n_sealed_ = 0;
  n_dropped_small_ = 0;
  for (size_t r = 0; r < conns.size(); ++r) {
    const Connection& c = conns[r];
    if (!(c.trans > 0)) { ++n_sealed_; continue; }
    if (c.trans <= floor_t) {
      // Keep it only if either cell would otherwise be left with nothing.
      if (degree[c.a] > 0 && degree[c.b] > 0) { ++n_dropped_small_; continue; }
    }
    conns[k++] = conns[r];
  }
  conns.resize(k);

  // THPRES acts on phase-potential differences, not transmissibility, so it
  // remains a separate connection property.  Resolve the EQLNUM pair only
  // after active numbering and connection merging are final.
  if (!region_thresholds_.empty()) {
    const std::vector<Real>* eq = array("EQLNUM");
    if (!eq || eq->size() != (size_t)n_global()) {
      ignored_.push_back("THPRES without a complete EQLNUM array");
    } else {
      for (Connection& c : conns) {
        const int r1 = (int)std::lround((*eq)[active_to_global_[c.a]]);
        const int r2 = (int)std::lround((*eq)[active_to_global_[c.b]]);
        for (const RegionThreshold& t : region_thresholds_)
          if (t.r1 == r1 && t.r2 == r2)
            c.threshold = std::max(c.threshold, t.pressure);
      }
    }
  }
}

void CornerPointGrid::apply_fault_multipliers() {
  if (faults_.empty()) return;

  // Mark every (cell, face) a fault names, with its multiplier.  A face can be
  // named by more than one fault, in which case the multipliers compound --
  // which is what a deck means when it declares overlapping fault segments.
  std::vector<Real> fmul((size_t)n_global() * 6, 1.0);
  for (const Fault& fl : faults_)
    for (const FaultFace& ff : fl.faces)
      for (int k = ff.k1 - 1; k <= ff.k2 - 1; ++k) {
        if (k < 0 || k >= nz) continue;
        for (int j = ff.j1 - 1; j <= ff.j2 - 1; ++j) {
          if (j < 0 || j >= ny) continue;
          for (int i = ff.i1 - 1; i <= ff.i2 - 1; ++i) {
            if (i < 0 || i >= nx) continue;
            fmul[(size_t)global_index(i, j, k) * 6 + ff.face] *= fl.mult;
          }
        }
      }

  // A fault multiplier belongs to a face, and a connection crosses one face
  // from each side, so a connection is scaled if either side names it.
  //
  // Which face that is depends on which cell sits on the low side, and that
  // cannot be assumed from the connection's own ordering.  `a` and `b` are
  // sorted by active index, which follows the global index, which is K-major;
  // so for a connection that a fault throw has offset in K, the cell with the
  // lower active index can be the one on the *high* I side.  Assigning I+ to
  // `a` regardless would then look up the wrong two faces -- and it would go
  // wrong only on the faulted connections, which are precisely the ones the
  // multipliers exist to scale.
  for (Connection& cn : conns) {
    const int ga = active_to_global_[cn.a];
    const int gb = active_to_global_[cn.b];
    const int ia = ga % nx, ja = (ga / nx) % ny, ka = ga / (nx * ny);
    const int ib = gb % nx, jb = (gb / nx) % ny, kb = gb / (nx * ny);

    int fa, fb;
    if (ia != ib)      { fa = (ia < ib) ? 1 : 0; fb = 1 - fa; }
    else if (ja != jb) { fa = (ja < jb) ? 3 : 2; fb = 5 - fa; }
    else               { fa = (ka < kb) ? 5 : 4; fb = 9 - fa; }

    const Real m = fmul[(size_t)ga * 6 + fa] * fmul[(size_t)gb * 6 + fb];
    if (m != 1.0) {
      cn.trans *= m;
      ++n_fault_conns_;
    }
  }
}

void CornerPointGrid::apply_region_multipliers() {
  if (regmults_.empty()) return;

  for (const RegionMult& rm : regmults_) {
    const std::vector<Real>* reg = array(rm.region_array);
    if (!reg) {
      ignored_.push_back("MULTREGT on undefined " + rm.region_array);
      continue;
    }
    for (Connection& cn : conns) {
      const int ga = active_to_global_[cn.a];
      const int gb = active_to_global_[cn.b];
      const int ia = ga % nx, ja = (ga / nx) % ny;
      const int ib = gb % nx, jb = (gb / nx) % ny;

      const int bit = (ib != ia) ? 1 : (jb != ja) ? 2 : 4;
      if (!(rm.dir & bit)) continue;

      const int r1 = (int)(*reg)[(size_t)ga];
      const int r2 = (int)(*reg)[(size_t)gb];
      if (r1 == r2) continue;                       // same region: no boundary

      const bool fwd = (rm.r1 < 0 || rm.r1 == r1) && (rm.r2 < 0 || rm.r2 == r2);
      const bool rev = (rm.r1 < 0 || rm.r1 == r2) && (rm.r2 < 0 || rm.r2 == r1);
      if (!fwd && !rev) continue;

      cn.trans *= rm.mult;
      ++n_regt_conns_;
    }
  }
}

std::vector<std::string> CornerPointGrid::fault_names() const {
  std::vector<std::string> n;
  n.reserve(faults_.size());
  for (const Fault& f : faults_) n.push_back(f.name);
  return n;
}

bool CornerPointGrid::set_fault_multiplier(const std::string& name, Real m) {
  bool found = false;
  for (Fault& f : faults_)
    if (f.name == name) { f.mult = m; found = true; }
  return found;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Block-centred geometry
// ---------------------------------------------------------------------------
// A block-centred deck gives cell sizes and the depth of the top layer, and
// leaves the corners implied.  Rather than carry a second geometry path
// through the connection builder, the sizes are converted once into the COORD
// and ZCORN a corner-point deck would have supplied, and everything after this
// point is identical for both kinds of deck.
//
// The pillars are vertical, at the accumulated x and y positions.  That is
// exact for a brick grid, which is what block-centred means.
//
// DX may legally vary with j, and DY with i, but such a deck does not describe
// a grid whose cells share faces: cell (0,0) and cell (0,1) would have
// different widths along the same pillar.  ECLIPSE resolves this by taking the
// sizes from the first row and column, and so does this, after saying so.
bool CornerPointGrid::synthesise_block_centred() {
  if (nx <= 0 || ny <= 0 || nz <= 0) return false;
  const size_t ncell = (size_t)nx * ny * nz;

  const std::vector<Real>* dxa = array("DX");
  const std::vector<Real>* dya = array("DY");
  const std::vector<Real>* dza = array("DZ");
  const bool have_v = !dxv_.empty() && !dyv_.empty() && !dzv_.empty();
  const bool have_a = dxa && dya && dza;
  if (!have_v && !have_a) return false;

  // Widths along each axis, taken from the vectors when given and otherwise
  // from the first row, column and column of the per-cell arrays.
  std::vector<Real> wx(nx), wy(ny);
  if (!dxv_.empty()) wx = dxv_;
  else for (int i = 0; i < nx; ++i) wx[i] = (*dxa)[(size_t)i];
  if (!dyv_.empty()) wy = dyv_;
  else for (int j = 0; j < ny; ++j) wy[j] = (*dya)[(size_t)j * nx];

  for (int i = 0; i < nx; ++i)
    if (!(wx[i] > 0)) throw std::runtime_error("DX/DXV: non-positive cell width");
  for (int j = 0; j < ny; ++j)
    if (!(wy[j] > 0)) throw std::runtime_error("DY/DYV: non-positive cell width");

  // Thickness is genuinely per cell: a block-centred deck varies DZ by layer
  // and often by cell, and the corner-point form represents that exactly.
  std::vector<Real> dz(ncell);
  if (!dzv_.empty()) {
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          dz[((size_t)k * ny + j) * nx + i] = dzv_[k];
  } else {
    if (dza->size() != ncell) throw std::runtime_error("DZ: wrong length");
    dz = *dza;
  }

  // TOPS gives the depth of the top face.  Decks supply it for the first layer
  // only, which is nx*ny values, or for every cell.  Anything below the first
  // layer is stacked on the layer above, which is what makes the column
  // continuous.
  const std::vector<Real>* tops = array("TOPS");
  std::vector<Real> ztop(ncell, 0.0);
  const size_t nlayer = (size_t)nx * ny;
  for (int k = 0; k < nz; ++k) {
    for (size_t c = 0; c < nlayer; ++c) {
      const size_t g = (size_t)k * nlayer + c;
      if (tops && tops->size() >= g + 1 && k > 0 && tops->size() == ncell)
        ztop[g] = (*tops)[g];
      else if (k == 0)
        ztop[g] = tops ? (*tops)[c] : 0.0;
      else
        ztop[g] = ztop[g - nlayer] + dz[g - nlayer];
    }
  }

  // Pillar positions: the cumulative sums of the widths.
  std::vector<Real> px(nx + 1, 0.0), py(ny + 1, 0.0);
  for (int i = 0; i < nx; ++i) px[i + 1] = px[i] + wx[i];
  for (int j = 0; j < ny; ++j) py[j + 1] = py[j] + wy[j];

  Real zmin = ztop[0], zmax = ztop[0];
  for (size_t g = 0; g < ncell; ++g) {
    zmin = std::min(zmin, ztop[g]);
    zmax = std::max(zmax, ztop[g] + dz[g]);
  }

  coord_.assign((size_t)6 * (nx + 1) * (ny + 1), 0.0);
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      const size_t p = (size_t)(j * (nx + 1) + i) * 6;
      coord_[p + 0] = px[i]; coord_[p + 1] = py[j]; coord_[p + 2] = zmin;
      coord_[p + 3] = px[i]; coord_[p + 4] = py[j]; coord_[p + 5] = zmax;
    }
  }

  // Eight corners per cell, indexed exactly as cell_corners() reads them.
  const int nxc = 2 * nx, nyc = 2 * ny;
  zcorn_.assign((size_t)8 * ncell, 0.0);
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const size_t g = ((size_t)k * ny + j) * nx + i;
        const Real zt = ztop[g], zb = ztop[g] + dz[g];
        for (int c = 0; c < 8; ++c) {
          const int a = c & 1, b = (c >> 1) & 1, cc = (c >> 2) & 1;
          const size_t zi =
              ((size_t)(2 * k + cc) * nyc + (2 * j + b)) * nxc + (2 * i + a);
          zcorn_[zi] = cc ? zb : zt;
        }
      }
    }
  }

  block_centred_ = true;
  return true;
}

void CornerPointGrid::build() {
  if (nx <= 0 || ny <= 0 || nz <= 0)
    throw std::runtime_error("grid dimensions were never set");
  // A block-centred deck gives no corners; build them from the cell sizes
  // before the checks that follow, so that both kinds of deck reach the same
  // geometry code.
  if (coord_.empty() && zcorn_.empty()) synthesise_block_centred();

  if (coord_.size() != (size_t)6 * (nx + 1) * (ny + 1))
    throw std::runtime_error(
        "no usable geometry: the deck gave neither COORD and ZCORN "
        "(corner-point) nor DX/DY/DZ or DXV/DYV/DZV with TOPS "
        "(block-centred)");
  if (zcorn_.size() != (size_t)8 * nx * ny * nz)
    throw std::runtime_error("ZCORN is missing or the wrong length");

  compute_geometry();
  number_active();
  build_connections();
}

std::string CornerPointGrid::summary() const {
  std::ostringstream os;
  const char* u = (units_ == Units::Metric) ? "METRIC" : "FIELD";

  double pv = 0, bv = 0, vmin = 1e300, vmax = 0;
  for (size_t a = 0; a < porv.size(); ++a) {
    pv += porv[a];
    bv += volume[a];
    vmin = std::min(vmin, (double)volume[a]);
    vmax = std::max(vmax, (double)volume[a]);
  }
  double dmin = 1e300, dmax = -1e300;
  for (Real d : depth) { dmin = std::min(dmin, (double)d); dmax = std::max(dmax, (double)d); }

  int nI = 0, nJ = 0, nK = 0, nN = 0;
  double tmin = 1e300, tmax = 0, tsum = 0;
  for (const Connection& c : conns) {
    switch (c.dir) { case 0: ++nI; break; case 1: ++nJ; break;
                     case 2: ++nK; break; default: ++nN; }
    tmin = std::min(tmin, (double)c.trans);
    tmax = std::max(tmax, (double)c.trans);
    tsum += c.trans;
  }

  char b[320];
  os << "Corner-point grid\n";
  std::snprintf(b, sizeof b, "  dimensions      : %d x %d x %d = %d cells\n",
                nx, ny, nz, n_global()); os << b;
  std::snprintf(b, sizeof b, "  active          : %d  (%d inactive, %d below MINPV %.4g)\n",
                n_active(), n_inactive_, n_minpv_, (double)min_pv_); os << b;
  if (pinch_active_) {
    std::snprintf(b, sizeof b, "  PINCH           : %.4g thickness, %s\n",
                  (double)pinch_thickness_, pinch_gap_ ? "GAP" : "NOGAP");
    os << b;
  }
  std::snprintf(b, sizeof b, "  units           : %s\n", u); os << b;
  std::snprintf(b, sizeof b, "  bulk volume     : %.6g total, cells %.4g .. %.4g\n",
                bv, vmin, vmax); os << b;
  std::snprintf(b, sizeof b, "  pore volume     : %.6g total\n", pv); os << b;
  std::snprintf(b, sizeof b, "  depth range     : %.2f .. %.2f\n", dmin, dmax); os << b;
  std::snprintf(b, sizeof b, "  faults          : %d named, %d connections scaled\n",
                (int)faults_.size(), n_fault_conns_); os << b;
  if (!regmults_.empty()) {
    std::snprintf(b, sizeof b, "  MULTREGT        : %d records, %d connections scaled\n",
                  (int)regmults_.size(), n_regt_conns_); os << b;
  }
  std::snprintf(b, sizeof b, "  connections     : %d  (I %d, J %d, K %d, non-neighbour %d)\n",
                (int)conns.size(), nI, nJ, nK, nN); os << b;
  if (n_sealed_) {
    std::snprintf(b, sizeof b, "  sealed shut     : %d connections dropped "
                  "(zero multiplier)\n", n_sealed_); os << b;
  }
  std::snprintf(b, sizeof b, "  transmissibility: %.4g .. %.4g, total %.6g\n",
                tmin, tmax, tsum); os << b;
  if (!ignored_.empty()) {
    // Report each distinct one once; a deck repeats keywords across includes.
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
