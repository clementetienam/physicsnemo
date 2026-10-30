// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// deck_token.hpp -- the shared lexer for ECLIPSE-style decks.
//
// The grid reader and the schedule reader consume the same grammar, and the
// subtleties in it are the kind that are wrong once and then wrong everywhere:
// `N*value` repeat counts, `--` comments, quoted names, and -- the one that
// costs the most -- free text after a record's terminating slash.  Norne writes
//
//     'PERMZ' 0.64  1 46 1 112 13 13 /  Tofte 2.1.3
//
// and a reader that treats what follows the slash as data does not merely
// mis-parse that record; it goes looking for a terminator it has already
// passed and swallows the next one whole.  Keeping one lexer means that rule,
// and every rule like it, is stated once.
#pragma once

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvrs {

class DeckTokenizer {
 public:
  // Decks reach hundreds of megabytes, so the file is read once and scanned in
  // place rather than streamed through iostreams.
  explicit DeckTokenizer(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    buf_.resize((size_t)n);
    f.read(&buf_[0], n);
    buf_.resize((size_t)f.gcount());
  }

  // Advance past whitespace and `--` comments.
  void skip() {
    while (p_ < buf_.size()) {
      const char c = buf_[p_];
      if (c == '-' && p_ + 1 < buf_.size() && buf_[p_ + 1] == '-') {
        while (p_ < buf_.size() && buf_[p_] != '\n') ++p_;
      } else if (std::isspace((unsigned char)c)) {
        ++p_;
      } else {
        return;
      }
    }
  }

  bool eof() { skip(); return p_ >= buf_.size(); }
  bool at_slash() { skip(); return p_ < buf_.size() && buf_[p_] == '/'; }

  // Step over a record terminator and any free text that follows it on the
  // same line.  See the note at the top of this file.
  void consume_slash() {
    ++p_;
    while (p_ < buf_.size() && buf_[p_] != '\n') ++p_;
  }

  void eat_slash() { if (at_slash()) consume_slash(); }

  // One whitespace-delimited token.  Quoted names come back without quotes; a
  // record terminator comes back as "/".
  std::string next() {
    skip();
    if (p_ >= buf_.size()) return {};
    if (buf_[p_] == '\'') {
      const size_t s = ++p_;
      while (p_ < buf_.size() && buf_[p_] != '\'') ++p_;
      std::string t = buf_.substr(s, p_ - s);
      if (p_ < buf_.size()) ++p_;
      return trim(t);
    }
    if (buf_[p_] == '/') { consume_slash(); return "/"; }
    const size_t s = p_;
    while (p_ < buf_.size() && !std::isspace((unsigned char)buf_[p_]) &&
           buf_[p_] != '/')
      ++p_;
    if (p_ == s) { ++p_; return next(); }
    return buf_.substr(s, p_ - s);
  }

  std::string peek() {
    const size_t save = p_;
    std::string t = next();
    p_ = save;
    return t;
  }

  // Is the next token quoted?  A quoted token is always a data field -- a well
  // name, a group, a report mnemonic -- and never a keyword, however much it
  // looks like one.  RPTSCHED's mnemonic list contains a literal 'WELSPECS',
  // so anything deciding "has the next keyword started?" has to ask this
  // first or it will mistake that mnemonic for the start of a well section.
  bool next_is_quoted() {
    skip();
    return p_ < buf_.size() && buf_[p_] == '\'';
  }

  // Read numbers up to the terminating slash, expanding `N*value` repeats.
  // `expect` reserves; it never truncates, so a length mismatch is visible to
  // the caller rather than hidden here.
  void read_values(std::vector<double>& out, size_t expect = 0) {
    if (expect) out.reserve(expect);
    for (;;) {
      skip();
      if (p_ >= buf_.size()) return;
      if (buf_[p_] == '/') { consume_slash(); return; }
      const size_t s = p_;
      while (p_ < buf_.size() && !std::isspace((unsigned char)buf_[p_]) &&
             buf_[p_] != '/')
        ++p_;
      const char* tok = buf_.c_str() + s;
      const size_t len = p_ - s;
      if (!len) continue;
      const char* star = (const char*)std::memchr(tok, '*', len);
      if (star) {
        // `N*value`, and the bare `N*` form meaning N defaulted entries.
        const long rep = std::strtol(tok, nullptr, 10);
        const double v =
            (star + 1 < tok + len) ? std::strtod(star + 1, nullptr) : 0.0;
        if (rep > 0) out.insert(out.end(), (size_t)rep, v);
      } else {
        // A trailing flag such as SPECGRID's 'F' is not a number.
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
      if (p_ >= buf_.size()) return;
      if (buf_[p_] == '/') { consume_slash(); return; }
      next();
    }
  }

  // TITLE is free text terminated by a line ending, not by '/'.  Eclipse
  // accepts the text either after the keyword or on the following line.
  void skip_title() {
    bool text_on_keyword_line = false;
    while (p_ < buf_.size() && buf_[p_] != '\n') {
      if (!std::isspace((unsigned char)buf_[p_]))
        text_on_keyword_line = true;
      ++p_;
    }
    if (!text_on_keyword_line && p_ < buf_.size()) {
      ++p_;
      while (p_ < buf_.size() && buf_[p_] != '\n') ++p_;
    }
  }

  // Read one record's fields, up to the terminating slash.
  //
  // A defaulted run is written `N*` and stands for N *positions*, not one.
  //
  //     'PROD1' 'OPEN' 'ORAT' 1000 4* 90 /
  //
  // has the 90 in field 8 (the bottom-hole pressure), because `4*` fills
  // fields 4 through 7.  Returning `4*` as a single entry would put the 90 in
  // field 5 and read it as a water rate -- the record still parses, the well
  // still runs, and every limit after the default is attached to the wrong
  // quantity.  So the run is expanded here, and each position is marked "1*"
  // for the accessors to recognise.  This is the opposite of what `N*` means
  // in an array, where it is a repeat count; that case is read_values().
  std::vector<std::string> read_record() {
    std::vector<std::string> out;
    for (;;) {
      skip();
      if (p_ >= buf_.size()) return out;
      if (buf_[p_] == '/') { consume_slash(); return out; }
      std::string t = next();
      if (t.empty() || t == "/") return out;
      const size_t star = t.find('*');
      bool numeric_prefix = star != std::string::npos && star > 0;
      for (size_t i = 0; i < star && numeric_prefix; ++i)
        if (!std::isdigit((unsigned char)t[i])) numeric_prefix = false;
      if (numeric_prefix && star + 1 == t.size()) {
        const long rep = std::strtol(t.c_str(), nullptr, 10);
        for (long i = 0; i < rep; ++i) out.push_back("1*");
      } else {
        out.push_back(std::move(t));
      }
    }
  }

  size_t pos() const { return p_; }

  // Is this token shaped like a keyword?
  static bool is_keyword(const std::string& t) {
    if (t.empty() || t.size() > 8) return false;
    if (!std::isalpha((unsigned char)t[0])) return false;
    for (char c : t)
      if (!std::isalnum((unsigned char)c) && c != '_' && c != '-' && c != '+')
        return false;
    return true;
  }

  // Has the current keyword's record list ended?
  //
  // Deck keywords terminate their record lists in two different ways and give
  // no syntactic hint which is which.  Some end with an empty record -- a bare
  // slash on its own line.  Others take a fixed number of records set
  // elsewhere in the deck (EQUIL takes NTEQUL, DENSITY takes NTPVT) and simply
  // stop, with the next keyword following immediately.
  //
  // Reading the second kind as though it were the first consumes the keyword
  // that follows and everything belonging to it.  That failure is silent and
  // its symptoms appear far away: a DENSITY swallowed by EQUIL shows up much
  // later as a field with no fluid densities at all.
  //
  // So a record list ends at whichever comes first: the empty record, the end
  // of input, or a bare token that can only be a keyword.  "Bare" matters --
  // a quoted token is a data field however much it reads like a keyword.
  bool at_records_end() {
    if (at_slash()) return true;
    if (eof()) return true;
    if (next_is_quoted()) return false;
    return looks_like_bare_keyword(peek());
  }

  // A token that can only be a keyword: alphabetic, at least three characters,
  // no digits.  Numeric fields fail on the first character, and names that a
  // deck cares about are quoted.
  static bool looks_like_bare_keyword(const std::string& t) {
    if (t.size() < 3 || t.size() > 8) return false;
    for (char c : t)
      if (!std::isalpha((unsigned char)c) && c != '_') return false;
    return true;
  }

  // A defaulted field, written `1*` or `3*` or bare `*`.
  static bool is_default(const std::string& t) {
    if (t.empty()) return true;
    if (t == "*") return true;
    const size_t star = t.find('*');
    if (star == std::string::npos) return false;
    for (size_t i = 0; i < star; ++i)
      if (!std::isdigit((unsigned char)t[i])) return false;
    return star + 1 == t.size();
  }

  // Field accessors that apply a default when the deck omitted the field.
  static double num(const std::vector<std::string>& r, size_t i, double dflt) {
    if (i >= r.size() || is_default(r[i])) return dflt;
    return std::strtod(r[i].c_str(), nullptr);
  }
  static int integer(const std::vector<std::string>& r, size_t i, int dflt) {
    if (i >= r.size() || is_default(r[i])) return dflt;
    return std::atoi(r[i].c_str());
  }
  static std::string text(const std::vector<std::string>& r, size_t i,
                          const std::string& dflt) {
    if (i >= r.size() || is_default(r[i])) return dflt;
    return r[i];
  }

 private:
  static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
  }

  std::string buf_;
  size_t p_ = 0;
};

}  // namespace nvrs
