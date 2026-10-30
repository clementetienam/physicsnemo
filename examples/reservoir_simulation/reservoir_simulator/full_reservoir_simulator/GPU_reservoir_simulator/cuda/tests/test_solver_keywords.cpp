// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// test_solver_keywords.cpp -- the stage-1 library keywords of the NVRS deck.
//
// LINSOLVER picks which multigrid sits inside CPR, and each library carries its
// own knobs.  A typo in one of them must stop the run, not silently select a
// default: a benchmark labelled "Ginkgo FP16" that actually ran FP32, or AmgX,
// is worse than no benchmark.  So every accepted spelling is checked to land
// in the Deck, and every rejected one to throw.
//
//   g++ -O2 -std=c++17 -I../include test_solver_keywords.cpp ../src/deck.cpp -o t && ./t

#include "deck.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace nvrs;

namespace {

int g_checks = 0, g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) { ++g_failures; std::printf("  FAIL  %s\n", what.c_str()); }
  else      std::printf("  ok    %s\n", what.c_str());
}

const char* kPath = "test_solver_keywords.DATA";

Deck parse_text(const std::string& body) {
  {
    std::ofstream f(kPath);
    f << body;
  }
  return parse_deck(kPath);
}

bool throws(const std::string& body) {
  try {
    (void)parse_text(body);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  std::printf("LINSOLVER and the stage-1 library keywords\n");

  {
    const Deck d = parse_text("");
    check(d.backend == 1, "default stage-1 library is AmgX");
    check(d.ginkgo_precision == 16, "default GINKGOPREC is 16");
  }

  for (const char* v : {"AMGX", "amgx", "1"})
    check(parse_text(std::string("LINSOLVER ") + v + "\n").backend == 1,
          std::string("LINSOLVER ") + v + " selects AmgX");
  for (const char* v : {"HYPRE", "hypre", "0"})
    check(parse_text(std::string("LINSOLVER ") + v + "\n").backend == 0,
          std::string("LINSOLVER ") + v + " selects HYPRE");
  for (const char* v : {"GINKGO", "ginkgo", "Ginkgo", "2"})
    check(parse_text(std::string("LINSOLVER ") + v + "\n").backend == 2,
          std::string("LINSOLVER ") + v + " selects Ginkgo");
  check(parse_text("AMGLIB GINKGO\n").backend == 2, "AMGLIB is an alias");

  for (const char* v : {"32", "FP32", "single"})
    check(parse_text(std::string("GINKGOPREC ") + v + "\n").ginkgo_precision == 32,
          std::string("GINKGOPREC ") + v + " is FP32");
  for (const char* v : {"16", "fp16", "HALF"})
    check(parse_text(std::string("GINKGOPREC ") + v + "\n").ginkgo_precision == 16,
          std::string("GINKGOPREC ") + v + " is FP16");

  check(throws("LINSOLVER AMGXX\n"), "LINSOLVER typo is rejected");
  check(throws("LINSOLVER 7\n"), "LINSOLVER 7 is rejected");
  check(throws("GINKGOPREC 8\n"), "GINKGOPREC 8 is rejected");
  check(throws("GINKGOPREC FP64\n"), "GINKGOPREC FP64 is rejected");

  // The Tensor Core keywords parse in every build; only the run needs TENSORCORE=1.
  check(parse_text("JACOBIAN TENSORCORE\n").jacobian == 2, "JACOBIAN TENSORCORE");
  check(parse_text("JACOBIAN ANALYTIC\n").jacobian == 1, "JACOBIAN ANALYTIC");
  check(parse_text("JACOBIAN ANALYTIC\nJACOBIAN TENSORCORE\n").jacobian == 2,
        "the last JACOBIAN in a deck wins");
  check(parse_text("TCPRECISION TF32\n").tc_precision == 32, "TCPRECISION TF32");
  check(parse_text("TCPRECISION FP32\n").tc_precision == 0, "TCPRECISION FP32");
  check(parse_text("TCVALIDATE 1\n").tc_validate == 1, "TCVALIDATE 1");
  check(parse_text("\n").tc_validate == 0, "TCVALIDATE is off by default");
  check(throws("JACOBIAN TENSORCOREX\n"), "JACOBIAN typo is rejected");
  check(throws("TCPRECISION FP16\n"), "TCPRECISION FP16 is rejected");

  std::remove(kPath);
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
