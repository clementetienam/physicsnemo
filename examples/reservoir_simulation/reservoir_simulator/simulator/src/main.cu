// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// main.cu — NVRS-CUDA driver.  Reads an OPM/Eclipse-style deck, decomposes the
// grid across MPI ranks (one GPU per rank → single/multi-GPU, single/multi
// node), then advances the reservoir with a Newton loop whose linear systems
// are solved by the CPR-preconditioned mixed-precision Krylov solver.  Fields,
// well rates and an insight summary are written to the output directory and the
// Python post-processor is invoked to render plots automatically.
#include "deck.hpp"
#include "tc_jacobian.hpp"
#include "field_model.hpp"
#include "grid.hpp"
#include "halo.hpp"
#include "halo_conn.hpp"
#include "io.hpp"
#include "well_report.hpp"
#include "linear_solver.hpp"
#include "physics.hpp"
#include "sequential.hpp"

#include <HYPRE.h>
#include <HYPRE_utilities.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

using namespace nvrs;

namespace {

namespace fs = std::filesystem;

// Locate the post-processor without assuming anything about the working
// directory.  Search order:
//   1. $NVRS_PLOT_SCRIPT               (explicit override)
//   2. <dir of argv[0]>/../scripts/    (build/nvrs_cuda → cuda/scripts)
//   3. <dir of argv[0]>/scripts/       (installed side by side)
//   4. ./scripts/                      (running from the source tree)
// Returns an empty path when the script cannot be found, in which case the
// end-of-run plotting is skipped with a clear message instead of a shell error.
fs::path find_plot_script(const char* argv0) {
  if (const char* env = std::getenv("NVRS_PLOT_SCRIPT")) {
    fs::path p(env);
    if (fs::exists(p)) return p;
  }
  std::error_code ec;
  fs::path exe = fs::absolute(fs::path(argv0 ? argv0 : ""), ec);
  fs::path dir = exe.parent_path();
  const fs::path candidates[] = {
      dir / ".." / "scripts" / "plot_results.py",
      dir / "scripts" / "plot_results.py",
      fs::path("scripts") / "plot_results.py",
  };
  for (const auto& c : candidates)
    if (fs::exists(c, ec)) return fs::weakly_canonical(c, ec);
  return {};
}

// Integer environment override.  Used for the HYPRE kernel switches, which
// have to be set before the deck is read.
int nvrs_env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  char* end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v) return fallback;
  return (int)n;
}

// Pick a Python interpreter: $NVRS_PYTHON wins, else python3, else python.
std::string find_python() {
  if (const char* env = std::getenv("NVRS_PYTHON")) return std::string(env);
#if defined(_WIN32)
  return "python";
#else
  return "python3";
#endif
}

// Quote a path for the shell so directories containing spaces still work.
std::string shell_quote(const std::string& s) {
#if defined(_WIN32)
  return "\"" + s + "\"";
#else
  std::string out = "'";
  for (char c : s) { if (c == '\'') out += "'\\''"; else out += c; }
  return out + "'";
#endif
}

// ---------------------------------------------------------------------------
// Launch banner, in the style OPM Flow prints at start-up.
// ---------------------------------------------------------------------------
constexpr int kBoxWidth = 78;   // total width including the '*' borders

void box_rule() { std::printf("%s\n", std::string(kBoxWidth, '*').c_str()); }

// One boxed line, centred if `centre` is set, otherwise indented.
void box_line(const std::string& text = "", bool centre = false) {
  const int inner = kBoxWidth - 2;
  std::string t = text.substr(0, (size_t)inner - 2);
  int pad_l = centre ? (inner - (int)t.size()) / 2 : 3;
  int pad_r = inner - (int)t.size() - pad_l;
  std::printf("*%s%s%s*\n", std::string(pad_l, ' ').c_str(), t.c_str(),
              std::string(pad_r, ' ').c_str());
}

// Which multigrid the banner should name.  The banner prints before the deck
// is parsed -- deliberately, so a run that dies in GPU binding has already
// said which host and how many ranks -- so this takes one cheap look at the
// file for LINSOLVER rather than reordering the startup.  It is not a second
// parser: anything it fails to understand falls through to the same default
// deck.hpp uses, and the authoritative line is still printed after the real
// parse.
const char* peek_backend(const char* deck_path) {
  int be = 1;  // deck.hpp default: 1 = AmgX, 0 = HYPRE, 2 = Ginkgo
  std::ifstream f(deck_path ? deck_path : "");
  std::string line;
  while (std::getline(f, line)) {
    const auto c = line.find("--");
    if (c != std::string::npos) line = line.substr(0, c);
    std::istringstream is(line);
    std::string k, v;
    if (!(is >> k >> v)) continue;
    if (k != "LINSOLVER" && k != "AMGLIB") continue;
    for (auto& ch : v) ch = (char)std::toupper((unsigned char)ch);
    if (v == "HYPRE" || v == "0") be = 0;
    else if (v == "AMGX" || v == "1") be = 1;
    else if (v == "GINKGO" || v == "2") be = 2;
  }
  // The same fallbacks main applies after the real parse, so the banner names
  // the library that will actually run.
#ifndef NVRS_HAVE_GINKGO
  if (be == 2) be = 1;
#endif
#ifndef NVRS_HAVE_AMGX
  if (be == 1) be = 0;
#endif
  if (be == 2) return "Ginkgo PGM AMG";
  return be == 1 ? "AmgX aggregation AMG" : "HYPRE BoomerAMG";
}

// Is stage 1 HYPRE?  Only then do the HYPRE build and runtime switches
// describe the run.
bool banner_uses_hypre(const char* amglib) {
  return amglib && std::strstr(amglib, "HYPRE") != nullptr;
}

// Does this run use AmgX for stage 1?  The banner prints the configuration of
// the multigrid actually in use and says nothing about the other library: a
// HYPRE build detail in an AmgX run is noise that has been read as fact more
// than once.  peek_backend() already resolves the deck keyword and the
// build, so it is the single source of that answer.
bool banner_uses_amgx(const char* amglib) {
  return amglib && std::strstr(amglib, "AmgX") != nullptr;
}

void print_banner(int nranks, const std::string& host, const char* amglib) {
  std::time_t now = std::time(nullptr);
  char when[80] = "unknown";
  if (std::tm* lt = std::localtime(&now))
    std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S %Z", lt);

  box_rule();
  box_line();
  box_line("N V R S   R E S E R V O I R   S I M U L A T O R", true);
  box_line();
  box_line("GPU-native black-oil and compositional reservoir simulation", true);
  box_line(std::string("Mixed-precision CPR: ") + amglib +
           " + block-Jacobi on CUDA", true);
  box_line();
  box_line("C. Etienam | T. Onishi | O. Ovcharenko | N. Luiken | I. Said",
           true);
  box_line("NVIDIA - Global Energy Team", true);
  box_line();
  box_rule();

  std::printf("  Launched : %s\n", when);
  std::printf("  Host     : %s\n", host.c_str());
  std::printf("  Parallel : %d MPI rank%s, one GPU per rank\n", nranks,
              nranks == 1 ? "" : "s");
  std::printf("  Precision: %s outer Krylov, %s CPR preconditioner\n",
              sizeof(Real) == 8 ? "FP64" : "FP32",
              sizeof(PrecReal) == 8 ? "FP64" : "FP32");

  // What HYPRE was BUILT with, not what it was asked for.  A HYPRE without
  // GPU-aware MPI stages every BoomerAMG halo exchange through host memory --
  // device to host, MPI, host to device, at every level of the hierarchy, on
  // every V-cycle, on every Newton iteration.  On one rank there are no
  // exchanges, so it costs nothing and nothing looks wrong; on several ranks it
  // can make the run slower than a single GPU.  Finding that took a multi-GPU
  // scaling study, so the configuration is printed on every run from now on.
  if (banner_uses_amgx(amglib)) {
    // Stage 1 is AmgX: state its hierarchy and why, and print nothing about
    // HYPRE, which this run never calls.
    std::printf("  AmgX     : %s hierarchy%s\n", amgx_algo_name(nranks),
                std::getenv("NVRS_AMGX_ALGO")
                    ? " (forced by NVRS_AMGX_ALGO)"
                    : (nranks > 1 ? " (automatic: distributed-capable)"
                                  : " (automatic: strongest on one rank)"));
    std::fflush(stdout);
    return;
  }
  if (!banner_uses_hypre(amglib)) {
    // Ginkgo: the version it was built against.  The precision and the
    // hierarchy are printed at the first setup, once the deck is known.
    std::printf("  Ginkgo   : %s, single GPU\n", ginkgo_version());
    std::fflush(stdout);
    return;
  }

  std::printf("  HYPRE    : %s",
#ifdef HYPRE_RELEASE_VERSION
              "v" HYPRE_RELEASE_VERSION
#else
              "version unknown"
#endif
  );
#ifdef HYPRE_SEQUENTIAL
  std::printf(", NO MPI (sequential build)");
#else
  std::printf(", MPI");
#endif
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
  std::printf(", CUDA");
#else
  std::printf(", NO CUDA (BoomerAMG runs on the host)");
#endif
#ifdef HYPRE_USING_GPU_AWARE_MPI
  std::printf(", GPU-aware MPI");
#else
  std::printf(", NO GPU-aware MPI");
#endif
#ifdef HYPRE_USING_UNIFIED_MEMORY
  std::printf(", unified memory");
#endif
  std::printf("\n");
#ifndef HYPRE_USING_GPU_AWARE_MPI
  if (nranks > 1)
    std::printf(
        "  [warn] HYPRE has no GPU-aware MPI, so every AMG halo exchange goes\n"
        "         through host memory.  %d ranks may well be SLOWER than 1.\n"
        "         Rebuild HYPRE with -DHYPRE_ENABLE_GPU_AWARE_MPI=ON.\n",
        nranks);
#endif
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Per-report-step convergence table.
//
// One row per report step: the progress bar the CuPy reference printed, plus
// the numbers you need to judge whether the answer is trustworthy — the
// Newton residual at the start and end of the step, the relative reduction
// actually achieved against the tolerance being applied, and the iteration
// counts that produced it.
// ---------------------------------------------------------------------------
void print_step_header(const Deck& d, int nsteps, bool from_dates) {
  // Say where the report times came from.  "%d report steps of 30 days"
  // over a run whose steps are 1 to 92 days apart is a lie that looks
  // like a setting.
  if (from_dates)
    std::printf("\n--- FORWARDING: %d report steps at the deck\'s own dates (to %g days) ---\n",
                nsteps, (double)d.t_end);
  else
    std::printf("\n--- FORWARDING: %d report steps of %g days (to %g days) ---\n",
                nsteps, (double)d.dt_report, (double)d.t_end);
  std::printf("  Newton : tol %.2e (relative), max %d iterations/step\n",
              (double)d.newton_tol, d.newton_max);
  std::string krylov = d.krylov
                           ? std::string("BiCGSTAB")
                           : "FGMRES restart " + std::to_string(d.gmres_restart);
  std::printf("  Linear : tol %.2e, max %d iterations, %s\n",
              (double)d.lin_tol, d.lin_maxit, krylov.c_str());
  if (!d.krylov)
    std::printf("  Arnoldi: CGS%d orthogonalisation (ORTHOPASSES %d)\n",
                std::clamp(d.ortho_passes, 1, 2), d.ortho_passes);
  // Report what the binary actually does, not what the deck asked for.  The
  // preconditioner precision is fixed at *compile* time by PrecReal
  // (-DNVRS_MIXED_PRECISION and the HYPRE build it links against), so the
  // deck's MIXED keyword cannot change it at run time.  Printing the deck
  // value would misreport the configuration a result was produced with.
  const char* amg_prec = (sizeof(PrecReal) == 4) ? "FP32" : "FP64";
  std::printf("  Solver : CPR preconditioner, %s AMG (compile-time), "
              "%s operator\n",
              amg_prec,
              d.emulate_fp64 == 1   ? "df64-emulated FP64"
              : d.emulate_fp64 == 2 ? "FP32-stored (FP64 master for residuals)"
                                    : "native FP64");
  if (d.cpr_decouple != 0)
    std::printf("  CPR    : stage-1 decoupling %s%s\n",
                d.cpr_decouple == 1 ? "dynamic row sum (DRS), DRSEPS "
                                    : "row sum",
                d.cpr_decouple == 1 ? std::to_string((double)d.drs_eps).c_str()
                                    : "");
  if (d.graph_min_reuse > 0)
    std::printf("  Graphs : capture stops below %.2f reuses per capture "
                "(checked after %d captures)\n",
                (double)d.graph_min_reuse, d.graph_guard_after);
  if (!d.dt_seq_file.empty())
    std::printf("  Time   : substeps imposed from %s (DTSEQFILE)\n",
                d.dt_seq_file.c_str());
  std::printf("  Operator: %s (BLOCKSPMV %d), CUDA graphs %s (CUDAGRAPH %d)\n",
              d.block_spmv ? "block storage, one index per block"
                           : "scalar CSR, one index per entry",
              d.block_spmv, d.cuda_graph ? "on for the Arnoldi tail" : "off",
              d.cuda_graph);
  // Reuse changes what the preconditioner *is* from one Newton iterate to the
  // next, so it has to be on the record next to the tolerances rather than
  // discovered from a timing table afterwards.
  if (d.scheme == 1)
    std::printf("  AMG    : rebuilt every solve (SEQUENTIAL alternates two "
                "matrices; nothing to reuse)\n");
  else if (d.amg_reuse > 0)
    std::printf("  AMG    : hierarchy reused for up to %d further Newton "
                "iterations, rebuilt early if Krylov exceeds %.2gx the fresh "
                "count (AMGREUSE)\n",
                d.amg_reuse, (double)d.amg_reuse_growth);
  else
    std::printf("  AMG    : rebuilt every Newton iteration (AMGREUSE 0)\n");
  // Only when it is on, and only where the other settings are listed, so a
  // run that used it says so and a run that did not stays unchanged.
  if (d.dt_relax > 1)
    std::printf("  dt     : a comfortable step recovers up to %d halving(s), "
                "target %d Newton/sub-step (DTRELAX/DTTARGET)\n",
                d.dt_relax, d.dt_target);
  std::printf("  Stage 2: %s\n",
              d.stage2 == 1
                  ? "ZLINE vertical block-tridiagonal line relaxation, "
                    "refactored every Newton iteration (STAGE2 ZLINE)"
              : d.stage2 == 2
                  ? "Ginkgo adaptive-precision block-Jacobi on each cell "
                    "block, FP64/FP32/FP16 per block (STAGE2 GINKGO)"
                  : "block-Jacobi on each cell block (STAGE2 JACOBI)");
  if (d.appleyard)
    std::printf("  Newton : modified Appleyard chop at the relperm endpoints "
                "(APPLEYARD 1)\n");
  if (d.pv_switch)
    std::printf("  Newton : black-oil u[2] switches Sg <-> Rs at phase "
                "appearance/disappearance (PVSWITCH 1)\n");
  if (d.line_search > 0)
    std::printf("  Newton : residual line search, up to %d alpha halving(s) "
                "(LINESEARCH)\n", d.line_search);
  if (d.line_best)
    std::printf("  Newton : retain the best finite trial when strict line "
                "descent is unavailable (LINEBEST 1)\n");
  if (d.lin_forcing_max > d.lin_tol)
    std::printf("  Linear : adaptive forcing %.2e .. %.2e "
                "(LINTOL/LINFORCINGMAX)\n",
                (double)d.lin_tol, (double)d.lin_forcing_max);
  // The scheme decides which equations are solved implicitly, so it belongs
  // on every run's record next to the tolerances, in both directions.
  if (d.aim)
    std::printf("  Scheme : ADAPTIVE IMPLICIT (AIM) — a cell runs explicit in "
                "saturation below throughput CFL %.2g; wells and fronts stay "
                "implicit.  AIM 0 for fully implicit.\n",
                (double)d.aim_cfl);
  else
    std::printf("  Scheme : FULLY IMPLICIT (FIM) — every cell implicit in "
                "pressure and saturation.  AIM 1 for adaptive implicit.\n");
  const bool deck_wants_fp32 = (d.mixed_precision != 0);
  const bool build_is_fp32 = (sizeof(PrecReal) == 4);
  if (deck_wants_fp32 != build_is_fp32)
    std::printf("  [warn] deck sets MIXED %d but this binary has a %s "
                "preconditioner, which is what this run uses. The precision "
                "is compile-time: run %s instead (built by ./nvrs.sh install, "
                "or make%s); the launch scripts pick it from MIXED.\n",
                d.mixed_precision, amg_prec,
                deck_wants_fp32 ? "nvrs_cuda" : "nvrs_cuda_fp64",
                deck_wants_fp32 ? "" : " FP64_PRECOND=1");
  std::printf(
      "\n  %-9s %8s  %11s  %11s  %10s %4s %6s %7s  %s\n",
      "step", "time[d]", "|R|_start", "|R|_end", "rel", "newt", "krylov",
      "wall[s]", "progress");
  std::printf("  %s\n", std::string(96, '-').c_str());
  std::fflush(stdout);
}

void report_progress(int step, int nsteps, Real t_days, Real r_start,
                     Real r_end, Real tol, int newton, int krylov,
                     double elapsed, bool step_ok, const char* how) {
  const int width = 20;
  const double frac = (nsteps > 0) ? (double)step / nsteps : 1.0;
  const int filled = (int)(frac * width + 0.5);
  std::string bar(filled, '#');
  bar.append(width - filled, '-');

  const double rel = (r_start > 0) ? (double)(r_end / r_start) : 0.0;

  // Report whether the STEP converged, and on which test -- not whether the
  // ratio alone happened to clear newton_tol.  A step converged on material
  // balance and CNV is converged; calling it "not converged" because the ratio
  // is 1e-2 describes a criterion the run is no longer using, and it made a
  // finished nine-year history look like a failure on every line.
  (void)tol;
  const char* tail = "";
  if (!step_ok) tail = "  <-- NOT CONVERGED";
  else if (how && how[0] && std::strcmp(how, "rel") != 0) tail = "  [mb+cnv]";

  // "110/247", not "110".  A bare step number says nothing about how much
  // is left, and on a deck whose report count is read from the schedule
  // rather than set in the deck, it is also the quickest way to see that
  // the run is stepping where you think it is.
  char pos[24];
  std::snprintf(pos, sizeof(pos), "%d/%d", step, nsteps);
  std::printf("  %-9s %8.1f  %11.4e  %11.4e  %10.3e %4d %6d %7.1f  [%s] %3.0f%%%s\n",
              pos, (double)t_days, (double)r_start, (double)r_end, rel, newton,
              krylov, elapsed, bar.c_str(), frac * 100.0, tail);
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Debug: summarise a device array (max magnitude and non-finite count).
//
// When a solve returns NaN the first question is whether the *input* was
// already bad.  Copying each piece of the linear system to the host and
// scanning it answers that directly, and separates "the assembly produced a\n// poisoned matrix" from "the assembly was clean and the solver diverged".
// Only reachable under NVRS_VERBOSE, so the host round-trip costs nothing in a
// normal run.
// ---------------------------------------------------------------------------
// NVRS_RESLOG=<dir>: one CSV row per Newton evaluation (newton.csv) and per
// linear solve (linear.csv), rank 0 only.  The values are already on the host,
// so the cost is one fprintf per iteration; used to compare the convergence of
// the mixed-precision and all-FP64 builds.
// ---------------------------------------------------------------------------
static FILE* reslog_file(int which) {
  static FILE* f[2] = {nullptr, nullptr};
  static bool tried[2] = {false, false};
  if (!tried[which]) {
    tried[which] = true;
    if (const char* d = std::getenv("NVRS_RESLOG")) {
      const std::string p = std::string(d) + (which ? "/linear.csv" : "/newton.csv");
      f[which] = std::fopen(p.c_str(), "w");
      if (f[which])
        std::fputs(which ? "step,sub,newton,lin_iters,lin_rel,converged,du_norm\n"
                         : "step,sub,newton,rnorm,rel,mb,cnv,accepted\n",
                   f[which]);
    }
  }
  return f[which];
}

template <typename T>
void scan_device_array(const T* dev, size_t n, const char* name) {
  if (!dev || n == 0) {
    std::printf("      %-8s : (absent)\n", name);
    return;
  }
  std::vector<T> h(n);
  if (cudaMemcpy(h.data(), dev, sizeof(T) * n, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    std::printf("      %-8s : (copy failed)\n", name);
    return;
  }
  double mx = 0.0, mn = 1e308;
  size_t bad = 0, zero = 0;
  for (size_t i = 0; i < n; ++i) {
    const double v = (double)h[i];
    if (!std::isfinite(v)) { ++bad; continue; }
    const double a = std::fabs(v);
    if (a == 0.0) { ++zero; continue; }
    mx = std::max(mx, a);
    mn = std::min(mn, a);
  }
  std::printf("      %-8s : n=%-9zu max|.|=%.4e  min|.|=%.4e  zeros=%zu"
              "  NON-FINITE=%zu%s\n",
              name, n, mx, (mn > 1e307 ? 0.0 : mn), zero, bad,
              bad ? "   <-- poisoned" : "");
}

void dump_system(const DeviceLinearSystem& sys, int step, int nw) {
  std::printf("    linear system at step %d, newton %d:\n", step, nw);
  scan_device_array(sys.aval64, (size_t)sys.nnz, "aval64");
  scan_device_array(sys.b, (size_t)sys.n, "b");
  scan_device_array(sys.pval, (size_t)sys.pnnz, "pval");
  scan_device_array(sys.dinv, (size_t)sys.cells * sys.bs * sys.bs, "dinv");
  scan_device_array(sys.aval_hi, (size_t)sys.nnz, "aval_hi");
  scan_device_array(sys.aval_lo, (size_t)sys.nnz, "aval_lo");
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Bind this rank to a GPU, before HYPRE is initialised.
//
// Three things matter here and all three were wrong in the original two-liner
// (`cudaGetDeviceCount(&ndev); if (ndev > 0) cudaSetDevice(rank % ndev);`):
//
//   1. The *node-local* rank selects the device, not the global rank.  With
//      `rank % ndev` a 2-node x 4-GPU job sends global ranks 4..7 to devices
//      0..3 only by luck of the arithmetic, and any other rank/GPU ratio binds
//      several ranks to one GPU while leaving others idle.
//   2. Both CUDA calls must be checked.  Silently ignoring them means a failed
//      bind surfaces much later as an inscrutable error from deep inside
//      HYPRE's thrust calls ("exclusive_scan: invalid device ordinal") rather
//      than at the point of failure.
//   3. cudaSetDevice is lazy: it records an intent but creates no context.
//      HYPRE_Init() (and the thrust/CUB machinery underneath BoomerAMG) then
//      initialises against whatever device it finds current.  Forcing context
//      creation here with cudaFree(0) makes the binding take effect *before*
//      HYPRE looks, which is what the HYPRE GPU documentation requires.
//
// Returns the bound device ordinal.  Set NVRS_DEVICE to override the choice.
int bind_gpu(int rank, int nranks) {
  // Node-local rank: split the world communicator by shared memory.
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &node_comm);
  int local_rank = 0, local_size = 1;
  MPI_Comm_rank(node_comm, &local_rank);
  MPI_Comm_size(node_comm, &local_size);
  MPI_Comm_free(&node_comm);

  // Cross-check the split against host names.  Under HPC-X in the cluster
  // container the shared-memory split has put every rank of a single node in
  // a group of its own: four ranks each reported local_rank=0/1, all four bound
  // GPU 0 of 4, and the 5M run took 64.8 s on one GPU, 95.3 s on "two" and
  // 153.1 s on "four".  Counting the ranks that share this rank's host name
  // does not depend on the MPI's shared-memory detection, so it wins whenever
  // it finds more ranks on the node than the split did.
  char host[MPI_MAX_PROCESSOR_NAME] = {};
  int hostlen = 0;
  MPI_Get_processor_name(host, &hostlen);
  std::vector<char> hosts((size_t)nranks * MPI_MAX_PROCESSOR_NAME, 0);
  MPI_Allgather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts.data(),
                MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
  int host_rank = 0, host_size = 0;
  for (int q = 0; q < nranks; ++q) {
    if (std::strncmp(&hosts[(size_t)q * MPI_MAX_PROCESSOR_NAME], host,
                     MPI_MAX_PROCESSOR_NAME) != 0)
      continue;
    if (q < rank) ++host_rank;
    ++host_size;
  }
  bool by_host = false;
  if (host_size > local_size) {
    local_rank = host_rank;
    local_size = host_size;
    by_host = true;
  }

  int ndev = 0;
  cudaError_t err = cudaGetDeviceCount(&ndev);
  if (err != cudaSuccess || ndev <= 0) {
    std::ostringstream os;
    os << "no usable CUDA device visible to rank " << rank << " ("
       << (err == cudaSuccess ? "device count is 0" : cudaGetErrorString(err))
       << "). Check CUDA_VISIBLE_DEVICES and that the container exposes GPUs.";
    throw std::runtime_error(os.str());
  }

  int dev = local_rank % ndev;
  if (const char* env = std::getenv("NVRS_DEVICE")) dev = std::atoi(env);
  if (dev < 0 || dev >= ndev) {
    std::ostringstream os;
    os << "rank " << rank << " cannot bind device " << dev << ": only " << ndev
       << " device(s) visible (ordinals 0.." << ndev - 1 << ").";
    throw std::runtime_error(os.str());
  }

  NVRS_CHECK_CUDA(cudaSetDevice(dev));
  // Force context creation now, so the binding is real before HYPRE_Init().
  NVRS_CHECK_CUDA(cudaFree(nullptr));

  int actual = -1;
  NVRS_CHECK_CUDA(cudaGetDevice(&actual));
  cudaDeviceProp prop{};
  NVRS_CHECK_CUDA(cudaGetDeviceProperties(&prop, actual));

  // One line per rank: with a multi-GPU job this is the first thing you want
  // to see, and it is what turns a binding problem into a two-second diagnosis.
  std::printf("[rank %d/%d] %s local_rank=%d/%d%s -> GPU %d of %d (%s, sm_%d%d)\n",
              rank, nranks, host, local_rank, local_size,
              by_host ? " (by host name; MPI's shared-memory split saw fewer)"
                      : "",
              actual, ndev, prop.name, prop.major, prop.minor);
  std::fflush(stdout);

  // Two ranks on one host bound to one GPU is never intended here unless
  // NVRS_DEVICE asked for it, and it is silent otherwise: the run completes,
  // just slower with every rank added.
  std::vector<int> devs(nranks, -1);
  MPI_Allgather(&actual, 1, MPI_INT, devs.data(), 1, MPI_INT, MPI_COMM_WORLD);
  if (rank == 0 && !std::getenv("NVRS_DEVICE") && nranks > 1) {
    int shared = 0;
    for (int a = 0; a < nranks; ++a)
      for (int b = a + 1; b < nranks; ++b)
        if (devs[a] == devs[b] &&
            std::strncmp(&hosts[(size_t)a * MPI_MAX_PROCESSOR_NAME],
                         &hosts[(size_t)b * MPI_MAX_PROCESSOR_NAME],
                         MPI_MAX_PROCESSOR_NAME) == 0)
          ++shared;
    if (shared > 0 && local_size <= ndev)
      std::printf("[warn] %d pair(s) of ranks on one host share a GPU although "
                  "the host has enough; multi-GPU timings from this run are "
                  "not a scaling measurement.\n", shared);
    std::fflush(stdout);
  }

  if (local_size > ndev && rank == 0)
    std::printf(
        "[warn] %d ranks share %d GPU(s) on this node; ranks will oversubscribe "
        "devices. Use -np %d, or enable MPS.\n",
        local_size, ndev, ndev);

  return actual;
}

// DUMPOP: write the coupled operator and the CPR pressure matrix to
// OUTDIR/op_J.bin and OUTDIR/op_Ap.bin.  Little-endian, no padding:
//
//   op_J.bin : int32 n, n_ext, nnz, bs, cells
//              int32 rowptr[n+1], int32 colind[nnz], float64 aval[nnz]
//   op_Ap.bin: int32 cells, pnnz, prec_bytes (4 = FP32 values, 8 = FP64)
//              int32 prowptr[cells+1], int32 pcolind[pnnz],
//              prec  pval[pnnz], prec pweight[cells*bs]
//
// tests/bspmv_bench.cu reads op_J.bin; scripts/cond_estimate.py reads both.
template <class T>
std::vector<T> device_copy(const T* d, size_t n) {
  std::vector<T> h(n);
  if (d && n) NVRS_CHECK_CUDA(cudaMemcpy(h.data(), d, sizeof(T) * n,
                                         cudaMemcpyDeviceToHost));
  return h;
}
void dump_operator(const std::string& dir, const DeviceLinearSystem& sys,
                   int step, int newton) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(dir, ec);
  const auto rp = device_copy(sys.rowptr, (size_t)sys.n + 1);
  const auto ci = device_copy(sys.colind, (size_t)sys.nnz);
  const auto av = device_copy(sys.aval64, (size_t)sys.nnz);
  {
    std::ofstream f(dir + "/op_J.bin", std::ios::binary);
    const int32_t h[5] = {sys.n, sys.n_ext, sys.nnz, sys.bs, sys.cells};
    f.write((const char*)h, sizeof(h));
    f.write((const char*)rp.data(), sizeof(int) * rp.size());
    f.write((const char*)ci.data(), sizeof(int) * ci.size());
    f.write((const char*)av.data(), sizeof(double) * av.size());
  }
  if (sys.prowptr && sys.pval && sys.pnnz > 0) {
    const auto prp = device_copy(sys.prowptr, (size_t)sys.cells + 1);
    const auto pci = device_copy(sys.pcolind, (size_t)sys.pnnz);
    const auto pv = device_copy(sys.pval, (size_t)sys.pnnz);
    const auto pw = device_copy(sys.pweight, (size_t)sys.cells * sys.bs);
    std::ofstream f(dir + "/op_Ap.bin", std::ios::binary);
    const int32_t h[3] = {sys.cells, sys.pnnz, (int32_t)sizeof(PrecReal)};
    f.write((const char*)h, sizeof(h));
    f.write((const char*)prp.data(), sizeof(int) * prp.size());
    f.write((const char*)pci.data(), sizeof(int) * pci.size());
    f.write((const char*)pv.data(), sizeof(PrecReal) * pv.size());
    f.write((const char*)pw.data(), sizeof(PrecReal) * pw.size());
  }
  std::printf("  [dumpop] step %d, Newton %d: %d rows, %d nonzeros, %d "
              "pressure nonzeros -> %s/op_J.bin, op_Ap.bin\n",
              step, newton, sys.n, sys.nnz, sys.pnnz, dir.c_str());
}

}  // namespace

// The horizontal rule the start-up reports are separated by.  Same width as
// the deck summary's own box, so the whole preamble reads as one document of
// several sections rather than a wall of text.
static const char* const kSectionRule =
    "======================================================\n";

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  bool jac_check = false;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--jac-check") jac_check = true;

  if (argc < 2) {
    if (rank == 0)
      std::cerr << "usage: " << argv[0] << " <deck.DATA> [--jac-check]\n"
                << "  mpirun -np <ranks> " << argv[0] << " deck.DATA\n"
                << "  --jac-check : compare analytic vs numerical Jacobian and exit\n";
    MPI_Finalize();
    return 1;
  }

  {
    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int hlen = 0;
    MPI_Get_processor_name(host, &hlen);
    if (rank == 0) print_banner(nranks, host, peek_backend(argv[1]));
  }

  // Bind one GPU per rank *before* HYPRE_Init — see bind_gpu().  A binding
  // failure here is fatal and must abort the whole job: if one rank dies
  // outside a collective the others hang forever in the next MPI call, which
  // is what turns a clear error into an apparent freeze.
  try {
    bind_gpu(rank, nranks);
  } catch (const std::exception& e) {
    std::cerr << "[rank " << rank << "] GPU binding failed: " << e.what()
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 2;
  }

  HYPRE_Init();
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

  // Which kernels HYPRE uses underneath, which is not the same question as
  // where its memory lives.  Left alone, HYPRE takes its own defaults; on a
  // CUDA build that means routing SpMV and the RAP triple products through
  // cuSPARSE.
  //
  // That is the wrong trade here.  The measured cost is a fixed charge per
  // level-operation -- about 62 microseconds on one rank and 186 on two, at
  // every level including the one holding half a million rows -- rather than
  // anything proportional to the work.  A vendor SpMV carries per-call setup,
  // and a parallel ParCSR matvec makes two of those calls per operation
  // instead of one, because the off-diagonal block only exists once the matrix
  // is distributed.  On a single rank there is no off-diagonal block at all,
  // which is exactly why none of this was visible until a second GPU appeared.
  //
  // Aramco's GPU BoomerAMG benchmark, which does scale across ranks, turns
  // both vendor paths off and uses HYPRE's own kernels.  Same defaults here,
  // with HYPRESPMV / HYPRESPGEMM to put them back for a comparison.
  //
  // HYPRE_SetGpuAwareMPI is a *runtime* switch.  The banner has only ever
  // reported the build-time macro, which says what HYPRE can do, not what it
  // is doing.
  HYPRE_SetSpMVUseVendor(nvrs_env_int("NVRS_HYPRE_SPMV", 0));
  HYPRE_SetSpGemmUseVendor(nvrs_env_int("NVRS_HYPRE_SPGEMM", 0));
#ifdef HYPRE_USING_GPU_AWARE_MPI
  HYPRE_SetGpuAwareMPI(nvrs_env_int("NVRS_HYPRE_GPUMPI", 1));
#endif
  // Only meaningful when BoomerAMG is the stage-1 multigrid; an AmgX run
  // never enters a HYPRE kernel, so reporting its switches would misdescribe
  // the run.  (The calls above are harmless either way and stay unconditional
  // so a deck that selects HYPRE mid-session is still correctly configured.)
  if (rank == 0 && banner_uses_hypre(peek_backend(argv[1])))
    std::printf("  HYPRE opt: SpMV vendor %d, SpGemm vendor %d, GPU-aware MPI "
                "%d (NVRS_HYPRE_SPMV / _SPGEMM / _GPUMPI)\n",
                nvrs_env_int("NVRS_HYPRE_SPMV", 0),
                nvrs_env_int("NVRS_HYPRE_SPGEMM", 0),
                nvrs_env_int("NVRS_HYPRE_GPUMPI", 1));

  int rc = 0;
  try {
    Deck deck = parse_deck(argv[1]);
    // AIM freezes explicit donors' mobilities inside the residual, so the
    // JACOBIAN TENSORCORE: the batched operator product is built and
    // selectable, but the per-connection operators L_props and L_flux are not
    // yet filled from this simulator's PVT evaluation, so the path cannot
    // produce a Jacobian here.  Rather than assemble something that looks
    // plausible and is wrong, say exactly what is missing and run the
    // assembly that is trusted.  See include/tc_jacobian.hpp.
    if (deck.jacobian == 2) {
      if (rank == 0)
        std::printf("  [warn] JACOBIAN TENSORCORE: the batched operator "
                    "product (%s) is built, but the operator fill from the "
                    "PVT tables is not implemented yet; assembling with "
                    "JACOBIAN ANALYTIC instead.
",
                    nvrs::tc::tc_backend_name());
      deck.jacobian = 1;
    }

    // Jacobian must be the analytic one that applies the same rule; the
    // numerical fallback would differentiate a residual it does not compute.
    if (deck.aim && deck.jacobian != 1) {
      if (rank == 0 && deck.aim_explicit)
        std::printf("  [warn] AIM needs JACOBIAN ANALYTIC (it freezes donor "
                    "mobilities inside the residual, and the numerical "
                    "Jacobian would differentiate a different one); running "
                    "fully implicit instead.\n");
      deck.aim = 0;
    }
    if (deck.aim && deck.model == Model::Compositional) {
      if (rank == 0 && deck.aim_explicit)
        std::printf("  [warn] AIM has no compositional rule yet; running "
                    "fully implicit instead.\n");
      deck.aim = 0;
    }
    if (rank == 0) print_deck_summary(deck);

    // --- the field-deck path -----------------------------------------------
    // FIELDDECK names a corner-point deck.  When present the grid, the wells
    // and the initial state all come from it, and this deck supplies only the
    // fluid and solver settings.  Loading it up front means the rest of the
    // run is identical either way: the same Reservoir, the same linear system,
    // the same Newton loop.
    std::unique_ptr<FieldModel> field;
    if (!deck.field_deck.empty()) {
      const Real grad[3] = {deck.grad_oil, deck.grad_water, deck.grad_gas};
      // Every rank reads the whole deck and computes the same partition, so
      // the communication lists agree without a negotiation round.  See
      // field_partition.hpp.
      field = load_field_model(deck.field_deck,
                               deck.have_gradients ? grad : nullptr,
                               deck.connate_water, rank, nranks,
                               deck.props_arrays, deck.trans_drop_rel);
      if (rank == 0) {
        // A rule between sections: the grid, the schedule and the solver input
        // are three separate reports and run together without one.
        std::printf("\n%s%s", kSectionRule, field->grid.summary().c_str());
        std::printf("%s%s", kSectionRule, field->schedule.summary().c_str());
        if (!deck.have_gradients)
          std::printf("  [warn] no GRADIENTS given: the initial state falls "
                      "back to PINIT/SWINIT rather than equilibrating.\n");

        // Where the initial water came from.  With capillary pressure in the
        // flux this is not a detail: a sharp contact and a deck-supplied
        // transition zone differ by the whole of Pcow across the contact.
        if (field->have_initial) {
          const auto& ini = field->initial;
          if (ini.n_swatinit > 0)
            std::printf("  [equil] SWATINIT used for %d of %d cell(s), Sw %.4f .. %.4f\n", ini.n_swatinit, (int)ini.sw.size(),
                        (double)ini.swatinit_lo, (double)ini.swatinit_hi);
          else
            std::printf("  [equil] no SWATINIT: sharp contacts, connate water above the OWC and Sw = 1 below\n");
        }
      }
      if (nranks > 1) {
        // Every rank reports its own share: an unstructured partition does not
        // balance perfectly, and the spread is worth seeing rather than
        // inferring from the timings later.
        for (int q = 0; q < nranks; ++q) {
          MPI_Barrier(MPI_COMM_WORLD);
          if (q == rank) std::printf("%s", field->part.summary().c_str());
          std::fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    Grid grid = Grid::make(deck.nx, deck.ny, deck.nz, deck.dx, deck.dy, deck.dz,
                           rank, nranks);

    // Collective, and it must run before anything else touches the halo.
    if (!field) halo_probe(grid);

    // Equinor's own saturation functions, in place of the run deck's
    // placeholder.  Before make_reservoir_field, which is where the tables are
    // uploaded -- the PVT read further down happens after that and is too late
    // for these.
    //
    // The column order is the trap.  ECLIPSE writes SWOF as Sw krw krow and
    // SGOF as Sg krg krog; this simulator's own deck format puts the OIL curve
    // second.  Reading Norne's file with the run deck's convention would swap
    // krw with krow -- silently, since both are plausible monotone curves --
    // so the swap is done here, once, in the open.
    if (field && deck.field_satfunc != 0) {
      PvtTables satsrc;
      try {
        satsrc.read(deck.field_deck);
      } catch (const std::exception&) {
      }
      // Refuse a table that cannot be right, rather than simulate it.
      //
      // The four corner values must each be the LARGE end of their curve.  A
      // parse that goes wrong does not produce noise, it produces a smooth
      // curve of the wrong shape: reading Norne's nineteen rows as two gave
      // krw = 0 across the whole range, so water was immobile everywhere, and
      // the run converged in 45 s and reported zero water production, zero
      // water cut and a water balance flat against Flow's 80 Mrb.  Every
      // panel of the comparison was wrong and nothing had failed.
      const auto& w = satsrc.swof();
      const auto& g = satsrc.sgof();
      const bool need_gas = deck.model == Model::BlackOil3P;
      bool sane = w.size() >= 3 &&
                  w.back().kr > (Real)1e-6 &&    // krw at maximum water
                  w.front().kro > (Real)1e-6;    // krow at connate water
      if (need_gas)
        sane = sane && g.size() >= 3 &&
               g.back().kr > (Real)1e-6 &&       // krg at maximum gas
               g.front().kro > (Real)1e-6;       // krog at zero gas
      if (!sane && rank == 0)
        std::printf("  [satfunc] REFUSED field saturation tables: SWOF %zu "
                    "rows, SGOF %zu rows; model needs %s. Falling back to "
                    "the run deck's curves.\n", w.size(), g.size(),
                    need_gas ? "both valid curves" : "a valid SWOF");
      if (sane) {
        // Capillary pressure is a pressure and follows the deck's units.
        const Real pc_scale =
            (field->input_units == Units::Metric) ? (Real)14.503773773 : (Real)1;
        const size_t nw_old = deck.swof.size(), ng_old = deck.sgof.size();
        deck.swof.clear(); deck.swof_pcow.clear();
        for (const auto& r : satsrc.swof()) {
          deck.swof.push_back({r.s, r.kro, r.kr});   // -> Sw krow krw
          deck.swof_pcow.push_back(r.pc * pc_scale);
        }
        deck.sgof.clear(); deck.sgof_pcog.clear();
        for (const auto& r : satsrc.sgof()) {
          deck.sgof.push_back({r.s, r.kro, r.kr});   // -> Sg krog krg
          deck.sgof_pcog.push_back(r.pc * pc_scale);
        }
        if (rank == 0) {
          Real pcw_lo = 0, pcw_hi = 0;
          for (Real v : deck.swof_pcow) {
            if (v < pcw_lo) pcw_lo = v;
            if (v > pcw_hi) pcw_hi = v;
          }
          std::printf("  [satfunc] using the FIELD deck's own curves: SWOF "
                      "%zu rows (was %zu), SGOF %zu rows (was %zu); Pcow "
                      "%.4f .. %.4f psi\n",
                      deck.swof.size(), nw_old, deck.sgof.size(), ng_old,
                      (double)pcw_lo, (double)pcw_hi);
          if (need_gas)
            std::printf("  [satfunc] krw/krow/krg/krog large endpoints: "
                        "%.4f %.4f %.4f %.4f\n",
                        (double)deck.swof.back()[2],
                        (double)deck.swof.front()[1],
                        (double)deck.sgof.back()[2],
                        (double)deck.sgof.front()[1]);
          else
            std::printf("  [satfunc] krw(Sw_max) %.4f, krow(Sw_min) %.4f "
                        "(both must be the large end)\n",
                        (double)deck.swof.back()[2],
                        (double)deck.swof.front()[1]);
        }
        if (satsrc.have_ehystr()) {
          deck.use_hysteresis = satsrc.ehystr_kr_model() == 0;
          if (rank == 0 && deck.use_hysteresis)
            std::printf("  [satfunc] EHYSTR HYSTMOD=0: Carlson oil/water + "
                        "gas/oil non-wetting relperm hysteresis on "
                        "(IMBNUM/I-endpoint scaling)\n");
          else if (rank == 0)
            std::printf("  [satfunc] WARNING: EHYSTR HYSTMOD=%d is not "
                        "implemented; hysteresis disabled\n",
                        satsrc.ehystr_kr_model());
        }
      } else if (rank == 0) {
        std::printf("  [satfunc] the field deck carries no SWOF/SGOF; keeping "
                    "the run deck's tables\n");
      }
    }

    Reservoir res = field ? make_reservoir_field(deck, *field)
                          : make_reservoir(deck, grid);
    res.dt = deck.dt_report;   // [days]

    // Tabulated PVT, read from the field deck itself.
    //
    // No new keyword: PVTO, PVTG, PVTW and DENSITY live in the PROPS section
    // of the deck FIELDDECK already names, reached through the same INCLUDE
    // chain the grid came down.  A field deck that carries tables should not
    // have to be told twice that it does, and a run should not quietly fall
    // back to four constants when the real curves were sitting in the file.
    //
    // PVTCONST 1 in the run deck overrides this and keeps the constants, which
    // is what makes a table-versus-constant comparison a one-line edit.
    PvtTables pvt_host;
    if (field && deck.pvt_tables != 0) {
      try {
        pvt_host.read(deck.field_deck);
      } catch (const std::exception& e) {
        if (rank == 0)
          std::printf("  [warn] PVT tables not read (%s); falling back to the "
                      "deck's constant B and Rs.\n", e.what());
      }
      const bool have_fluid_tables = pvt_host.have_pvto() ||
                                     pvt_host.have_pvtg();
      const bool have_property_data = have_fluid_tables ||
                                      pvt_host.have_pvtw() ||
                                      pvt_host.have_density() ||
                                      pvt_host.have_rock();
      if (have_property_data) {
        // The run works in the units to_field_units() put the grid in, so the
        // tables must be moved the same way before they are uploaded.
        if (field->input_units == Units::Metric) pvt_host.to_field_units();
        else                                     pvt_host.set_field_units(true);
        // VAPOIL 0 removes vaporised oil.  Set before any region is
        // flattened or uploaded, so every region carries the same value and
        // the host evaluator the report uses agrees with the kernel.
        pvt_host.set_rv_scale(deck.vapoil ? (Real)1 : (Real)0);
        if (!deck.vapoil)
          std::printf("  VAPOIL 0: vaporised oil removed; Rv is zero "
                      "everywhere and the gas phase carries no stock-tank "
                      "oil.\n");
        // ROCK, if the field deck gave it.  With tables present `ct` means
        // rock compressibility alone, which is what ROCK states, so the field
        // deck's own number replaces the deck default rather than sitting
        // beside it unused.  Said out loud: a compressibility that silently
        // changed would be very hard to notice in a result.
        if (pvt_host.have_rock()) {
          if (rank == 0)
            std::printf("  [rock] ROCK: compressibility %.3e /psi at %.1f psi "
                        "(replaces CT %.3e)\n",
                        (double)pvt_host.rock_cr(),
                        (double)pvt_host.rock_pref(), (double)res.ct);
          res.ct = pvt_host.rock_cr();
          res.rock_pref = pvt_host.rock_pref();
          res.have_rock = 1;
        }

        if (have_fluid_tables) res.pvt = upload_pvt(pvt_host, 0);
        // The undersaturated anchor is the deck's own RS keyword, not r.rs.
        // Those differ: with PVTCONST clear, r.rs is the built-in correlation
        // -- about 5650 SCF/STB against Norne's 674 -- and an anchor that high
        // never binds, which pins every cell to the saturated line and gives
        // the oil a negative compressibility.
        res.rs_max = (deck.rs_const > 0) ? deck.rs_const : res.rs;

        // PVTNUM: every region on the device, and the region each cell
        // belongs to.  A one-region deck lands on region 0 everywhere
        // and behaves exactly as it did through res.pvt.
        if (have_fluid_tables) {
          int nreg = 0;
          res.pvt_regions = upload_pvt_regions(pvt_host, &nreg);
          res.n_pvt_regions = nreg;
          const std::vector<Real>* pn = field->grid.array("PVTNUM");
          const int ncells = field->part.n_local_ext();
          std::vector<int> reg(ncells, 0);
          int nbad = 0;
          for (int c = 0; c < ncells; ++c) {
            int r0 = 0;
            if (pn && c < (int)field->part.active_of_local.size()) {
              const int a = field->part.active_of_local[c];
              const int gidx = (a >= 0 && a < field->grid.n_active())
                                   ? field->grid.global_of_active(a) : -1;
              if (gidx >= 0 && gidx < (int)pn->size())
                r0 = (int)std::lround((*pn)[gidx]) - 1;  // 1-based
            }
            if (r0 < 0 || r0 >= nreg) { r0 = 0; ++nbad; }
            reg[c] = r0;
          }
          NVRS_CHECK_CUDA(cudaMalloc(&res.pvt_cell, sizeof(int) * ncells));
          NVRS_CHECK_CUDA(cudaMemcpy(res.pvt_cell, reg.data(),
                                     sizeof(int) * ncells,
                                     cudaMemcpyHostToDevice));
          if (rank == 0) {
            std::vector<int> count(nreg, 0);
            for (int r0 : reg) ++count[r0];
            std::printf("  PVTNUM   : %d region(s);", nreg);
            for (int q = 0; q < nreg; ++q)
              std::printf(" %d:%d cell(s)", q + 1, count[q]);
            if (nbad) std::printf("  (%d out of range -> region 1)", nbad);
            std::printf("\n");
          }
        }
        if (rank == 0) {
          std::printf("\n%s%s", kSectionRule, pvt_host.summary().c_str());
          // Prove the tables reached the kernel, rather than assume it.
          // Reading a table, uploading it and USING it are three claims, and
          // the summary above only supports the first.  This runs the kernel's
          // own evaluator on the device, at the initial reservoir pressure,
          // and prints what came back beside the deck's constants.  If the
          // b-factors come back as 1.000 the tables are not live however
          // complete the summary looks.
          const Real p_ref = 0.5 * (Real)(3774.0 + 4544.0);
          const PvtCell dp = probe_device_pvt(res.pvt, p_ref, res.rs);
          std::printf("  in use on the device, at %.0f psi:\n", (double)p_ref);
          std::printf("    valid           : %s\n",
                      res.pvt.valid ? "yes" : "NO -- running on constants");
          std::printf("    Bo %7.4f (deck %7.4f)   muo %6.4f (deck %6.4f)\n",
                      (double)(dp.bo > 0 ? 1.0 / dp.bo : 0.0), (double)res.bo,
                      (double)dp.muo, (double)res.uo);
          std::printf("    Bg %.3e (deck %.3e)   mug %6.4f (deck %6.4f)\n",
                      (double)(dp.bg > 0 ? 1.0 / dp.bg : 0.0), (double)res.bg,
                      (double)dp.mug, (double)res.ug);
          std::printf("    Rs %7.1f (deck %7.1f) scf/stb,  dRs/dp %.4g\n",
                      (double)dp.rs, (double)res.rs, (double)dp.drs);
          std::printf("    Rv %.4e stb/scf, dRv/dp %.4g  (VAPOIL)\n",
                      (double)dp.rv, (double)dp.drv);
          std::printf("    compressibility : oil %.3e  water %.3e  gas %.3e "
                      "/psi   (deck CT %.3e)\n",
                      (double)(dp.bo > 0 ? dp.dbo / dp.bo : 0.0),
                      (double)(dp.bw > 0 ? dp.dbw / dp.bw : 0.0),
                      (double)(dp.bg > 0 ? dp.dbg / dp.bg : 0.0),
                      (double)res.ct);
          // PVTNUM is honoured now: the per-region cell counts are
          // printed above, where they are actually computed.
        }
      } else if (rank == 0) {
        std::printf("  [note] no PVTO/PVTG in %s: running on the deck's "
                    "constant B and Rs.\n", deck.field_deck.c_str());
      }
    }

    // The dissolved-gas reference state depends on the uploaded PVTO tables:
    // oil in a free-gas cell is saturated, whereas oil-only cells retain their
    // RSVD charge.  Initialise it only after PVTNUM and all device tables exist.
    drsdt_initialize(res);
    // PVSWITCH changes what u[2] MEANS, cell by cell: Sg, Rs or Rv.  Only the
    // connection-list assembler reads it that way, and that is the FIELDDECK
    // path with the analytic three-phase Jacobian.  The structured Cartesian
    // kernel and the finite-difference fallback still read u[2] as Sg, so
    // switching on one of those would hand them Rs and Rv as saturations.
    // use_conn_list is not set until build_sparsity_field below, hence the
    // test on the path itself.
    if (res.pv_switch && !(field && res.jacobian == 1 && res.bs == 3)) {
      if (rank == 0)
        std::printf("  [warn] PVSWITCH needs a corner-point FIELDDECK run with "
                    "the analytic three-phase Jacobian; the structured "
                    "assembler reads u[2] as Sg.  Ignored.\n");
      res.pv_switch = 0;
    }
    // The coupled well block is built from FIELDDECK schedule wells.  A deck
    // without one never uploads an entry, so the setting is inert; say so
    // rather than let it look as if it had been applied.
    if (deck.well_schur && !field && rank == 0)
      std::printf("  [note] WELLSCHUR couples FIELDDECK schedule wells; this "
                  "deck has none, so it has no effect.\n");
    blackoil_pvswitch_initialize(res);

    DeviceLinearSystem sys;
    sys.bs = res.bs;
    if (field) build_sparsity_field(res, sys, *field);
    else       build_sparsity(res, sys);

    // What the solver actually received.
    //
    // Everything below is cheap and none of it is guesswork, which is the
    // point: a field model that assembles to nothing looks exactly like one
    // that converged instantly, and the run ends in a fraction of a second
    // either way. These are the quantities that separate the two -- if the
    // saturation tables are empty every mobility is zero, if the
    // transmissibilities are zero there is no flux, and in both cases the
    // residual is machine noise and Newton exits without doing anything.
    if (field && rank == 0) {
      auto range = [](const Real* dev, size_t n, double& lo, double& hi,
                      double& sum) {
        std::vector<Real> h(n);
        cudaMemcpy(h.data(), dev, sizeof(Real) * n, cudaMemcpyDeviceToHost);
        lo = 1e300; hi = -1e300; sum = 0;
        for (size_t i = 0; i < n; ++i) {
          lo = std::min(lo, (double)h[i]);
          hi = std::max(hi, (double)h[i]);
          sum += (double)h[i];
        }
      };
      double lo, hi, sum;
      const int nc = sys.cells;
      std::printf("%s  solver input\n", kSectionRule);
      std::printf("    cells      : %d owned, %d with ghosts\n", nc, sys.cells_ext);
      std::printf("    matrix     : %d rows, %d nonzeros, max degree %d\n",
                  sys.n, sys.nnz, res.max_degree);
      range(res.poro, nc, lo, hi, sum);
      std::printf("    porosity   : %.4g .. %.4g\n", lo, hi);
      range(res.cellvol, nc, lo, hi, sum);
      std::printf("    accumulation volume: %.4g .. %.4g  (before porosity)\n",
                  lo, hi);
      range(res.conn_trans, res.nconn, lo, hi, sum);
      std::printf("    trans      : %.4g .. %.4g\n", lo, hi);
      range(res.celldepth, nc, lo, hi, sum);
      std::printf("    depth      : %.1f .. %.1f\n", lo, hi);
      {
        std::vector<Real> h((size_t)nc * res.bs);
        cudaMemcpy(h.data(), res.u, sizeof(Real) * h.size(),
                   cudaMemcpyDeviceToHost);
        double plo = 1e300, phi = -1e300, slo = 1e300, shi = -1e300;
        for (int c = 0; c < nc; ++c) {
          plo = std::min(plo, (double)h[res.bs * c]);
          phi = std::max(phi, (double)h[res.bs * c]);
          if (res.bs >= 2) {
            slo = std::min(slo, (double)h[res.bs * c + 1]);
            shi = std::max(shi, (double)h[res.bs * c + 1]);
          }
        }
        std::printf("    pressure   : %.2f .. %.2f\n", plo, phi);
        std::printf("    Sw         : %.4f .. %.4f\n", slo, shi);
      }
      const bool missing_sat_tables =
          res.tables.nsw == 0 || (res.bs == 3 && res.tables.nsg == 0);
      std::printf("    sat tables : SWOF %d rows, SGOF %d rows%s\n",
                  res.tables.nsw, res.tables.nsg,
                  missing_sat_tables
                      ? "   <-- required saturation table is empty" : "");
      std::printf("    gravity    : %s, grad %.4f, rho w/o/g %.4f/%.4f/%.4f\n",
                  res.gravity ? "on" : "OFF", (double)res.grav_grad,
                  (double)res.rhow, (double)res.rhoo, (double)res.rhog);
      {
        int nwell = 0;
        if (field->schedule.n_steps() > 0)
          for (const Well& w : field->schedule.step(0).wells)
            if (w.state.open) ++nwell;
        std::printf("    wells open : %d at the first report step\n", nwell);
      }
      std::printf("%s", kSectionRule);
      std::fflush(stdout);
    }
    if (deck.emulate_fp64) {
      NVRS_CHECK_CUDA(cudaMalloc(&sys.aval_hi, sizeof(float) * sys.nnz));
      NVRS_CHECK_CUDA(cudaMalloc(&sys.aval_lo, sizeof(float) * sys.nnz));
    }

    // --- Jacobian cross-check: analytic vs numerical on the initial state ---
    if (jac_check) {
      if (deck.model == Model::Compositional) {
        if (rank == 0)
          std::cout << "--jac-check: analytic path is black-oil only; "
                       "compositional uses the numerical Jacobian.\n";
      } else {
        std::vector<Real> hnum(sys.nnz), hana(sys.nnz);
        int save = res.jacobian;
        res.jacobian = 0; assemble_blackoil(res, sys);
        NVRS_CHECK_CUDA(cudaMemcpy(hnum.data(), sys.aval64, sizeof(Real) * sys.nnz,
                                   cudaMemcpyDeviceToHost));
        res.jacobian = 1; assemble_blackoil(res, sys);
        NVRS_CHECK_CUDA(cudaMemcpy(hana.data(), sys.aval64, sizeof(Real) * sys.nnz,
                                   cudaMemcpyDeviceToHost));
        res.jacobian = save;
        Real maxabs = 0, maxrel = 0, refmax = 0;
        for (int i = 0; i < sys.nnz; ++i) {
          Real d = std::fabs(hnum[i] - hana[i]);
          Real s = std::fabs(hnum[i]) + std::fabs(hana[i]);
          maxabs = std::max(maxabs, d);
          if (s > 1e-12) maxrel = std::max(maxrel, 2 * d / s);
          refmax = std::max(refmax, std::fabs(hnum[i]));
        }
        if (rank == 0) {
          std::cout << "\n=== Jacobian cross-check (analytic vs numerical) ===\n"
                    << "  nnz              : " << sys.nnz << "\n"
                    << "  max |J_num|      : " << refmax << "\n"
                    << "  max abs diff     : " << maxabs << "\n"
                    << "  max rel diff     : " << maxrel << "\n"
                    << "  verdict          : "
                    << (maxrel < 1e-4 ? "PASS (agree to O(eps))"
                                      : "CHECK (rel diff > 1e-4 — investigate)")
                    << "\n====================================================\n";
        }
      }
      free_reservoir(res);
      HYPRE_Finalize();
      MPI_Finalize();
      return 0;
    }

    SolverConfig cfg;
    cfg.krylov = deck.krylov;
    cfg.tol = deck.lin_tol;
    cfg.max_iters = deck.lin_maxit;
    cfg.restart = deck.gmres_restart;
    cfg.ortho_passes = std::clamp(deck.ortho_passes, 1, 2);
    cfg.use_cprw = deck.use_cprw != 0;
    cfg.mixed = deck.mixed_precision != 0;
    // EMULATE 0/1/2: native FP64, df64, FP32-stored with an FP64 master.
    cfg.operator_mode = std::clamp(deck.emulate_fp64, 0, 2);
    cfg.emulate_fp64 = cfg.operator_mode == 1;
    cfg.block_spmv = deck.block_spmv != 0;
    cfg.cuda_graph = deck.cuda_graph != 0;
    cfg.graph_min_reuse = (double)deck.graph_min_reuse;
    cfg.graph_guard_after = std::max(1, deck.graph_guard_after);
    if (const char* e = std::getenv("NVRS_BLOCKSPMV")) cfg.block_spmv = std::atoi(e) != 0;
    if (const char* e = std::getenv("NVRS_CUDA_GRAPH")) cfg.cuda_graph = std::atoi(e) != 0;
    if (const char* e = std::getenv("NVRS_GRAPH_MIN_REUSE")) cfg.graph_min_reuse = std::atof(e);
    cfg.amg_reuse = std::max(0, deck.amg_reuse);
    cfg.stage2 = deck.stage2;
    cfg.amg_reuse_growth = (double)deck.amg_reuse_growth;
    cfg.amg_max_coarse = deck.amg_max_coarse;
    cfg.amg_print_level = deck.amg_print_level;
    cfg.amg_coarse_relax = deck.amg_coarse_relax;
    // LINSOLVER.  A binary built without AmgX cannot honour a request for
    // it, and silently running the other library would make every benchmark
    // label wrong, so say so and fall back.
    cfg.backend = deck.backend == 2   ? SolverConfig::kGinkgo
                  : deck.backend == 1 ? SolverConfig::kAmgx
                                      : SolverConfig::kHypre;
    cfg.ginkgo_precision = deck.ginkgo_precision;
#ifndef NVRS_HAVE_GINKGO
    if (cfg.backend == SolverConfig::kGinkgo) {
      cfg.backend = SolverConfig::kAmgx;
      if (rank == 0)
        std::printf("  [warn] LINSOLVER GINKGO requested but this binary was "
                    "built without Ginkgo; using AmgX. Run "
                    "benchmark/install_ginkgo.sh and rebuild with GINKGO_DIR "
                    "set.\n");
    }
#endif
#ifndef NVRS_HAVE_AMGX
    if (cfg.backend == SolverConfig::kAmgx) {
      cfg.backend = SolverConfig::kHypre;
      if (rank == 0)
        std::printf("  [warn] LINSOLVER AMGX requested but this binary was "
                    "built without AmgX; using HYPRE. Rebuild with "
                    "AMGX_DIR set.\n");
    }
#endif
#ifndef NVRS_HAVE_GINKGO
    if (cfg.stage2 == 2) {
      cfg.stage2 = 0;
      if (rank == 0)
        std::printf("  [warn] STAGE2 GINKGO requested but this binary was built "
                    "without Ginkgo; using the built-in block-Jacobi.\n");
    }
#endif
    if (cfg.stage2 == 2 && nranks > 1) {
      if (rank == 0)
        std::fprintf(stderr, "ERROR: STAGE2 GINKGO runs on one GPU in this "
                             "build (%d ranks requested).\n", nranks);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    // Ginkgo is wired single-GPU.  Stop here rather than inside the first
    // linear solve, with the one sentence that says what to do instead.
    if (cfg.backend == SolverConfig::kGinkgo && nranks > 1) {
      if (rank == 0)
        std::fprintf(stderr, "ERROR: LINSOLVER GINKGO runs on one GPU in this "
                             "build (%d ranks requested). Run it with one "
                             "rank, or use LINSOLVER AMGX for several GPUs.\n",
                     nranks);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (rank == 0) {
      if (cfg.backend == SolverConfig::kGinkgo)
        std::printf("  Linear   : stage-1 multigrid from Ginkgo, FP%d\n",
                    cfg.ginkgo_precision);
      else
        std::printf("  Linear   : stage-1 multigrid from %s\n",
                    cfg.backend == SolverConfig::kAmgx ? "AmgX" : "HYPRE");
    }
    if (rank == 0 && cfg.use_cprw)
      std::printf("  Linear   : CPRW global eliminated-well pressure correction%s\n",
                  deck.use_mgr ? " requested (not supported with MGR)" : "");
    cfg.use_mgr = deck.use_mgr;
    cfg.mgr_relax_type = deck.mgr_relax_type;
    cfg.mgr_relax_sweeps = deck.mgr_relax_sweeps;
    cfg.mgr_interp_type = deck.mgr_interp_type;
    cfg.mgr_global_smooth = deck.mgr_global_smooth;
    cfg.mgr_global_sweeps = deck.mgr_global_sweeps;
    cfg.agg_num_levels = deck.agg_num_levels;
    cfg.amg_seq_threshold = deck.amg_seq_threshold;
    cfg.amg_nongalerkin_tol = (double)deck.amg_nongalerkin_tol;
    cfg.agg_trunc_factor = (double)deck.agg_trunc_factor;
    cfg.agg_pmax = deck.agg_pmax;
    // AMGFAST fills in whatever the deck did not set explicitly, so a
    // single keyword gives the recommended multi-GPU bundle and any
    // individual keyword still overrides it.
    // What is left of AMGFAST after two of its three parts turned out to be
    // host-only in HYPRE: coarsen faster, and keep the interpolation sparse
    // while doing it.  Both are ordinary coarsening/interpolation settings and
    // run on the device.
    //
    // Neither AMGSEQTHRESH nor AMGNONGALERKIN is set here.  Both segfault
    // against a device build -- see the notes in configure_amg -- and both were
    // aimed squarely at the coarse levels, so this bundle now attacks the same
    // cost indirectly, by never building those levels in the first place.
    if (deck.amg_fast) {
      if (cfg.agg_trunc_factor == 0.0) cfg.agg_trunc_factor = 0.3;
      if (cfg.agg_pmax == 0) cfg.agg_pmax = 4;
      if (deck.agg_num_levels <= 1) cfg.agg_num_levels = 2;
      // Stop coarsening early and keep the coarsest grid on the device.
      // These two belong together: a large coarse grid is only affordable
      // because the coarse solve is no longer Gaussian elimination, which
      // HYPRE gathers and factorises densely on every rank.
      //
      // Measured on 1M cells, one GB200: seven levels at 2.07 ms per
      // V-cycle become three at 1.11 ms.  A V-cycle here is dominated by
      // per-level overhead, not by arithmetic -- level 0 is a few hundred
      // microseconds of memory traffic inside a cycle costing five times
      // that with no communication anywhere -- so removing levels is what
      // pays.  Whole run: 96.5 s to 54.1 s.
      if (deck.amg_coarse_relax == 9) cfg.amg_coarse_relax = 18;
      if (cfg.amg_max_coarse == 0) cfg.amg_max_coarse = 6000;
    }
    // The sequential scheme hands the solver two *different* matrices through
    // one DeviceLinearSystem — an elliptic pressure system and a transport
    // system, back to back — so consecutive setup() calls are not consecutive
    // iterates on the same problem and there is nothing to reuse.  The
    // fingerprint in CprSolver::setup would catch matrices at different
    // addresses; these share one, so rule it out here instead.
    if (deck.scheme == 1) cfg.amg_reuse = 0;
    CprSolver solver(grid, res.bs, cfg);
    // A field partition's ghosts are not a K-slab seam, so the operator apply
    // must exchange them through the same unstructured halo the assembly uses.
    // Null on a structured run, which keeps the K-slab path.
    solver.set_conn_halo(res.halo_conn);

    // The RESERVOIR's grid, not the run deck's box.
    //
    // For a field run the two are nothing alike: `grid` is Grid::make from the
    // run deck's DIMENS, which a field deck never sets, so it is 1 x 1 x 1.
    // The writer gathers grid_.n_local() cells per snapshot -- so it gathered
    // ONE, wrote one value, and every field came out blank with the rock
    // entirely NaN.  res.grid is sized to the cells this rank actually owns
    // (44,431 of Norne's active cells), and for a Cartesian run it is the same
    // object, so this is correct either way.
    Writer writer(deck, res.grid);
    // res.grid says rank 0 of 1 on a field run; the writer must not believe it
    // or every rank writes meta.json over the top of every other.
    writer.set_mpi(rank, nranks);
    // Hand the writer the field's real geometry.
    //
    // Without it the sidecar describes the RUN deck: a 1 x 1 x 1 box and no
    // wells, from a run that solved 44,431 cells with 36 wells in them.  With
    // it, a corner-point run writes the same files a Cartesian run writes --
    // states scattered onto the full index box, NaN where the grid is inactive
    // -- plus the cgrid table, which is what lets a plot draw the field's true
    // dipping, faulted shape rather than a cuboid.
    if (field && rank == 0) {
      Writer::FieldGeometry fg;
      const auto& cg = field->grid;
      fg.nx = cg.nx; fg.ny = cg.ny; fg.nz = cg.nz;
      const int na = cg.n_active();

      // Global solver row -> active cell, so the writer can put a multi-rank
      // gather back into active order.  Built exactly the way field_model.cu
      // numbers the rows: ranks in order, and within a rank its owned cells
      // ascending in active index.  Every rank holds the whole partition, so
      // this needs no communication.
      {
        const auto& P = field->part;
        std::vector<long long> cnt(P.nranks, 0), off(P.nranks + 1, 0);
        for (int a = 0; a < na; ++a) ++cnt[P.owner_of_active[a]];
        for (int q = 0; q < P.nranks; ++q) off[q + 1] = off[q] + cnt[q];
        std::vector<long long> seen(P.nranks, 0);
        fg.active_of_row.assign(na, -1);
        for (int a = 0; a < na; ++a) {
          const int q = P.owner_of_active[a];
          const long long row = off[q] + seen[q]++;
          if (row >= 0 && row < na) fg.active_of_row[(size_t)row] = a;
        }
      }

      fg.active_to_global.resize(na);
      fg.cgrid.resize((size_t)na * 6);
      for (int a = 0; a < na; ++a) {
        const int g = cg.global_of_active(a);
        fg.active_to_global[a] = g;
        // 1-based i, j, k, as a cgrid table is indexed.
        const int i = g % cg.nx;
        const int j = (g / cg.nx) % cg.ny;
        const int k = g / (cg.nx * cg.ny);
        float* row = fg.cgrid.data() + (size_t)a * 6;
        row[0] = (float)(i + 1);
        row[1] = (float)(j + 1);
        row[2] = (float)(k + 1);
        row[3] = (a < (int)cg.centroid_x.size()) ? (float)cg.centroid_x[a] : 0.f;
        row[4] = (a < (int)cg.centroid_y.size()) ? (float)cg.centroid_y[a] : 0.f;
        // Depth is positive downwards; negate so the plot's z axis points up
        // and the structure reads the way a section of a field reads.
        row[5] = (a < (int)cg.depth.size()) ? -(float)cg.depth[a] : 0.f;
      }
      // One marker per well, at its first completion, typed by what it does at
      // the end of the schedule -- which is when the picture is drawn.
      if (field->schedule.n_steps() > 0) {
        const auto& last = field->schedule.step(field->schedule.n_steps() - 1);
        for (const Well& w : last.wells) {
          if (w.completions.empty()) continue;
          Writer::WellXY wx;
          wx.name = w.name;
          wx.i = w.completions.front().i + 1;
          wx.j = w.completions.front().j + 1;
          // Water and gas injection are different colours because they are
          // different things.  Norne's C-4H is both at different times.
          if (w.state.type != WellType::Injector) wx.kind = "prod";
          else if (w.state.inj_phase == InjectedPhase::Gas) wx.kind = "ginj";
          else wx.kind = "winj";
          fg.wells.push_back(wx);
        }
      }
      // The grid's own pore volume and depth, per ACTIVE cell, for the
      // ECLIPSE INIT file.  Without these it wrote the run deck's uniform
      // 50 x 50 x 20 ft box for every cell of a field whose volumes span two
      // and a half orders of magnitude, and every volumetric comparison drawn
      // from that file was wrong by the ratio.
      {
        const auto& cg2 = field->grid;
        const int na2 = cg2.n_active();
        // CornerPointGrid::build() converts FIELD geometry from ft^3 to
        // reservoir barrels at construction, and to_field_units() converts a
        // METRIC grid from m^3 to the same unit. Thus cg2.porv is already rb
        // for both inputs. Dividing a FIELD grid here a second time made SPE10
        // appear to hold 5.614583x too little water in the comparison report.
        std::vector<float> pv(na2), dp(na2);
        for (int a = 0; a < na2; ++a) {
          const double v = (a < (int)cg2.porv.size()) ? (double)cg2.porv[a]
                                                      : 0.0;
          pv[a] = (float)v;
          dp[a] = (a < (int)cg2.depth.size()) ? (float)cg2.depth[a] : 0.f;
        }
        // And the DECK's porosity, not the solver's.
        //
        // field_model multiplies poro by NTG for the accumulation term, which
        // is right for the equations and wrong for a file labelled PORO: the
        // INIT went out with 0.0061 .. 0.347 against Flow's 0.134 .. 0.347,
        // the two INIT files disagreed by 58%, and the report said so --
        // "the comparison below is between two different reservoirs".  PORV
        // above already carries NTG, so writing it here too counts it twice.
        std::vector<float> pr;
        if (const std::vector<Real>* pd = cg2.array("PORO")) {
          pr.resize(na2);
          for (int a = 0; a < na2; ++a) {
            const int gidx = cg2.global_of_active(a);
            pr[a] = (gidx >= 0 && gidx < (int)pd->size())
                        ? (float)(*pd)[gidx] : 0.f;
          }
        }
        writer.set_cell_geometry(std::move(pv), std::move(dp));
        if (!pr.empty()) writer.set_cell_porosity(std::move(pr));
      }
      writer.set_field_grid(fg);
    }


    // Newton work vector.  Allocated over the extended (owned + ghost) space
    // because it is handed to the solver as an SpMV operand; only the owned
    // prefix carries the Newton update.
    Real* du = nullptr;
    NVRS_CHECK_CUDA(cudaMalloc(&du, sizeof(Real) * sys.n_ext));
    NVRS_CHECK_CUDA(cudaMemset(du, 0, sizeof(Real) * sys.n_ext));

    // When to report.
    //
    // DTREPORT puts them on a uniform grid: 3304 days / 30 = 110 steps.  The
    // field deck does not have a uniform grid -- Norne's schedule names 247
    // dates, days apart during the drilling and months apart later -- and Flow
    // reports on exactly those.  Sampling the two simulators at different
    // dates makes a per-timestep comparison meaningless before it starts:
    // every difference plot would be differencing a June state against a
    // July one.
    //
    // So a field run follows the deck's dates.  DTREPORT still governs a
    // Cartesian deck, which has no schedule to follow, and REPORTDATES 0
    // forces the uniform grid back for a field run that wants it.
    std::vector<Real> report_days;
    if (field && deck.use_schedule && deck.report_dates) {
      // ScheduleStep::day is an ABSOLUTE day number, not days since
      // START: Norne's run to 3304 days carries days 10179 .. 13483.
      // Subtract the start, exactly as schedule_step_at() does.  Without
      // it every date failed the `d <= TEND` test, no report times were
      // built, and the run fell back to the uniform DTREPORT grid -- 110
      // steps against Flow's 247.  The schedule summary hides the
      // convention by printing back().day - front().day, which is 3304
      // either way.
      const double t0 = field->schedule.start_day();
      for (int i = 0; i < field->schedule.n_steps(); ++i) {
        const double d = field->schedule.step(i).day - t0;
        if (d > 1e-9 && d <= (double)deck.t_end + 1e-9)
          report_days.push_back((Real)d);
      }
      std::sort(report_days.begin(), report_days.end());
      report_days.erase(std::unique(report_days.begin(), report_days.end()),
                        report_days.end());
    }
    if (report_days.empty()) {
      const int n = std::max(1, (int)std::lround(deck.t_end / deck.dt_report));
      for (int i = 1; i <= n; ++i) report_days.push_back(deck.dt_report * i);
    }
    // Say where the report times came from, and if they did not come from
    // the schedule, say which input stopped them.  Guessing at this from the
    // step count alone has already cost several runs.
    if (rank == 0) {
      const int nsched = field ? field->schedule.n_steps() : 0;
      std::printf("  report times  : %d step(s), field=%d use_schedule=%d\n",
                  (int)report_days.size(), field ? 1 : 0, deck.use_schedule);
      std::printf("                  report_dates=%d schedule_steps=%d tend=%g\n",
                  deck.report_dates, nsched, (double)deck.t_end);
      if (nsched > 0)
        std::printf("                  schedule days %g .. %g (since start)\n",
                    (double)(field->schedule.step(0).day
                             - field->schedule.start_day()),
                    (double)(field->schedule.step(nsched - 1).day
                             - field->schedule.start_day()));
      std::fflush(stdout);
    }
    // A TEND shorter than the schedule silently shortens the comparison.
    // Say so, with the value that would cover it.
    if (rank == 0 && field && deck.report_dates && deck.use_schedule) {
      const int nsched = field->schedule.n_steps();
      const int dropped = nsched - (int)report_days.size();
      if (dropped > 0 && nsched > 0) {
        const double last = field->schedule.step(nsched - 1).day
                            - field->schedule.start_day();
        std::printf("  [note] TEND %g leaves out %d of the schedule\'s %d dates; TEND %g covers them all\n",
                    (double)deck.t_end, dropped, nsched, last);
      }
    }
    const int nsteps = (int)report_days.size();
    // The interval this report step has to cover, which is no longer a
    // constant: Norne's dates are 1 to 92 days apart.
    auto step_dt = [&](int step) -> Real {
      const Real prev = (step > 1) ? report_days[step - 2] : (Real)0;
      return report_days[step - 1] - prev;
    };

    // Carry the timestep that worked into the next report step.
    //
    // Without this, res.dt is reset to DTREPORT at the start of every report
    // step, so a run that needs quarter-length steps re-discovers that fact
    // twenty times over: it spends up to NEWTONMAX iterations failing at the
    // full step, twice that failing at half, and only then does the work that
    // counts.  Measured on 46 x 112 x 22, roughly half of all Newton
    // iterations were thrown away like that.  Flow does not do this -- its log
    // shows 1, 1.8, 3.96, 8.7 ... 100 days, carried forward and grown.
    //
    // chop_hint is that memory, in halvings of DTREPORT.  After a success it
    // relaxes by one level so the step can grow back when the flood settles
    // down, which costs one failed attempt if it was too optimistic and saves
    // every attempt after that when it was not.
    int chop_hint = 0;
    // The continuous counterpart of chop_hint, in days.  Zero until the first
    // report step has planned one, after which it is what the controller last
    // asked for and the next step starts from.
    res.cnv_tol = deck.opm_conv ? deck.cnv_tol : (Real)0;
    res.cpr_decouple = deck.cpr_decouple;
    res.drs_eps = deck.drs_eps;
    Real dt_next = 0;
    // Flow's PID controller retains the last three accepted relative changes.
    // Initialising all three to the tolerance is its neutral history.
    Real dt_pid_error[3] = {deck.dt_pid_tol, deck.dt_pid_tol, deck.dt_pid_tol};
    // The longest dt the recent past says is safe, carried across report steps
    // with dt_next.  Zero means no ceiling has been learned yet.
    Real dt_ceiling = 0;
    // A probe temporarily raises dt_ceiling after a run of cheap solves. Keep
    // the last proven ceiling so a failed probe returns there instead of
    // ratcheting the whole run down through DTFAILCAP.
    int cap_easy_run = 0;
    Real cap_probe_floor = 0;
    // Work done on attempts that were later discarded.  It is in the wall
    // clock either way; the only question is whether it is also in the count.
    long long nnewt_lost = 0, nlin_lost = 0;
    long long nsub_accepted = 0, nline_trials = 0, nline_no_descent = 0;
    long long nline_assembly_handoffs = 0;
    int steps_chopped = 0;
    int clean_steps = 0;   // consecutive steps that converged first try
    std::vector<Real> step_times;
    std::vector<int> newton_iters, linear_iters;

    // The CuPy-faithful sequential scheme needs its own scalar workspace.
    SeqSolver seq;
    const bool sequential = (deck.scheme == 1);
    if (sequential) seq_build(res, deck, seq);

    // NVRS_VERBOSE=1 traces every Newton iteration and its linear solve.
    const bool verbose = std::getenv("NVRS_VERBOSE") != nullptr;
    bool warned_linear = false;
    // One line, once, if OPMCONV cannot be honoured on this grid.
    bool warned_opmconv = false;
    bool dumped_op = false;          // DUMPOP writes once

    auto t0 = std::chrono::high_resolution_clock::now();
    // Split the wall clock so the report can say how much of it was the
    // linear solve, which is the part the mixed-precision CPR work is
    // actually about.  cudaDeviceSynchronize before reading the clock:
    // the kernels are asynchronous and an unsynchronised timer would
    // charge the launch, not the work.
    double t_assemble = 0.0, t_linear = 0.0, t_well_control = 0.0;
    auto tic = [] {
      cudaDeviceSynchronize();
      return std::chrono::high_resolution_clock::now();
    };
    auto toc = [&](std::chrono::high_resolution_clock::time_point a) {
      cudaDeviceSynchronize();
      return std::chrono::duration<double>(
                 std::chrono::high_resolution_clock::now() - a).count();
    };
    // What WEFAC actually says, so its effect is a number and not a claim.
    if (rank == 0 && field && field->schedule.n_steps() > 0) {
      double lo = 1e30, hi = -1e30, sum = 0;
      int n = 0, below = 0;
      for (int i = 0; i < field->schedule.n_steps(); ++i)
        for (const Well& w : field->schedule.step(i).wells) {
          const double e = (double)w.state.efficiency;
          if (!(e > 0) || e > 1) continue;
          lo = std::min(lo, e); hi = std::max(hi, e); sum += e; ++n;
          if (e < 0.999) ++below;
        }
      if (n > 0)
        std::printf("  WEFAC    : %d record(s), %.3f .. %.3f, mean %.3f; %d below 1 (%.0f%%)\n",
                    n, lo, hi, sum / n, below, 100.0 * below / n);
    }

    const bool dates_from_deck =
        (field && deck.use_schedule && deck.report_dates != 0);
    if (rank == 0) print_step_header(deck, nsteps, dates_from_deck);

    // DTSEQFILE: the end times of an imposed sequence of accepted substeps
    // ("t_start dt" per line, days).  With it, each substep runs from the
    // current time to the next listed end time, whatever the controller would
    // have chosen; a substep that fails is still chopped, and the sequence is
    // rejoined at the next listed end.  This is how the reference's own
    // accepted steps are imposed (benchmark/opm_timesteps.py writes them from
    // a Flow log), and how every rank count of a scaling study is given the
    // same steps.
    std::vector<double> dtseq_ends;
    if (!deck.dt_seq_file.empty()) {
      std::ifstream f(deck.dt_seq_file);
      std::string line;
      while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        double t0 = 0, dt = 0;
        if (is >> t0 >> dt && dt > 0) dtseq_ends.push_back(t0 + dt);
      }
      std::sort(dtseq_ends.begin(), dtseq_ends.end());
      dtseq_ends.erase(std::unique(dtseq_ends.begin(), dtseq_ends.end(),
                                   [](double a, double b) {
                                     return std::fabs(a - b) < 1e-7;
                                   }),
                       dtseq_ends.end());
      if (rank == 0)
        std::printf("  [dtseq] %zu imposed substep ends read from %s\n",
                    dtseq_ends.size(), deck.dt_seq_file.c_str());
      if (dtseq_ends.empty() && rank == 0)
        std::printf("  [warn] DTSEQFILE %s gave no substeps; the adaptive "
                    "controller is used instead\n", deck.dt_seq_file.c_str());
    }
    // Every accepted substep, as (start day, length).  Written at the end as
    // OUTDIR/dt_sequence.txt, which DTSEQFILE reads back.
    std::vector<std::pair<double, double>> accepted_dt;
    // Host time spent writing restart and summary data inside the wall clock.
    double t_io = 0.0;

    Real t_days = 0;
    // Optional discrete material ledger.  Backward Euler uses the well source
    // evaluated at each accepted sub-step endpoint, so q_end*dt is the exact
    // source volume seen by the residual.  Report-step trapezoids can be very
    // misleading when controls or phase splits move inside a long interval.
    const bool substep_balance = std::getenv("NVRS_SUBSTEP_BALANCE") != nullptr;
    double sub_qo = 0, sub_qwp = 0, sub_qwi = 0, sub_qgp = 0, sub_qgi = 0;
    const bool exact_well_cumulative =
        field && res.bs == 2 && res.use_conn_list && res.jacobian == 1;
    const bool residual_balance_active =
        field && res.use_conn_list && res.jacobian == 1 && res.bs == 3;
    double residual_balance[3] = {0, 0, 0};
    writer.snapshot(res, 0, t_days);          // initial state
    // The initial state gets a rate row too, so the well curves start at t = 0
    // like every other series in the report rather than one step in.
    if (field)
    {
      // Apply the controls before recording, or the t = 0 row is whatever
      // apply_schedule_step's bottom-hole fallback gives: the report opened
      // with a peak field rate of 193 k STB/day at day zero, which is the
      // uncontrolled state and not a rate the field ever made.
      const int ss0 = schedule_step_at(*field, (double)t_days);
      if (deck.use_schedule) {
        apply_producer_rate_targets(res, *field, ss0, deck, &pvt_host);
        apply_injector_rate_targets(res, *field, ss0, deck, &pvt_host);
      }
      std::vector<WellRateRow> wr =
          field_well_rates(deck, *field, res, ss0, &pvt_host);
      reduce_well_rates(wr, nranks);
      writer.record_well_rates(t_days, wr);
      if (exact_well_cumulative)
        writer.record_well_cumulative(0.0, 0.0);
    }

    // Snapshot of the state at the start of a report step, so a failed attempt
    // can be rolled back and retried on a shorter timestep.
    Real* u_save = nullptr;
    NVRS_CHECK_CUDA(cudaMalloc(&u_save, sizeof(Real) * sys.n_ext));
    // The state at the start of the sub-step now being attempted, so a failure
    // can be undone without discarding the sub-steps that already converged.
    // Allocated whatever DTKEEP says: one field is cheap and a conditional
    // allocation is one more thing to get wrong.
    Real* u_sub = nullptr;
    NVRS_CHECK_CUDA(cudaMalloc(&u_sub, sizeof(Real) * sys.n_ext));
    // Fully implicit line-search rollback.  u_sub belongs to the timestep
    // controller and must continue to hold the accepted sub-step boundary.
    Real* u_newton = nullptr;
    NVRS_CHECK_CUDA(cudaMalloc(&u_newton, sizeof(Real) * sys.n_ext));
    Real *gp_save = nullptr, *gp_sub = nullptr, *gp_newton = nullptr;
    const size_t gp_bytes = sizeof(Real) * (size_t)res.grid.n_local_ext();
    if (res.gas_primary) {
      NVRS_CHECK_CUDA(cudaMalloc(&gp_save, gp_bytes));
      NVRS_CHECK_CUDA(cudaMalloc(&gp_sub, gp_bytes));
      NVRS_CHECK_CUDA(cudaMalloc(&gp_newton, gp_bytes));
    }
    // Shadows of the clamp diagnostic, rolled back wherever the state is.
    // Without these the accumulator would keep the limiter's effect on
    // iterations belonging to sub-steps that were thrown away, and read as a
    // sink for oil the field never actually lost.
    Real *cd_save = nullptr, *cd_sub = nullptr, *cd_newton = nullptr;
    if (res.clamp_dso) {
      NVRS_CHECK_CUDA(cudaMalloc(&cd_save, sizeof(Real) * (size_t)res.grid.n_local_ext()));
      NVRS_CHECK_CUDA(cudaMalloc(&cd_sub, sizeof(Real) * (size_t)res.grid.n_local_ext()));
      NVRS_CHECK_CUDA(cudaMalloc(&cd_newton, sizeof(Real) * (size_t)res.grid.n_local_ext()));
    }
    // Accepted-step histories are part of the nonlinear state even though
    // they are not primary unknowns.  A dyadic retry rolls the primary state
    // back to the report boundary, so it must roll these back with it; keeping
    // extrema from a discarded future trajectory corrupts Carlson scanning
    // curves and VAPPARS on every subsequent retry.
    const size_t hist_bytes =
        sizeof(Real) * (size_t)res.grid.n_local_ext();
    Real *rs_save = nullptr, *so_save = nullptr;
    Real *sg_hy_save = nullptr, *sw_ow_hy_save = nullptr;
    auto alloc_history_shadow = [&](Real* history, Real** shadow) {
      if (history) NVRS_CHECK_CUDA(cudaMalloc(shadow, hist_bytes));
    };
    alloc_history_shadow(res.rs_cap, &rs_save);
    alloc_history_shadow(res.so_max, &so_save);
    alloc_history_shadow(res.sg_hy, &sg_hy_save);
    alloc_history_shadow(res.sw_ow_hy, &sw_ow_hy_save);
    auto save_history = [&]() {
      if (rs_save) NVRS_CHECK_CUDA(cudaMemcpy(
          rs_save, res.rs_cap, hist_bytes, cudaMemcpyDeviceToDevice));
      if (so_save) NVRS_CHECK_CUDA(cudaMemcpy(
          so_save, res.so_max, hist_bytes, cudaMemcpyDeviceToDevice));
      if (sg_hy_save) NVRS_CHECK_CUDA(cudaMemcpy(
          sg_hy_save, res.sg_hy, hist_bytes, cudaMemcpyDeviceToDevice));
      if (sw_ow_hy_save) NVRS_CHECK_CUDA(cudaMemcpy(
          sw_ow_hy_save, res.sw_ow_hy, hist_bytes, cudaMemcpyDeviceToDevice));
    };
    auto restore_history = [&]() {
      if (rs_save) NVRS_CHECK_CUDA(cudaMemcpy(
          res.rs_cap, rs_save, hist_bytes, cudaMemcpyDeviceToDevice));
      if (so_save) NVRS_CHECK_CUDA(cudaMemcpy(
          res.so_max, so_save, hist_bytes, cudaMemcpyDeviceToDevice));
      if (sg_hy_save) NVRS_CHECK_CUDA(cudaMemcpy(
          res.sg_hy, sg_hy_save, hist_bytes, cudaMemcpyDeviceToDevice));
      if (sw_ow_hy_save) NVRS_CHECK_CUDA(cudaMemcpy(
          res.sw_ow_hy, sw_ow_hy_save, hist_bytes, cudaMemcpyDeviceToDevice));
    };

    for (int step = 1; step <= nsteps; ++step) {
      // The field deck's schedule drives the wells.  Schedule::read snapshots
      // the running controls when DATES advances the clock.  Consequently the
      // snapshot at the END of an interval contains the controls that were in
      // force throughout that interval: records written after the preceding
      // DATES were folded into it.  Selecting the snapshot at the interval's
      // start applies every change one report step late.  That is especially
      // visible on Norne around day 500, where B-1H is introduced and C-3H is
      // opened, recompleted, shut and converted in closely spaced records.
      const Real dt_rep = step_dt(step);
      // Which schedule record governs this report step.  Kept out here so the
      // Newton loop below can re-solve the well controls against it.
      int sched_step_now = -1;
      if (field && deck.use_schedule) {
        // Use the interval END, not its start.  The first interval is unchanged
        // (both lookups select snapshot zero); every later interval now sees
        // the controls at the date on which the preceding report was written.
        const double day = (double)report_days[step - 1];
        const int ss = schedule_step_at(*field, day);
        sched_step_now = ss;

        // Announce what the schedule does before the step that does it.
        //
        // These events are the content of a field schedule, and a run that
        // does not report them gives no way to connect a timestep that will
        // not converge to the thing that caused it.  Norne turns C-4H from a
        // producer into a gas injector three months in; that took a crash and
        // a bisection to discover, which it should not have.
        // A well changing role is banner-worthy rather than log-worthy: it is
        // the single most common reason a run that has been converging stops,
        // and it should be impossible to scroll past.
        static int last_reported = -1;
        if (rank == 0 && ss > last_reported) {
          for (int q = std::max(last_reported, 0); q < ss; ++q) {
            const std::vector<WellEvent> ev =
                well_changes(field->schedule.step(q),
                             field->schedule.step(q + 1));
            // The difference q -> q+1 consists of records written immediately
            // after DATES q, so q is the date on which the event took effect.
            const char* when = field->schedule.step(q).date.c_str();
            for (const WellEvent& e : ev) {
              if (!e.major) continue;
              std::printf(
                  "\n  ****************************************************"
                  "**********\n"
                  "  ***  WELL CHANGES ROLE   %-38s ***\n"
                  "  ***  %-58s ***\n"
                  "  ****************************************************"
                  "**********\n\n",
                  when, e.text.c_str());
            }
            for (const WellEvent& e : ev)
              if (!e.major)
                std::printf("  [well] %s  (%s)\n", e.text.c_str(), when);
          }
          last_reported = ss;
        }

        if (ss >= 0) {
          int shut_no_bhp = 0;
          static int inj_no_bhp_total = 0;
          int inj_no_bhp = 0;
          apply_schedule_step(res, *field, ss, deck.prod_bhp, &shut_no_bhp,
                              deck.inj_bhp, &inj_no_bhp);
          // An injector with no ceiling is a source term with no state
          // dependence: the residual keeps a piece nothing can reduce and
          // Newton freezes on it.  Say so once.
          if (inj_no_bhp > 0 && inj_no_bhp_total == 0 && rank == 0) {
            inj_no_bhp_total = inj_no_bhp;
            std::printf("  [well] %d injector(s) have no pressure ceiling; their rate cannot be limited by the formation.  Set INJBHP in the deck.\n",
                        inj_no_bhp);
            std::fflush(stdout);
          }
          // apply_schedule_step gives every producer its BHP limit as an
          // OPERATING pressure.  For a well the deck controls by rate that
          // is not what was asked for, so replace it with the pressure that
          // meets the target -- keeping the limit as a limit.  Collective.
          apply_producer_rate_targets(res, *field, ss, deck, &pvt_host);
          apply_injector_rate_targets(res, *field, ss, deck, &pvt_host);
          // Reported once: a producer whose deck record defaults the bottom-
          // hole pressure cannot be run pressure-controlled, and silently
          // leaving it shut would look like a field that simply does not
          // produce.
          static bool said = false;
          if (shut_no_bhp > 0 && !said && rank == 0) {
            said = true;
            std::printf("  [warn] %d producer(s) have no bottom-hole pressure "
                        "in the schedule and are shut. Set PRODBHP to give "
                        "them a flowing pressure.\n", shut_no_bhp);
          }
        }
      }

      // ---------------------------------------------------------------------
      // Adaptive report step, mirroring the CuPy reference's outer
      // `while not conv: dt = Tt / 2**IT` loop.  A fully-implicit Newton solve
      // is not guaranteed to converge from an arbitrary starting point; when it
      // does not, the answer is a shorter timestep, not a longer iteration.
      // Each attempt halves dt and runs 2^chop sub-steps to cover the same
      // report interval.
      // ---------------------------------------------------------------------
      if (cd_save)
        NVRS_CHECK_CUDA(cudaMemcpy(cd_save, res.clamp_dso, sizeof(Real) * (size_t)res.grid.n_local_ext(),
                                   cudaMemcpyDeviceToDevice));
      NVRS_CHECK_CUDA(cudaMemcpy(u_save, res.u, sizeof(Real) * sys.n_ext,
                                 cudaMemcpyDeviceToDevice));
      if (gp_save) NVRS_CHECK_CUDA(cudaMemcpy(
          gp_save, res.gas_primary, gp_bytes, cudaMemcpyDeviceToDevice));
      save_history();
      int nnewt = 0, nlin = 0;
      Real r0 = 0, rlast = 0;
      bool step_ok = false;
      // Simulated time completed inside this report step. Zero unless DTKEEP
      // is on, in which case a failed attempt leaves the converged prefix in
      // place and only the remainder is retried.
      // DTCONTROL 1 implies DTKEEP: see deck.hpp.
      const bool itercount = (deck.dt_control != 0);
      const bool dt_keep = (deck.dt_keep != 0) || itercount;
      const bool pid_active = deck.dt_pid &&
                              deck.model != Model::Compositional;
      Real t_done = 0;

      // DTMAX sets how finely the report step is divided before Newton is even
      // asked; the max_chop halvings on top of that stay available for genuine
      // convergence failures.  Without it every report step is one jump of
      // DTREPORT, because the loop below only shortens dt when Newton fails and
      // Newton converges happily on a badly under-resolved step.
      int chop0 = 0;
      if (deck.dt_max > 0)
        while (chop0 < 24 &&
               dt_rep / (Real)(1 << chop0) > deck.dt_max * (1 + 1e-9))
          ++chop0;
      // Start where the last report step succeeded, not from scratch.
      if (chop_hint > chop0) chop0 = chop_hint;
      // Which test converged the last sub-step, for the report line.
      // What the step converged ON, kept for the line under the step table.
      // The relative residual is not among them: report_progress already
      // prints it, and carrying a second copy only produced a warning.
      Real conv_mb = 0, conv_signed_mb = 0, conv_cnv = 0;
      const char* conv_how = "";

      // Under iteration-count control there are no levels to descend, so the
      // outer loop runs once and the retry budget is counted inside.
      const int chop_last = itercount ? chop0 : chop0 + deck.max_chop;
      const long long nnewt_lost_at_entry = nnewt_lost;
      for (int chop = chop0; chop <= chop_last && !step_ok; ++chop) {
        // Cover what is left, at this halving level. With nothing completed
        // this is exactly dt_rep / 2^chop in 2^chop sub-steps, which is the
        // old behaviour; with a converged prefix kept it is the remainder,
        // divided so the sub-steps are equal and land on the boundary.
        const Real dt_level = dt_rep / (Real)(1 << chop);
        const Real remaining = dt_rep - t_done;
        int nsub = (int)std::ceil((double)remaining / (double)dt_level - 1e-9);
        if (nsub < 1) nsub = 1;
        res.dt = remaining / (Real)nsub;

        // Iteration-count control plans nothing ahead.  Each sub-step's length
        // comes from how the previous one converged, so nsub stops being a
        // plan and becomes a bound that the t_done test below reaches first.
        // The first report step of a run has no history and starts from what
        // the lattice would have chosen, which is DTMAX-limited already.
        Real dt_try = res.dt;
        int  nfail = 0;
        int  nsubs_done = 0;
        if (itercount) {
          if (dt_next > 0) {
            dt_try = dt_next;
          } else if (field && field->schedule.tuning_initial_step() > 0) {
            // Flow starts Norne at TSINIT=1 day and then lets its controller
            // grow.  Starting at the whole first 8-day report interval makes
            // phase appearance occur in one coarse jump and changes the
            // solution even though Newton happens to accept it.
            dt_try = field->schedule.tuning_initial_step();
          }
          if (dt_ceiling > 0 && dt_try > dt_ceiling) dt_try = dt_ceiling;
          if (deck.dt_max > 0 && dt_try > deck.dt_max) dt_try = deck.dt_max;
          if (dt_try > remaining) dt_try = remaining;
          res.dt = dt_try;
          nsub = 1 << 24;
        }

        // What a converged or failed sub-step does to the controller.  Both
        // the sequential and fully-implicit paths call these, so the two
        // cannot drift apart.
        //
        // Growth/decay is Flow's PIDAndIterationCount rule (the iteration
        // half; the failure ceiling below approximates its PID half):
        //
        //   used <= target: dt *= 1 + (target-used)/target * growth_damping
        //   used >  target: dt /= 1 + (used-target)/target * decay_damping
        //
        // The denominator is TARGET, not USED*damping.  The latter was the
        // old NVRS formula and turned Flow's 1 -> 3 day startup growth into
        // roughly 1 -> 1.3, changing the phase-appearance path before the
        // first Norne report.
        // The asymmetry is deliberate in Flow and worth keeping: a step that
        // converged easily is weak evidence that a much longer one would, while
        // a step that struggled is strong evidence that a shorter one is
        // needed, so the rule shrinks at full strength and grows at a third of
        // it.  DTMAXGROW then caps what any single success can do.
        auto on_success = [&](int iters_used, Real state_error,
                              Real controller_dt) {
          ++nsub_accepted;
          if (dt_keep) {
            accepted_dt.emplace_back((double)t_days + (double)t_done,
                                     (double)res.dt);
            t_done += res.dt;
          }
          ++nsubs_done;
          if (!itercount) return;
          nfail = 0;
          // Reaching the raised ceiling proves the probe. Until then retain
          // cap_probe_floor so an unrelated failure cannot erase the last
          // demonstrated stable length.
          if (cap_probe_floor > 0 &&
              res.dt >= (Real)0.98 * dt_ceiling) {
            if (rank == 0)
              std::printf("  [dt-probe] %.6g days accepted; promoted as the "
                          "new proven ceiling\n", (double)res.dt);
            cap_probe_floor = 0;
          }
          const double used = (double)(iters_used > 0 ? iters_used : 1);
          const double tgt = (double)(deck.dt_target > 0 ? deck.dt_target : 8);
          double f;
          if (used > tgt) {
            f = 1.0 / (1.0 + (used - tgt) / tgt *
                                (double)deck.dt_decay_damp);
          } else {
            f = 1.0 + (tgt - used) / tgt *
                          (double)deck.dt_grow_damp;
          }
          if (pid_active && deck.dt_pid_tol > 0) {
            const double tol = (double)deck.dt_pid_tol;
            // PIDTimeStepControl shifts its three-sample history before it
            // evaluates the estimate.  At this point NVRS has not shifted
            // yet, so the matching samples are old e1, old e2, current --
            // not old e0, old e1, current.  Using the latter makes both the
            // proportional and derivative factors one accepted step stale.
            const double e0 = std::max((double)dt_pid_error[1], 1.0e-30);
            const double e1 = std::max((double)dt_pid_error[2], 1.0e-30);
            const double e2 = std::max((double)state_error, 1.0e-30);
            double pid_f;
            if (e2 > tol) {
              pid_f = tol / e2;
            } else {
              pid_f = std::pow(e1 / e2, 0.075) *
                      std::pow(tol / e2, 0.175) *
                      std::pow((e1 * e1) / (e0 * e2), 0.01);
            }
            if (std::isfinite(pid_f)) f = std::min(f, pid_f);
            dt_pid_error[0] = dt_pid_error[1];
            dt_pid_error[1] = dt_pid_error[2];
            dt_pid_error[2] = state_error;
          }
          if (f > (double)deck.dt_max_grow) f = (double)deck.dt_max_grow;
          if (f < 0.1) f = 0.1;
          const bool probe_easy =
              iters_used <= std::max(1, deck.dt_cap_probe_iters) &&
              (!pid_active || state_error <= deck.dt_pid_tol);
          cap_easy_run = probe_easy ? cap_easy_run + 1 : 0;
          if (deck.dt_cap_probe > 0 && dt_ceiling > 0 &&
              cap_probe_floor <= 0 &&
              cap_easy_run >= deck.dt_cap_probe &&
              deck.dt_cap_probe_factor > 1) {
            cap_probe_floor = dt_ceiling;
            dt_ceiling *= deck.dt_cap_probe_factor;
            if (deck.dt_max > 0 && dt_ceiling > deck.dt_max)
              dt_ceiling = deck.dt_max;
            cap_easy_run = 0;
            if (rank == 0)
              std::printf("  [dt-probe] %d cheap solves: testing ceiling "
                          "%g -> %g days\n", deck.dt_cap_probe,
                          (double)cap_probe_floor, (double)dt_ceiling);
          }
          // A report boundary may truncate the physical step to the remaining
          // interval.  That is an output constraint, not nonlinear feedback:
          // retain the controller's untruncated request as the basis for the
          // next proposal.  Feeding the short remainder back here produced a
          // deterministic sawtooth in SPE10 (3.996 -> 1.534 -> ... -> 0.711
          // days) and paid extra Newton solves solely to land on output dates.
          // If the snap-to-boundary rule enlarged the physical step slightly,
          // use that demonstrated length instead.
          const Real accepted_base = std::max(controller_dt, res.dt);
          Real cand = accepted_base * (Real)f;
          // Earn the way back rather than jumping over the cliff.  The
          // ceiling binds here; how fast it relaxes depends on the grid.
          if (dt_ceiling > 0 && cand > dt_ceiling) cand = dt_ceiling;
          if (deck.dt_max > 0 && cand > deck.dt_max) cand = deck.dt_max;
          dt_try = cand;
          // A ceiling is evidence from one failed substep, not a sentence for
          // the whole report interval. Relax it after every accepted solve so
          // a sequence of cheap 1--2 Newton steps can recover smoothly, while
          // still preventing the immediate grow-back into the failed length.
          //
          // That is the corner-point (FIELDDECK) policy, validated on Norne
          // with DTCAPGROW 1.1.  A Cartesian deck keeps the once-per-report-
          // step relaxation at the end of the report step instead.  SPE10
          // leaves DTCAPGROW at its 1.2 default, and relaxing that per
          // substep regrew the ceiling into the failed length within two
          // successes: 8.08 d failed, the ceiling became 6.06 d, and
          // 6.06 * 1.2 * 1.2 = 8.73 d was tried and failed again.
          if (field && dt_ceiling > 0) {
            dt_ceiling *= deck.dt_cap_grow;
            if (deck.dt_max > 0 && dt_ceiling >= deck.dt_max) dt_ceiling = 0;
          }
        };
        // Returns true when the sub-step should be retried at the shorter
        // length, false when the attempt is over and the caller should end it.
        auto on_failure = [&](int iters_lost, int lin_lost) -> bool {
          nnewt_lost += iters_lost;
          nlin_lost += lin_lost;
          if (dt_keep)
            NVRS_CHECK_CUDA(cudaMemcpy(res.u, u_sub, sizeof(Real) * sys.n_ext,
                                       cudaMemcpyDeviceToDevice));
          if (dt_keep && gp_sub)
            NVRS_CHECK_CUDA(cudaMemcpy(res.gas_primary, gp_sub, gp_bytes,
                                       cudaMemcpyDeviceToDevice));
          if (dt_keep && cd_sub)
            NVRS_CHECK_CUDA(cudaMemcpy(res.clamp_dso, cd_sub, sizeof(Real) * (size_t)res.grid.n_local_ext(),
                                       cudaMemcpyDeviceToDevice));
          if (!itercount) return false;
          // The hierarchy was built for a pressure matrix this state no longer
          // produces, and the AIM rates describe fluxes that were never
          // reached.  Both are the lattice path's reasoning, unchanged.
          solver.invalidate_amg();
          if (deck.aim) aim_reset_implicit(res);
          const Real was = res.dt;
          // Remember the length that failed.  Without this the controller
          // grows straight back into it: on Norne a failure at 0.97 days was
          // followed two successes later by an attempt at 1.27.
          // Do not turn the two-phase startup homotopy into a permanent
          // stability bound.  SPE10 must first cross its sharp initial
          // pressure transient (1 -> .33 -> .1089 day); the .33-day failure
          // says nothing about the stable length after even one accepted
          // state.  Three-phase/NORNE retains its validated behaviour.
          const bool can_learn_cap =
              res.bs != 2 || res.accepted_state_id > 0;
          if (deck.dt_fail_cap > 0 && can_learn_cap) {
            Real cap = was * deck.dt_fail_cap;
            if (cap_probe_floor > 0) {
              cap = std::max(cap, cap_probe_floor);
              if (rank == 0)
                std::printf("  [dt-probe] %g days failed; restoring proven "
                            "ceiling %g days\n", (double)was, (double)cap);
              cap_probe_floor = 0;
              cap_easy_run = 0;
            }
            if (dt_ceiling <= 0 || cap < dt_ceiling) dt_ceiling = cap;
          }
          dt_try = res.dt * deck.dt_restart;
          ++nfail;
          const bool again = (nfail <= deck.max_chop) && (dt_try > (Real)1e-8);
          if (rank == 0)
            std::printf("  [chop] step %d: %g days did not converge; %s "
                        "(restart %d of %d, %g of %g days stand)\n",
                        step, (double)was,
                        again ? "retrying shorter" : "giving up",
                        nfail, deck.max_chop, (double)t_done, (double)dt_rep);
          if (again && rank == 0)
            std::printf("  [dt] step %d: dt %g -> %g days\n", step,
                        (double)was, (double)dt_try);
          return again;
        };
        if (chop > chop0) {
          // Roll back and retry on the shorter dt. Without DTKEEP that
          // means the whole report step; with it the state is already sitting
          // at the last converged sub-step boundary, put there when the
          // failure was detected, and there is nothing to undo here.
          if (!dt_keep)
            NVRS_CHECK_CUDA(cudaMemcpy(res.u, u_save, sizeof(Real) * sys.n_ext,
                                       cudaMemcpyDeviceToDevice));
          if (!dt_keep && gp_save)
            NVRS_CHECK_CUDA(cudaMemcpy(res.gas_primary, gp_save, gp_bytes,
                                       cudaMemcpyDeviceToDevice));
          if (!dt_keep)
            restore_history();
          if (!dt_keep && cd_save)
            NVRS_CHECK_CUDA(cudaMemcpy(res.clamp_dso, cd_save, sizeof(Real) * (size_t)res.grid.n_local_ext(),
                                       cudaMemcpyDeviceToDevice));
          // The rollback throws the state away, so any AMG hierarchy held over
          // from the failed attempt was built for a pressure matrix that no
          // longer exists.  Retrying on a shorter dt with it would start the
          // retry from the preconditioner that just failed to get anywhere.
          solver.invalidate_amg();
          // The rates measured during the discarded attempt describe a state
          // that no longer exists; go back to fully implicit and re-measure
          // once the retry has a converged substep of its own.
          if (deck.aim) aim_reset_implicit(res);
          // The retry is the interesting assembly, not the quiet first one.
          // Open the diagnostic budget so the step that is actually in trouble
          // reports what it sees; a budget spent on step 1 says nothing about
          // the step that failed.
          if (rank == 0) res.diag_budget = 1;
          if (rank == 0)
            std::printf("  [chop] step %d did not converge; retrying with "
                        "dt = %g days in %d sub-steps\n",
                        step, (double)res.dt, nsub);
          if (rank == 0 && dt_keep && t_done > 0)
            std::printf("  [chop] step %d: keeping %g of %g days already "
                        "converged; retrying the remainder\n",
                        step, (double)t_done, (double)dt_rep);
          // And say WHERE the residual is.  A norm cannot distinguish a
          // single runaway well from an equation that is off everywhere,
          // and guessing between those has cost this step two wrong
          // diagnoses.
          if (rank == 0 && field)
            report_worst_cells(res, sys.b, field.get());
          // The counters describe work that still stands when the prefix
          // is kept, so they are only cleared when it is not -- but what they
          // held was really spent, so it moves to the run total rather than
          // disappearing.
          if (!dt_keep) {
            nnewt_lost += nnewt;
            nlin_lost += nlin;
            nnewt = nlin = 0;
          }
        }

        // Trace the FIRST retry of a step that failed, and only its first
        // sub-step.  A step that converges says nothing worth reading; the
        // one just chopped is the only place the answer is, and waiting for
        // the sixth halving to repeat it five more times is noise.  The full
        // --verbose trace also dumps the matrix to disk every iteration,
        // which is not what is wanted here.
        const bool trace_retry = (rank == 0 && chop == chop0 + 1);
        bool all_subs_ok = true;
        for (int sub = 0; sub < nsub && all_subs_ok; ++sub) {
          Real controller_dt = res.dt;
          if (itercount) {
            const Real left = dt_rep - t_done;
            if (left <= (Real)1e-12 * dt_rep) break;
            controller_dt = dt_try;
            res.dt = (dt_try < left) ? dt_try : left;
            // Land on the report boundary rather than leaving a sliver behind
            // it.  A tenth of a step is cheaper to absorb into this one than to
            // pay a whole Newton solve for.
            if (left - res.dt < (Real)0.1 * res.dt) res.dt = left;
            // DTSEQFILE: run to the next imposed end.  After a failure keep
            // the chopped length, but never step past that end.
            if (!dtseq_ends.empty() && dt_keep) {
              const double tnow = (double)t_days + (double)t_done;
              const auto it = std::upper_bound(dtseq_ends.begin(),
                                               dtseq_ends.end(), tnow + 1e-7);
              if (it != dtseq_ends.end()) {
                const Real to_end = (Real)(*it - tnow);
                const Real want = nfail == 0 ? to_end
                                             : std::min(dt_try, to_end);
                controller_dt = want;
                res.dt = (want < left) ? want : left;
              }
            }
          }
          // The boundary this sub-step starts from. If it fails, this is what
          // the state returns to, and everything before it stands.
          if (dt_keep)
            NVRS_CHECK_CUDA(cudaMemcpy(u_sub, res.u, sizeof(Real) * sys.n_ext,
                                       cudaMemcpyDeviceToDevice));
          if (dt_keep && gp_sub)
            NVRS_CHECK_CUDA(cudaMemcpy(gp_sub, res.gas_primary, gp_bytes,
                                       cudaMemcpyDeviceToDevice));
          if (dt_keep && cd_sub)
            NVRS_CHECK_CUDA(cudaMemcpy(cd_sub, res.clamp_dso, sizeof(Real) * (size_t)res.grid.n_local_ext(),
                                       cudaMemcpyDeviceToDevice));
          // advance one sub-step: freeze u^n and refresh pressure-dependent PVT
          NVRS_CHECK_CUDA(cudaMemcpy(res.u_old, res.u, sizeof(Real) * sys.n,
                                     cudaMemcpyDeviceToDevice));
          if (res.gas_primary_old)
            NVRS_CHECK_CUDA(cudaMemcpy(res.gas_primary_old, res.gas_primary,
                                       gp_bytes, cudaMemcpyDeviceToDevice));
          if (deck.aim) {
            // Explicit donors read the neighbour's u_old across rank seams.
            // The flags come from rates measured at the last CONVERGED state
            // (none yet on the first substep, which therefore runs FIM).
            // res.grid is this rank's cell count presented as a 1-D run,
            // with rank 0 of 1 -- so the structured exchange returns without
            // sending anything and the donor ghosts stay stale.  Take the
            // connection halo where there is one.
            if (res.halo_conn)
              static_cast<HaloConn*>(res.halo_conn)->exchange(res.u_old, res.bs);
            else
              halo_exchange(res.u_old, res.bs, res.grid);
            aim_update_flags(res, sub == 1 ? 1 : 0);
          }
          refresh_pvt(res);

          bool converged = false;
          // The last iteration's state, kept so a failure can say what it was
          // still failing on rather than only that it failed.
          Real fail_mb = 0, fail_signed_mb = 0, fail_cnv = 0, fail_rel = 0;
          bool fail_mb_ok = false, fail_cnv_ok = false, fail_stalled = false;
          // What this sub-step alone cost.  nnewt accumulates over the report
          // step, and the controller wants the length of the last solve, not
          // the total.
          const int nnewt_sub0 = nnewt;
          const int nlin_sub0 = nlin;
          const bool nw_unused_first = (sub == 0);
          if (sequential) {
            // Reference scheme: pressure solve, Darcy velocities, then an
            // implicit fractional-flow saturation update per phase.
            Real seq_res = 0;
            converged = seq_step(res, deck, seq, solver, res.dt, &nnewt, &nlin,
                                 &seq_res);
            if (nw_unused_first) { r0 = seq_res + 1e-30; }
            rlast = seq_res;
            if (!converged) {
              if (on_failure(nnewt - nnewt_sub0, nlin - nlin_sub0)) continue;
              all_subs_ok = false;
            } else {
              const Real change = (pid_active || deck.dt_cap_probe > 0)
                                      ? blackoil_relative_change(res, u_sub)
                                      : (Real)0;
              on_success(nnewt - nnewt_sub0, change, controller_dt);
              ++res.accepted_state_id;
            }
            continue;
          }
          // Previous iteration's residual, to tell progress from stalling.
          Real rprev = 0;
          Real merit_prev = std::numeric_limits<Real>::infinity();
          // Flow watches component residuals for a two-cycle, then damps later
          // corrections in this timestep by 0.1 at a time, down to 0.5.
          std::vector<std::array<Real, 3>> phase_residual_history;
          const char* oscillation_env =
              std::getenv("NVRS_OSCILLATION_DAMPING");
          const bool oscillation_damping = res.bs == 2 &&
              (!oscillation_env || std::atoi(oscillation_env) != 0);
          Real update_relaxation = 1.0;

          // Consecutive iterations that did not reduce the residual.  See
          // NEWTONSTALLMAX: the step is lost long before NEWTONMAX says so.
          int stall_run = 0;
          // An accepted line-search trial has already assembled the exact
          // state the next Newton iteration starts from.  Hand it forward
          // instead of repeating well control and the same assembly.
          bool have_trial_assembly = false;
          Real trial_assembly_rnorm = 0;
          for (int nw = 0; nw < deck.newton_max; ++nw) {
            Real rnorm;
            if (have_trial_assembly) {
              rnorm = trial_assembly_rnorm;
              have_trial_assembly = false;
              ++nline_assembly_handoffs;
            } else {
              // Re-solve each rate-controlled well's flowing pressure against
              // the CURRENT state, before assembling with it.
              //
              // Doing this once per report step froze the mobilities at the
              // start of the step: the pressure that met the target then no
              // longer meets it thirty days later, and the well delivers
              // whatever the drift gives.  OPM carries the well equation inside
              // the coupled system, which is the same statement taken to its
              // limit -- the control is satisfied AT the solution rather than
              // at the state the step began in.
              //
              // Re-solving every Newton iteration is the explicit version of
              // that: a fixed-point iteration on the well constraint alongside
              // Newton on the reservoir.  It costs one device-to-host copy of
              // the state per iteration and converges with it.  WELLITER 0
              // restores the old once-per-step behaviour.
              if (field && deck.well_iter)
              {
                const auto twc = std::chrono::high_resolution_clock::now();
                // Injector rate control is the same nonlinear constraint as a
                // producer target. Recompute its common BHP and completion
                // allocation at the current state; otherwise the start-of-step
                // injectivity is frozen and the assembly clips a well such as
                // Norne C-1H as its gas/water mobility changes through the step.
                apply_injector_rate_targets(res, *field, sched_step_now, deck,
                                            &pvt_host, true);
                apply_producer_rate_targets(res, *field, sched_step_now, deck,
                                            &pvt_host);
                t_well_control += std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - twc).count();
              }
              auto ta = tic();
              rnorm = (deck.model == Model::Compositional)
                          ? assemble_compositional(res, sys)
                          : assemble_blackoil(res, sys);
              t_assemble += toc(ta);
            }
            rlast = rnorm;
            // Say what the pressure operator looks like before the first solve
            // of the run.  It costs one copy back, once, and it is the
            // difference between diagnosing a stalled Krylov method and
            // guessing at it.  NVRS_CPR_DIAG=n reports the first n assemblies.
            if (rank == 0 && deck.model != Model::Compositional) {
              char what[64];
              std::snprintf(what, sizeof what, "step %d, Newton %d", step, nw);
              report_pressure_operator(sys, what);
            }
            // A non-finite residual cannot be recovered by more iterations, but
            // it can often be recovered by a shorter timestep — so treat it as
            // a failed attempt rather than a fatal error, and only give up once
            // the chopping budget is exhausted.
            if (!std::isfinite((double)rnorm)) {
              if (rank == 0)
                std::printf("  [chop] non-finite residual at step %d, sub %d, "
                            "newton %d\n", step, sub, nw);
              // Leave converged false and let the outcome handler below decide.
              // Setting all_subs_ok here as well would end the sub loop even
              // when the controller wanted to retry at a shorter length.
              break;
            }
            // Refresh the double-word (df64) hi/lo pairs for the emulated-FP64 SpMV.
            if (deck.emulate_fp64) split_operator_df64(sys);
            // DUMPOP step newton: the operator and the CPR pressure matrix as
            // they are at this Newton iteration, for tests/bspmv_bench.cu (the
            // kernels on the real Norne pattern) and scripts/cond_estimate.py
            // (cond(A_p) and cond(A_p) * u32).  One rank only.
            if (step == deck.dump_op_step && nw == deck.dump_op_newton &&
                !dumped_op) {
              dumped_op = true;
              if (nranks == 1) {
                dump_operator(deck.out_dir, sys, step, nw);
              } else if (rank == 0) {
                std::printf("  [warn] DUMPOP needs one rank; skipped\n");
              }
            }
            if (nw == 0) r0 = rnorm + 1e-30;
            Real rel = rnorm / r0;
            // Material balance, in OPM's sense: the absolute SIGNED sum of
            // each component residual as a fraction of fluid in place. The
            // per-equation L2 norm is retained for diagnostics, but is not
            // Flow's MB criterion: local residuals which cancel physically
            // must cancel in this field balance too.
            Real mb = 0, signed_mb = 0, cnv = 0;
            const bool bo_model = (deck.model != Model::Compositional);
            // OPMCONV's two criteria are built from the pore volume, the CNV
            // maxima and the signed component sums that the assembly reduces
            // alongside the residual.  Only the connection-list kernel fills
            // those slots; the three fixed-stencil kernels leave them zero.
            // A zero pore volume therefore means "not measured", not "no
            // fluid", and the criteria must not be read as satisfied -- with
            // the L2 fallback disabled below, a Cartesian deck would accept
            // every step at Newton 1 with the residual untouched.
            const bool opm_active =
                deck.opm_conv && bo_model && res.pv_local > 0;
            if (!opm_active && deck.opm_conv && bo_model && rank == 0 &&
                !warned_opmconv) {
              warned_opmconv = true;
              std::printf("  [warn] OPMCONV was requested, but this grid is "
                          "assembled by a kernel that does not compute the "
                          "OPM convergence metrics. Falling back to the "
                          "relative-residual test (NEWTONTOL) for this run.\n");
            }
            if (deck.mb_tol > 0 && bo_model && res.pv_local > 0) {
              const Real bph[3] = {(Real)1.0 / res.bo, (Real)1.0 / res.bw,
                                   (Real)1.0 / res.bg};
              for (int a = 0; a < 3; ++a) {
                const Real scale = res.pv_local * bph[a];
                if (scale > 0)
                  mb = std::max(mb, res.res_eq[a] * res.dt / scale);
                if (scale > 0)
                  signed_mb = std::max(
                      signed_mb, std::abs(res.res_sum[a]) * res.dt / scale);
                cnv = std::max(cnv, res.cnv[a]);
              }
            }
            // OPM's test, when asked for: a component passes on the strict
            // tolerance OR on the relaxed one, provided the cells that violate
            // hold a small enough share of the pore volume.  Checked every
            // iteration rather than only after a stall, which is also what OPM
            // does and is why our Newton counts run long.
            bool cnv_ok_opm = true;
            Real cnv_tol_eff = deck.cnv_tol;
            // Flow relaxes both tests on its last Newton iteration
            // (ToleranceMbRelaxed, ToleranceCnvRelaxed); MBTOLRELAXED > 0
            // does the same here.
            const bool final_relax = opm_active && deck.mb_tol_relaxed > 0 &&
                                     nw == deck.newton_max - 1;
            const Real mb_tol_eff =
                final_relax ? std::max(deck.mb_tol, deck.mb_tol_relaxed)
                            : deck.mb_tol;
            if (opm_active) {
              // Flow classifies a cell by its worst component.  Relax every
              // component only when the UNION of strict-CNV-violating cells
              // occupies less than RelaxedMaxPvFraction.  Three independent
              // fractions could incorrectly relax three disjoint regions.
              const Real bad_frac = (res.pv_local > 0)
                                        ? res.cnv_bad_pv / res.pv_local
                                        : (Real)0;
              const bool use_relaxed = bad_frac < deck.cnv_relax_pv ||
                                       final_relax;
              cnv_tol_eff = use_relaxed ? deck.cnv_relaxed : deck.cnv_tol;
              for (int a = 0; a < 3; ++a) {
                const Real tol = cnv_tol_eff;
                if (!(res.cnv[a] < tol)) { cnv_ok_opm = false; break; }
              }
            }
            phase_residual_history.push_back(
                {res.cnv[0], res.cnv[1], res.cnv[2]});
            bool oscillating = false;
            if (oscillation_damping && deck.opm_conv &&
                phase_residual_history.size() >= 3) {
              const auto& f0 = phase_residual_history[phase_residual_history.size() - 1];
              const auto& f1 = phase_residual_history[phase_residual_history.size() - 2];
              const auto& f2 = phase_residual_history[phase_residual_history.size() - 3];
              for (int a = 0; a < res.bs && a < 3; ++a) {
                const Real den = std::max(std::abs(f0[a]), (Real)1e-300);
                const Real d1 = std::abs(f0[a] - f2[a]) / den;
                const Real d2 = std::abs(f0[a] - f1[a]) / den;
                if (d1 < (Real)0.2 && d2 > (Real)0.2) {
                  oscillating = true;
                  break;
                }
              }
              if (oscillating) {
                update_relaxation = std::max((Real)0.5,
                                             update_relaxation - (Real)0.1);
                if ((verbose || trace_retry) && rank == 0)
                  std::printf("               Flow oscillation damping: %.2f\n",
                              (double)update_relaxation);
              }
            }

            // OPMCONV is a complete convergence policy, not an additional
            // early-exit beside the legacy relative L2 ratio. Previously the
            // L2 exit could accept a coherent water residual even when MB and
            // CNV failed. Gas dominates the combined norm, so that defect was
            // nearly invisible there and accumulated into the missing Norne
            // water bank after roughly half the history.
            const bool by_rel =
                !opm_active &&
                ((rel < deck.newton_tol) || (rnorm < 1e-8));
            const Real opm_merit =
                (opm_active && deck.mb_tol > 0 &&
                 cnv_tol_eff > 0)
                    ? std::max(signed_mb / deck.mb_tol, cnv / cnv_tol_eff)
                    : std::numeric_limits<Real>::infinity();
            const Real nonlinear_ratio =
                (nw > 0 && rprev > 0) ? rnorm / rprev : (Real)1;
            // The abort policy must watch the same scaled component criteria
            // as convergence. Raw L2 is gas-dominated and can stall while the
            // signed water balance is still improving.
            const bool stalled = (nw > 0) &&
                (opm_active
                     ? !(opm_merit < (Real)0.999 * merit_prev)
                     : (rnorm > deck.newton_stall * rprev));
            // A sufficiently short step can satisfy MB and CNV at Newton zero
            // because both scale with dt. Require at least one Newton update
            // so time cannot advance merely because dt was chopped small.
            const bool mb_ok = (nw > 0) && signed_mb < mb_tol_eff;
            const bool cnv_ok = opm_active
                                    ? cnv_ok_opm
                                    : (deck.cnv_tol <= 0 || cnv < deck.cnv_tol);
            // Flow evaluates MB+CNV every Newton iteration. Requiring an L2
            // stall here both wasted already-converged iterations and made the
            // later relative-L2 bypass more likely to choose the answer.
            bool by_opm = opm_active && mb_ok && cnv_ok;
            // MBTOLRELAXED: Flow accepts MB < ToleranceMbRelaxed and CNV <
            // ToleranceCnvRelaxed on its last iteration, so no state it
            // accepts is worse than that.  When NEWTONSTALLMAX is about to
            // abandon this attempt, a state inside that envelope is accepted
            // here instead of being thrown away and the step chopped: the
            // same acceptance bound as Flow, reached without grinding out the
            // remaining iterations of a stalled solve.
            if (!by_opm && opm_active && deck.mb_tol_relaxed > 0 &&
                deck.newton_stall_max > 0 && stalled &&
                stall_run + 1 >= deck.newton_stall_max && nw > 0) {
              bool relaxed_ok =
                  signed_mb < std::max(deck.mb_tol, deck.mb_tol_relaxed);
              for (int a = 0; a < 3 && relaxed_ok; ++a)
                relaxed_ok = res.cnv[a] < deck.cnv_relaxed;
              by_opm = relaxed_ok;
            }
            // Count the stall before rprev is overwritten.  A step that is
            // still descending has not stalled however slowly it moves.
            if (stalled) ++stall_run; else stall_run = 0;
            fail_mb = mb; fail_signed_mb = signed_mb;
            fail_cnv = cnv; fail_rel = rel;
            fail_mb_ok = mb_ok; fail_cnv_ok = cnv_ok; fail_stalled = stalled;
            rprev = rnorm;
            merit_prev = opm_merit;
            if (rank == 0)
              if (FILE* rl = reslog_file(0))
                std::fprintf(rl, "%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%d\n", step, sub,
                             nw, (double)rnorm, (double)rel, (double)mb,
                             (double)cnv, (by_rel || by_opm) ? 1 : 0);
            if (by_rel || by_opm) {
              converged = true;
              conv_mb = mb; conv_signed_mb = signed_mb; conv_cnv = cnv;
              conv_how = by_rel ? "rel" : "mb+cnv";
              break;
            }

            // Not converging and no longer descending: the remaining
            // iterations cost what a useful one costs and buy nothing, and the
            // controller is going to retry shorter regardless.
            if (deck.newton_stall_max > 0 &&
                stall_run >= deck.newton_stall_max) {
              if (rank == 0)
                std::printf("  [why] step %d sub %d abandoned at Newton %d: "
                            "residual stalled for %d iteration(s) at rel "
                            "%.3e\n", step, sub, nw, stall_run, (double)rel);
              break;
            }

            if (verbose && rank == 0) dump_system(sys, step, nw);

            // Eisenstat--Walker-style forcing, bounded below by LINTOL. A
            // distant Newton iterate does not need a more accurate linear
            // correction than its nonlinear model can justify; as nonlinear
            // progress improves, return smoothly to the validated tolerance.
            Real linear_tol = deck.lin_tol;
            if (deck.lin_forcing_max > deck.lin_tol) {
              const Real ew = (nw == 0)
                                  ? deck.lin_forcing_max
                                  : (Real)0.5 * nonlinear_ratio * nonlinear_ratio;
              linear_tol = std::min(deck.lin_forcing_max,
                                    std::max(deck.lin_tol, ew));
            }
            solver.set_tolerance(linear_tol);
            auto tl = tic();
            solver.setup(sys);
            NVRS_CHECK_CUDA(cudaMemset(du, 0, sizeof(Real) * sys.n_ext));
            SolveStats st = solver.solve(sys, du, true);
            t_linear += toc(tl);
            nlin += st.iters;
            if (rank == 0)
              if (FILE* rl = reslog_file(1))
                std::fprintf(rl, "%d,%d,%d,%d,%.6e,%d,%.6e\n", step, sub, nw,
                             st.iters, (double)st.final_residual,
                             st.converged ? 1 : 0, (double)st.sol_norm);

            // Per-iteration trace.  When a step diverges the useful question is
            // whether the *linear* solve failed (large final residual, hit the
            // iteration cap) or succeeded and handed back a huge update (a bad
            // Jacobian).  Printing both separates those two cases immediately.
            if ((verbose || trace_retry) && rank == 0)
              std::printf(
                  "    sub %2d newton %2d  |R|=%.4e  rel=%.3e  mb=%.2e "
                  "cnv=%.2e  lin: %3d iters, rel %.2e%s   |du|=%.4e\n",
                  sub, nw, (double)rnorm, (double)rel, (double)mb, (double)cnv,
                  st.iters, (double)st.final_residual,
                  st.converged ? "" : " NOT CONVERGED", (double)st.sol_norm);
            if ((verbose || (trace_retry && sub == 0)) && rank == 0 &&
                deck.model != Model::Compositional)
              std::printf("               oil %.4e  water %.4e  gas %.4e\n",
                          (double)res.res_eq[0], (double)res.res_eq[1],
                          (double)res.res_eq[2]);
            if (rank == 0 && !st.converged && !warned_linear) {
              std::printf(
                  "  [warn] the linear solver did not reach its tolerance "
                  "%.2e (rel %.2e "
                  "after %d iterations) at step %d, Newton %d. Newton is being "
                  "advanced with an inexact update; raise LINMAXIT or loosen "
                  "LINTOL.\n",
                  (double)linear_tol, (double)st.final_residual, st.iters,
                  step, nw);
              warned_linear = true;
            }

            if (deck.line_search <= 0) {
              newton_update(res, du, update_relaxation);
            } else {
              // Backtrack the fully implicit update against the actual
              // nonlinear residual.  Appleyard/DSMAX make newton_update
              // nonlinear in alpha, so each trial starts from an identical
              // reservoir state (and identical clamp diagnostic), rather
              // than successively scaling an already clipped update.
              NVRS_CHECK_CUDA(cudaMemcpy(u_newton, res.u,
                                         sizeof(Real) * sys.n_ext,
                                         cudaMemcpyDeviceToDevice));
              if (gp_newton) NVRS_CHECK_CUDA(cudaMemcpy(
                  gp_newton, res.gas_primary, gp_bytes,
                  cudaMemcpyDeviceToDevice));
              if (cd_newton)
                NVRS_CHECK_CUDA(cudaMemcpy(
                    cd_newton, res.clamp_dso,
                    sizeof(Real) * (size_t)res.grid.n_local_ext(),
                    cudaMemcpyDeviceToDevice));

              Real alpha = 1.0;
              Real rtrial = rnorm;
              Real best_alpha = 0;
              Real best_score = std::numeric_limits<Real>::infinity();
              int halvings = 0;
              bool accepted = false;
              for (; halvings <= deck.line_search; ++halvings) {
                ++nline_trials;
                NVRS_CHECK_CUDA(cudaMemcpy(res.u, u_newton,
                                           sizeof(Real) * sys.n_ext,
                                           cudaMemcpyDeviceToDevice));
                if (gp_newton) NVRS_CHECK_CUDA(cudaMemcpy(
                    res.gas_primary, gp_newton, gp_bytes,
                    cudaMemcpyDeviceToDevice));
                if (cd_newton)
                  NVRS_CHECK_CUDA(cudaMemcpy(
                      res.clamp_dso, cd_newton,
                      sizeof(Real) * (size_t)res.grid.n_local_ext(),
                      cudaMemcpyDeviceToDevice));
                newton_update(res, du, alpha * update_relaxation);
                // A line-search residual must use the well controls belonging to
                // the trial state. Injector BHP limits are active-set constraints,
                // and even smooth producer common-BHP allocations change the long-
                // term state path enough to collapse later timesteps if frozen.
                // Keep the same injector-then-producer ordering used at the top of
                // the nonlinear iteration.
                if (field && deck.well_iter) {
                  const auto twc = std::chrono::high_resolution_clock::now();
                  apply_injector_rate_targets(res, *field, sched_step_now,
                                              deck, &pvt_host, true);
                  apply_producer_rate_targets(res, *field, sched_step_now,
                                              deck, &pvt_host, true);
                  t_well_control += std::chrono::duration<double>(
                      std::chrono::high_resolution_clock::now() - twc).count();
                }
                auto tls = tic();
                rtrial = (deck.model == Model::Compositional)
                             ? assemble_compositional(res, sys)
                             : assemble_blackoil(res, sys);
                t_assemble += toc(tls);
                Real trial_merit = std::numeric_limits<Real>::infinity();
                Real trial_cnv_metric = std::numeric_limits<Real>::infinity();
                if (opm_active && deck.mb_tol > 0) {
                  Real trial_signed = 0, trial_cnv = 0;
                  const Real bph_trial[3] = {
                      (Real)1.0 / res.bo, (Real)1.0 / res.bw,
                      (Real)1.0 / res.bg};
                  for (int a = 0; a < 3; ++a) {
                    const Real scale = res.pv_local * bph_trial[a];
                    if (scale > 0)
                      trial_signed = std::max(
                          trial_signed,
                          std::abs(res.res_sum[a]) * res.dt / scale);
                    trial_cnv = std::max(trial_cnv, res.cnv[a]);
                  }
                  trial_cnv_metric = trial_cnv;
                  const Real bad_frac = res.cnv_bad_pv / res.pv_local;
                  const Real trial_cnv_tol =
                      bad_frac < deck.cnv_relax_pv ? deck.cnv_relaxed
                                                   : deck.cnv_tol;
                  if (trial_cnv_tol > 0)
                    trial_merit = std::max(trial_signed / deck.mb_tol,
                                           trial_cnv / trial_cnv_tol);
                }
                const char* cnv_env = std::getenv("NVRS_LINE_CNV_DESCENT");
                const bool require_raw_cnv = res.bs == 2 &&
                    (!cnv_env || std::atoi(cnv_env) != 0);
                // Guard raw CNV while it exceeds Flow's relaxed tolerance.
                // Below that threshold material balance is dominant, and
                // demanding monotonically decreasing CNV can reject the
                // pressure update which is actually finishing the solve.
                const bool guard_raw_cnv =
                    opm_active && require_raw_cnv && cnv > deck.cnv_relaxed;
                const bool descends = opm_active
                    ? (trial_merit < opm_merit &&
                       (!guard_raw_cnv || trial_cnv_metric < cnv))
                    : rtrial < rnorm;
                Real trial_score = opm_active
                                       ? trial_merit
                                       : rtrial / std::max(rnorm, (Real)1e-300);
                if (guard_raw_cnv)
                  trial_score = std::max(
                      trial_score,
                      trial_cnv_metric / std::max(cnv, (Real)1e-300));
                if (std::isfinite((double)trial_score) &&
                    trial_score < best_score) {
                  best_score = trial_score;
                  best_alpha = alpha;
                }
                if (std::isfinite((double)rtrial) && descends) {
                  accepted = true;
                  break;
                }
                alpha *= (Real)0.5;
              }
              if (!accepted) {
                ++nline_no_descent;
                const Real last_alpha = alpha * (Real)2.0;
                alpha = (deck.line_best && best_alpha > 0)
                            ? best_alpha : last_alpha;
                NVRS_CHECK_CUDA(cudaMemcpy(res.u, u_newton,
                                           sizeof(Real) * sys.n_ext,
                                           cudaMemcpyDeviceToDevice));
                if (gp_newton) NVRS_CHECK_CUDA(cudaMemcpy(
                    res.gas_primary, gp_newton, gp_bytes,
                    cudaMemcpyDeviceToDevice));
                if (cd_newton)
                  NVRS_CHECK_CUDA(cudaMemcpy(
                      res.clamp_dso, cd_newton,
                      sizeof(Real) * (size_t)res.grid.n_local_ext(),
                      cudaMemcpyDeviceToDevice));
                newton_update(res, du, alpha * update_relaxation);
              }
              const int used_halvings =
                  std::min(halvings, deck.line_search);
              if ((trace_retry || verbose) && rank == 0 && used_halvings > 0)
                std::printf("               line search: alpha %.5g after "
                            "%d halving(s), |Rtrial| %.4e%s%s\n",
                            (double)(alpha * update_relaxation), used_halvings,
                            (double)rtrial,
                            accepted ? "" : " (no descending trial)",
                            (!accepted && deck.line_best && best_alpha > 0)
                                ? " (best finite trial retained)" : "");
              if (accepted) {
                have_trial_assembly = true;
                trial_assembly_rnorm = rtrial;
              }
            }
            ++nnewt;
          }
          if (!converged) {
            // Why it failed, not just that it did.  Two changes to the
            // timestep controller were made on guesses about this and both
            // were wrong; the answer is cheap to print and was not being
            // printed anywhere the failing path could reach.
            if (rank == 0) {
              const char* blocker =
                  !fail_stalled  ? "still descending, never stalled"
                  : !fail_mb_ok  ? "material balance"
                  : !fail_cnv_ok ? "CNV"
                                 : "relative residual";
              std::printf("  [why] step %d sub %d failed after %d Newton at "
                          "dt %g d: blocked on %s  (rel %.3e tol %.1e, "
                          "l2mb %.3e, signed %.3e tol %.1e, "
                          "cnv %.3e tol %.1e)\n",
                          step, sub, nnewt - nnewt_sub0, (double)res.dt,
                          blocker, (double)fail_rel, (double)deck.newton_tol,
                          (double)fail_mb, (double)fail_signed_mb,
                          (double)deck.mb_tol, (double)fail_cnv,
                          (double)deck.cnv_tol);
              if (deck.opm_conv && res.pv_local > 0)
                std::printf("  [why] step %d sub %d CNV pore-volume share: "
                            "union %.4f; oil %.4f water %.4f gas %.4f "
                            "(relaxed under %g)\n",
                            step, sub,
                            (double)(res.cnv_bad_pv / res.pv_local),
                            (double)(res.cnv_pv[0] / res.pv_local),
                            (double)(res.cnv_pv[1] / res.pv_local),
                            (double)(res.cnv_pv[2] / res.pv_local),
                            (double)deck.cnv_relax_pv);
              if (deck.opm_conv)
                std::printf("  [why] signed residual rates [oil water gas] = "
                            "[%+.6e %+.6e %+.6e] surface units/day\n",
                            (double)res.res_sum[0],
                            (double)res.res_sum[1],
                            (double)res.res_sum[2]);
              // Only for the first failure of a report step: it copies
              // back from the device, and step 51 failed ten times.
              if (field && nfail == 0)
                report_worst_cells(res, sys.b, field.get());
            }
            // Undo only this sub-step. The converged prefix is what makes an
            // overshoot cheap, and cheap overshoots are what let the step
            // length grow.
            if (on_failure(nnewt - nnewt_sub0, nlin - nlin_sub0)) continue;

            all_subs_ok = false;
          } else {
            // The converged residual is accumulation + production - injection
            // - internal flux divergence. Summed over the field, internal
            // faces cancel, so this is a cheap exact ledger of what the
            // nonlinear acceptance left unconserved in this substep.
            if (residual_balance_active)
              for (int a = 0; a < 3; ++a)
                residual_balance[a] +=
                    (double)res.res_sum[a] * (double)res.dt;
            // DRSDT, on the converged sub-step state and nowhere else.
            //
            // The ceiling is a function of the state at the END of an accepted
            // step.  Moving it inside the Newton loop would make the residual
            // chase a target that shifts under it, which is a good way to stop
            // a solver converging without changing any equation it reports.
            drsdt_update(res, res.dt);
            vappars_update(res);
            hyst_update(res);
            // The exact per-substep ledger.  It feeds the cumulative summary
            // vectors only where exact_well_cumulative is established (the
            // two-phase field path); NVRS_SUBSTEP_BALANCE also accumulates it
            // on three-phase runs, for the [substep-balance] audit line, so a
            // closure can be measured against the discrete sources the
            // residual actually saw rather than report-step trapezoids.
            if (exact_well_cumulative || substep_balance) {
              sub_qo += (double)res.well_prod[0] * (double)res.dt;
              sub_qwp += (double)res.well_prod[1] * (double)res.dt;
              sub_qwi += (double)res.well_inj[1] * (double)res.dt;
              sub_qgp += (double)res.well_prod[2] * (double)res.dt;
              sub_qgi += (double)res.well_inj[2] * (double)res.dt;
            }
            const Real change = (pid_active || deck.dt_cap_probe > 0)
                                    ? blackoil_relative_change(res, u_sub)
                                    : (Real)0;
            on_success(nnewt - nnewt_sub0, change, controller_dt);
            ++res.accepted_state_id;
            // AIM: the state is converged, so the fluxes in it are real.  This
            // is the only place the throughput rates can honestly be measured;
            // the next substep's flags come from them.
            if (deck.aim) aim_measure_rates(res);
          }
        }
        // Under the controller the sub loop ends when the report interval is
        // covered, which all_subs_ok does not describe: a sub-step may have
        // failed, been retried shorter, and succeeded.
        step_ok = itercount ? (t_done >= dt_rep - (Real)1e-9 * dt_rep)
                            : all_subs_ok;
        if (step_ok && itercount) {
          // The controller has already chosen it; carry it across the report
          // boundary rather than re-deriving one from the lattice.
          dt_next = dt_try;
          // Cartesian decks: one relaxation per report step, so a ceiling
          // learned from a failure survives long enough to mean something.
          // FIELDDECK runs relax per substep instead; see on_success.
          if (!field && dt_ceiling > 0) {
            dt_ceiling *= deck.dt_cap_grow;
            if (deck.dt_max > 0 && dt_ceiling >= deck.dt_max) dt_ceiling = 0;
          }
          if (rank == 0)
            std::printf("  [dt] step %d covered in %d sub-step(s); next dt "
                        "%g days\n", step, nsubs_done, (double)dt_next);
        } else if (step_ok) {
          // Only lengthen the step after DTGROW consecutive steps that
          // converged on the first attempt.  Relaxing after every success
          // meant a step that had just been chopped immediately tried twice
          // that length again, so dt oscillated and each overshoot cost a
          // full discarded attempt.
          if (chop == chop0) {
            if (++clean_steps >= std::max(1, deck.dt_grow)) {
              // How much headroom did the step have?
              //
              // Relaxing exactly one halving discards the only signal the step
              // just produced.  A report step that converged in two Newton
              // iterations per sub-step against a target of eight is saying
              // the length is four times shorter than it needs to be, and
              // under the one-halving rule it took two more clean steps to act
              // on half of that.  Meanwhile a single failure drops six or
              // eight halvings at once.  The descent was geometric and the
              // ascent linear, which is why the Norne run parked six halvings
              // below its report steps and stayed there.
              //
              // DTRELAX caps how far one success may recover.  At its default
              // of 1 this is exactly the old rule.
              int relax = 1;
              const int nsub_done = 1 << chop;
              if (deck.dt_relax > 1 && nnewt > 0 && deck.dt_target > 0) {
                const double per_sub =
                    (double)nnewt / (double)std::max(1, nsub_done);
                if (per_sub > 0.0 && per_sub < (double)deck.dt_target) {
                  const double ratio = (double)deck.dt_target / per_sub;
                  relax = 1 + (int)std::floor(std::log2(ratio));
                  if (relax > deck.dt_relax) relax = deck.dt_relax;
                  if (relax < 1) relax = 1;
                }
              }
              chop_hint = (chop > relax) ? chop - relax : 0;
              clean_steps = 0;
            } else {
              chop_hint = chop;
            }
          } else {
            // Needed chopping to get here; hold this length rather than
            // immediately asking for twice it again.
            chop_hint = chop;
            clean_steps = 0;
          }
          if (rank == 0 && chop > 0)
            std::printf("  [dt] step %d converged at dt = %g days; next step "
                        "starts at %g\n", step,
                        (double)(dt_rep / (1 << chop)),
                        (double)(dt_rep / (1 << chop_hint)));
        }
      }

      if (!step_ok) {
        std::ostringstream os;
        os << "report step " << step << " failed to converge after "
           << deck.max_chop << " timestep halvings (down to dt = "
           << (double)dt_rep / (1 << (chop0 + deck.max_chop))
           << " days).\n"
           << "  Try a smaller DTREPORT, a larger NEWTONMAX, tighter DPMAX/"
              "DSMAX step limits, or check the well controls give a sensible "
              "voidage balance.";
        throw std::runtime_error(os.str());
      }
      if (nnewt_lost > nnewt_lost_at_entry) ++steps_chopped;
      res.dt = dt_rep;              // restore for the next report step
      t_days = report_days[step - 1];
      {
        // Restart and per-step output, timed so the wall clock can be split
        // into simulation and I/O the way the reference reports it.
        const auto tio = std::chrono::steady_clock::now();
        writer.snapshot(res, step, t_days);
        t_io += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - tio).count();
      }
      // Well rates, taken here because this is the only place the schedule and
      // the device state are both in hand.  finalize() cannot reconstruct them
      // for a field deck: it locates a well by index arithmetic on the deck's
      // NX/NY/NZ, and a field deck's are placeholders (Norne's say nz = 1 for a
      // 22-layer grid), so every well falls outside the bounds test and the
      // whole well section of the report vanishes without a word.
      if (field) {
        const int sched_step = schedule_step_at(*field, (double)t_days);
        std::vector<WellRateRow> wr =
            field_well_rates(deck, *field, res, sched_step, &pvt_host);
        reduce_well_rates(wr, nranks);   // collective: every rank
        writer.record_well_rates(t_days, wr);
        if (exact_well_cumulative)
          writer.record_well_cumulative(sub_qo, sub_qwp);
        if (substep_balance && rank == 0)
          std::printf("  [substep-balance] oil produced %.6g STB; water "
                      "injected %.6g - produced %.6g = %.6g STB; gas "
                      "injected %.6g - produced %.6g Mscf\n",
                      sub_qo, sub_qwi, sub_qwp, sub_qwi - sub_qwp,
                      sub_qgi * 0.001, sub_qgp);
        if (rank == 0 && residual_balance_active)
          std::printf("  [residual-balance] cumulative accepted defect: "
                      "oil %.6g STB, water %.6g STB, gas %.6g Mscf\n",
                      residual_balance[0], residual_balance[1],
                      residual_balance[2] * 0.001);

        // Oil balance, in stock-tank volumes on both sides.
        //
        // The stock-tank oil in place carries the shrinkage that solution gas
        // causes below the bubble point, so its fall between report steps IS
        // the oil that left, and the reported rates have to account for it.
        // Comparing SUM(PORV * So) instead -- a reservoir volume -- cannot
        // tell a rate that is too low from oil that merely shrank.
        {
          // Accumulated every step, because it is only a sum over rows
          // computed anyway.  The oil in place is NOT: it copies the
          // state off the device and evaluates the PVT for every cell,
          // so it runs only on the steps that report it.
          static double cum_rate = 0, t_prev = 0, n_first = -1;
          static double cum_wp = 0, cum_wi = 0, w_first = -1;
          // The previous interval's rates, for the trapezoid below.
          static double qo_prev = 0, qwp_prev = 0, qwi_prev = 0;
          double qo = 0, qwp = 0, qwi = 0;
          for (const WellRateRow& q : wr) {
            if (q.type < 0) { qo += (double)q.qo; qwp += (double)q.qw; }
            else if (q.type > 0) qwi += (double)q.qw;
          }
          // Trapezoid, not the right-hand rectangle this used to be.
          //
          // q(t_n) * (t_n - t_{n-1}) evaluates the rate at the END of every
          // interval and applies it across the whole of it.  On a profile that
          // is mostly declining -- which a field's is -- that is a systematic
          // UNDER-estimate, and it biases the very percentages these lines
          // exist to report.  benchmark/compare.py integrates both simulators
          // with np.trapezoid, so the in-run balance was not even measuring the
          // same quantity as the report it is supposed to corroborate.
          if (t_prev > 0) {
            const double dt = (double)t_days - t_prev;
            cum_rate += 0.5 * (qo + qo_prev) * dt;
            cum_wp += 0.5 * (qwp + qwp_prev) * dt;
            cum_wi += 0.5 * (qwi + qwi_prev) * dt;
          }
          t_prev = (double)t_days;
          qo_prev = qo; qwp_prev = qwp; qwi_prev = qwi;

          // EVERY report step, not every fortieth.
          //
          // The percentage below has read 77% since the run began, and a
          // number sampled six times over 247 steps cannot say whether the
          // oil goes missing steadily or in a few identifiable places.  The
          // per-step defect can, and this costs one device copy and a walk
          // over 44k cells against a step that takes seconds.
          if (true) {
            double w_now = 0, clamp_now = 0;
            const double n_now =
                field_oil_in_place(*field, res, &pvt_host, &w_now, &clamp_now);
            if (n_first < 0) { n_first = n_now; w_first = w_now; }
            const double from_field = n_first - n_now;
            // The same two quantities over THIS step alone.
            static double n_prev_step = -1, rate_prev_step = 0;
            const double d_field = (n_prev_step < 0) ? 0.0 : n_prev_step - n_now;
            const double d_rate = cum_rate - rate_prev_step;
            n_prev_step = n_now; rate_prev_step = cum_rate;
            if (rank == 0 && from_field > 0)
              std::printf("  [oil] in place %.4g STB, field says %.4g left, "
                          "rates say %.4g (%.0f%%)\n",
                          n_now, from_field, cum_rate,
                          100.0 * cum_rate / from_field);
            // Where the missing oil is, per step and in total.  `clamp` is the
            // Newton limiter's own contribution, measured in the same STB, so
            // the two are directly comparable: if it tracks the defect, the
            // limiter is the sink; if it is near zero, the sink is elsewhere
            // and the per-step column says which steps to look at.
            if (rank == 0)
              std::printf("  [oil-defect] step %d t=%.4g d: field -%.5g STB, "
                          "rates %.5g, defect %.5g (%.1f%% of step); "
                          "cumulative defect %.5g, clamp %.5g\n",
                          step, (double)t_days, d_field, d_rate,
                          d_field - d_rate,
                          d_field != 0 ? 100.0 * (d_field - d_rate) / d_field
                                       : 0.0,
                          from_field - cum_rate, clamp_now);
            // Water: the field's own change against injected minus
            // produced.  Both are stock-tank, both come from this run,
            // and water is conserved -- so this separates an injection
            // that is too large from a production that is merely
            // reported too large, which `gained` alone cannot.
            const double dW = w_now - w_first;
            const double net = cum_wi - cum_wp;
            if (rank == 0 && w_first > 0)
              std::printf("  [wat] in place %.4g STB, field gained %.4g, "
                          "rates say injected %.4g - produced %.4g = %.4g "
                          "(%.0f%%)\n",
                          w_now, dW, cum_wi, cum_wp, net,
                          dW != 0 ? 100.0 * net / dW : 0.0);

            // Crossflow, on the same schedule as the balances above.
            //
            // Every rate printed here is a NET sum over completions, and a
            // producer flowing both ways at once satisfies any balance while
            // reporting a rate that describes no well.  On Norne that is not
            // hypothetical: B-4H and D-4H report large negative oil against
            // Flow's positive while their gas is large and positive, which is
            // one well producing at one end and injecting at the other.
            //
            // Naming the worst offender is enough to tell that apart from a
            // rate target that is simply too small -- and the two want
            // opposite fixes, so guessing between them is expensive.
            if (rank == 0) {
              double xtot_o = 0;
              const WellRateRow* worst = nullptr;
              int n_xflow = 0;
              for (const WellRateRow& q : wr) {
                if (q.type >= 0 || q.nperf_reverse <= 0) continue;
                ++n_xflow;
                xtot_o += (double)q.xflow_o;
                if (!worst || q.xflow_o > worst->xflow_o) worst = &q;
              }
              if (worst && xtot_o > 0)
                std::printf("  [xflow] %d producer(s) flowing both ways: "
                            "%.4g STB/day of oil re-injected, worst %s "
                            "(%.4g STB/day back in through %d of %d "
                            "completions, net %.4g)\n",
                            n_xflow, xtot_o, worst->name.c_str(),
                            (double)worst->xflow_o, worst->nperf_reverse,
                            worst->nperf, (double)worst->qo);
            }
          }
        }
      }
      step_times.push_back(t_days);
      newton_iters.push_back(nnewt);
      linear_iters.push_back(nlin);

      if (rank == 0) {
        double el = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t0).count();
        report_progress(step, nsteps, t_days, r0, rlast, deck.newton_tol, nnewt,
                        nlin, el, step_ok, conv_how);
        if (step_ok && conv_mb > 0)
          std::printf("         mb %.2e (tol %.0e)   signed %.2e (tol %.0e)"
                      "   cnv %.2e (tol %.0e)\n",
                      (double)conv_mb, (double)deck.mb_tol,
                      (double)conv_signed_mb, (double)deck.mb_tol,
                      (double)conv_cnv, (double)deck.cnv_tol);
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();
    if (rank == 0) {
      std::printf("  %s\n", std::string(96, '-').c_str());
      if (nnewt_lost > 0 || nlin_lost > 0)
        std::printf("  discarded: %lld Newton and %lld linear iteration(s) on "
                    "%d of %d report step(s) that were retried; this work is in "
                    "the wall clock below but not in the per-step counts\n",
                    nnewt_lost, nlin_lost, steps_chopped, nsteps);
      std::printf("  simulation finished in %.2f s (%.2f s per report step)\n",
                  wall, nsteps ? wall / nsteps : wall);
      std::printf("  nonlinear work: %lld accepted substeps; %lld line-search "
                  "trial assemblies; %lld accepted-trial assemblies reused; "
                  "%lld updates had no descending trial; well-control refresh %.2f s\n",
                  nsub_accepted, nline_trials, nline_assembly_handoffs,
                  nline_no_descent, t_well_control);
    }
    // Rates that do not depend on how long the steps were: Newton per accepted
    // substep and per simulated day, retried work included and excluded.  A
    // run that takes more, shorter steps can have fewer Newton per step and
    // still do more work per day, so the per-day figure is the comparable one.
    long long nnewt_acc = 0, nlin_acc = 0;
    for (int v : newton_iters) nnewt_acc += v;
    for (int v : linear_iters) nlin_acc += v;
    const double sim_days = (double)t_days;
    if (rank == 0) {
      std::printf("  I/O      : restart and step output %.2f s inside the wall "
                  "(%.1f%%)\n", t_io, 100.0 * t_io / (wall > 0 ? wall : 1));
      std::printf("  Rates    : %.3f Newton per accepted substep; %.4f Newton "
                  "per simulated day (%.4f with retried work); %.1f days per "
                  "substep on average\n",
                  nsub_accepted ? (double)nnewt_acc / nsub_accepted : 0.0,
                  sim_days > 0 ? (double)nnewt_acc / sim_days : 0.0,
                  sim_days > 0 ? (double)(nnewt_acc + nnewt_lost) / sim_days
                               : 0.0,
                  nsub_accepted ? sim_days / (double)nsub_accepted : 0.0);
      // The accepted substeps, for DTSEQFILE: impose this run's steps on
      // another (a different rank count, backend or operator arm).
      if (!accepted_dt.empty()) {
        std::ofstream f(deck.out_dir + "/dt_sequence.txt");
        f << "# accepted substeps: t_start_days dt_days\n";
        f.precision(17);
        for (const auto& s : accepted_dt) f << s.first << " " << s.second << "\n";
      }
    }

    // The preconditioner's own accounting: how much of the linear solve was
    // HYPRE's FP32 BoomerAMG against the block-Jacobi second stage.  That is
    // the split the mixed-precision claim rests on, and it cannot be recovered
    // from the outside.
    const CprTimings& ct = solver.timings();

    // Multi-rank runs have been slower than single-rank ones, and wall clock
    // alone cannot apportion that between the halo exchange, the Krylov
    // reductions and BoomerAMG's own communication.  Report the halo directly,
    // and take the max across ranks rather than rank 0's view: the slowest rank
    // is what the others wait for.
    if (nranks > 1) {
      const HaloStats& hs = halo_stats();
      // Which halo did the work decides which flag describes it: the field
      // path never touches the structured one, so asking halo_cuda_aware()
      // there would label the transport by an implementation that did not run.
      const bool ca = field ? halo_conn_cuda_aware() : halo_cuda_aware();
      double mine[3] = {hs.seconds, hs.stage_seconds, hs.bytes};
      double worst[3] = {0, 0, 0};
      MPI_Reduce(mine, worst, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      long calls = hs.calls, calls_max = 0;
      MPI_Reduce(&calls, &calls_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        // "within operator+halo": the exchange runs inside op_apply, so this
        // time is part of that figure rather than additional to it.  Without
        // saying so the two lines read as double counting.
        std::printf("  Halo     : %ld exchanges, %.2f s (%.1f%% of wall, "
                    "within operator+halo below), %.2f GB moved%s\n",
                    calls_max, worst[0], 100.0 * worst[0] / (wall > 0 ? wall : 1),
                    worst[2] / 1e9,
                    ca ? "" : ", host-staged");
        if (!ca && worst[1] > 0)
          std::printf("             of which %.2f s was cudaMemcpy staging; "
                      "a CUDA-aware MPI removes it\n", worst[1]);
      }
    }

    if (rank == 0 && cfg.amg_reuse > 0) {
      const long asked = ct.n_amg_setup + ct.n_amg_reuse;
      // Cost per surviving setup, which is the number to watch when deciding
      // whether reuse is worth raising: it is what each skipped rebuild saved.
      const double per = ct.n_amg_setup ? ct.amg_setup / (double)ct.n_amg_setup
                                        : 0.0;
      std::printf("  AMG setup: %ld of %ld rebuilt (%ld reused, %ld cut short "
                  "by the growth guard), %.1f ms each, %.2f s total\n",
                  ct.n_amg_setup, asked, ct.n_amg_reuse, ct.n_amg_forced,
                  1e3 * per, ct.amg_setup);
    }
    // Split the linear solve into its three parts.  These have always been
    // measured and never printed, which left "restLinr" as one opaque number
    // covering the V-cycles, the operator applications and the second CPR
    // stage.  With identical hierarchies and identical iteration counts across
    // rank counts, this is the only remaining place the difference can be.
    if (rank == 0) {
      const double per_amg = ct.n_amg_solve
                                 ? 1e3 * ct.amg_solve / (double)ct.n_amg_solve
                                 : 0.0;
      // The operator figure includes its halo exchange, which is also counted
      // in the Halo line above; the two overlap on purpose and must not be
      // added together.  Whatever is left after these is kernel launches, the
      // Givens rotations on the host, and the update of x from the basis.
      std::printf("  Linear   : AMG V-cycles %.2f s (%ld applies, %.2f ms each)"
                  ", operator+halo %.2f s, stage-2 %.2f s, CGS%d %.2f s,"
                  " IJVector %.2f s\n",
                  ct.amg_solve, ct.n_amg_solve, per_amg, ct.spmv, ct.stage2,
                  cfg.ortho_passes, ct.ortho, ct.ijvec);
      std::printf("  Operator : %s storage, %ld block and %ld scalar applies;"
                  " CUDA graphs %s (%ld captured, %ld replays)\n",
                  cfg.block_spmv ? "block" : "scalar CSR", ct.spmv_block,
                  ct.spmv_scalar, cfg.cuda_graph ? "on" : "off",
                  ct.graph_captures, ct.graph_replays);
      std::printf("  Arith    : operator %s%s\n",
                  cfg.operator_mode == 2   ? "FP32-stored, FP64 master for residuals"
                  : cfg.operator_mode == 1 ? "df64"
                                           : "native FP64",
                  cfg.operator_mode == 2
                      ? (" (" + std::to_string(ct.spmv_fp32) +
                         " FP32 applies)").c_str()
                      : "");
      if (ct.graph_captures > 0) {
        const long reuses = ct.graph_replays - ct.graph_captures;
        std::printf("  Graphs   : %.2f reuses per capture (%ld launches "
                    "beyond the one after each capture); capture + "
                    "instantiate %.3f s in all, %.1f us each%s\n",
                    (double)reuses / (double)ct.graph_captures, reuses,
                    ct.graph_capture_seconds,
                    1e6 * ct.graph_capture_seconds / (double)ct.graph_captures,
                    ct.graph_guard_fired
                        ? (", guard fired after " +
                           std::to_string(ct.graph_guard_at) + " captures")
                              .c_str()
                        : "");
      }
      std::printf("  FGMRES   : true residual checked at %ld recursive "
                  "convergence(s); largest true/recursive ratio %.3g; %ld "
                  "solve(s) continued because the true residual had not met "
                  "the tolerance\n",
                  ct.fgmres_gap_checks, ct.fgmres_gap_max,
                  ct.fgmres_gap_restarts);
    }
    writer.set_run_stats(t_io, (long long)nsub_accepted, nnewt_acc,
                         (long long)nnewt_lost, nlin_acc, sim_days, wall,
                         cfg.operator_mode, deck.cpr_decouple);
    writer.finalize(deck, res, step_times, newton_iters, linear_iters, wall,
                    t_assemble, t_linear, ct);

    cudaFree(du);
    cudaFree(u_save);
    cudaFree(u_sub);
    cudaFree(u_newton);
    if (gp_save) cudaFree(gp_save);
    if (gp_sub) cudaFree(gp_sub);
    if (gp_newton) cudaFree(gp_newton);
    if (cd_save) cudaFree(cd_save);
    if (cd_sub) cudaFree(cd_sub);
    if (cd_newton) cudaFree(cd_newton);
    if (rs_save) cudaFree(rs_save);
    if (so_save) cudaFree(so_save);
    if (sg_hy_save) cudaFree(sg_hy_save);
    if (sw_ow_hy_save) cudaFree(sw_ow_hy_save);
    if (sequential) seq_free(seq);
    free_reservoir(res);

    // Auto-render plots + insights (rank 0 only; skipped in the HM inner loop).
    // NVRS_AUTOPLOT=0 skips the plots (minutes on a field deck) without editing
    // the deck; timing runs such as scripts/perf_ab.sh use it.
    const char* ap_env = std::getenv("NVRS_AUTOPLOT");
    const bool autoplot = ap_env ? std::atoi(ap_env) != 0 : deck.autoplot != 0;
    if (rank == 0 && autoplot && (deck.make_vtk || deck.make_csv)) {
      std::filesystem::path script = find_plot_script(argv[0]);
      if (script.empty()) {
        std::cout << "(plotting skipped — plot_results.py not found; set "
                     "NVRS_PLOT_SCRIPT or run it manually on "
                  << deck.out_dir << ")\n";
      } else {
        std::string cmd = shell_quote(find_python()) + " " +
                          shell_quote(script.string()) + " " +
                          shell_quote(deck.out_dir);
        std::cout << "rendering plots: " << cmd << "\n";
        int prc = std::system(cmd.c_str());
        if (prc != 0)
          std::cout << "(plotting failed — run:\n  " << cmd << "\nmanually)\n";
      }
    }
  } catch (const std::exception& e) {
    // Abort the whole job rather than unwinding this rank alone.  The other
    // ranks are almost certainly inside a collective (an MPI_Allreduce in the
    // Krylov inner products, or HYPRE's own communication); if this rank exits
    // quietly they block forever and the failure looks like a hang instead of
    // an error.  MPI_Abort tears the job down immediately with the message
    // still on screen.
    std::cerr << "[rank " << rank << "] error: " << e.what() << std::endl;
    std::cerr.flush();
    MPI_Abort(MPI_COMM_WORLD, 2);
    rc = 2;
  }

  halo_finalize();
  HYPRE_Finalize();
  MPI_Finalize();
  return rc;
}
