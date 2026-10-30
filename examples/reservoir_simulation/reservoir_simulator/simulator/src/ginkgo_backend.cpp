// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// ginkgo_backend.cpp -- Ginkgo as the stage-1 pressure solver inside CPR, in
// half precision.
//
// Like amgx_backend.cu this replaces exactly one thing, the multigrid on the
// pressure block.  The outer FP64 FGMRES, the operator, the quasi-IMPES
// weights and the bs x bs stage 2 are untouched.  What is new is the
// precision of that multigrid: Ginkgo (1.9 and later) instantiates its
// matrices, smoothers and multigrid levels for IEEE half precision, so the
// whole V-cycle -- hierarchy, Jacobi smoothing, coarse solve and the vectors
// it cycles -- runs in FP16.  GINKGOPREC 32 builds the identical hierarchy in
// FP32, so the effect of the precision can be measured with nothing else
// changed.
//
// The pressure matrix is Jacobi-scaled and the right-hand side normalised
// before narrowing (see include/ginkgo_kernels.hpp); both precisions go
// through the same scaling so the FP16/FP32 comparison is like for like.
//
// Hierarchy: Ginkgo's parallel graph matching (PGM) aggregation, Jacobi-
// relaxed Richardson smoothing, one V-cycle per application from a zero
// initial guess -- the same "improve the vector once" contract the AmgX and
// HYPRE backends honour.  Single GPU: the backend is wired for one rank, and
// main.cu refuses a multi-rank Ginkgo run before any work starts.
//
// Tunable without rebuilding, as the AmgX backend is:
//   NVRS_GINKGO_SWEEPS       pre/post smoothing sweeps          (default 1)
//   NVRS_GINKGO_RELAX        Jacobi relaxation factor           (default 0.9)
//   NVRS_GINKGO_COARSE_ITERS Jacobi sweeps on the coarsest level (default 10)
//   NVRS_GINKGO_MAXLEVELS    maximum hierarchy depth            (default 24)
//   NVRS_GINKGO_MINCOARSE    stop coarsening below this many rows (default 32)
//   NVRS_GINKGO_CYCLE        V, W or F                          (default V)
//   NVRS_GINKGO_NVTX         1 = Ginkgo's own NVTX ranges in an Nsight trace
//
// STAGE2 GINKGO swaps the stage-2 smoother for Ginkgo's adaptive-precision
// block-Jacobi on the same bs x bs cell blocks of the coupled FP64 operator:
// each block inverse is stored in FP64, FP32 or FP16 according to its
// conditioning.  It is independent of LINSOLVER and tuned by
//   NVRS_GINKGO_S2_ACCURACY  relative accuracy that drives the choice (0.1)
//   NVRS_GINKGO_S2_PREC      adaptive (default), or 64 / 32 / 16 for all blocks
//
// This file is compiled by the host C++ compiler (it includes the Ginkgo
// headers); the device kernels it needs are in ginkgo_kernels.cu.  Without
// NVRS_HAVE_GINKGO it compiles to stubs, and main.cu falls back to AmgX.
#include "linear_solver.hpp"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef NVRS_HAVE_GINKGO

#include "ginkgo_kernels.hpp"

#include <ginkgo/ginkgo.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <memory>

namespace {

inline double now_synced() {
  cudaDeviceSynchronize();
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : fallback;
}

double env_dbl(const char* name, double fallback) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::atof(v) : fallback;
}

char env_cycle() {
  const char* v = std::getenv("NVRS_GINKGO_CYCLE");
  if (!v || !*v) return 'V';
  const char c = (char)std::toupper((unsigned char)*v);
  return (c == 'W' || c == 'F') ? c : 'V';
}

struct Knobs {
  int sweeps, coarse_iters, max_levels, min_coarse;
  double relax;
  char cycle;
};

Knobs read_knobs() {
  Knobs k;
  k.sweeps = std::max(1, env_int("NVRS_GINKGO_SWEEPS", 1));
  k.relax = env_dbl("NVRS_GINKGO_RELAX", 0.9);
  k.coarse_iters = std::max(1, env_int("NVRS_GINKGO_COARSE_ITERS", 10));
  k.max_levels = std::max(2, env_int("NVRS_GINKGO_MAXLEVELS", 24));
  k.min_coarse = std::max(2, env_int("NVRS_GINKGO_MINCOARSE", 32));
  k.cycle = env_cycle();
  return k;
}

// One multigrid factory, for either value type.  Everything below is the
// value type V: the level generator, the smoother and the coarse solver.
template <typename V>
std::shared_ptr<gko::LinOpFactory> make_factory(
    std::shared_ptr<const gko::Executor> exec, const Knobs& k) {
  using Ir = gko::solver::Ir<V>;
  using Jacobi = gko::preconditioner::Jacobi<V, std::int32_t>;
  using Pgm = gko::multigrid::Pgm<V, std::int32_t>;
  const V relax = static_cast<V>(static_cast<float>(k.relax));

  auto smoother = gko::share(
      Ir::build()
          .with_solver(Jacobi::build().with_max_block_size(1u))
          .with_relaxation_factor(relax)
          .with_criteria(gko::stop::Iteration::build().with_max_iters(
              static_cast<gko::size_type>(k.sweeps)))
          .on(exec));
  auto coarsest = gko::share(
      Ir::build()
          .with_solver(Jacobi::build().with_max_block_size(1u))
          .with_relaxation_factor(relax)
          .with_criteria(gko::stop::Iteration::build().with_max_iters(
              static_cast<gko::size_type>(k.coarse_iters)))
          .on(exec));
  auto level = gko::share(Pgm::build().with_deterministic(true).on(exec));

  auto cycle = k.cycle == 'W'   ? gko::solver::multigrid::cycle::w
               : k.cycle == 'F' ? gko::solver::multigrid::cycle::f
                                : gko::solver::multigrid::cycle::v;
  return gko::share(
      gko::solver::Multigrid::build()
          .with_max_levels(static_cast<gko::size_type>(k.max_levels))
          .with_min_coarse_rows(static_cast<gko::size_type>(k.min_coarse))
          .with_pre_smoother(smoother)
          .with_post_uses_pre(true)
          .with_mg_level(level)
          .with_coarsest_solver(coarsest)
          .with_cycle(cycle)
          .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
          .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
          .on(exec));
}

// The pressure matrix as a Ginkgo CSR of value type V.  Row pointers and
// column indices are views of the simulator's own device arrays -- the
// sparsity never changes -- and only the values are Ginkgo's, rewritten with
// the scaled coefficients at every rebuild.
template <typename V>
std::shared_ptr<gko::LinOp> make_matrix(
    std::shared_ptr<const gko::Executor> exec,
    const nvrs::DeviceLinearSystem& sys, void** values) {
  using Csr = gko::matrix::Csr<V, std::int32_t>;
  auto rows = gko::array<std::int32_t>::view(
      exec, static_cast<gko::size_type>(sys.cells) + 1, sys.prowptr);
  auto cols = gko::array<std::int32_t>::view(
      exec, static_cast<gko::size_type>(sys.pnnz), sys.pcolind);
  gko::array<V> vals(exec, static_cast<gko::size_type>(sys.pnnz));
  auto A = gko::share(Csr::create(
      exec,
      gko::dim<2>(static_cast<gko::size_type>(sys.cells),
                  static_cast<gko::size_type>(sys.cells)),
      std::move(vals), std::move(cols), std::move(rows),
      std::make_shared<typename Csr::classical>()));
  *values = static_cast<void*>(A->get_values());
  return A;
}

template <typename V>
std::shared_ptr<gko::LinOp> make_vector(
    std::shared_ptr<const gko::Executor> exec, int cells, void** values) {
  auto v = gko::share(gko::matrix::Dense<V>::create(
      exec, gko::dim<2>(static_cast<gko::size_type>(cells), 1)));
  *values = static_cast<void*>(v->get_values());
  return v;
}

[[noreturn]] void ginkgo_fail(const char* where, const std::exception& e) {
  std::fprintf(stderr, "Ginkgo error in %s: %s\n", where, e.what());
  std::fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 3);
  std::abort();
}

}  // namespace

#if GINKGO_ENABLE_HALF
static_assert(sizeof(gko::half) == 2,
              "gko::half must be a 16-bit IEEE binary16 value");
#endif

// ===========================================================================
// State.  At global scope, matching the forward declaration in
// linear_solver.hpp, as amgx_impl and hypre_amg_impl are.
// ===========================================================================
struct ginkgo_impl {
  bool built = false;
  bool announced = false;
  int prec = 16;
  int cells = 0, pnnz = 0;
  long long row0 = 0;
  int device = 0;
  Knobs knobs{};
  std::shared_ptr<gko::Executor> exec;
  std::shared_ptr<gko::LinOpFactory> factory;
  std::shared_ptr<gko::LinOp> A, b, x;
  std::unique_ptr<gko::LinOp> solver;
  void* a_vals = nullptr;   // A's values: std::uint16_t* (FP16) or float*
  void* b_vals = nullptr;   // b's values, same type
  void* x_vals = nullptr;   // x's values, same type
  float* d = nullptr;       // Jacobi scale, cells
  float* work = nullptr;    // scaled right-hand side before narrowing, cells
  // --- STAGE2 GINKGO -------------------------------------------------------
  int s2_n = -1, s2_bs = -1, s2_nnz = -1;
  bool s2_announced = false;
  std::shared_ptr<gko::LinOpFactory> s2_factory;
  std::shared_ptr<gko::LinOp> s2_A;          // view of the coupled FP64 CSR
  std::unique_ptr<gko::LinOp> s2_prec;       // the generated block-Jacobi
  nvrs::Real* s2_y = nullptr;                // M^-1 (r - A z), n entries
};

namespace {

// One executor for everything Ginkgo does in this process, created on first
// use by whichever of the stage-1 backend or stage 2 gets there first.
void ensure_exec(ginkgo_impl& G) {
  if (G.exec) return;
  NVRS_CHECK_CUDA(cudaGetDevice(&G.device));
  G.exec = gko::CudaExecutor::create(G.device, gko::ReferenceExecutor::create());
  // Under Nsight, name what Ginkgo is doing (generate, apply, each level)
  // rather than leaving a timeline of anonymous gko:: kernels.  Off by
  // default: the hook costs a range push per operation.
  if (env_int("NVRS_GINKGO_NVTX", 0) != 0)
    G.exec->add_logger(gko::log::ProfilerHook::create_nvtx());
}

}  // namespace

namespace nvrs {

bool ginkgo_available() { return true; }

const char* ginkgo_version() {
  static const std::string v =
      std::to_string(GKO_VERSION_MAJOR) + "." +
      std::to_string(GKO_VERSION_MINOR) + "." +
      std::to_string(GKO_VERSION_PATCH) +
#if GINKGO_ENABLE_HALF
      " (half precision enabled)";
#else
      " (built WITHOUT half precision)";
#endif
  return v.c_str();
}

// ---------------------------------------------------------------------------
// Build or refresh the hierarchy.
// ---------------------------------------------------------------------------
void CprSolver::setup_ginkgo(const DeviceLinearSystem& sys) {
  if (!ginkgo_) ginkgo_ = new ginkgo_impl();
  auto& G = *ginkgo_;

  // Same reuse rule as the other two backends.
  if (G.built && reuse_left_ > 0 && amg_pval_ == (const void*)sys.pval &&
      amg_cells_ == sys.cells && amg_pnnz_ == sys.pnnz) {
    --reuse_left_;
    ++timings_.n_amg_reuse;
    amg_fresh_ = false;
    return;
  }

  try {
    if (!G.built) {
      G.prec = cfg_.ginkgo_precision == 32 ? 32 : 16;
#if !GINKGO_ENABLE_HALF
      if (G.prec == 16) {
        std::fprintf(stderr,
                     "ERROR: GINKGOPREC 16 needs a Ginkgo built with half "
                     "precision (GINKGO_ENABLE_HALF=ON), and this one was "
                     "not.  Rebuild it with benchmark/install_ginkgo.sh, or "
                     "set GINKGOPREC 32.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
#endif
      if (grid_.nranks > 1) {
        std::fprintf(stderr, "ERROR: the Ginkgo backend is single-GPU; use "
                             "LINSOLVER AMGX on several ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
      G.cells = sys.cells;
      G.pnnz = sys.pnnz;
      G.row0 = sys.row0;
      G.knobs = read_knobs();
      ensure_exec(G);
      NVRS_CHECK_CUDA(cudaMalloc((void**)&G.d, sizeof(float) * (size_t)G.cells));
      NVRS_CHECK_CUDA(cudaMalloc((void**)&G.work, sizeof(float) * (size_t)G.cells));
      if (G.prec == 16) {
#if GINKGO_ENABLE_HALF
        G.A = make_matrix<gko::half>(G.exec, sys, &G.a_vals);
        G.b = make_vector<gko::half>(G.exec, G.cells, &G.b_vals);
        G.x = make_vector<gko::half>(G.exec, G.cells, &G.x_vals);
        G.factory = make_factory<gko::half>(G.exec, G.knobs);
#endif
      } else {
        G.A = make_matrix<float>(G.exec, sys, &G.a_vals);
        G.b = make_vector<float>(G.exec, G.cells, &G.b_vals);
        G.x = make_vector<float>(G.exec, G.cells, &G.x_vals);
        G.factory = make_factory<float>(G.exec, G.knobs);
      }
      G.built = true;
      if (grid_.rank == 0)
        std::printf("  Ginkgo   : %s, FP%d PGM aggregation AMG, %c-cycle, "
                    "%d sweep%s (relax %.2f), %d rows\n",
                    ginkgo_version(), G.prec, G.knobs.cycle, G.knobs.sweeps,
                    G.knobs.sweeps == 1 ? "" : "s", G.knobs.relax, G.cells);
    }

    const double t = now_synced();
    gk::jacobi_scale(G.cells, G.row0, sys.prowptr, sys.pcolind, sys.pval, G.d);
    if (G.prec == 16)
      gk::scale_matrix_f16(G.cells, G.row0, sys.prowptr, sys.pcolind, sys.pval,
                           G.d, static_cast<std::uint16_t*>(G.a_vals));
    else
      gk::scale_matrix_f32(G.cells, G.row0, sys.prowptr, sys.pcolind, sys.pval,
                           G.d, static_cast<float*>(G.a_vals));
    G.solver = G.factory->generate(G.A);
    G.exec->synchronize();
    timings_.amg_setup += now_synced() - t;
    ++timings_.n_amg_setup;

    if (!G.announced && grid_.rank == 0) {
      G.announced = true;
      if (auto* mg = dynamic_cast<gko::solver::Multigrid*>(G.solver.get())) {
        const auto& levels = mg->get_mg_level_list();
        gko::size_type coarse = static_cast<gko::size_type>(G.cells);
        if (!levels.empty())
          coarse = levels.back()->get_coarse_op()->get_size()[0];
        std::printf("  Ginkgo   : hierarchy of %zu level%s, coarsest %zu rows\n",
                    (size_t)levels.size() + 1, levels.empty() ? "" : "s",
                    (size_t)coarse);
      }
    }
  } catch (const std::exception& e) {
    ginkgo_fail("setup", e);
  }

  reuse_left_ = std::max(0, cfg_.amg_reuse);
  iters_fresh_ = -1;
  amg_fresh_ = true;
  amg_pval_ = (const void*)sys.pval;
  amg_cells_ = sys.cells;
  amg_pnnz_ = sys.pnnz;
}

// ---------------------------------------------------------------------------
// One V-cycle.  bp holds the restricted residual and xp receives the
// correction, both device arrays of `cells` PrecReal, exactly as for AmgX.
// The scaling and narrowing are timed as marshalling (ev_ijvec_), the cycle
// itself as the multigrid (ev_amg_), so the columns mean what they mean for
// the other backends.
// ---------------------------------------------------------------------------
void CprSolver::apply_ginkgo(const DeviceLinearSystem& sys, PrecReal* bp,
                             PrecReal* xp) {
  auto& G = *ginkgo_;
  (void)sys;
  try {
    ev_ijvec_.tic();
    const float s =
        G.prec == 16
            ? gk::scale_rhs_f16(G.cells, bp, G.d, G.work,
                                static_cast<std::uint16_t*>(G.b_vals))
            : gk::scale_rhs_f32(G.cells, bp, G.d,
                                static_cast<float*>(G.b_vals));
    ev_ijvec_.toc();

    ev_amg_.tic();
    G.solver->apply(G.b.get(), G.x.get());
    ev_amg_.toc();
    ++timings_.n_amg_solve;
    ++timings_.amg_iters;

    ev_ijvec_.tic();
    if (G.prec == 16)
      gk::unscale_f16(G.cells, static_cast<const std::uint16_t*>(G.x_vals),
                      G.d, s, xp);
    else
      gk::unscale_f32(G.cells, static_cast<const float*>(G.x_vals), G.d, s,
                      xp);
    ev_ijvec_.toc();
  } catch (const std::exception& e) {
    ginkgo_fail("apply", e);
  }
}

// ---------------------------------------------------------------------------
// STAGE2 GINKGO: Ginkgo's adaptive-precision block-Jacobi.
// ---------------------------------------------------------------------------
void CprSolver::setup_ginkgo_stage2(const DeviceLinearSystem& sys) {
  if (!ginkgo_) ginkgo_ = new ginkgo_impl();
  auto& G = *ginkgo_;
  using Jac = gko::preconditioner::Jacobi<double, std::int32_t>;
  try {
    ensure_exec(G);
    if (grid_.nranks > 1 || sys.n_ext != sys.n) {
      std::fprintf(stderr, "ERROR: STAGE2 GINKGO is single-GPU (the coupled "
                           "operator must have no ghost columns).\n");
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
    if (!G.s2_factory || G.s2_n != sys.n || G.s2_bs != sys.bs ||
        G.s2_nnz != sys.nnz) {
      const int cells = sys.n / sys.bs;
      gko::array<std::int32_t> bp(G.exec->get_master(),
                                  static_cast<gko::size_type>(cells) + 1);
      for (int c = 0; c <= cells; ++c) bp.get_data()[c] = c * sys.bs;

      gko::precision_reduction storage = gko::precision_reduction::autodetect();
      const char* pm = std::getenv("NVRS_GINKGO_S2_PREC");
      std::string mode = "adaptive";
      if (pm && *pm) {
        const int p = std::atoi(pm);
        if (p == 64) { storage = gko::precision_reduction(0, 0); mode = "FP64"; }
        if (p == 32) { storage = gko::precision_reduction(0, 1); mode = "FP32"; }
        if (p == 16) { storage = gko::precision_reduction(0, 2); mode = "FP16"; }
      }
      const double acc = env_dbl("NVRS_GINKGO_S2_ACCURACY", 0.1);

      G.s2_factory = gko::share(
          Jac::build()
              .with_max_block_size(static_cast<gko::uint32>(sys.bs))
              .with_block_pointers(bp)
              .with_storage_optimization(storage)
              .with_accuracy(acc)
              .on(G.exec));
      using Csr = gko::matrix::Csr<double, std::int32_t>;
      G.s2_A = gko::share(Csr::create(
          G.exec,
          gko::dim<2>(static_cast<gko::size_type>(sys.n),
                      static_cast<gko::size_type>(sys.n)),
          gko::array<double>::view(G.exec, static_cast<gko::size_type>(sys.nnz),
                                   sys.aval64),
          gko::array<std::int32_t>::view(
              G.exec, static_cast<gko::size_type>(sys.nnz), sys.colind),
          gko::array<std::int32_t>::view(
              G.exec, static_cast<gko::size_type>(sys.n) + 1, sys.rowptr),
          std::make_shared<Csr::classical>()));
      if (G.s2_y) cudaFree(G.s2_y);
      NVRS_CHECK_CUDA(
          cudaMalloc((void**)&G.s2_y, sizeof(Real) * (size_t)sys.n));
      G.s2_n = sys.n;
      G.s2_bs = sys.bs;
      G.s2_nnz = sys.nnz;
      if (grid_.rank == 0)
        std::printf("  Stage 2  : Ginkgo block-Jacobi, %d x %d blocks, %s "
                    "storage (accuracy %.3g)\n",
                    sys.bs, sys.bs, mode.c_str(), acc);
    }

    // The operator values are a view, so regenerating reads the current
    // Jacobian; the blocks change every Newton iteration, like ZLINE's factor.
    const double t = now_synced();
    G.s2_prec = G.s2_factory->generate(G.s2_A);
    G.exec->synchronize();
    timings_.stage2_setup += now_synced() - t;

    if (!G.s2_announced && grid_.rank == 0) {
      G.s2_announced = true;
      if (auto* jac = dynamic_cast<Jac*>(G.s2_prec.get())) {
        auto bw = jac->get_parameters().storage_optimization.block_wise;
        if (bw.get_size() == jac->get_num_blocks() && bw.get_size() > 0) {
          bw.set_executor(G.exec->get_master());
          size_t n64 = 0, n32 = 0, n16 = 0;
          for (gko::size_type i = 0; i < bw.get_size(); ++i) {
            const auto pr = bw.get_const_data()[i];
            const int red = (int)pr.get_preserving() + (int)pr.get_nonpreserving();
            (red == 0 ? n64 : red == 1 ? n32 : n16)++;
          }
          const double nb = (double)bw.get_size();
          std::printf("  Stage 2  : first build, blocks stored FP64 %.1f%%, "
                      "FP32 %.1f%%, FP16 %.1f%%\n",
                      100.0 * n64 / nb, 100.0 * n32 / nb, 100.0 * n16 / nb);
        }
      }
    }
  } catch (const std::exception& e) {
    ginkgo_fail("stage-2 setup", e);
  }
}

void CprSolver::apply_ginkgo_stage2(const DeviceLinearSystem& sys,
                                    const Real* r, Real* Az, Real* z) {
  auto& G = *ginkgo_;
  const int n = sys.n;
  try {
    gk::stage2_residual(n, r, Az);
    using Dense = gko::matrix::Dense<double>;
    auto b = Dense::create(
        G.exec, gko::dim<2>(static_cast<gko::size_type>(n), 1),
        gko::array<double>::view(G.exec, static_cast<gko::size_type>(n), Az),
        1);
    auto x = Dense::create(
        G.exec, gko::dim<2>(static_cast<gko::size_type>(n), 1),
        gko::array<double>::view(G.exec, static_cast<gko::size_type>(n),
                                 G.s2_y),
        1);
    G.s2_prec->apply(b.get(), x.get());
    gk::stage2_update(n, G.s2_y, Az, z);
  } catch (const std::exception& e) {
    ginkgo_fail("stage-2 apply", e);
  }
}

void CprSolver::destroy_ginkgo() {
  if (!ginkgo_) return;
  auto& G = *ginkgo_;
  G.s2_prec.reset();
  G.s2_factory.reset();
  G.s2_A.reset();
  if (G.s2_y) cudaFree(G.s2_y);
  // Ginkgo objects first: they free device memory through the executor, and
  // must do so while the CUDA context is still alive.
  G.solver.reset();
  G.factory.reset();
  G.A.reset();
  G.b.reset();
  G.x.reset();
  G.exec.reset();
  if (G.d) cudaFree(G.d);
  if (G.work) cudaFree(G.work);
  delete ginkgo_;
  ginkgo_ = nullptr;
}

}  // namespace nvrs

#else  // NVRS_HAVE_GINKGO

struct ginkgo_impl {};

namespace nvrs {

bool ginkgo_available() { return false; }
const char* ginkgo_version() { return "not built"; }

// main.cu falls back to AmgX before any of these can be reached; they abort
// rather than silently doing nothing if that guard is ever removed.
void CprSolver::setup_ginkgo(const DeviceLinearSystem&) {
  std::fprintf(stderr, "NVRS: built without Ginkgo; use LINSOLVER AMGX or "
                       "HYPRE, or run benchmark/install_ginkgo.sh.\n");
  MPI_Abort(MPI_COMM_WORLD, 3);
}
void CprSolver::apply_ginkgo(const DeviceLinearSystem&, PrecReal*, PrecReal*) {}
void CprSolver::destroy_ginkgo() {}
void CprSolver::setup_ginkgo_stage2(const DeviceLinearSystem&) {
  std::fprintf(stderr, "NVRS: built without Ginkgo; STAGE2 GINKGO is not "
                       "available.\n");
  MPI_Abort(MPI_COMM_WORLD, 3);
}
void CprSolver::apply_ginkgo_stage2(const DeviceLinearSystem&, const Real*,
                                    Real*, Real*) {}

}  // namespace nvrs

#endif  // NVRS_HAVE_GINKGO
