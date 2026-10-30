// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// amgx_backend.cu — AmgX as the stage-1 pressure solver inside CPR.
//
// This file replaces exactly one thing: the multigrid that solves the pressure
// block.  Everything that carries the mixed-precision argument is untouched and
// lives elsewhere — the outer Krylov is still FP64 FGMRES over a df64-emulated
// FP64 operator (cpr_solver.cu), the quasi-IMPES weights that form the pressure
// system are still FP64 (assemble_*.cu), and stage 2 is still the bs x bs
// block-Jacobi correction.  AmgX runs in dFFI mode, so the preconditioner is
// FP32 exactly as the FP32 build of HYPRE was.  Swapping the backend does not
// move the precision boundary; it moves which library sits inside it.
//
// Why offer it at all: BoomerAMG's *setup* is the one phase that never scaled
// here.  Measured on 200x200x25, a rebuild costs 42 ms on one GB200, 94 on two
// and 140 on four, while the local problem shrinks — and the ratio is the same
// on a grid nine times smaller, which makes it latency in the parallel
// coarsening rather than work.  AmgX's default is aggregation AMG, whose setup
// is much cheaper to build in parallel than classical PMIS coarsening, and its
// setup runs entirely on the device.
//
// The whole file compiles to nothing without NVRS_HAVE_AMGX, so a machine with
// no AmgX still builds and runs on HYPRE.
#include "linear_solver.hpp"

#ifdef NVRS_HAVE_AMGX

#include <amgx_c.h>
#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// AmgX returns a status code from every call.  Ignoring them turns a
// configuration mistake into a wrong answer rather than an error, so every call
// goes through here.
void amgx_check(AMGX_RC rc, const char* what, const char* file, int line) {
  if (rc == AMGX_RC_OK) return;
  char msg[4096] = {0};
  AMGX_get_error_string(rc, msg, (int)sizeof(msg));
  std::fprintf(stderr, "AmgX error at %s:%d in %s -> %s\n", file, line, what,
               msg);
  MPI_Abort(MPI_COMM_WORLD, 3);
}
#define NVRS_AMGX(x) amgx_check((x), #x, __FILE__, __LINE__)

// Wall clock with the device drained first, so a setup time is the setup and
// not the queue in front of it.  cpr_solver.cu has the same helper in its own
// anonymous namespace; both backends must measure the same way or the setup
// column is not a comparison.
inline double now_synced() {
  cudaDeviceSynchronize();
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// AmgX prints through a callback rather than to stdout.  Route it to rank 0 so
// a four-rank run does not print everything four times.
int g_amgx_rank = 0;
void amgx_print(const char* msg, int /*length*/) {
  if (g_amgx_rank == 0) std::fputs(msg, stdout);
}

// dFFI: device, Float matrix, Float vector, Int index -- the FP32 solver that
// keeps the mixed-precision arrangement identical to the FP32 HYPRE build.
// dDDI is the all-double mode, selected automatically if this binary was built
// without NVRS_FP32_PRECOND so that PrecReal is double.
#if defined(NVRS_FP32_PRECOND) && NVRS_FP32_PRECOND
constexpr AMGX_Mode kAmgxMode = AMGX_mode_dFFI;
constexpr const char* kAmgxModeName = "dFFI (FP32)";
#else
constexpr AMGX_Mode kAmgxMode = AMGX_mode_dDDI;
constexpr const char* kAmgxModeName = "dDDI (FP64)";
#endif

// Classical PMIS + extended interpolation, one V-cycle per application.  On a
// heterogeneous field this is the far better preconditioner: on SPE10 model 2
// it halves the Krylov count against aggregation and, over four report steps
// on one GB200, halves the run (117 s against 237 s).
//
// It is single-rank only, and that is a property of the library rather than a
// choice: AmgX's DISTRIBUTED classical setup exhausts device memory building
// its communication maps (markCoarseFinePoints -> CommsMPIHostBufferStream::
// do_setup -> cudaMallocAsync fails) within seconds of the first setup, at 2,
// 3 and 4 ranks alike, with either interpolator (D1 or D2), and on a problem
// each rank holds less of than the single-GPU run that succeeds.  Aggregation
// has no such trouble, so choose_amgx_config() picks by rank count.
const char* classical_amgx_config() {
  // Three knobs are overridable from the environment, because the right values
  // depend on the field and finding them is an experiment, not a derivation.
  // Each costs one run to test and needs no rebuild:
  //
  //   NVRS_AMGX_STRENGTH  strength threshold, default 0.5.  This decides which
  //       couplings the coarsening treats as strong.  On a sharply
  //       heterogeneous field -- Norne's permeability spans 0.65 to 3,996 mD
  //       -- a high threshold marks most couplings weak, the hierarchy
  //       coarsens too aggressively, and the V-cycle stops being an effective
  //       preconditioner.  0.25 is the usual value for such problems.
  //   NVRS_AMGX_SWEEPS    pre- and post-smoothing sweeps, default 1.  Two
  //       sweeps roughly double the cost of an apply and can more than halve
  //       the Krylov count on a hard system.
  //   NVRS_AMGX_CYCLE     V (default) or W.  A W-cycle spends more work on the
  //       coarse levels, which is where a nearly incompressible pressure
  //       problem is hardest.
  static std::string cfg;
  if (!cfg.empty()) return cfg.c_str();

  auto env_or = [](const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string(fallback);
  };
  const std::string strength = env_or("NVRS_AMGX_STRENGTH", "0.5");
  const std::string sweeps = env_or("NVRS_AMGX_SWEEPS", "1");
  const std::string cycle = env_or("NVRS_AMGX_CYCLE", "V");

  cfg =
      "config_version=2,"
      "determinism_flag=1,"
      "solver(main)=AMG,"
      "main:algorithm=CLASSICAL,"
      "main:selector=PMIS,"
      "main:interpolator=D2,"
      "main:strength_threshold=" + strength + ","
      "main:max_iters=1,"
      "main:cycle=" + cycle + ","
      "main:presweeps=" + sweeps + ","
      "main:postsweeps=" + sweeps + ","
      "main:smoother=BLOCK_JACOBI,"
      "main:relaxation_factor=0.8,"
      "main:coarse_solver=DENSE_LU_SOLVER,"
      "main:max_levels=24,"
      "main:min_coarse_rows=32,"
      "main:tolerance=0.0,"
      "main:norm=L2,"
      "main:monitor_residual=0,"
      "main:store_res_history=0,"
      "main:print_solve_stats=0,"
      "main:print_grid_stats=0";
  return cfg.c_str();
}

// Aggregation AMG applied as a single V-cycle, which is what a preconditioner
// is: max_iters=1 and tolerance=0 mean "improve the vector once", never "solve
// to a tolerance".  That mirrors HYPRE_BoomerAMGSetMaxIter(1) + SetTol(0) on
// the other backend, so a comparison between them is a comparison of two
// multigrids and not of two stopping rules.
const char* default_amgx_config() {
  return
      "config_version=2,"
      "determinism_flag=1,"
      "solver(main)=AMG,"
      "main:algorithm=AGGREGATION,"
      "main:selector=SIZE_8,"
      "main:max_iters=1,"
      "main:cycle=V,"
      "main:presweeps=1,"
      "main:postsweeps=1,"
      "main:smoother=BLOCK_JACOBI,"
      "main:relaxation_factor=0.8,"
      "main:coarse_solver=DENSE_LU_SOLVER,"
      "main:max_levels=24,"
      "main:min_coarse_rows=32,"
      "main:tolerance=0.0,"
      "main:norm=L2,"
      "main:monitor_residual=0,"
      "main:store_res_history=0,"
      "main:print_solve_stats=0,"
      "main:print_grid_stats=0";
}

// AmgX's host-buffer MPI communicator works with every MPI implementation.
// MPI_DIRECT is faster where device-pointer transport has been verified, but
// merely advertising CUDA support is insufficient: some OpenMPI builds accept
// the pointer and then fault in MPI_Isend during distributed matrix upload.
// Keep the portable transport as the default and allow a verified installation
// to opt in with NVRS_AMGX_COMM=MPI_DIRECT.
const char* amgx_communicator() {
  const char* c = std::getenv("NVRS_AMGX_COMM");
  return (c && *c) ? c : "MPI";
}

// Which built-in configuration to use, from the rank count.  Classical is the
// better preconditioner and the faster run, but only works on one rank (see
// classical_amgx_config); aggregation is what survives a decomposition.
// NVRS_AMGX_ALGO=CLASSICAL|AGGREGATION forces one either way -- CLASSICAL on
// several ranks is left available deliberately, so that a future AmgX can be
// retested with one environment variable and no rebuild.
const char* choose_amgx_config(int nranks, const char** name,
                               const char** why) {
  const char* a = std::getenv("NVRS_AMGX_ALGO");
  if (a && (*a == 'c' || *a == 'C')) {
    *name = "classical PMIS"; *why = "forced by NVRS_AMGX_ALGO";
    return classical_amgx_config();
  }
  if (a && (*a == 'a' || *a == 'A')) {
    *name = "aggregation"; *why = "forced by NVRS_AMGX_ALGO";
    return default_amgx_config();
  }
  if (nranks > 1) {
    *name = "aggregation";
    *why = "automatic: the distributed-capable hierarchy";
    return default_amgx_config();
  }
  *name = "classical PMIS";
  *why = "automatic: the stronger hierarchy, single rank";
  return classical_amgx_config();
}

}  // namespace

// ===========================================================================
// State.  Declared here rather than in the header so the AmgX types stay out
// of the public interface, the same arrangement hypre_amg_impl uses.
// ===========================================================================
struct amgx_impl {
  bool initialized = false;      // AMGX_initialize, once per process
  bool built = false;            // handles created and structure uploaded
  AMGX_config_handle cfg = nullptr;
  AMGX_resources_handle rsrc = nullptr;
  AMGX_matrix_handle A = nullptr;
  AMGX_vector_handle b = nullptr, x = nullptr;
  AMGX_solver_handle solver = nullptr;
  int cells = 0, pnnz = 0;
  long long row0 = 0;
  long long n_global = 0;
  MPI_Comm comm = MPI_COMM_NULL;
  int device = 0;
};

namespace nvrs {

// ---------------------------------------------------------------------------
// Build or refresh the pressure hierarchy.
// ---------------------------------------------------------------------------
void CprSolver::setup_amgx(const DeviceLinearSystem& sys) {
  if (!amgx_) amgx_ = new amgx_impl();
  auto& X = *amgx_;

  // Same reuse rule as the HYPRE path, for the same reason: a hierarchy is
  // expensive to build and stays usable for several Newton iterations.  The
  // growth guard in solve() abandons it as soon as it costs more Krylov
  // iterations than a fresh one would.
  if (X.built && reuse_left_ > 0 && amg_pval_ == (const void*)sys.pval &&
      amg_cells_ == sys.cells && amg_pnnz_ == sys.pnnz) {
    --reuse_left_;
    ++timings_.n_amg_reuse;
    amg_fresh_ = false;
    return;
  }

  if (!X.built) {
    X.comm = MPI_COMM_WORLD;
    X.cells = sys.cells;
    X.pnnz = sys.pnnz;
    X.row0 = sys.row0;
    g_amgx_rank = grid_.rank;
    NVRS_CHECK_CUDA(cudaGetDevice(&X.device));

    if (!X.initialized) {
      NVRS_AMGX(AMGX_initialize());
      NVRS_AMGX(AMGX_register_print_callback(&amgx_print));
      NVRS_AMGX(AMGX_install_signal_handler());
      X.initialized = true;
    }

    // AMGXCONFIG names an AmgX config file, so the whole solver can be
    // retuned without rebuilding.  Otherwise the built-in string above.
    const char* cfg_file = std::getenv("NVRS_AMGX_CONFIG");
    if (cfg_file && *cfg_file) {
      NVRS_AMGX(AMGX_config_create_from_file(&X.cfg, cfg_file));
      if (grid_.rank == 0)
        std::printf("  AmgX     : config from %s\n", cfg_file);
    } else {
      const char *name = "", *why = "";
      const char* cfg = choose_amgx_config(grid_.nranks, &name, &why);
      NVRS_AMGX(AMGX_config_create(&X.cfg, cfg));
      if (grid_.rank == 0)
        std::printf("  AmgX     : %s hierarchy on %d rank%s (%s)\n", name,
                    grid_.nranks, grid_.nranks == 1 ? "" : "s", why);
    }
    // AmgX would otherwise install its own MPI error handling; keep ours.
    NVRS_AMGX(AMGX_config_add_parameters(&X.cfg, "exception_handling=1"));
    if (grid_.nranks > 1) {
      std::string comm = std::string("communicator=") + amgx_communicator();
      NVRS_AMGX(AMGX_config_add_parameters(&X.cfg, comm.c_str()));
    }

    // One rank, one GPU -- the binding main.cu already made.
    NVRS_AMGX(AMGX_resources_create(&X.rsrc, X.cfg, &X.comm, 1, &X.device));
    NVRS_AMGX(AMGX_matrix_create(&X.A, X.rsrc, kAmgxMode));
    NVRS_AMGX(AMGX_vector_create(&X.b, X.rsrc, kAmgxMode));
    NVRS_AMGX(AMGX_vector_create(&X.x, X.rsrc, kAmgxMode));
    NVRS_AMGX(AMGX_solver_create(&X.solver, X.rsrc, kAmgxMode, X.cfg));

    // --- the distributed structure, uploaded once -------------------------
    // sys.pcolind already holds *global* pressure column ids, which is what
    // the global upload wants, and the rows each rank owns are the contiguous
    // range [row0, row0 + cells).  So the partition is described by gathering
    // one integer per rank.
    std::vector<int> counts(grid_.nranks), offsets(grid_.nranks + 1, 0);
    int mine = sys.cells;
    MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, X.comm);
    for (int r = 0; r < grid_.nranks; ++r)
      offsets[r + 1] = offsets[r] + counts[r];
    X.n_global = offsets[grid_.nranks];

    // partition_vector[global row] = owning rank.  AmgX takes this on the
    // host; it is built once and thrown away, unlike the matrix itself.
    std::vector<int> partition((size_t)X.n_global);
    for (int r = 0; r < grid_.nranks; ++r)
      for (int i = offsets[r]; i < offsets[r + 1]; ++i) partition[i] = r;

    // The structure has to come from the host for this call, so stage it once.
    // The *values* never take this path again: replace_coefficients below
    // reads sys.pval straight from the device on every rebuild.
    std::vector<int> h_rowptr(sys.cells + 1), h_colind(sys.pnnz);
    std::vector<PrecReal> h_val(sys.pnnz);
    NVRS_CHECK_CUDA(cudaMemcpy(h_rowptr.data(), sys.prowptr,
                               sizeof(int) * (sys.cells + 1),
                               cudaMemcpyDeviceToHost));
    NVRS_CHECK_CUDA(cudaMemcpy(h_colind.data(), sys.pcolind,
                               sizeof(int) * sys.pnnz, cudaMemcpyDeviceToHost));
    NVRS_CHECK_CUDA(cudaMemcpy(h_val.data(), sys.pval,
                               sizeof(PrecReal) * sys.pnnz,
                               cudaMemcpyDeviceToHost));

    // The _32 suffix is not optional.  AMGX_matrix_upload_all_global reads
    // the global column indices as int64 whatever the solver mode is, so
    // handing it our int32 array walked off the end of the buffer inside
    // loadDistributedMatrixPartitionVec<long> and segfaulted.  pcolind is
    // int32 because the pressure system is one row per cell and a grid that
    // needed more than two billion of them would not fit on the machine.
    NVRS_AMGX(AMGX_matrix_upload_all_global_32(
        X.A, (int)X.n_global, sys.cells, sys.pnnz, 1, 1, h_rowptr.data(),
        h_colind.data(), h_val.data(), nullptr, 1, 1, partition.data()));

    NVRS_AMGX(AMGX_vector_bind(X.b, X.A));
    NVRS_AMGX(AMGX_vector_bind(X.x, X.A));
    X.built = true;

    if (grid_.rank == 0)
      std::printf("  AmgX     : %s, aggregation AMG, 1 V-cycle per apply, "
                  "%lld global rows on %d rank%s\n",
                  kAmgxModeName, X.n_global, grid_.nranks,
                  grid_.nranks == 1 ? "" : "s");
  } else {
    // Structure is static; only the coefficients move, and they move on the
    // device.  This is the AmgX equivalent of skipping the IJ assembly.
    NVRS_AMGX(AMGX_matrix_replace_coefficients(X.A, sys.cells, sys.pnnz,
                                               sys.pval, nullptr));
  }

  {
    const double t = now_synced();
    NVRS_AMGX(AMGX_solver_setup(X.solver, X.A));
    timings_.amg_setup += now_synced() - t;
    ++timings_.n_amg_setup;
  }

  reuse_left_ = std::max(0, cfg_.amg_reuse);
  iters_fresh_ = -1;
  amg_fresh_ = true;
  amg_pval_ = (const void*)sys.pval;
  amg_cells_ = sys.cells;
  amg_pnnz_ = sys.pnnz;
}

// ---------------------------------------------------------------------------
// One V-cycle.  bp holds the restricted residual, xp receives the correction;
// both are device arrays of `cells` PrecReal, produced and consumed by the
// same k_extract_p / k_scatter_p kernels the HYPRE path uses.
// ---------------------------------------------------------------------------
void CprSolver::apply_amgx(const DeviceLinearSystem& sys, PrecReal* bp,
                           PrecReal* xp) {
  auto& X = *amgx_;
  NVRS_AMGX(AMGX_vector_upload(X.b, sys.cells, 1, bp));
  NVRS_AMGX(AMGX_vector_upload(X.x, sys.cells, 1, xp));

  ev_amg_.tic();
  NVRS_AMGX(AMGX_solver_solve_with_0_initial_guess(X.solver, X.b, X.x));
  ev_amg_.toc();
  ++timings_.n_amg_solve;

  NVRS_AMGX(AMGX_vector_download(X.x, xp));
}

void CprSolver::destroy_amgx() {
  if (!amgx_) return;
  auto& X = *amgx_;
  if (X.built) {
    AMGX_solver_destroy(X.solver);
    AMGX_vector_destroy(X.x);
    AMGX_vector_destroy(X.b);
    AMGX_matrix_destroy(X.A);
    AMGX_resources_destroy(X.rsrc);
    AMGX_config_destroy(X.cfg);
    X.built = false;
  }
  if (X.initialized) {
    AMGX_finalize();
    X.initialized = false;
  }
  delete amgx_;
  amgx_ = nullptr;
}

// Name only -- the run banner states the hierarchy before any setup runs.
// Kept beside choose_amgx_config so the two cannot drift apart.
const char* amgx_algo_name(int nranks) {
  const char *name = "", *why = "";
  (void)choose_amgx_config(nranks, &name, &why);
  return name;
}

}  // namespace nvrs

#else   // NVRS_HAVE_AMGX

namespace nvrs {
// Without AmgX the question is moot: the deck's LINSOLVER AMGX falls back to
// HYPRE, and the banner never asks for this name.
const char* amgx_algo_name(int) { return "aggregation"; }
}  // namespace nvrs

#endif  // NVRS_HAVE_AMGX
