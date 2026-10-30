// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// linear_solver.hpp — the CPR-preconditioned Krylov solver.
//
// This is the reservoir-simulator generalisation of the mixed-precision CPR
// reference (Mixed_precision/src/hypre_cusolver_mixed_cpr.cu).  The reference
// hard-wired a 2-unknown/cell system; here the block size `bs` is arbitrary
// (2 for two-phase, 3 for three-phase black oil, nc+1 for compositional).
//
//   * Outer Krylov  : FP64 FGMRES (Givens QR) OR FP64 BiCGSTAB   — user choice
//   * Preconditioner: CPR = [ HYPRE BoomerAMG on the pressure block ]
//                            ∘ [ bs×bs block-Jacobi second stage ]
//   * Precision     : AMG + stage-2 in FP32 by default (mixed); FP64 optional;
//                     df64-emulated FP64 operator optional (paper §7).
//   * Distribution  : HYPRE IJ matrix uses the global row range [row0,row0+n)
//                     from the K-slab decomposition, so the pressure solve runs
//                     across all ranks / GPUs / nodes with no renumbering.
//
// The full (coupled) operator A is stored once in device CSR (FP64) and applied
// with cuSPARSE SpMV; only the scalar *pressure* sub-block is handed to HYPRE.
#pragma once

#include "common.hpp"
#include "grid.hpp"

#include <mpi.h>
#include <vector>

// Forward declarations to keep HYPRE headers out of the public interface.
struct hypre_amg_impl;
struct hypre_mgr_impl;
struct amgx_impl;
struct ginkgo_impl;

namespace nvrs {

// Which AmgX hierarchy a run of `nranks` ranks will use, as a printable name
// ("classical PMIS" or "aggregation"), so the run banner can state it before
// the first solver setup happens.  Defined in amgx_backend.cu, and answers
// "aggregation" in a build without AmgX, where the question is moot.
const char* amgx_algo_name(int nranks);

// Whether this binary was linked against Ginkgo, and the version it was built
// with ("1.11.0"), or "not built" without it.  Defined in ginkgo_backend.cpp.
bool ginkgo_available();
const char* ginkgo_version();

// Device-resident coupled linear system  A x = b  in block-CSR-as-scalar-CSR.
struct DeviceLinearSystem {
  int  bs     = 2;      // unknowns per cell (block size)
  int  cells  = 0;      // local interior (owned) cells
  int  n      = 0;      // owned unknowns = bs * cells        → CSR *rows*
  // Owned + ghost unknowns = bs * cells_ext.  The coupled CSR has `n` rows but
  // `n_ext` columns, because a boundary cell couples across the K-slab seam to
  // a ghost cell filled by halo_exchange().  Every vector handed to the SpMV as
  // an operand must therefore be allocated with n_ext entries.
  int  cells_ext = 0;
  int  n_ext  = 0;
  int  nnz    = 0;      // nonzeros in the coupled operator
  long long row0 = 0;   // global row offset of the pressure block (per rank)
  int  cells_global = 0;

  // Coupled operator (scalar CSR over all unknowns), FP64 + FP32 mirror.
  int*   rowptr = nullptr;
  int*   colind = nullptr;
  Real*  aval64 = nullptr;
  float* aval_hi = nullptr;   // df64 hi slice (emulation path only)
  float* aval_lo = nullptr;   // df64 lo slice

  // Pressure sub-block CSR (one row per cell), values in HYPRE precision.
  int*      prowptr = nullptr;
  int*      pcolind = nullptr;   // *global* pressure column ids
  PrecReal* pval    = nullptr;
  int       pnnz    = 0;
  // Row weights used to build pval, one bs-vector per cell (quasi-IMPES: the
  // first row of the inverse cell block).  CPR is only a valid preconditioner
  // if the *same* weighting restricts the residual, A_p = W A P with rhs W r;
  // restricting with e_0 while building A_p with W solves a different problem
  // and the outer Krylov stalls.
  PrecReal* pweight = nullptr;   // cells * bs

  // Per-cell bs×bs block-diagonal inverse for the stage-2 smoother.
  PrecReal* dinv = nullptr;      // cells * bs * bs, row-major per block

  // Optional matrix-free low-rank-per-well correction. The pointed-to entries
  // are owned by Reservoir and refreshed with the well controls; production
  // mode uses one BHP mode and experimental coupled modes use three.
  const void* well_schur = nullptr;
  int well_schur_count = 0, well_schur_groups = 0, well_schur_modes = 1;

  // Assembly topology, shared with the Reservoir (set by build_sparsity, not
  // owned here): self + 6 face neighbours per cell, and each member's scalar
  // column offset within its CSR row.  The ZLINE stage-2 factorisation reads
  // the vertical coupling blocks out of the coupled CSR through these.
  const int* nbr7  = nullptr;    // cells * 7
  const int* bcol7 = nullptr;    // cells * 7

  Real* b = nullptr;             // rhs (device, FP64)
};

struct SolverConfig {
  int  krylov = 0;            // 0 = FGMRES, 1 = BiCGSTAB
  Real tol = 1e-6;
  int  max_iters = 100;
  int  restart = 30;          // FGMRES restart
  int  ortho_passes = 2;      // 1 = CGS1, 2 = CGS2 (robust default)
  bool use_cprw = false;      // include eliminated wells in pressure CPR stage
  bool mixed = true;          // FP32 AMG preconditioner
  bool emulate_fp64 = false;  // df64-emulated FP64 operator
  // The arithmetic of the Krylov operator apply J z (EMULATE):
  //   0  native FP64 values
  //   1  df64: the FP64 value stored as two FP32 slices (emulate_fp64)
  //   2  FP32-stored values (the hi slice alone), FP64 accumulation.  The
  //      inexact-Krylov bound (Simoncini and Szyld 2003) lets the matvec be
  //      perturbed by up to eta * |r0| / |r_j| at step j; with eta = 1e-2 an
  //      FP32 value (relative error 6e-8) is inside that from the first step.
  //      The residual that decides convergence, at every restart and at
  //      exit, is still formed with the FP64 master copy (op_apply_master).
  // -1 means "follow emulate_fp64" (0 or 1), which is what older callers set.
  int  operator_mode = -1;
  bool block_spmv = true;     // block-indexed operator apply (BLOCKSPMV)
  bool cuda_graph = true;     // graph-replayed Arnoldi tail (CUDAGRAPH)
  // CUDAGRAPHMINREUSE: turn graph capture off for the rest of the run when,
  // after graph_guard_after captures, the graphs have been launched fewer than
  // this many times per capture beyond the launch that follows each capture.
  // A capture plus instantiate costs tens of microseconds; a graph that is
  // replayed about once before its signature changes cannot pay that back.
  // 0 disables the guard (the published behaviour).
  double graph_min_reuse = 0.0;
  int    graph_guard_after = 192;
  // BoomerAMG knobs (defaults = the GPU-tuned strong cycle from the reference).
  int coarsen_type = 8, relax_type = 18, interp_type = 6;
  int num_sweeps = 2, agg_num_levels = 1, pmax = 4;
  double strong_threshold = 0.5, trunc_factor = 0.3;
  int amg_print_level = 0;
  // Size at which BoomerAMG stops coarsening and solves the coarse system
  // directly.  This is the single most important multi-GPU knob, and 64 (the
  // value tuned for one rank) is the wrong one for several.
  //
  // Every extra level costs a round of neighbour exchanges in the setup and two
  // more in each V-cycle, and those rounds are pure latency once a level holds
  // only a few hundred rows per rank.  Coarsening a million cells to 64 takes
  // roughly ten levels; stopping at a few thousand removes the three or four
  // cheapest-per-row and most-expensive-per-message of them.  Measured on
  // 1,000,000 cells, AMG setup went 46.7 s on one GPU to 140.9 s on two for the
  // same number of rebuilds -- that 3x is what this targets.
  //
  // 0 selects a rank-aware default: 64 on one rank, 2000 on several.
  int amg_max_coarse = 0;

  // --- coarse-level cost control (AMGFAST) ---------------------------
  // Three things that matter on several GPUs and not at all on one.
  //
  // amg_seq_threshold is the important one.  Below this many rows the rest
  // of the hierarchy is gathered onto a single rank and solved there,
  // instead of staying spread across every GPU holding a few hundred rows
  // apiece and exchanging messages about them.  Those bottom levels are
  // nearly all latency, and they are what makes AMG setup grow with rank
  // count while the iteration counts stay flat.
  //
  // amg_nongalerkin_tol drops small entries from the coarse operators, so
  // RAP stays sparse rather than filling in as the levels coarsen.  It
  // changes the operator, so it is off unless asked for.  The usual recipe
  // is nothing on the fine level and a little more further down.
  //
  // agg_trunc_factor / agg_pmax keep the interpolation sparse on the
  // aggressive levels, where it is otherwise much denser than on the rest.
  // Relaxation on the COARSEST grid only.  HYPRE's default is 9, Gaussian
  // elimination, and on a GPU build that is the single most expensive thing
  // in a multi-rank V-cycle: the coarse operator is gathered across ranks and
  // the dense solve runs on the HOST, so each cycle pays a device-to-host
  // copy, an MPI gather and a host-to-device copy.  It is invisible on one
  // rank, where there is nothing to gather.
  //
  // 18 (l1-Jacobi) keeps the coarsest level on the device.  It is a weaker
  // solve, but the coarsest grid here holds nineteen rows inside a
  // preconditioner that is only ever applied for one V-cycle, so the
  // iteration count is what decides whether that matters.
  int amg_coarse_relax = 9;         // 9 = GE (HYPRE default), 18 = l1-Jacobi
  int amg_seq_threshold = 0;        // 0 = leave HYPRE's default
  double amg_nongalerkin_tol = 0.0; // 0 = off
  double agg_trunc_factor = 0.0;
  int agg_pmax = 0;
  // Store the interpolation transposes at setup so the V-cycle's restriction is
  // a matvec instead of a transposed matvec.  HYPRE's own GPU example turns
  // this on ("keep transpose to avoid SpMTV"); a transposed SpMV on a
  // distributed ParCSR matrix is far more expensive than the memory the stored
  // transpose costs.
  int keep_transpose = 1;

  // --- AMG hierarchy reuse -------------------------------------------------
  // Rebuilding the BoomerAMG hierarchy from scratch on every Newton iteration
  // is the single largest cost in the run: on 46 x 112 x 22 it is 30 s of a
  // 65 s solve on one GPU, and 158 s of 225 s on four, because the coarsening
  // is a chain of latency-bound neighbour exchanges and global reductions whose
  // cost barely falls as the grid is split.  Within one timestep the pressure
  // coefficients move very little between Newton iterates, so the hierarchy
  // built for iterate k is still a good preconditioner for iterate k+1 --
  // OPM Flow makes the same trade (--cpr-reuse-setup).
  //
  //   amg_reuse        maximum consecutive setups to skip.  0 restores the old
  //                    rebuild-every-iteration behaviour exactly.
  //   amg_reuse_growth rebuild early if the Krylov count on the stale hierarchy
  //                    exceeds this multiple of the count the fresh one needed.
  //                    This is what keeps reuse from quietly trading a cheap
  //                    setup for many more expensive iterations.
  int amg_reuse = 4;
  double amg_reuse_growth = 1.3;

  // --- preconditioner choice ----------------------------------------
  // 0 = the hand-rolled CPR above.  1 = HYPRE's MGR, which performs the
  // same pressure/saturation reduction inside the library, from the real
  // block structure rather than from a diagonal approximation of it.
  // Both run FP32 because HYPRE is built FP32, so the mixed-precision
  // arrangement is identical either way.
  // Which library provides the stage-1 pressure multigrid.  Everything
  // else -- FP64 FGMRES, the df64-emulated operator, the quasi-IMPES
  // weights, the bs x bs stage-2 correction -- is the same either way, and
  // both run their multigrid in FP32, so the mixed-precision boundary does
  // not move.  Only the multigrid does.
  enum Backend { kHypre = 0, kAmgx = 1, kGinkgo = 2 };
  int backend = kAmgx;
  // Ginkgo only (LINSOLVER GINKGO): the precision the WHOLE multigrid runs in,
  // hierarchy, smoothers, coarse solve and the vectors it cycles.  16 is IEEE
  // half precision, the point of offering this backend at all; 32 is the same
  // Ginkgo hierarchy in FP32, so the two can be compared with nothing else
  // changed.  Either way the pressure system is symmetrically Jacobi-scaled
  // and the right-hand side normalised before it is narrowed, because field
  // units put pressure-matrix entries well past FP16's 65,504.
  int ginkgo_precision = 16;
  // --- CPR stage 2 ---------------------------------------------------
  // 0 = per-cell bs x bs block-Jacobi (the default; one inverse per cell).
  // 1 = ZLINE: block-tridiagonal line relaxation along each vertical grid
  //     column, factored once per Newton iterate and applied by block
  //     Thomas substitution.  Columns are independent, so the sweep needs
  //     no colouring and parallelises over the areal grid.  This is the
  //     textbook stage 2 for strong vertical coupling: on SPE10 the cells
  //     are 20 x 10 x 2 ft, so vertical transmissibility is ~100x the
  //     horizontal and the saturation error the multigrid cannot see lies
  //     along the columns.  Under a Z-cut decomposition (pz > 1) each rank
  //     relaxes its own column segment; the cross-rank coupling is simply
  //     left to the outer Krylov, like every other off-rank term here.
  // 2 = Ginkgo's adaptive-precision block-Jacobi on the same bs x bs cell
  //     blocks: each inverse is stored in FP64, FP32 or FP16 according to its
  //     conditioning (STAGE2 GINKGO).  Single GPU, needs a Ginkgo build.
  int stage2 = 0;
  int use_mgr = 0;
  int mgr_relax_type = 18;      // l1-Jacobi: parallel, unlike Gauss-Seidel
  int mgr_relax_sweeps = 2;
  int mgr_interp_type = 2;      // diagonal-scaled (Jacobi) interpolation
  // MGR global smoother.  0 = block Jacobi, which is what was intended and
  // is GPU-supported; 16 is ILU, which is neither.
  int mgr_global_smooth = 0;
  int mgr_global_sweeps = 1;
};

// Where the linear solve actually went.  The mixed-precision CPR work is a
// claim about the preconditioner, so "linear solve time" as one number cannot
// support or refute it: the interesting split is HYPRE's FP32 BoomerAMG on the
// pressure block against the bs x bs block-Jacobi second stage against the
// FP64 operator applications driving the outer Krylov.  Every timer
// synchronises the device before reading the clock, since the kernels are
// asynchronous and an unsynchronised timer measures launch latency.
struct CprTimings {
  double amg_setup = 0.0;    // HYPRE_BoomerAMGSetup, once per Newton iterate
  double amg_solve = 0.0;    // HYPRE_BoomerAMGSolve, once per Krylov iterate
  double stage2 = 0.0;       // block-Jacobi correction, including its operator
  // Building the stage-2 operator where that is a separate step: Ginkgo's
  // adaptive-precision block-Jacobi generates (inverts and chooses a
  // precision for) every cell block once per setup.  Zero otherwise, since
  // the built-in block-Jacobi inverses come from assembly.
  double stage2_setup = 0.0;
  double spmv = 0.0;         // FP64 (or df64) operator applications
  // CGS orthogonalisation: its projection pass(es), reductions, and the host
  // read of the Hessenberg column.  Once the vendor SpMV was
  // out of the V-cycle this became the largest single term in the solve,
  // and it had never been measured.
  double ortho = 0.0;
  // Handing the residual to HYPRE and taking the correction back: IJVector
  // SetValues, Assemble and GetValues, once around every AMG apply.  Pure
  // marshalling, no arithmetic, and never measured until now.
  double ijvec = 0.0;
  long   n_amg_setup = 0;
  long   n_amg_solve = 0;
  long   amg_iters = 0;      // summed HYPRE_BoomerAMGGetNumIterations
  // Setups that were skipped because the previous hierarchy was reused, and of
  // those skips, how many were cut short by the Krylov-growth guard rather than
  // by the reuse limit.  A large n_amg_forced means amg_reuse is set too high
  // for this problem: the hierarchy is going stale faster than it is being
  // rebuilt, and the guard is paying for it in extra iterations.
  long   n_amg_reuse = 0;
  long   n_amg_forced = 0;
  // CUDAGRAPH bookkeeping: graphs captured (and instantiated), and replays.
  // Many captures per replay means the system signature keeps changing and
  // the graphs are not paying for themselves.
  long   graph_captures = 0;
  // Every cudaGraphLaunch, INCLUDING the launch that immediately follows each
  // capture.  Reuses proper are graph_replays - graph_captures.
  long   graph_replays = 0;
  // Host time spent recording and instantiating graphs (begin capture through
  // cudaGraphInstantiate), so the cost of a capture is a measurement.
  double graph_capture_seconds = 0.0;
  // The CUDAGRAPHMINREUSE guard: whether it fired, and after how many captures.
  int    graph_guard_fired = 0;
  long   graph_guard_at = 0;
  // Operator applies taken by the block kernel, and by the scalar one.
  long   spmv_block = 0;
  long   spmv_scalar = 0;
  // Of those, the applies made with FP32-stored values (EMULATE 2).
  long   spmv_fp32 = 0;
  // FGMRES residual gap.  The Givens recursion gives |r_j| without forming it;
  // at every restart and at exit the true residual b - J x is formed with the
  // FP64 operator.  fgmres_gap_max is the largest ratio true / recursive seen
  // at a point where the recursion had declared convergence, and
  // fgmres_gap_restarts counts the solves that had to continue because the
  // true residual had not in fact reached the tolerance.
  double fgmres_gap_max = 0.0;
  long   fgmres_gap_checks = 0;
  long   fgmres_gap_restarts = 0;
};

struct SolveStats {
  int iters = 0;
  Real final_residual = 0;   // ||b - Ax|| / ||b||
  Real sol_norm = 0;         // ||x||, i.e. the size of the Newton update
  bool converged = false;
};

// The solver owns cuSPARSE/cuBLAS handles, HYPRE AMG state and Krylov scratch.
class CprSolver {
 public:
  CprSolver(const Grid& grid, int block_size, const SolverConfig& cfg);

  // Hand the solver the unstructured halo a field run uses.
  //
  // Without this the operator apply exchanges ghosts through the STRUCTURED
  // K-slab halo built from the grid it was constructed with -- and for a field
  // run that grid is the run deck's DIMENS, which a field deck never sets.
  // The messages it posts then have nothing to do with the real ghost layout,
  // which comes from the partition (Norne's four ranks hold 504, 1062, 1064
  // and 500 ghosts), so the sends and receives do not pair and MPI_Waitall
  // never returns.  That is a hang in the first linear solve, on any AMG
  // backend, because it is ours and not theirs.
  //
  // Pass Reservoir::halo_conn.  Null (a structured run) keeps the K-slab path.
  void set_conn_halo(void* halo_conn) { conn_halo_ = halo_conn; }
  ~CprSolver();

  // (Re)build the AMG hierarchy from the current pressure sub-block.  Call once
  // per pressure matrix; for a frozen-coefficient sequential-implicit step the
  // hierarchy can be reused across Newton iterations.
  //
  // With cfg.amg_reuse > 0 this may keep the previous hierarchy *and* the
  // previous pressure matrix, in which case the call is nearly free.  The
  // preconditioner is then the one frozen at the last rebuild; that is still a
  // valid preconditioner for both outer Krylov methods, since it is constant
  // for the whole of the solve() that follows.
  void setup(const DeviceLinearSystem& sys);

  // Which preconditioner is actually in use, for the run banner.
  bool using_mgr() const { return cfg_.use_mgr != 0; }

  // Discard any reusable hierarchy, so the next setup() rebuilds.  Call this
  // whenever the state jumps discontinuously -- a timestep chop rolls the
  // solution back, so the pressure matrix that follows has nothing to do with
  // the one the current hierarchy was built from -- or before handing the
  // solver a *different* linear system through the same DeviceLinearSystem.
  void invalidate_amg();

  // Solve A x = b into device vector x (length sys.n).  x is used as the
  // initial guess unless zero_initial_guess is true.  The explicit flag lets
  // callers that have just cleared x avoid one otherwise redundant A*0 SpMV.
  SolveStats solve(const DeviceLinearSystem& sys, Real* x,
                   bool zero_initial_guess = false);

  // Change only the outer inexact-Krylov stopping tolerance. The CPR
  // hierarchy and matrix are untouched, so Newton may set a forcing term per
  // iteration without rebuilding anything extra.
  void set_tolerance(Real tol) { cfg_.tol = tol; }

 private:
  Grid grid_;
  // Unstructured halo for a field partition; null on a structured run.
  void* conn_halo_ = nullptr;
  int  bs_;
  SolverConfig cfg_;
  hypre_amg_impl* amg_ = nullptr;
  hypre_mgr_impl* mgr_ = nullptr;
  amgx_impl* amgx_ = nullptr;
  ginkgo_impl* ginkgo_ = nullptr;
  // AMGPRINT is a one-shot: the level table is the same every rebuild.
  bool amg_printed_ = false;
  bool vecpath_said_ = false;

  cusparseHandle_t sp_ = nullptr;
  cublasHandle_t   blas_ = nullptr;
  cusparseSpMatDescr_t matA_ = nullptr;
  cusparseDnVecDescr_t vx_ = nullptr, vy_ = nullptr;
  void*  spmv_buf_ = nullptr;
  size_t spmv_bufsz_ = 0;
  // Shape the cached cuSPARSE descriptor was built for; one solver is
  // reused across systems of different shape, so it must be rebuilt when
  // the shape changes rather than silently describing the old matrix.
  int mat_rows_ = -1, mat_cols_ = -1, mat_nnz_ = -1;

  // Krylov scratch.  Vectors that are ever used as an SpMV *operand* are
  // allocated with n_ext entries and carry a ghost region; the Krylov algebra
  // itself only ever touches the first n (owned) entries.  V_/Z_ are stored
  // column-major with leading dimension next_ for exactly that reason.
  int n_alloc_ = 0;    // owned length the scratch was sized for
  int next_ = 0;       // extended (owned + ghost) column stride
  Real *r_ = nullptr, *Ax_ = nullptr, *tmp_ = nullptr;
  Real *V_ = nullptr, *Z_ = nullptr, *hdev_ = nullptr, *sdev_ = nullptr, *yv_ = nullptr;
  Real *p_ = nullptr, *phat_ = nullptr, *s_ = nullptr, *shat_ = nullptr,
       *t_ = nullptr, *v_ = nullptr, *rhat_ = nullptr;  // BiCGSTAB vectors
  Real* well_schur_dot_ = nullptr;
  int well_schur_alloc_ = 0;
  // Full-system Woodbury workspace.  The sparse reservoir operator is solved
  // once for the physical RHS and once for each active eliminated-well mode;
  // keeping these vectors resident avoids allocating O(cells) storage at every
  // Newton iteration.
  Real* well_lr_rhs_ = nullptr;
  Real* well_lr_z_ = nullptr;
  int well_lr_next_alloc_ = 0, well_lr_modes_alloc_ = 0;
  // CPRW: pressure-space responses Q = M_p U_w and the inverse of the tiny
  // global well Schur matrix I + V_w^T Q. This exposes eliminated well modes
  // to the global AMG stage instead of leaving them only to local stage 2.
  Real* cprw_u_ = nullptr;
  PrecReal* cprw_q_ = nullptr;
  Real* cprw_kinv_ = nullptr;
  Real* cprw_coeff_ = nullptr;
  int* cprw_active_ = nullptr;
  int cprw_cells_alloc_ = 0, cprw_modes_alloc_ = 0;
  int cprw_na_ = 0, cprw_nfull_ = 0;
  PrecReal *rp_ = nullptr, *dp_ = nullptr;
  // ZLINE stage-2 factors (cfg_.stage2 == 1): per-cell inverse of the
  // modified block diagonal, the two vertical coupling blocks, and the
  // forward-substitution workspace.  Rebuilt by setup_zline() every setup().
  PrecReal *zl_tinv_ = nullptr, *zl_l_ = nullptr, *zl_u_ = nullptr,
           *zl_y_ = nullptr;
  int zl_cells_ = 0;

  // --- AMG reuse bookkeeping ------------------------------------------------
  // How many more setups may be skipped before the next forced rebuild.
  int reuse_left_ = 0;
  // Krylov iterations the *fresh* hierarchy needed, which is the yardstick the
  // growth guard compares against.  Negative means "no measurement yet".
  int iters_fresh_ = -1;
  // Whether the hierarchy currently held was built from the system handed to
  // the last setup() call, so solve() knows which side of the comparison the
  // count it is about to record belongs to.
  bool amg_fresh_ = true;
  // Fingerprint of the pressure sub-block the hierarchy was built from.  One
  // CprSolver is reused across different systems (the sequential scheme feeds
  // it a scalar pressure system and a transport system through the same
  // DeviceLinearSystem), and reusing a hierarchy across that switch would
  // precondition with the wrong matrix.  Cheap insurance against a future
  // caller.
  const void* amg_pval_ = nullptr;
  int amg_cells_ = -1, amg_pnnz_ = -1;

  // --- per-iteration timing without per-iteration synchronisation ----------
  // The CPR phase timers used to bracket each region with
  // cudaDeviceSynchronize().  apply_cpr runs once per Krylov iteration, so that
  // was four full pipeline drains per iteration and tens of thousands per run:
  // a few percent on one GPU, and worse on several, because each drain also
  // destroys the overlap between compute and MPI progress that multi-rank
  // performance depends on.
  //
  // Record CUDA events instead.  cudaEventRecord is asynchronous and costs
  // roughly a microsecond, and the elapsed times are summed once per solve
  // rather than once per iteration -- the same numbers, two orders of
  // magnitude fewer synchronisations.
  struct EventSpan {
    std::vector<cudaEvent_t> beg, end;
    size_t used = 0;
    void reserve_pair();
    void tic();
    void toc();
    double drain();          // sums the recorded spans and rewinds
    ~EventSpan();
  };
  // ev_spmv_ covers the operator application, halo included.  CprTimings has
  // carried an spmv field from the start but nothing ever wrote to it, so
  // the operator half of the linear solve was invisible.
  EventSpan ev_amg_, ev_stage2_, ev_spmv_, ev_ortho_, ev_ijvec_;

  CprTimings timings_;

 public:
  // Cumulative, over the whole run; main.cu reads it once at the end.
  const CprTimings& timings() const { return timings_; }

 private:

  void ensure_scratch(const DeviceLinearSystem& sys);
  // Extract the vertical coupling blocks from the coupled CSR and factor the
  // per-column block tridiagonal (ZLINE stage 2).  Called from setup().
  void setup_zline(const DeviceLinearSystem& sys);
  // `x` is non-const because the halo exchange writes its ghost region before
  // the SpMV; the owned entries are never modified.
  void op_apply(const DeviceLinearSystem& sys, Real* x, Real* y);
  // The operator with its FP64 master values, whatever EMULATE says.  Used
  // for the residual that decides convergence.  For EMULATE 0 and 1 this is
  // op_apply itself, so their results are unchanged.
  void op_apply_master(const DeviceLinearSystem& sys, Real* x, Real* y);
  // Set only inside op_apply_master, to route one apply through FP64 values.
  bool master_apply_ = false;
  // EMULATE resolved to 0 (FP64), 1 (df64) or 2 (FP32-stored).
  int operator_mode() const {
    return cfg_.operator_mode >= 0 ? cfg_.operator_mode
                                   : (cfg_.emulate_fp64 ? 1 : 0);
  }
  void apply_well_schur(const DeviceLinearSystem& sys, const Real* x, Real* y);
  void apply_cpr(const DeviceLinearSystem& sys, const Real* r, Real* z, Real* tmpAz);
  // apply_cpr in two halves.  The head restricts the residual and runs the
  // stage-1 multigrid (a library call, with host synchronisation inside); the
  // tail is the CPRW correction, the prolongation and the second stage, all
  // our own kernels, which is what CUDAGRAPH captures.  Returns the pressure
  // correction slot the tail reads.
  PrecReal* apply_cpr_head(const DeviceLinearSystem& sys, const Real* r);
  void apply_cpr_tail(const DeviceLinearSystem& sys, const Real* r, Real* z,
                      Real* tmpAz, PrecReal* xslot);
  // The Gram-Schmidt projections of w against the first nc basis vectors,
  // and |w|^2 into ndev_.  No host read: fgmres fetches everything at once.
  void ortho_project(int n, int nc, Real* w, bool cgs2);

  // --- block operator storage (BLOCKSPMV) -----------------------------------
  // One local cell index per bs x bs block, built from the scalar CSR the
  // first time a sparsity pattern is seen and checked against it on the
  // device.  bcol_state_: 0 not built, 1 usable, -1 the pattern does not have
  // the block layout (the scalar kernels are used, and the run says so).
  bool block_ready(const DeviceLinearSystem& sys);
  int* bcol_ = nullptr;
  int  bcol_alloc_ = 0;
  int* bcol_bad_ = nullptr;
  int  bcol_state_ = 0;
  const void* bcol_rowptr_ = nullptr;
  const void* bcol_colind_ = nullptr;
  int bcol_nnz_ = -1, bcol_cells_ = -1, bcol_bs_ = -1;

  // --- CUDA graphs (CUDAGRAPH) ------------------------------------------------
  // st_ is the stream every capturable kernel is launched on: the legacy
  // default stream normally, the capture stream gs_ while a graph is being
  // recorded.  gs_ is a blocking stream, so work on it is ordered against the
  // legacy stream the rest of the solver (and the AMG libraries) use.
  cudaStream_t st_ = 0;
  cudaStream_t gs_ = nullptr;
  void* cublas_ws_ = nullptr;         // fixed cuBLAS workspace for capture
  bool capturing_ = false;
  // NCCL for the Krylov tail on several ranks (nccl_comm.hpp): created once,
  // at the first solve, when CUDAGRAPH is on and NCCL is built in.  With it
  // the tail's halo exchanges and reductions are stream-ordered, so the tail
  // is captured on every rank; without it several ranks keep the MPI path.
  bool nccl_ = false;
  bool nccl_tried_ = false;
  bool graphs_off_ = false;         // a capture failed; never try again
  std::vector<unsigned long long> gsig_;
  std::vector<cudaGraphExec_t> gexec_;   // [column * 3 + phase]
  Real* ndev_ = nullptr;            // |w|^2, packed after hdev_ and sdev_
  bool graphs_usable(const DeviceLinearSystem& sys) const;
  void graph_signature(const DeviceLinearSystem& sys, const PrecReal* xslot,
                       bool cgs2, std::vector<unsigned long long>& sig) const;
  void graphs_clear();
  void ensure_small_scratch(const DeviceLinearSystem& sys);
  template <class F>
  void graph_phase(int column, int phase, EventSpan& ev, F&& body);
  void setup_cprw(const DeviceLinearSystem& sys);
  void apply_cprw(const DeviceLinearSystem& sys, PrecReal* pressure_update);
  // One place that configures a BoomerAMG instance, so the pressure solve
  // inside CPR and the coarse solve inside MGR are identical and a
  // comparison between them is a comparison of the reduction alone.
  void configure_amg(void* solver) const;
  void setup_mgr(const DeviceLinearSystem& sys);
  void apply_mgr(const DeviceLinearSystem& sys, const Real* r, Real* z);
  // Defined in amgx_backend.cu, compiled only with NVRS_HAVE_AMGX.
  void setup_amgx(const DeviceLinearSystem& sys);
  void apply_amgx(const DeviceLinearSystem& sys, PrecReal* bp, PrecReal* xp);
  // One stage-1 pressure apply x = M_p^{-1} b through whichever backend is
  // configured (AmgX, Ginkgo or HYPRE BoomerAMG), from and into our own
  // staging buffers.  CPRW builds its well columns with it.
  void apply_stage1(const DeviceLinearSystem& sys, PrecReal* bp, PrecReal* xp);
  void destroy_amgx();
  // Defined in ginkgo_backend.cpp; stubs that abort without NVRS_HAVE_GINKGO.
  void setup_ginkgo(const DeviceLinearSystem& sys);
  void apply_ginkgo(const DeviceLinearSystem& sys, PrecReal* bp, PrecReal* xp);
  void destroy_ginkgo();
  // STAGE2 GINKGO: generate the adaptive block-Jacobi from the coupled FP64
  // operator (every setup), and apply z += M^-1 (r - A z), leaving M^-1 (r - A z)
  // in Az exactly as the built-in stage 2 does for the well correction.
  void setup_ginkgo_stage2(const DeviceLinearSystem& sys);
  void apply_ginkgo_stage2(const DeviceLinearSystem& sys, const Real* r,
                           Real* Az, Real* z);
  SolveStats fgmres(const DeviceLinearSystem& sys, Real* x,
                    bool zero_initial_guess);
  SolveStats bicgstab(const DeviceLinearSystem& sys, Real* x,
                      bool zero_initial_guess);
  // MPI-reduced dot / norm across ranks.
  Real gdot(int n, const Real* a, const Real* b);
  Real gnrm2(int n, const Real* a);
  // In-place cross-rank sum of a device vector.  Uses device pointers directly
  // when MPI is CUDA-aware, otherwise stages through `host`.
  void allreduce_dev(Real* dev, int count, std::vector<Real>& host);
  std::vector<Real> redbuf_;   // staging for allreduce_dev, reused
  Real* redev_ = nullptr;      // device receive buffer for allreduce_dev
  int   redev_count_ = 0;
};

}  // namespace nvrs
