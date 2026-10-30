// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// deck.hpp — the input deck.  The deck grammar follows the OPM Flow / Eclipse
// convention: uppercase keywords introduce sections, one record per line, a
// slash `/` terminates a record list, and `--` starts a comment.  A single
// MODEL keyword selects the physics (BLACKOIL or COMPOSITIONAL) exactly as the
// `skill` brief requests.  The parser is deliberately small and dependency-free
// so the compiled executable can read a deck and start simulating immediately.
#pragma once

#include "common.hpp"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace nvrs {

struct WellSpec {
  std::string name;
  int i = 0, j = 0;          // 1-based areal location (as in the deck)
  int type = 1;              //  1 = injector, -1 = producer
  Real rate = 0.0;           //  target surface rate [STB/D] (inj>0), or -1 → BHP
  Real rw   = 0.5;           //  wellbore radius [ft]
  Real skin = 0.0;
  Real pwf  = 100.0;         //  BHP target for producers [psia]
  std::string perf = "all";  //  "all" layers or "last" layer only
};

// Relative-permeability table row (Sat, Kr_o, Kr_{w|g}).
using SatRow = std::array<Real, 3>;

struct Deck {
  // ---- MODEL / RUNSPEC --------------------------------------------------
  Model       model      = Model::BlackOil3P;
  int         ncomp      = 3;      // # black-oil phases OR # hydrocarbon comps
  std::string title      = "NVRS-CUDA reservoir model";
  std::string perm_path;           // optional external permeability file

  // --- field deck (corner-point) -----------------------------------------
  // FIELDDECK names an ECLIPSE-format corner-point deck.  When given, the
  // grid, the wells and the initial state come from it, and the keywords in
  // this deck supply only the fluid and solver settings.  See field_model.hpp.
  std::string field_deck;
  // Phase pressure gradients for equilibration, in the field deck's own
  // pressure and length units.  They cannot be derived without PVT tables and
  // are never guessed, so equilibration is skipped unless these are given.
  Real grad_oil = 0, grad_water = 0, grad_gas = 0;
  bool have_gradients = false;
  // Connate water for the hydrocarbon columns, where the field deck's grid
  // carries no SWL array.
  Real connate_water = 0;
  // Use the field deck's schedule to drive well controls at report steps.
  int  use_schedule = 1;
  // Report at the field deck's own dates rather than every DTREPORT days.
  // On by default, because a comparison against Flow -- which reports at
  // those dates -- otherwise samples the two simulators at different times.
  // 0 restores the uniform DTREPORT grid.  Ignored without a FIELDDECK.
  int  report_dates = 1;
  // Flowing bottom-hole pressure for a producer whose schedule record does not
  // state one, in the run deck's pressure units.
  //
  // This is needed because of a genuine mismatch.  This simulator's producers
  // are pressure-controlled; a history-matched deck's producers are rate
  // controlled, and its WCONHIST records give observed rates and default the
  // bottom-hole pressure entirely -- all 2,109 of Norne's do.  Without a value
  // here such a well would be handed a BHP of zero and become a sink pulling
  // the full reservoir pressure across every completion, which no timestep can
  // survive.  Left unset, those wells stay shut and are reported.
  Real prod_bhp = 0;
  // Largest drawdown allowed for a producer we are controlling ourselves
  // -- one whose deck record defaults the bottom-hole pressure AND states
  // no rate.  PRODBHP is then not the deck's instruction but our
  // invention, and on Norne it is 2900 psi against 4100+ psi of
  // reservoir: C-3H converts from gas injection to production and opens
  // with 1200 psi of drawdown across every completion, into rock its
  // neighbours have been filling with gas.  The gas residual arrives at
  // 6.6e9 and no timestep chop settles it.  Capping the drawdown bounds
  // an invented control; it does not touch a well the deck actually
  // controls.  0 disables.  [psi]
  Real prod_draw_max = 500;
  // WELLXFLOW -- may a producer's completion flow backwards into the formation?
  //
  // 0 (default): no.  A rate-controlled producer's flowing pressure is solved
  // so that only the completions above it carry the target, and the rest are
  // shut for the step.  This is OPM's own !allow_cf branch in
  // StandardWell::computePerfRate.
  //
  // 1: the previous behaviour -- one division over every completion, letting
  // the ones below the flowing pressure subtract from the total.  That is
  // neither of OPM's two branches: OPM's reverse-flow path uses TOTAL mobility
  // distributed by the wellbore mixture, so its reverse flow always carries
  // the well's own composition, while this one applies each phase's local
  // formation mobility and can take oil in at one completion while pushing gas
  // out of another.  On Norne that is B-4H and D-4H reporting large negative
  // oil beside large positive gas.
  //
  // 2: allow reverse completion flow, as WELSPECS defaults to, but carry the
  // producer wellbore's mixture on that flow.  This is the physical OPM
  // branch; unlike mode 1 it never injects a cell's local phase composition
  // backwards through a common wellbore.
  //
  // Kept as a switch so a run can be put back exactly as it was, which is the
  // only way to attribute a change in the comparison to this and not to
  // something else.  See include/well_control.hpp.
  int well_crossflow = 0;
  // Pressure ceiling for an injector whose deck record defaults one.
  //
  // Without a ceiling the injector is a constant source: the kernel's
  // injectivity cap is gated on pwf > 0, so a defaulted limit disables it
  // and the well delivers its full stated rate whatever the cell pressure
  // does.  The residual then contains a term independent of the unknowns
  // and Newton freezes on it -- |R| identical to five digits for sixty
  // iterations while |du| bounces, which is what Norne's step 49 does at
  // 9.1e7, the size of the gas rate C-4H is being given.
  //
  // 0 leaves such a well uncapped.  [psia]
  Real inj_bhp = 0;
  // Re-solve rate-controlled wells' flowing pressure every Newton
  // iteration rather than once per report step.  1 (default) is closer to
  // what OPM does, which carries the well equation in the coupled system;
  // 0 restores the frozen-at-step-start behaviour for comparison.
  int  well_iter = 1;
  // Pressure part of eliminated rate-controlled well equations, applied as a
  // matrix-free rank-one correction.  Opt-in while benchmarked against the
  // established Norne trajectory.
  int  well_schur = 0;
  // OPM-style black-oil composition switching.  In a cell containing free
  // gas the third primary is Sg; after gas disappears it becomes Rs, and after
  // oil disappears it becomes Rv. This is
  // opt-in while the coupled implementation is benchmarked on field decks.
  int  pv_switch = 0;

  // Optional per-cell rock files, same format and order as PERMFILE (one
  // value per global cell, natural order).  POROFILE replaces the linear
  // perm -> [poro_min, poro_max] map; PERMZFILE replaces Kz = KZMULT * Kx.
  // SPE10 model 2 needs both: its porosity is an independent measurement and
  // its vertical permeability is not a scalar multiple of the horizontal.
  std::string poro_path;
  std::string permz_path;
  std::string out_dir    = "OUTPUT";

  // ---- DIMENS / GRID ----------------------------------------------------
  int  nx = 1, ny = 1, nz = 1;
  Real dx = 50, dy = 50, dz = 20;  // [ft] uniform cell sizes
  Real kz_mult = 0.2;              // Kz = kz_mult * Kx
  // CPR stage 2: 0 = bs x bs block-Jacobi (default), 1 = ZLINE, the vertical
  // block-tridiagonal line relaxation, 2 = Ginkgo's adaptive-precision
  // block-Jacobi (deck keyword STAGE2 JACOBI | ZLINE | GINKGO).
  int stage2 = 0;
  // Take SWOF and SGOF from the FIELD deck rather than this one
  // (FIELDSATFUNC 1, the default).
  //
  // decks/NORNE.DATA carries a hand-written 10-row SWOF and 8-row SGOF, and
  // that placeholder is what every NVRS run of Norne used, while OPM Flow
  // read Equinor's SCAL_NORNE.INC -- around 19 rows, different curves.  The
  // two simulators were being compared on different rock, which is not a
  // benchmark.  Set to 0 to go back to the run deck's own tables.
  int field_satfunc = 1;

  // EHYSTR HYSTMOD=0: Carlson non-wetting relative-permeability hysteresis
  // in both oil/water and gas/oil, using
  // IMBNUM and I-prefixed imbibition endpoints.  Other modes stay disabled
  // until their distinct scanning-curve laws are implemented.
  int use_hysteresis = 0;

  // Read per-cell arrays from PROPS/REGIONS/SOLUTION (PROPSARRAYS 1).
  //
  // Off by default because it changes more than it looks like it does: it is
  // what makes ENDSCALE's endpoints visible, but it also hands equilibration
  // the deck's own SWL and EQLNUM, and those set the initial saturations and
  // which EQUIL record each cell uses.  Turning it on is choosing a different
  // initial state, so it is a deliberate act rather than a side effect.
  int props_arrays = 0;

  // Endpoint scaling in the relative permeability (USEENDSCALE 1).
  //
  // Off by default, and deliberately so.  Reading the arrays and USING them
  // are separate decisions: on Norne the arrays also feed equilibration,
  // which changes the initial state on their own, so a run that changes both
  // at once cannot say which one moved the answer.  Turn this on to scale the
  // curves; leave it off to get per-cell connate water in the initial state
  // and nothing else.
  int use_endscale = 0;

  // Modified Appleyard chop in the Newton update (APPLEYARD 1): saturation
  // updates that cross a relperm endpoint are truncated at the endpoint, the
  // classic fully-implicit enabler for large timesteps on hard fields.
  int appleyard = 0;
  // Implicitness (deck keyword AIM 0|1, or IMPLICIT FIM|AIM).
  //
  // DEFAULT ON.  Adaptive implicit: a cell whose throughput CFL over the
  // coming substep is below AIMCFL runs explicit in saturation (its face
  // mobilities are frozen at t^n and contribute no saturation couplings),
  // while wells and every cell at or near a front stay fully implicit.  The
  // accumulation term is implicit everywhere, so each row stays well posed
  // and the Newton loop, the residual test and the CPR solve are unchanged.
  //
  // Measured on SPE10 model 2, 2000 days on one GB200: 402.6 s fully
  // implicit against 342.1 s adaptive, at a lower Newton count (1984 ->
  // 1789) and a lower Krylov count (203k -> 176k).  AIM 0 restores FIM.
  //
  // Black-oil only, and it needs the analytic Jacobian, since the residual
  // it forms is the one those kernels differentiate.  Both conditions fall
  // back to FIM in main.cu -- silently when AIM was merely the default,
  // with a warning when the deck asked for it (aim_explicit).
  int  aim = 1;
  int  aim_explicit = 0;
  Real aim_cfl = 0.5;
  Real perm_min = 5, perm_max = 500;   // [mD] mapping range for synthetic K
  Real poro_min = 0.05, poro_max = 0.30;

  // ---- PROPS (fluid) ----------------------------------------------------
  Real uw = 1.0, uo = 2.5, ug = 0.02;   // viscosities [cP]
  Real bw = 1.0, bo = 1.2, bg = 0.005;  // FVFs
  Real rhow = 1.0, rhoo = 0.8, rhog = 0.01;  // densities [g/cc]
  Real cfw = 1e-5, cfo = 1e-5, ct = 2e-5;    // compressibilities [1/psi]
  Real swi = 0.1, swr = 0.1;
  Real p_init = 3000, p_bub = 3000, p_atm = 14.6959;
  // Solution gas-oil ratio [SCF/STB].  Only consulted when pvt_const is set;
  // otherwise Rs comes from the Standing correlation in pvt.hpp.
  Real rs_const = 500.0;
  // VAPOIL: 1 (default) keeps PVTG's saturated Rv, 0 removes vaporised oil.
  int  vapoil = 1;
  // PVTCONST: hold B_o, B_g, R_s and mu_g fixed at the deck values instead of
  // re-evaluating the correlations at the field-mean pressure every report
  // step.  The correlations in pvt.hpp are a 1:1 port of the CuPy reference
  // and are not a dimensionally consistent field-unit property set (they give
  // B_g of order 1 RB/SCF and R_s of 5658 SCF/STB), so a deck that has to
  // describe the same fluid to another simulator cannot use them.  With this
  // set, the fluid is exactly the constants below and can be written into an
  // ECLIPSE PVT table verbatim.  See benchmark/make_decks.py.
  int  pvt_const = 0;

  // Read PVTO/PVTG/PVTW out of the field deck's PROPS section.  -1 = do it
  // whenever a field deck is given and carries them; 0 = never; 1 = require.
  //
  // This is deliberately NOT pvt_const.  The two questions look like one and
  // are not: pvt_const chooses between the deck's constants and the built-in
  // correlations, while this chooses whether tabulated PVT replaces either.
  // Conflating them once already cost a day.  Clearing pvt_const to enable the
  // tables also handed r.rs, r.bo, r.bg and r.ug to the correlations and let
  // refresh_pvt() rewrite them every step -- so the undersaturated anchor
  // became the correlation's 5650 SCF/STB instead of the deck's 674, the cap
  // in dev_rs_state() could never bind, every cell was pinned to the saturated
  // line, and the oil came out with a NEGATIVE compressibility of -7.1e-5/psi.
  // None of which is visible from the keyword's name.
  int  pvt_tables = -1;
  Real sw_init = 0.2, sg_init = 0.02, so_init = 0.78;
  int  gravity = 1;

  // Which well curves the report should draw (deck keyword WELLPLOT).
  //
  //   WELLPLOT ALL                 every quantity below
  //   WELLPLOT WOPR WWCT WGOR      only these
  //   WELLPLOT NONE                skip the section
  //
  // A field with thirty-six wells produces a lot of curves and most of a
  // report is usually not being read; asking for the two that matter is
  // faster to render and far faster to look at.  Empty means ALL, so a deck
  // that says nothing keeps the old behaviour.
  //
  // WOPR/WWPR/WGPR are written by the well post-process; WWCT and WGOR are
  // derived from them, so any of the five may be asked for.
  std::vector<std::string> well_plot;

  std::vector<SatRow> swof;   // water-oil table  (Sw, Krow, Krw)
  std::vector<SatRow> sgof;   // gas-oil table    (Sg, Krog, Krg)
  // Column four of each: the capillary pressures, parallel to the rows above.
  //
  // Kept beside the tables rather than widening SatRow, which every existing
  // three-element initialiser would otherwise have to grow.  Empty when the
  // deck gives three columns; all-zero when it gives four zeros, which is the
  // same thing physically and is treated as such.
  std::vector<Real> swof_pcow;   // oil-water Pc = p_o - p_w  [psi]
  std::vector<Real> sgof_pcog;   // gas-oil   Pc = p_g - p_o  [psi]

  // ---- Compositional (EOS) — only used when model == Compositional ------
  std::vector<std::string> comp_names;    // component names
  std::vector<Real> comp_z;     // overall feed mole fractions (ZMF)
  std::vector<Real> comp_Tc;    // critical T [R]           (TCRIT)
  std::vector<Real> comp_Pc;    // critical P [psia]        (PCRIT)
  std::vector<Real> comp_acf;   // acentric factors         (ACF)
  std::vector<Real> comp_vmL;   // liquid molar volume [ft^3/lbmol] (VMOL)
  Real temperature = 663.0;     // reservoir T [R] (TEMP; ≈ 200 F)
  Real z_factor    = 0.9;       // gas compressibility factor (ZFACTOR)

  // ---- Physical constants (deck-overridable; nothing hard-wired) --------
  // Unit system the RUN DECK is written in (keyword METRIC or FIELD).
  //
  // The solver works in field units, so a metric deck is converted once, on
  // load, by to_field_units() below.  `metric_input` records what it was
  // written in, for reporting; every value in this struct is in field units
  // by the time parse_deck() returns.
  int metric_input = 0;
  // Set once to_field_units() has run, so the banner can report the system
  // the deck was WRITTEN in.  Without it a converted deck is indistinguishable
  // from one that was field units all along, and a wrong METRIC keyword is
  // invisible.
  int was_metric = 0;

  Real darcy_const = 1.127e-3;  // field-unit transmissibility constant (DARCY)
  Real grav_grad   = 0.4335;    // psi/ft per (g/cc) fluid density (GRAVGRAD)
  Real re_factor   = 0.2;       // Peaceman equivalent-radius / dx (REFACTOR)
  Real fd_eps      = 1e-7;      // numerical-Jacobian FD step scale (FDEPS)
  // Relative cutoff applied to corner-point connection transmissibilities.
  // Zero preserves every positive connection (Flow-equivalent topology).
  // The historical 1e-6 default remains for tuned cases such as Norne;
  // high-contrast models can disable it with TRANSDROP 0.
  Real trans_drop_rel = 1e-6;

  // ---- SOLVER -----------------------------------------------------------
  // Primary solver is always CPR (constrained pressure residual).  `krylov`
  // selects the outer accelerator wrapped around the CPR preconditioner:
  //   0 = FGMRES (default),  1 = BiCGSTAB.
  int  krylov      = 0;
  Real lin_tol     = 1e-6;
  int  lin_maxit   = 100;
  int  gmres_restart = 30;
  // Classical Gram-Schmidt passes in FGMRES (ORTHOPASSES, 1 or 2). CGS2 is
  // the robust general default; a well-preconditioned large problem may use
  // CGS1 to remove the second projection, global reduction, and update.
  int  ortho_passes = 2;
  // CPRW includes the low-rank, eliminated well/control Jacobian in the
  // global pressure preconditioner (currently supported by AmgX).
  int  use_cprw = 0;
  // How many consecutive BoomerAMG setups may be skipped by reusing the
  // hierarchy built for an earlier Newton iterate (AMGREUSE; 0 rebuilds every
  // iteration, which is what every run before this keyword did).  A stale
  // hierarchy is abandoned early if it starts costing AMGREUSEGROWTH times the
  // Krylov iterations a fresh one needed.  See SolverConfig in
  // linear_solver.hpp for why this matters so much on multiple GPUs.
  int  amg_reuse = 4;
  // BoomerAMG coarse-grid cutoff (AMGMAXCOARSE); 0 = rank-aware default,
  // 64 on one rank and 2000 on several.  See linear_solver.hpp.
  int  amg_max_coarse = 0;
  // AMGPRINT 1 dumps HYPRE's hierarchy once, for comparing rank counts.
  int  amg_print_level = 0;
  // AMGCOARSERELAX 18 keeps the coarsest grid on the GPU; 9 is HYPRE's
  // host-side Gaussian elimination.  See linear_solver.hpp.
  int  amg_coarse_relax = 9;
  // PRECOND: 0 = hand-rolled CPR, 1 = HYPRE MGR.  See linear_solver.hpp.
  // LINSOLVER: AMGX (default), HYPRE or GINKGO.  See linear_solver.hpp
  // -- this selects the stage-1 multigrid only; the outer FP64 Krylov, the
  // operator and stage 2 are identical either way.  A binary built without
  // the requested library falls back and says so.
  int  backend = 1;    // 1 = AmgX, 0 = HYPRE, 2 = Ginkgo
  // GINKGOPREC 16 (default) or 32: the precision of the Ginkgo multigrid.
  int  ginkgo_precision = 16;
  int  use_mgr = 0;
  int  mgr_relax_type = 18;
  int  mgr_relax_sweeps = 2;
  int  mgr_interp_type = 2;
  int  mgr_global_smooth = 0;   // 0 = block Jacobi; 16 = ILU (not on GPU)
  int  mgr_global_sweeps = 1;
  // AMGFAST: turn on the multi-GPU coarse-level bundle in one keyword --
  // gather tiny coarse grids onto one rank, drop small entries from the
  // coarse operators, and keep interpolation sparse on the aggressive
  // levels.  Off by default: it changes the preconditioner, so it should be
  // measured rather than assumed.
  int  amg_fast = 0;
  int  amg_seq_threshold = 0;
  Real amg_nongalerkin_tol = 0.0;
  Real agg_trunc_factor = 0.0;
  int  agg_pmax = 0;
  int  agg_num_levels = 1;
  Real amg_reuse_growth = 1.3;
  int  mixed_precision = 1;   // 1 → FP32 AMG preconditioner (mixed), 0 → FP64
  // Default ON: the FP64 operator is evaluated in double-word (df64, Dekker /
  // QD error-free-transform) arithmetic on FP32 ALUs, while the CPR
  // preconditioner (AMG + stage-2) runs in FP32.  This is the default
  // mixed-precision CPR variant.  Set EMULATE 0 for a native-FP64 operator.
  // 0 native FP64, 1 df64 (default), 2 FP32-stored operator in the Krylov
  // matvec with the FP64 master used for every residual that decides
  // convergence (see SolverConfig::operator_mode).
  int  emulate_fp64 = 1;      // 1 → double-word df64 emulated FP64 operator
  // Operator storage for the Krylov SpMV.  1 applies the coupled Jacobian
  // block by block: one cell-level column index per bs x bs block instead of
  // one per scalar entry, four lanes per row (include/block_spmv.cuh).  0 is
  // the scalar CSR path (one thread per row for df64, cuSPARSE for FP64),
  // which reproduces earlier runs bit for bit.
  int  block_spmv = 1;
  // CUDA graphs for the part of each Arnoldi step after the AMG apply: the
  // second CPR stage, the Arnoldi SpMV and the Gram-Schmidt projections are
  // captured once per basis column and replayed.  On several ranks the tail's
  // halo exchanges and reductions go through NCCL, which is stream-ordered and
  // capturable, so the whole tail replays there too; a build without NCCL, or
  // NVRS_NCCL=0, keeps several ranks on MPI and uncaptured.  A capture that
  // fails turns graphs off for the run and says so.
  int  cuda_graph = 1;
  // CUDAGRAPHMINREUSE r: stop capturing once graphs are reused fewer than r
  // times per capture (after CUDAGRAPHGUARDAFTER captures).  0 = never stop.
  Real graph_min_reuse = 0;
  int  graph_guard_after = 192;
  // CPR stage-1 decoupling (CPRDECOUPLE), three-phase connection-list
  // assembly only:
  //   0 QUASI  quasi-IMPES: the first row of the inverse cell block (default,
  //            the published runs)
  //   1 DRS    dynamic row sum after Gries, Stueben et al. (SPE J. 2014):
  //            the pressure row is the sum of those component equations whose
  //            pressure row is diagonally dominant to within DRSEPS
  //            (a_ii >= DRSEPS * sum_j |a_ij|); quasi-IMPES where none is
  //   2 SUM    plain row sum of the component equations
  int  cpr_decouple = 0;
  Real drs_eps = 0.1;
  // DTSEQFILE: impose an accepted-substep sequence instead of the adaptive
  // controller.  One "t_start dt" pair (days) per line; lines starting with #
  // are comments.  Written by any run as dt_sequence.txt and by
  // benchmark/opm_timesteps.py from a Flow log.  A substep that fails to
  // converge is still chopped, and the chop is counted.
  std::string dt_seq_file;
  // DUMPOP step newton: write the Jacobian (CSR, FP64) and the CPR pressure
  // matrix A_p at that report step and Newton iteration to OUTDIR/op_*.bin,
  // for the kernel micro-benchmark and the condition estimate.  -1 = off.
  int  dump_op_step = -1;
  int  dump_op_newton = 0;
  // Jacobian assembly: 0 = numerical (finite-difference, all block sizes),
  //                    1 = analytic (exact derivatives; black-oil 2-phase).
  // Numerical scheme:
  //   1 = SEQUENTIAL — the CuPy reference formulation: elliptic pressure solve
  //       with total mobility and fixed-rate well sources, then an implicit
  //       fractional-flow saturation update.  This is what produced RESULTS/.
  //   0 = FIM — the fully-coupled block Newton in assemble_blackoil.cu.
  // Both use the same mixed-precision CPR solver.
  //
  // Defaults to FIM: that is the path with a completed 20-step run behind it.
  // SEQUENTIAL is opt-in until it has one too.
  int  scheme = 0;
  // JACOBIAN: 0 = NUMERICAL (finite differences, any block size), 1 =
  // ANALYTIC (exact derivatives, two- and three-phase), 2 = TENSORCORE (the
  // operator-product assembly; see tc_jacobian.hpp).
  int  jacobian = 0;

  // ---- TENSOR-CORE JACOBIAN ---------------------------------------------
  // TCVALIDATE 1: assemble the connection blocks both ways, analytic and
  // operator-product, and report the block-wise difference before the run
  // continues on the analytic path.  This is how a new factorisation is
  // checked against the assembly that is already trusted, rather than by
  // watching Newton misbehave later.
  int  tc_validate = 0;
  // TCPRECISION: 32 = TF32 Tensor Core multiply with FP32 accumulation
  // (needs SM80 or newer); 0 = FP32 on CUDA cores, which runs anywhere and
  // is the fallback when the device has no TF32 datapath.
  int  tc_precision = 32;

  // ---- SCHEDULE / TIME --------------------------------------------------
  Real dt_report  = 100;   // [days] report/output interval
  // DTMAX: upper bound on the *internal* timestep, independent of the
  // reporting interval.  0 disables it, which makes every report step a
  // single jump of dt_report.  That is stable -- a fully implicit scheme
  // is stable at any CFL -- but not accurate: on the benchmark deck a
  // 100-day step pushes ~32 cell pore volumes out of each injector, and
  // the saturation front becomes numerical diffusion.  The chop loop in
  // main.cu only shortens dt when Newton *fails*, and Newton converges
  // happily on a badly under-resolved step, so it is no substitute.
  Real dt_max     = 0;     // [days] 0 = no limit
  // DTGROW: consecutive first-attempt successes required before the step
  // length is doubled.  1 reproduces the original behaviour, which relaxed
  // after every success including ones that only converged because they had
  // just been chopped -- so dt doubled, overshot, chopped back, and doubled
  // again.  On BENCH3P that oscillation wasted an attempt on 11 of 20 steps.
  int  dt_grow    = 2;
  // DTRELAX: how many halvings the step may recover in one go, and DTTARGET:
  // the Newton count per sub-step that counts as comfortable.
  //
  // The descent is geometric and the ascent was not. One hard report step
  // drops the step length by six or eight halvings at once and holds it, while
  // recovery was one halving per DTGROW clean steps: from eight halvings down,
  // sixteen consecutive clean steps to get back. On Norne, with 85 chops
  // across 247 report steps, a clean run that long does not occur, and the run
  // spends its life six or seven halvings below the report step. The median
  // step there is 0.219 days against Flow's 7.2, and almost all of it is
  // inherited rather than required: only 85 steps chopped, the rest converged
  // first try at a length some earlier step imposed.
  //
  // With DTRELAX above 1, a step that converged in far fewer Newton iterations
  // than DTTARGET recovers proportionally, floor(log2(target/achieved))
  // halvings, capped at DTRELAX. A step that took two Newton iterations
  // against a target of eight is saying the length is four times shorter than
  // it needs to be, and that signal was being discarded.
  //
  // DTRELAX = 1 is the old behaviour exactly, and is the default, so no
  // existing result moves without being asked to.
  // DTKEEP: on a convergence failure, roll back to the start of the sub-step
  // that failed rather than to the start of the report step.
  //
  // Today a failure discards the whole report step and retries it at half the
  // length, so sub-steps that already converged are computed again. The cost
  // of an overshoot is therefore the entire step, which is why the timestep
  // cannot be allowed to grow quickly: measured on Norne, relaxing up to four
  // halvings made the next step chop 65% of the time against 28% at one
  // halving, and the extra chops ate the gain. Flow can afford to grow because
  // its mistakes cost one sub-step.
  //
  // With DTKEEP 1 the converged sub-steps are kept, the failed one is retried
  // at half the length, and only the remaining interval is re-planned. This
  // changes the sequence of states the simulator visits, so it is off by
  // default and every published number stands.
  int  dt_keep    = 0;
  int  dt_relax   = 1;
  int  dt_target  = 8;

  // How the step length is chosen at all.
  //
  //   0  the dyadic scheme above: dt_rep / 2^chop, a failure raises chop.
  //   1  OPM Flow's controller: a continuous dt carried across report steps,
  //      scaled after every converged sub-step by how many Newton iterations
  //      it took, and restarted at DTRESTART of its length when one fails.
  //
  // The lattice is the problem the second option exists to remove.  With 2^-k
  // lengths a failure can drop six levels at once while recovery climbs one
  // level per DTGROW clean steps, so the run settles far below the length it
  // could carry and never comes back.  A continuous dt has no levels to fall
  // through: the same information (Newton iterations used against the target)
  // moves it both ways, by the same rule, immediately.
  //
  // DTCONTROL 1 implies DTKEEP.  A controller that adjusts dt continuously is
  // meaningless if a failure throws away the interval it was controlling.
  //
  // Defaults below are Flow's own, so DTCONTROL 1 alone reproduces its scheme
  // rather than a variant of it.
  int  dt_control    = 0;
  Real dt_restart    = 0.33;  // SolverRestartFactor: failed sub-step retries here
  Real dt_max_grow   = 3.0;   // SolverMaxGrowth: cap on one step's growth
  Real dt_decay_damp = 1.0;   // TimeStepControlDecayDampingFactor
  Real dt_grow_damp  = 3.2;   // TimeStepControlGrowthDampingFactor
  int  dt_pid         = 0;     // also apply Flow's relative-change PID limiter
  Real dt_pid_tol     = 0.1;   // TimeStepControlTolerance
  // Optional cautious reopening of a learned failure ceiling. After this many
  // consecutive cheap accepted substeps, raise it by dt_cap_probe_factor.
  // Zero preserves the fixed ceiling. A failed probe restores the last proven
  // ceiling rather than ratcheting the run down.
  int  dt_cap_probe = 0;
  int  dt_cap_probe_iters = 2;
  Real dt_cap_probe_factor = 1.25;

  // A ceiling remembered from the last failure.
  //
  // The iteration-count rule on its own oscillates: a sub-step converging in
  // two Newton iterations against a target of eight is told to grow by 1.94,
  // so two successes carry dt from 0.32 to 1.27 days and straight back into
  // the 0.97 that had just failed.  Flow does not use that rule alone -- its
  // default takes the smaller of it and a PID controller on the relative
  // change in the solution, and near the stability limit the PID term
  // collapses and stops the growth before the failure.
  //
  // These two constants stand in for that term.  A failure drops the ceiling
  // to DTFAILCAP of the length that failed; each success relaxes it by
  // DTCAPGROW.  The controller still climbs, but has to earn its way back
  // rather than jumping over the cliff it just fell off.  DTFAILCAP 0 turns
  // the ceiling off and gives the unmodified iteration-count rule.
  // Convergence, OPM's way.
  //
  //   0  the strict test: cnv < CNVTOL for every phase, every cell, and the
  //      mb+cnv pair consulted only once Newton has stalled.
  //   1  OPM's BlackoilModel::getReservoirConvergence: a component passes
  //      when its worst cell is under CNVTOL, OR when the cells that violate
  //      hold less than CNVRELAXPV of the pore volume and the worst of them
  //      is still under CNVRELAXED.  Tested at every Newton iteration, as in
  //      Flow; the legacy relative-L2 exit is not used in this mode.
  //
  // This is what separates a 7-day step from a 0.9-day one.  A single cell
  // beside a gas injector is over tolerance at any step length worth taking,
  // and the strict test throws away the whole step for it while Flow carries
  // on.  No timestep controller can recover a length the convergence test
  // refuses, which is why tuning the controller alone did nothing.
  int  opm_conv     = 0;
  Real cnv_relax_pv = 0.03;   // RelaxedMaxPvFraction
  Real cnv_relaxed  = 1.0;    // OPM ToleranceCnvRelaxed default

  // Give up on a Newton solve that has stopped descending.
  //
  // A failed sub-step costs the full NEWTONMAX, and on Norne that is sixty
  // iterations spent watching mb sit at 1.3 times its tolerance while the
  // relative residual stays around 1e-1.  The step is lost by iteration three
  // or four and the rest is pure waste: 40,020 discarded Newton iterations
  // over a run, about as much work as the whole baseline performs.
  //
  // This many consecutive iterations that fail to reduce the residual by
  // NEWTONSTALL, with the convergence tests still unmet, ends the attempt. The
  // timestep controller then retries shorter, which is what was going to
  // happen anyway, just fifty iterations later.  0 disables it.
  int  newton_stall_max = 0;

  Real dt_fail_cap = 0.75;
  Real dt_cap_grow = 1.2;
  Real t_end      = 2000;  // [days] active simulation period
  int  newton_substeps = 10;
  int  newton_max      = 20;
  // Material-balance tolerance, adapted from OPM Flow's MB criterion.
  //
  // newton_tol alone is a RATIO to the residual this step happened to start
  // with, and that is a moving target: a sub-step that starts close to its
  // answer has a small R0, so 1e-3 of it is a very small absolute residual,
  // and eventually Newton is asked to chase round-off.  The first sub-step of
  // a chopped step converges in four iterations and a later one cannot
  // converge at all, which looks like the physics getting harder and is
  // really the target getting harder.
  //
  // |sum R|.dt/(pore volume . b) is the field material-balance error used by
  // Flow.  It means the same thing on every step.  In OPMCONV mode it is a
  // required criterion; the relative L2 ratio must never bypass it.  0
  // disables the MB criterion only for the legacy convergence mode.
  Real mb_tol          = 1e-6;
  // Flow's ToleranceMbRelaxed: on the last Newton iteration (NEWTONMAX) Flow
  // accepts MB below this and CNV below CNVRELAXED in every cell.  Without it
  // a strict MBTOL of 1e-7 is tighter than the reference it is meant to
  // match.  0 (default) keeps the published behaviour: no final relaxation.
  Real mb_tol_relaxed  = 0;
  // CNV: the worst single cell, as a fraction of its own contents.  MB is a
  // sum over the field, so one badly wrong cell can hide behind every cell
  // that is right -- exactly the cell a gas injector is filling.  Flow tests
  // both and so does this in OPMCONV mode. Flow's default is 1e-2 and this
  // matches it.
  Real cnv_tol         = 1e-2;
  // How much an iteration must still be buying before the mb+cnv exit is
  // allowed to end the step.  0.5 means "while each iteration is at least
  // halving the residual, keep going".  Without it Newton stops the instant
  // material balance is satisfied, which on Norne is around 1e-2 of the
  // starting residual while the very next iteration would have reached 1e-4.
  // 1.0 restores the old behaviour; 0 disables the mb+cnv exit entirely and
  // leaves only the relative test.
  Real newton_stall    = 0.5;
  Real newton_tol      = 5e-3;
  // Maximum alpha halvings used to find a residual-reducing fully implicit
  // Newton update.  Zero preserves the undamped update.  This is separate
  // from the sequential solver's saturation line search.
  int  line_search     = 0;
  // When no line-search trial strictly descends, retain the least-bad trial
  // rather than blindly taking the smallest alpha. Opt-in for A/B testing.
  int  line_best       = 0;
  // Maximum adaptive inexact-Newton linear tolerance. Zero disables forcing;
  // otherwise the tolerance varies between LINTOL and this value.
  Real lin_forcing_max = 0;
  // Newton safeguarding, mirroring the CuPy reference's line search and
  // timestep chopping.  Without these a fully-implicit step takes the raw
  // Newton update however large it is, and a strongly nonlinear step diverges.
  Real dp_max   = 500.0;   // absolute max |dp| [psi], if dp_max_rel <= 0
  Real dp_max_rel = 0.0;   // relative max |dp|/p; Flow default is 0.3
  Real ds_max   = 0.2;     // max |dS| per cell per Newton iteration [-]
  int  max_chop = 6;       // halvings of dt before a report step is abandoned

  // ---- SUMMARY ----------------------------------------------------------
  int make_csv = 1, make_vtk = 1, make_summary = 1, make_gassmann = 1;
  int autoplot = 1;   // 0 → skip the end-of-run Python plotting (HM inner loop)
  // ECLIPSE / OPM Flow compatible output (EGRID, INIT, UNRST, SMSPEC, UNSMRY,
  // RSM).  ECLOUT 0 disables it; CASENAME sets the ECLIPSE base name.
  int ecl_out = 1;
  std::string case_name = "NVRS";

  std::vector<WellSpec> wells;

  int n_cells() const { return nx * ny * nz; }
  int n_prod()  const;
  int n_inj()   const;
};

// Parse a deck file.  Throws std::runtime_error with a helpful message on any
// malformed keyword so the executable fails fast with a clear diagnostic.
Deck parse_deck(const std::string& path);

// Convert a run deck written in METRIC into field units, in place.
//
// Called by parse_deck() when the deck says METRIC; idempotent, and a no-op
// on a deck that is already in field units.  Exposed for the deck tests.
void to_field_units(Deck& d);

// Echo the resolved deck to stdout (used at start-up for provenance).
void print_deck_summary(const Deck& d);

}  // namespace nvrs
