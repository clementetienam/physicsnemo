// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// physics.hpp — reservoir state + the physics/assembly interface.  Two families
// share one Newton–CPR driver:
//   * BlackOil (2- or 3-phase)   — assemble_blackoil.cu
//   * Compositional (nc-comp EOS)— assemble_compositional.cu
// Both expose the same contract: given the current primary-variable field they
// fill a DeviceLinearSystem (coupled CSR Jacobian J, rhs = -R, pressure block,
// block-diagonal inverse) for the CPR solver, then apply a Newton update.
#pragma once

#include "common.hpp"
#include "deck.hpp"
#include "fluid.hpp"
#include "grid.hpp"
#include "linear_solver.hpp"
#include "pvt_tables.hpp"

namespace nvrs {

// Rock + fluid static fields (device), plus the evolving primary variables.
struct Reservoir {
  Grid grid;
  Model model;
  // Monotone identity of the accepted reservoir state.  Host-side controls
  // use it to cache quantities which must remain fixed throughout one Newton
  // solve but be refreshed before the next internal timestep.
  long long accepted_state_id = 0;
  // Deck-fixed PVT: refresh_pvt is a no-op, so B_o, B_g, R_s and mu_g keep the
  // values make_reservoir copied out of the deck.
  int  pvt_const = 0;
  int  bs = 2;                // unknowns per cell
  int  ncomp = 3;

  // Static rock (device), length n_local cells.
  Real* perm = nullptr;       // Kx [mD]  (Ky=Kx)
  Real* poro = nullptr;       // porosity [-]
  // Per-cell Kz [mD], only when the deck gives PERMZFILE; otherwise null and
  // Kz = kz_mult * Kx.  Consumed by the face-transmissibility builder alone,
  // so no kernel reads it.
  Real* permz = nullptr;
  Real  kz_mult = 0.2;

  // Primary variables u (device), length bs * cells, cell-major:
  //   black oil:  [p, Sw]      or [p, Sw, Sg]
  //   compos.  :  [p, N_1..N_nc]  (overall molar densities)
  Real* u = nullptr;
  Real* u_old = nullptr;      // previous timestep (for accumulation)
  // Meaning of black-oil u[2], stored as Real so the existing scalar halo can
  // exchange it: 0 = Sg, 1 = Rs, 2 = Rv. The old array belongs to u_old. Null keeps
  // the legacy fixed-[p,Sw,Sg] formulation bit-for-bit unchanged.
  Real* gas_primary = nullptr;
  Real* gas_primary_old = nullptr;
  int pv_switch = 0;

  // DIAGNOSTIC.  Per cell, the oil-saturation change caused by CLAMPING in the
  // Newton update -- the Appleyard endpoint truncation, the per-iteration
  // ds_max clip, and the hard [swi, 1] and [0, 1 - Sw] bounds -- accumulated
  // over every Newton iteration of the run.
  //
  // Sw and Sg are the unknowns and So = 1 - Sw - Sg is the DEPENDENT phase, so
  // every one of those clamps lands in full on oil with no matching well or
  // flux term.  Weighted by pore volume and shrinkage on the host (see
  // field_oil_in_place), this is the stock-tank oil the update creates or
  // destroys outside the mass balance -- which is the quantity the [oil]
  // balance line reports missing.  Negative means oil destroyed.
  Real* clamp_dso = nullptr;

  // Well source arrays (device), length cells: signed surface rate per cell.
  Real* q_water = nullptr;
  Real* q_oil   = nullptr;
  Real* q_gas   = nullptr;

  // --- assembly topology (device), built once by build_sparsity ---
  int*  nbr     = nullptr;   // cells*7  (self + 6 face neighbours; -1 = none,
                             //           >= n_local() = ghost cell)
  int*  bcol    = nullptr;   // cells*7  scalar column offset of block in its row
  int*  pcolpos = nullptr;   // cells*7  pressure-block column position
  Real* tface   = nullptr;   // cells*6  face transmissibility (harmonic K·A/L)
  // Global K index of every local cell, interior *and* ghost (length
  // n_local_ext()).  Cell-centre depth must be measured from the global datum,
  // otherwise each rank would apply gravity from its own slab origin and the
  // answer would depend on the rank count.
  int*  cellk   = nullptr;

  // --- general connection list ------------------------------------------
  // The arrays above describe a 7-point stencil: six faces at fixed slots,
  // one transmissibility each.  That is true of a Cartesian box and false of
  // every field model.  Corner-point geometry with faults gives a cell any
  // number of neighbours -- on Norne the degree runs from 1 to 17 with a mean
  // of 5.81 -- because a throw juxtaposes one cell against parts of several.
  //
  // So the connectivity is also published as a CSR adjacency, which is the
  // form that covers both.  For a structured grid it holds exactly the same
  // six neighbours and the two agree entry for entry; for a corner-point grid
  // it holds whatever the geometry produced.  Assembly kernels that iterate
  // this instead of the fixed slots work unchanged on either.
  //
  // Gravity is the one place the fixed stencil hid an assumption: it knew a
  // face was vertical because its index was >= 4.  A connection carries the
  // depth difference between its two cells instead, which is the quantity
  // that was actually wanted and is well defined for a fault connection that
  // is neither vertical nor horizontal.
  int*  conn_ptr   = nullptr;  // cells+1   CSR offsets into the arrays below
  int*  conn_nbr   = nullptr;  // nconn     local index of the other cell
  Real* conn_trans = nullptr;  // nconn     transmissibility
  Real* conn_dz    = nullptr;  // nconn     depth(neighbour) - depth(cell)
  Real* conn_threshold = nullptr; // nconn  phase-potential dead band
  int*  conn_bcol  = nullptr;  // nconn     scalar column offset of that block
  int*  conn_pcol  = nullptr;  // nconn     pressure-block column position
  int*  self_bcol  = nullptr;  // cells     column offset of the self block
  int*  self_pcol  = nullptr;  // cells     pressure column of the self block
  int   nconn      = 0;        // total connection entries (each pair twice)
  int   max_degree = 0;        // largest number of neighbours of any cell

  // Accumulation volume per cell, in the same units as the scalar the uniform
  // path uses.  The kernels multiply this by `poro`.  For a field model it is
  // therefore PORV/(PORO*NTG), not necessarily geometric bulk volume: this
  // preserves deck MULTPV/MINPV while r.poro remains the physical property
  // written to INIT.  Null on a uniform grid, where one scalar says so.
  Real* cellvol = nullptr;
  // The unstructured halo, when the grid came from a field deck.  Null on a
  // structured grid, where the slab exchange in halo.cu applies instead.
  void* halo_conn = nullptr;
  // Cell centre depth, for wells and for reporting.  Null on a uniform grid,
  // where depth follows from the k index.
  Real* celldepth = nullptr;
  // Select the connection-list assembly.  Forced on for a grid that is not a
  // 7-point stencil, since the fixed-slot kernels cannot represent one;
  // optional on a structured grid, where the two paths agree and running both
  // is how that agreement is checked.  Deck keyword CONNLIST.
  int   use_conn_list = 0;

  void* wellcells = nullptr; // device array of per-cell well parameters (opaque)
  void* wellcells_host = nullptr; // canonical host mirror (owned)
  int* well_update_cells = nullptr; // compact completion scatter indices
  void* well_update_values = nullptr; // compact WellCell staging (device)
  int well_update_capacity = 0;
  int* well_state_cells = nullptr; // completion indices for compact state gather
  Real* well_state_values = nullptr; // compact gathered primary variables
  int well_state_capacity = 0;
  void* assembly_pvt_cache = nullptr; // current per-cell assembly properties
  int assembly_pvt_capacity = 0;
  int assembly_pvt_ready = 0; // one-shot: controls prepared cache for assembly
  double* well_region_partials = nullptr; // deterministic GPU regional sums
  int well_region_capacity = 0; // number of partial blocks allocated
  Real* assembly_residual_stats = nullptr; // persistent 23-value reduction
  unsigned long long* assembly_cnv_stats = nullptr; // persistent 3 maxima
  Real* timestep_change_stats = nullptr; // persistent [numerator,denominator]
  void* well_schur = nullptr; // device WellSchurEntry list (field models only)
  int well_schur_count = 0, well_schur_capacity = 0, well_schur_groups = 0;
  int well_schur_enabled = 0;

  // Rel-perm tables (device) + host mirror kept for well post-processing.
  SatTables tables;

  // PVT / fluid scalars (copied from the deck; PVT recomputed per step).
  Real uw, uo, ug, bw, bo, bg, rs;
  Real swi, swr, cfo, p_bub, p_atm;
  // Total compressibility [1/psi], rock plus fluid (deck CT).  With
  // constant formation volume factors this is the ONLY pressure
  // dependence in the accumulation term, and therefore the only thing
  // that keeps the pressure operator non-singular under depletion.
  Real ct = 2e-5;
  // A tabulated-PVT deck with ROCK uses an absolute pore-volume multiplier
  // about this reference pressure.  Without ROCK, ct retains its legacy role
  // as a timestep-local total-compressibility fallback for PVTCONST models.
  int  have_rock = 0;
  Real rock_pref = 0;

  // Tabulated PVT, when the deck supplies PVTO/PVTG/PVTW.  Where these are
  // valid the scalars above are ignored, ct reverts to meaning rock
  // compressibility alone, and the fluid's own compressibility comes from the
  // tables' pressure derivatives -- which for gas is an order of magnitude
  // larger than the constant it replaces, and varies by a factor of eight
  // across the pressure range of a single run.
  DevicePvt pvt;
  // Per-region tables and the region each cell belongs to, from PVTNUM.
  // Null falls back to `pvt` above, which is region 1 for every cell.
  DevicePvt* pvt_regions = nullptr;   // device array, n_pvt_regions entries
  int* pvt_cell = nullptr;            // device, one region index per cell
  int n_pvt_regions = 0;
  // The dissolved gas an oil retains where there is no free gas to saturate
  // it.  Rs is not a primary unknown here, so this is the undersaturated
  // branch's anchor; the deck's RS keyword sets it.
  Real rs_max = 0;
  // Per-cell undersaturated Rs anchor from RSVD, on the device, or null to use
  // the scalar above.  A deck that states RSVD means the anchor VARIES with
  // depth and equilibration region -- Norne's five tables span 675 to 531
  // scf/stb -- and one number for the whole field puts the crest value
  // everywhere.
  Real* rs_max_cell = nullptr;

  // Dissolved-gas composition history.  `rs_cap` is the largest dissolved gas
  // each cell is allowed to hold RIGHT NOW.  Oil-only cells start at their
  // RSVD charge; cells containing free gas start at Rs_sat.  It is walked
  // after every accepted step, never up by more than drsdt per day.
  //
  // Without it Rs is a pure function of pressure -- min(Rs_sat(p), anchor) --
  // so gas that came out of solution when a cell depleted goes straight back
  // in when the cell repressurises.  VAPPARS cancels Norne's earlier DRSDT 0,
  // but the compact three-primary-variable formulation still needs this
  // conservative history: changing Rs upward outside the coupled nonlinear
  // solve would create dissolved-gas component inventory.
  //
  // Null when neither DRSDT nor VAPPARS requires composition history.
  Real* rs_cap = nullptr;
  Real drsdt = -1;                 // < 0: not stated, no limit

  // VAPPARS: historical maximum oil saturation and its Rv depletion exponent.
  // The schedule makes this mutually exclusive with DRSDT/DRVDT.
  Real* so_max = nullptr;
  Real vap1 = 0, vap2 = 0;

  // EHYSTR.  `sg_hy` is the largest Sg and `sw_ow_hy` the smallest Sw+Sg
  // each cell has reached at an accepted step. Carlson's gas/oil and
  // oil/water scanning curves read them in relperm3 (fluid.hpp),
  // and it is walked forward the same way rs_cap is: once per accepted step,
  // never inside the Newton loop, because it is a function of the converged
  // state and moving it between iterations would make the residual chase a
  // target that shifts under it.  Null when the deck states no hysteresis.
  Real* sg_hy = nullptr;
  Real* sw_ow_hy = nullptr;
  int use_hysteresis = 0;

  // Assemblies still to report on.  The driver opens this when a step has
  // already been chopped, so the quiet first attempt stays quiet and the
  // retry -- the one that is actually in trouble -- says what it sees.  A
  // budget spent on step 1 tells you nothing about the step that failed.
  int diag_budget = 0;

  // The last assembly's residual, split by equation: oil, water, gas.
  //
  // OPM Flow does not test one norm over all three -- it normalises each
  // equation by the fluid in place and requires every one of them to pass
  // (its CNV and MB criteria).  This simulator tests a single L2 norm, which
  // on Norne is 9621 times larger for gas than for oil, so what it is really
  // testing is the gas equation and the other two ride along.  Keeping the
  // split lets the trace show whether they are actually converging.
  // The real MPI geometry, which `grid` does NOT carry on a field run.
  // There, grid is this rank's cell count presented as a 1-D run with rank 0
  // of 1 -- deliberately, since an unstructured model has no i,j,k extents to
  // decompose.  But every collective in the assembly was guarded by
  // grid.nranks, so on a field deck with four ranks the Newton residual was
  // never reduced: each rank tested convergence against its own local norm,
  // the ranks left the Newton loop at different iterations, and whichever
  // left first stranded the others in the next collective.  Guard on these
  // instead; they are the truth on both paths.
  int mpi_rank = 0, mpi_nranks = 1;

  Real res_eq[3] = {0, 0, 0};
  // Signed global residual sums from the last assembly [surface units/day].
  // Multiplying a converged substep's value by dt gives its exact discrete
  // component-balance defect; internal connection fluxes cancel globally.
  Real res_sum[3] = {0, 0, 0};
  // Surface well rates reduced during the last assembly.  Accepted substeps
  // integrate these without copying the full reservoir state back to host.
  Real well_prod[3] = {0, 0, 0};
  Real well_inj[3] = {0, 0, 0};
  // Cell blocks the stage-2 inverse could not form, and injector cells that
  // hit their bottom-hole limit, from the last assembly.  The first is the
  // one to watch where a phase is vanishing: when So reaches zero the oil
  // row loses its pressure entry, and the block CPR inverts for its weights
  // can go singular in exactly the cells a gas injector is filling.
  int n_singular = 0, n_inj_bhp = 0;
  // Pore volume owned by this rank [rb], summed during the assembly.  It is
  // the scale a residual has to be judged against: R is a surface volume per
  // day, so R.dt/(pore volume . b) is the fraction of the cell's contents
  // that the equation is out by -- a number that means the same thing on
  // every step, unlike a ratio to the residual this particular step started
  // with.
  Real pv_local = 0;
  // CNV per equation: the worst single cell, as a fraction of its own
  // contents.  A material balance is a sum over the field and lets one badly
  // wrong cell hide behind every cell that is right, which is why OPM tests
  // both and why converging on MB alone is not the same as converging.
  Real cnv[3] = {0, 0, 0};
  // Pore volume [rb] held by the cells that violate the strict CNV tolerance,
  // per phase (diagnostic) and for the union over all phases (acceptance).
  // OPM classifies each cell by its worst component and applies the relaxed
  // tolerance only when that union occupies less than RelaxedMaxPvFraction.
  Real cnv_pv[3] = {0, 0, 0};
  Real cnv_bad_pv = 0;
  // The strict tolerance the kernel measures cnv_pv against.  Set from the
  // deck at start-up; zero disables the accumulation.
  Real cnv_tol = 0;
  // CPR stage-1 decoupling (deck CPRDECOUPLE / DRSEPS): 0 quasi-IMPES,
  // 1 dynamic row sum, 2 plain row sum.
  int  cpr_decouple = 0;
  Real drs_eps = 0.1;

  // Modified Appleyard chop in the Newton update (deck keyword APPLEYARD 1):
  // truncate a saturation update that crosses a relperm endpoint to land just
  // past the endpoint, instead of letting one iteration leap mobility regimes.
  int appleyard = 0;
  // Adaptive implicit (deck keyword AIM 1; default FIM).  aimflag is a
  // cells_ext device array: 1 = fully implicit, 0 = explicit-in-saturation
  // for the coming substep, chosen by the throughput-CFL rule against
  // aim_cfl (AIMCFL).  aim_expfrac is the last measured explicit fraction.
  Real* aimflag = nullptr;
  // Per-cell throughput [pore volumes/day], measured from a CONVERGED state;
  // the flags are dt * aim_rate < aim_cfl.  aim_have_rate is 0 until the
  // first converged substep, and the run is fully implicit until then.
  Real* aim_rate = nullptr;
  int  aim = 0;
  int  aim_have_rate = 0;
  Real aim_cfl = 0.5;
  Real aim_expfrac = 0;
  Real rhow, rhoo, rhog;
  int  gravity = 1;
  Real dt = 1.0;
  Real dp_max = 500.0, dp_max_rel = 0.0, ds_max = 0.2;
  int  jacobian = 0;          // 0 = numerical FD, 1 = analytic

  // Physical constants (from the deck — nothing hard-wired in kernels).
  Real darcy_const = 1.127e-3;
  Real grav_grad   = 0.4335;
  Real re_factor   = 0.2;
  Real fd_eps      = 1e-7;
  Real temperature = 663.0;   // [R]
  Real z_factor    = 0.9;

  // Compositional EOS component properties (device arrays, length ncomp).
  Real* comp_Tc  = nullptr;   // [R]
  Real* comp_Pc  = nullptr;   // [psia]
  Real* comp_acf = nullptr;   // acentric factor
  Real* comp_vmL = nullptr;   // liquid molar volume [ft^3/lbmol]
  // Injected composition (deck ZMF), host-side, length ncomp.
  Real comp_zinj[kMaxComponents] = {};
};

// Allocate device fields, upload rock + tables, set initial conditions.
Reservoir make_reservoir(const Deck& d, const Grid& g);
void free_reservoir(Reservoir& r);

// Build the fixed block-sparsity (7-point stencil) once; sizes the CSR arrays
// inside `sys` and fills rowptr/colind for both the coupled and pressure CSRs.
void build_sparsity(const Reservoir& r, DeviceLinearSystem& sys);

// Fill J and rhs = -R for the current state (black oil).  Returns ||R||.
Real assemble_blackoil(Reservoir& r, DeviceLinearSystem& sys);
// Evaluate the RESV-control regional state on the GPU while preparing the
// per-cell property cache consumed by the immediately following assembly.
// Returns false for configurations which do not use that cache.
bool blackoil_region_average(Reservoir& r, double out[8]);
// OPM TimeStepControl's squared relative change in the primary variables,
// between the beginning and converged end of one internal timestep.
Real blackoil_relative_change(Reservoir& r, const Real* previous);
// Convert initially gas-free cells from Sg to Rs after the device PVT tables
// have been uploaded, and exchange their primary-variable meanings.
void blackoil_pvswitch_initialize(Reservoir& r);
// Apply the OPM Sg<->Rs appearance/disappearance test after a Newton update.
void blackoil_pvswitch_adapt(Reservoir& r);

// Report what the CPR pressure operator actually looks like.
//
// The AMG in stage one is built for an M-matrix: positive diagonal, negative
// off-diagonals, and every row at least weakly diagonally dominant.  Nothing
// enforces that -- it is a property the discretisation happens to have, and it
// is lost quietly.  When it goes, the symptom is not a wrong answer but a
// Krylov method that stalls at a fixed relative residual however many
// iterations it is given, which is indistinguishable from a dozen other
// causes if all you have is the stall.
//
// So measure it.  This reads the assembled pressure matrix back and counts the
// rows that violate each property, which turns "the linear solver is
// struggling" into a number that either is or is not the problem.  Called once
// per run by default; NVRS_CPR_DIAG=n reports the first n calls.
void report_pressure_operator(const DeviceLinearSystem& sys, const char* when);

// Fill J and rhs = -R for the current state (compositional).  Returns ||R||.
Real assemble_compositional(Reservoir& r, DeviceLinearSystem& sys);

// Apply a Newton update  u += damp * du  with saturation/volume safeguards.
void newton_update(Reservoir& r, const Real* du, Real damp);

// AIM (no-ops unless the deck set AIM 1; compositional stays implicit).
//   aim_measure_rates  throughput per pore volume from a CONVERGED state
//   aim_update_flags   flags for the coming substep at its dt; verbose
//                      prints the explicit fraction from rank 0
//   aim_reset_implicit back to fully implicit (run start, timestep chop)
void aim_measure_rates(Reservoir& r);
void aim_update_flags(Reservoir& r, int verbose);
void aim_reset_implicit(Reservoir& r);

// Recompute pressure-dependent PVT (BO, BG, RS) from the mean pressure — the
// sequential PVT refresh used by the Python reference between timesteps.
// DRSDT: walk each cell's dissolved-gas ceiling after an ACCEPTED timestep.
//
// Must be called once per accepted step and never inside the Newton loop --
// the ceiling is a function of the step's converged state, and moving it
// between iterations would make the residual chase a target that shifts under
// it.  A no-op when the deck states no DRSDT.
void drsdt_initialize(Reservoir& r);
void drsdt_update(Reservoir& r, Real dt);
// VAPPARS: advance historical maximum So after an accepted timestep.
void vappars_update(Reservoir& r);

// EHYSTR: walk each cell's historical-high Sg after an accepted step.  Same
// rule as drsdt_update -- once per accepted step, never inside the Newton
// loop.  A no-op when the deck states no hysteresis.
void hyst_update(Reservoir& r);

void refresh_pvt(Reservoir& r);

// Split the FP64 coupled operator into df64 (hi,lo) FP32 slices for the
// double-word emulated-FP64 SpMV.  Call after every assemble when EMULATE is on.
void split_operator_df64(DeviceLinearSystem& sys);

}  // namespace nvrs
