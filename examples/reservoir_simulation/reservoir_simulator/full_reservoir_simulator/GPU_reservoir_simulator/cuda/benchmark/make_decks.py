#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# make_decks.py — generate the NVRS deck and the ECLIPSE deck for a
# head-to-head benchmark against OPM Flow, from a single parameter set.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# ---------------------------------------------------------------------------
# Why generate both decks from one place
# ---------------------------------------------------------------------------
# NVRS reads its own compact deck format; Flow reads ECLIPSE.  Maintaining the
# two by hand guarantees they drift, and a benchmark whose two inputs disagree
# measures nothing.  Everything physical is declared once in CASES below and
# emitted twice.
#
# ---------------------------------------------------------------------------
# Making the two simulators solve the SAME equations
# ---------------------------------------------------------------------------
# NVRS's residual (src/assemble_blackoil.cu) is
#
#     V phi / dt * (b_p S_p - b_p S_p^n)  -  sum_faces b_p T lambda_p dPhi_p
#         - b_p WI lambda_p (p - p_wf)  =  0,        b_p = 1 / B_p
#
# with phi and B_p constant.  Matching that in ECLIPSE requires five things,
# each of which is done explicitly below:
#
#   1. CONSTANT PVT.  NVRS normally re-evaluates B_o, B_g, R_s and mu_g once a
#      report step from the correlations in include/pvt.hpp.  Those are a 1:1
#      port of the CuPy reference and are not a dimensionally consistent
#      field-unit property set (B_g of order 1 RB/SCF, R_s of 5658 SCF/STB), so
#      they cannot be written into an ECLIPSE PVT table and mean the same
#      thing.  The NVRS deck therefore sets PVTCONST 1, which pins the fluid to
#      the deck constants; the ECLIPSE deck holds exactly those constants.
#
#   2. ZERO FLUID COMPRESSIBILITY.  NVRS's accumulation is V*phi/dt with phi
#      fixed and b_p fixed, i.e. the system is incompressible.  PVTW and PVCDO
#      therefore carry zero compressibility, which keeps B_o and B_w exactly
#      constant and every flux term identical.
#
#      ROCK is the one exception, and it is deliberate.  Flow's default CPR
#      preconditioner extracts its pressure equation using weights built from
#      d(accumulation)/dp.  With every compressibility at zero that derivative
#      is identically zero, the pressure matrix has no strong connections, DUNE
#      reports "Could not build any aggregates. Probably no connected nodes",
#      and the coarse direct solve then fails with "matrix is singular" and the
#      run dies at time step 0.  A small ordinary sandstone value (4e-6 /psi)
#      restores a well-defined pressure equation.  Its physical effect is a
#      pore volume varying by c_r * dp ~ 8e-4 over the run -- three orders of
#      magnitude below the differences the benchmark reports.
#
#   3. NO DISSOLVED GAS.  NVRS's gas equation carries an R_s * b_o * S_o term.
#      Setting RS 0 removes it, and the ECLIPSE deck omits DISGAS, so both are
#      immiscible three-phase.
#
#   4. MATCHED GRAVITY.  NVRS uses the *surface* density as the reservoir
#      density in the gravity head (rho * GRAVGRAD, GRAVGRAD = 0.4335 psi/ft
#      per g/cc); ECLIPSE uses rho_surface / B.  The densities below are
#      therefore chosen so that rho_surface / B is the same for all three
#      phases and equal to what NVRS uses.  Consequences: the hydrostatic
#      gradient is identical in the two runs, and there is no buoyancy for the
#      two to disagree about.  0.4335 psi/ft per g/cc and 1/144 psi/ft per
#      lb/ft3 are the same constant (62.428 / 144 = 0.4335).
#
#   5. IDENTICAL WELL INDEX.  NVRS uses a fixed equivalent radius
#      re = REFACTOR * dx rather than the anisotropic Peaceman formula Flow
#      would compute for itself.  The connection transmissibility factor is
#      therefore written explicitly into COMPDAT item 8 (which ECLIPSE defines
#      as exactly NVRS's WI), so no well-model difference remains.
#
# ---------------------------------------------------------------------------
# The one difference that cannot be removed
# ---------------------------------------------------------------------------
# For THREE-PHASE flow the oil relative permeability conventions differ:
#
#     NVRS      kro = (So / (1 - Swi - Swr)) * krow(Sw) * krog(Sg)   [Baker]
#     ECLIPSE   kro = (Sg*krog + (Sw - Swco)*krow) / (Sg + Sw - Swco)
#
# Flow implements the ECLIPSE default plus STONE1/STONE2; it has no Baker
# option, and NVRS has no Stone option, so a mobile-gas run cannot be matched
# exactly.  Rather than hide that, the benchmark ships two cases:
#
#   BENCH2P  two-phase oil/water.  NVRS's two-phase kernel uses ANALYTIC Corey
#            curves (include/fluid.hpp relperm2: krw = S^2, kro = (1-S)^2 with
#            S = (Sw - Swi)/(1 - Swi - Swr)) and ignores SWOF entirely, so the
#            ECLIPSE SWOF below tabulates those exact curves.  Every term in
#            the two residuals is then identical and any difference in the
#            answer is discretisation, linear solver and time stepping alone.
#            This is the case that measures NVRS.
#
#   BENCH3P  full three-phase with mobile gas, tabulated SWOF/SGOF shared
#            verbatim.  Everything matches except the kro convention above.
#            This is the case that exercises the three-phase code path.
#
# ---------------------------------------------------------------------------
#   python3 make_decks.py [outdir]        # default: alongside this script
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA = os.path.abspath(os.path.join(HERE, ".."))
# data/ lives beside cuda/, not inside it; PERMFILE in the decks is
# "../data/..." relative to the cuda/ working directory.
SIM = os.path.abspath(os.path.join(CUDA, ".."))

# ===========================================================================
# The single source of physical truth
# ===========================================================================
SHARED = dict(
    # --- grid ---------------------------------------------------------------
    # The permeability file is a 40 x 40 x 3 field covering a 2000 x 2000 x 60
    # ft reservoir.  A case may ask for more cells: the reservoir keeps the
    # same physical extent and the same rock, and the cells get smaller.  A
    # larger case is therefore the SAME problem at higher resolution -- same
    # pore volume, same wells in the same physical spots, same injected
    # fraction -- so its answer stays comparable with the base case and the
    # only thing that grows is the linear system.  That is what makes it a
    # meaningful performance comparison rather than a different simulation.
    base_dims=(40, 40, 3),              # shape of the permeability file
    extent=(2000.0, 2000.0, 60.0),      # [ft] 40*50, 40*50, 3*20
    nx=40, ny=40, nz=3,                 # per-case; dx, dy, dz follow from these
    tops=8000.0,                        # [ft] depth of the top face of layer 1
    kz_mult=0.2,                        # Kz = kz_mult * Kx
    permfile=os.path.join(SIM, "data", "iglesias3.out"),
    poro_min=0.05, poro_max=0.30,

    # --- fluid (constant; see note 1 and 2 above) ---------------------------
    # B_o and B_w are 1.0 so that surface and reservoir rates coincide and so
    # that rho_surface / B = rho_surface, which is what NVRS assumes.
    bw=1.0, bo=1.0, bg=0.9,             # [RB/STB, RB/STB, RB/MSCF]
    uw=0.5, uo=2.5, ug=0.02,            # [cP]
    rho_res=49.0,                       # [lb/ft3] reservoir density, all phases
    rs=0.0,                             # [MSCF/STB] no dissolved gas
    swi=0.1, swr=0.1,
    # Rock compressibility [1/psi], ECLIPSE side only.  NVRS holds phi
    # constant, so strictly this should be zero -- but Flow's default CPR
    # preconditioner builds its pressure equation from d(accumulation)/dp, and
    # with every compressibility at zero that derivative vanishes, the pressure
    # matrix has no strong connections, AMG reports "Could not build any
    # aggregates" and the coarse direct solve fails on a singular matrix.  A
    # small, ordinary sandstone value restores a well-defined pressure equation.
    # It is confined to the ROCK keyword so that B_o, B_w and B_g stay exactly
    # constant and every flux term remains identical to NVRS's; the only
    # difference is a pore volume that varies by c_r * dp ~ 8e-4 over the run.
    rock_comp=4.0e-6,

    # --- initial state ------------------------------------------------------
    p_init=3000.0,                      # [psia]
    sw_init=0.2,

    # --- wells --------------------------------------------------------------
    rw=0.5, skin=0.0,                   # [ft]
    re_factor=0.2,                      # re = re_factor * dx
    darcy_const=1.127e-3,               # field-unit transmissibility constant
    # Sized for roughly half a pore volume over the schedule: enough to
    # drive a clear front past the producers, not enough to sweep the
    # field to residual oil and leave nothing moving.  selfcheck()
    # prints the actual fraction for the real permeability field.
    q_inj=500.0,                        # [STB/D] water per injector
    p_wf=2800.0,                        # [psia] producer BHP
    # Given as base-grid (40 x 40) indices; make_case() remaps them onto the
    # case grid so every case puts its wells at the same physical location.
    inj=[("I1", 8, 8), ("I2", 33, 8), ("I3", 8, 33), ("I4", 33, 33)],
    prod=[("P1", 20, 20), ("P2", 12, 28), ("P3", 28, 12)],

    # --- linear solver ------------------------------------------------------
    # The two linear tolerances are NOT the same number and must not be forced
    # to match.  They are inner tolerances feeding different nonlinear
    # convergence tests: Flow decides a Newton step has converged on its CNV
    # and material-balance criteria, NVRS on the relative drop in the residual
    # against NEWTONTOL.
    #
    # 1e-2 on both sides, and NOT because they must match -- because measurement
    # says 1e-2 is the better value for NVRS too.  Tightening it to 1e-6 on
    # 46 x 112 x 22 changed step-1 Newton iterations from 43 to 49 while taking
    # Krylov from 250 to 992 and the run from 82 s to 188 s.  It bought nothing.
    #
    # The earlier reasoning for tightening it -- that LINTOL must be below
    # NEWTONTOL -- was wrong.  They measure different residuals: LINTOL is the
    # relative reduction of the LINEAR residual within one Newton step, and
    # NEWTONTOL the reduction of the NONLINEAR residual across the step.  An
    # inexact Newton method with a forcing term of 1e-2 converges perfectly
    # well; that is why Flow's own default is 1e-2.
    #
    # The timestep chopping that prompted the change has a different cause
    # entirely, and it is fixed in main.cu: NVRS reset dt to DTREPORT at the
    # start of every report step, so it re-discovered the same too-large step
    # and threw away the same Newton iterations twenty times over.
    lin_tol_nvrs=1e-2,
    lin_tol_opm=1e-2,
    lin_maxit=200,

    # How many consecutive Newton iterations may reuse the BoomerAMG hierarchy
    # built for an earlier one.  Rebuilding it every iteration was 30 s of a
    # 65 s run on one GB200 and 158 s of 225 s on four, because the parallel
    # coarsening is latency-bound and does not divide when the grid does.
    #
    # This is not a thumb on the scale.  Flow reuses its CPR setup by default
    # (--cpr-reuse-setup), the technique is standard in every production CPR
    # implementation, and NVRS abandons a stale hierarchy as soon as it starts
    # costing AMGREUSEGROWTH times the Krylov iterations a fresh one needed --
    # so a case where reuse does not hold up pays one bad solve, not a wrong
    # answer.  The report prints rebuilds against reuses for both to check.
    # Raised from 4 and 1.3 after measuring what a rebuild is worth.
    # On 200x200x25 a setup is 42 ms on one GB200 and 94 ms on two,
    # against a V-cycle of about 1 ms, so one avoided rebuild pays for
    # scores of extra Krylov iterations.  The old pair rebuilt on 61 of
    # 147 Newton iterations; these rebuild on 15 of 59.
    amg_reuse=10,
    amg_reuse_growth=2.0,

    # Aggressive coarsening with the coarsest grid relaxed on the device
    # rather than eliminated on the host.  A V-cycle here is dominated by
    # per-level overhead, not arithmetic -- level 0 is a few hundred
    # microseconds of memory traffic inside a cycle costing several times
    # that on a single rank, with nothing to communicate -- so the win is
    # in having fewer levels.  Seven at 2.07 ms became three at 1.11 ms.
    #
    # This tunes NVRS's own preconditioner and does not touch Flow.  It is
    # the same kind of choice as --cpr-reuse-setup on Flow's side: both
    # simulators are run as their authors would run them, and neither
    # setting changes the answer either produces.
    amg_fast=1,

    # --- schedule -----------------------------------------------------------
    dt_report=100.0, nsteps=20,         # [days]
    # Maximum internal timestep, 0 = one step per report interval.
    #
    # This was briefly set to 5 days on the theory that NVRS's 100-day steps
    # were under-resolved next to Flow.  Measured, that was wrong twice over:
    # Flow's own summary reports 24 timesteps for 20 report steps and
    # "stepsize 100 days" by report step 18, so Flow is not taking small steps
    # either; and refining NVRS 20x moved the disagreement with Flow by under
    # 1% (water RMS 0.2022 -> 0.2029) while costing 4x the runtime.  The
    # remaining difference is not resolution, so both simulators take the same
    # 100-day steps and the comparison stays cheap.  Set DTMAX in the deck to
    # re-run the experiment.
    #
    # Deliberately still 0 even though DTMAX was the single largest saving
    # on MILLION.DATA, where it took the run from 23.0 s to 5.4 s.  That
    # deck fails at its 100-day report step and chops four times to reach
    # 6.25 days, so capping it there only skips a search whose answer was
    # already known; the steps actually taken are unchanged.  These cases
    # converge at 100 days, and Flow takes 100-day steps too, so a cap here
    # would make NVRS solve a finer problem than Flow and stop the two
    # being comparable.  If a run starts printing [chop] lines, set dt_max
    # to whatever dt it settles at -- that is a saving, not a thumb on the
    # scale.
    dt_max=0.0,                         # [days]
)

def _remap(idx, n_base, n_new):
    """Base-grid cell index -> case-grid index at the same physical position."""
    return max(1, min(n_new, int(round((idx - 0.5) * n_new / n_base + 0.5))))


def make_case(three_phase, sg_init, dims=None):
    """One case: phases, initial gas, and how finely the reservoir is gridded."""
    P = dict(SHARED)
    P["three_phase"] = three_phase
    P["sg_init"] = sg_init
    if dims:
        P["nx"], P["ny"], P["nz"] = (int(d) for d in dims)
    ex, ey, ez = P["extent"]
    P["dx"] = ex / P["nx"]
    P["dy"] = ey / P["ny"]
    P["dz"] = ez / P["nz"]
    bx, by, _bz = P["base_dims"]
    for key in ("inj", "prod"):
        P[key] = [(nm, _remap(i, bx, P["nx"]), _remap(j, by, P["ny"]))
                  for nm, i, j in SHARED[key]]
    return P


# Sizes are exact block refinements of the 40 x 40 x 3 field, so the
# nearest-neighbour resampling in load_rock() is exact rather than approximate.
CASES = {
    "BENCH2P":    make_case(False, 0.00),                    #      4,800 cells
    "BENCH3P":    make_case(True,  0.02),                    #      4,800
    "BENCH2P_L":  make_case(False, 0.00, (120, 120, 9)),     #    129,600
    "BENCH3P_L":  make_case(True,  0.02, (120, 120, 9)),     #    129,600
    "BENCH2P_XL": make_case(False, 0.00, (200, 200, 15)),    #    600,000
    "BENCH3P_XL": make_case(True,  0.02, (200, 200, 15)),    #    600,000
}

# Tabulated curves for the three-phase case.  NVRS reads these verbatim from
# its deck; the ECLIPSE deck receives the identical numbers.
SWOF3 = [(0.00, 1.0000, 0.0000), (0.30, 1.0000, 0.0000), (0.35, 0.5900, 0.0100),
         (0.40, 0.3200, 0.0200), (0.45, 0.1800, 0.0340), (0.50, 0.0800, 0.0460),
         (0.55, 0.0300, 0.0680), (0.60, 0.0100, 0.0900), (0.65, 0.0010, 0.1280),
         (0.70, 0.0001, 0.1660), (0.75, 0.0000, 0.2000), (0.80, 0.0000, 0.2400),
         (1.00, 0.0000, 0.2400)]
SGOF3 = [(0.00, 1.0000, 0.0000), (0.03, 0.7500, 0.0000), (0.05, 0.5900, 0.0200),
         (0.10, 0.3200, 0.0900), (0.15, 0.1800, 0.1600), (0.20, 0.0800, 0.2400),
         (0.25, 0.0300, 0.3300), (0.30, 0.0100, 0.4300), (0.35, 0.0010, 0.5500),
         (0.40, 0.0000, 0.6700), (0.45, 0.0000, 0.8100), (0.50, 0.0000, 1.0000),
         (1.00, 0.0000, 1.0000)]


# ===========================================================================
# Rock
# ===========================================================================
def load_rock(P):
    """(perm, poro) as flat arrays in ECLIPSE natural order (i fastest).

    Reproduces src/physics_host.cu: the permeability file holds one value per
    global cell in natural order, and porosity is the linear map of
    permeability onto [poro_min, poro_max].
    """
    bx, by, bz = P["base_dims"]
    nbase = bx * by * bz
    # data/ is tracked with Git LFS.  A checkout made without the LFS payload
    # leaves a 3-line pointer file behind, and np.loadtxt's complaint about the
    # word "version" gives no hint what actually went wrong.
    with open(P["permfile"], "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    if head.startswith("version https://git-lfs"):
        raise SystemExit(
            "%s is a Git LFS pointer, not the permeability field. "
            "Fetch the real data first:  git lfs install && git lfs pull"
            % P["permfile"])
    raw = np.loadtxt(P["permfile"]).ravel()
    if raw.size < nbase:
        raise SystemExit("PERMFILE has %d values, need %d" % (raw.size, nbase))
    # Natural order (i fastest, then j, then k) is exactly C order for a
    # (nz, ny, nx) array, so this reshape and the ravel below are both no-ops
    # on the byte layout ECLIPSE expects.
    base = raw[:nbase].astype(float).reshape((bz, by, bx))

    nx, ny, nz = P["nx"], P["ny"], P["nz"]
    if (nx, ny, nz) != (bx, by, bz):
        # Nearest-neighbour block mapping.  Refining a heterogeneous field by
        # interpolation would smooth it and change the permeability
        # distribution, so a refined case would no longer be the same rock;
        # block replication keeps min, max, mean and histogram identical and
        # refines only the flow solution.
        ix = np.minimum(np.arange(nx) * bx // nx, bx - 1)
        iy = np.minimum(np.arange(ny) * by // ny, by - 1)
        iz = np.minimum(np.arange(nz) * bz // nz, bz - 1)
        base = base[np.ix_(iz, iy, ix)]

    perm = base.ravel()
    lo, hi = perm.min(), perm.max()
    t = (perm - lo) / (hi - lo) if hi > lo else np.zeros_like(perm)
    poro = P["poro_min"] + (P["poro_max"] - P["poro_min"]) * t
    return perm, poro


def well_index(P, k_md):
    """NVRS's Peaceman well index [cP.rb/day/psi].

    WI = 2 pi C k h / (ln(re/rw) + skin), re = re_factor * dx.  ECLIPSE defines
    COMPDAT item 8 identically, so this number transfers without conversion.
    """
    import math
    re = P["re_factor"] * P["dx"]
    denom = math.log(re / P["rw"]) + P["skin"]
    if denom <= 0:
        raise SystemExit("ln(re/rw) + skin = %.4g <= 0: well index would be "
                         "negative and producers would inject" % denom)
    return 2.0 * math.pi * P["darcy_const"] * k_md * P["dz"] / denom


def corey_swof(P, n=201):
    """SWOF tabulating NVRS's analytic two-phase Corey curves exactly.

    include/fluid.hpp relperm2 uses S = (Sw - Swi)/(1 - Swi - Swr) with
    krw = S^2 and kro = (1 - S)^2, and never consults a table.  Tabulating the
    same closed form on a fine grid is what makes BENCH2P an exact match.
    """
    swi, swr = P["swi"], P["swr"]
    sw = np.linspace(swi, 1.0 - swr, n)
    S = (sw - swi) / (1.0 - swi - swr)
    krw, krow = S ** 2, (1.0 - S) ** 2
    rows = [(0.0, 1.0, 0.0)] if swi > 0 else []
    rows += [(float(a), float(b), float(c)) for a, b, c in zip(sw, krow, krw)]
    if 1.0 - swr < 1.0:
        rows.append((1.0, 0.0, float(krw[-1])))
    return rows


# ===========================================================================
# Emitters
# ===========================================================================
def fmt_array(vals, per_line=8, fmt="%.6g"):
    """An ECLIPSE array body, run-length compressed where it pays."""
    out, i, n = [], 0, len(vals)
    toks = []
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        run = j - i + 1
        s = fmt % vals[i]
        toks.append("%d*%s" % (run, s) if run > 1 else s)
        i = j + 1
    for k in range(0, len(toks), per_line):
        out.append(" " + " ".join(toks[k:k + per_line]))
    return "\n".join(out)


def write_nvrs_deck(name, P, path):
    """The NVRS deck (cuda/decks format)."""
    perm, _ = load_rock(P)
    g = 62.428                      # lb/ft3 per g/cc
    rho_gcc = P["rho_res"] / g      # NVRS uses this directly as reservoir density
    ncomp = 3 if P["three_phase"] else 2
    so_init = 1.0 - P["sw_init"] - P["sg_init"]

    L = []
    A = L.append
    A("-- " + "=" * 69)
    A("--  %s — NVRS side of the OPM Flow benchmark" % name)
    A("--")
    A("--  GENERATED by benchmark/make_decks.py.  Do not edit by hand: the")
    A("--  ECLIPSE deck %s_OPM.DATA is emitted from the same" % name)
    A("--  parameters and the comparison is only meaningful while the two")
    A("--  agree.  Re-run make_decks.py instead.")
    A("--")
    A("--  PVTCONST 1 pins B_o, B_g, R_s and mu_g to the constants below so")
    A("--  that the fluid can be written into an ECLIPSE PVT table verbatim.")
    A("-- " + "=" * 69)
    A("")
    A("TITLE    %s OPM-FLOW BENCHMARK %dx%dx%d"
      % (name, P["nx"], P["ny"], P["nz"]))
    A("")
    A("NCOMP    %d" % ncomp)
    A("MODEL    BLACKOIL")
    A("OUTDIR   OUTPUT_%s" % name)
    A("CASENAME %s" % name)
    A("")
    A("-- ---- GRID ----------------------------------------------------------")
    A("NX       %d" % P["nx"])
    A("NY       %d" % P["ny"])
    A("NZ       %d" % P["nz"])
    A("DX       %g" % P["dx"])
    A("DY       %g" % P["dy"])
    A("DZ       %g" % P["dz"])
    A("KZMULT   %g" % P["kz_mult"])
    # NVRS reads one permeability value per cell, so a refined case needs the
    # resampled field on disk.  Writing it for every case -- rather than
    # pointing the base case straight at data/iglesias3.out -- guarantees the
    # two simulators are handed bit-identical rock instead of two independent
    # resamplings that agree only if this script is deterministic.
    permout = os.path.join(os.path.dirname(os.path.abspath(path)),
                           "%s_perm.txt" % name)
    np.savetxt(permout, perm, fmt="%.8g")
    # PERMFILE is resolved relative to the working directory, which run.sh and
    # run_comparison.sh both set to cuda/.  Fall back to the absolute path when
    # a relative one cannot be formed (a different drive on Windows).
    try:
        permref = os.path.relpath(permout, CUDA).replace("\\", "/")
    except ValueError:
        permref = permout.replace("\\", "/")
    A("PERMFILE %s" % permref)
    A("POROMIN  %g" % P["poro_min"])
    A("POROMAX  %g" % P["poro_max"])
    A("")
    A("-- ---- PROPS ---------------------------------------------------------")
    A("PVTCONST 1             -- hold the fluid fixed; see the header")
    A("UW       %g" % P["uw"])
    A("UO       %g" % P["uo"])
    A("UG       %g" % P["ug"])
    A("BW       %g" % P["bw"])
    A("BO       %g" % P["bo"])
    A("BG       %g" % P["bg"])
    A("RS       %g             -- no dissolved gas" % P["rs"])
    A("-- Equal reservoir densities: the hydrostatic gradient is identical to")
    A("-- ECLIPSE's and there is no buoyancy for the two runs to disagree on.")
    A("RHOW     %.6f" % rho_gcc)
    A("RHOO     %.6f" % rho_gcc)
    A("RHOG     %.6f" % rho_gcc)
    A("CFW      0.0")
    A("CFO      0.0")
    A("CT       0.0")
    A("SWI      %g" % P["swi"])
    A("SWR      %g" % P["swr"])
    A("PINIT    %g" % P["p_init"])
    A("PBUB     %g" % P["p_init"])
    A("PATM     14.6959")
    A("SWINIT   %g" % P["sw_init"])
    A("SGINIT   %g" % P["sg_init"])
    A("SOINIT   %g" % so_init)
    A("GRAVITY  1")
    A("DARCY    %g" % P["darcy_const"])
    A("REFACTOR %g" % P["re_factor"])
    A("")
    if P["three_phase"]:
        A("-- ---- SWOF: (Sw, Krow, Krw) ----------------------------------------")
        A("SWOF")
        for r in SWOF3:
            A(" %5.2f  %7.4f %7.4f" % r)
        A("/")
        A("")
        A("-- ---- SGOF: (Sg, Krog, Krg) ----------------------------------------")
        A("SGOF")
        for r in SGOF3:
            A(" %5.2f  %7.4f %7.4f" % r)
        A("/")
    else:
        A("-- Two-phase: the NVRS kernel uses analytic Corey curves")
        A("--     krw = S^2,  kro = (1-S)^2,  S = (Sw-SWI)/(1-SWI-SWR)")
        A("-- and never reads SWOF, so no table is given here.  The ECLIPSE")
        A("-- deck tabulates exactly those curves.")
    A("")
    A("-- ---- SOLVER --------------------------------------------------------")
    A("KRYLOV   FGMRES")
    A("LINTOL   %g           -- inexact-Newton forcing term, as Flow uses"
      % P["lin_tol_nvrs"])
    A("LINMAXIT %d" % P["lin_maxit"])
    A("RESTART  30")
    if P["amg_fast"]:
        A("AMGFAST  1             -- coarsen aggressively, coarsest grid")
        A("                       --    relaxed on the device not the host")
    A("AMGREUSE %d            -- Newton iterations that may reuse the AMG"
      % P["amg_reuse"])
    A("AMGREUSEGROWTH %-8g-- ...hierarchy, unless Krylov exceeds this x"
      % P["amg_reuse_growth"])
    A("                       --    the count the fresh one needed")
    A("MIXED    1             -- FP32 BoomerAMG inside the CPR preconditioner")
    A("EMULATE  1             -- df64 emulated FP64 operator")
    A("JACOBIAN ANALYTIC")
    A("")
    A("-- ---- TIME ----------------------------------------------------------")
    A("DTREPORT   %g" % P["dt_report"])
    if P["dt_max"] > 0:
        A("DTMAX      %g           -- internal timestep cap" % P["dt_max"])
    A("TEND       %g" % (P["dt_report"] * P["nsteps"]))
    A("NEWTONMAX  20")
    A("NEWTONSTALLMAX 4       -- abandon a sub-step that stops reducing the")
    A("                       --   residual; a stall does not recover, and the")
    A("                       --   full Newton budget is the price of every")
    A("                       --   lost attempt")
    A("DTKEEP     1           -- carry dt ACROSS report steps, as Flow does,")
    A("DTCONTROL  1           --   and rescale it from the last converged")
    A("                       --   sub-step's Newton count instead of resetting")
    A("                       --   onto a dyadic lattice at every report step")
    A("OPMCONV    1           -- Flow's getReservoirConvergence: the signed")
    A("                       --   component material balance and the local CNV")
    A("                       --   bound, tested at every Newton iteration.")
    A("                       --   Leave CNVRELAXPV and CNVRELAXED at their")
    A("                       --   defaults (0.03 and 1.0), which are Flow's")
    A("                       --   own; tightening CNVRELAXED to CNVTOL makes")
    A("                       --   the relaxed test identical to the strict one")
    A("                       --   and OPMCONV worthless.")
    A("LINESEARCH 6           -- backtrack the fully implicit Newton update")
    A("                       --   until the residual descends.  The accepted")
    A("                       --   trial is carried into the next iteration, so")
    A("                       --   a search that takes alpha = 1 costs nothing.")
    A("APPLEYARD  1           -- chop a saturation update that crosses a relperm")
    A("                       --   endpoint so it lands just past it, rather")
    A("                       --   than leaping between mobility regimes.")
    A("NEWTONTOL  5e-3")
    A("DPMAX      500         -- per-cell pressure and saturation change")
    A("DSMAX      0.2         --   limits per Newton iteration")
    A("")
    A("-- ---- OUTPUT --------------------------------------------------------")
    A("MAKECSV      1")
    A("MAKESUMMARY  1")
    A("ECLOUT       1         -- the comparison reads the ECLIPSE output")
    A("AUTOPLOT     0         -- benchmark/compare.py does the plotting")
    A("")
    A("-- ---- WELLS:  I  J  TYPE  RATE  RW  SKIN  PWF  PERF  NAME ------------")
    A("WELLS")
    for nm, i, j in P["inj"]:
        A("  %2d  %2d   1   %g   %g   %g  %g   all %s"
          % (i, j, P["q_inj"], P["rw"], P["skin"], P["p_wf"], nm))
    for nm, i, j in P["prod"]:
        A("  %2d  %2d  -1     -1   %g   %g  %g   all %s"
          % (i, j, P["rw"], P["skin"], P["p_wf"], nm))
    A("/")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    return path


def write_opm_deck(name, P, path):
    """The ECLIPSE deck for OPM Flow."""
    perm, poro = load_rock(P)
    nx, ny, nz = P["nx"], P["ny"], P["nz"]
    ncell = nx * ny * nz
    three = P["three_phase"]
    # Surface densities that reproduce the shared reservoir density: ECLIPSE
    # divides by B, NVRS does not.  See note 4 in the module header.
    rho_o = P["rho_res"] * P["bo"]
    rho_w = P["rho_res"] * P["bw"]
    rho_g = P["rho_res"] * P["bg"]
    swof = SWOF3 if three else corey_swof(P)

    L = []
    A = L.append
    A("-- " + "=" * 69)
    A("--  %s_OPM — ECLIPSE side of the NVRS / OPM Flow benchmark" % name)
    A("--")
    A("--  GENERATED by benchmark/make_decks.py from the same parameters as")
    A("--  %s.DATA.  Do not edit by hand." % name)
    A("--")
    A("--  Everything is constant and incompressible so that the two")
    A("--  simulators solve the same discrete system; see make_decks.py.")
    A("-- " + "=" * 69)
    A("")
    A("RUNSPEC")
    A("")
    A("TITLE")
    A(" %s OPM-FLOW BENCHMARK %dx%dx%d" % (name, nx, ny, nz))
    A("")
    A("DIMENS")
    A(" %d %d %d /" % (nx, ny, nz))
    A("")
    A("OIL")
    A("WATER")
    if three:
        A("GAS")
    A("-- No DISGAS and no VAPOIL: the phases are immiscible, matching RS 0")
    A("-- on the NVRS side.")
    A("")
    A("FIELD")
    A("")
    # Item 3 is the maximum number of rows in a saturation table.  The
    # two-phase SWOF below has a couple of hundred, so a fixed 60 understates
    # it; size it from what is actually written.
    nssfun = max(len(swof), len(SGOF3) if three else 0) + 20
    A("TABDIMS")
    A(" 1 1 %d 60 /" % nssfun)
    A("")
    A("WELLDIMS")
    A(" %d %d %d %d /" % (len(P["inj"]) + len(P["prod"]), nz,
                          len(P["prod"]), len(P["inj"]) + len(P["prod"])))
    A("")
    A("START")
    A(" 1 'JAN' 2020 /")
    A("")
    A("UNIFIN")
    A("UNIFOUT")
    A("")
    A("GRID")
    A("")
    A("INIT")
    A("")
    A("DXV")
    A(" %d*%g /" % (nx, P["dx"]))
    A("DYV")
    A(" %d*%g /" % (ny, P["dy"]))
    A("DZV")
    A(" %d*%g /" % (nz, P["dz"]))
    A("")
    A("TOPS")
    A(" %d*%g /" % (nx * ny, P["tops"]))
    A("")
    A("-- Permeability from %s, natural order, identical to the PERMFILE the"
      % os.path.basename(P["permfile"]))
    A("-- NVRS deck reads.  PERMY and PERMZ are derived with COPY/MULTIPLY")
    A("-- rather than written out: at %d cells three explicit arrays would"
      % ncell)
    A("-- treble the deck for no added information, and NVRS forms them the")
    A("-- same way (Ky = Kx, Kz = KZMULT * Kx).")
    A("PERMX")
    A(fmt_array(perm))
    A("/")
    A("")
    A("COPY")
    A(" 'PERMX' 'PERMY' /")
    A(" 'PERMX' 'PERMZ' /")
    A("/")
    A("")
    A("MULTIPLY")
    A(" 'PERMZ' %g /" % P["kz_mult"])
    A("/")
    A("")
    A("PORO")
    A(fmt_array(poro))
    A("/")
    A("")
    A("PROPS")
    A("")
    A("-- Zero compressibility throughout: NVRS's accumulation is V*phi/dt")
    A("-- with phi and 1/B constant, i.e. an incompressible system.")
    A("PVTW")
    A("-- Pref      Bw     Cw     muw    viscosibility")
    A(" %8.2f  %6.4f  0.0  %6.4f  0.0 /" % (P["p_init"], P["bw"], P["uw"]))
    A("")
    A("PVCDO")
    A("-- Pref      Bo     Co     muo    viscosibility")
    A(" %8.2f  %6.4f  0.0  %6.4f  0.0 /" % (P["p_init"], P["bo"], P["uo"]))
    A("")
    if three:
        A("-- Dry gas.  Bg is constant to match the NVRS constant; the table is")
        A("-- given a 0.01%% decrease across the pressure range only because")
        A("-- ECLIPSE requires Bg to be strictly decreasing.")
        A("PVDG")
        A("--    P        Bg       mug")
        A(" %8.2f  %9.6f  %7.5f" % (14.7, P["bg"], P["ug"]))
        A(" %8.2f  %9.6f  %7.5f /" % (10000.0, P["bg"] * 0.9999, P["ug"]))
        A("")
    A("-- Rock compressibility is the one place a non-zero compressibility is")
    A("-- kept: Flow's CPR preconditioner needs d(accumulation)/dp to be")
    A("-- non-zero to extract a pressure equation at all.  It changes the pore")
    A("-- volume by c_r * dp ~ %.1e over the run.  See make_decks.py."
      % (P["rock_comp"] * 200.0))
    A("ROCK")
    A("-- Pref     Cr")
    A(" %8.2f  %.3e /" % (P["p_init"], P["rock_comp"]))
    A("")
    A("DENSITY")
    A("--   oil     water    gas          [lb/ft3, surface]")
    A("-- Chosen so rho_surface / B = %.4f lb/ft3 for every phase, which is"
      % P["rho_res"])
    A("-- the reservoir density NVRS uses.  No buoyancy, identical head.")
    if three:
        A(" %8.4f %8.4f %8.4f /" % (rho_o, rho_w, rho_g))
    else:
        A(" %8.4f %8.4f %8.4f /" % (rho_o, rho_w, rho_o))
    A("")
    # COLUMN ORDER.  NVRS's deck format is (Sw, Krow, Krw) and (Sg, Krog, Krg),
    # which is how SWOF3/SGOF3 and corey_swof() are stored above.  ECLIPSE is
    # the other way round -- SWOF is (Sw, KRW, KROW, PCOW) and SGOF is
    # (Sg, KRG, KROG, PCOG) -- so the two relperm columns are SWAPPED on the
    # way out.  Emitting them in NVRS's order makes Flow read krow as the water
    # relperm, which starts at 1.0 and decreases, and it aborts with
    #     "In SWOF table SATNUM = 1, first value of krw should be 0."
    A("SWOF")
    if not three:
        A("-- NVRS's analytic two-phase Corey curves, tabulated:")
        A("--   krw = S^2,  kro = (1-S)^2,  S = (Sw - %g)/(1 - %g - %g)"
          % (P["swi"], P["swi"], P["swr"]))
    A("--   Sw        Krw       Krow      Pcow")
    for sw, krow, krw in swof:
        A(" %7.5f  %8.6f  %8.6f  0.0" % (sw, krw, krow))
    A("/")
    A("")
    if three:
        A("SGOF")
        A("--   Sg        Krg       Krog      Pcog")
        for sg, krog, krg in SGOF3:
            A(" %7.5f  %8.6f  %8.6f  0.0" % (sg, krg, krog))
        A("/")
        A("")
        A("-- NOTE: no STONE1/STONE2, so Flow uses the ECLIPSE default")
        A("-- saturation-weighted three-phase oil relative permeability, while")
        A("-- NVRS uses a Baker product.  This is the one model difference the")
        A("-- benchmark cannot remove; BENCH2P has no such difference.")
        A("")
    A("SOLUTION")
    A("")
    A("-- Explicit initial state, matching NVRS's uniform initialisation")
    A("-- exactly.  No EQUIL: an equilibration would put OPM in hydrostatic")
    A("-- balance and NVRS not, which is a difference in the initial")
    A("-- condition rather than in the solver.")
    A("PRESSURE")
    A(" %d*%g /" % (ncell, P["p_init"]))
    A("")
    A("SWAT")
    A(" %d*%g /" % (ncell, P["sw_init"]))
    A("")
    if three:
        A("SGAS")
        A(" %d*%g /" % (ncell, P["sg_init"]))
        A("")
    A("RPTRST")
    A(" 'BASIC=2' /")
    A("")
    A("SUMMARY")
    A("")
    A("-- One vector per well for each quantity the comparison plots.")
    for kw in ("WOPR", "WWPR", "WGPR", "WWCT", "WGOR", "WBHP", "WWIR"):
        if kw in ("WGPR", "WGOR") and not three:
            continue
        A(kw)
        A("/")
    A("FOPR")
    A("FWPR")
    A("FOPT")
    A("FWPT")
    A("FPR")
    A("TCPU")
    A("")
    A("SCHEDULE")
    A("")
    A("WELSPECS")
    for nm, i, j in P["inj"]:
        A(" '%s' 'INJ' %d %d %g 'WATER' /" % (nm, i, j, P["tops"]))
    for nm, i, j in P["prod"]:
        A(" '%s' 'PRO' %d %d %g 'OIL' /" % (nm, i, j, P["tops"]))
    A("/")
    A("")
    A("-- Item 8 is the connection transmissibility factor.  ECLIPSE defines it")
    A("-- as 2 pi c k h / (ln(re/rw) + skin), which is exactly NVRS's well")
    A("-- index, so giving it explicitly removes the well model as a source of")
    A("-- difference: NVRS uses re = %g * dx = %g ft rather than the"
      % (P["re_factor"], P["re_factor"] * P["dx"]))
    A("-- anisotropic Peaceman radius Flow would derive for itself.")
    A("COMPDAT")
    for nm, i, j in list(P["inj"]) + list(P["prod"]):
        for k in range(1, nz + 1):
            c = (i - 1) + nx * ((j - 1) + ny * (k - 1))
            wi = well_index(P, perm[c])
            A(" '%s' %d %d %d %d 'OPEN' 1* %11.6f %6.3f 1* %g 1* 'Z' /"
              % (nm, i, j, k, k, wi, 2.0 * P["rw"], P["skin"]))
    A("/")
    A("")
    A("WCONINJE")
    for nm, _i, _j in P["inj"]:
        A(" '%s' 'WATER' 'OPEN' 'RATE' %g 1* 100000 /" % (nm, P["q_inj"]))
    A("/")
    A("")
    A("WCONPROD")
    for nm, _i, _j in P["prod"]:
        A(" '%s' 'OPEN' 'BHP' 5* %g /" % (nm, P["p_wf"]))
    A("/")
    A("")
    A("TSTEP")
    A(" %d*%g /" % (P["nsteps"], P["dt_report"]))
    A("")
    A("END")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    return path


def check_sat_tables(swof, sgof):
    """Apply the rules OPM's RelpermDiagnostics enforces, before Flow sees them.

    Tables are passed in NVRS order (S, Kro*, Kr*); the ECLIPSE emitter swaps
    the two relperm columns.  A swapped or non-monotonic table is not a parse
    error -- Flow reads the whole deck, builds the grid, distributes the wells,
    and only then aborts in the saturation-function diagnostics with a message
    buried under one backtrace per MPI rank.  Checking here turns a 32-rank
    abort into one line.

    Returns a list of complaints; empty means the tables satisfy the rules.
    """
    bad = []

    def monotone(name, vals, non_decreasing):
        for i in range(1, len(vals)):
            d = vals[i] - vals[i - 1]
            if (d < -1e-12) if non_decreasing else (d > 1e-12):
                bad.append("%s is not %s at row %d (%.6g -> %.6g)"
                           % (name, "non-decreasing" if non_decreasing
                              else "non-increasing", i, vals[i - 1], vals[i]))
                return

    sw = [r[0] for r in swof]
    krow = [r[1] for r in swof]
    krw = [r[2] for r in swof]
    if abs(krw[0]) > 1e-12:
        bad.append("SWOF: first krw is %.6g, must be 0" % krw[0])
    if abs(krow[-1]) > 1e-12:
        bad.append("SWOF: last krow is %.6g, must be 0" % krow[-1])
    monotone("SWOF Sw", sw, True)
    monotone("SWOF krw", krw, True)
    monotone("SWOF krow", krow, False)
    for i in range(1, len(sw)):
        if sw[i] - sw[i - 1] <= 0:
            bad.append("SWOF: Sw is not strictly increasing at row %d" % i)
            break
    swco = sw[0]

    if sgof:
        sg = [r[0] for r in sgof]
        krog = [r[1] for r in sgof]
        krg = [r[2] for r in sgof]
        if abs(krg[0]) > 1e-12:
            bad.append("SGOF: first krg is %.6g, must be 0" % krg[0])
        if abs(krog[-1]) > 1e-12:
            bad.append("SGOF: last krog is %.6g, must be 0" % krog[-1])
        monotone("SGOF Sg", sg, True)
        monotone("SGOF krg", krg, True)
        monotone("SGOF krog", krog, False)
        # The two families have to agree where they meet: at zero gas the oil
        # relperm must be the one SWOF gives at connate water.
        if abs(krog[0] - krow[0]) > 1e-9:
            bad.append("krog(Sg=0) = %.6g but krow(Sw=Swco) = %.6g; they must "
                       "be equal" % (krog[0], krow[0]))
        if abs(sg[-1] - (1.0 - swco)) > 1e-9:
            bad.append("SGOF: maximum Sg is %.6g, must be 1 - Swco = %.6g"
                       % (sg[-1], 1.0 - swco))
    return bad


def write_solver_json(name, P, path, gpu):
    """OPM's linear-solver settings, as the JSON --linear-solver accepts.

    This is what actually puts Flow's linear solve on the GPU.  It is a
    separate mechanism from --accelerator-mode, which selects the older BDA
    bridge; "solver": "gpubicgstab" with a GPU preconditioner selects the
    newer gpuistl path, and that is the combination proven on the Aramco runs.

    NOTE ON THE ALGORITHMS.  OPMGPUILU0 is a single-stage ILU(0).  NVRS runs a
    two-stage CPR -- FP32 BoomerAMG on the pressure block, then a bs x bs
    block-Jacobi correction.  These are different preconditioners, and the
    report says so rather than implying a like-for-like CPR comparison.  What
    IS matched is the tolerance, the iteration cap, and the outer FP64 Krylov.

    store_factorization_as_float is OPM's mixed-precision knob: it keeps the
    ILU factors in FP32 while the Krylov stays FP64, which is the same trade
    NVRS makes with an FP32 AMG.  Enabled here so the two sides are making
    comparable precision choices.
    """
    if gpu:
        body = {
            "tol": P["lin_tol_opm"],
            "maxiter": P["lin_maxit"],
            "verbosity": 0,
            "solver": "gpubicgstab",
            "preconditioner": {
                "type": "OPMGPUILU0",
                "split_matrix": True,
                "tune_gpu_kernels": True,
                "store_factorization_as_float": True,
            },
        }
    else:
        body = {
            "tol": P["lin_tol_opm"],
            "maxiter": P["lin_maxit"],
            "verbosity": 0,
            "solver": "bicgstab",
            "preconditioner": {"type": "ParOverILU0", "relaxation": 1.0},
        }
    import json as _json
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        _json.dump(body, fh, indent=2)
        fh.write("\n")
    return path


def selfcheck(P):
    """Print the numbers a reader would otherwise have to trust."""
    perm, poro = load_rock(P)
    print("    permeability  %.4g - %.4g mD   (mean %.4g)"
          % (perm.min(), perm.max(), perm.mean()))
    print("    porosity      %.4g - %.4g" % (poro.min(), poro.max()))
    print("    well index    %.6g - %.6g cP.rb/day/psi"
          % (well_index(P, perm.min()), well_index(P, perm.max())))
    pv = poro.sum() * P["dx"] * P["dy"] * P["dz"] / 5.61458
    print("    pore volume   %.4g bbl" % pv)
    inj_total = P["q_inj"] * len(P["inj"]) * P["dt_report"] * P["nsteps"]
    print("    injected      %.4g bbl over %g days (%.1f%% of pore volume)"
          % (inj_total, P["dt_report"] * P["nsteps"], 100.0 * inj_total / pv))
    print("    reservoir density %.4f lb/ft3 for all phases -> hydrostatic "
          "gradient %.5f psi/ft" % (P["rho_res"], P["rho_res"] / 144.0))
    swof = SWOF3 if P["three_phase"] else corey_swof(P)
    bad = check_sat_tables(swof, SGOF3 if P["three_phase"] else None)
    if bad:
        for b in bad:
            print("    !! %s" % b)
        raise SystemExit("saturation tables would be rejected by OPM Flow; "
                         "fix them before running the benchmark")
    print("    saturation tables: OPM endpoint rules satisfied "
          "(%d SWOF rows%s)"
          % (len(swof), ", %d SGOF rows" % len(SGOF3)
             if P["three_phase"] else ""))
    if not P["three_phase"]:
        # The whole claim of BENCH2P rests on the emitted SWOF reproducing
        # relperm2.  Check it the way Flow will use it: interpolate the table
        # and compare against the closed form on a grid the table does not
        # share knots with.
        tab = np.array(corey_swof(P))
        sw = np.linspace(P["swi"] + 1e-6, 1.0 - P["swr"] - 1e-6, 97)
        S = (sw - P["swi"]) / (1.0 - P["swi"] - P["swr"])
        kro_t = np.interp(sw, tab[:, 0], tab[:, 1])
        krw_t = np.interp(sw, tab[:, 0], tab[:, 2])
        worst = max(np.abs(kro_t - (1 - S) ** 2).max(),
                    np.abs(krw_t - S ** 2).max())
        print("    SWOF vs analytic relperm2: max |difference| = %.3e "
              "over %d interpolated points" % (worst, sw.size))


def main(argv):
    """make_decks.py [--out DIR] [--grid NX NY NZ] [CASE ...]

    With no case names every predefined case is written.  Naming cases matters
    for the large ones: BENCH2P_XL is 600,000 cells and takes a while to
    format, so run_comparison.sh asks only for the cases it is about to run.
    """
    args = list(argv[1:])
    outdir, grid, names = HERE, None, []
    while args:
        a = args.pop(0)
        if a == "--out":
            outdir = args.pop(0)
        elif a == "--grid":
            grid = (int(args.pop(0)), int(args.pop(0)), int(args.pop(0)))
        elif a.startswith("-"):
            raise SystemExit("unknown option %s -- %s" % (a, main.__doc__))
        else:
            names.append(a)
    # A bare directory argument is the historical calling convention.
    if len(names) == 1 and os.path.isdir(names[0]):
        outdir, names = names[0], []

    cases = dict(CASES)
    if grid:
        # An explicit grid overrides the size of every case asked for, so a
        # one-off resolution can be run without editing this file.
        for nm in (names or list(cases)):
            base = cases.get(nm)
            if base is None:
                raise SystemExit("unknown case %s; known: %s"
                                 % (nm, ", ".join(sorted(CASES))))
            cases[nm] = make_case(base["three_phase"], base["sg_init"], grid)
    unknown = [n for n in names if n not in cases]
    if unknown:
        raise SystemExit("unknown case(s) %s; known: %s"
                         % (", ".join(unknown), ", ".join(sorted(CASES))))
    selected = [(n, cases[n]) for n in names] if names else list(cases.items())

    os.makedirs(outdir, exist_ok=True)
    for name, P in selected:
        if not os.path.exists(P["permfile"]):
            raise SystemExit("permeability file not found: %s" % P["permfile"])
        print("[decks] %s (%s, %d x %d x %d = %s cells)"
              % (name, "three-phase" if P["three_phase"] else "two-phase",
                 P["nx"], P["ny"], P["nz"],
                 format(P["nx"] * P["ny"] * P["nz"], ",d")))
        a = write_nvrs_deck(name, P, os.path.join(outdir, name + ".DATA"))
        b = write_opm_deck(name, P, os.path.join(outdir, name + "_OPM.DATA"))
        for gpu in (True, False):
            write_solver_json(
                name, P,
                os.path.join(outdir, "%s_solver_%s.json"
                             % (name, "gpu" if gpu else "cpu")), gpu)
        selfcheck(P)
        print("    cell size     %.4g x %.4g x %.4g ft"
              % (P["dx"], P["dy"], P["dz"]))
        print("    linear tol: NVRS %g, OPM %g; max %d iterations"
              % (P["lin_tol_nvrs"], P["lin_tol_opm"], P["lin_maxit"]))
        print("    AMG hierarchy reused for up to %d further Newton iterations"
              % P["amg_reuse"])
        print("    -> %s" % os.path.basename(a))
        print("    -> %s  (%.1f MB)"
              % (os.path.basename(b), os.path.getsize(b) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
