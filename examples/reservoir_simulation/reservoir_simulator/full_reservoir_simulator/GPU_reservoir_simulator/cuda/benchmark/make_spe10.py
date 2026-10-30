#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# make_spe10.py — the SPE10 model 2 benchmark on NVRS-CUDA.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# ---------------------------------------------------------------------------
# The case
# ---------------------------------------------------------------------------
# The Tenth SPE Comparative Solution Project, model 2 (Christie & Blunt,
# SPE 72469, 2001): 60 x 220 x 85 = 1,122,000 cells at 20 x 10 x 2 ft, the
# geostatistical Brent-sequence permeability field with a ~10^7 contrast,
# two-phase oil/water, a central water injector and four corner producers,
# 2000 days.  It is the standard stress test for algebraic multigrid on
# reservoir pressure systems, which is exactly what the backend study of
# scripts/backend_study.sh measures.
#
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# The rock arrays are the public-domain dataset published by Christie & Blunt
# (https://www.spe.org/web/csp/datasets/set02.htm), in the ECLIPSE include
# form redistributed by the Open Porous Media initiative under the Open
# Database License (github.com/OPM/opm-data, spe10model2/).  The three files
#
#     SPE10MODEL2_PERM.INC   (PERMX, PERMY, PERMZ; 57.9 MB)
#     SPE10MODEL2_PHI.INC    (PORO; 19.3 MB)
#     SPE10_MODEL2.DATA      (the reference ECLIPSE deck, for provenance)
#
# are DOWNLOADED ON FIRST USE into cuda/dataset/SPE10/ (git-ignored) rather
# than committed: 77 MB of third-party public data does not belong in this
# repository when one URL reproduces it bit-for-bit.  A checkout that already
# has the files there, put by hand from any mirror, is used as-is and nothing
# is fetched.
#
# ---------------------------------------------------------------------------
# What is faithful and what is not
# ---------------------------------------------------------------------------
# Faithful to the spec:
#   * the grid, cell sizes and the full kx and per-cell kz fields (NVRS reads
#     kz through the PERMZFILE keyword; ky = kx holds in the dataset and is
#     verified below before the deck is written);
#   * the independent porosity field (POROFILE), floored at POROMIN because
#     the published file replaces zero porosity with 1e-7 to avoid dead
#     cells, and an accumulation term of 1e-7 conditions the matrix without
#     representing any storage;
#   * the relative permeabilities: the spec's quadratic Corey curves with
#     Swc = Sor = 0.2 and endpoint 1.0 are ANALYTICALLY identical to NVRS's
#     two-phase kernel (include/fluid.hpp relperm2: krw = S^2,
#     kro = (1-S)^2, S = (Sw-SWI)/(1-SWI-SWR)) at SWI = SWR = 0.2;
#   * viscosities (0.3 / 3.0 cP), densities (64 / 53 lb/ft3), the initial
#     state (6000 psia, connate water: the OWC of the reference deck sits at
#     the base of the grid), the well pattern, rates and BHPs, and the
#     2000-day horizon.
#
# Deliberate simplifications, shared identically by both solver backends:
#   * dead oil with constant B (PVTCONST 1, B = 1.01, zero compressibility),
#     where the spec's PVDO carries ~1e-6/psi.  NVRS's accumulation is
#     incompressible by construction; this is the same simplification every
#     other NVRS benchmark deck makes and it does not touch the pressure
#     system's off-diagonal structure, which is what the multigrid sees.
#   * The injector is surface-rate controlled at 5000 STB/D where the
#     reference deck uses 5000 RB/D reservoir-voidage control; at B_w = 1.01
#     the two differ by 1%.
#   * NVRS report steps are 100 days with its own adaptive substepping,
#     rather than the reference deck's fixed 1..100-day TSTEP ramp; the
#     chop-hint controller discovers an equivalent ramp on the first report
#     step and MAXCHOP 8 lets it reach 0.39-day steps if the field needs
#     them.
#
#   python3 make_spe10.py [--out DIR] [--tend DAYS]
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(CUDA, "dataset", "SPE10")

NX, NY, NZ = 60, 220, 85
NCELL = NX * NY * NZ
DX, DY, DZ = 20.0, 10.0, 2.0

OPM_DATA_RAW = "https://raw.githubusercontent.com/OPM/opm-data/master/spe10model2/"
FILES = ["SPE10MODEL2_PERM.INC", "SPE10MODEL2_PHI.INC", "SPE10_MODEL2.DATA"]


def fetch(name):
    """The include file from cuda/dataset/SPE10/, downloading it if missing.

    The download goes to a temporary name and is renamed only when complete,
    so an interrupted fetch cannot leave a truncated file that a later run
    would silently accept as the dataset.
    """
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        return path
    os.makedirs(DATA, exist_ok=True)
    url = OPM_DATA_RAW + name
    print("[spe10] downloading %s" % url)
    import urllib.request
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def read_arrays(path, wanted):
    """{keyword: np.array} for the ECLIPSE array keywords in `wanted`.

    Handles comments (--), run-length tokens (N*V) and the closing slash.
    A pure-alphabetic token opens a new keyword; anything the caller did not
    ask for is skipped without being stored.
    """
    out = {}
    cur = None
    vals = []

    def flush():
        nonlocal cur, vals
        if cur in wanted:
            out[cur] = np.asarray(vals, dtype=np.float64)
        cur, vals = None, []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.split("--")[0]
            for tok in line.replace("/", " / ").split():
                if tok == "/":
                    flush()
                elif tok.replace("_", "").isalpha():
                    flush()
                    cur = tok
                elif cur in wanted:
                    if "*" in tok:
                        n, v = tok.split("*")
                        vals.extend([float(v)] * int(n))
                    else:
                        vals.append(float(tok))
    flush()
    missing = [k for k in wanted if k not in out]
    if missing:
        raise SystemExit("%s: keywords not found: %s" % (path, missing))
    return out


def load_spe10():
    perm = read_arrays(fetch("SPE10MODEL2_PERM.INC"),
                       ("PERMX", "PERMY", "PERMZ"))
    phi = read_arrays(fetch("SPE10MODEL2_PHI.INC"), ("PORO",))
    fetch("SPE10_MODEL2.DATA")          # provenance only
    for k, a in list(perm.items()) + list(phi.items()):
        if a.size != NCELL:
            raise SystemExit("%s has %d values, expected %d"
                             % (k, a.size, NCELL))
    # NVRS assumes Ky = Kx.  The dataset satisfies it; verify rather than
    # trust, because a violated assumption here would silently change the
    # problem being benchmarked.
    kx, ky = perm["PERMX"], perm["PERMY"]
    rel = np.max(np.abs(kx - ky) / np.maximum(np.abs(kx), 1e-30))
    if rel > 1e-6:
        raise SystemExit("PERMY differs from PERMX (max rel diff %.3g); "
                         "NVRS has no independent Ky" % rel)
    return kx, perm["PERMZ"], phi["PORO"]


def write_deck(outdir, tend):
    kx, kz, phi = load_spe10()
    os.makedirs(outdir, exist_ok=True)

    def save(name, arr):
        p = os.path.join(outdir, name)
        np.savetxt(p, arr, fmt="%.8g")
        try:
            return os.path.relpath(p, CUDA).replace("\\", "/")
        except ValueError:
            return p.replace("\\", "/")

    permref = save("SPE10_perm.txt", kx)
    permzref = save("SPE10_permz.txt", kz)
    pororef = save("SPE10_poro.txt", phi)

    g = 62.428                       # lb/ft3 per g/cc
    rho_w, rho_o = 64.0 / g, 53.0 / g

    L = []
    A = L.append
    A("-- " + "=" * 69)
    A("--  SPE10 — Tenth SPE Comparative Solution Project, model 2")
    A("--  Christie & Blunt, SPE 72469 (2001).  60 x 220 x 85 = 1,122,000")
    A("--  cells, two-phase oil/water, central injector, corner producers.")
    A("--")
    A("--  GENERATED by benchmark/make_spe10.py from the public dataset;")
    A("--  do not edit by hand, re-run the generator instead.  The header")
    A("--  of make_spe10.py records exactly what is faithful to the spec")
    A("--  and what is simplified.")
    A("-- " + "=" * 69)
    A("")
    A("TITLE    SPE10 MODEL 2 60x220x85")
    A("")
    A("NCOMP    2")
    A("MODEL    BLACKOIL")
    A("OUTDIR   OUTPUT_SPE10")
    A("CASENAME SPE10")
    A("")
    A("-- ---- GRID ----------------------------------------------------------")
    A("NX       %d" % NX)
    A("NY       %d" % NY)
    A("NZ       %d" % NZ)
    A("DX       %g" % DX)
    A("DY       %g" % DY)
    A("DZ       %g" % DZ)
    A("PERMFILE  %s" % permref)
    A("PERMZFILE %s          -- per-cell Kz: SPE10's kv/kh is not a scalar"
      % permzref)
    A("POROFILE  %s          -- independent measured porosity" % pororef)
    A("POROMIN  1e-4          -- floor under the file's 1e-7 dead-cell stand-in")
    A("")
    A("-- ---- PROPS ---------------------------------------------------------")
    A("PVTCONST 1             -- dead oil, constant B; see the generator header")
    A("UW       0.3")
    A("UO       3.0")
    A("BW       1.01")
    A("BO       1.01")
    A("RS       0")
    A("RHOW     %.6f" % rho_w)
    A("RHOO     %.6f" % rho_o)
    A("CFW      0.0")
    A("CFO      0.0")
    A("CT       0.0")
    A("SWI      0.2           -- with SWR, makes relperm2 the spec's exact curves")
    A("SWR      0.2")
    A("PINIT    6000")
    A("PBUB     6000")
    A("PATM     14.6959")
    A("SWINIT   0.2           -- connate water: the reference OWC is at the base")
    A("SGINIT   0.0")
    A("SOINIT   0.8")
    A("GRAVITY  1")
    A("DARCY    1.127e-3")
    A("REFACTOR 0.2")
    A("")
    A("-- Two-phase: the NVRS kernel uses analytic Corey curves")
    A("--     krw = S^2,  kro = (1-S)^2,  S = (Sw-SWI)/(1-SWI-SWR)")
    A("-- which at SWI = SWR = 0.2 are exactly the SPE10 model 2 relative")
    A("-- permeabilities, so no SWOF table is needed or read.")
    A("")
    A("-- ---- SOLVER --------------------------------------------------------")
    A("KRYLOV   FGMRES")
    A("LINTOL   1e-2")
    A("LINMAXIT 200")
    A("RESTART  30")
    A("AMGFAST  1")
    A("AMGREUSE 10")
    A("AMGREUSEGROWTH 2.0")
    A("MIXED    1")
    A("EMULATE  1")
    A("JACOBIAN ANALYTIC")
    A("")
    A("-- ---- TIME ----------------------------------------------------------")
    A("DTREPORT   100")
    A("TEND       %g" % tend)
    A("DTGROW     2")
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
    A("DPMAX      500")
    A("DSMAX      0.2")
    A("MAXCHOP    8           -- the 10^7 contrast may want 0.39-day steps early")
    A("")
    A("-- ---- OUTPUT --------------------------------------------------------")
    A("MAKECSV      1")
    A("MAKESUMMARY  1")
    A("ECLOUT       1         -- field snapshots; scripts/spe10_report.py")
    A("                       --    reads these to render spe10.html")
    A("AUTOPLOT     0")
    A("")
    A("-- ---- WELLS:  I  J  TYPE  RATE  RW  SKIN  PWF  PERF  NAME ------------")
    A("-- The reference pattern: INJ at (30,110) rate-controlled with a")
    A("-- 10,000 psi BHP cap, producers at the four corners at 4000 psi BHP,")
    A("-- all perforated through every layer.  rw = 1.0 ft is the reference")
    A("-- deck's 2 ft wellbore diameter.")
    A("WELLS")
    A("  30 110   1   5000   1.0   0.0  10000   all INJ")
    A("   1   1  -1     -1   1.0   0.0   4000   all P1")
    A("  60   1  -1     -1   1.0   0.0   4000   all P2")
    A("  60 220  -1     -1   1.0   0.0   4000   all P3")
    A("   1 220  -1     -1   1.0   0.0   4000   all P4")
    A("/")

    path = os.path.join(outdir, "SPE10.DATA")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")

    # --- selfcheck: the numbers a reader would otherwise have to trust -----
    pv = phi.sum() * DX * DY * DZ / 5.61458
    inj = 5000.0 * tend
    print("[spe10] kx    %.4g - %.4g mD (mean %.4g)"
          % (kx.min(), kx.max(), kx.mean()))
    print("[spe10] kz    %.4g - %.4g mD (mean %.4g)"
          % (kz.min(), kz.max(), kz.mean()))
    print("[spe10] kx contrast %.3g, ky == kx verified"
          % (kx.max() / max(kx.min(), 1e-30)))
    print("[spe10] poro  %.3g - %.3g (mean %.4g; %.1f%% of cells at the "
          "1e-4 floor)" % (phi.min(), phi.max(), phi.mean(),
                           100.0 * np.mean(phi < 1e-4)))
    print("[spe10] pore volume %.4g bbl; injection %.4g bbl over %g days "
          "(%.1f%% PV)" % (pv, inj, tend, 100.0 * inj / pv))
    print("[spe10] -> %s" % path)
    return path


def main(argv):
    args = list(argv[1:])
    outdir, tend = HERE, 2000.0
    while args:
        a = args.pop(0)
        if a == "--out":
            outdir = args.pop(0)
        elif a == "--tend":
            tend = float(args.pop(0))
        else:
            raise SystemExit("usage: make_spe10.py [--out DIR] [--tend DAYS]")
    write_deck(outdir, tend)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
