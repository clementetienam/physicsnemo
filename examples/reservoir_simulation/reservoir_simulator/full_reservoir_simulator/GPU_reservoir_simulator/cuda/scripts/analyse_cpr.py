#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
"""Assemble the CPR pressure operator on the host and inspect it.

Why this exists
---------------
When the linear solve stalls, the question is whether the Krylov method is
struggling with a hard but correct matrix, or whether the matrix handed to the
preconditioner is not the one intended.  Those look identical from the outside:
a relative residual that stops falling.

Classical algebraic multigrid -- the stage-1 solver inside CPR -- rests on
assumptions about the operator it coarsens.  It expects something close to an
M-matrix: a positive diagonal, off-diagonals of the opposite sign, and rows
that are at least weakly diagonally dominant.  Where those hold, PMIS
coarsening and the interpolation built on it work.  Where they do not, AMG can
converge arbitrarily slowly on a matrix that is perfectly well posed, and no
amount of tightening the tolerance or adding iterations helps.

So this rebuilds the pressure operator from the same connection list and the
same initial state the simulator uses, and measures those properties directly.
It is host-side NumPy and independent of the CUDA path, which is the point: if
the two disagree, that is itself the answer.

    python analyse_cpr.py <field-deck.DATA> --deck <run-deck.DATA>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpgrid_ref as R  # noqa: E402


# ---------------------------------------------------------------------------
# Relative permeability, matching fluid.hpp's table interpolation.
# ---------------------------------------------------------------------------
def read_sat_tables(run_deck):
    """SWOF and SGOF from a run deck: (Sw, krow, krw) and (Sg, krog, krg)."""
    swof, sgof, cur = [], [], None
    with open(run_deck, errors="replace") as f:
        for raw in f:
            line = raw.split("--")[0].strip()
            if not line:
                continue
            if line == "/":
                cur = None
                continue
            head = line.split()[0]
            if head in ("SWOF", "SGOF"):
                cur = head
                continue
            if cur is None:
                continue
            tk = line.rstrip("/").split()
            if len(tk) >= 3:
                try:
                    row = [float(tk[0]), float(tk[1]), float(tk[2])]
                except ValueError:
                    cur = None
                    continue
                (swof if cur == "SWOF" else sgof).append(row)
    return np.array(swof), np.array(sgof)


def scalars_from_deck(run_deck):
    out = {}
    with open(run_deck, errors="replace") as f:
        for raw in f:
            line = raw.split("--")[0].strip()
            tk = line.split()
            if len(tk) >= 2:
                try:
                    out[tk[0]] = float(tk[1])
                except ValueError:
                    pass
    return out


def interp(x, xp, fp):
    return np.interp(x, xp, fp)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("field_deck")
    ap.add_argument("--deck", required=True, help="the NVRS run deck")
    args = ap.parse_args()

    print("CPR pressure-operator analysis")
    print("==============================")

    # ---- grid + connections -------------------------------------------------
    res, arr = R.build(args.field_deck, verbose=False)
    a, b, t = arr["conn_a"], arr["conn_b"], arr["conn_t"]
    # The same negligible-connection filter cpgrid applies, so the operator
    # measured here is the operator the solver is handed.
    keep = t > 1e-6 * t.max()
    a, b, t = a[keep], b[keep], t[keep]
    depth_m, porv_m, vol_m = arr["depth"], arr["porv"], arr["volume"]
    n = res["n_active"]
    print(f"  cells {n}, connections {t.size}")
    print(f"  trans {t.min():.4g} .. {t.max():.4g}  "
          f"({np.log10(t.max()/t.min()):.1f} decades)")

    # ---- to field units, as the solver sees them ---------------------------
    M3_BBL, M_FT, BAR_PSI = 6.2898107704, 3.280839895, 14.503773773
    t = t * (M3_BBL / BAR_PSI)
    depth = depth_m * M_FT
    porv = porv_m * M3_BBL
    vol = vol_m * M3_BBL

    D = scalars_from_deck(args.deck)
    swof, sgof = read_sat_tables(args.deck)
    print(f"  SWOF {len(swof)} rows, SGOF {len(sgof)} rows")

    uw, uo, ug = D.get("UW", 1.0), D.get("UO", 2.5), D.get("UG", 0.02)
    bw, bo, bg = D.get("BW", 1.0), D.get("BO", 1.2), D.get("BG", 0.005)
    rhow, rhoo, rhog = D.get("RHOW", 1.0), D.get("RHOO", 0.8), D.get("RHOG", 0.01)
    gg = D.get("GRAVGRAD", 0.4335)
    rs = D.get("RS", 500.0)
    dt = D.get("DTMAX", D.get("DTREPORT", 30.0))

    # ---- initial state ------------------------------------------------------
    grad = (D.get("GRADOIL", 0.0706), D.get("GRADWATER", 0.1001),
            D.get("GRADGAS", 0.0186))
    swc = D.get("SWCONNATE", 0.1)
    deck = R.Deck()
    deck.read(args.field_deck)
    # The active set after ACTNUM and MINPV, so EQLNUM lines up with `depth`.
    shp = (deck.nz, deck.ny, deck.nx)
    act = deck.arrays["ACTNUM"].reshape(-1) != 0
    active_idx = np.nonzero(act)[0]
    if active_idx.size != depth_m.size:
        active_idx = active_idx[:depth_m.size]
    eq = _equil(args.field_deck, deck, grad, swc, depth_m, active_idx)
    p = eq["p"] * BAR_PSI
    Sw, So, Sg = eq["sw"], eq["so"], eq["sg"]
    print(f"  pressure {p.min():.1f} .. {p.max():.1f} psi, "
          f"Sw {Sw.min():.3f} .. {Sw.max():.3f}")

    # ---- mobilities ---------------------------------------------------------
    krw = interp(Sw, swof[:, 0], swof[:, 2])
    krow = interp(Sw, swof[:, 0], swof[:, 1])
    krg = interp(Sg, sgof[:, 0], sgof[:, 2])
    krog = interp(Sg, sgof[:, 0], sgof[:, 1])
    # Baker-style product for three-phase oil, as fluid.hpp does.
    kro = krow * krog
    Mw = krw / (uw * bw)
    Mo = kro / (uo * bo)
    Mg = krg / (ug * bg)
    print(f"  mobility Mw {Mw.min():.3g}..{Mw.max():.3g}  "
          f"Mo {Mo.min():.3g}..{Mo.max():.3g}  Mg {Mg.min():.3g}..{Mg.max():.3g}")
    print(f"  cells with Mo == 0 : {int((Mo <= 0).sum())} of {n}")
    print(f"  cells with Mw == 0 : {int((Mw <= 0).sum())} of {n}")
    print(f"  cells with Mg == 0 : {int((Mg <= 0).sum())} of {n}")

    # ---- self blocks --------------------------------------------------------
    acc = vol * (porv / np.where(vol > 0, vol, 1)) / dt   # V*phi/dt
    acc = porv / dt
    ibo, ibw, ibg = 1.0 / bo, 1.0 / bw, 1.0 / bg

    sJ = np.zeros((n, 3, 3))
    sJ[:, 0, 1] += -acc * ibo
    sJ[:, 0, 2] += -acc * ibo
    sJ[:, 1, 1] += acc * ibw
    sJ[:, 2, 1] += -acc * rs * ibo
    sJ[:, 2, 2] += acc * (ibg - rs * ibo)

    # Total compressibility: the pressure derivative of the accumulation, and
    # the only thing standing between this operator and a singular Neumann
    # Laplacian when the formation volume factors are constants.
    ct = D.get("CT", 0.0)
    sJ[:, 0, 0] += acc * ibo * So * ct
    sJ[:, 1, 0] += acc * ibw * Sw * ct
    sJ[:, 2, 0] += acc * (ibg * Sg + rs * ibo * So) * ct
    print(f"  total compressibility CT = {ct:g} /psi")

    dz = depth[b] - depth[a]
    dp = p[b] - p[a]
    dPw = dp - gg * rhow * dz
    dPo = dp - gg * rhoo * dz
    dPg = dp - gg * rhog * dz
    up_w = np.where(dPw > 0, b, a)
    up_o = np.where(dPo > 0, b, a)
    up_g = np.where(dPg > 0, b, a)
    MwU, MoU, MgU = Mw[up_w], Mo[up_o], Mg[up_g]

    # Pressure derivatives only -- that is all the pressure block needs.
    # dR_oil/dp_i = + bo*T*Mo, dR_oil/dp_j = - bo*T*Mo, and so on.
    d_oil = ibo * t * MoU
    d_wat = ibw * t * MwU
    d_gas = ibg * t * MgU + rs * ibo * t * MoU

    np.add.at(sJ[:, 0, 0], a, d_oil); np.add.at(sJ[:, 0, 0], b, d_oil)
    np.add.at(sJ[:, 1, 0], a, d_wat); np.add.at(sJ[:, 1, 0], b, d_wat)
    np.add.at(sJ[:, 2, 0], a, d_gas); np.add.at(sJ[:, 2, 0], b, d_gas)

    # ---- CPR weights: first row of the inverse of the self block -----------
    det = np.linalg.det(sJ)
    n_sing = int((np.abs(det) < 1e-300).sum())
    print(f"\n  self blocks with |det| ~ 0 : {n_sing}")
    inv = np.zeros_like(sJ)
    ok = np.abs(det) > 0
    inv[ok] = np.linalg.inv(sJ[ok])
    w = np.where(ok[:, None], inv[:, 0, :], 1.0)     # fallback weights = 1

    # ---- the pressure operator ---------------------------------------------
    # diagonal: w . [dR_a/dp_i]
    diag = w[:, 0] * sJ[:, 0, 0] + w[:, 1] * sJ[:, 1, 0] + w[:, 2] * sJ[:, 2, 0]
    # off-diagonal for connection (a,b), as seen from a and from b
    off_a = -(w[a, 0] * d_oil + w[a, 1] * d_wat + w[a, 2] * d_gas)
    off_b = -(w[b, 0] * d_oil + w[b, 1] * d_wat + w[b, 2] * d_gas)

    print("\n  pressure operator")
    print(f"    diagonal      : {diag.min():.4g} .. {diag.max():.4g}")
    print(f"    non-positive  : {int((diag <= 0).sum())} of {n}")
    npos_off = int((off_a > 0).sum() + (off_b > 0).sum())
    print(f"    off-diagonals : {off_a.min():.4g} .. {off_a.max():.4g}")
    print(f"    POSITIVE off-diagonals (M-matrix violation): "
          f"{npos_off} of {2*t.size}  ({100.0*npos_off/(2*t.size):.1f}%)")

    rowsum = np.zeros(n)
    np.add.at(rowsum, a, np.abs(off_a))
    np.add.at(rowsum, b, np.abs(off_b))
    dom = np.abs(diag) - rowsum
    print(f"    rows NOT diagonally dominant : "
          f"{int((dom < 0).sum())} of {n}  ({100.0*(dom<0).mean():.1f}%)")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rowsum > 0, np.abs(diag) / rowsum, np.inf)
    print(f"    |diag|/sum|offdiag| : min {np.nanmin(ratio):.4g}, "
          f"median {np.nanmedian(ratio):.4g}")

    # A weighted sum of rows with mixed signs is the classic way a quasi-IMPES
    # pressure operator stops being an M-matrix.
    print(f"\n  CPR weights w = first row of inv(self block)")
    print(f"    w0 {w[:,0].min():.4g} .. {w[:,0].max():.4g}")
    print(f"    w1 {w[:,1].min():.4g} .. {w[:,1].max():.4g}")
    print(f"    w2 {w[:,2].min():.4g} .. {w[:,2].max():.4g}")
    mixed = int(((w[:, 0] < 0) | (w[:, 1] < 0) | (w[:, 2] < 0)).sum())
    print(f"    cells with a NEGATIVE weight : {mixed} of {n}")


def gather_text(path, seen=None, depth=0):
    """The deck and every file it INCLUDEs, comments stripped.

    EQUIL sits in the SOLUTION section, which the grid reader stops before, and
    in an included file besides.  This walks the tree independently.
    """
    import os, re
    if seen is None:
        seen = set()
    path = os.path.abspath(path)
    if path in seen or depth > 8 or not os.path.exists(path):
        return ""
    seen.add(path)
    raw = re.sub(r"--[^\n]*", "", open(path, errors="replace").read())
    parts = [raw]
    d = os.path.dirname(path)
    for m in re.finditer(r"INCLUDE\s+'([^']+)'", raw):
        inc = m.group(1)
        if inc.startswith("./"):
            inc = inc[2:]
        parts.append(gather_text(os.path.join(d, inc), seen, depth + 1))
    return "\n".join(parts)


def read_equil(field_deck):
    """EQUIL records: datum depth, datum pressure, OWC, Pc, GOC, Pc."""
    import re
    txt = gather_text(field_deck)
    m = re.search(r"^[ 	]*EQUIL[ 	]*$", txt, re.M)
    if not m:
        return []
    recs = []
    for line in txt[m.end():].splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "/":
            break
        tk = line.rstrip("/").split()
        vals = []
        for x in tk[:6]:
            try:
                vals.append(float(x))
            except ValueError:
                break
        if len(vals) < 6:
            break
        recs.append(vals)
        if len(recs) >= 50:
            break
    return recs


def _equil(field_deck, deck, grad, swc_default, depth, active_idx):
    """The same three-leg equilibration field_model performs."""
    go, gw, gg_ = grad
    recs = read_equil(field_deck)
    if not recs:
        raise SystemExit("no EQUIL records found")
    print(f"  EQUIL regions: {len(recs)}")

    eqlnum = deck.arrays.get("EQLNUM")
    if eqlnum is not None:
        reg = np.clip(eqlnum[active_idx].astype(int) - 1, 0, len(recs) - 1)
    else:
        reg = np.zeros(depth.size, int)

    p = np.zeros(depth.size)
    sw = np.full(depth.size, swc_default)
    so = np.zeros(depth.size)
    sg = np.zeros(depth.size)
    for r, rec in enumerate(recs):
        zd, pd, owc, _pcw, goc, _pcg = rec
        sel = reg == r
        if not sel.any():
            continue
        z = depth[sel]
        po = pd + go * (z - zd)
        pw = (pd + go * (owc - zd)) + gw * (z - owc)
        pg = (pd + go * (goc - zd)) + gg_ * (z - goc)
        water = z >= owc
        gas = z < goc
        oil = ~water & ~gas
        p[sel] = np.where(water, pw, np.where(gas, pg, po))
        sw[sel] = np.where(water, 1.0, swc_default)
        so[sel] = np.where(oil, 1.0 - swc_default, 0.0)
        sg[sel] = np.where(gas, 1.0 - swc_default, 0.0)
    return {"p": p, "sw": sw, "so": so, "sg": sg}


_DECK_PATH = {}


def deck_path_of(_deck):
    return _DECK_PATH["path"]


if __name__ == "__main__":
    _DECK_PATH["path"] = sys.argv[1]
    main()
