#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""spe10_rock_fig.py -- SPE10 model 2 permeability and porosity on the native
60 x 220 x 85 grid (no coarsening), permeability on a log10 scale.

    python3 spe10_rock_fig.py <SPE10 INIT file> [output folder]

The three visible faces of the block are drawn cell by cell.  Writes
spe10_perm.png and spe10_poro.png.  compare.py uses the same drawing for the
Rock section of the SPE10 report.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

NX, NY, NZ = 60, 220, 85
DX, DY, DZ = 20.0, 10.0, 2.0          # ft
VEXAG = 4.0                           # vertical exaggeration of the drawing
WELLS = [("INJ", 30, 110, "blue"), ("P1", 0, 0, "green"), ("P2", NX - 1, 0, "green"),
         ("P3", NX - 1, NY - 1, "green"), ("P4", 0, NY - 1, "green")]


def pad(a):
    """Face values (n, m) -> vertex-shaped (n+1, m+1) for plot_surface."""
    return np.pad(a, ((0, 1), (0, 1)), mode="edge")


def draw(vals, norm, cmap, label, path, ticks=None, ticklabels=None):
    cm = plt.get_cmap(cmap)
    x = np.arange(NX + 1) * DX
    y = np.arange(NY + 1) * DY
    z = -np.arange(NZ + 1) * DZ * VEXAG
    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    kw = dict(rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)
    # top, k = 0
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax.plot_surface(X, Y, np.zeros_like(X), facecolors=cm(norm(pad(vals[0].T))), **kw)
    # front, j = 0
    X, Z = np.meshgrid(x, z, indexing="ij")
    ax.plot_surface(X, np.zeros_like(X), Z, facecolors=cm(norm(pad(vals[:, 0, :].T))), **kw)
    # side, i = NX - 1
    Y, Z = np.meshgrid(y, z, indexing="ij")
    ax.plot_surface(np.full_like(Y, x[-1]), Y, Z, facecolors=cm(norm(pad(vals[:, :, -1].T))), **kw)
    for name, i, j, c in WELLS:
        xi, yj = (i + 0.5) * DX, (j + 0.5) * DY
        ax.plot([xi, xi], [yj, yj], [z[-1], 160], color=c, lw=2.5)
        ax.text(xi, yj, 180, name, color=c, fontsize=10, weight="bold")
    ax.set_box_aspect((NX * DX, NY * DY, NZ * DZ * VEXAG))
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x (i), ft")
    ax.set_ylabel("y (j), ft")
    ax.set_zticks([])
    ax.set_title(label, fontsize=13, weight="bold")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.04)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklabels)
    cb.set_label(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def as_nz_ny_nx(field):
    """The report holds fields as (nx, ny, nz); the drawing wants (nz, ny, nx)."""
    a = np.asarray(field, dtype=float)
    if a.shape == (NX, NY, NZ):
        a = a.T
    if a.shape != (NZ, NY, NX):
        raise ValueError("not an SPE10 model 2 field: shape %s" % (a.shape,))
    return a


def draw_perm(perm, path):
    perm = as_nz_ny_nx(perm)
    kmin = max(float(np.nanmin(perm[perm > 0])), 1e-3)
    kmax = float(np.nanmax(perm))
    dec = np.arange(np.floor(np.log10(kmin)), np.ceil(np.log10(kmax)) + 1)
    draw(np.clip(np.nan_to_num(perm, nan=kmin), kmin, kmax), LogNorm(kmin, kmax), "jet",
         "Permeability, mD (log$_{10}$ scale)", path,
         ticks=10.0 ** dec, ticklabels=["$10^{%d}$" % d for d in dec])
    return kmin, kmax


def draw_poro(poro, path):
    poro = as_nz_ny_nx(poro)
    lo, hi = np.nanpercentile(poro[poro > 1e-3], [1, 99])
    draw(np.nan_to_num(poro, nan=lo), Normalize(lo, hi), "jet", "Porosity", path)
    return lo, hi


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    import ecl

    def field(kw):
        for k, _t, v in ecl.read_unformatted(sys.argv[1]):
            if k == kw:
                return np.asarray(v, dtype=float).reshape((NZ, NY, NX))
        raise KeyError(kw)

    out = sys.argv[2] if len(sys.argv) > 2 else "."
    kmin, kmax = draw_perm(field("PERMX"), os.path.join(out, "spe10_perm.png"))
    lo, hi = draw_poro(field("PORO"), os.path.join(out, "spe10_poro.png"))
    print("perm %.3g-%.3g mD, poro 1-99%% %.3f-%.3f" % (kmin, kmax, lo, hi))


if __name__ == "__main__":
    main()
