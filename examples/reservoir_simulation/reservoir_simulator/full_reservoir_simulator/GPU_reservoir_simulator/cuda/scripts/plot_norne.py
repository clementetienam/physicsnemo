#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# plot_norne.py -- render a CORNER-POINT run as the field it is.
#
# Why this is separate from the SPE10 path
# ----------------------------------------
# A Cartesian model is a box, and drawing it as an array of voxels is not an
# approximation -- it is the geometry.  A corner-point model is not.  Its cells
# are hexahedra standing on pillars: the layers dip, the thickness varies, and
# a fault throw sets one column against another so that cells with adjacent
# (i, j, k) can be tens of metres apart in space.  Drawn as a regular array,
# Norne comes out as a rectangular slab -- correct in topology, wrong in every
# respect anyone looks at a field map for.  The faults vanish, the structure
# vanishes, and the closure that holds the oil vanishes with it.
#
# So this reads cgrid.bin, which the writer emits for a corner-point run: six
# float32 columns per active cell -- i, j, k (1-based) then the x, y, z of the
# cell centre.  Scattering those onto the index box gives one surface per
# layer, and plot_surface draws the real dipping, faulted structure.
#
# Four panels, as the reference figure has them:
#   voxel view with wells | side view (elev 30, azim 30)
#   top view (elev 90)    | depth-averaged map
#
# Wells stand proud of the reservoir so their names are readable against the
# structure rather than buried in it, and they are coloured by what they DO:
#
#   blue   water injector      red    gas injector      green  producer
#
# Three colours and not two, because a water injector and a gas injector do
# opposite things to a reservoir; on Norne the same well is often both at
# different times, and which one it is at the plotted date is the point.
#
#   python3 plot_norne.py OUTPUT_NORNE               # last step, all fields
#   python3 plot_norne.py OUTPUT_NORNE --step 40
#   python3 plot_norne.py OUTPUT_NORNE --field water
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm

# The four fields a black-oil snapshot carries, in the order io.cu writes them.
PLANES = ["pressure", "water", "oil", "gas"]
LABEL = {"pressure": "Pressure (psia)",
         "water": "Water saturation",
         "oil": "Oil saturation",
         "gas": "Gas saturation",
         "perm": "Log K (mD)",
         "porosity": "Porosity"}

WELL_STYLE = {"winj": ("blue", "water injector"),
              "ginj": ("red", "gas injector"),
              "prod": ("green", "oil/water/gas producer")}


def load_meta(outdir):
    with open(os.path.join(outdir, "meta.json")) as f:
        return json.load(f)


def load_plane(outdir, step, which, nbox):
    """One field of one snapshot, as a flat array over the full index box."""
    path = os.path.join(outdir, "state_%04d.bin" % step)
    a = np.fromfile(path, dtype=np.float32)
    if a.size < nbox * len(PLANES):
        raise SystemExit("%s holds %d floats, expected %d -- is this a "
                         "corner-point run written by the current writer?"
                         % (path, a.size, nbox * len(PLANES)))
    k = PLANES.index(which)
    return a[k * nbox:(k + 1) * nbox]


def load_rock(outdir, nbox, which):
    path = os.path.join(outdir, "rock.bin")
    a = np.fromfile(path, dtype=np.float32)
    k = 0 if which == "perm" else 1
    return a[k * nbox:(k + 1) * nbox]


def coordinate_cubes(outdir, meta):
    """Xcor, Ycor, Zcor on the index box, NaN where the grid is inactive.

    cgrid.bin is per ACTIVE cell; the plot is indexed by the box.  Everything
    the cells do not occupy stays NaN so it is simply not drawn."""
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    raw = np.fromfile(os.path.join(outdir, "cgrid.bin"), dtype=np.float32)
    if raw.size % 6:
        raise SystemExit("cgrid.bin is not a multiple of six columns")
    t = raw.reshape((-1, 6))
    X = np.full((nx, ny, nz), np.nan)
    Y = np.full((nx, ny, nz), np.nan)
    Z = np.full((nx, ny, nz), np.nan)
    i = t[:, 0].astype(int) - 1
    j = t[:, 1].astype(int) - 1
    k = t[:, 2].astype(int) - 1
    X[i, j, k] = t[:, 3]
    Y[i, j, k] = t[:, 4]
    Z[i, j, k] = t[:, 5]
    return X, Y, Z


def draw_wells(ax, wells, nz, z_top, fontsize, lw):
    """A stick per well, standing proud of the reservoir, labelled and typed.

    The stick runs from below the field to above it so the name sits clear of
    the structure; buried in it, the labels are unreadable on a model with
    thirty-six wells."""
    for w in wells:
        colour, _ = WELL_STYLE.get(w.get("type", "prod"), ("black", ""))
        x, y = w["i"] - 1, w["j"] - 1
        ax.plot([x, x], [y, y], [0, z_top], colour, linewidth=lw)
        ax.text(x, y, z_top, w.get("name", ""), color=colour,
                weight="bold", fontsize=fontsize)


def legend_handles():
    return [mlines.Line2D([], [], color=c, linewidth=2, label=lab)
            for c, lab in WELL_STYLE.values()]


def top_layer(Z):
    """Index of the layer with the shallowest cells.

    ZCORN is positive downwards and the writer negates it, so the top of the
    reservoir is the LARGEST z.  Which k that is depends on the deck, so it is
    measured rather than assumed to be zero."""
    means = [np.nanmean(Z[:, :, k]) if np.isfinite(Z[:, :, k]).any() else -np.inf
             for k in range(Z.shape[2])]
    return int(np.argmax(means))


def plot_field(outdir, meta, values, varii, step, t_days, out_png):
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    wells = meta.get("wells", [])

    # Fortran order: the writer's global index is i + nx*(j + ny*k).
    cube = np.reshape(values, (nx, ny, nz), order="F").astype(float)
    finite = np.isfinite(cube)
    if not finite.any():
        raise SystemExit("every cell is inactive -- nothing to draw")
    lo, hi = float(np.nanmin(cube)), float(np.nanmax(cube))
    span = (hi - lo) if hi > lo else 1.0
    norm = mpl.colors.Normalize(vmin=lo, vmax=hi)

    # Two rows of three: voxels and two structural angles on top,
    # the map view, the depth-averaged map, and room for the legend
    # below.  Three angles because one is never enough on a faulted
    # field -- a throw that is obvious along strike can be invisible
    # down dip, and which is which depends on the fault.
    fig = plt.figure(figsize=(24, 15), dpi=130)

    # ---- 1. voxels, with wells ---------------------------------------------
    ax = fig.add_subplot(231, projection="3d")
    shade = (cube - lo) / span
    colours = cm.jet(np.ma.masked_invalid(shade))
    colours[~finite, :3] = 1.0
    ax.voxels(finite, facecolors=colours, alpha=0.5, edgecolor="none",
              shade=True)
    m = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    m.set_array([])
    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.set_box_aspect([nx, ny, nz])
    ax.set_proj_type("ortho")
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo["tick"]["inward_factor"] = 0
        a._axinfo["tick"]["outward_factor"] = 0.4
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=30, azim=45)
    draw_wells(ax, wells, nz, nz * 2 + 2, 5, 1)
    ax.legend(handles=legend_handles(), loc="lower left", fontsize=9)
    cb = plt.colorbar(m, ax=ax, orientation="horizontal", shrink=0.5)
    cb.set_label(LABEL.get(varii, varii), fontsize=9)
    ax.set_title("%s with well locations" % LABEL.get(varii, varii),
                 fontsize=16, weight="bold")

    # ---- 2 & 3. the STRUCTURE -----------------------------------------------
    # The top of the reservoir, drawn at its real depth.
    #
    # Not every layer stacked: twenty-two opaque surfaces render as a solid
    # block and the top one hides the other twenty-one, which is how a plot
    # that has the true geometry still comes out looking like a box.  The top
    # surface alone is the structure map -- it carries the dip and every fault
    # throw, because a throw is a step in exactly this surface.
    #
    # Depth is exaggerated by default.  Norne spans about 1,950 ft vertically
    # across 27,000 ft of map, so at true scale the structure is a sixth of a
    # degree of tilt and invisible.  ZEXAG scales it; 1.0 is true scale.
    X, Y, Z = coordinate_cubes(outdir, meta)
    ktop = top_layer(Z)
    zex = float(os.environ.get("ZEXAG", "12"))
    Zt = Z[:, :, ktop]
    zmid = np.nanmean(Zt)
    Zs = zmid + (Zt - zmid) * zex
    Ct = cube[:, :, ktop]
    fl = (Ct - lo) / span
    cols = cm.jet(np.ma.masked_invalid(fl))
    cols[~np.isfinite(Ct), :3] = 1.0

    for pos, (elev, azim, name) in enumerate(
            [(30, 45, "structure, 30/45"),
             (28, -60, "structure, side view"),
             (90, -90, "structure, map view")]):
        axs = fig.add_subplot(232 + pos, projection="3d")
        axs.plot_surface(X[:, :, ktop], Y[:, :, ktop], Zs, facecolors=cols,
                         edgecolor="none", shade=True, rstride=1, cstride=1,
                         antialiased=False)
        axs.set_facecolor("white")
        for lim, arr in ((axs.set_xlim, X), (axs.set_ylim, Y)):
            lim(np.nanmin(arr), np.nanmax(arr))
        zlo, zhi = float(np.nanmin(Zs)), float(np.nanmax(Zs))
        if zhi > zlo:
            axs.set_zlim(zlo, zhi)
        axs.view_init(elev, azim)
        axs.grid(False)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        axs.set_xlabel("X"); axs.set_ylabel("Y")
        axs.set_zlabel("depth (x%g)" % zex)
        ztop = zhi + 0.25 * (zhi - zlo + 1.0)
        for w in wells:
            colour, _ = WELL_STYLE.get(w.get("type", "prod"), ("black", ""))
            i, j = w["i"] - 1, w["j"] - 1
            if not (0 <= i < nx and 0 <= j < ny):
                continue
            xw, yw, zw = X[i, j, ktop], Y[i, j, ktop], Zs[i, j]
            if not (np.isfinite(xw) and np.isfinite(yw) and np.isfinite(zw)):
                continue
            axs.plot([xw, xw], [yw, yw], [zw, ztop], colour, linewidth=1.8)
            axs.text(xw, yw, ztop, w.get("name", ""), color=colour,
                     weight="bold", fontsize=7)
        cb = fig.colorbar(m, ax=axs, orientation="horizontal", shrink=0.55)
        cb.set_label(LABEL.get(varii, varii), fontsize=9)
        axs.set_title("%s - %s" % (LABEL.get(varii, varii), name),
                      weight="bold", fontsize=14)

    # ---- 4. depth-averaged map ---------------------------------------------
    ax4 = fig.add_subplot(236)
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(cube, axis=2)
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    pc = ax4.pcolormesh(XX.T, YY.T, avg, cmap="jet", shading="auto")
    cb = plt.colorbar(pc, ax=ax4)
    cb.set_label(LABEL.get(varii, varii), fontsize=9)
    for w in wells:
        colour, _ = WELL_STYLE.get(w.get("type", "prod"), ("black", ""))
        ax4.plot(w["i"] - 1, w["j"] - 1, "o", color=colour, markersize=4)
        ax4.text(w["i"] - 1, w["j"] - 1, w.get("name", ""), color=colour,
                 fontsize=5, weight="bold")
    ax4.set_xlabel("X", fontsize=14); ax4.set_ylabel("Y", fontsize=14)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_title("Depth-averaged %s" % LABEL.get(varii, varii),
                  weight="bold", fontsize=16)

    title = meta.get("title", "field")
    fig.suptitle(r"%s  [$N_x=%d$, $N_y=%d$, $N_z=%d$]   step %d, %.0f days"
                 % (title, nx, ny, nz, step, t_days),
                 weight="bold", fontsize=15)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png



def _frame(job):
    """One structural frame: the top surface at its real depth, one report step.

    A separate figure from plot_field's four-panel sheet because an animation
    wants one view large, not four small ones, and because the frames have to
    share a colour scale -- rescaling per frame makes a still field look like
    it is boiling."""
    (outdir, varii, step, nbox, lo, hi, zex, out_png) = job
    meta = load_meta(outdir)
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    times = meta.get("times", [])
    t_days = times[step] if step < len(times) else 0.0
    vals = load_plane(outdir, step, varii, nbox)
    cube = np.reshape(vals, (nx, ny, nz), order="F").astype(float)

    X, Y, Z = coordinate_cubes(outdir, meta)
    ktop = top_layer(Z)
    Zt = Z[:, :, ktop]
    zmid = np.nanmean(Zt)
    Zs = zmid + (Zt - zmid) * zex
    Ct = cube[:, :, ktop]
    span = (hi - lo) if hi > lo else 1.0
    cols = cm.jet(np.ma.masked_invalid((Ct - lo) / span))
    cols[~np.isfinite(Ct), :3] = 1.0

    fig = plt.figure(figsize=(10, 7.5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X[:, :, ktop], Y[:, :, ktop], Zs, facecolors=cols,
                    edgecolor="none", shade=True, rstride=1, cstride=1,
                    antialiased=False)
    ax.set_xlim(np.nanmin(X), np.nanmax(X))
    ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
    zlo, zhi = float(np.nanmin(Zs)), float(np.nanmax(Zs))
    if zhi > zlo:
        ax.set_zlim(zlo, zhi)
    ax.view_init(28, -60)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_zlabel("depth (x%g)" % zex)

    ztop = zhi + 0.25 * (zhi - zlo + 1.0)
    for w in meta.get("wells", []):
        colour, _ = WELL_STYLE.get(w.get("type", "prod"), ("black", ""))
        i, j = w["i"] - 1, w["j"] - 1
        if not (0 <= i < nx and 0 <= j < ny):
            continue
        xw, yw, zw = X[i, j, ktop], Y[i, j, ktop], Zs[i, j]
        if not (np.isfinite(xw) and np.isfinite(yw) and np.isfinite(zw)):
            continue
        ax.plot([xw, xw], [yw, yw], [zw, ztop], colour, linewidth=1.5)

    m = cm.ScalarMappable(cmap=cm.jet,
                          norm=mpl.colors.Normalize(vmin=lo, vmax=hi))
    m.set_array([])
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", shrink=0.55, pad=0.02)
    cb.set_label(LABEL.get(varii, varii), fontsize=9)
    ax.set_title("%s   step %d,  %.0f days"
                 % (LABEL.get(varii, varii), step, t_days),
                 fontsize=13, weight="bold")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def structure_movie(outdir, meta, varii, workers=0, fps=3):
    """Animate the structure over the report steps.

    The existing evolution GIFs are flat maps of the index box.  On a field
    with 1,950 ft of relief and sixty-one faults, a flat map cannot show where
    the fluid is going -- the front follows the structure, and the structure is
    the thing that has been thrown away."""
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    nbox = nx * ny * nz
    nsnap = int(meta.get("nsnap", len(meta.get("times", []))))
    if nsnap < 2:
        return None
    try:
        from PIL import Image
    except Exception:
        print("  (no PIL: skipping %s animation)" % varii)
        return None

    # One colour scale for the whole sequence, from the first and last steps.
    lo, hi = np.inf, -np.inf
    for s in (0, nsnap - 1):
        v = load_plane(outdir, s, varii, nbox)
        f = v[np.isfinite(v)]
        if f.size:
            lo = min(lo, float(f.min())); hi = max(hi, float(f.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None

    zex = float(os.environ.get("ZEXAG", "12"))
    tmp = os.path.join(outdir, "_frames_%s" % varii)
    os.makedirs(tmp, exist_ok=True)
    # Cap the frame count: 111 report steps at this size is minutes of render
    # and a GIF nobody waits through.
    cap = int(os.environ.get("NVRS_MOVIE_FRAMES", "40"))
    steps = list(range(nsnap)) if nsnap <= cap else         sorted(set(np.linspace(0, nsnap - 1, cap).astype(int).tolist()))
    jobs = [(outdir, varii, s, nbox, lo, hi, zex,
             os.path.join(tmp, "f%04d.png" % s)) for s in steps]

    n = workers if workers > 0 else min(len(jobs), (os.cpu_count() or 4))
    if n > 1:
        with ProcessPoolExecutor(max_workers=n) as ex:
            frames = list(ex.map(_frame, jobs))
    else:
        frames = [_frame(j) for j in jobs]

    out = os.path.join(outdir, "norne_evolution_%s.gif" % varii)
    imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0, optimize=True)
    for f in frames:
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    return out


def _render(job):
    """One field, one figure.  Top level so it can be sent to a worker."""
    outdir, varii, step, nbox = job
    meta = load_meta(outdir)
    times = meta.get("times", [])
    t_days = times[step] if step < len(times) else 0.0
    if varii in ("perm", "porosity"):
        vals = load_rock(outdir, nbox, varii)
        if varii == "perm":
            vals = np.log10(np.maximum(vals, 1e-6))
        out = os.path.join(outdir, "norne_%s.png" % varii)
    else:
        vals = load_plane(outdir, step, varii, nbox)
        out = os.path.join(outdir, "norne_%s_%04d.png" % (varii, step))
    return plot_field(outdir, meta, vals, varii, step, t_days, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir")
    ap.add_argument("--step", type=int, default=-1,
                    help="report step to draw; -1 for the last")
    ap.add_argument("--field", default="all",
                    help="pressure|water|oil|gas|perm|porosity|all")
    ap.add_argument("--movie", action="store_true",
                    help="also animate the structure over the report steps")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel renderers; 0 = one per field, 1 = serial")
    args = ap.parse_args()

    meta = load_meta(args.outdir)
    if not meta.get("corner_point"):
        raise SystemExit(
            "%s is not a corner-point run -- meta.json has no \"corner_point\".\n"
            "A Cartesian model is a box and scripts/plot_results.py already "
            "draws it correctly; this renderer exists for the geometry that one "
            "cannot show." % args.outdir)

    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    nbox = nx * ny * nz
    times = meta.get("times", [])
    nsnap = meta.get("nsnap", len(times))
    step = args.step if args.step >= 0 else max(nsnap - 1, 0)
    t_days = times[step] if step < len(times) else 0.0

    fields = ([args.field] if args.field != "all"
              else PLANES + ["perm", "porosity"])

    # Each field is an independent figure over the same read-only inputs, and
    # the voxel pass is slow enough on 113,344 cells that doing six of them in
    # sequence is the whole runtime.  Render them at once.  Processes and not
    # threads: matplotlib's state is per-process and the work is CPU-bound, so
    # threads would serialise on the GIL and share a figure manager besides.
    jobs = [(args.outdir, varii, step, nbox) for varii in fields]
    workers = args.workers if args.workers > 0 else len(jobs)
    if workers <= 1 or len(jobs) == 1:
        for j in jobs:
            print("  wrote %s" % _render(j))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            for out in ex.map(_render, jobs):
                print("  wrote %s" % out)

    # The animation runs after the stills, whichever way those were rendered.
    # It used to sit only on the parallel branch, so asking for a single field
    # -- which takes the serial path -- returned before ever reaching it.
    if args.movie:
        for varii in [f for f in fields if f in PLANES]:
            g = structure_movie(args.outdir, meta, varii, args.workers)
            if g:
                print("  wrote %s" % g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
