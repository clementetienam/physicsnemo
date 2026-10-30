#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# plot_results.py — zero-configuration post-processor for NVRS-CUDA output.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# ---------------------------------------------------------------------------
# Design goals
# ---------------------------------------------------------------------------
# 1. AUTOMATIC.  Run it with no arguments and it finds the output directories
#    itself (anything containing a meta.json, searched from the CWD and from
#    the repository root), processes every one it finds, and writes a browsable
#    HTML report.  No paths, no flags, no configuration.
#
#        python3 plot_results.py                  # find and plot everything
#        python3 plot_results.py OUTPUT           # one specific directory
#        python3 plot_results.py --watch          # re-plot as the run advances
#
# 2. NEVER FATAL.  The simulator invokes this at the end of a run.  A plotting
#    problem must never turn a good simulation into a failed job, so every
#    renderer is individually guarded and the script exits 0 unless explicitly
#    asked to be strict (--strict).  A missing matplotlib degrades to a printed
#    text summary rather than a traceback.
#
# 3. COMPLETE.  Field maps per layer, 3-D voxel views with wells, orthogonal
#    slice panels, animated evolution (GIF + MP4), field averages, well rates,
#    water cut, cumulative production, solver convergence — everything the deck
#    produced.
#
# ---------------------------------------------------------------------------
# Input sources
# ---------------------------------------------------------------------------
#   NVRS native    meta.json + state_%04d.bin + RSM_FVM.csv   (written by io.cu)
#   ECLIPSE/OPM    CASE.EGRID + CASE.UNRST + CASE.SMSPEC/UNSMRY
#                  — reads this simulator's output *and* OPM Flow's, so the two
#                    can be plotted side by side
#   MATLAB/NumPy   .mat / .npy / .npz, 3-D (k,j,i) or 4-D (t,k,j,i)
#   ASCII          flat whitespace/comma separated, with --dims NX NY NZ
#
#   python3 plot_results.py                              # auto-discover
#   python3 plot_results.py OUTPUT                        # one directory
#   python3 plot_results.py --ecl OUTPUT/NVRS             # an ECLIPSE case
#   python3 plot_results.py --file p.mat --dims 40 40 5   # a raw array
# ---------------------------------------------------------------------------
import argparse
import glob
import json
import math
import os
import sys
import time

# --- Optional dependencies -------------------------------------------------
# numpy is required for anything numeric; matplotlib for rendering; PIL only
# for GIFs.  Each is probed once and its absence downgrades the output rather
# than aborting the run.
try:
    import numpy as np
except Exception as exc:                                    # pragma: no cover
    print(f"[plot] numpy unavailable ({exc}); nothing to do.")
    sys.exit(0)

HAVE_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")                # headless: no DISPLAY on compute nodes
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib.lines as mlines
except Exception as exc:                                    # pragma: no cover
    HAVE_MPL = False
    print(f"[plot] matplotlib unavailable ({exc}); writing text summary only.")

HAVE_PIL = True
try:
    from PIL import Image
except Exception:
    HAVE_PIL = False

# Sibling modules: 3-D/slice rendering, generic array loaders, ECLIPSE reader.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import viz3d
    import loaders
    HAVE_VIZ = True
except Exception as exc:                                    # pragma: no cover
    HAVE_VIZ = False
    print(f"[plot] 3-D/slice rendering unavailable ({exc})")

try:
    import ecl
    HAVE_ECL = True
except Exception:
    HAVE_ECL = False


# ===========================================================================
# Small utilities
# ===========================================================================
def progress_bar(total, done, length=20, icon="#", bar="-"):
    """The CuPy reference's ProgressBar, so a long render never looks hung."""
    frac = 0.0 if total <= 0 else float(done) / total
    filled = int(frac * length)
    return f"[{icon * filled}{bar * (length - filled)}] {frac * 100:5.1f}%"


def guarded(label):
    """Decorator: a renderer that fails logs one line and does not kill the run."""
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:
                print(f"[plot]   ! {label} skipped: {type(exc).__name__}: {exc}")
                return None
        return wrapper
    return deco


def human(x, unit=""):
    """Compact number formatting for report tables."""
    if x is None:
        return "-"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.2f} G{unit}"
    if ax >= 1e6:
        return f"{x/1e6:.2f} M{unit}"
    if ax >= 1e3:
        return f"{x/1e3:.2f} k{unit}"
    if ax >= 1 or x == 0:
        return f"{x:.3g} {unit}".strip()
    return f"{x:.4g} {unit}".strip()


def savefig(fig, path, dpi=120):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Discovery
# ===========================================================================
def discover_outdirs(explicit):
    """Return the list of output directories to process.

    An explicit argument always wins.  Otherwise search, in order, the current
    directory and the repository tree above this script, for directories that
    contain a meta.json.  Deduplicated, most-recently-modified first.
    """
    if explicit:
        return [d for d in explicit if os.path.isdir(d)]

    roots = [os.getcwd()]
    here = os.path.dirname(os.path.abspath(__file__))
    roots += [os.path.abspath(os.path.join(here, "..")),          # cuda/
              os.path.abspath(os.path.join(here, "..", ".."))]    # simulator/

    found = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        # Depth-limited walk: output dirs live near the top, and an unbounded
        # walk over a repository with build trees is needlessly slow.
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if depth >= 3:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "build", "__pycache__", ".cache")]
            if "meta.json" in filenames:
                real = os.path.realpath(dirpath)
                found.setdefault(real, os.path.getmtime(dirpath))
    return [d for d, _ in sorted(found.items(), key=lambda kv: -kv[1])]


def load_meta(d):
    with open(os.path.join(d, "meta.json")) as f:
        return json.load(f)


def read_state(path, ncell):
    """Read one snapshot: returns (P, Sw, So, Sg) or None if truncated."""
    a = np.fromfile(path, dtype=np.float32)
    if a.size < 4 * ncell:
        return None
    return (a[0 * ncell:1 * ncell], a[1 * ncell:2 * ncell],
            a[2 * ncell:3 * ncell], a[3 * ncell:4 * ncell])


def layer_of(vec, nx, ny, nz, k):
    """Extract layer k. C-order (k,j,i), matching the writer in io.cu."""
    return vec.reshape((nz, ny, nx))[k]


# ===========================================================================
# Renderers
# ===========================================================================
def finite_range(field, pad=1e-30):
    """Colour limits over the cells that HAVE a value.

    field.min() on an array holding NaN returns NaN, and NaN silently poisons
    every comparison downstream: the limits come back NaN, matplotlib falls
    back to 0-1, and a 2,900-3,500 psia field is painted uniformly at the top
    of the scale.  A corner-point grid puts NaN in 61% of the box, so this is
    the normal case and not an edge one."""
    a = np.asarray(field, dtype=float)
    f = a[np.isfinite(a)]
    if f.size == 0:
        return 0.0, 1.0
    lo, hi = float(f.min()), float(f.max())
    if hi - lo < pad:
        hi = lo + pad
    return lo, hi


# Well markers on a map view.
#
# Three kinds and two shapes: an upward triangle takes fluid OUT, a downward
# triangle puts it IN, and the colour says what.  Green producer, blue water
# injector, red gas injector.  The previous code tested type == "inj", which no
# longer matches anything -- the writer distinguishes water from gas injection
# -- so every injector silently vanished from the maps and only producers were
# drawn.  Names go on the markers because on a field with thirty-six wells a
# legend cannot tell you which one is flooding.
WELL_MARK = {"prod": ("green", "^", "producer"),
             "winj": ("blue", "v", "water injector"),
             "ginj": ("red", "v", "gas injector")}


def well_kind(w):
    t = str(w.get("type", "prod")).lower()
    if t in WELL_MARK:
        return t
    # Sidecars written before the split only said "inj"; those are water.
    return "winj" if t.startswith("inj") else "prod"


def draw_wells_2d(ax, wells, label_names=True, size=45, fontsize=5):
    """Markers and names for every well, grouped so the legend stays short."""
    if not wells:
        return
    seen = set()
    for kind, (colour, marker, label) in WELL_MARK.items():
        pts = [w for w in wells if well_kind(w) == kind]
        if not pts:
            continue
        ax.scatter([w["i"] - 1 for w in pts], [w["j"] - 1 for w in pts],
                   marker=marker, s=size, c=colour, edgecolors="black",
                   linewidths=0.6, label=label, zorder=5)
        seen.add(label)
        if label_names:
            for w in pts:
                ax.annotate(str(w.get("name", "")),
                            (w["i"] - 1, w["j"] - 1),
                            textcoords="offset points", xytext=(3, 3),
                            fontsize=fontsize, color=colour, weight="bold",
                            zorder=6)
    if seen:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)


@guarded("field maps")
def plot_corner_point(outdir, meta):
    """Hand a corner-point run to the renderer that can draw it.

    Returns the PNGs it produced, so the report can embed them.

    A Cartesian model IS a box, so the voxel and imshow views below are its
    geometry, not an approximation of it.  A corner-point model is not: its
    layers dip, its cells are hexahedra on pillars, and a fault throw puts
    cells with adjacent indices in quite different places.  Drawn on the index
    box it comes out a rectangular slab with the faults and the structure gone
    -- which is most of what a field map is read for.  So that case goes to
    plot_norne.py, which has the cell centres and can draw the real surface."""
    import glob
    import subprocess
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "plot_norne.py")
    if not os.path.exists(script):
        print("[plot] corner-point renderer missing: %s" % script)
        return []
    print("[plot]   corner-point geometry: drawing the structure as well")
    try:
        r = subprocess.run([sys.executable, script, outdir, "--movie"],
                           capture_output=True, text=True)
        for line in (r.stdout or "").splitlines():
            print("[plot]   %s" % line.strip())
        if r.returncode != 0:
            # Say why, rather than leaving a run with no structural plots and
            # no explanation for it.
            tail = (r.stderr or "").strip().splitlines()[-6:]
            print("[plot] corner-point renderer failed (rc=%d)" % r.returncode)
            for line in tail:
                print("[plot]   %s" % line)
    except Exception as e:
        print("[plot] corner-point renderer failed: %s" % e)
    return (sorted(glob.glob(os.path.join(outdir, "norne_*.png")))
            + sorted(glob.glob(os.path.join(outdir, "norne_*.gif"))))


def plot_field_maps(outdir, meta, states, times, wells):
    """Per-snapshot maps of the mid layer for each state variable."""
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    ncell = nx * ny * nz
    kmid = nz // 2

    # One colour map for every field, and jet for all of them.
    #
    # Four different maps meant four different readings of "high": Blues runs
    # dark at the top, copper runs light, and comparing a water front against
    # a gas front across two frames required re-learning the scale each time.
    # Jet is what the structural figures use, so the animation and the stills
    # now agree, and it is what a reservoir engineer reads without thinking.
    specs = [("pressure", 0, "Pressure (psia)", "jet"),
             ("water", 1, "Water saturation", "jet"),
             ("oil", 2, "Oil saturation", "jet"),
             ("gas", 3, "Gas saturation", "jet")]

    # A shared colour range per variable keeps an animation readable — a
    # per-frame autoscale makes a steady field look like it is churning.
    ranges = {}
    for name, idx, _, _ in specs:
        lo, hi = math.inf, -math.inf
        for path in states:
            r = read_state(path, ncell)
            if r is None:
                continue
            v = r[idx]
            # nanmin/nanmax, not min/max.  A corner-point field carries NaN in
            # every inactive cell, and min(inf, nan) returns inf in Python --
            # so the scan finished with lo = +inf, hi = -inf, the guard below
            # fired, and EVERY field was drawn on a 0-1 colour scale.  On
            # saturations that is invisible because 0-1 is right; on pressure
            # it painted a 2,900-3,500 psia field uniformly at 1.0.
            fv = v[np.isfinite(v)]
            if fv.size:
                lo = min(lo, float(np.nanmin(fv)))
                hi = max(hi, float(np.nanmax(fv)))
        if lo > hi:
            lo, hi = 0.0, 1.0
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        ranges[name] = (lo, hi)

    made = {name: 0 for name, _, _, _ in specs}
    for s, path in enumerate(states):
        r = read_state(path, ncell)
        if r is None:
            continue
        t = times[s] if s < len(times) else s
        for name, idx, label, cmap in specs:
            field = r[idx]
            if name == "gas" and finite_range(field)[1] <= 1e-9:
                continue                       # no free gas: skip an empty plot
            lo, hi = ranges[name]
            fig, ax = plt.subplots(figsize=(5.4, 4.3))
            im = ax.imshow(layer_of(field, nx, ny, nz, kmid), origin="lower",
                           cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, label=label)
            draw_wells_2d(ax, wells)
            ax.set_title(f"{label}   k={kmid}   t = {t:g} days", fontsize=10)
            ax.set_xlabel("i"); ax.set_ylabel("j")
            savefig(fig, os.path.join(outdir, f"{name}_{s:04d}.png"), dpi=110)
            made[name] += 1
    return made


@guarded("layer panel")
def plot_layer_panel(outdir, meta, states):
    """Final-state pressure and Sw across every K layer, one panel per layer."""
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    ncell = nx * ny * nz
    if not states:
        return
    r = read_state(states[-1], ncell)
    if r is None:
        return
    for field, name, label, cmap in ((r[0], "pressure", "Pressure (psia)", "turbo"),
                                     (r[1], "water", "Water saturation", "Blues")):
        ncol = min(4, nz)
        nrow = int(math.ceil(nz / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow),
                                 squeeze=False)
        lo, hi = finite_range(field)
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        for k in range(nrow * ncol):
            ax = axes[k // ncol][k % ncol]
            if k >= nz:
                ax.axis("off")
                continue
            im = ax.imshow(layer_of(field, nx, ny, nz, k), origin="lower",
                           cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
            ax.set_title(f"k = {k}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.subplots_adjust(top=0.86 if nrow > 1 else 0.72)
        fig.colorbar(im, ax=axes.ravel().tolist(), label=label, shrink=0.85)
        fig.suptitle(f"{label} — final state, all layers", fontsize=11)
        savefig(fig, os.path.join(outdir, f"layers_{name}.png"))


@guarded("animations")
def make_gifs(outdir, made):
    """Assemble the per-snapshot PNGs into GIFs (needs Pillow)."""
    if not HAVE_PIL:
        print("[plot]   - GIFs skipped (Pillow not installed: pip install pillow)")
        return []
    gifs = []
    for name in ("pressure", "water", "oil", "gas"):
        if made.get(name, 0) < 2:
            continue
        files = glob.glob(os.path.join(outdir, f"{name}_*.png"))
        # Numeric order: sorted() would play frame 10 between 1 and 2.
        files = viz3d.sort_frames(files) if HAVE_VIZ else sorted(files)
        if len(files) < 2:
            continue
        frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE)
                  for f in files]
        out = os.path.join(outdir, f"Evolution_{name}.gif")
        frames[0].save(out, format="GIF", append_images=frames[1:],
                       save_all=True, duration=420, loop=0)
        gifs.append(os.path.basename(out))
    return gifs


# ===========================================================================
# 3-D voxel views, orthogonal slices and movies
# ===========================================================================
def _render_frame(job):
    """Render one 3-D voxel view and one slice panel.

    Module level and taking a plain tuple so it can be dispatched to a process
    pool: a closure is not picklable.  Returns the two paths written.
    """
    (field, title, p3, ps, wells, lo, hi, label, cmap) = job
    viz3d.plot3d_voxels(field, title, p3, wells=wells, vmin=lo, vmax=hi,
                        cbar_label=label, cmap=cmap)
    viz3d.plot_slices(field, title, ps, wells=wells, vmin=lo, vmax=hi,
                      cbar_label=label, cmap=cmap)
    return p3, ps
@guarded("3-D and slice rendering")
def render_field_series(outdir, name, data4d, times, wells, max_frames=24):
    """Render a (nt, nz, ny, nx) field as 3-D voxel views, slice panels and movies.

    A shared colour range across all time steps is used deliberately: a
    per-frame autoscale makes a steady field look like it is churning.
    """
    if not HAVE_VIZ:
        return {}
    a = np.asarray(data4d, dtype=float)
    if a.ndim == 3:
        a = a[None, ...]
    nt = a.shape[0]
    label, cmap = loaders.LABELS.get(name, (name, "jet"))

    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {}
    lo, hi = finite_range(finite)
    if hi - lo < 1e-30:
        hi = lo + 1e-30
    # A saturation that never leaves zero carries no information: skip it.
    if name in ("gas", "water", "oil") and hi <= 1e-9:
        return {}

    # ax.voxels costs seconds per frame, so the frames are rendered across
    # processes.  concurrent.futures is in the standard library, so this needs
    # no third-party package and there is no reason to render a reduced set.
    step = max(1, int(math.ceil(nt / float(max_frames))))
    idx = list(range(0, nt, step))
    if idx[-1] != nt - 1:
        idx.append(nt - 1)

    jobs = []
    for s in idx:
        t = times[s] if s < len(times) else s
        jobs.append((a[s], "%s   t = %g days" % (label, t),
                     os.path.join(outdir, "vox_%s_%04d.png" % (name, s)),
                     os.path.join(outdir, "slice_%s_%04d.png" % (name, s)),
                     wells, lo, hi, label, cmap))

    # Render frames in parallel across cores, as the CuPy reference does with
    # joblib Parallel(n_jobs=-1).  A 3-D voxel render is seconds of matplotlib
    # per frame; serially that is minutes of apparent hang with no output.
    workers = min(len(jobs), max(1, (os.cpu_count() or 1)))
    print("[plot]   %s: rendering %d 3-D frames on %d worker(s)" %
          (name, len(jobs), workers), flush=True)
    pairs = []
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for n, res in enumerate(ex.map(_render_frame, jobs), 1):
                pairs.append(res)
                print("[plot]   %s: %s" % (name, progress_bar(len(jobs), n)),
                      flush=True)
    except Exception as exc:
        # A sandbox that forbids subprocesses falls back to serial rather than
        # losing the figures entirely.
        print("[plot]   %s: parallel render unavailable (%s); serial"
              % (name, type(exc).__name__), flush=True)
        pairs = []
        for n, job in enumerate(jobs, 1):
            pairs.append(_render_frame(job))
            print("[plot]   %s: %s" % (name, progress_bar(len(jobs), n)), flush=True)
    v3d = [p for p, _ in pairs]
    vsl = [q for _, q in pairs]

    movies = []
    if len(v3d) > 1:
        movies += viz3d.make_movie(v3d, os.path.join(outdir, f"Evolution_{name}_3D")) or []
        movies += viz3d.make_movie(vsl, os.path.join(outdir, f"Evolution_{name}_slices")) or []
    return {"vox": v3d, "slice": vsl, "movies": movies}


@guarded("rock properties")
def plot_rock(outdir, meta, wells, box_views=True):
    """Permeability and porosity, as the reference's Permeability figure.

    rock.bin holds two float32 planes of nx*ny*nz written once by io.cu:
    permeability [mD] then porosity.  Permeability is shown on a log scale —
    a reservoir field spans orders of magnitude and a linear map collapses it
    to one colour, which is why the reference labels its colourbar
    "Log 10 - K(mD)".
    """
    if not HAVE_VIZ:
        return []
    path = os.path.join(outdir, meta.get("rock", "rock.bin"))
    if not os.path.exists(path):
        return []
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    ncell = nx * ny * nz
    a = np.fromfile(path, dtype=np.float32)
    if a.size < 2 * ncell:
        print("[plot]   ! rock.bin is short (%d of %d values)" % (a.size, 2 * ncell))
        return []
    perm = a[:ncell].reshape((nz, ny, nx)).astype(float)
    poro = a[ncell:2 * ncell].reshape((nz, ny, nx)).astype(float)

    made = []
    logk = np.log10(np.maximum(perm, 1e-6))
    for field, name, label, cmap, fname in (
            (logk, "perm", "Log10 permeability (mD)", "jet", "Permeability"),
            (poro, "poro", "Porosity", "viridis", "Porosity")):
        lo, hi = finite_range(field)
        if hi - lo < 1e-30:
            hi = lo + 1e-30
        # The index-box renderings are skipped on a corner-point grid for the
        # same reason the state ones are: they draw a rectangular slab where
        # the field dips and faults, and plot_norne.py has already drawn the
        # permeability and porosity on the real structure.  The layer maps
        # below are kept either way -- a map view of one layer is honest on
        # any grid, and is how a static model is usually read.
        if box_views:
            p3 = os.path.join(outdir, "%s.png" % fname)
            ps = os.path.join(outdir, "%s_slices.png" % fname)
            viz3d.plot3d_voxels(field, label, p3, wells=wells, vmin=lo,
                                vmax=hi, cbar_label=label, cmap=cmap)
            viz3d.plot_slices(field, label, ps, wells=wells, vmin=lo, vmax=hi,
                              cbar_label=label, cmap=cmap)
            made += [os.path.basename(p3), os.path.basename(ps)]

        # Layer-by-layer map view: how a geologist reads a static model.
        ncol = min(4, nz)
        nrow = int(math.ceil(nz / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow),
                                 squeeze=False)
        flat = field.reshape(-1)
        for k in range(nrow * ncol):
            ax = axes[k // ncol][k % ncol]
            if k >= nz:
                ax.axis("off")
                continue
            im = ax.imshow(layer_of(flat, nx, ny, nz, k), origin="lower",
                           cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
            # Wells on every layer panel: which layer a well is completed in
            # is the question these grids get read for.
            draw_wells_2d(ax, wells, label_names=False, size=14)
            leg = ax.get_legend()
            if leg is not None and k:
                leg.remove()            # one legend for the sheet, not 22
            ax.set_title("k = %d" % k, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.subplots_adjust(top=0.86 if nrow > 1 else 0.72)
        fig.colorbar(im, ax=axes.ravel().tolist(), label=label, shrink=0.85)
        fig.suptitle("%s - all layers" % label, fontsize=11)
        savefig(fig, os.path.join(outdir, "layers_%s.png" % name))
        made.append("layers_%s.png" % name)

    # Over the ACTIVE cells.  perm.min() on an array that is 61% NaN reports
    # nan-nan, which reads as "the rock did not load" -- and it said exactly
    # that while the figures beside it were drawing correctly.
    pk_lo, pk_hi = finite_range(perm)
    po_lo, po_hi = finite_range(poro)
    print("[plot]   rock: permeability %.4g-%.4g mD, porosity %.3f-%.3f"
          " (over %d active cells)"
          % (pk_lo, pk_hi, po_lo, po_hi, int(np.isfinite(perm).sum())),
          flush=True)
    return made


def load_native_series(outdir, meta, states):
    """Read every snapshot into {name: (nt, nz, ny, nx)}."""
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    ncell = nx * ny * nz
    acc = {"pressure": [], "water": [], "oil": [], "gas": []}
    for path in states:
        r = read_state(path, ncell)
        if r is None:
            continue
        for key, arr in zip(("pressure", "water", "oil", "gas"), r):
            acc[key].append(arr.reshape((nz, ny, nx)))
    return {k: np.stack(v, axis=0) for k, v in acc.items() if v}


@guarded("field evolution")
def plot_field_evolution(outdir, meta, states, times):
    """Field-average pressure and saturations, plus recovery factor."""
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    ncell = nx * ny * nz
    st = {k: [] for k in ("pmin", "pmax", "pavg", "sw", "so", "sg")}
    for path in states:
        r = read_state(path, ncell)
        if r is None:
            continue
        P, SW, SO, SG = r
        st["pmin"].append(float(P.min())); st["pmax"].append(float(P.max()))
        st["pavg"].append(float(P.mean()))
        st["sw"].append(float(SW.mean())); st["so"].append(float(SO.mean()))
        st["sg"].append(float(SG.mean()))
    if not st["pavg"]:
        return None
    t = list(times[:len(st["pavg"])]) or list(range(len(st["pavg"])))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t, st["pavg"], "-o", ms=3, color="#1f77b4", label="mean")
    ax[0].fill_between(t, st["pmin"], st["pmax"], alpha=0.18, color="#1f77b4",
                       label="min–max")
    ax[0].set_xlabel("time (days)"); ax[0].set_ylabel("pressure (psia)")
    ax[0].set_title("Reservoir pressure"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, st["sw"], "-o", ms=3, label="$S_w$", color="#1f77b4")
    ax[1].plot(t, st["so"], "-o", ms=3, label="$S_o$", color="#d62728")
    if max(st["sg"]) > 1e-6:
        ax[1].plot(t, st["sg"], "-o", ms=3, label="$S_g$", color="#2ca02c")
    ax[1].set_xlabel("time (days)"); ax[1].set_ylabel("field-average saturation")
    ax[1].set_title("Field saturations"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    so0 = st["so"][0] if st["so"][0] > 1e-12 else 1.0
    rec = [100.0 * (so0 - v) / so0 for v in st["so"]]
    ax[2].plot(t, rec, "-o", ms=3, color="#9467bd")
    ax[2].set_xlabel("time (days)"); ax[2].set_ylabel("recovery factor (%)")
    ax[2].set_title("Oil recovery"); ax[2].grid(alpha=0.3)

    savefig(fig, os.path.join(outdir, "field_evolution.png"))
    return st


# The well quantities a deck may ask for, and how each is obtained.
#
# WOPR, WWPR, WGPR and WBHP are written per well by the post-process.  WWCT and
# WGOR are ratios of those, and are formed here rather than stored, so the two
# can never disagree about what a water cut is.
WELL_QTY = {
    "WOPR": ("Oil rate (STB/day)",       "WOPR", None),
    "WWPR": ("Water rate (STB/day)",     "WWPR", None),
    "WGPR": ("Gas rate (MSCF/day)",      "WGPR", None),
    "WWCT": ("Water cut (fraction)",     "WWCT", ("WWPR", "WOPR")),
    "WGOR": ("Gas-oil ratio (MSCF/STB)", None,   ("WGPR", "WOPR")),
    "WBHP": ("Bottom-hole pressure (psia)", "WBHP", None),
}
WELL_QTY_ORDER = ["WOPR", "WWPR", "WGPR", "WWCT", "WGOR", "WBHP"]


def well_legend(ax, handles=None, labels=None, rows=3, fontsize=6.5,
                fig=None):
    """Well names UNDER the axes, in columns, in as few rows as will do.

    matplotlib's default puts them in one column inside the axes, and a
    tight bounding box then grows the figure to fit that column: with 33
    producers the cumulative figure came out 732 x 1301 -- taller than it was
    wide, the plot squeezed into the top third and a long ribbon of names
    trailing below it.  The plot should be the subject of the picture.

    `fig` draws one legend for a whole figure instead of one per subplot,
    which is what a grid of panels sharing the same wells wants."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    n = len(handles)
    if n == 0:
        return
    ncol = max(1, min(11, int(math.ceil(n / float(rows)))))
    kw = dict(fontsize=fontsize, ncol=ncol, frameon=False,
              handlelength=1.3, columnspacing=1.1, handletextpad=0.4,
              labelspacing=0.35)
    if fig is not None:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), **kw)
    else:
        ax.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14), **kw)


def requested_well_curves(meta):
    """Which curves to draw, from the deck's WELLPLOT.

    Empty or ALL means every one; NONE means skip the section.  Anything the
    deck names that is not a known quantity is reported rather than ignored --
    a typo in a deck should not silently produce a report with a section
    missing from it."""
    want = [str(x).upper() for x in (meta.get("well_plot") or [])]
    if not want or "ALL" in want:
        return WELL_QTY_ORDER
    if "NONE" in want:
        return []
    good = [q for q in WELL_QTY_ORDER if q in want]
    bad = [q for q in want if q not in WELL_QTY]
    if bad:
        print("[plot]   ! WELLPLOT: unknown %s (known: %s)"
              % (", ".join(bad), ", ".join(WELL_QTY_ORDER)))
    return good


@guarded("well curves")
def plot_well_curves(outdir, meta):
    """One figure per requested quantity, every well overlaid.

    A figure per quantity rather than a panel per well: the question these are
    read for is which well is behind the field's behaviour, and that is a
    comparison across wells at one time, not a tour of thirty-six small
    charts."""
    # Each of these used to be a bare `return []`, and the report simply came
    # out with no well curves and nothing said about why.  The one that
    # actually fired on Norne was the last: the writer emitted a full header
    # and rows holding only a time, so no row matched the header width and the
    # array came out empty.  Silence there cost an afternoon.
    def skip(why):
        print("[plot]   ! well curves skipped: %s" % why, flush=True)
        return []

    wanted = requested_well_curves(meta)
    if not wanted:
        return skip("deck WELLPLOT selects none")
    csv_path = os.path.join(outdir, "RSM_FVM.csv")
    if not os.path.exists(csv_path):
        return skip("no RSM_FVM.csv in %s" % outdir)
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.reader(f))
    if len(rows) < 2:
        return skip("RSM_FVM.csv has a header and no data rows")
    header = rows[0]
    good = [r for r in rows[1:] if len(r) == len(header)]
    if not good:
        return skip("RSM_FVM.csv header lists %d columns, data rows carry %d"
                    % (len(header), len(rows[1])))
    if len(good) < len(rows) - 1:
        print("[plot]   ! %d of %d well-rate rows had the wrong width"
              % (len(rows) - 1 - len(good), len(rows) - 1), flush=True)
    data = np.array([[float(x) for x in r] for r in good])
    if data.size == 0:
        return skip("RSM_FVM.csv parsed to nothing")
    t = data[:, 0]

    def well_name(h):
        """Well name out of a column header like "B-1H-WOPR(BBL/DAY)".

        Split from the RIGHT.  Splitting from the left is correct only for a
        name with no hyphen in it, and a field deck's names are full of them:
        every one of Norne's 36 wells is B-1H, C-4AH, E-3H and so on, so the
        left split labelled every curve "B" or "C" and collapsed six wells
        onto one key."""
        return h.rsplit("-", 1)[0]

    def series(tag):
        """{well name: column} for a stored tag."""
        out = {}
        for j in range(1, len(header)):
            h = header[j]
            if "-" not in h or tag not in h:
                continue
            out[well_name(h)] = data[:, j]
        return out

    made = []
    for q in wanted:
        label, tag, ratio = WELL_QTY[q]
        if tag and series(tag):
            curves = series(tag)
            if q == "WWCT":
                # Stored as a percentage; report the fraction, as OPM does.
                curves = {k: v / 100.0 if np.nanmax(v) > 1.5 else v
                          for k, v in curves.items()}
        elif ratio:
            num, den = series(ratio[0]), series(ratio[1])
            curves = {}
            for k in num:
                if k not in den:
                    continue
                d = np.array(den[k], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.where(np.abs(d) > 1e-12,
                                 np.array(num[k], dtype=float) / d, np.nan)
                curves[k] = v
        else:
            continue
        curves = {k: v for k, v in curves.items()
                  if np.any(np.isfinite(v)) and np.nanmax(np.abs(v)) > 0}
        if not curves:
            continue

        # One PANEL PER WELL, five across.
        #
        # Overlaying thirty-three producers on one axes produced a figure
        # whose legend was longer than its plot and in which no individual
        # well could be followed.  A rate belongs to a well; the grid is what
        # lets you read thirty-three of them.  Twenty-five to a page, so a
        # bigger field gets more pages rather than smaller panels.
        names = sorted(curves)
        per_page, ncol = 25, 5
        pages = int(math.ceil(len(names) / float(per_page)))
        for pg in range(pages):
            page = names[pg * per_page:(pg + 1) * per_page]
            nrow = int(math.ceil(len(page) / float(ncol)))
            fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                     figsize=(3.4 * ncol, 2.5 * nrow))
            for i, name in enumerate(page):
                ax = axes[i // ncol][i % ncol]
                ax.plot(t, curves[name], lw=1.4, color="#1f77b4")
                ax.set_title(name, fontsize=9, weight="bold")
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=7)
                if i // ncol == nrow - 1:
                    ax.set_xlabel("days", fontsize=8)
                if i % ncol == 0:
                    ax.set_ylabel(label, fontsize=8)
            for k in range(len(page), nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")
            ttl = "%s - one panel per well" % q
            if pages > 1:
                ttl += "  (%d of %d)" % (pg + 1, pages)
            fig.suptitle(ttl, fontsize=13, weight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fn = "wellcurve_%s%s.png" % (q, "" if pages == 1 else "_%d" % (pg + 1))
            savefig(fig, os.path.join(outdir, fn), dpi=110)
            made.append(fn)
        print("[plot]   wrote %s curves: %d well(s) over %d page(s)"
              % (q, len(names), pages))
    return made


@guarded("well rates")
def plot_well_rates(outdir):
    """Producer rates, water cut and cumulative production from RSM_FVM.csv."""
    csv_path = os.path.join(outdir, "RSM_FVM.csv")
    if not os.path.exists(csv_path):
        return None
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.reader(f))
    if len(rows) < 2:
        return None
    header = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:] if len(r) == len(header)])
    if data.size == 0:
        return None
    t = data[:, 0]

    def cols(tag):
        return [(j, header[j]) for j in range(1, len(header)) if tag in header[j]]

    panels = [("WOPR", "oil rate (BBL/day)", "Producer oil rate"),
              ("WWPR", "water rate (BBL/day)", "Producer water rate"),
              ("WGPR", "gas rate (MSCF/day)", "Producer gas rate"),
              ("WWCT", "water cut (%)", "Producer water cut")]
    active = [(tag, yl, ti) for tag, yl, ti in panels if cols(tag)]
    if not active:
        return None

    ncol = 2
    nrow = int(math.ceil(len(active) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.0 * nrow),
                             squeeze=False)
    for n, (tag, ylab, title) in enumerate(active):
        ax = axes[n // ncol][n % ncol]
        for j, name in cols(tag):
            ax.plot(t, data[:, j], "-o", ms=2.5, label=name.rsplit("-", 1)[0])
        ax.set_xlabel("time (days)"); ax.set_ylabel(ylab)
        ax.set_title(title); ax.grid(alpha=0.3)
    for n in range(len(active), nrow * ncol):
        axes[n // ncol][n % ncol].axis("off")
    # One legend for the figure: all four panels draw the same wells, so four
    # copies of the same 33 names is three copies of clutter.
    h, l = axes[0][0].get_legend_handles_labels()
    well_legend(None, h, l, fig=fig)
    savefig(fig, os.path.join(outdir, "well_rates.png"))

    # Cumulative production: trapezoidal integration of each producer rate.
    totals = {}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    plotted = False
    # One colour per WELL and one line style per PHASE, rather than a separate
    # legend entry for each pairing.  With 33 producers the latter is 66
    # entries saying the same 33 names twice, which is what made this figure
    # taller than it was wide.
    colour = {}
    for tag, dash, scale in (("WOPR", "-", 1e-6), ("WWPR", "--", 1e-6)):
        for j, name in cols(tag):
            well = name.rsplit("-", 1)[0]
            if well not in colour:
                colour[well] = "C%d" % (len(colour) % 10)
            cum = np.concatenate([[0.0], np.cumsum(
                0.5 * (data[1:, j] + data[:-1, j]) * np.diff(t))])
            ax.plot(t, cum * scale, dash, lw=1.2, color=colour[well],
                    label=well if tag == "WOPR" else None)
            totals[name] = float(cum[-1])
            plotted = True
    if plotted:
        ax.set_xlabel("time (days)"); ax.set_ylabel("cumulative production (MMBBL)")
        ax.set_title("Cumulative production"); ax.grid(alpha=0.3)
        h, l = ax.get_legend_handles_labels()
        # Two keys for the styles, since colour no longer says which phase.
        h = h + [mlines.Line2D([], [], color="0.35", ls="-", label="oil"),
                 mlines.Line2D([], [], color="0.35", ls="--", label="water")]
        l = l + ["oil", "water"]
        well_legend(ax, h, l)
        savefig(fig, os.path.join(outdir, "cumulative_production.png"))
    else:
        plt.close(fig)
    return totals


@guarded("solver convergence")
def plot_convergence(outdir, meta):
    """Newton and linear iteration counts per report step."""
    newton = meta.get("newton_iters") or []
    linear = meta.get("linear_iters") or []
    if not newton and not linear:
        return
    steps = np.arange(1, max(len(newton), len(linear)) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if newton:
        ax[0].bar(steps[:len(newton)], newton, color="#1f77b4")
        ax[0].set_ylabel("Newton iterations")
        ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    if linear:
        ax[1].bar(steps[:len(linear)], linear, color="#ff7f0e")
        ax[1].set_ylabel("linear (Krylov) iterations")
    for a, title in zip(ax, ("Newton iterations per report step",
                             f"{meta.get('krylov', 'Krylov')} iterations per report step")):
        a.set_xlabel("report step"); a.set_title(title); a.grid(alpha=0.3, axis="y")
    savefig(fig, os.path.join(outdir, "solver_convergence.png"))


@guarded("text summary")
def append_summary(outdir, meta, st, totals):
    """Append quantitative trends to summary.txt (created by the simulator)."""
    if not st or not st.get("pavg"):
        return
    path = os.path.join(outdir, "summary.txt")
    with open(path, "a") as s:
        s.write("\nQuantitative field trends (post-processed)\n")
        s.write("------------------------------------------\n")
        s.write(f"* Mean pressure: {st['pavg'][0]:.1f} -> {st['pavg'][-1]:.1f} psia "
                f"({st['pavg'][-1] - st['pavg'][0]:+.1f} over the run).\n")
        s.write(f"* Mean water saturation: {st['sw'][0]:.3f} -> {st['sw'][-1]:.3f} "
                f"(waterflood front advance).\n")
        so0, so1 = st["so"][0], st["so"][-1]
        rf = 100.0 * (so0 - so1) / so0 if so0 > 1e-12 else 0.0
        s.write(f"* Mean oil saturation:   {so0:.3f} -> {so1:.3f} "
                f"(recovery factor = {rf:.2f}%).\n")
        if max(st["sg"]) > 1e-4:
            s.write(f"* Mean gas saturation:   {st['sg'][0]:.3f} -> {st['sg'][-1]:.3f} "
                    f"(solution-gas liberation as p drops below Pb).\n")
        if totals:
            for name, v in totals.items():
                s.write(f"* Cumulative {name}: {human(v, 'BBL')}\n")
        nit = meta.get("newton_iters") or []
        lit = meta.get("linear_iters") or []
        if nit:
            s.write(f"* Newton: {sum(nit)} total, {sum(nit)/len(nit):.2f} per step.\n")
        if lit and sum(nit):
            s.write(f"* Krylov: {sum(lit)} total, "
                    f"{sum(lit)/max(sum(nit),1):.2f} per Newton iteration "
                    f"(CPR keeps this near-constant under refinement).\n")


# Total bytes of embedded media.  Chrome copes with a large single document,
# but there is no point inlining a gigabyte of animation.
EMBED_BUDGET = 96 * 1024 * 1024


def data_uri(path, state):
    """Inline a file as a data: URI, within a running byte budget.

    The report is routinely copied off the machine that produced it — scp'd,
    attached, opened from a different directory — and relative <img src> then
    resolves to nothing and the page renders as a wall of broken icons.
    Embedding makes index.html a single self-contained file that works
    anywhere.  Anything that does not fit the budget keeps its relative link.
    """
    mime = {".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
            ".mp4": "video/mp4"}.get(os.path.splitext(path)[1].lower())
    if not mime:
        return None
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if state["used"] + size > EMBED_BUDGET:
        state["skipped"] += 1
        return None
    import base64
    try:
        with open(path, "rb") as f:
            payload = base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None
    state["used"] += size
    return "data:%s;base64,%s" % (mime, payload)


@guarded("HTML report")
def write_report(outdir, meta, st, totals, gifs, movies=None,
                 corner_pngs=None, wellcurves=None, box_views=True):
    """A single self-contained index.html tying every figure together."""
    embed = {"used": 0, "skipped": 0}

    def img(fname, caption):
        full = os.path.join(outdir, fname)
        if not os.path.exists(full):
            return ""
        src = data_uri(full, embed) or fname
        return ('<figure><img src="%s" alt="%s" loading="lazy">'
                '<figcaption>%s</figcaption></figure>' % (src, caption, caption))

    rows = [("Title", meta.get("title", "-")),
            ("Model", meta.get("model", "-")),
            ("Grid", f"{meta.get('nx')} x {meta.get('ny')} x {meta.get('nz')} "
                     f"({meta.get('nx',0)*meta.get('ny',0)*meta.get('nz',0)} cells)"),
            ("Cell size", f"{meta.get('dx')} x {meta.get('dy')} x {meta.get('dz')} ft"),
            ("Outer Krylov", meta.get("krylov", "-")),
            ("Mixed precision", str(meta.get("mixed_precision", "-"))),
            ("df64 emulated operator", str(meta.get("emulate_fp64", "-"))),
            ("MPI ranks / GPUs", str(meta.get("nranks", "-"))),
            ("Report steps", str(meta.get("nsnap", "-"))),
            ("Wall clock", f"{meta.get('wall_seconds', 0):.2f} s")]
    nit = meta.get("newton_iters") or []
    lit = meta.get("linear_iters") or []
    if nit:
        rows.append(("Newton iterations", f"{sum(nit)} total"))
    if lit:
        rows.append(("Krylov iterations", f"{sum(lit)} total"))
    if st and st.get("so"):
        so0, so1 = st["so"][0], st["so"][-1]
        if so0 > 1e-12:
            rows.append(("Recovery factor", f"{100.0*(so0-so1)/so0:.2f} %"))

    table = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    gif_html = "".join(img(g, f"Evolution — {g.split('_')[-1].split('.')[0]}")
                       for g in (gifs or []))

    # 3-D voxel views and slice panels: show the final frame of each variable,
    # with the movies alongside.
    def latest(pattern):
        f = sorted(glob.glob(os.path.join(outdir, pattern)))
        return os.path.basename(f[-1]) if f else None

    # Structural views first, when the grid has structure to show.  On a
    # corner-point field these are the plots that answer "where is the oil":
    # the index-box views below cannot show a fault or a dip.
    corner_html = ""
    for f in (corner_pngs or []):
        base = os.path.basename(f)
        what = base.replace("norne_", "").rsplit(".", 1)[0]
        corner_html += img(base, "Structure — %s" % what.replace("_", " "))

    # Only figures THIS run produced.  latest() globs the output directory,
    # and a directory that has been run before still holds the voxel views
    # from when they were drawn -- so switching them off left the old ones
    # embedded, and the report kept showing a box that nothing had rendered.
    vox_html, slice_html = "", ""
    if not box_views:
        names = []
    else:
        names = ["pressure", "water", "oil", "gas", "perm", "poro"]
    for name in names:
        v = latest(f"vox_{name}_*.png")
        if v:
            vox_html += img(v, f"3-D voxel view — {name} (final step)")
        s = latest(f"slice_{name}_*.png")
        if s:
            slice_html += img(s, f"Orthogonal slices — {name} (final step)")

    # Static rock properties, rendered once per run.
    rock_html = ""
    rock_figs = [("layers_perm.png", "Permeability — all K layers"),
                 ("layers_poro.png", "Porosity — all K layers")]
    if box_views:
        # Only when this run drew them.  These have fixed names, so a
        # directory that had been run before still held them and the report
        # kept listing a box nothing had rendered.
        rock_figs = [("Permeability.png", "Permeability — log10 mD"),
                     ("Porosity.png", "Porosity"),
                     ("Permeability_slices.png",
                      "Permeability — orthogonal slices"),
                     ("Porosity_slices.png", "Porosity — orthogonal slices")]             + rock_figs
    for f, cap in rock_figs:
        rock_html += img(f, cap)
    if rock_html:
        rock_html = ('<h2>Rock properties</h2><div class="grid">%s</div>'
                     % rock_html)

    movie_html = ""
    for m in (movies or []):
        b = os.path.basename(m)
        if b.endswith(".gif"):
            movie_html += img(b, f"Animation — {b}")
        elif b.endswith(".mp4"):
            src = data_uri(m, embed) or b
            movie_html += ('<figure><video controls loop muted src="%s" '
                           'style="max-width:100%%"></video>'
                           '<figcaption>Animation — %s</figcaption></figure>'
                           % (src, b))

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NVRS-CUDA report — {meta.get('title','')}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
        max-width: 1100px; color: #1a1a1a; line-height: 1.5; }}
 h1 {{ border-bottom: 3px solid #76b900; padding-bottom: .3rem; }}
 h2 {{ margin-top: 2.5rem; color: #333; border-bottom: 1px solid #ddd; }}
 table {{ border-collapse: collapse; margin: 1rem 0; }}
 th, td {{ border: 1px solid #ddd; padding: .4rem .8rem; text-align: left; }}
 th {{ background: #f5f5f5; width: 15rem; }}
 figure {{ margin: 1rem 0; }}
 img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
 figcaption {{ font-size: .85rem; color: #666; margin-top: .3rem; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }}
</style></head><body>
<h1>NVRS-CUDA simulation report</h1>
<p><em>{meta.get('title','')}</em></p>
<table>{table}</table>

<h2>Field evolution</h2>
<div class="grid">
{img('field_evolution.png', 'Field-average pressure, saturations and recovery factor')}
{img('solver_convergence.png', 'Newton and Krylov iteration counts per report step')}
</div>

<h2>Well response</h2>
<div class="grid">
{img('well_rates.png', 'Producer rates and water cut')}
{''.join(img(f, f.replace('wellcurve_', '').replace('.png', '') + ' - one panel per well') for f in (wellcurves or []))}
{img('cumulative_production.png', 'Cumulative oil and water production')}
</div>

{rock_html}{('<h2>Field structure</h2><p class="note">Drawn on the '
   'corner-point geometry: the layers dip and the faults throw, which the '
   'index-box views below cannot show.</p><div class="grid">'
   + corner_html + '</div>') if corner_html else ''}
{('<h2>3-D views</h2><div class="grid">' + vox_html + '</div>')
   if vox_html.strip() else ''}
{('<h2>Orthogonal slices</h2><div class="grid">' + slice_html + '</div>')
   if slice_html.strip() else ''}

<h2>Final state by layer</h2>
<div class="grid">
{img('layers_pressure.png', 'Pressure, all K layers, final report step')}
{img('layers_water.png', 'Water saturation, all K layers, final report step')}
</div>

<h2>Animated evolution</h2>
<div class="grid">{gif_html}{movie_html}</div>

<hr><p style="font-size:.8rem;color:#888">
Generated by scripts/plot_results.py &middot; NVRS-CUDA &middot;
C. Etienam, T. Onishi, O. Ovcharenko, N. Luiken, I. Said</p>
</body></html>"""
    out = os.path.join(outdir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    mb = embed["used"] / (1024.0 * 1024.0)
    note = "" if not embed["skipped"] else (
        "  (%d file(s) over the budget kept as relative links)" % embed["skipped"])
    print("[plot]   report: %.1f MB of figures embedded, self-contained%s"
          % (mb, note), flush=True)
    # The report is usually produced on a compute node and read somewhere else,
    # so print the absolute path and a ready-to-paste fetch command rather than
    # a relative name the reader cannot act on.
    try:
        import socket
        host = socket.gethostname()
        print("[plot]   fetch the report with:", flush=True)
        print("           scp %s:%s ." % (host, os.path.abspath(out)), flush=True)
        print("         or serve the folder:", flush=True)
        print("           python3 -m http.server 8000 --directory %s"
              % os.path.abspath(outdir), flush=True)
    except Exception:
        pass


# ===========================================================================
# Driver
# ===========================================================================
def process(outdir):
    print(f"[plot] === {outdir} ===")
    try:
        meta = load_meta(outdir)
    except Exception as exc:
        print(f"[plot]   ! cannot read meta.json: {exc}")
        return False

    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    times = meta.get("times", [])
    wells = meta.get("wells", [])
    states = sorted(glob.glob(os.path.join(outdir, "state_*.bin")))
    print(f"[plot]   grid {nx}x{ny}x{nz}, {len(states)} snapshots, "
          f"{len(wells)} wells")

    if not states:
        print("[plot]   ! no state_*.bin snapshots found")
        return False

    if not HAVE_MPL:
        # Text-only degradation: still give the user the numbers.
        ncell = nx * ny * nz
        r = read_state(states[-1], ncell)
        if r is not None:
            P, SW, SO, SG = r
            print(f"[plot]   final: <p>={P.mean():.1f} psia  <Sw>={SW.mean():.3f}  "
                  f"<So>={SO.mean():.3f}  <Sg>={SG.mean():.3f}")
        return True

    made = plot_field_maps(outdir, meta, states, times, wells) or {}
    plot_layer_panel(outdir, meta, states)
    gifs = make_gifs(outdir, made) or []
    st = plot_field_evolution(outdir, meta, states, times)
    totals = plot_well_rates(outdir)
    wellcurves = plot_well_curves(outdir, meta) or []
    plot_convergence(outdir, meta)

    # One decision, used by everything that draws on the index box.
    box_views = (not meta.get("corner_point")) or os.environ.get("NVRS_BOX_VIEWS")
    plot_rock(outdir, meta, wells, box_views=bool(box_views))

    # A corner-point run also gets drawn as the field it is.  The views below
    # are indexed by the box, which for Norne is a rectangular slab with the
    # faults and the dip gone; plot_norne.py has the cell centres and can draw
    # the real surface.  Both are produced -- the box views are still the way
    # to read a single layer, and the structural views are the way to see where
    # that layer sits.
    corner_pngs = []
    if meta.get("corner_point"):
        corner_pngs = plot_corner_point(outdir, meta) or []

    # 3-D voxel views with wells, orthogonal slices, and movies.
    #
    # Skipped on a corner-point grid.  These are drawn on the index box, which
    # for a faulted field is a rectangular slab: the dip is gone, the throws
    # are gone, and the inactive 61% of the box is drawn as reservoir.  The
    # result looks like a grey brick and says nothing the structural figures
    # above do not say better.  NVRS_BOX_VIEWS=1 brings them back.
    movies = []
    if HAVE_VIZ and not box_views:
        print("[plot]   index-box 3-D views skipped (corner-point grid); "
              "NVRS_BOX_VIEWS=1 to draw them anyway")
    if HAVE_VIZ and box_views:
        series = load_native_series(outdir, meta, states)
        for name, arr in series.items():
            res = render_field_series(outdir, name, arr, times, wells)
            if res:
                movies += res.get("movies", [])

    append_summary(outdir, meta, st, totals)
    write_report(outdir, meta, st, totals, gifs, movies, corner_pngs,
                 wellcurves, box_views=bool(box_views))
    # The PNGs have now been consumed twice: make_gifs() above assembled the
    # frames into the animations, and write_report() base64-embedded every
    # figure into the page.  Only from this point are they duplicates -- and
    # worse than duplicates, since they carry fixed names that next run's glob
    # picks up, which is how a stale box view kept reappearing in a
    # corner-point report.  So clear them here and nowhere earlier, and only
    # if the page they were embedded into actually exists: if write_report
    # raised, the figures are all that survived and deleting them would throw
    # the run away.  GIFs stay -- they are the other deliverable.
    html = os.path.join(outdir, "index.html")
    if not os.environ.get("NVRS_KEEP_PNG") and os.path.exists(html):
        gone = 0
        for f in (glob.glob(os.path.join(outdir, "*.png"))
                  + glob.glob(os.path.join(outdir, "*.PNG"))):
            try:
                os.remove(f)
                gone += 1
            except OSError:
                pass
        if gone:
            print("[plot]   cleared %d loose PNG(s): %d GIF(s) and the report "
                  "hold them now (NVRS_KEEP_PNG=1 to keep)"
                  % (gone, len(glob.glob(os.path.join(outdir, "*.gif")))))
    print(f"[plot]   done -> {os.path.join(outdir, 'index.html')}")
    return True


# ===========================================================================
# Alternate sources: ECLIPSE cases and raw arrays
# ===========================================================================
def process_eclipse(base, outdir=None):
    """Plot an ECLIPSE/OPM case (CASE.EGRID + CASE.UNRST [+ summary])."""
    if not HAVE_ECL:
        print("[plot] ECLIPSE reader unavailable (scripts/ecl.py missing)")
        return False
    case = ecl.EclipseCase(base)
    outdir = outdir or (os.path.dirname(os.path.abspath(base)) or ".")
    print(f"[plot] === ECLIPSE case {base} ===")
    try:
        nx, ny, nz = case.dims
    except Exception as exc:
        print(f"[plot]   ! {exc}")
        return False
    times = case.restart_times()
    print(f"[plot]   grid {nx}x{ny}x{nz}, {len(times)} restart steps")

    wells = []          # ECLIPSE well locations are not in EGRID/UNRST
    got = False
    for kw, name in (("PRESSURE", "pressure"), ("SWAT", "water"),
                     ("SOIL", "oil"), ("SGAS", "gas")):
        arr = case.restart_field(kw)
        if arr is None:
            continue
        got = True
        render_field_series(outdir, name, arr, list(times), wells)

    # Static properties from INIT, if present.
    for kw, name in (("PERMX", "perm"), ("PORO", "poro")):
        arr = case.init_field(kw)
        if arr is not None:
            render_field_series(outdir, name, arr[None, ...], [0.0], wells)
            got = True

    # Summary curves.
    t, vectors = case.summary()
    if t is not None and vectors and HAVE_MPL:
        plot_summary_vectors(outdir, t, vectors)
        got = True
    if got:
        print(f"[plot]   done -> {outdir}")
    else:
        print("[plot]   ! nothing readable in that case")
    return got


@guarded("summary curves")
def plot_summary_vectors(outdir, t, vectors):
    """Plot SMSPEC/UNSMRY time series, grouped by mnemonic."""
    groups = {}
    for label, v in vectors.items():
        groups.setdefault(v["keyword"], []).append((label, v))
    keys = sorted(groups)
    if not keys:
        return
    ncol = 2
    nrow = int(math.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.8 * nrow),
                             squeeze=False)
    for n, kw in enumerate(keys):
        ax = axes[n // ncol][n % ncol]
        unit = ""
        for label, v in groups[kw]:
            unit = v["unit"]
            ax.plot(t, v["values"], "-", lw=1.4,
                    label=v["well"] if v["well"] not in ("", ":+:+:+:+") else kw)
        ax.set_xlabel("time (days)")
        ax.set_ylabel(f"{kw} ({unit})" if unit else kw)
        ax.set_title(kw); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for n in range(len(keys), nrow * ncol):
        axes[n // ncol][n % ncol].axis("off")
    savefig(fig, os.path.join(outdir, "summary_vectors.png"))


def process_raw_file(path, dims=None, outdir=None, key=None, times=None):
    """Plot a raw .mat / .npy / ASCII field file."""
    if not HAVE_VIZ:
        print("[plot] raw-array plotting needs viz3d/loaders")
        return False
    outdir = outdir or (os.path.dirname(os.path.abspath(path)) or ".")
    os.makedirs(outdir, exist_ok=True)
    print(f"[plot] === {path} ===")
    try:
        fields = loaders.load_any(path, dims=dims, key=key)
    except Exception as exc:
        print(f"[plot]   ! {exc}")
        return False

    any_done = False
    for raw_name, arr in fields.items():
        canon = loaders.canonical_name(raw_name) or raw_name
        nt, nz, ny, nx = arr.shape
        print(f"[plot]   {raw_name!r} -> {canon}: {nt} step(s), {nx}x{ny}x{nz}")
        tt = list(times) if times else list(range(nt))
        if render_field_series(outdir, canon, arr, tt, []):
            any_done = True
    if any_done:
        print(f"[plot]   done -> {outdir}")
    return any_done


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Automatic post-processor for NVRS-CUDA output. "
                    "With no arguments it discovers every output directory itself.")
    ap.add_argument("outdirs", nargs="*",
                    help="output directories (default: auto-discover)")
    ap.add_argument("--ecl", metavar="CASE",
                    help="plot an ECLIPSE/OPM case: path to CASE (no extension) "
                         "or to any of CASE.EGRID/.UNRST/.SMSPEC. Reads this "
                         "simulator's output and OPM Flow's alike.")
    ap.add_argument("--file", metavar="PATH",
                    help="plot a raw field file (.mat, .npy, .npz, or flat ASCII). "
                         "3-D arrays are (k,j,i); 4-D arrays are (t,k,j,i).")
    ap.add_argument("--dims", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                    help="grid dimensions; required for flat ASCII input and "
                         "used to disambiguate axis order otherwise")
    ap.add_argument("--key", metavar="NAME",
                    help="with --file: plot only this array from a .mat/.npz")
    ap.add_argument("--out", metavar="DIR",
                    help="where to write figures (default: alongside the input)")
    ap.add_argument("--watch", action="store_true",
                    help="re-plot every --interval seconds while a run advances")
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between passes in --watch mode (default 30)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if nothing could be plotted "
                         "(default is to always exit 0 so a plotting problem "
                         "never fails a simulation job)")
    args = ap.parse_args(argv)

    dims = tuple(args.dims) if args.dims else None

    def one_pass():
        # Explicit sources take precedence over auto-discovery.
        if args.file:
            return process_raw_file(args.file, dims=dims, outdir=args.out,
                                    key=args.key)
        if args.ecl:
            return process_eclipse(args.ecl, outdir=args.out)
        dirs = discover_outdirs(args.outdirs)
        if not dirs:
            print("[plot] no output directories found "
                  "(looked for any directory containing meta.json)")
            return False
        return any(process(d) for d in dirs)

    if args.watch:
        print(f"[plot] watch mode, every {args.interval:g}s — Ctrl-C to stop")
        try:
            while True:
                one_pass()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[plot] stopped")
        return 0

    ok = one_pass()
    return 0 if (ok or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
