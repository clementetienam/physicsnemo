#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# spe10_report.py -- render one self-contained page for an SPE10 model 2 run.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# The backend study answers "how fast".  This answers "what did it do":
#
#   * the rock, as three-dimensional cutaway blocks and as areal layers;
#   * pressure and saturation in three dimensions, still and animated over
#     the report steps;
#   * well response -- WOPR, WWPR, WWCT, WBHP -- per producer;
#   * recovery factor and material balance, computed from the fields
#     themselves rather than reported by the simulator;
#   * where the time went, and the run log.
#
# Everything is embedded as base64, so the page travels as one file.
#
#   python3 scripts/spe10_report.py <run-dir> [--out spe10.html] [--log FILE]
#
# The three-dimensional views are voxel blocks drawn by scripts/viz3d.py, the
# same renderer benchmark/compare.py uses: the block is drawn semi-
# transparently so the interior is visible rather than only its outer faces,
# with the wells overlaid as labelled sticks.
import base64
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecl    # noqa: E402
import viz3d  # noqa: E402  (the shared voxel renderer, as compare.py uses)

NX, NY, NZ = 60, 220, 85
DX, DY, DZ = 20.0, 10.0, 2.0
GREEN = "#76b900"

# Colour scales, taken from the ones the rest of the repository already uses
# so that a field looks the same wherever it is plotted: scripts/loaders.py
# LABELS for the rock, benchmark/compare.py FIELDS for the dynamic fields.
CMAP_PERM = "viridis"
CMAP_PORO = "cividis"
CMAP_SW = "Blues"
CMAP_PRES = "turbo"

# viz3d sets the box aspect from voxel counts, which is right for a roughly
# cubic model and wrong for this one: after decimation SPE10 is about
# 12 x 44 x 17 voxels, so the vertical would be drawn nearly three times the
# x extent when the reservoir is 1200 x 2200 x 170 ft -- a slab.  The aspect
# is therefore reset from the physical dimensions, with a modest vertical
# exaggeration so that the 85 layers remain legible.
ZEXAG = 4.0

# The SPE10 well pattern, in the form viz3d expects: one-based deck indices.
WELLS = [
    {"i": 30, "j": 110, "name": "INJ", "type": "inj"},
    {"i": 1,   "j": 1,   "name": "P1", "type": "prod"},
    {"i": 60,  "j": 1,   "name": "P2", "type": "prod"},
    {"i": 60,  "j": 220, "name": "P3", "type": "prod"},
    {"i": 1,   "j": 220, "name": "P4", "type": "prod"},
]


# ---------------------------------------------------------------------------
# encoding helpers
# ---------------------------------------------------------------------------
# Full-resolution figures and animation frames: nothing on the page is
# downsampled to save size.
REPORT_DPI = int(os.environ.get("NVRS_REPORT_DPI", "200"))


def fig_to_b64(fig, dpi=REPORT_DPI):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def frame_from_fig(fig, dpi=REPORT_DPI):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# frame rendering, in parallel
#
# A voxel block is a few seconds of matplotlib, and the two animations are
# forty-odd of them, so rendering them one after another dominates the run of
# this script.  benchmark/compare.py has the same problem with its panel
# sequence and solves it with a process pool, each worker writing a PNG that
# the parent collects; the same arrangement is used here.  Figures cannot be
# sent between processes, so the worker receives the array it must draw and
# returns a path.
# ---------------------------------------------------------------------------
def _frame_worker(job):
    """Render one animation frame.  Runs in a worker process."""
    (path, field, title, cbar_label, cmap, vmin, vmax, log, wells) = job
    fig = fig3d(field, title, cbar_label, cmap=cmap, vmin=vmin, vmax=vmax,
                log=log, wells=wells, figsize=(6.6, 4.4))
    fig.savefig(path, format="png", dpi=int(os.environ.get("NVRS_PANEL_DPI", "130")),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_frames(fields, times, label, cbar_label, cmap, vmin, vmax,
                  workers, tmpdir, tag):
    """Render one frame per field, in parallel, and return them as images.

    Falls back to rendering in this process when a pool cannot be created,
    which keeps the report working on a machine where forking is restricted.
    """
    from PIL import Image
    jobs, paths = [], []
    for i, F in enumerate(fields):
        p = os.path.join(tmpdir, "%s_%04d.png" % (tag, i))
        jobs.append((p, np.asarray(F, dtype=float),
                     "%s -- %.0f days" % (label, times[i]),
                     cbar_label, cmap, vmin, vmax, False, WELLS))
        paths.append(p)

    n = len(jobs)
    print("    %s: %d frames on %d worker(s)" % (tag, n, workers))
    done = 0
    if workers > 1 and n > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for _ in ex.map(_frame_worker, jobs):
                    done += 1
                    if done % 5 == 0 or done == n:
                        print("      %s %d/%d" % (tag, done, n))
        except Exception as exc:
            print("    [warn] pool unavailable (%s); rendering serially" % exc)
            done = 0
    if done < n:
        for j in jobs:
            if not os.path.exists(j[0]):
                _frame_worker(j)
            done += 1
            if done % 5 == 0 or done == n:
                print("      %s %d/%d" % (tag, done, n))

    return [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]


def gif_to_b64(frames, ms=420):
    from PIL import Image
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    buf = io.BytesIO()
    pal[0].save(buf, format="GIF", save_all=True, append_images=pal[1:],
                duration=ms, loop=0, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# three-dimensional cutaway rendering
# ---------------------------------------------------------------------------
def fig3d(F, title, cbar_label, cmap="jet", vmin=None, vmax=None, log=False,
          wells=None, figsize=(7.2, 4.6)):
    """One voxel block, rendered by viz3d -- the same path compare.py uses.

    viz3d.draw_voxels block-averages to a voxel budget, renders the whole
    block semi-transparently so the interior is visible, and overlays the
    wells as vertical sticks that continue above the top face. Using it here
    rather than a bespoke renderer keeps every three-dimensional figure in
    this repository looking and reading the same way.

    A logarithmic field is passed as log10 with the colourbar relabelled,
    since draw_voxels normalises linearly.
    """
    G = np.asarray(F, dtype=float)
    if log:
        G = np.log10(np.maximum(G, 1e-4))
        if vmin is not None:
            vmin = np.log10(max(vmin, 1e-4))
        if vmax is not None:
            vmax = np.log10(max(vmax, 1e-4))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    norm = viz3d.draw_voxels(ax, G, wells=wells or WELLS, vmin=vmin,
                             vmax=vmax, cmap=cmap, elev=28, azim=42,
                             well_fontsize=8)
    ax.set_box_aspect((NX * DX, NY * DY, NZ * DZ * ZEXAG))
    ax.set_title(title, fontsize=11, pad=4)
    m = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    m.set_array([])
    cb = fig.colorbar(m, ax=ax, fraction=0.026, pad=0.04, shrink=0.72)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return fig


def panel3d(fields, titles, suptitle, cbar_label, cmap="jet",
            vmin=None, vmax=None, log=False):
    """A row of voxel blocks sharing one colour scale."""
    n = len(fields)
    fig = plt.figure(figsize=(4.7 * n, 4.3))
    norm = None
    for idx, (F, t) in enumerate(zip(fields, titles)):
        G = np.asarray(F, dtype=float)
        if log:
            G = np.log10(np.maximum(G, 1e-4))
        ax = fig.add_subplot(1, n, idx + 1, projection="3d")
        lo = (np.log10(max(vmin, 1e-4)) if (log and vmin is not None)
              else vmin)
        hi = (np.log10(max(vmax, 1e-4)) if (log and vmax is not None)
              else vmax)
        norm = viz3d.draw_voxels(ax, G, wells=WELLS, vmin=lo, vmax=hi,
                                 cmap=cmap, elev=28, azim=42,
                                 axis_labels=False, well_fontsize=7)
        ax.set_box_aspect((NX * DX, NY * DY, NZ * DZ * ZEXAG))
        ax.set_title(t, fontsize=10, pad=2)
    m = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    m.set_array([])
    cb = fig.colorbar(m, ax=fig.axes, fraction=0.016, pad=0.02, shrink=0.72)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(suptitle, fontsize=12, y=0.99)
    return fig


def layer_panel(F, layers, title, cmap, norm, unit):
    fig, axes = plt.subplots(1, len(layers), figsize=(3.05 * len(layers), 4.2))
    if len(layers) == 1:
        axes = [axes]
    im = None
    for ax, k in zip(axes, layers):
        im = ax.imshow(F[k].T, origin="lower", aspect="auto",
                       cmap=cmap, norm=norm)
        ax.set_title("layer %d" % (k + 1), fontsize=9)
        ax.set_xlabel("i", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("j", fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def find_case(run_dir):
    for root, _dirs, files in os.walk(run_dir):
        for f in files:
            if f.upper().endswith(".UNRST"):
                return os.path.join(root, f[:-6])
    return None


def load_rock(case, run_dir):
    perm = permz = poro = porv = None
    if case is not None:
        for name, kw in (("perm", "PERMX"), ("permz", "PERMZ"),
                         ("poro", "PORO"), ("porv", "PORV")):
            try:
                v = case.init_field(kw)
            except Exception:
                v = None
            if name == "perm":
                perm = v
            elif name == "permz":
                permz = v
            elif name == "poro":
                poro = v
            else:
                porv = v
    if perm is not None and poro is not None:
        return perm, permz, poro, porv

    for c in (run_dir, os.path.join(run_dir, ".."),
              os.path.join(run_dir, "..", "runs", "SPE10")):
        if os.path.exists(os.path.join(c, "SPE10_perm.txt")):
            def rd(n):
                p = os.path.join(c, n)
                if not os.path.exists(p):
                    return None
                v = np.loadtxt(p)
                return (v[:NX * NY * NZ].reshape((NZ, NY, NX))
                        if v.size >= NX * NY * NZ else None)
            return (perm if perm is not None else rd("SPE10_perm.txt"),
                    permz if permz is not None else rd("SPE10_permz.txt"),
                    poro if poro is not None else rd("SPE10_poro.txt"),
                    porv)
    print("  [warn] rock not found in INIT or as generated text files")
    return perm, permz, poro, porv


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = [h.strip() for h in fh.readline().strip().split(",")]
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    cols = {}
    for i, name in enumerate(head):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[i]))
            except (ValueError, IndexError):
                vals.append(np.nan)
        cols[name] = np.asarray(vals)
    return cols


def time_axis(cols):
    """The time column, however it is spelled.

    NVRS writes "Time(DAY)"; the summary-vector fallback writes "TIME".
    """
    for k in cols:
        if k.upper().replace(" ", "").startswith("TIME"):
            return cols[k]
    for k in cols:
        if k.upper().startswith("DAYS"):
            return cols[k]
    return None


def match(cols, kw):
    """Columns carrying keyword kw, keyed by well name.

    NVRS names its CSV columns "<WELL>-WOPR(BBL/DAY)", and the summary
    vectors are "WOPR:WELL", so the keyword is looked for anywhere in the
    name and the well is whatever surrounds it.
    """
    out = {}
    for k, v in cols.items():
        ku = k.upper()
        if kw not in ku:
            continue
        if "-" in k and ku.index(kw) > 0:      # <WELL>-KEYWORD(UNIT)
            name = k.split("-", 1)[0].strip()
        elif ":" in k:                          # KEYWORD:WELL
            name = k.split(":", 1)[1].strip()
        else:
            name = k.strip()
        if name:
            out[name] = v
    return out


def deck_injection(run_dir):
    """(total injection rate [STB/day], Bw) from the deck the run used.

    The CSV carries producers only, so the injected volume for the material
    balance comes from the deck.  Injection here is at a constant rate for
    the whole schedule, which is what makes this a valid independent figure
    rather than a restatement of the simulator's own accounting.
    """
    cands = []
    for root, _d, files in os.walk(run_dir):
        for f in files:
            if f.upper().endswith(".DATA"):
                cands.append(os.path.join(root, f))
    cands += [os.path.join(run_dir, "..", "runs", "SPE10", "SPE10.DATA")]
    for path in cands:
        if not os.path.exists(path):
            continue
        q, bw, in_wells = 0.0, 1.0, False
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.split("--")[0].strip()
                if not line:
                    continue
                up = line.upper()
                if up.startswith("BW"):
                    try:
                        bw = float(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                if up.startswith("WELLS") or up.startswith("WELSPECS"):
                    in_wells = True
                    continue
                if in_wells:
                    if line.startswith("/"):
                        break
                    tk = line.replace("/", " ").split()
                    if len(tk) >= 4:
                        try:
                            if int(tk[2]) > 0:          # injector
                                q += float(tk[3])
                        except ValueError:
                            pass
        if q > 0:
            print("  deck: %s -- injection %.4g STB/day, Bw %.4g"
                  % (os.path.basename(path), q, bw))
            return q, bw
    return None, 1.0


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def section_rock(perm, permz, poro):
    parts = []
    layers = [0, NZ // 3, 2 * NZ // 3, NZ - 1]

    if perm is not None:
        lo = max(float(perm[perm > 0].min()), 1e-4)
        norm = LogNorm(vmin=lo, vmax=float(perm.max()))
        parts.append(("Permeability in three dimensions",
                      fig_to_b64(fig3d(perm,
                                       "Horizontal permeability",
                                       "log10 Kx [mD]", cmap=CMAP_PERM,
                                       log=True)),
                      "The Brent-sequence field. The upper part of the model "
                      "is the Tarbert formation, a prograding near-shore "
                      "sequence; below it the Upper Ness is fluvial, and its "
                      "channels are the streaks that carry the flood. The "
                      "colour scale is logarithmic because the contrast "
                      "across this model is %.2g."
                      % (float(perm.max()) / max(float(perm.min()), 1e-30))))

        parts.append(("Permeability by layer",
                      layer_panel(perm, layers, "", cm.viridis, norm, "mD"),
                      "Areal maps at four depths. The character changes with "
                      "depth: smooth and correlated in the Tarbert, "
                      "channelised in the Upper Ness."))

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.4))
        a1.hist(np.log10(np.maximum(perm.ravel(), 1e-4)), bins=90,
                color=GREEN, alpha=0.85)
        a1.set_xlabel(r"$\log_{10} K_x$ [mD]", fontsize=9)
        a1.set_ylabel("cells", fontsize=9)
        a1.set_title("permeability distribution", fontsize=10)
        a1.tick_params(labelsize=8)
        km = np.array([perm[k].mean() for k in range(perm.shape[0])])
        a2.plot(km, np.arange(1, perm.shape[0] + 1), color=GREEN, lw=1.6)
        a2.set_xscale("log")
        a2.invert_yaxis()
        a2.set_xlabel("layer mean $K_x$ [mD]", fontsize=9)
        a2.set_ylabel("layer", fontsize=9)
        a2.set_title("vertical profile", fontsize=10)
        a2.grid(alpha=0.25)
        a2.tick_params(labelsize=8)
        fig.tight_layout()
        parts.append(("Permeability statistics", fig_to_b64(fig),
                      "The distribution is multi-modal and spans seven "
                      "orders of magnitude, which is what makes this a "
                      "solver benchmark rather than merely a large model."))

    if poro is not None:
        pn = Normalize(vmin=float(poro.min()), vmax=float(poro.max()))
        parts.append(("Porosity in three dimensions",
                      fig_to_b64(fig3d(poro, "Porosity", "porosity [-]",
                                       cmap=CMAP_PORO)),
                      "Porosity is an independent measurement in this "
                      "dataset rather than a function of permeability, which "
                      "is why the deck uses POROFILE. Mean %.4f, range %.3g "
                      "to %.3g." % (float(poro.mean()), float(poro.min()),
                                    float(poro.max()))))
        parts.append(("Porosity by layer",
                      layer_panel(poro, layers, "", cm.cividis, pn, "-"),
                      "The same four depths as above."))

    if permz is not None and perm is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(perm > 0, permz / perm, np.nan)
        good = ratio[np.isfinite(ratio) & (ratio > 0)]
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.hist(np.log10(good), bins=80, color="#4E86C6", alpha=0.85)
        ax.set_xlabel(r"$\log_{10}(K_z/K_x)$", fontsize=9)
        ax.set_ylabel("cells", fontsize=9)
        ax.set_title("vertical to horizontal permeability ratio", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        parts.append(("Anisotropy", fig_to_b64(fig),
                      "The ratio is not a constant, which is why the deck "
                      "supplies PERMZFILE rather than KZMULT. Cells are "
                      "20 x 10 x 2 ft, so the transmissibility ratio is "
                      "larger still: the system is strongly anisotropic in "
                      "the vertical, and that is what the CPR second stage "
                      "has to contend with."))
    return parts


def section_fields(case, times, sw, pr, workers, tmpdir):
    parts = []
    nsteps = len(sw)
    mid = nsteps // 2

    # --- water saturation, three voxel blocks sharing one scale -----------
    parts.append((
        "Water saturation in three dimensions",
        fig_to_b64(panel3d([sw[0], sw[mid], sw[-1]],
                           ["initial",
                            "%.0f days" % times[mid],
                            "%.0f days" % times[-1]],
                           "Water saturation", "Sw [-]", cmap=CMAP_SW,
                           vmin=0.2, vmax=0.8)),
        "The block is drawn semi-transparently, so the flood is visible "
        "inside it rather than only on the faces. The injector is the black "
        "stick at (30,110) and the producers are the red sticks at the "
        "corners. The flood does not advance as a front: it follows the "
        "high-permeability channels, which is precisely the behaviour this "
        "benchmark exists to test."))

    # --- animation --------------------------------------------------------
    try:
        frames = render_frames(sw, times, "Water saturation", "Sw [-]",
                               CMAP_SW, 0.2, 0.8, workers, tmpdir, "swat")
        if len(frames) > 1:
            parts.append(("Flood evolution", gif_to_b64(frames),
                          "One frame per report step over the full 2000 "
                          "days.", "gif"))
    except Exception as exc:
        print("  [warn] saturation animation skipped: %s" % exc)

    if pr:
        allp = np.concatenate([pr[0].ravel(), pr[-1].ravel()])
        plo = float(np.percentile(allp, 1))
        phi = float(np.percentile(allp, 99))
        parts.append((
            "Pressure in three dimensions",
            fig_to_b64(panel3d([pr[0], pr[min(mid, len(pr) - 1)], pr[-1]],
                               ["initial",
                                "%.0f days" % times[mid],
                                "%.0f days" % times[len(pr) - 1]],
                               "Pressure", "pressure [psia]",
                               cmap=CMAP_PRES, vmin=plo, vmax=phi)),
            "The injector is rate-controlled and the producers are held on "
            "bottom-hole pressure, so the pressure field is a map of the "
            "resistance of the rock between them."))

        try:
            frames = render_frames(pr, times, "Pressure", "pressure [psia]",
                                   CMAP_PRES, plo, phi, workers, tmpdir,
                                   "pres")
            if len(frames) > 1:
                parts.append(("Pressure evolution", gif_to_b64(frames),
                              "The transient established in the opening "
                              "steps, and its slow evolution as the flood "
                              "changes the mobility field.", "gif"))
        except Exception as exc:
            print("  [warn] pressure animation skipped: %s" % exc)
    return parts


def section_wells(cols):
    if not cols:
        return []
    t = time_axis(cols)
    if t is None:
        return []
    parts = []
    panels = (("WOPR", "oil rate [STB/d]", "Oil production rate (WOPR)"),
              ("WWPR", "water rate [STB/d]", "Water production rate (WWPR)"),
              ("WWCT", "water cut [-]", "Water cut (WWCT)"),
              ("WBHP", "BHP [psia]", "Bottom-hole pressure (WBHP)"))
    have = [(p, lab, ttl) for p, lab, ttl in panels if match(cols, p)]
    if not have:
        return parts

    rows = (len(have) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11.0, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (p, lab, ttl) in zip(axes, have):
        for name, v in sorted(match(cols, p).items()):
            ax.plot(t, v, lw=1.5, label=name)
        ax.set_xlabel("time [days]", fontsize=9)
        ax.set_ylabel(lab, fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, frameon=False, ncol=2)
    for ax in axes[len(have):]:
        ax.axis("off")
    fig.tight_layout()
    parts.append(("Well response", fig_to_b64(fig),
                  "Breakthrough is the arrival of the channelled flood, and "
                  "the four producers break through at different times "
                  "because the channels do not reach them equally. The water "
                  "cut curves are the clearest statement of that."))
    return parts


def section_balance(cols, porv, sw, times, poro=None, q_inj=None, bw=1.0):
    """Recovery factor and material balance, from the fields themselves."""
    parts, rows = [], []
    if porv is None and poro is not None:
        porv = poro * (DX * DY * DZ / 5.61458)      # ft^3 -> reservoir bbl
    if porv is None or not sw:
        return parts, rows

    pv = porv.astype(np.float64)
    pvt = float(pv.sum())
    oip = np.array([float((pv * (1.0 - s)).sum()) for s in sw])
    wip = np.array([float((pv * s).sum()) for s in sw])
    rf = 1.0 - oip / oip[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 3.6))
    a1.plot(times, 100.0 * rf, color=GREEN, lw=2.0)
    a1.set_xlabel("time [days]", fontsize=9)
    a1.set_ylabel("recovery factor [% of OOIP]", fontsize=9)
    a1.set_title("Oil recovery", fontsize=10)
    a1.grid(alpha=0.25)
    a1.tick_params(labelsize=8)

    a2.plot(times, oip / oip[0], lw=1.8, color="#7C4F20", label="oil in place")
    a2.plot(times, wip / pvt, lw=1.8, color="#3F7FC1",
            label="water saturation, field average")
    a2.set_xlabel("time [days]", fontsize=9)
    a2.set_ylabel("fraction", fontsize=9)
    a2.set_title("In-place volumes", fontsize=10)
    a2.grid(alpha=0.25)
    a2.tick_params(labelsize=8)
    a2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    parts.append(("Recovery", fig_to_b64(fig),
                  "Computed from the saturation field and the pore volume "
                  "rather than read from a report: oil in place at each step "
                  "is the sum over cells of pore volume times one minus "
                  "water saturation, and the recovery factor is its "
                  "fractional decline from the initial state. Final recovery "
                  "%.2f%% of %.4g reservoir bbl originally in place."
                  % (100.0 * rf[-1], oip[0])))

    # --- material balance -------------------------------------------------
    # Change in water in place, from the saturation field, against water
    # injected less water produced, from the deck and the well rates.  The
    # two are computed from independent sources, so their agreement is a
    # statement about conservation rather than a restatement of one number.
    dwip = wip - wip[0]
    net = None
    if cols is not None:
        t = time_axis(cols)
        if t is not None:
            qi = None
            inj = match(cols, "WWIR")
            if inj:
                qi_rate = sum(inj.values())
                qi = np.concatenate(([0.0], np.cumsum(
                    0.5 * (qi_rate[1:] + qi_rate[:-1]) * np.diff(t))))
            elif q_inj:
                qi = q_inj * t            # constant rate for the schedule
            prod = match(cols, "WWPR")
            if qi is not None:
                if prod:
                    qp_rate = sum(prod.values())
                    qp = np.concatenate(([0.0], np.cumsum(
                        0.5 * (qp_rate[1:] + qp_rate[:-1]) * np.diff(t))))
                else:
                    qp = np.zeros_like(qi)
                net = np.interp(times, t, (qi - qp) * bw)

    if net is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.where(np.abs(net) > 0,
                           (dwip - net) / np.maximum(np.abs(net), 1e-30), 0.0)
        fig, (b1, b2) = plt.subplots(1, 2, figsize=(11.0, 3.6))
        b1.plot(times, dwip / 1e6, lw=2.0, color="#3F7FC1",
                label="change in water in place")
        b1.plot(times, net / 1e6, lw=1.6, ls="--", color="#222",
                label="net injected (injected - produced)")
        b1.set_xlabel("time [days]", fontsize=9)
        b1.set_ylabel("volume [MMbbl]", fontsize=9)
        b1.set_title("Water material balance", fontsize=10)
        b1.grid(alpha=0.25)
        b1.tick_params(labelsize=8)
        b1.legend(fontsize=8, frameon=False)

        b2.plot(times[1:], 100.0 * err[1:], lw=1.8, color="#C62828")
        b2.set_xlabel("time [days]", fontsize=9)
        b2.set_ylabel("closure error [% of net injected]", fontsize=9)
        b2.set_title("Material balance closure", fontsize=10)
        b2.grid(alpha=0.25)
        b2.tick_params(labelsize=8)
        fig.tight_layout()
        parts.append(("Material balance", fig_to_b64(fig),
                      "The change in water in place should equal the water "
                      "injected less the water produced. The two are "
                      "computed independently -- one from the saturation "
                      "field, one from the well rates -- so their agreement "
                      "is a check that the simulation conserved mass. Final "
                      "closure error %.3f%% of the net injected volume."
                      % (100.0 * err[-1])))
        rows.append(("Material balance closure, final",
                     "%.3f %%" % (100.0 * err[-1])))

    rows.append(("Oil in place, initial [reservoir bbl]", "%.4g" % oip[0]))
    rows.append(("Oil in place, final [reservoir bbl]", "%.4g" % oip[-1]))
    rows.append(("Recovery factor, final", "%.2f %%" % (100.0 * rf[-1])))
    rows.append(("Pore volume [reservoir bbl]", "%.4g" % pvt))
    return parts, rows


def run_info_rows(run_dir, log_path, times, perm=None, poro=None, porv=None):
    """What was run, on what, when, and over which period.

    The simulator counts in days from the start of the schedule; the SPE10
    reference deck dates that start at 1 February 1967, so the calendar
    range is given alongside the elapsed days rather than in place of them.
    """
    import datetime

    rows = []
    ncell = NX * NY * NZ
    rows.append(("Model", "SPE10 model 2 (Christie &amp; Blunt, SPE 72469)"))
    rows.append(("Grid", "%d &times; %d &times; %d = %s cells"
                 % (NX, NY, NZ, "{:,}".format(ncell))))
    rows.append(("Cell size", "%g &times; %g &times; %g ft" % (DX, DY, DZ)))
    rows.append(("Reservoir extent", "%s &times; %s &times; %s ft"
                 % ("{:,.0f}".format(NX * DX), "{:,.0f}".format(NY * DY),
                    "{:,.0f}".format(NZ * DZ))))
    rows.append(("Formulation", "Two-phase black oil (oil and water)"))
    rows.append(("Wells", "1 injector at (30,110), 4 producers at the "
                          "corners, perforated over all 85 layers"))
    if perm is not None:
        rows.append(("Permeability range", "%.3g to %.4g mD (contrast %.2g)"
                     % (float(perm.min()), float(perm.max()),
                        float(perm.max()) / max(float(perm.min()), 1e-30))))
    if poro is not None:
        rows.append(("Porosity range", "%.3g to %.3g (mean %.4f)"
                     % (float(poro.min()), float(poro.max()),
                        float(poro.mean()))))
    if porv is not None:
        rows.append(("Pore volume", "%.4g reservoir bbl"
                     % float(np.asarray(porv).sum())))

    if times is not None and len(times):
        span = float(times[-1]) - float(times[0])
        start = datetime.date(1967, 2, 1)
        d0 = start + datetime.timedelta(days=float(times[0]))
        d1 = start + datetime.timedelta(days=float(times[-1]))
        rows.append(("Simulated period",
                     "%.0f days &nbsp;(%s to %s, dated from the reference "
                     "deck's start of 1 February 1967)"
                     % (span, d0.strftime("%d %b %Y"),
                        d1.strftime("%d %b %Y"))))
        rows.append(("Report steps", "%d, %.0f days apart"
                     % (len(times) - 1 if len(times) > 1 else 1,
                        span / max(len(times) - 1, 1))))

    # The run itself, from the header run_spe10.sh writes.
    if log_path and os.path.exists(log_path):
        want = (("started", "Run started"), ("host", "Host"),
                ("exe", "Executable"), ("ranks", "MPI ranks"),
                ("wall", "Wall clock, including start-up and output"))
        got = {}
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                for key, _lab in want:
                    if s.startswith(key) and ":" in s:
                        got.setdefault(key, s.split(":", 1)[1].strip())
                if s.startswith("finished"):
                    got.setdefault("finished", s.split(":", 1)[1].strip())
        for key, lab in want:
            if key in got:
                rows.append((lab, got[key]))
        if "finished" in got:
            rows.append(("Run finished", got["finished"]))
    else:
        rows.append(("Report generated",
                     datetime.datetime.now().strftime("%d %b %Y %H:%M")))
    return rows


def performance_rows(run_dir):
    meta = None
    for root, _d, files in os.walk(run_dir):
        if "meta.json" in files:
            try:
                meta = json.load(open(os.path.join(root, "meta.json")))
            except Exception:
                meta = None
            break
    rows = []
    if meta:
        for key, label in (("wall_seconds", "Wall clock [s]"),
                           ("linear_seconds", "Linear solve [s]"),
                           ("assembly_seconds", "Assembly [s]"),
                           ("amg_setup_seconds", "AMG setup [s]"),
                           ("amg_solve_seconds", "AMG V-cycles [s]"),
                           ("stage2_seconds", "CPR stage 2 [s]"),
                           ("newton_iterations", "Newton iterations"),
                           ("krylov_iterations", "Krylov iterations")):
            if key in meta:
                v = meta[key]
                rows.append((label,
                             "%.1f" % v if isinstance(v, float) else str(v)))
    return rows


def read_log_tail(path, nlines=400):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return "".join(fh.readlines()[-nlines:])


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
# Taken from benchmark/compare.py so the two reports are the same document
# in appearance: the same typography, the same NVIDIA-green rules, the same
# note blocks and figure captions, and the same dark-mode treatment, which
# follows the reader's system preference through color-scheme.
CSS = """
 :root { color-scheme: light dark; }
 body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif; margin: 0 auto; max-width: 1500px;
        padding: 1.5rem; line-height: 1.5; }
 h1 { border-bottom: 3px solid #76b900; padding-bottom: .3rem; }
 h2 { border-bottom: 1px solid #ccc; padding-bottom: .2rem; margin-top: 2.2rem; }
 h3 { margin-top: 1.6rem; color: #555; }
 table { border-collapse: collapse; margin: 1rem 0; font-size: .88rem; }
 th, td { border: 1px solid #bbb; padding: .3rem .6rem; text-align: right; }
 th { background: rgba(118,185,0,.16); text-align: left; }
 td:first-child, th:first-child { text-align: left; }
 figure { margin: 0 0 1.2rem 0; }
 /* compare.py fills the column because its panels are 15 inches wide; the
    figures here are a third of that, so they are shown at their natural
    size and merely prevented from overflowing. */
 figure img { max-width: 100%; width: auto; height: auto;
        border: 1px solid #ddd; border-radius: 4px; background: #fff;
        display: block; }
 figcaption { font-size: .85rem; color: #666; padding-top: .3rem; }
 .grid { display: grid; grid-template-columns: repeat(auto-fit,
         minmax(430px, 1fr)); gap: 1.1rem; }
 .wide { display: block; }
 .note { background: rgba(118,185,0,.09); border-left: 4px solid #76b900;
         padding: .7rem 1rem; margin: 1rem 0; font-size: .92rem; }
 .byline { font-size: .9rem; color: #666; margin: .2rem 0 1.2rem 0; }
 code { background: rgba(127,127,127,.16); padding: .1rem .3rem;
        border-radius: 3px; }
 pre { background: rgba(127,127,127,.12); padding: .8rem 1rem;
       border-radius: 4px; overflow-x: auto; font-size: .78rem;
       line-height: 1.45; max-height: 32rem; }
 @media (prefers-color-scheme: dark) {
   body { background: #16181c; color: #e6e6e6; }
   th, td { border-color: #444; }
   h2 { border-color: #444; }
   h3 { color: #b8b8b8; }
   figure img { border-color: #333; }
   figcaption, .byline { color: #9a9a9a; }
 }
"""


def build(run_dir, out_path, log_path, workers=0, keep_frames=False):
    workers = workers or min(os.cpu_count() or 1, 12)
    tmpdir = os.path.join(run_dir, "frames")
    os.makedirs(tmpdir, exist_ok=True)

    base = find_case(run_dir)
    case = ecl.EclipseCase(base) if base else None
    if case is None:
        print("  [warn] no .UNRST under %s -- the deck needs ECLOUT 1"
              % run_dir)
    else:
        print("  reading %s" % base)

    perm, permz, poro, porv = load_rock(case, run_dir)

    times, sw, pr = np.array([]), [], []
    if case is not None:
        times = np.asarray(case.restart_times(), dtype=float).ravel()
        sw_all = case.restart_field("SWAT")
        pr_all = case.restart_field("PRESSURE")
        if sw_all is not None and times.size:
            n = min(len(times), len(sw_all))
            times = times[:n]
            sw = [sw_all[i] for i in range(n)]
            pr = ([pr_all[i] for i in range(min(n, len(pr_all)))]
                  if pr_all is not None else [])
            print("  fields: %d report steps, %.0f to %.0f days"
                  % (n, times[0], times[-1]))

    cols = None
    for root, _d, files in os.walk(run_dir):
        if "RSM_FVM.csv" in files:
            cols = read_csv(os.path.join(root, "RSM_FVM.csv"))
            print("  wells: RSM_FVM.csv (%d columns)" % len(cols))
            break
    if not cols and case is not None:
        try:
            t, series = case.summary()
            if t is not None and len(t):
                cols = {"TIME": np.asarray(t, dtype=float)}
                cols.update({k: np.asarray(v, dtype=float)
                             for k, v in series.items()})
                print("  wells: summary vectors (%d series)" % len(series))
        except Exception as exc:
            print("  [warn] summary vectors unavailable: %s" % exc)

    print("  rendering figures (the 3-D animations take a minute)")
    blocks = [("Rock", section_rock(perm, permz, poro))]
    if sw:
        blocks.append(("Pressure and saturation",
                       section_fields(case, times, sw, pr, workers, tmpdir)))
    blocks.append(("Well response", section_wells(cols)))
    q_inj, bw = deck_injection(run_dir)
    bal_parts, bal_rows = section_balance(cols, porv, sw, times, poro,
                                          q_inj, bw)
    blocks.append(("Recovery and material balance", bal_parts))
    perf_rows = performance_rows(run_dir)
    log_text = read_log_tail(log_path)

    info_rows = run_info_rows(run_dir, log_path, times, perm, poro, porv)

    html = [
        '<!doctype html>',
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>SPE10 model 2 &mdash; NVRS-CUDA</title>",
        "<style>%s</style></head><body>" % CSS,
        "<h1>SPE10 model 2 &mdash; NVRS-CUDA</h1>",
        '<p class="byline">Clement Etienam &middot; Tsubasa Onishi &middot; '
        "Oleg Ovcharenko &middot; Nick Luiken &middot; Issam Said</p>",
        '<div class="note">The Tenth SPE Comparative Solution Project, '
        "model 2: the standard test of a solver's tolerance of "
        "heterogeneity, built as an upscaling exercise that most of the "
        "original participants could not run on the fine grid. This page "
        "reports one run of it on NVRS-CUDA &mdash; the rock, the flood, "
        "the wells, the recovery and the material balance.</div>",
    ]

    # --- run information ---------------------------------------------------
    html.append("<h2>Run information</h2>")
    html.append("<table><tr><th>Item</th><th>Value</th></tr>")
    for label, value in info_rows:
        html.append("<tr><td>%s</td><td>%s</td></tr>" % (label, value))
    html.append("</table>")

    # --- figures -----------------------------------------------------------
    for name, parts in blocks:
        if not parts:
            continue
        html.append("<h2>%s</h2>" % name)
        for item in parts:
            title, payload, note = item[0], item[1], item[2]
            kind = item[3] if len(item) > 3 else "png"
            html.append("<h3>%s</h3>" % title)
            html.append('<figure class="wide">')
            html.append('<img src="data:image/%s;base64,%s" alt="%s">'
                        % ("gif" if kind == "gif" else "png", payload, title))
            html.append("<figcaption>%s</figcaption>" % note)
            html.append("</figure>")

    if bal_rows or perf_rows:
        html.append("<h2>Summary</h2>")
        html.append("<table><tr><th>Quantity</th><th>Value</th></tr>")
        for label, value in bal_rows + perf_rows:
            html.append("<tr><td>%s</td><td>%s</td></tr>" % (label, value))
        html.append("</table>")

    if log_text:
        html.append("<h2>Run log</h2>")
        html.append('<div class="note">The tail of the run as it was '
                    "printed. The complete log is beside this page.</div>")
        html.append("<pre>%s</pre>"
                    % log_text.replace("&", "&amp;").replace("<", "&lt;"))

    html.append('<hr><p style="font-size:.8rem;color:#888">Generated by '
                "scripts/spe10_report.py &middot; NVRS-CUDA &middot; "
                "Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko, Nick Luiken, "
                "Issam Said"
                "<br>Dataset: Christie &amp; Blunt, SPE 72469 (2001), public "
                "domain; ECLIPSE include form redistributed by the Open "
                "Porous Media initiative under ODbL.</p>")
    html.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))
    print("wrote %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))

    # The frames are embedded in the page, so the intermediates are scratch.
    if not keep_frames:
        try:
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)
        except OSError:
            pass


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    run_dir, out, log = argv[1], "spe10.html", None
    workers, keep = 0, False
    args = argv[2:]
    while args:
        a = args.pop(0)
        if a == "--out":
            out = args.pop(0)
        elif a == "--log":
            log = args.pop(0)
        elif a == "--workers":
            workers = int(args.pop(0))
        elif a == "--keep-frames":
            keep = True
        else:
            raise SystemExit("unknown option %s" % a)
    build(run_dir, out, log, workers, keep)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
