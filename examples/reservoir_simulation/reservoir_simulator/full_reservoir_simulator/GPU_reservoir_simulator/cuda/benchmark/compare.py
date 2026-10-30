#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# compare.py — OPM Flow vs NVRS-CUDA, side by side.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# ---------------------------------------------------------------------------
# What this produces
# ---------------------------------------------------------------------------
# A single self-contained compare.html holding, per benchmark case:
#
#   * one 3 x 4 panel of 3-D voxel renderings per report step — columns are
#     OPM Flow, NVRS-CUDA and the difference, rows are pressure, water, oil and
#     gas saturation — plus an animation of the sequence;
#   * one figure per well quantity (WOPR, WWPR, WWCT, WGOR, WBHP) overlaying
#     both simulators for every producer, with the per-well difference below;
#   * a wall-clock comparison and a per-step, per-field error table.
#
# It reads ECLIPSE output on BOTH sides — Flow's own, and the EGRID/UNRST/
# SMSPEC/UNSMRY that NVRS writes when ECLOUT is set — so the two runs go
# through one reader and one code path, and nothing can differ because of how
# a file was parsed.
#
# This is deliberately a SECOND report.  scripts/plot_results.py still owns
# index.html for a single NVRS run; nothing here touches it.
#
#   python3 compare.py                       # discover runs/ next to this file
#   python3 compare.py --runs path/to/runs
# ---------------------------------------------------------------------------
import argparse
import base64
import glob
import json
import math
import mimetypes
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

# No VTK by default.  PyVista pulls in VTK, and on the compute nodes VTK
# reached for an X server that is not there ("bad X server connection") and
# then corrupted the heap inside the parallel panel workers
# ("malloc_consolidate(): unaligned fastbin chunk"), which hung the report.
# Every field and rock figure is drawn with matplotlib instead.  The PyVista
# renderer is kept only behind NVRS_USE_PYVISTA=1, for a workstation with a
# working display, and is never imported otherwise.
pv = None
if os.environ.get("NVRS_USE_PYVISTA", "0") == "1":
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
    except Exception:
        pv = None

import ecl
import plot_norne
import viz3d

# NVRS_REF_LABEL / NVRS_CMP_LABEL: the same report for two NVRS runs, the
# reference in the "opm" slot (precision_fp64_vs_mixed.sh puts the all-FP64
# run there and the mixed run beside it).  Every figure text and the page are
# relabelled; the data paths are unchanged.
_RELABEL = []
if os.environ.get("NVRS_REF_LABEL"):
    _ref = os.environ["NVRS_REF_LABEL"]
    _cmp = os.environ.get("NVRS_CMP_LABEL", "NVRS-CUDA")
    _RELABEL = [("OPM Flow vs NVRS-CUDA", "%s vs %s" % (_ref, _cmp)),
                ("OPM - NVRS", "%s - %s" % (_ref, _cmp)),
                ("OPM − NVRS", "%s − %s" % (_ref, _cmp)),
                ("OPM Flow", _ref),
                ("NVRS-CUDA (AmgX)", _cmp),
                ("NVRS-CUDA (AMGX)", _cmp),
                ("NVRS-CUDA", _cmp)]
    import matplotlib.text as _mtext
    _set_text = _mtext.Text.set_text

    def _relabelled(self, s):
        _set_text(self, relabel(s) if isinstance(s, str) else s)
    _mtext.Text.set_text = _relabelled


def relabel(s):
    for a, b in _RELABEL:
        s = s.replace(a, b)
    return s

# Rows of the field panel.  "gas" is dropped automatically when the case has
# no gas phase.
# One colour scale for every field: jet, as asked for.
#
# The rows used to carry a colour each -- blue for water, green for oil, red
# for gas -- which reads well on its own and badly in a grid, because the eye
# then compares a shade of blue against a shade of green and cannot.  One map
# across all four rows means a value has one colour wherever it appears.
# Turbo retains the familiar rainbow reservoir-viewer appearance while being
# smoother and more perceptually uniform than the legacy jet map.  Difference
# fields always use a separate zero-centred red-white-blue palette below.
# The colour maps the reports used before the September merge.
FIELDS = [("pressure", "Pressure (psia)", "jet"),
          ("water", "Water saturation", "jet"),
          ("oil", "Oil saturation", "jet"),
          ("gas", "Gas saturation", "jet")]
DIFF_CMAP = "RdBu_r"

# Well quantities.  Plot WGPR directly so gas agreement remains visible even
# when a producer is shut or watered out.  WGOR is also derived from WGPR/WOPR
# on both sides, but only while oil flow is large enough to define a useful
# ratio; dividing numerical shut-in noise by numerical shut-in noise otherwise
# produces spectacular and meaningless 1e20-scale spikes.
# One colour per simulator, used everywhere a simulator appears.
#
# A reader should not have to check a legend twice.  Red is the reference,
# OPM Flow, everywhere it appears; each NVRS backend keeps its own colour
# (AmgX green, HYPRE blue, Ginkgo amber/purple/teal) in the curves, the
# accuracy bars and the timing charts alike.
SRC_COLOUR = {
    "OPM Flow":           "#d62728",   # red -- the reference
    "NVRS-CUDA (AmgX)":   "#76B900",   # NVIDIA green
    "NVRS-CUDA (HYPRE)":  "#1f77b4",   # blue
    "NVRS-CUDA (Ginkgo FP16)": "#EF9100",   # amber
    "NVRS-CUDA (Ginkgo FP32)": "#952FC6",   # purple
    "NVRS-CUDA (Ginkgo FP16 + adaptive bJ)": "#1DBBA4",   # teal
    "NVRS-CUDA (AmgX, FP64 operator)": "#3B5D00",   # dark green
    "NVRS-CUDA":          "#76B900",   # NVIDIA green
}
SRC_STYLE = {
    "OPM Flow":           ("-", 1.25),
    "NVRS-CUDA (AmgX)":   ("--", 1.10),
    "NVRS-CUDA (HYPRE)":  ("-.", 1.10),
    "NVRS-CUDA (Ginkgo FP16)": (":", 1.30),
    "NVRS-CUDA (Ginkgo FP32)": ((0, (5, 2, 1, 2)), 1.20),
    "NVRS-CUDA (Ginkgo FP16 + adaptive bJ)": ((0, (1, 1)), 1.40),
    "NVRS-CUDA (AmgX, FP64 operator)": ((0, (8, 3)), 1.10),
    "NVRS-CUDA":          ("--", 1.10),
}


def src_colour(label):
    return SRC_COLOUR.get(label, "#7f7f7f")


def src_style(label):
    """(linestyle, linewidth).  Pass the style as linestyle=, never as the
    positional format string: the Ginkgo FP32 and adaptive-bJ styles are dash
    tuples, which plot() reads as a third data array ("Input could not be
    cast to an at-least-1D NumPy array"), and that removed every well curve
    and the injection plot from the Norne report."""
    return SRC_STYLE.get(label, (":", 1.4))


WELL_QTY = [("WOPR", "Oil rate (STB/day)", None),
            ("WWPR", "Water rate (STB/day)", None),
            ("WWCT", "Water cut (fraction)", ("WWPR", "WOPR", "cut")),
            ("WGPR", "Gas rate (MSCF/day)", None),
            ("WGOR", "Gas-oil ratio (MSCF/STB)", ("WGPR", "WOPR", "ratio")),
            ("WBHP", "Bottom-hole pressure (psia)", None)]

# Default budget per report.  A self-contained page is only useful if it can
# actually be opened, and base64 inflates every figure by a third, so the budget
# is enforced rather than hoped for.  --max-mb overrides it.
#
# It is 20 MB rather than 10 because the evolution animation has to fit,
# and it is the first thing anyone looks at.  Fifteen is enough for the whole
# page once the animation is carried as H.264 rather than as a GIF; see the
# note where the movies are chosen.  Pass --max-mb to raise or lower it.
EMBED_BUDGET = 60 * 1024 * 1024

# Resolution of every static figure in the report.  The figures used to be
# written at 95 to 130 dpi, which made panel text soft once the page scaled
# them; 200 keeps them sharp at full width.  NVRS_REPORT_DPI overrides, and
# the embed budget above was raised so the sharper figures still fit.
REPORT_DPI = int(os.environ.get("NVRS_REPORT_DPI", "200"))
# The per-step 3-D panels also feed the animation, which grows with the square
# of this, so they get their own, lower, setting.
PANEL_DPI = int(os.environ.get("NVRS_PANEL_DPI", "130"))


def log(msg):
    print("[cmp] " + msg, flush=True)


def guarded(label):
    """A renderer that fails logs one line and does not kill the report."""
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:
                # Where it failed, not only what: a skipped section is
                # otherwise one line with nothing to act on.
                tb = traceback.extract_tb(exc.__traceback__)
                at = ("%s:%d in %s" % (os.path.basename(tb[-1].filename),
                                       tb[-1].lineno, tb[-1].name)
                      if tb else "?")
                log("  ! %s skipped: %s: %s  (at %s)"
                    % (label, type(exc).__name__, exc, at))
                return None
        return wrapper
    return deco


def human(x, unit=""):
    """Compact number formatting: 1.23 M, 4.56 k, and so on."""
    if x is None or not np.isfinite(x):
        return "-"
    ax = abs(x)
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if ax >= scale:
            return "%.2f %s%s" % (x / scale, suffix, unit)
    return ("%.3g %s" % (x, unit)).strip()


def data_uri(path, budget, force=False):
    """Inline a file as a data: URI so the report survives being copied off.

    force=True embeds regardless of the budget.  Used for the animation, which
    is the point of the page rather than an extra on it: a report that quietly
    lost it is worse than a report that is larger than intended.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if not force and budget["used"] + size > budget.get("limit", EMBED_BUDGET):
        budget["skipped"] += 1
        return None
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    budget["used"] += size
    return "data:%s;base64,%s" % (mime, b64)


# ===========================================================================
# Loading
# ===========================================================================
def find_bases(rundir):
    """(opm_base, nvrs_base) for a runs/<CASE> directory, or (None, None).

    A "base" is the ECLIPSE case path without extension.  Both sides are found
    by looking for the EGRID, which each simulator writes exactly once.
    """
    def first_egrid(*subs):
        # Several names, because a case may hold more than one NVRS run: the
        # same simulator with AmgX and with HYPRE is two runs, and the field
        # comparison needs one of them, not both.  First match wins.
        for sub in subs:
            hits = sorted(glob.glob(os.path.join(rundir, sub, "**", "*.EGRID"),
                                    recursive=True))
            if hits:
                return os.path.splitext(hits[0])[0]
        return None
    # NVRS_FIELD_ARM=nvrs_hypre (say) puts that run first, so the 3-D field
    # panels and the movie show it against Flow.
    pref = [os.environ["NVRS_FIELD_ARM"]] if os.environ.get("NVRS_FIELD_ARM") else []
    return (first_egrid("opm"),
            first_egrid(*pref, "nvrs", "nvrs_amgx", "nvrs_hypre", "nvrs_ginkgo",
                        "nvrs_ginkgo32", "nvrs_ginkgo_abj", "nvrs_amgx_fp64"))


def find_nvrs_bases(rundir):
    """Every NVRS run in the case, as [(label, base), ...].

    find_bases() deliberately picks ONE for the field comparison -- the 3-D
    panels difference two states and a third would have nothing to be
    differenced against.  The well curves are not like that: a rate is a
    scalar in time, three lines fit on one axes, and which backend a
    discrepancy belongs to is exactly what the reader wants to know.
    """
    out = []
    for sub, label in (("nvrs", "NVRS-CUDA"),
                       ("nvrs_amgx", "NVRS-CUDA (AmgX)"),
                       ("nvrs_hypre", "NVRS-CUDA (HYPRE)"),
                       ("nvrs_ginkgo", "NVRS-CUDA (Ginkgo FP16)"),
                       ("nvrs_ginkgo32", "NVRS-CUDA (Ginkgo FP32)"),
                       ("nvrs_ginkgo_abj",
                        "NVRS-CUDA (Ginkgo FP16 + adaptive bJ)"),
                       ("nvrs_amgx_fp64", "NVRS-CUDA (AmgX, FP64 operator)")):
        hits = sorted(glob.glob(os.path.join(rundir, sub, "**", "*.EGRID"),
                                recursive=True))
        if hits:
            out.append((label, os.path.splitext(hits[0])[0]))
    return out


def load_case(base):
    """Read an ECLIPSE case into {times, fields{name -> (nt,nz,ny,nx)}, dims}.

    Oil saturation is not stored by either simulator: ECLIPSE convention is to
    write SWAT and SGAS and leave So = 1 - Sw - Sg implied.  It is formed here
    identically for both sides.
    """
    c = ecl.EclipseCase(base)
    nx, ny, nz = c.dims
    t = c.restart_times()
    P = c.restart_field("PRESSURE")
    SW = c.restart_field("SWAT")
    SG = c.restart_field("SGAS")
    # A METRIC run writes pressures in bar.  Saturations are dimensionless and
    # are left alone.  Verified against the files: OPM's Norne holds
    # 260.3 .. 312.5 bar at t = 0 and NVRS 3774.3 .. 4543.9 psi, and
    # 260.3 x 14.5038 = 3776 -- the same initial state, reported twice in two
    # systems.  Uncorrected this showed up as a pressure RMS of 3728 psi at
    # t = 0, before a single timestep.
    if P is not None and (c.unit_system or "").upper() == "METRIC":
        P = np.asarray(P, dtype=float) * _BAR_TO_PSI
        log("  [units] %s fields were METRIC: PRESSURE bar -> psia"
            % os.path.basename(base))
    if P is None or SW is None:
        raise RuntimeError("no PRESSURE/SWAT in %s.UNRST" % base)
    if SG is None:
        SG = np.zeros_like(SW)
    SO = 1.0 - SW - SG
    n = min(len(t), P.shape[0], SW.shape[0], SG.shape[0])
    return dict(dims=(nx, ny, nz), times=np.asarray(t[:n], dtype=float),
                fields={"pressure": P[:n], "water": SW[:n],
                        "oil": SO[:n], "gas": SG[:n]},
                base=base, case=c)


def align_times(ta, tb, rtol=1e-3):
    """Index pairs (ia, ib) whose report times agree.

    Both decks carry the same TSTEP, so the times should be identical; matching
    on value rather than position means a run that chopped a step, stopped
    early or wrote an extra initial record still lines up instead of silently
    comparing step 4 against step 5.
    """
    pairs = []
    if ta.size == 0 or tb.size == 0:
        return pairs
    span = max(float(np.max(ta)), 1.0)
    tol = rtol * span

    # Never wider than a quarter of the closest two report times.
    #
    # A relative tolerance alone is 3.3 days on Norne, and Norne's schedule
    # has dates one day apart.  A date present in one run and absent from the
    # other would then pair with its NEIGHBOUR and be reported as a match --
    # the exact failure the value-matching was introduced to prevent, just at
    # a finer scale.  The two runs write different numbers of restarts (65
    # against 248, because RPTRST governs one and DTREPORT the other), so
    # missing dates are the normal case here, not the exceptional one.
    gaps = [np.diff(x) for x in (ta, tb) if x.size > 1]
    gaps = [float(g[g > 0].min()) for g in gaps if np.any(g > 0)]
    if gaps:
        tol = min(tol, 0.25 * min(gaps))

    worst = 0.0
    for ia, t in enumerate(ta):
        ib = int(np.argmin(np.abs(tb - t)))
        dt = abs(float(tb[ib]) - float(t))
        if dt <= tol:
            pairs.append((ia, ib, float(t)))
            worst = max(worst, dt)
    if pairs and worst > 0:
        log("  time match: worst offset %.4g day(s), tolerance %.4g" % (worst, tol))
    return pairs


# ===========================================================================
# Field panel: columns OPM / NVRS / difference, rows pressure and saturations
# ===========================================================================
# Corner-point geometry for the panels, loaded once per worker process.
#
# The single-run report already draws Norne as the dipping, faulted surface it
# is; this page was still drawing it as a cuboid of voxels, which shows the
# index topology and hides the geometry -- on a faulted field that is most of
# what there is to see.  Same source as plot_norne.py: cgrid.bin, six float32
# columns per ACTIVE cell (i, j, k 1-based, then the centre x, y, z).
_CGRID_CACHE = {}
_PVGRID_CACHE = {}


def _corner_point(cgrid_path, dims):
    """(X, Y, Z, ktop) on the index box, or None if there is no cgrid.bin."""
    key = (cgrid_path, dims)
    if key in _CGRID_CACHE:
        return _CGRID_CACHE[key]
    out = None
    try:
        # plot_norne already reads cgrid.bin onto the index box and finds the
        # shallowest layer -- the single-run report draws Norne with them.
        # Calling them beats keeping a second copy that can drift.
        nz, ny, nx = dims
        meta = {"nx": nx, "ny": ny, "nz": nz}
        X, Y, Z = plot_norne.coordinate_cubes(os.path.dirname(cgrid_path), meta)
        out = (X, Y, Z, plot_norne.top_layer(Z))
    except Exception as exc:
        log("  ! corner-point geometry unusable (%s); drawing the index box"
            % exc)
        out = None
    _CGRID_CACHE[key] = out
    return out


def _pv_block_geometry(egrid_path, dims, target=None):
    """An exact-corner PyVista grid and optional fine-to-block reduction map.

    Native cells are retained by default up to 1.5 million cells.  The old
    10k target averaged SPE10 into 5x5x5 display blocks and visibly erased its
    lower-channel pressure structure.  Very large models can still request a
    bounded display grid with VIZ_TARGET_CELLS.
    """
    if pv is None or not egrid_path or not os.path.exists(egrid_path):
        return None
    if target is None:
        target = int(os.environ.get("VIZ_TARGET_CELLS", "1500000"))
    target = max(1, int(target))
    zex = float(os.environ.get("ZEXAG", "1"))
    key = (egrid_path, tuple(dims), target, zex)
    if key in _PVGRID_CACHE:
        return _PVGRID_CACHE[key]

    nz, ny, nx = dims
    rec = ecl.read_unformatted(egrid_path)
    first = lambda kw: next((v for k, _t, v in rec if k == kw), None)
    coord = first("COORD")
    zcorn = first("ZCORN")
    actnum = first("ACTNUM")
    if coord is None or zcorn is None:
        return None
    coord = np.asarray(coord, dtype=float).reshape((ny + 1, nx + 1, 6))
    zcorn = np.asarray(zcorn, dtype=float).reshape((2 * nz, 2 * ny, 2 * nx))
    active = (np.ones((nz, ny, nx), dtype=bool) if actnum is None else
              np.asarray(actnum).reshape((nz, ny, nx)) > 0)

    stride = max(1, int(np.ceil((nx * ny * nz / float(target)) ** (1.0 / 3.0))))
    nbi = (nx + stride - 1) // stride
    nbj = (ny + stride - 1) // stride
    nbk = (nz + stride - 1) // stride
    nfull = nbi * nbj * nbk
    fine_id = np.broadcast_to(
        (np.arange(nx)[None, None, :] // stride)
        + nbi * ((np.arange(ny)[None, :, None] // stride)
                 + nbj * (np.arange(nz)[:, None, None] // stride)),
        (nz, ny, nx)).ravel()
    occupancy = np.bincount(fine_id, weights=active.ravel().astype(float),
                            minlength=nfull)
    live_ids = np.flatnonzero(occupancy > 0)
    bi = live_ids % nbi
    bj = (live_ids // nbi) % nbj
    bk = live_ids // (nbi * nbj)
    i0, j0, k0 = bi * stride, bj * stride, bk * stride
    i1 = np.minimum(i0 + stride - 1, nx - 1)
    j1 = np.minimum(j0 + stride - 1, ny - 1)
    k1 = np.minimum(k0 + stride - 1, nz - 1)

    corners = np.empty((live_ids.size, 8, 3), dtype=float)
    for c in range(8):
        a, b, cc = c & 1, (c >> 1) & 1, (c >> 2) & 1
        ii = i1 if a else i0
        jj = j1 if b else j0
        kk = k1 if cc else k0
        z = zcorn[2 * kk + cc, 2 * jj + b, 2 * ii + a]
        pillar = coord[jj + b, ii + a]
        xt, yt, zt = pillar[:, 0], pillar[:, 1], pillar[:, 2]
        xb, yb, zb = pillar[:, 3], pillar[:, 4], pillar[:, 5]
        dz = zb - zt
        frac = np.where(np.abs(dz) > 1e-12, (z - zt) / dz, 0.0)
        corners[:, c, 0] = xt + frac * (xb - xt)
        corners[:, c, 1] = yt + frac * (yb - yt)
        corners[:, c, 2] = z

    # Large projected map coordinates (Norne Y is about 7.3e6 m) make VTK's
    # off-screen camera/depth calculations lose a model only hundreds of
    # metres thick. Recenter for rendering; all distances and cell geometry
    # remain exact, and well coordinates receive the same offset below.
    xmid = 0.5 * (float(np.nanmin(corners[:, :, 0]))
                  + float(np.nanmax(corners[:, :, 0])))
    ymid = 0.5 * (float(np.nanmin(corners[:, :, 1]))
                  + float(np.nanmax(corners[:, :, 1])))
    corners[:, :, 0] -= xmid
    corners[:, :, 1] -= ymid
    zmid = 0.5 * (float(np.nanmin(corners[:, :, 2]))
                  + float(np.nanmax(corners[:, :, 2])))
    corners[:, :, 2] = -(corners[:, :, 2] - zmid) * zex
    vtk_order = np.array([0, 1, 3, 2, 4, 5, 7, 6])
    points = corners[:, vtk_order].reshape((-1, 3))
    conn = np.arange(points.shape[0], dtype=np.int64).reshape((-1, 8))
    cells = np.column_stack((np.full(live_ids.size, 8, dtype=np.int64),
                             conn)).ravel()
    ctypes = np.full(live_ids.size, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, points)
    out = dict(grid=grid, fine_id=fine_id, live_ids=live_ids,
               nfull=nfull, active=active.ravel(), xy_origin=(xmid, ymid))
    _PVGRID_CACHE[key] = out
    return out


def _pv_block_values(field, geom):
    v = np.asarray(field, dtype=float).ravel()
    mask = np.isfinite(v) & geom["active"]
    sums = np.bincount(geom["fine_id"][mask], weights=v[mask],
                       minlength=geom["nfull"])
    count = np.bincount(geom["fine_id"][mask], minlength=geom["nfull"])
    avg = np.full(geom["nfull"], np.nan)
    np.divide(sums, count, out=avg, where=count > 0)
    return avg[geom["live_ids"]]


def _pv_full_surface(mesh):
    """Native-cell full-field envelope reduced to its external faces.

    The unstructured grid deliberately keeps eight corner points per cell so
    faults and pinches cannot accidentally weld across a discontinuity.  For
    SPE10 that is almost nine million vertices *per actor*.  Sending nine such
    actors to one VTK window can silently drop the first opaque actor (OPM
    pressure), and the stand-alone rock windows can lose their only actor.

    Exact point welding joins only coincident corners (not separated fault
    faces), after which extracting the boundary can actually remove buried
    faces.  The displayed colours remain native cell values -- no
    interpolation, averaging, scatter, or resampling -- while the rasterizer
    sees a small, robust Petrel-style filled full-field envelope instead of
    millions of invisible interior faces.
    """
    # Every hexahedron was constructed with private corner points so faults
    # cannot be welded accidentally.  Without this exact-coordinate merge,
    # VTK regards all six faces of every cell as external and extraction still
    # returns millions of quads for SPE10.
    mesh.cell_data["source_cell"] = np.arange(mesh.n_cells, dtype=np.int64)
    welded = mesh.clean(tolerance=0.0, produce_merge_map=False)
    return welded.extract_surface(pass_cellid=False, pass_pointid=False,
                                  algorithm="dataset_surface")


def _pv_surface_with_values(field, block):
    """A cheap scalar-specific view of the cached full-field envelope."""
    if "surface" not in block:
        surface = _pv_full_surface(block["grid"].copy(deep=False))
        # Cleaning retains the explicit source cell number and the
        # surface filter carries it to every visible face. Cache that mapping
        # with the topology: only the scalar array changes between columns,
        # rows, and report dates.
        source = np.asarray(surface.cell_data["source_cell"],
                            dtype=np.int64).copy()
        block["surface"] = surface
        block["surface_source"] = source
    mesh = block["surface"].copy(deep=False)
    values = _pv_block_values(field, block)
    mesh.cell_data["value"] = values[block["surface_source"]]
    return mesh


def _pv_changed_surface(field, baseline, block, threshold):
    """Boundary of native cells whose value changed by at least threshold."""
    values = _pv_block_values(field, block)
    initial = _pv_block_values(baseline, block)
    selected = np.flatnonzero(
        np.isfinite(values) & np.isfinite(initial)
        & (np.abs(values - initial) >= float(threshold)))
    if selected.size == 0:
        return None
    mesh = block["grid"].extract_cells(selected)
    mesh.cell_data["value"] = values[selected]
    # Selected neighbouring cells must share exact coincident points before
    # extracting their envelope, otherwise every buried face is rendered.
    mesh = mesh.clean(tolerance=0.0, produce_merge_map=False)
    return mesh.extract_surface(pass_cellid=False, pass_pointid=False,
                                algorithm="dataset_surface")


def _pv_well_tubes(well_xyz, bounds):
    """Batch dense well sets into one VTK actor per well type."""
    dz = max(bounds[5] - bounds[4], 1.0)
    radius = 0.0025 * max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    grouped = {}
    for w, xw, yw, _zw in well_xyz:
        colour = {"prod": "#159447", "winj": "#1464d2",
                  "ginj": "#d62728", "inj": "#1464d2"}.get(
                      str(w.get("type", "prod")), "#20252b")
        line = pv.Line((xw, yw, bounds[4]),
                       (xw, yw, bounds[5] + 0.14 * dz))
        grouped.setdefault(colour, []).append(line.tube(radius=radius))
    return [(pv.merge(parts, merge_points=False), colour)
            for colour, parts in grouped.items()]


def _pv_merge_dense_scene(mesh, well_tubes, opacity, context=None):
    """One actor containing transparent context, result, and dense wells."""
    pieces = []
    if context is not None:
        context = context.copy(deep=False)
        context.cell_data["opacity"] = np.full(context.n_cells, 0.10)
        pieces.append(context)
    mesh = mesh.copy(deep=False)
    mesh.cell_data["opacity"] = np.full(mesh.n_cells, float(opacity))
    pieces.append(mesh)
    for tubes, _colour in well_tubes:
        tubes = tubes.copy(deep=False)
        tubes.cell_data["value"] = np.full(tubes.n_cells, np.nan)
        tubes.cell_data["opacity"] = np.ones(tubes.n_cells)
        pieces.append(tubes)
    return pv.merge(pieces, merge_points=False)


def _pv_xy_scale(X, Y, bounds):
    """Scale cgrid coordinates onto EGRID units (notably Norne ft -> m)."""
    ratios = []
    for values, span in ((X, bounds[1] - bounds[0]),
                         (Y, bounds[3] - bounds[2])):
        raw = float(np.nanmax(values) - np.nanmin(values))
        if raw > 0.0 and span > 0.0:
            ratios.append(span / raw)
    scale = float(np.median(ratios)) if ratios else 1.0
    # Active-cell-centre extents are slightly smaller than the full EGRID
    # corner extent, so their raw ratio is only a unit-system detector. Once
    # it identifies feet versus metres, use the exact conversion; applying
    # the approximate extent ratio to a 24-million-foot northing introduces a
    # false offset of hundreds of kilometres.
    if 0.25 < scale < 0.40:
        return 0.3048
    return 1.0 if 0.80 < scale < 1.25 else scale


def _pv_reservoir_camera(plotter, zoom=1.08):
    """Reference-style reservoir view: azimuth -105°, elevation 40°."""
    azim = math.radians(float(os.environ.get("VIZ_AZIM", "-105")))
    elev = math.radians(float(os.environ.get("VIZ_ELEV", "40")))
    bounds = plotter.bounds
    centre = np.array([(bounds[0] + bounds[1]) * 0.5,
                       (bounds[2] + bounds[3]) * 0.5,
                       (bounds[4] + bounds[5]) * 0.5])
    radius = 2.5 * max(bounds[1] - bounds[0], bounds[3] - bounds[2],
                       bounds[5] - bounds[4], 1.0)
    direction = np.array([math.cos(elev) * math.cos(azim),
                          math.cos(elev) * math.sin(azim),
                          math.sin(elev)])
    plotter.camera_position = [centre + radius * direction, centre,
                               (0.0, 0.0, 1.0)]
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(zoom)


def _draw_top_surface(ax, geom, field, vlo, vhi, cmap, wells, elev, azim,
                    label_sep=0.055, label_wells=True, label_fontsize=None):
    """The reservoir's top surface, coloured by `field` (nz, ny, nx).

    The matplotlib renderer the reports used until 15 September, restored as
    the fallback for machines without PyVista: the sampled cell cloud that
    replaced it drew a few thousand translucent squares, which on Norne read
    as loose sticks rather than a field.  NVRS_FALLBACK_ZEXAG sets its
    vertical exaggeration (default 12, as before).

    `label_sep` is the smallest gap between two well labels, as a fraction of
    the field's extent.  Names closer than that are dropped rather than drawn
    on top of each other.
    """
    placed = []
    X, Y, Z, ktop = geom
    zex = float(os.environ.get("NVRS_FALLBACK_ZEXAG", "12"))
    Zt = Z[:, :, ktop]
    zmid = np.nanmean(Zt)
    Zs = zmid + (Zt - zmid) * zex
    # field is (nz, ny, nx); the geometry is (nx, ny, nz).
    Ct = np.asarray(field)[ktop].T
    span = (vhi - vlo) or 1.0
    fl = (Ct - vlo) / span
    cols = plt.get_cmap(cmap)(np.ma.masked_invalid(np.clip(fl, 0.0, 1.0)))
    cols[~np.isfinite(Ct), :3] = 1.0
    ax.plot_surface(X[:, :, ktop], Y[:, :, ktop], Zs, facecolors=cols,
                    edgecolor="none", shade=True, rstride=1, cstride=1,
                    antialiased=False)
    ax.set_facecolor("white")
    for lim, arr in ((ax.set_xlim, X), (ax.set_ylim, Y)):
        lim(np.nanmin(arr), np.nanmax(arr))
    zlo, zhi = float(np.nanmin(Zs)), float(np.nanmax(Zs))
    if zhi > zlo:
        ax.set_zlim(zlo, zhi)
    ax.view_init(elev, azim)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    nx, ny = X.shape[0], X.shape[1]
    # Low enough that the tallest label still clears the axes title.
    # The sticks used to reach a quarter of the field's relief above it and
    # the names then sat on top of the heading.
    ztop = zhi + 0.10 * (zhi - zlo + 1.0)
    xspan = float(np.nanmax(X) - np.nanmin(X)) or 1.0
    yspan = float(np.nanmax(Y) - np.nanmin(Y)) or 1.0
    # Sticks in 3-D now; names later, in 2-D.
    #
    # Names are laid out by place_well_labels(), called just before the figure
    # is saved: colour bars and tight_layout move and shrink the axes after
    # this function returns, and names placed against the old axes box ended
    # up overlapping.  Here only the stick tops are recorded.
    tops = []
    for w in (wells or []):
        col = {"prod": "green", "winj": "blue", "ginj": "red",
               "inj": "blue"}.get(str(w.get("type", "prod")), "black")
        i, j = int(w.get("i", 0)) - 1, int(w.get("j", 0)) - 1
        if not (0 <= i < nx and 0 <= j < ny):
            continue
        xw, yw, zw = X[i, j, ktop], Y[i, j, ktop], Zs[i, j]
        if not (np.isfinite(xw) and np.isfinite(yw) and np.isfinite(zw)):
            continue
        ax.plot([xw, xw], [yw, yw], [zw, ztop], col, linewidth=1.2)
        tops.append((float(xw), float(yw), float(ztop),
                     str(w.get("name", "")), col))
    if label_wells:
        ax._nvrs_well_tops = tops
        ax._nvrs_label_fs = label_fontsize


def place_well_labels(fig):
    """Lay out every recorded well name on `fig` without overlap.

    Call immediately before savefig, after colour bars and tight_layout.
    Each stick top is projected to the screen and its name goes in the
    nearest free slot: straight above in tiers, then sideways, then below,
    never into the top 6% of the axes where the title sits and never outside
    them.  A thin leader line joins a moved name to its stick.  Names are
    bold (NVRS_WELL_LABEL_FONTSIZE, default 6.5 pt; smaller in dense grids).
    """
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import proj3d
    for ax in fig.axes:
        tops = getattr(ax, "_nvrs_well_tops", None)
        if not tops:
            continue
        fs = ax._nvrs_label_fs or float(
            os.environ.get("NVRS_WELL_LABEL_FONTSIZE", "6.5"))
        proj = ax.get_proj()
        ax_w = max(ax.bbox.width, 1.0)
        ax_h = max(ax.bbox.height, 1.0)
        char_w = fs * 0.80 / 72.0 * fig.dpi / ax_w
        line_h = fs * 1.6 / 72.0 * fig.dpi / ax_h
        pts = []
        for xw, yw, zt, name, col in tops:
            xp, yp, _ = proj3d.proj_transform(xw, yw, zt, proj)
            ux, uy = ax.transAxes.inverted().transform(
                ax.transData.transform((xp, yp)))
            pts.append((ux, uy, name, col))
        boxes = []

        def free(b):
            pad = 0.010
            return all(b[2] + pad < o[0] or b[0] - pad > o[2] or
                       b[3] + pad < o[1] or b[1] - pad > o[3] for o in boxes)

        cands_rel = []
        for tier in range(24):
            for shift in (0.0, -0.6, 0.6, -1.2, 1.2, -1.8, 1.8, -2.4, 2.4):
                for sign in (1, -1):
                    cands_rel.append((tier + abs(shift) * 0.9 +
                                      (0.0 if sign > 0 else 3.0),
                                      tier, shift, sign))
        cands_rel.sort(key=lambda c: c[0])
        for ux, uy, name, col in sorted(pts, key=lambda t: (t[1], t[0])):
            if not name:
                continue
            wbox = len(name) * char_w + 0.014
            spot = None
            for _c, tier, shift, sign in cands_rel:
                ly = (uy + 0.015 + tier * line_h * 1.15 if sign > 0 else
                      uy - 0.015 - line_h - tier * line_h * 1.15)
                lx = ux + shift * wbox
                b = (lx - wbox / 2, ly, lx + wbox / 2, ly + line_h)
                if b[3] > 0.94 or b[1] < 0.02 or b[0] < 0.0 or b[2] > 1.0:
                    continue
                if free(b):
                    spot = (lx, ly, b)
                    break
            if spot is None:
                continue          # no free slot left: the stick stays unnamed
            lx, ly, b = spot
            boxes.append(b)
            if abs(lx - ux) > 1e-6 or ly - uy > 0.02 or ly < uy:
                fig.add_artist(Line2D([ux, lx], [uy, ly + (0 if ly > uy else line_h)],
                                      transform=ax.transAxes, color=col,
                                      linewidth=0.5, alpha=0.8, zorder=19))
            ax.text2D(lx, ly, name, transform=ax.transAxes, color=col,
                      weight="bold", fontsize=fs, ha="center", va="bottom",
                      zorder=20, clip_on=False,
                      bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                ec=col, lw=0.4, alpha=0.9))
        ax._nvrs_well_tops = None

def _draw_structure(ax, geom, field, vlo, vhi, cmap, wells, elev, azim,
                    label_sep=0.055, label_wells=True, label_fontsize=None):
    """A sampled physical-coordinate cell cloud coloured by `field`."""
    X, Y, Z, ktop = geom
    # Physical coordinates are the default.  Vertical exaggeration remains an
    # explicit opt-in (for example ZEXAG=12) rather than silently changing the
    # model's aspect ratio in an accuracy comparison.
    zex = float(os.environ.get("ZEXAG", "1"))
    zraw_lo, zraw_hi = float(np.nanmin(Z)), float(np.nanmax(Z))
    zmid = 0.5 * (zraw_lo + zraw_hi)
    Zs = zmid + (Z - zmid) * zex
    Zt = Zs[:, :, ktop]
    # field is (nz, ny, nx); the geometry is (nx, ny, nz).
    span = (vhi - vlo) or 1.0
    colour = plt.get_cmap(cmap)
    # A thin field (Norne: 22 layers) reads best as its top surface, drawn the
    # way these reports drew it before the PyVista renderer arrived.  A thick
    # model (SPE10: 85 layers) keeps the sampled cloud below, whose point is
    # the interior.  NVRS_FALLBACK_3D=surface|cloud forces either.
    style = os.environ.get("NVRS_FALLBACK_3D", "auto").lower()
    if style == "surface" or (style == "auto" and X.shape[2] <= 40):
        _draw_top_surface(ax, geom, field, vlo, vhi, cmap, wells, elev, azim,
                          label_sep=label_sep, label_wells=label_wells,
                          label_fontsize=label_fontsize)
        return
    # Plot cells throughout the volume, not a skin.  Roughly 20k points keeps
    # 41-frame reports responsive while sampling every logical direction and
    # retaining the exact corner-point centroids.  This is especially useful
    # for SPE10: its flat top contains no information about its 85 K layers.
    target = 20000.0
    stride = max(1, int(np.ceil((X.size / target) ** (1.0 / 3.0))))
    sl = (slice(None, None, stride),) * 3
    values = np.asarray(field).transpose(2, 1, 0)[sl]
    xs, ys, zs = X[sl], Y[sl], Zs[sl]
    live = (np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
            & np.isfinite(values))
    rgba = colour(np.clip((values[live] - vlo) / span, 0.0, 1.0))
    rgba[:, 3] = 0.34
    ax.scatter(xs[live], ys[live], zs[live], c=rgba, s=3.0,
               marker="s", linewidths=0, depthshade=False,
               rasterized=True)
    ax.set_facecolor("white")
    for lim, arr in ((ax.set_xlim, X), (ax.set_ylim, Y)):
        lim(np.nanmin(arr), np.nanmax(arr))
    zlo, zhi = float(np.nanmin(Zs)), float(np.nanmax(Zs))
    if zhi > zlo:
        ax.set_zlim(zlo, zhi)
    # mplot3d defaults to a nearly cubic display box even when X spans 3,000
    # ft and Y spans 11,000 ft. Preserve physical X:Y:Z (or the explicitly
    # requested ZEXAG above) instead of drawing index geometry.
    xspan = float(np.nanmax(X) - np.nanmin(X)) or 1.0
    yspan = float(np.nanmax(Y) - np.nanmin(Y)) or 1.0
    zspan = zhi - zlo if zhi > zlo else min(xspan, yspan) * 0.02
    ax.set_box_aspect((xspan, yspan, zspan))
    ax.view_init(elev, azim)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    nx, ny = X.shape[0], X.shape[1]
    # Low enough that the tallest label still clears the axes title.
    # The sticks used to reach a quarter of the field's relief above it and
    # the names then sat on top of the heading.
    ztop = zhi + 0.10 * (zhi - zlo + 1.0)
    for w in (wells or []):
        col = {"prod": "green", "winj": "blue", "ginj": "red",
               "inj": "blue"}.get(str(w.get("type", "prod")), "black")
        i, j = int(w.get("i", 0)) - 1, int(w.get("j", 0)) - 1
        if not (0 <= i < nx and 0 <= j < ny):
            continue
        xw, yw, zw = X[i, j, ktop], Y[i, j, ktop], Zt[i, j]
        if not (np.isfinite(xw) and np.isfinite(yw) and np.isfinite(zw)):
            continue
        ax.plot([xw, xw], [yw, yw], [zw, ztop], col, linewidth=1.4)


def _panel_worker_pyvista(job, block):
    """Petrel-style filled, translucent hexahedral block-model panel."""
    (path, title, rows, wells, _elev, _azim, cgrid_path, _dims,
     _egrid_path) = job
    nrow = len(rows)
    # Keep each of the three side-by-side reservoir views large enough to
    # inspect channels and fronts in the embedded report. The old 1472 px
    # canvas left fewer than 500 pixels per model after scalar bars/margins.
    plotter = pv.Plotter(off_screen=True, shape=(nrow, 3),
                         window_size=(1800, max(1120, 400 * nrow)),
                         border=False)
    plotter.set_background("#f4f6f8")
    # VTK 9.7's off-screen/Xvfb depth-peeling path silently drops the first
    # mesh actor in a render window.  That made OPM pressure blank while the
    # later NVRS actor survived. These are boundary-only full-field surfaces, so
    # ordinary alpha blending is both sufficient and reliable.
    opacity = float(os.environ.get("BLOCK_OPACITY", "0.62"))
    zex = float(os.environ.get("ZEXAG", "1"))
    geometry_note = ("physical X:Y:Z" if abs(zex - 1.0) <= 1e-12
                     else "physical X:Y; Z x%g" % zex)
    columns = ("OPM Flow", "NVRS-CUDA", "OPM - NVRS")

    # Well trajectories use the same physical centroids as the simulator.
    well_xyz = []
    cg = _corner_point(cgrid_path, _dims) if cgrid_path else None
    if cg is not None:
        X, Y, Z, ktop = cg
        xorigin, yorigin = block.get("xy_origin", (0.0, 0.0))
        xy_scale = _pv_xy_scale(X, Y, block["grid"].bounds)
        zmid = 0.5 * (float(np.nanmin(Z)) + float(np.nanmax(Z)))
        ztop = zmid + (Z[:, :, ktop] - zmid) * zex
        for w in wells or []:
            i, j = int(w.get("i", 0)) - 1, int(w.get("j", 0)) - 1
            if 0 <= i < X.shape[0] and 0 <= j < X.shape[1]:
                if np.isfinite(X[i, j, ktop]) and np.isfinite(ztop[i, j]):
                    well_xyz.append((w, float(X[i, j, ktop] * xy_scale
                                              - xorigin),
                                     float(Y[i, j, ktop] * xy_scale
                                           - yorigin),
                                     float(ztop[i, j])))

    bounds = block["grid"].bounds
    dz = max(bounds[5] - bounds[4], 1.0)
    well_tubes = _pv_well_tubes(well_xyz, bounds)
    dense_wells = len(well_xyz) > 12
    for r, row in enumerate(rows):
        label, cmap, a, b, d = row[:5]
        is_pressure = label.lower().startswith("pressure")
        evolution_threshold = (float(os.environ.get(
            "PRESSURE_EVOLUTION_THRESHOLD", "1000")) if is_pressure else
            float(os.environ.get("SATURATION_EVOLUTION_THRESHOLD", "0.05")))
        lo = (float(row[5]) if len(row) >= 8 else
              float(min(np.nanmin(a), np.nanmin(b))))
        hi = (float(row[6]) if len(row) >= 8 else
              float(max(np.nanmax(a), np.nanmax(b))))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-30:
            hi = lo + 1e-30
        dm = (float(row[7]) if len(row) >= 8 else
              float(np.nanmax(np.abs(d))))
        if not np.isfinite(dm) or dm < 1e-30:
            dm = 1e-30
        for c, (field, cm_, vlo, vhi) in enumerate(
                ((a, cmap, lo, hi), (b, cmap, lo, hi),
                 (d, DIFF_CMAP, -dm, dm))):
            plotter.subplot(r, c)
            # Show the complete reservoir envelope. The cached boundary is a
            # rendering optimisation only; scalars remain cell-centred and at
            # native resolution. Applying the same geometry to every row also
            # prevents pressure and saturation panels from implying different
            # spatial support.
            mesh = _pv_surface_with_values(field, block)
            # OPM and NVRS use identical Turbo maps and identical limits in
            # columns 1 and 2.  Column 3 gets an explicitly named delta legend;
            # previously it repeated `label`, making blue-white-red appear to
            # be the pressure/saturation scale.
            bar_title = (label + "\nTurbo — shared OPM/NVRS scale"
                         if c < 2 else
                         "Δ %s\nOPM - NVRS" % label)
            sb = dict(title=bar_title, title_font_size=9, label_font_size=7,
                      n_labels=3, fmt="%.3g", vertical=True,
                      position_x=0.86, position_y=0.12,
                      width=0.09, height=0.68)
            changed = None
            if c < 2 and len(row) >= 10:
                changed = _pv_changed_surface(
                    field, row[8 + c], block, evolution_threshold)
                # A quiet full-field shell preserves reservoir context while
                # the opaque changed-cell envelope exposes the moving front.
                # Dense Norne well sets use a single composite actor below:
                # VTK 9.7 off-screen drops this geometry when a renderer has
                # multiple actors. Per-cell opacity keeps the complete field
                # visible without creating another actor.
                if not dense_wells:
                    plotter.add_mesh(
                        mesh, color="#c7cdd3", style="surface", opacity=0.10,
                        show_edges=False, lighting=True, ambient=0.45,
                        diffuse=0.55, show_scalar_bar=False, reset_camera=True)
            display_mesh = changed if changed is not None else mesh
            if dense_wells:
                display_mesh = _pv_merge_dense_scene(
                    display_mesh, well_tubes,
                    0.92 if changed is not None else
                    (1.0 if is_pressure else opacity),
                    context=mesh if changed is not None else None)
            plotter.add_mesh(
                display_mesh, scalars="value", cmap=cm_, clim=(vlo, vhi),
                style="surface",
                opacity=("opacity" if dense_wells else
                         (0.92 if changed is not None else
                          (1.0 if is_pressure else opacity))),
                show_edges=False,
                lighting=True, ambient=0.28, diffuse=0.72, specular=0.18,
                specular_power=18.0,
                nan_color="#263238", nan_opacity=(1.0 if dense_wells else 0.0),
                interpolate_before_map=False,
                show_scalar_bar=True, scalar_bar_args=sb, reset_camera=True)
            if not dense_wells:
                for tubes, colour in well_tubes:
                    plotter.add_mesh(tubes, color=colour, lighting=True,
                                     reset_camera=False)
            heading = "%s — %s" % (columns[c], label)
            if c < 2:
                unit = " psi" if is_pressure else ""
                heading += " — |change from t0| ≥ %g%s" % (
                    evolution_threshold, unit)
            else:
                heading += (" — opaque full field" if is_pressure
                            else " — translucent full field")
            if r == 0 and c == 0:
                heading += "\n%s | %s" % (title, geometry_note)
            plotter.add_text(heading, position="upper_left", font_size=8,
                             color="#20252b")
            _pv_reservoir_camera(plotter, zoom=1.18)

    plotter.screenshot(path, transparent_background=False)
    plotter.close()
    return path


def _rock_figure_pyvista(path, title, field, cmap, block, vlo, vhi,
                         full_lo, full_hi, wells=None, cgrid_path=None,
                         dims=None):
    """Render one static property as the same filled block model as panels."""
    # A deliberately wide viewport reserves the right margin for the scalar
    # bar.  With the former 1050 px canvas and 1.15 camera zoom, SPE10's long
    # model extended underneath the legend.
    plotter = pv.Plotter(off_screen=True, window_size=(1250, 760))
    plotter.set_background("#f4f6f8")
    # Keep this consistent with the report-step renderer.  Enabling depth
    # peeling here drops the sole rock actor on VTK 9.7 off-screen renders,
    # leaving only wells and the colourbar in the saved PNG.
    mesh = _pv_surface_with_values(field, block)
    plotter.add_mesh(
        mesh, scalars="value", cmap=cmap, clim=(vlo, vhi),
        style="surface",
        opacity=float(os.environ.get("BLOCK_OPACITY", "0.62")),
        show_edges=False, lighting=True, ambient=0.28, diffuse=0.72,
        specular=0.18, specular_power=18.0, nan_opacity=0.0,
        scalar_bar_args=dict(
            title=("%s\n1-99%% colour range\nfull %.4g-%.4g"
                   % (title, full_lo, full_hi)),
            title_font_size=11, label_font_size=9, n_labels=5, fmt="%.4g",
            vertical=True, position_x=0.895, position_y=0.18,
            width=0.075, height=0.62),
        reset_camera=True)
    zex = float(os.environ.get("ZEXAG", "1"))
    note = ("physical X:Y:Z" if abs(zex - 1.0) <= 1e-12
            else "physical X:Y; Z x%g" % zex)

    # Draw the schedule wells in the same physical coordinate system as the
    # block model. Tubes remain visible through the translucent rock. Names
    # are intentionally omitted: Norne's dense well pattern becomes unreadable
    # when dozens of labels are projected into a static 3-D viewport.
    cg = (_corner_point(cgrid_path, dims)
          if cgrid_path and dims is not None else None)
    if cg is not None:
        X, Y, Z, ktop = cg
        xorigin, yorigin = block.get("xy_origin", (0.0, 0.0))
        xy_scale = _pv_xy_scale(X, Y, block["grid"].bounds)
        zmid = 0.5 * (float(np.nanmin(Z)) + float(np.nanmax(Z)))
        ztop = zmid + (Z[:, :, ktop] - zmid) * zex
        bounds = block["grid"].bounds
        dz = max(bounds[5] - bounds[4], 1.0)
        radius = 0.0025 * max(bounds[1] - bounds[0],
                              bounds[3] - bounds[2])
        well_xyz = []
        for w in wells or []:
            i, j = int(w.get("i", 0)) - 1, int(w.get("j", 0)) - 1
            if not (0 <= i < X.shape[0] and 0 <= j < X.shape[1]):
                continue
            xw, yw, zw = (X[i, j, ktop] * xy_scale - xorigin,
                          Y[i, j, ktop] * xy_scale - yorigin, ztop[i, j])
            if not (np.isfinite(xw) and np.isfinite(yw)
                    and np.isfinite(zw)):
                continue
            well_xyz.append((w, float(xw), float(yw), float(zw)))
        well_tubes = _pv_well_tubes(well_xyz, bounds)
        if len(well_xyz) > 12:
            # Rebuild the one existing actor rather than adding dozens of
            # actors; see the dense-well path in the field panel renderer.
            plotter.remove_scalar_bar()
            plotter.clear_actors()
            mesh = _pv_merge_dense_scene(
                mesh, well_tubes,
                float(os.environ.get("BLOCK_OPACITY", "0.62")))
            plotter.add_mesh(
                mesh, scalars="value", cmap=cmap, clim=(vlo, vhi),
                style="surface", opacity="opacity", show_edges=False,
                lighting=True, ambient=0.30, diffuse=0.70, specular=0.20,
                specular_power=20.0, nan_color="#263238", nan_opacity=1.0,
                interpolate_before_map=False, show_scalar_bar=True,
                scalar_bar_args=dict(
                    title=("%s\n1-99%% colour range\nfull %.4g-%.4g"
                           % (title, full_lo, full_hi)),
                    title_font_size=11, label_font_size=9, n_labels=5,
                    fmt="%.4g", vertical=True, position_x=0.895,
                    position_y=0.18, width=0.075, height=0.62),
                reset_camera=True)
        else:
            for tubes, colour in well_tubes:
                plotter.add_mesh(tubes, color=colour, lighting=True,
                                 reset_camera=False)
    plotter.add_text("%s full field | %s" % (title, note), position="upper_left",
                     font_size=12, color="#20252b")
    _pv_reservoir_camera(plotter, zoom=0.94)
    plotter.screenshot(path, transparent_background=False)
    plotter.close()
    return path


def _injector_cell(wells):
    """0-based (i, j) of the first injector, where a movie's cut-away goes."""
    for w in wells or []:
        if "inj" in str(w.get("type", "")).lower():
            try:
                return int(w["i"]) - 1, int(w["j"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
    return None


def _panel_worker(job):
    """Render one report step's panel.  Runs in a worker process."""
    (path, title, rows, wells, elev, azim, cgrid_path, dims, egrid_path) = job
    if pv is not None:
        try:
            block = _pv_block_geometry(egrid_path, dims)
            if block is not None:
                return _panel_worker_pyvista(job, block)
        except Exception as exc:
            log("  ! PyVista block renderer failed (%s); using fallback" % exc)
    geom = _corner_point(cgrid_path, dims) if cgrid_path else None
    # A thick grid is drawn as voxel blocks even when it wrote corner-point
    # geometry: SPE10 comes in through a field deck (85 layers), and on the
    # corner-point path its interior became a sparse cloud of sticks.  Thin
    # corner-point fields (Norne, 22 layers) keep the surface renderer.
    if (geom is not None and geom[0].shape[2] > 40
            and os.environ.get("NVRS_FALLBACK_3D", "auto").lower() == "auto"):
        geom = None
    nrow = len(rows)
    fig = plt.figure(figsize=(15.5, 3.5 * nrow), dpi=95)
    # right < 1 leaves room for the difference colourbar's tick labels, which
    # sit outside the bar itself and are otherwise clipped by the figure edge.
    fig.subplots_adjust(left=0.035, right=0.945,
                        top=0.94 if geom is not None else 0.87,
                        bottom=0.01, wspace=0.02, hspace=0.02)
    for r, row in enumerate(rows):
        label, cmap, a, b, d = row[:5]
        # Shared scale across the two simulators so the eye compares colour to
        # colour.  When render_panels supplies global limits, every time step
        # also maps a value to that same colour.
        lo = (float(row[5]) if len(row) >= 8 else
              float(np.nanmin(b)) if a is None else
              float(min(np.nanmin(a), np.nanmin(b))))
        hi = (float(row[6]) if len(row) >= 8 else
              float(np.nanmax(b)) if a is None else
              float(max(np.nanmax(a), np.nanmax(b))))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-30:
            hi = lo + 1e-30
        m = (float(row[7]) if len(row) >= 8 else
             1e-30 if d is None else float(np.nanmax(np.abs(d))))
        if not np.isfinite(m) or m < 1e-30:
            m = 1e-30

        axes = []
        for c, (field, cm_, vlo, vhi) in enumerate(
                ((a, cmap, lo, hi), (b, cmap, lo, hi),
                 (d, DIFF_CMAP, -m, m))):
            ax = fig.add_subplot(nrow, 3, r * 3 + c + 1, projection="3d")
            if field is None:
                # NVRS ran past the end of Flow's output: no Flow field and
                # no difference at this time, said in place of the block.
                ax.set_axis_off()
                ax.text2D(0.5, 0.5, "no OPM Flow output\nat this time",
                          transform=ax.transAxes, ha="center", va="center",
                          fontsize=12, color="#777777")
            elif geom is not None:
                _draw_structure(ax, geom, field, vlo, vhi, cm_, wells,
                                elev, azim, label_wells=(r == 0 and c == 0),
                                label_fontsize=5.5)
            else:
                # Cartesian grid (BENCH, SPE10): the block drawn face by
                # face, cut away to the injector's column so the moving
                # front shows (the closed block's faces barely change).
                # Well sticks on every panel, names only in row 1, column 1.
                viz3d.draw_voxels(ax, np.nan_to_num(field, nan=vlo),
                                  wells=wells, vmin=vlo, vmax=vhi, cmap=cm_,
                                  elev=elev, azim=azim, well_fontsize=8,
                                  axis_labels=False,
                                  label_wells=(r == 0 and c == 0),
                                  cut=_injector_cell(wells))
            if r == 0:
                # The voxel block's well names stand above its top face, so
                # its titles sit higher to clear them.
                ax.set_title(("OPM Flow", "NVRS-CUDA",
                              "OPM − NVRS")[c], fontsize=13,
                             pad=14 if geom is not None else 30)
            axes.append(ax)

        # Name the row down the left edge; without axis labels there is
        # otherwise nothing tying a row to its variable except the colourbar.
        pos = axes[0].get_position()
        fig.text(0.012, pos.y0 + pos.height / 2, label, rotation=90,
                 va="center", ha="center", fontsize=12)

        sm = cm.ScalarMappable(cmap=plt.get_cmap(cmap),
                               norm=matplotlib.colors.Normalize(lo, hi))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[:2], fraction=0.020, pad=0.02, label=label)
        if d is not None:
            sd = cm.ScalarMappable(cmap=plt.get_cmap(DIFF_CMAP),
                                   norm=matplotlib.colors.Normalize(-m, m))
            sd.set_array([])
            fig.colorbar(sd, ax=axes[2], fraction=0.030, pad=0.02,
                         label="Δ " + label)

    fig.suptitle(title, fontsize=16, y=0.995)
    place_well_labels(fig)
    fig.savefig(path, dpi=PANEL_DPI)
    plt.close(fig)
    return path


def render_panels(outdir, case, A, B, pairs, wells, workers, cgrid_path=None,
                  tail=()):
    """One panel PNG per aligned report step, rendered in parallel.

    tail lists (ib, t) for NVRS report steps after Flow's last one: they are
    drawn with the NVRS column only, so a Flow run that stopped early (SPE10's
    Flow is run for the first report steps only) does not cut the movie short.
    """
    dims = tuple(A["fields"][FIELDS[0][0]].shape[1:])
    names = [f for f, _l, _c in FIELDS
             if np.nanmax(np.abs(A["fields"][f])) > 0
             or np.nanmax(np.abs(B["fields"][f])) > 0]
    # Each step scaled on its own, shared between the two simulators, as the
    # reports did before the September merge.
    jobs, paths = [], []
    for s, (ia, ib, t) in enumerate(pairs):
        rows = []
        for f, label, cmap in FIELDS:
            if f not in names:
                continue
            a = A["fields"][f][ia]
            b = B["fields"][f][ib]
            rows.append((label, cmap, a, b, a - b))
        p = os.path.join(outdir, "panel_%s_%04d.png" % (case, s))
        jobs.append((p, "%s  —  t = %g days" % (case, t), rows, wells,
                     30, 45, cgrid_path, dims, A["base"] + ".EGRID"))
        paths.append(p)
    for s2, (ib, t) in enumerate(tail):
        rows = []
        for f, label, cmap in FIELDS:
            if f not in names:
                continue
            rows.append((label, cmap, None, B["fields"][f][ib], None))
        p = os.path.join(outdir, "panel_%s_%04d.png" % (case, len(pairs) + s2))
        jobs.append((p, "%s  —  t = %g days (NVRS only; Flow stopped at %g days)"
                     % (case, t, float(pairs[-1][2]) if pairs else 0.0),
                     rows, wells, 30, 45, cgrid_path, dims,
                     A["base"] + ".EGRID"))
        paths.append(p)

    log("  %s: %d panels x %d rows on %d worker(s)"
        % (case, len(jobs), len(names), workers))
    done = 0
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(_panel_worker, jobs):
                done += 1
                log("  %s: %s" % (case, progress(len(jobs), done)))
    else:
        for j in jobs:
            _panel_worker(j)
            done += 1
            log("  %s: %s" % (case, progress(len(jobs), done)))
    return paths, names


def progress(total, done, length=20):
    frac = 0.0 if total <= 0 else float(done) / total
    n = int(frac * length)
    return "[%s%s] %5.1f%%" % ("#" * n, "-" * (length - n), 100 * frac)


# ===========================================================================
# Well curves
# ===========================================================================
# --- summary units ---------------------------------------------------------
#
# ECLIPSE records the unit of every summary vector in the SMSPEC, and a deck
# written in METRIC produces a summary in metric: rates in sm3/day, pressures
# in barsa, volumes in sm3.  NVRS solves and reports in field units.  Comparing
# the two without converting puts a 6.29x on every rate and a 14.5x on every
# pressure, which is what made OPM's bottom-hole pressures read as ~250 against
# NVRS's ~2900 -- not a modelling difference, a unit.
#
# Driven entirely by the unit string the file declares, so a field-unit summary
# passes through untouched and nothing has to be assumed about the deck.
_M3_TO_BBL = 6.289810770
_M3_TO_FT3 = 35.31466672
_BAR_TO_PSI = 14.503773773

def _is_gas(keyword):
    """WGPR, FGPT, WGIR ... the phase letter is the second character."""
    k = (keyword or "").upper()
    return len(k) > 1 and k[1] == "G" and not k.startswith("WGOR")         and not k.startswith("FGOR")

def _unit_factor(keyword, unit):
    """Multiplier taking one metric summary vector into field units."""
    u = (unit or "").strip().upper().replace(" ", "")
    k = (keyword or "").upper()
    if not u:
        return 1.0, unit
    if u in ("BARSA", "BARS", "BAR"):
        return _BAR_TO_PSI, "PSIA"
    if u in ("SM3/SM3", "SM3/SM3)"):          # gas-oil ratio
        return _M3_TO_FT3 / _M3_TO_BBL / 1000.0, "MSCF/STB"
    if u in ("SM3/DAY", "SM3/D"):
        if _is_gas(k):
            return _M3_TO_FT3 / 1000.0, "MSCF/DAY"
        return _M3_TO_BBL, "STB/DAY"
    if u == "SM3":
        if _is_gas(k):
            return _M3_TO_FT3 / 1000.0, "MSCF"
        return _M3_TO_BBL, "STB"
    if u in ("RM3/DAY", "RM3/D"):
        return _M3_TO_BBL, "RB/DAY"
    if u == "RM3":
        return _M3_TO_BBL, "RB"
    if u == "M":
        return 1.0 / 0.3048, "FT"
    return 1.0, unit                          # already field, or dimensionless

def to_field_units(vec, label=""):
    """Convert a metric summary in place; report once what was converted."""
    if not vec:
        return vec
    changed = {}
    for name, e in vec.items():
        f, newu = _unit_factor(e.get("keyword", name.split(":")[0]),
                               e.get("unit", ""))
        if f != 1.0:
            e["values"] = np.asarray(e["values"], dtype=float) * f
            changed.setdefault((e.get("unit", ""), newu), 0)
            changed[(e.get("unit", ""), newu)] += 1
            e["unit"] = newu
    if changed:
        parts = ", ".join("%s->%s x%d" % (a or "?", b, n)
                          for (a, b), n in sorted(changed.items()))
        log("  [units] %s summary was METRIC, converted: %s"
            % (label or "case", parts))
    return vec


def init_field_field_units(case, kw):
    """An INIT array in field units, whatever the run was written in.

    PORV is a reservoir volume: m3 in a metric run, rb in a field one.
    PERMX is millidarcy in both and PORO is dimensionless, so only the
    volumes and lengths move.
    """
    c = case["case"] if isinstance(case, dict) else case
    v = c.init_field(kw)
    if v is None:
        return None
    if (c.unit_system or "").upper() != "METRIC":
        return v
    k = kw.upper()
    if k == "PORV":
        return np.asarray(v, dtype=float) * _M3_TO_BBL
    if k in ("DEPTH", "DZ", "DX", "DY", "TOPS"):
        return np.asarray(v, dtype=float) / 0.3048
    return v


def summary_of(case):
    t, vec = case["case"].summary()
    return (np.asarray(t, dtype=float) if t is not None else None),         to_field_units(vec, "loaded case")


def summary_of_base(base):
    """(times, vectors) straight from an ECLIPSE base path.

    The well curves used to take LOADED cases, which meant they could only be
    drawn for runs whose restart fields had already been read -- and passing a
    base path where a loaded case was expected gave

        ! well curves skipped: TypeError: string indices must be integers

    Reading the summary directly also decouples them from the field loading:
    a run whose EGRID a reader rejects still has perfectly good well vectors,
    and there is no reason to lose both.
    """
    if not base:
        return (None, None)
    try:
        t, vec = ecl.EclipseCase(base).summary()
    except Exception as exc:
        log("  ! no summary for %s: %s" % (os.path.basename(base), exc))
        return (None, None)
    return (np.asarray(t, dtype=float) if t is not None else None),         to_field_units(vec, os.path.basename(base))


def field_series(vec, kw):
    """A field summary vector, accepting both field-key naming conventions."""
    e = vec.get(kw)
    if e is None:
        for key, cand in vec.items():
            if key == kw or key.startswith(kw + ":"):
                e = cand
                break
    return None if e is None else np.asarray(e["values"], dtype=float)


@guarded("field injection")
def render_field_injection(outdir, case, opm_base, nvrs_sources):
    """Plot accepted injection so a water-volume deficit has a cause."""
    t_ref, v_ref = summary_of_base(opm_base)
    if t_ref is None:
        return None
    sources = [("OPM Flow", t_ref, v_ref)]
    for label, base in (nvrs_sources or []):
        t, v = summary_of_base(base)
        if t is not None:
            sources.append((label, t, v))

    quantities = [("FWIR", "Water injection (STB/day)"),
                  ("FGIR", "Gas injection (MSCF/day)")]
    live = [(kw, label) for kw, label in quantities
            if any(field_series(v, kw) is not None for _, _, v in sources)]
    if not live:
        return None

    fig, axes = plt.subplots(len(live), 1, squeeze=False,
                             figsize=(12.5, 3.3 * len(live)))
    for row, (kw, label) in enumerate(live):
        ax = axes[row][0]
        for sname, t, v in sources:
            y = field_series(v, kw)
            if y is None:
                continue
            n = min(len(t), len(y))
            ls, lw = src_style(sname)
            ax.plot(t[:n], y[:n], linestyle=ls, color=src_colour(sname), lw=lw,
                    label=sname)
        ax.set_ylabel(label)
        ax.set_xlabel("days")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.suptitle("%s - field injection actually accepted" % case,
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(outdir, "field_injection_%s.png" % case)
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def well_series(vec, kw, well, derive=None):
    """One well vector, deriving flowing WWCT/WGOR from their phase rates."""
    if derive is None:
        e = vec.get("%s:%s" % (kw, well))
        return None if e is None else np.asarray(e["values"], dtype=float)
    num, den, operation = derive
    a = vec.get("%s:%s" % (num, well))
    b = vec.get("%s:%s" % (den, well))
    if a is None or b is None:
        return None
    a = np.asarray(a["values"], dtype=float)
    b = np.asarray(b["values"], dtype=float)
    # Match OPM Summary.cpp: producer phase rates are non-negative and division
    # by an exactly zero denominator returns zero.  Do not hide low-rate points
    # in the plotted series; the diagnostic mask below excludes them from R2
    # and NRMSE without changing the physical curve the user sees.
    out = np.zeros(a.shape, dtype=float)
    if operation == "cut":
        ap = np.maximum(a, 0.0)
        bp = np.maximum(b, 0.0)
        liquid = ap + bp
        ok = liquid > 0.0
        out[ok] = ap[ok] / liquid[ok]
    else:
        ap = np.maximum(a, 0.0)
        bp = np.maximum(b, 0.0)
        ok = bp > 0.0
        out[ok] = ap[ok] / bp[ok]
    return out


def constant_reference_scale(ref, kw):
    """The yardstick R2 uses when the OPM reference does not vary.

    R2 divides by the reference's variance, which is zero for a producer held
    at a fixed bottom-hole pressure (every SPE10 and BENCH3P producer, and
    Norne wells on their BHP limit) or a water cut that stays at zero.  The
    ratio is then undefined -- the report used to print "ill-conditioned" and
    count the well as failing even when NVRS reproduced it exactly.  Instead
    the error is measured against 5% of the constant level (1 for a water
    cut, which is already a fraction), the same scale the diagnostics' NRMSE
    uses: an exact match scores 1.0, a 10 psi miss on 4000 psi 0.9975.
    """
    ref = np.asarray(ref, dtype=float)
    if kw == "WWCT":
        return 1.0
    floor = 0.001 if kw == "WGPR" else 1.0
    level = float(np.max(np.abs(ref))) if ref.size else 0.0
    return max(0.05 * level, floor)


def _is_constant(ref, scale):
    ref = np.asarray(ref, dtype=float)
    if ref.size < 2:
        return True
    return float(np.std(ref)) <= 1.0e-9 * max(scale, 1.0e-30) + 1.0e-24


def _well_activity(vec, well):
    """Reference-time flowing mask from the well's own phase rates."""
    phase = []
    for kw in ("WOPR", "WWPR", "WGPR"):
        e = vec.get("%s:%s" % (kw, well))
        if e is not None:
            phase.append((kw, np.asarray(e["values"], dtype=float)))
    if not phase:
        return None
    n = min(len(x[1]) for x in phase)
    active = np.zeros(n, dtype=bool)
    for kw, values in phase:
        values = np.nan_to_num(values[:n], nan=0.0)
        # WGPR is stored in MSCF/day; 0.001 MSCF/day is one SCF/day.
        threshold = 0.001 if kw == "WGPR" else 1.0
        active |= np.abs(values) >= threshold
    return active


def _interp_finite(t_src, y_src, t_dst):
    """Linear interpolation that does not bridge through stored NaNs."""
    n = min(len(t_src), len(y_src))
    t = np.asarray(t_src[:n], dtype=float)
    y = np.asarray(y_src[:n], dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    out = np.full(len(t_dst), np.nan)
    if ok.sum() >= 2:
        out[:] = np.interp(t_dst, t[ok], y[ok], left=np.nan, right=np.nan)
    return out


def _interp_previous(t_src, y_src, t_dst):
    """Sample a piecewise-constant schedule/state series without ramping it."""
    n = min(len(t_src), len(y_src))
    t = np.asarray(t_src[:n], dtype=float)
    y = np.asarray(y_src[:n])
    ok = np.isfinite(t)
    out = np.zeros(len(t_dst), dtype=y.dtype)
    if not ok.any():
        return out
    t = t[ok]
    y = y[ok]
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    pos = np.searchsorted(t, np.asarray(t_dst, dtype=float), side="right") - 1
    have = pos >= 0
    out[have] = y[pos[have]]
    return out


def _ratio_metric_mask(vec, well, derive, t_src, t_dst):
    """Exclude shut/trace-flow ratios from accuracy scores, not from plots."""
    if derive is None:
        return np.ones(len(t_dst), dtype=bool)
    num, den, operation = derive
    a = vec.get("%s:%s" % (num, well))
    b = vec.get("%s:%s" % (den, well))
    if a is None or b is None:
        return np.zeros(len(t_dst), dtype=bool)
    av = _interp_finite(t_src, np.asarray(a["values"], dtype=float), t_dst)
    bv = _interp_finite(t_src, np.asarray(b["values"], dtype=float), t_dst)
    if operation == "cut":
        scale = np.maximum(av, 0.0) + np.maximum(bv, 0.0)
    else:
        scale = np.maximum(bv, 0.0)  # oil denominator for WGOR
    # One STB/day is far below a flowing Norne producer, but above numerical
    # residue.  Requiring it in both simulations prevents a tiny denominator
    # in either curve from dominating R2 by orders of magnitude.
    return np.isfinite(scale) & (scale >= 1.0)


def compute_well_diagnostics(opm_base, nvrs_sources, producers):
    """Scale-aware active-period metrics for every well quantity/backend.

    Raw R2 remains in the report, but it is ill-conditioned for a nearly
    constant or mostly shut reference.  These rows add the evidence needed to
    decide whether a low score is physical: active-time NRMSE, cumulative
    rate error and opening/closing shifts.
    """
    t_ref, v_ref = summary_of_base(opm_base)
    if t_ref is None:
        return None
    sources = []
    for label, base in (nvrs_sources or []):
        t, v = summary_of_base(base)
        if t is not None:
            sources.append((label, t, v))
    if not sources:
        return None

    rows = []
    rate_kw = {"WOPR", "WWPR", "WGPR"}
    for kw, _label, derive in WELL_QTY:
        for well in producers:
            y_ref0 = well_series(v_ref, kw, well, derive)
            a_ref0 = _well_activity(v_ref, well)
            if y_ref0 is None or a_ref0 is None:
                continue
            n = min(len(t_ref), len(y_ref0), len(a_ref0))
            if n < 2:
                continue
            tc = np.asarray(t_ref[:n], dtype=float)
            y_ref = np.asarray(y_ref0[:n], dtype=float)
            a_ref = np.asarray(a_ref0[:n], dtype=bool)

            for source, t_sim, v_sim in sources:
                y_sim0 = well_series(v_sim, kw, well, derive)
                a_sim0 = _well_activity(v_sim, well)
                if y_sim0 is None or a_sim0 is None:
                    continue
                m = min(len(t_sim), len(y_sim0), len(a_sim0))
                ts = np.asarray(t_sim[:m], dtype=float)
                post = ts > 0.0
                y_sim = _interp_finite(
                    ts[post], np.asarray(y_sim0[:m], dtype=float)[post], tc)
                # Activity is a schedule state, not a linearly varying value.
                # Linear interpolation invents half-open wells and can move a
                # transition by one irregular Norne report interval.
                a_sim = _interp_previous(
                    ts[post], np.asarray(a_sim0[:m], dtype=bool)[post],
                    tc).astype(bool)

                valid = a_ref & np.isfinite(y_ref) & np.isfinite(y_sim)
                if derive is not None:
                    valid &= _ratio_metric_mask(v_ref, well, derive, tc, tc)
                    valid &= _ratio_metric_mask(v_sim, well, derive,
                                                t_sim, tc)
                n_active = int(valid.sum())
                if n_active < 2:
                    continue
                ref = y_ref[valid]
                sim = y_sim[valid]
                err = sim - ref
                var = float(np.mean((ref - ref.mean()) ** 2))
                cscale = constant_reference_scale(ref, kw)
                const_ref = _is_constant(ref, cscale)
                if const_ref:
                    r2 = 1.0 - float(np.mean(err * err)) / (cscale * cscale)
                else:
                    r2 = 1.0 - float(np.mean(err * err)) / var

                p05, p95 = np.percentile(ref, [5.0, 95.0])
                characteristic = max(abs(float(p95 - p05)),
                                     0.05 * float(np.max(np.abs(ref))),
                                     1.0e-12)
                # Fractions have a natural unit scale. Rates and pressures use
                # the robust active range, with a small magnitude floor for a
                # nearly constant history target.
                if kw == "WWCT":
                    characteristic = 1.0
                elif kw in rate_kw or kw == "WBHP":
                    characteristic = max(
                        characteristic,
                        0.05 * float(np.percentile(np.abs(ref), 95.0)),
                        1.0 if kw != "WGPR" else 0.001)
                nrmse = math.sqrt(float(np.mean(err * err))) / characteristic

                cum_err = None
                if kw in rate_kw:
                    rf = np.where(np.isfinite(y_ref), np.maximum(y_ref, 0.0), 0.0)
                    sf = np.where(np.isfinite(y_sim), np.maximum(y_sim, 0.0), 0.0)
                    trap = getattr(np, "trapezoid", None)
                    if trap is None:  # NumPy before 2.0
                        trap = np.trapz
                    cr = float(trap(rf, tc))
                    cs = float(trap(sf, tc))
                    if abs(cr) > 1.0e-12:
                        cum_err = 100.0 * (cs - cr) / abs(cr)

                ref_idx = np.flatnonzero(a_ref)
                sim_idx = np.flatnonzero(a_sim)
                open_shift = close_shift = None
                if ref_idx.size and sim_idx.size:
                    open_shift = float(tc[sim_idx[0]] - tc[ref_idx[0]])
                    close_shift = float(tc[sim_idx[-1]] - tc[ref_idx[-1]])

                # Water breakthrough: the first report time at which the water
                # cut reaches 1%, in both runs.  The offset, in days and in
                # report steps, is what decides whether a large negative R2 on
                # a late-breakthrough well is one step of timing (a metric
                # artefact) or a physics difference.
                bt_ref = bt_sim = bt_days = bt_steps = None
                if kw == "WWCT":
                    hit_r = np.flatnonzero(np.nan_to_num(y_ref) >= 0.01)
                    hit_s = np.flatnonzero(np.nan_to_num(y_sim) >= 0.01)
                    if hit_r.size:
                        bt_ref = float(tc[hit_r[0]])
                    if hit_s.size:
                        bt_sim = float(tc[hit_s[0]])
                    if hit_r.size and hit_s.size:
                        bt_days = bt_sim - bt_ref
                        bt_steps = int(hit_s[0] - hit_r[0])

                rows.append(dict(source=source, well=well, quantity=kw,
                                 active=n_active, r2=r2, nrmse=100.0 * nrmse,
                                 constant_reference=const_ref,
                                 cumulative=cum_err,
                                 open_shift=open_shift,
                                 close_shift=close_shift,
                                 breakthrough_ref_day=bt_ref,
                                 breakthrough_sim_day=bt_sim,
                                 breakthrough_offset_days=bt_days,
                                 breakthrough_offset_steps=bt_steps))

    low = [r for r in rows if not np.isfinite(r["r2"]) or r["r2"] < 0.80]
    low.sort(key=lambda r: (float("-inf") if not np.isfinite(r["r2"])
                            else r["r2"], -r["nrmse"]))
    return dict(rows=rows, low=low, threshold=0.80)


@guarded("well curves")
def render_well_curves(outdir, case, opm_base, nvrs_sources, producers,
                       per_page=30):
    """One PANEL PER WELL, every simulator on it, gridded five across.

    The previous figure put every well on one axes: thirty-three producers,
    two lines each, sixty-six entries in the legend.  Nothing in it could be
    read.  A rate is only meaningful against the same rate from another
    simulator, so the comparison belongs inside one well's panel, and the
    grid is what puts thirty-three of those on a page.

    Twenty-five panels to a page (five by five).  A case with more wells gets
    more pages rather than smaller panels, since the point is legibility.
    """
    tA, vA = summary_of_base(opm_base)
    if tA is None:
        log("  ! no SMSPEC/UNSMRY for OPM; skipping well curves")
        return []

    # Every source that has a summary, OPM first.
    sources = [("OPM Flow", tA, vA)]
    for label, base in (nvrs_sources or []):
        t, v = summary_of_base(base)
        if t is not None:
            sources.append((label, t, v))
    if len(sources) < 2:
        log("  ! only one simulator has well output; skipping well curves")
        return []

    # Colour and style come from the shared palette, so a simulator is the
    # same colour here as in the accuracy bars and the timing charts.

    made = []
    for kw, label, derive in WELL_QTY:
        # Keep only wells some source actually reports.
        live = []
        for w in producers:
            if any(well_series(v, kw, w, derive) is not None
                   for _, _, v in sources):
                live.append(w)
        if not live:
            continue

        # A quantity gets one axis range across every producer and simulator.
        # Per-panel autoscaling made a 100 STB/day well look as important as a
        # 10,000 STB/day well and prevented magnitude comparisons by eye.
        all_y = []
        all_t = []
        for w in live:
            for _sname, t, v in sources:
                y = well_series(v, kw, w, derive)
                if y is None:
                    continue
                n = min(len(t), len(y))
                tx = np.asarray(t[:n], dtype=float)
                yy = np.asarray(y[:n], dtype=float)
                if kw == "WBHP":
                    active = _well_activity(v, w)
                    if active is not None:
                        na = min(n, len(active))
                        yy = yy.copy()
                        yy[:na] = np.where(active[:na], yy[:na], 0.0)
                        if na < n:
                            yy[na:] = 0.0
                keep = np.isfinite(tx) & np.isfinite(yy) & (tx > 0.0)
                if np.any(keep):
                    all_t.append(tx[keep])
                    all_y.append(yy[keep])
        if all_y:
            values = np.concatenate(all_y)
            ymin, ymax = float(np.min(values)), float(np.max(values))
            if kw == "WWCT":
                ymin, ymax = 0.0, 1.0
            elif kw in ("WOPR", "WWPR", "WGPR", "WGOR") and ymin >= 0.0:
                ymin = 0.0
                ymax *= 1.04
            else:
                pad = 0.04 * max(ymax - ymin, abs(ymax), 1.0)
                ymin, ymax = ymin - pad, ymax + pad
            if ymax <= ymin:
                ymax = ymin + 1.0
            xmax = max(float(np.max(x)) for x in all_t)
        else:
            ymin, ymax, xmax = 0.0, 1.0, 1.0

        for page0 in range(0, len(live), per_page):
            page = live[page0:page0 + per_page]
            ncol = 5
            nrow = int(math.ceil(len(page) / float(ncol)))
            fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                     figsize=(3.5 * ncol, 2.6 * nrow))
            for i, w in enumerate(page):
                ax = axes[i // ncol][i % ncol]
                drew = False
                for sname, t, v in sources:
                    y = well_series(v, kw, w, derive)
                    if y is None:
                        continue
                    col = src_colour(sname)
                    ls, lw = src_style(sname)
                    n = min(len(t), len(y))
                    tx = np.asarray(t[:n], dtype=float)
                    yy = np.asarray(y[:n], dtype=float)
                    # Summary rates are end-of-timestep quantities. NVRS also
                    # retains the instantaneous initialized-state rate at day
                    # zero, whereas Flow starts after its first accepted step.
                    # Joining that drawdown spike to the first report endpoint
                    # destroys the useful plot scale and is not a like-time
                    # comparison, so show only post-step samples here.
                    if kw == "WBHP":
                        active = _well_activity(v, w)
                        if active is not None:
                            na = min(n, len(active))
                            yy = yy.copy()
                            yy[:na] = np.where(active[:na], yy[:na], 0.0)
                            if na < n:
                                yy[na:] = 0.0
                    keep = np.isfinite(tx) & np.isfinite(yy) & (tx > 0.0)
                    ax.plot(tx[keep], yy[keep], linestyle=ls, color=col, lw=lw,
                            marker="o" if np.count_nonzero(keep) < 3 else None,
                            label=sname if i == 0 else None)
                    drew = True
                ax.set_title(w, fontsize=9, weight="bold")
                ax.set_xlim(0.0, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_facecolor("#fbfcfd")
                ax.minorticks_on()
                ax.grid(which="major", color="#9aa4ad", alpha=0.32,
                        linewidth=0.55)
                ax.grid(which="minor", color="#c8ced4", alpha=0.16,
                        linewidth=0.35)
                ax.tick_params(which="both", direction="in", top=True,
                               right=True, labelsize=7, width=0.6)
                for spine in ax.spines.values():
                    spine.set_color("#59636d")
                    spine.set_linewidth(0.65)
                if kw != "WWCT":
                    ax.ticklabel_format(axis="y", style="sci",
                                        scilimits=(-3, 4), useMathText=True)
                    ax.yaxis.get_offset_text().set_fontsize(7)
                if not drew:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="0.5")
                # Axis labels only on the edges: inside a grid they are noise.
                if i // ncol == nrow - 1:
                    ax.set_xlabel("days", fontsize=8)
                if i % ncol == 0:
                    ax.set_ylabel(label, fontsize=8)
            for k in range(len(page), nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")

            handles = []
            for sname, _t, _v in sources:
                ls, lw = src_style(sname)
                handles.append(Line2D([], [], color=src_colour(sname),
                                      linestyle=ls, linewidth=lw,
                                      label=sname))
            if handles:
                fig.legend(handles=handles, loc="upper center",
                           ncol=min(len(handles), 3), frameon=False,
                           fontsize=10, bbox_to_anchor=(0.5, 0.035))
            pages = int(math.ceil(len(live) / float(per_page)))
            ttl = "%s — %s by producer (shared x/y scales)" % (case, label)
            if pages > 1:
                ttl += "  (%d of %d)" % (page0 // per_page + 1, pages)
            fig.suptitle(ttl, fontsize=13, weight="bold")
            fig.tight_layout(rect=(0, 0.045 + 0.02 * (len(handles) > 3), 1,
                                   0.97))

            suffix = "" if pages == 1 else "_%d" % (page0 // per_page + 1)
            pth = os.path.join(outdir, "well_%s_%s%s.png" % (case, kw, suffix))
            fig.savefig(pth, dpi=REPORT_DPI, bbox_inches="tight")
            plt.close(fig)
            made.append(pth)
        log("  %s: %d producer panel(s)" % (kw, len(live)))
    return made


@guarded("well accuracy")
def render_well_r2(outdir, case, opm_base, nvrs_sources, producers,
                   per_page=30, collect=None):
    """R2 against OPM Flow THROUGH TIME, one panel per well.

    Elementwise, on the residual that step actually has: ten OPM values
    against ten NVRS values give ten R2 values.  The denominator is the OPM
    series' own variance, so a step that matches exactly reads 1.0 and the
    mean of the curve is the ordinary R2 of the whole history -- printed in
    the corner, so the panel gives both the number and where it came from.

    OPM is the reference, so it has no line: one line per NVRS backend, in
    that backend's colour.  R2 can go negative and is drawn there rather than
    clipped, since a well tracked worse than a flat line through the
    reference is a different failure from one merely imprecise.

    `collect`, when a dict, receives each panel's number -- the whole-history
    R2 of that well -- as collect[(backend label, quantity, well)], so the
    backend table averages exactly the values these charts print.
    """
    tA, vA = summary_of_base(opm_base)
    if tA is None:
        return []
    sources = []
    for label, base in (nvrs_sources or []):
        t, v = summary_of_base(base)
        if t is not None:
            sources.append((label, t, v))
    if not sources:
        return []

    def r2_series(ya, tb, yb, kw=None):
        """R2 at each timestep, from that step's own residual.

        Elementwise: OPM (a, b, c) against NVRS (d, e, f) gives residuals
        a-d, b-e, c-f, and each becomes

            R2_k = 1 - (ref_k - sim_k)^2 / var(ref)

        with var(ref) the mean squared deviation of the OPM series about its
        own mean -- the same denominator the ordinary R2 uses, so a step that
        matches exactly reads 1.0, and the MEAN of this series is exactly the
        ordinary R2 of the whole history.  Each point is therefore that step's
        own contribution to the total, which is what makes the curve worth
        plotting rather than a running average that hides where the error came
        from.
        """
        n = min(len(tA), len(ya))
        m = min(len(tb), len(yb))
        if n < 2 or m < 2:
            return None
        tc = np.asarray(tA[:n], dtype=float)
        ref = np.asarray(ya[:n], dtype=float)
        ts = np.asarray(tb[:m], dtype=float)
        ys = np.asarray(yb[:m], dtype=float)
        # Do not interpolate through NVRS's initialized-state rate at t=0;
        # Flow has no like-time sample there and the first implicit transient
        # is intentionally steep.
        post = ts > 0.0
        sim = _interp_finite(ts[post], ys[post], tc)
        ok = np.isfinite(ref) & np.isfinite(sim)
        if ok.sum() < 2:
            return None
        var = float(np.mean((ref[ok] - ref[ok].mean()) ** 2))
        cscale = constant_reference_scale(ref[ok], kw)
        if _is_constant(ref[ok], cscale):
            # Constant reference (fixed-BHP producer, zero water cut): its
            # variance is zero, so measure against 5% of the level instead
            # (constant_reference_scale).  The well keeps its panel and its
            # place in the backend table's mean.
            var = cscale * cscale
        out = np.full(n, np.nan)
        out[ok] = 1.0 - (ref[ok] - sim[ok]) ** 2 / var
        return (tc, out)

    made = []
    for kw, label, derive in WELL_QTY:
        rows = []
        for w in producers:
            ya = well_series(vA, kw, w, derive)
            if ya is None:
                continue
            lines = []
            for sname, tb, vb in sources:
                yb = well_series(vb, kw, w, derive)
                r = None if yb is None else r2_series(ya, tb, yb, kw)
                if r is not None:
                    lines.append((sname, r[0], r[1]))
                    f = r[1][np.isfinite(r[1])]
                    if collect is not None and f.size:
                        collect[(sname, kw, w)] = float(f.mean())
            if lines:
                rows.append((w, lines))
        if not rows:
            continue

        for page0 in range(0, len(rows), per_page):
            page = rows[page0:page0 + per_page]
            ncol = 5
            nrow = int(math.ceil(len(page) / float(ncol)))
            fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                     figsize=(3.5 * ncol, 2.6 * nrow))
            # A common floor, so panels can be compared to each other.
            lo = 0.0
            for _, lines in page:
                for _, _, y in lines:
                    f = y[np.isfinite(y)]
                    if f.size:
                        lo = min(lo, float(f.min()))
            lo = max(lo, -1.0)
            for i, (w, lines) in enumerate(page):
                ax = axes[i // ncol][i % ncol]
                for si, (sname, tc, y) in enumerate(lines):
                    ax.plot(tc, np.clip(y, lo, None), "-",
                            color=src_colour(sname), lw=1.6,
                            label=sname if i == 0 else None)
                    f = y[np.isfinite(y)]
                    if f.size:
                        # The mean of the per-step values IS the ordinary R2
                        # of the whole history, so annotating it costs nothing
                        # and gives the single number as well as the curve.
                        # Stacked, since two annotations at one anchor
                        # overprint and neither can be read.
                        ax.annotate("%.3f" % float(f.mean()),
                                    xy=(0.98, 0.05 + 0.115 * si),
                                    xycoords="axes fraction", ha="right",
                                    va="bottom", fontsize=7.5,
                                    fontweight="bold",
                                    color=src_colour(sname),
                                    bbox=dict(boxstyle="round,pad=0.12",
                                              facecolor="white",
                                              edgecolor="none", alpha=0.9),
                                    zorder=10)
                ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
                ax.axhline(0.0, color="0.2", lw=0.9)
                ax.set_ylim(lo - 0.05 * (1.0 - lo), 1.0 + 0.10 * (1.0 - lo))
                ax.set_title(w, fontsize=9, weight="bold")
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=7)
                if i // ncol == nrow - 1:
                    ax.set_xlabel("days", fontsize=8)
                if i % ncol == 0:
                    ax.set_ylabel("R² vs OPM", fontsize=8)
            for k in range(len(page), nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")

            h, l = axes[0][0].get_legend_handles_labels()
            if h:
                fig.legend(h, l, loc="upper center", ncol=min(len(h), 3),
                           frameon=False, fontsize=10,
                           bbox_to_anchor=(0.5, 0.035))
            pages = int(math.ceil(len(rows) / float(per_page)))
            ttl = ("%s - %s: R² per timestep against OPM Flow, per producer"
                   % (case, kw))
            if pages > 1:
                ttl += "  (%d of %d)" % (page0 // per_page + 1, pages)
            fig.suptitle(ttl, fontsize=13, weight="bold")
            fig.tight_layout(rect=(0, 0.045 + 0.02 * (len(h) > 3), 1, 0.96))
            suffix = "" if pages == 1 else "_%d" % (page0 // per_page + 1)
            pth = os.path.join(outdir, "wellr2_%s_%s%s.png" % (case, kw, suffix))
            fig.savefig(pth, dpi=REPORT_DPI, bbox_inches="tight")
            plt.close(fig)
            made.append(pth)
        log("  %s: per-step R2 for %d producer(s)" % (kw, len(rows)))
    return made


# ===========================================================================
# Metrics
# ===========================================================================
def _uniform3(a, w):
    """Box mean over a w-cube, via a summed-area table.  No scipy needed."""
    pad = w // 2
    b = np.pad(a, pad, mode="edge")
    c = b.cumsum(0).cumsum(1).cumsum(2)
    c = np.pad(c, ((1, 0), (1, 0), (1, 0)), mode="constant")
    n = a.shape
    sl = lambda o, d: slice(o, o + n[d])
    tot = (c[sl(w, 0), sl(w, 1), sl(w, 2)] - c[sl(0, 0), sl(w, 1), sl(w, 2)]
           - c[sl(w, 0), sl(0, 1), sl(w, 2)] - c[sl(w, 0), sl(w, 1), sl(0, 2)]
           + c[sl(0, 0), sl(0, 1), sl(w, 2)] + c[sl(0, 0), sl(w, 1), sl(0, 2)]
           + c[sl(w, 0), sl(0, 1), sl(0, 2)] - c[sl(0, 0), sl(0, 1), sl(0, 2)])
    return tot / float(w ** 3)


def accuracy_triplet(a, b, win=7):
    """R^2, relative L2 and SSIM between two fields.

    Three numbers because they fail in different ways and a single one hides
    it.  R^2 says whether NVRS reproduces the VARIATION -- it can be poor while
    the fields agree closely, if the reference barely varies.  Relative L2 is
    the honest magnitude of the disagreement.  SSIM is the only one that cares
    where the error IS: two fields can match in norm while the front sits in
    the wrong place, and on a flood that is the whole answer.

    NaN cells -- the inactive part of a corner-point box -- are excluded from
    all three rather than treated as zeros, which would otherwise dominate a
    field that is 39% active.
    """
    A3 = np.asarray(a, dtype=float)
    B3 = np.asarray(b, dtype=float)
    vol = (A3.ndim == 3 and A3.shape == B3.shape)
    a = A3.ravel()
    b = B3.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan"), float("nan"), float("nan")
    av, bv = a[ok], b[ok]

    ss_res = float(np.sum((av - bv) ** 2))
    ss_tot = float(np.sum((av - np.mean(av)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    den = float(np.sqrt(np.sum(av * av)))
    l2 = float(np.sqrt(ss_res)) / den if den > 0 else float("nan")

    # SSIM.  Windowed on the volume when the shape is known and big enough,
    # since a global SSIM over a whole field averages away exactly the local
    # structure the measure exists to catch; global otherwise.
    dr = float(np.nanmax(av) - np.nanmin(av))
    if dr <= 0:
        return r2, l2, 1.0 if ss_res == 0 else 0.0
    C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    windowed = vol and min(A3.shape) >= win and A3.size <= 8e6
    if windowed:
        A, B = A3, B3
        M = np.isfinite(A) & np.isfinite(B)
        A = np.where(M, A, 0.0); B = np.where(M, B, 0.0)
        mu_a, mu_b = _uniform3(A, win), _uniform3(B, win)
        sa = _uniform3(A * A, win) - mu_a * mu_a
        sb = _uniform3(B * B, win) - mu_b * mu_b
        sab = _uniform3(A * B, win) - mu_a * mu_b
        num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
        den2 = (mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2)
        smap = np.where(den2 > 0, num / np.maximum(den2, 1e-30), 1.0)
        ssim = float(np.mean(smap[M])) if M.any() else float("nan")
    else:
        ma, mb = float(np.mean(av)), float(np.mean(bv))
        va, vb = float(np.var(av)), float(np.var(bv))
        cab = float(np.mean((av - ma) * (bv - mb)))
        ssim = (((2 * ma * mb + C1) * (2 * cab + C2)) /
                ((ma * ma + mb * mb + C1) * (va + vb + C2)))
    return r2, l2, ssim


def field_metrics(A, B, pairs, names):
    """Per-step, per-field agreement.

    Reported as an absolute RMS and as a relative RMS normalised by the range
    the field actually spans in the reference run, because an RMS of 3 psi
    means something quite different on a 200 psi drawdown than on a 0.6
    saturation swing.
    """
    rows = []
    for s, (ia, ib, t) in enumerate(pairs):
        rec = {"step": s, "time": t}
        for f in names:
            a = A["fields"][f][ia]
            b = B["fields"][f][ib]
            d = a - b
            rms = float(np.sqrt(np.nanmean(d * d)))
            linf = float(np.nanmax(np.abs(d)))
            # Normalising by the range the reference actually spans is only
            # meaningful once the field has some variation.  At t = 0 every
            # field is uniform, so the span is ~0 and the ratio would be a
            # large meaningless number; report it as unavailable instead.
            span = float(np.nanmax(a) - np.nanmin(a))
            scale = max(abs(float(np.nanmean(a))), 1e-12)
            rel = rms / span if span > 1e-6 * scale else float("nan")
            r2, l2, ssim = accuracy_triplet(a, b)
            rec[f] = (rms, linf, rel)
            rec[f + "_acc"] = (r2, l2, ssim)
        rows.append(rec)
    return rows


# Rows of the accuracy figure, in the order a reservoir engineer reads them.
ACC_ROWS = [("pressure", "Pressure"), ("water", "Water saturation"),
            ("gas", "Gas saturation"), ("oil", "Oil saturation")]
# Columns: the metric, which tuple slot it is, its reference value, log scale.
ACC_COLS = [("R$^2$", 0, 1.0, False), ("Relative L2", 1, 0.0, True),
            ("SSIM", 2, 1.0, False)]


@guarded("accuracy chart")
def accuracy_chart(outdir, case, rows, names):
    """NVRS-CUDA against OPM Flow, per report step: four rows by three columns.

    Rows are the fields -- pressure, water, gas, oil.  Columns are the three
    measures, because they fail differently and one number hides it:

      R^2          whether the VARIATION is reproduced.  It can look poor while
                   the fields agree closely, if the reference barely varies.
      relative L2  the honest magnitude of the disagreement.
      SSIM         the only one that cares WHERE the error is.  Two fields can
                   agree in norm with the flood front in the wrong place, and
                   on a waterflood the front is the whole answer.

    All three are measures BETWEEN the two simulators, so there is no separate
    OPM curve to draw: OPM against itself is 1, 0 and 1 by construction.  It is
    the dashed reference on each panel and the gap is what you read.

    Per step and not once at the end, because agreement is not a constant.  The
    two start from the same state; the question is whether they stay together
    as the wells work.  A drift that only opens up when a gas injector does is
    invisible in a run average and plain here.
    """
    if not rows:
        return None
    t = [r["time"] for r in rows]
    live = [(f, lab) for f, lab in ACC_ROWS
            if f in names and (f + "_acc") in rows[0]]
    if not live:
        return None

    fig, axes = plt.subplots(len(live), 3, figsize=(16, 3.4 * len(live)),
                             squeeze=False)
    for ri, (f, flab) in enumerate(live):
        series = [r[f + "_acc"] for r in rows]
        for ci, (mlab, which, ref, logy) in enumerate(ACC_COLS):
            ax = axes[ri][ci]
            y = [c[which] for c in series]
            # The reference line carries OPM Flow's colour like every other
            # place OPM Flow appears, rather than a neutral grey that has to be
            # read off a legend.
            ax.axhline(ref, color=src_colour("OPM Flow"), linestyle="--",
                       linewidth=1.5, label="OPM Flow (reference)")
            if np.any(np.isfinite(y)):
                ax.plot(t, y, marker="o", markersize=3, linewidth=1.5,
                        color=src_colour("NVRS-CUDA"), label="NVRS-CUDA")
            if logy and any(np.isfinite(v) and v > 0 for v in y):
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if ri == 0:
                ax.set_title(mlab, fontsize=13, weight="bold")
            if ci == 0:
                ax.set_ylabel(flab + chr(10) + mlab, fontsize=10)

            else:
                ax.set_ylabel(mlab, fontsize=9)
            if ri == len(live) - 1:
                ax.set_xlabel("time (days)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8, loc="best")
            fin = [v for v in y if np.isfinite(v)]
            if fin:
                ax.annotate("mean %.5g" % float(np.mean(fin)),
                            xy=(0.98, 0.06), xycoords="axes fraction",
                            ha="right", fontsize=8, color="0.25")

    fig.suptitle("%s  -  NVRS-CUDA against OPM Flow, per report step" % case,
                 fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, "accuracy_%s.png" % case)
    fig.savefig(out, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


@guarded("material balance")
def material_balance(outdir, case, A, B, pairs, producers, q_inj_total):
    """Surface-water inventory against reported injection and production."""
    # Both runs use the same matched deck. Read the constitutive constants
    # written by NVRS so static INIT PORV is corrected for rock compression
    # and Sw is converted from reservoir volume to stock-tank water.
    phys = dict(bw=1.0, cw=0.0, pref=0.0, cr=0.0)
    for mp in sorted(glob.glob(os.path.join(outdir, "**", "meta.json"),
                               recursive=True)):
        try:
            with open(mp, encoding="utf-8", errors="replace") as fh:
                md = json.load(fh)
            if "water_fvf_ref" in md:
                phys = dict(bw=float(md.get("water_fvf_ref", 1.0)),
                            cw=float(md.get("water_compressibility", 0.0)),
                            pref=float(md.get("rock_ref_pressure", 0.0)),
                            cr=float(md.get("rock_compressibility", 0.0)))
                break
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    rows = []
    for label, C, style in (("OPM Flow", A, "-"), ("NVRS-CUDA", B, "--")):
        porv = init_field_field_units(C, "PORV")
        if porv is None:
            log("  ! %s has no PORV in INIT; skipping material balance" % label)
            plt.close(fig)
            return None
        pv = np.nan_to_num(porv, nan=0.0)
        idx = [(ia if C is A else ib) for ia, ib, _t in pairs]
        t = np.array([tt for _a, _b, tt in pairs], dtype=float)
        W = []
        for i in idx:
            pressure = C["fields"]["pressure"][i]
            mult = np.exp((phys["cr"] + phys["cw"])
                          * (pressure - phys["pref"])) / max(phys["bw"], 1e-30)
            W.append(float(np.nansum(pv * C["fields"]["water"][i] * mult)))
        W = np.asarray(W)
        ax.plot(t, (W - W[0]) / 1e6, style, lw=2.0,
                label="%s  water in place" % label)
        rows.append((label, t, W))

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Change in surface water in place (million STB)")
    ax.set_title("%s - water balance: pressure-corrected water inventory" % case)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    path = os.path.join(outdir, "balance_%s.png" % case)
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

    days = float(pairs[-1][2]) if pairs else 0.0
    table = []
    for label, C, (_l, t, W) in (("OPM Flow", A, rows[0]),
                                 ("NVRS-CUDA", B, rows[1])):
        _tt, vec = summary_of(C)
        prod = 0.0
        injected = 0.0
        if _tt is not None and _tt.size:
            fwpt = vec.get("FWPT")
            if fwpt is not None and len(fwpt["values"]):
                prod = float(np.asarray(fwpt["values"], dtype=float)[-1])
            else:
                qw = np.zeros(_tt.size, dtype=float)
                for w in producers:
                    e = vec.get("WWPR:%s" % w)
                    if e is not None:
                        v = np.asarray(e["values"], dtype=float)[:_tt.size]
                        qw[:v.size] += v
                prod = float(np.trapezoid(qw, _tt)) if hasattr(np, "trapezoid") \
                    else float(np.trapz(qw, _tt))

            qwi = np.zeros(_tt.size, dtype=float)
            fwir = vec.get("FWIR")
            if fwir is not None:
                v = np.asarray(fwir["values"], dtype=float)[:_tt.size]
                qwi[:v.size] = v
            else:
                for key, ent in vec.items():
                    if key.startswith("WWIR:"):
                        v = np.asarray(ent["values"], dtype=float)[:_tt.size]
                        qwi[:v.size] += v
            ti = np.asarray(_tt, dtype=float)
            if ti.size and ti[0] > 0:
                ti = np.r_[0.0, ti]
                qwi = np.r_[qwi[0], qwi]
            injected = float(np.trapezoid(qwi, ti)) \
                if hasattr(np, "trapezoid") else float(np.trapz(qwi, ti))
        gain = W[-1] - W[0]
        implied = gain + prod
        err = (100.0 * (implied - injected) / injected) \
            if injected > 0 else float("nan")
        table.append(dict(name=label, w0=W[0] / 1e6, wend=W[-1] / 1e6,
                          gain=gain / 1e6, produced=prod / 1e6,
                          injected=injected / 1e6,
                          implied=implied / 1e6, err=err))
    return dict(path=path, table=table, days=days)


def _norm_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def ks_normal(x):
    """Kolmogorov-Smirnov distance from the best-fit normal, and its 5% bound.

    Returned so the report can say whether a permeability field is Gaussian in
    log space rather than asserting it.  Because the mean and standard
    deviation are estimated from the same sample the Lilliefors bound
    0.886/sqrt(n) applies, not the 1.36/sqrt(n) of a fully specified normal.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    sd = x.std(ddof=1)
    if n < 8 or not np.isfinite(sd) or sd <= 0:
        return float("nan"), float("nan")
    cdf = _norm_cdf((x - x.mean()) / sd)
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n)))
    return d, 0.886 / math.sqrt(n)


def moments(x):
    """(mean, sd, skewness, excess kurtosis)."""
    x = np.asarray(x, dtype=float)
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    if sd <= 0:
        return mu, sd, 0.0, 0.0
    z = (x - mu) / sd
    return mu, sd, float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


@guarded("rock figures")
def rock_figures(outdir, case, A, B, wells):
    """Permeability and porosity, and a check that both runs got the same rock.

    Both decks come from one field, so the two INIT files should agree exactly.
    Verifying it here is what lets the rest of the report claim it is comparing
    solvers rather than two different reservoirs.
    """
    perm = A["case"].init_field("PERMX")
    poro = A["case"].init_field("PORO")
    # Both decks carry the same rock, so the NVRS INIT is a faithful source
    # when Flow's does not have the arrays (SPE10's section vanished that way).
    for kw in ("PERMX", "PORO"):
        if (perm if kw == "PERMX" else poro) is None:
            v = B["case"].init_field(kw)
            if v is not None:
                log("  rock: %s from the NVRS INIT (not in OPM's)" % kw)
                if kw == "PERMX":
                    perm = v
                else:
                    poro = v
    if perm is None or poro is None:
        log("  ! rock figures: no %s in either INIT file"
            % " or ".join(k for k, v in (("PERMX", perm), ("PORO", poro))
                          if v is None))
        return None

    agree = {}
    for kw in ("PERMX", "PORO"):
        a = A["case"].init_field(kw)
        b = B["case"].init_field(kw)
        if a is not None and b is not None and a.shape == b.shape:
            denom = max(float(np.nanmax(np.abs(a))), 1e-30)
            agree[kw] = float(np.nanmax(np.abs(a - b))) / denom

    made = {}
    # Same geometry as the field panels.  These were still voxels on the index
    # box while the panels beside them drew the real structure, so the page
    # showed Norne two different shapes on one screen.
    cgrid_path = os.path.join(os.path.dirname(B["base"]), "cgrid.bin")
    if not os.path.exists(cgrid_path):
        cg = sorted(glob.glob(os.path.join(outdir, "**", "cgrid.bin"),
                              recursive=True))
        cgrid_path = cg[0] if cg else None
    cg = [cgrid_path] if cgrid_path else []
    geom = _corner_point(cg[0], tuple(np.asarray(perm).shape)) if cg else None
    # Same rule as the field panels: a thick grid is drawn as voxel blocks.
    if (geom is not None and geom[0].shape[2] > 40
            and os.environ.get("NVRS_FALLBACK_3D", "auto").lower() == "auto"):
        geom = None
    block = None
    if pv is not None:
        try:
            # Flow's reference EGRID is authoritative for both simulators.
            block = _pv_block_geometry(A["base"] + ".EGRID",
                                       tuple(np.asarray(perm).shape))
        except Exception as exc:
            log("  ! PyVista rock renderer unavailable (%s); using fallback"
                % exc)
    for field, fname, label, cmap in (
            (np.nan_to_num(perm, nan=np.nan), "perm", "Permeability (mD)",
             "jet"),
            (np.nan_to_num(poro, nan=np.nan), "poro", "Porosity", "jet")):
        path = os.path.join(outdir, "rock_%s_%s.png" % (case, fname))
        f = np.asarray(field)
        if case == "SPE10":
            # The native 60 x 220 x 85 grid cell by cell, permeability on a
            # log10 scale: seven decades do not fit a translucent voxel block.
            try:
                import spe10_rock_fig
                (spe10_rock_fig.draw_perm if fname == "perm"
                 else spe10_rock_fig.draw_poro)(f, path)
                made[fname] = path
                continue
            except Exception as exc:
                log("  ! SPE10 %s figure failed (%s: %s); using the voxel plot"
                    % (fname, type(exc).__name__, exc))
                plt.close("all")
        fin = f[np.isfinite(f)]
        if fin.size:
            vlo = float(np.percentile(fin, 1.0))
            vhi = float(np.percentile(fin, 99.0))
        else:
            vlo, vhi = 0.0, 1.0
        if not np.isfinite(vlo) or not np.isfinite(vhi) or vhi <= vlo:
            vlo = float(np.nanmin(f)) if fin.size else 0.0
            vhi = vlo + 1e-30
        if block is not None:
            _rock_figure_pyvista(
                path, label, f, cmap, block, vlo, vhi,
                float(np.nanmin(f)) if fin.size else vlo,
                float(np.nanmax(f)) if fin.size else vhi,
                wells=wells, cgrid_path=cgrid_path,
                dims=tuple(np.asarray(perm).shape))
        elif geom is not None:
            fig = plt.figure(figsize=(10.0, 7.5))
            ax = fig.add_subplot(111, projection="3d")
            # Colour range from the 1st to 99th percentile, not the extremes.
            #
            # Permeability is lognormal: Norne runs 0.655 to 3,997 mD with a
            # median of 211, so a scale pinned to the maximum puts almost
            # every cell in the bottom eighth of the colour map and the field
            # reads as one flat colour.  Clipping the SCALE keeps the axis in
            # millidarcy -- which is what was asked for -- while letting the
            # variation show.  The colourbar is marked so the clip is not
            # mistaken for the data's range.
            _draw_structure(ax, geom, f, vlo, vhi, cmap, wells, 30, 45,
                            label_sep=0.07)
            sm = cm.ScalarMappable(cmap=plt.get_cmap(cmap),
                                   norm=matplotlib.colors.Normalize(vlo, vhi))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            clipped = fin.size and (float(np.nanmin(f)) < vlo
                                    or float(np.nanmax(f)) > vhi)
            cb.set_label(label + (" (1-99%% range; full %.4g-%.4g)"
                                  % (float(np.nanmin(f)), float(np.nanmax(f))))
                         if clipped else label, fontsize=9)
            ax.set_title(label, fontsize=12, weight="bold", y=1.06)
            fig.tight_layout()
            place_well_labels(fig)
            fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
            plt.close(fig)
        else:
            # Cartesian grid: the voxel block, on the same 1-99 percentile
            # colour range as the corner-point figure.  SPE10's permeability
            # spans seven decades, and on its full range the block is one
            # colour.
            clipped = fin.size and (float(np.nanmin(f)) < vlo
                                    or float(np.nanmax(f)) > vhi)
            # Permeability spanning more than three decades goes on a log
            # colour scale, still in mD.
            logscale = (fname == "perm" and vlo > 0 and vhi / vlo > 1e3)
            full = (" (1-99%% range; full %.4g-%.4g)"
                    % (float(np.nanmin(f)), float(np.nanmax(f)))
                    if clipped else "")
            try:
                viz3d.plot3d_voxels(
                    np.asarray(f, dtype=float), label, path, wells=wells,
                    vmin=vlo, vmax=vhi,
                    cbar_label=label + (", log scale" if logscale else "") + full,
                    cmap=cmap, log=logscale)
            except Exception as exc:
                # A figure on a linear scale beats no Rock section at all.
                log("  ! %s voxel plot failed (%s: %s); drawing it on a "
                    "linear scale" % (fname, type(exc).__name__, exc))
                plt.close("all")
                viz3d.plot3d_voxels(
                    np.asarray(f, dtype=float), label, path, wells=wells,
                    vmin=vlo, vmax=vhi, cbar_label=label + full, cmap=cmap)
        made[fname] = path
    return dict(figs=made, agree=agree,
                perm=np.asarray(perm).ravel(), poro=np.asarray(poro).ravel())


@guarded("insights")
def insights(A, B, rock):
    """Computed commentary on the reservoir and how it performed.

    Everything here is derived from the two runs' own output files; nothing is
    asserted that the data does not support.  Returned as (heading, html)
    pairs.
    """
    out = []
    nx, ny, nz = A["dims"]
    porv = init_field_field_units(A, "PORV")
    pv = None if porv is None else np.nan_to_num(porv, nan=0.0)
    three = float(np.nanmax(np.abs(A["fields"]["gas"]))) > 1e-9

    out.append(("Reservoir", (
        "%d &times; %d &times; %d = %s cells, %s. %s"
        % (nx, ny, nz, format(nx * ny * nz, ",d"),
           ("pore volume %.4g million reservoir bbl" % (pv.sum() / 1e6))
           if pv is not None else "pore volume unknown",
           "Three-phase black oil with a mobile gas phase." if three
           else "Two-phase oil/water waterflood; no free gas."))))

    if rock is not None:
        k = np.asarray(rock["perm"], dtype=float)
        k = k[np.isfinite(k) & (k > 0)]
        phi = np.asarray(rock["poro"], dtype=float)
        phi = phi[np.isfinite(phi)]
        _mu, _sd, skew, kurt = moments(k)
        _lmu, _lsd, lskew, lkurt = moments(np.log(k))
        # Cap the sample: the KS statistic is a max over order statistics and
        # 600k points costs more than it adds.
        d_k, crit = ks_normal(k[:20000])
        d_lk, crit_l = ks_normal(np.log(k[:20000]))
        gauss = ("<b>log-normal</b> &mdash; Gaussian in ln k, which is what a "
                 "sequential-Gaussian or multi-Gaussian generator produces"
                 if np.isfinite(d_lk) and d_lk <= crit_l else
                 "<b>Gaussian in k itself</b>"
                 if np.isfinite(d_k) and d_k <= crit else
                 "<b>non-Gaussian</b> in both k and ln k at the 5% level")
        out.append(("Permeability field", (
            "%s values from %.4g to %.4g mD, mean %.4g, median %.4g. "
            "Skewness %.3g and excess kurtosis %.3g in k; %.3g and %.3g in "
            "ln k. Kolmogorov-Smirnov distance from a fitted normal is %.4f "
            "in k and %.4f in ln k, against a 5%% Lilliefors bound of %.4f "
            "&mdash; so the field is %s."
            % (format(k.size, ",d"), np.nanmin(k), np.nanmax(k), np.nanmean(k),
               float(np.median(k)), skew, kurt, lskew, lkurt,
               d_k, d_lk, crit_l, gauss))))

        k50 = float(np.percentile(k, 50))
        k841 = float(np.percentile(k, 15.9))   # the value 84.1% of cells exceed
        V = (k50 - k841) / k50 if k50 > 0 else float("nan")
        het = ("essentially homogeneous" if V < 0.25 else
               "moderately heterogeneous" if V < 0.5 else
               "heterogeneous" if V < 0.75 else "very heterogeneous")
        out.append(("Heterogeneity", (
            "Dykstra-Parsons coefficient "
            "V = (k<sub>50</sub> &minus; k<sub>84.1</sub>)/k<sub>50</sub> "
            "= %.3f, i.e. <b>%s</b>. Porosity runs %.3f to %.3f, mean %.3f. "
            "Porosity is mapped linearly from permeability, so the two are "
            "perfectly correlated by construction &mdash; this is a synthetic "
            "field, not a geological model, and no porosity-permeability "
            "scatter should be read into it."
            % (V, het, np.nanmin(phi), np.nanmax(phi), np.nanmean(phi)))))

        if rock.get("agree"):
            worst = max(rock["agree"].values())
            out.append(("Same rock in both runs", (
                "Largest relative difference between the two INIT files: %s. %s"
                % (", ".join("%s %.2e" % (kk, vv)
                             for kk, vv in sorted(rock["agree"].items())),
                   "The two simulators were handed identical rock, so what "
                   "follows compares solvers." if worst < 1e-5 else
                   "<b>These disagree &mdash; the comparison below is between "
                   "two different reservoirs, and nothing else in this report "
                   "should be trusted until that is fixed.</b>"))))

    for label, C in (("OPM Flow", A), ("NVRS-CUDA", B)):
        t, vec = summary_of(C)
        if t is None or not t.size:
            continue
        names = sorted({v["well"] for v in vec.values()
                        if v.get("well") and v["well"] != ":+:+:+:+"})
        qo = np.zeros(t.size, dtype=float)
        per_well = {}
        for w in names:
            e = vec.get("WOPR:%s" % w)
            if e is None:
                continue
            v = np.asarray(e["values"], dtype=float)[:t.size]
            per_well[w] = v
            qo[:v.size] += v
        if not per_well or not np.any(qo):
            continue
        # NVRS writes an instantaneous Peaceman diagnostic at t=0, whereas
        # Flow's first summary sample is its first accepted timestep.  That
        # diagnostic is useful for debugging initialisation, but it is not a
        # produced interval: treating it as one invented several million STB
        # in a sparsely reported run and made the displayed peak incomparable.
        active = (t > 0.0) & np.isfinite(t) & np.isfinite(qo)
        if not np.any(active):
            continue
        ta, qoa = t[active], qo[active]

        # Prefer the simulator's accepted cumulative.  Apart from avoiding
        # the synthetic t=0 sample, this is exact across internal substeps;
        # trapezoidal integration of 25-day report samples is not.
        fopt = field_series(vec, "FOPT")
        fopt_last = None
        if fopt is not None:
            n = min(t.size, fopt.size)
            good = (t[:n] > 0.0) & np.isfinite(t[:n]) & np.isfinite(fopt[:n])
            if np.any(good):
                fopt_last = float(fopt[:n][good][-1])
        if fopt_last is not None:
            cum = fopt_last
        else:
            cum = (float(np.trapezoid(qoa, ta))
                   if hasattr(np, "trapezoid") else
                   float(np.trapz(qoa, ta)))
        peak_i = int(np.argmax(qoa))
        peak, peak_t = float(qoa[peak_i]), float(ta[peak_i])
        bw, bv = max(((w, float(np.max(np.asarray(v)[active[:len(v)]])))
                      for w, v in per_well.items()
                      if np.any(active[:len(v)])),
                     key=lambda kv: kv[1])
        ooip = oip_end = None
        if pv is not None:
            ooip = float(np.nansum(pv * np.nan_to_num(C["fields"]["oil"][0],
                                                      nan=0.0)))
            oip_end = float(np.nansum(pv * np.nan_to_num(C["fields"]["oil"][-1],
                                                         nan=0.0)))
        # Recovery on the FIELD's own basis: the reservoir volume of oil that
        # left, over the reservoir volume that started there.  Both come from
        # sum(PORV * So), so the formation volume factor cancels and no PVT is
        # needed -- where cum/ooip silently divided stock-tank by reservoir and
        # reported 32.0% for a field that recovered 41.7%.
        rf = (100.0 * (ooip - oip_end) / ooip)             if (ooip and oip_end is not None) else float("nan")

        # Cross-check the reported rates against the saved saturation field.
        # Oil produced has to equal the drop in oil in place, and the two are
        # computed from completely independent parts of the output: the
        # summary vectors on one side, PORV and So on the other.  A simulator
        # whose rates and fields disagree is telling two different stories and
        # every curve above is suspect.  This is what caught the accumulation
        # term being in ft^3 while the fluxes were in bbl/day.
        # The two figures are in DIFFERENT VOLUME BASES and always were.
        #
        # sum(PORV * So) is a RESERVOIR volume; the rate-based cumulative is
        # stock-tank.  They differ by the oil formation volume factor, so
        # requiring them to be equal was requiring Bo = 1 -- and the report
        # then called OPM's own material balance a contradiction, at a ratio
        # of 0.77, which is 1/1.30.  Flow was right and the check was wrong.
        #
        # Without the PVT tables here the honest form is the implied Bo: the
        # reservoir volume that left divided by the stock-tank volume
        # produced.  A physical answer is roughly 1.1 to 1.8 for a live oil;
        # outside that the two really are describing different runs, and that
        # is worth saying.  Inside it, quote it and move on.
        cross = ""
        if ooip is not None and oip_end is not None:
            from_field = ooip - oip_end
            if abs(from_field) > 1e-30 and cum > 0:
                bo_implied = from_field / cum
                # A band on this number is the wrong instrument, and the
                # previous two attempts at one were both wrong.
                #
                # sum(PORV * So) is a RESERVOIR volume and production is
                # stock-tank, so their ratio looks like a formation volume
                # factor -- but for a LIVE oil it is not one.  Below the
                # bubble point solution gas evolves and the oil shrinks, so
                # sum(PORV * So) falls with no oil produced at all.  The
                # ratio therefore mixes Bo with shrinkage, and a value above
                # Bo can mean the rates are too low OR that more gas came out
                # of solution.  One number cannot separate them, and calling
                # 1.786 a 27% shortfall asserted one reading of two.
                #
                # So: flag only what is arithmetically impossible -- a ratio
                # below 1 says more stock-tank oil was produced than the
                # reservoir volume that left, which no PVT permits.  The real
                # balance is stock-tank on both sides, and NVRS now reports
                # it directly as the [oil] line in its own log.
                # Restart fields and summary totals are single precision, so
                # do not turn a shared one-percent rounding gap into a warning.
                ok = bo_implied >= 0.98
                cross = (" The saturation field says <b>%s reservoir bbl</b> "
                         "of oil left (oil in place fell from %s to %s), which "
                         "against the <b>%s STB</b> the rates report implies "
                         "B<sub>o</sub> = <b>%.3f</b>. %s"
                         % (human(from_field), human(ooip), human(oip_end),
                            human(cum), bo_implied,
                            "For a live oil this ratio mixes B<sub>o</sub> "
                            "with the shrinkage that solution gas causes below "
                            "the bubble point, so it is indicative rather than "
                            "a balance; compare it against the other "
                            "simulator's rather than against 1."
                            if ok else
                            "<b>Materially below 1: more stock-tank "
                            "oil was produced than the reservoir volume that "
                            "left the reservoir.</b>"))
            if False:
                ratio = cum / from_field
                cross = (" The saturation field says <b>%s STB</b> of oil left "
                         "the reservoir (oil in place fell from %s to %s), "
                         "which is %.2fx the rate-based figure.%s"
                         % (human(from_field), human(ooip), human(oip_end),
                            ratio,
                            "" if 0.9 <= ratio <= 1.1 else
                            " <b>These must agree; they do not, so the rate "
                            "curves and the field panels are describing "
                            "different runs.</b>"))
        # A recovery factor above 100% is not a good result, it is a
        # contradiction: the well rates and the saved saturation field are
        # describing different amounts of oil.  Say so rather than printing it
        # as a headline number.
        flag = ""
        if np.isfinite(rf) and rf > 100.0:
            flag = (" <b>That exceeds 100%, which is impossible: the reported "
                    "well rates and the saved saturation fields disagree about "
                    "how much oil left the reservoir. Treat the rate curves "
                    "and the field panels as measuring different things until "
                    "that is resolved.</b>")
        out.append(("Recovery &mdash; %s" % label, (
            "Peak field oil rate <b>%s STB/day</b> at t = %g days; best single "
            "well %s at %s STB/day. Cumulative oil over %g days is "
            "<b>%s STB</b>. Oil in place at t = 0 is %s reservoir bbl, and "
            "the recovery factor -- the reservoir volume that left over the "
            "reservoir volume that started there, so B<sub>o</sub> cancels -- "
            "is <b>%s</b>.%s"
            % (human(peak), peak_t, bw, human(bv), t[-1], human(cum),
               human(ooip) if ooip else "unknown",
               ("%.1f%%" % rf) if np.isfinite(rf) else "unknown",
               flag + cross))))
    return out


# Each phase has a SHORT CODE, a fixed hue, and a fixed position in the legend.
# The code is what makes the two bars readable side by side: the linear solve is
# LS on Flow's bar and LS1..LS4 on NVRS's, so the eye can match the part being
# compared without tracing shades of green between two rows that decompose
# differently.  Nothing else in the chart is green or begins with LS.
RUNTIME_PHASES = [
    ("setup",     "SU",  "Setup / deck read",        "#607d8b"),
    ("assembly",  "AS",  "Assembly (Jacobian)",      "#1f77b4"),
    ("linear",    "LS",  "Linear solve (total)",     "#2e7d32"),
    ("amg_setup", "LS1", "AMG setup (HYPRE)",        "#1b5e20"),
    ("amg_solve", "LS2", "AMG V-cycles (FP32)",      "#66bb6a"),
    ("stage2",    "LS3", "Block-Jacobi stage 2",     "#a5d6a7"),
    ("krylov",    "LS4", "Krylov / SpMV (FP64)",     "#00897b"),
    ("props",     "PR",  "Fluid properties",         "#c9a227"),
    ("prepost",   "PP",  "Pre / post step",          "#8e5ea2"),
    ("output",    "OU",  "Output write",             "#e07b39"),
    ("rest",      "ST",  "Start-up / unaccounted",   "#cfcfcf"),
]
RT_COLOUR = {k: c for k, _s, _n, c in RUNTIME_PHASES}
RT_CODE = {k: s for k, s, _n, c in RUNTIME_PHASES}
RT_NAME = {k: "%s - %s" % (s, n) for k, s, n, _c in RUNTIME_PHASES}
# Every key whose time belongs to the linear solve, for the LS-total bracket.
RT_LINEAR = ("linear", "amg_setup", "amg_solve", "stage2", "krylov")


def _phase_split(r):
    """(ordered [(key, seconds)], accounted, total) for one run.

    A side that reported its CPR internals gets its linear bar subdivided; one
    that did not keeps a single linear segment.  Both stay in the LS family.
    """
    total = float(r["seconds"])
    bd = dict(r.get("breakdown") or {})

    # Two vocabularies for the same phases.
    #
    # Flow's log is parsed into setup/assembly/linear/...; NVRS writes its own
    # meta.json with assembly_seconds/linear_seconds/amg_setup_seconds/...
    # Nothing here matched the latter, so every NVRS run drew a single
    # unlabelled bar: no setup, no assembly, no linear solve, no AMG split --
    # on the one chart whose entire purpose is showing where the time went.
    # Normalise rather than rename at the source, so timing.json files that
    # already exist read correctly without re-running anything.
    for k in list(bd):
        if k.endswith("_seconds"):
            bd.setdefault(k[: -len("_seconds")], bd[k])
    if bd.get("amg_solve") or bd.get("amg_setup"):
        bd["krylov"] = max(bd.get("linear", 0.0) - bd.get("amg_setup", 0.0)
                           - bd.get("amg_solve", 0.0) - bd.get("stage2", 0.0),
                           0.0)
        order = ["setup", "assembly", "amg_setup", "amg_solve", "stage2",
                 "krylov", "props", "prepost", "output"]
    else:
        order = ["setup", "assembly", "linear", "props", "prepost", "output"]
    parts = [(k, float(bd.get(k) or 0.0)) for k in order]
    parts = [(k, v) for k, v in parts if v > 0]
    return parts, sum(v for _k, v in parts), total


def _seg_label(ax, x0, width, y, code, pct, dark_text=False, span=None):
    """Code above, share below, inside the segment -- as much as will fit.

    Whether it fits is a question about the segment's width on the AXIS, not
    its share of its own bar: a quarter of a 46 s bar on a 500 s axis is two
    percent of the axis, and labelling it wrote text across its neighbours.
    """
    col = "#333333" if dark_text else "white"
    if width <= 0:
        return
    if span:
        frac = width / float(span)
        if frac < 0.035:
            return
        if frac < 0.07:
            ax.text(x0 + width / 2, y, code, va="center", ha="center",
                    fontsize=8.5, fontweight="bold", color=col)
            return
    if pct >= 8.0:
        ax.text(x0 + width / 2, y - 0.10, code, va="center", ha="center",
                fontsize=9.5, fontweight="bold", color=col)
        ax.text(x0 + width / 2, y + 0.13, "%.0f%%" % pct, va="center",
                ha="center", fontsize=8.5, color=col)
    elif pct >= 3.5:
        ax.text(x0 + width / 2, y, code, va="center", ha="center",
                fontsize=8.5, fontweight="bold", color=col)


@guarded("runtime chart")
def runtime_chart(outdir, case, timing):
    """Wall clock by phase, in two panels.

    TOP -- absolute seconds, so the comparison is of real elapsed time, with
    whatever a simulator did not account for drawn in grey rather than silently
    dropped.  On this benchmark that grey is most of Flow's bar: MPI start-up
    on a 4,800-cell model swamps the time it spends simulating.  Hiding it
    would flatter NVRS.

    BOTTOM -- the same phases as a share of each simulator's OWN accounted
    time, which is the panel to read for "where does the work go".  The bars
    decompose differently and one is dominated by start-up, so raw percentages
    of wall clock say more about MPI than about either solver.

    An LS bracket spans the linear-solve group on each bar with its total, so
    "how much of this run was the linear solve" is one number on both sides even
    though NVRS splits it four ways and Flow does not split it at all.
    """
    rows = [r for r in timing if r.get("seconds")]
    if len(rows) < 2:
        return None

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(13, 2.2 + 1.55 * len(rows)),
        gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.42))
    seen, totals = [], []
    # The axis span the labels are judged against (set_xlim below).
    span_top = max(_phase_split(r)[2] for r in rows) * 1.30

    # ---- top: absolute ----------------------------------------------------
    for y, r in enumerate(rows):
        parts, accounted, total = _phase_split(r)
        left = ls_lo = ls_hi = 0.0
        for key, v in parts:
            ax.barh(y, v, left=left, height=0.5, color=RT_COLOUR[key],
                    edgecolor="white", linewidth=0.8,
                    label=RT_NAME[key] if key not in seen else None)
            if key not in seen:
                seen.append(key)
            if key in RT_LINEAR:
                if ls_hi == 0.0:
                    ls_lo = left
                ls_hi = left + v
            _seg_label(ax, left, v, y, RT_CODE[key], 100.0 * v / total,
                       span=span_top)
            left += v
        rest = total - accounted
        if rest > 1e-9:
            ax.barh(y, rest, left=left, height=0.5, color=RT_COLOUR["rest"],
                    edgecolor="white", linewidth=0.8,
                    label=RT_NAME["rest"] if "rest" not in seen else None)
            if "rest" not in seen:
                seen.append("rest")
            _seg_label(ax, left, rest, y, "ST", 100.0 * rest / total,
                       dark_text=True, span=span_top)
        totals.append(total)

    # Any number of runs, not two.  The comparison that matters is not just
    # OPM against NVRS but NVRS against itself with a different stage-one
    # multigrid -- AmgX or HYPRE -- and "fast = 1 - slow" could only ever
    # express a pair.
    slow = int(np.argmax(totals))
    fast = int(np.argmin(totals))
    speedup = totals[slow] / totals[fast] if totals[fast] > 0 else float("nan")

    # Name the winner on its own bar, in red.  The statement in the gap below
    # says how much slower the other one is; this says which one to look at
    # first, and a reader should not have to work that out from two bar lengths.
    # One annotation per bar, to its right, on its own line: the total, then
    # how it stands against the fastest.  Nothing is placed over another bar,
    # which is where the old centred FASTEST banner and the boxed ratio kept
    # landing once there were more than two runs.
    pad = span_top * 0.008
    for y, r in enumerate(rows):
        ax.text(totals[y] + pad, y, "%.2f s" % totals[y], va="center",
                ha="left", fontsize=10, fontweight="bold")
        if not np.isfinite(speedup) or totals[fast] <= 0:
            continue
        if y == fast and speedup > 1.0:
            note, colr = "FASTEST (%.2fx vs slowest)" % speedup, "#b00020"
        elif y != fast:
            note, colr = ("%.2fx slower than fastest"
                          % (totals[y] / totals[fast])), "#b00020"
        else:
            continue
        ax.text(totals[y] + pad, y + 0.34, note, va="center", ha="left",
                fontsize=8.5, fontweight="bold", color=colr)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["%s\n%s" % (r["name"], r.get("solver", ""))
                        for r in rows], fontsize=8)
    ax.set_xlabel("Run time (seconds)")
    ax.set_title("%s - total wall clock, including start-up" % case,
                 fontsize=12)
    ax.set_xlim(0, span_top)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # ---- bottom: share of each run's own accounted time -------------------
    for y, r in enumerate(rows):
        parts, accounted, total = _phase_split(r)
        if accounted <= 0:
            continue
        left = ls_lo = ls_tot = 0.0
        ls_started = False
        for key, v in parts:
            pct = 100.0 * v / accounted
            axn.barh(y, pct, left=left, height=0.5, color=RT_COLOUR[key],
                     edgecolor="white", linewidth=0.8)
            if key in RT_LINEAR:
                if not ls_started:
                    ls_lo, ls_started = left, True
                ls_tot += pct
            _seg_label(axn, left, pct, y, RT_CODE[key], pct, span=118.0)
            left += pct
        # The bracket: one LS number per bar, however many pieces it came in.
        if ls_started:
            yb = y - 0.33
            axn.plot([ls_lo, ls_lo + ls_tot], [yb, yb], color="#1b5e20",
                     lw=1.8, solid_capstyle="butt", clip_on=False)
            for xe in (ls_lo, ls_lo + ls_tot):
                axn.plot([xe, xe], [yb, yb + 0.05], color="#1b5e20", lw=1.8,
                         clip_on=False)
            axn.text(ls_lo + ls_tot / 2, yb - 0.06,
                     "LS total %.0f%%" % ls_tot, va="bottom", ha="center",
                     fontsize=9, fontweight="bold", color="#1b5e20",
                     clip_on=False)
        axn.text(101.5, y, "%.2f s accounted" % accounted, va="center",
                 fontsize=9)
    axn.set_yticks(range(len(rows)))
    axn.set_yticklabels([r["name"] for r in rows], fontsize=9)
    axn.set_xlabel("Share of that simulator's accounted time (%)")
    axn.set_title("Where the work goes, start-up excluded "
                  "(LS = linear solve, same meaning on both bars)", fontsize=11)
    axn.set_xlim(0, 118)
    axn.grid(axis="x", alpha=0.3)
    axn.invert_yaxis()
    # Headroom so the LS bracket and its label sit inside the axes.
    lo, hi = axn.get_ylim()
    axn.set_ylim(lo, hi - 0.30)

    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, fontsize=8, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), frameon=False)
    path = os.path.join(outdir, "runtime_%s.png" % case)
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# Below roughly this many cells per GPU there is not enough work in a report
# step to cover what each extra rank costs: the AMG coarsening's per-level
# neighbour exchanges and global reductions are latency-bound and barely shrink
# as the grid is split, and the coarse levels of the V-cycle end up holding a
# few hundred rows per rank.  The figure is a rule of thumb, not a threshold the
# code enforces; it is here so a reader can tell "this decomposition was never
# going to pay" apart from "the parallel implementation is broken", which look
# identical in a wall-clock column.
DECOMP_FLOOR = 500_000


def scaling_verdict(table, ncells):
    """Say *why* the scaling curve looks the way it does, from the numbers.

    Efficiency alone cannot distinguish a solver that parallelises badly from a
    problem that was too small to divide, and those call for opposite responses
    -- fix the code, or stop adding GPUs.  The iteration counts settle it: they
    are a property of the algorithm, so if they hold steady across rank counts
    the preconditioner is rank-invariant and every second lost is overhead.
    """
    if len(table) < 2:
        return None
    lo, hi = table[0], table[-1]
    if not (lo["ranks"] and hi["ranks"] > lo["ranks"]):
        return None

    bits = []
    cpg = hi.get("cells_per_gpu") or 0
    scaled = hi["eff"] >= 60.0

    if cpg and ncells:
        bits.append(
            "%s cells over %d GPUs is %s cells per GPU."
            % ("{:,}".format(ncells), hi["ranks"], "{:,}".format(cpg)))
        if cpg < DECOMP_FLOOR and not scaled:
            bits.append(
                "That is below the ~%s cells per GPU this solver needs before "
                "a decomposition pays for itself, so the flat-to-negative "
                "curve above is the expected result for a deck this size "
                "rather than a defect &mdash; run it on one GPU."
                % "{:,}".format(DECOMP_FLOOR))

    # Did the preconditioner change, or did the same algorithm just cost more?
    if lo["krylov"] and hi["krylov"]:
        growth = hi["krylov"] / float(lo["krylov"])
        if growth > 1.15:
            bits.append(
                "Krylov iterations rose %.0f%% (%d to %d), so BoomerAMG is "
                "coarsening differently on %d ranks and the preconditioner "
                "itself got weaker &mdash; that is an algorithmic cost, not "
                "overhead."
                % (100 * (growth - 1), lo["krylov"], hi["krylov"], hi["ranks"]))
        else:
            bits.append(
                "Newton and Krylov counts held to within %.0f%% across all "
                "rank counts, so the preconditioner is rank-invariant%s."
                % (100 * abs(growth - 1),
                   " and the decomposition is doing its job" if scaled
                   else ": the same iterations simply cost more"))

    # Which phase absorbed the loss?
    if lo["amg_setup"] and hi["amg_setup"]:
        s = hi["amg_setup"] / lo["amg_setup"]
        share = 100.0 * hi["amg_setup"] / hi["seconds"] if hi["seconds"] else 0
        # Only claim the rebuild count was held fixed if it actually was; with
        # reuse enabled the guard can fire a different number of times at each
        # rank count, and then the growth is not a clean per-setup comparison.
        same = (lo["setups"] and hi["setups"]
                and abs(hi["setups"] - lo["setups"]) <= 0.05 * lo["setups"])
        if s > 1.3:
            bits.append(
                "AMG setup grew %.1fx (%.1f s to %.1f s%s) and is now %.0f%% "
                "of the run. Parallel coarsening is a chain of small neighbour "
                "exchanges and global reductions per level; that cost is "
                "nearly independent of how many cells each rank holds, so "
                "splitting the grid adds it without dividing it."
                % (s, lo["amg_setup"], hi["amg_setup"],
                   " for the same number of rebuilds" if same else "", share))
            if not hi.get("reuses"):
                bits.append(
                    "The hierarchy is being rebuilt on every Newton iteration "
                    "here (<code>AMGREUSE 0</code>). Reusing it across "
                    "iterations removes most of that cost on one GPU and most "
                    "of the scaling penalty on several.")

    return " ".join(bits) if bits else None


@guarded("scaling")
def scaling_study(rundir, case, wells):
    """NVRS on 1..N GPUs: wall clock, speed-up against ideal, and efficiency.

    Also checks that the answer did not change.  A parallel run that is fast
    but wrong is worse than a slow one, and a K-slab decomposition has exactly
    the failure mode where it would not show up in the timings: a halo exchange
    that drops a layer still converges, still reports plausible rates, and only
    differs in the field.  So every rank count is compared against the 1-GPU
    result cell by cell.
    """
    path = os.path.join(rundir, "scaling.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            rows = (json.load(fh) or {}).get("nvrs") or []
    except (OSError, ValueError):
        return None
    rows = [r for r in rows if r.get("seconds")]
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r["ranks"])

    n = np.array([r["ranks"] for r in rows], dtype=float)
    t = np.array([r["seconds"] for r in rows], dtype=float)
    speed = t[0] / t
    ideal = n / n[0]
    eff = 100.0 * speed / ideal

    # --- did the answer survive being split up? ---------------------------
    invariance = []
    ref = None
    for r in rows:
        base = None
        for eg in sorted(glob.glob(os.path.join(r.get("outdir", ""), "**",
                                                "*.EGRID"), recursive=True)):
            base = os.path.splitext(eg)[0]
            break
        if not base:
            continue
        try:
            c = load_case(base)
        except Exception:
            continue
        if ref is None:
            ref = c
            invariance.append((r["ranks"], 0.0, 0.0))
            continue
        k = min(ref["fields"]["pressure"].shape[0],
                c["fields"]["pressure"].shape[0]) - 1
        if k < 0:
            continue
        dp = float(np.nanmax(np.abs(ref["fields"]["pressure"][k]
                                    - c["fields"]["pressure"][k])))
        dsw = float(np.nanmax(np.abs(ref["fields"]["water"][k]
                                     - c["fields"]["water"][k])))
        invariance.append((r["ranks"], dp, dsw))

    # Each scaling run wrote its own meta.json.  "More GPUs made it slower" has
    # several possible causes that a wall-clock number cannot separate:
    #
    #   * a global phase got more expensive.  BoomerAMG's SETUP is rebuilt every
    #     Newton iteration and its parallel coarsening is communication-bound,
    #     so it can cost more on N ranks than the solve saves.
    #   * the preconditioner CHANGED.  BoomerAMG coarsens differently on N
    #     ranks, so the Krylov count can rise: more iterations, each cheaper,
    #     net slower.
    #   * the ranks did not actually get separate GPUs, in which case they
    #     serialise and nothing scales.
    #
    # The first two are distinguishable from the phase times and iteration
    # counts, which are already in meta.json, so read them rather than guess.
    for r in rows:
        r["meta"] = {}
        for mp in sorted(glob.glob(os.path.join(r.get("outdir", ""), "**",
                                                "meta.json"), recursive=True)):
            try:
                with open(mp) as fh:
                    r["meta"] = json.load(fh) or {}
            except (OSError, ValueError):
                r["meta"] = {}
            break

    def mget(r, key, default=0.0):
        v = r["meta"].get(key)
        return float(v) if isinstance(v, (int, float)) else default

    def isum(r, key):
        v = r["meta"].get(key)
        return int(sum(v)) if isinstance(v, list) and v else 0

    have_meta = any(r["meta"] for r in rows)

    ncol = 3 if have_meta else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.6 * ncol, 4.0))
    a1, a2 = axes[0], axes[1]
    a1.plot(n, t, "o-", color="#76b900", lw=2.2, markersize=7)
    for xi, yi in zip(n, t):
        a1.annotate("%.1f s" % yi, (xi, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    a1.set_xlabel("GPUs")
    a1.set_ylabel("Wall clock (s)")
    a1.set_title("%s - NVRS-CUDA wall clock" % case, fontsize=11)
    a1.set_xscale("log", base=2)
    a1.set_xticks(n)
    a1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a1.grid(alpha=0.3)

    a2.plot(n, ideal, "--", color="#999999", lw=1.6, label="ideal (linear)")
    a2.plot(n, speed, "o-", color="#76b900", lw=2.2, markersize=7,
            label="measured")
    for xi, si, ei in zip(n, speed, eff):
        a2.annotate("%.2fx\n%.0f%% eff" % (si, ei), (xi, si),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=9)
    a2.set_xlabel("GPUs")
    a2.set_ylabel("Speed-up vs 1 GPU")
    a2.set_title("Strong scaling (parallel efficiency)", fontsize=11)
    a2.set_xscale("log", base=2)
    a2.set_xticks(n)
    a2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a2.grid(alpha=0.3)
    a2.legend(fontsize=9)

    # --- third panel: where the time went at each rank count --------------
    if have_meta:
        a3 = axes[2]
        PH = [("amg_setup_seconds", "AMG setup", "#1b5e20"),
              ("amg_solve_seconds", "AMG V-cycles", "#66bb6a"),
              ("stage2_seconds", "Block-Jacobi", "#a5d6a7"),
              ("krylov", "Krylov / SpMV", "#00897b"),
              ("assembly_seconds", "Assembly", "#1f77b4"),
              ("other", "Other", "#cfcfcf")]
        x = np.arange(len(rows), dtype=float)
        bottoms = np.zeros(len(rows))
        for key, name, colour in PH:
            if key == "krylov":
                vals = np.array([max(mget(r, "linear_seconds")
                                     - mget(r, "amg_setup_seconds")
                                     - mget(r, "amg_solve_seconds")
                                     - mget(r, "stage2_seconds"), 0.0)
                                 for r in rows])
            elif key == "other":
                vals = np.array([max(mget(r, "wall_seconds")
                                     - mget(r, "linear_seconds")
                                     - mget(r, "assembly_seconds"), 0.0)
                                 for r in rows])
            else:
                vals = np.array([mget(r, key) for r in rows])
            if not np.any(vals > 0):
                continue
            a3.bar(x, vals, bottom=bottoms, width=0.6, color=colour,
                   edgecolor="white", linewidth=0.7, label=name)
            bottoms += vals
        a3.set_xticks(x)
        a3.set_xticklabels(["%d" % r["ranks"] for r in rows])
        a3.set_xlabel("GPUs")
        # The left panels time the process from outside and so include MPI
        # start-up; this one is the simulator's own accounting.  The two totals
        # will not match exactly and should not be read as if they did.
        a3.set_ylabel("Seconds (simulator-internal)")
        a3.set_title("Where the time goes at each rank count", fontsize=11)
        a3.grid(axis="y", alpha=0.3)
        a3.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(rundir, "scaling_%s.png" % case)
    fig.savefig(out, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

    ncells = 0
    for r in rows:
        m = r["meta"]
        if all(isinstance(m.get(k), (int, float)) for k in ("nx", "ny", "nz")):
            ncells = int(m["nx"]) * int(m["ny"]) * int(m["nz"])
            break

    table = [dict(ranks=int(r["ranks"]), seconds=float(r["seconds"]),
                  speedup=float(sp), eff=float(e),
                  cells_per_gpu=(ncells // int(r["ranks"])) if ncells else 0,
                  amg_setup=mget(r, "amg_setup_seconds"),
                  amg_solve=mget(r, "amg_solve_seconds"),
                  linear=mget(r, "linear_seconds"),
                  newton=isum(r, "newton_iters"),
                  krylov=isum(r, "linear_iters"),
                  setups=int(mget(r, "amg_setups")),
                  reuses=int(mget(r, "amg_reuses")))
             for r, sp, e in zip(rows, speed, eff)]

    return dict(path=out, table=table, have_meta=have_meta,
                ncells=ncells, invariance=invariance,
                verdict=scaling_verdict(table, ncells))


def read_timing(rundir):
    p = os.path.join(rundir, "timing.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}


# ===========================================================================
# Report
# ===========================================================================
def html_head(title):
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
 :root { color-scheme: light dark; }
 body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif; margin: 0 auto; max-width: 1800px;
        padding: 1.5rem; line-height: 1.5; }
 h1 { border-bottom: 3px solid #76b900; padding-bottom: .3rem; }
 h2 { border-bottom: 1px solid #ccc; padding-bottom: .2rem; margin-top: 2.2rem; }
 h3 { margin-top: 1.6rem; color: #555; }
 table { border-collapse: collapse; margin: 1rem 0; font-size: .88rem; }
 th, td { border: 1px solid #bbb; padding: .3rem .6rem; text-align: right; }
 th { background: rgba(118,185,0,.16); text-align: left; }
 td:first-child, th:first-child { text-align: left; }
 figure { margin: 0 0 .55rem 0; }
 figure img, figure video { width: 100%; height: auto; border: 1px solid #ddd;
        border-radius: 4px; background: #fff; }
 figcaption { font-size: .85rem; color: #666; padding-top: .3rem; }
 .grid { display: grid; grid-template-columns: repeat(auto-fit,
         minmax(430px, 1fr)); gap: .55rem; }
 .wide { display: block; }
 .note { background: rgba(118,185,0,.09); border-left: 4px solid #76b900;
         padding: .7rem 1rem; margin: 1rem 0; font-size: .92rem; }
 .warn { background: rgba(220,140,0,.10); border-left: 4px solid #d88c00; }
 code { background: rgba(127,127,127,.16); padding: .1rem .3rem;
        border-radius: 3px; }
 dl { margin: 1rem 0; }
 dt { font-weight: 600; margin-top: .8rem; color: #4a7c00; }
 dd { margin: .2rem 0 0 0; }
 @media (prefers-color-scheme: dark) { dt { color: #9ede3a; } }
 @media (prefers-color-scheme: dark) {
   body { background: #16181c; color: #e6e6e6; }
   th, td { border-color: #444; }
   h2 { border-color: #444; }
   figure img, figure video { border-color: #333; }
 }
</style></head><body>
""".replace("__TITLE__", title)


def _fits(paths, limit):
    """Total on-disk size of `paths`, and whether it fits the budget.

    base64 adds about a third, so a file only fits if 4/3 of it does.
    """
    tot = 0
    for p in paths:
        try:
            tot += os.path.getsize(p)
        except OSError:
            pass
    return tot, (tot * 4 // 3) <= limit


def write_report(outpath, sections, generated, limit=EMBED_BUDGET):
    budget = {"used": 0, "skipped": 0, "limit": limit}

    # Decide up front whether the animations can be afforded.  They are by far
    # the largest thing on the page and the least load-bearing -- every frame
    # they contain is also a static panel or a step in the table -- so if the
    # budget is tight they are the first thing to go, rather than letting them
    # consume it and starve the figures that carry the argument.
    fixed = []
    for sec in sections:
        for key in ("runtime_fig",):
            if sec.get(key):
                fixed.append(sec[key])
        for k in ("wells", "panels"):
            v = sec.get(k) or []
            fixed += [p[0] if isinstance(p, (tuple, list)) else p for p in v]
        if sec.get("balance"):
            fixed.append(sec["balance"]["path"])
        if sec.get("rock"):
            fixed += list((sec["rock"].get("figs") or {}).values())
    fixed_bytes, _ok = _fits(fixed, limit)
    spare = max(limit - fixed_bytes * 4 // 3, 0)
    for sec in sections:
        movies = sec.get("movies") or []
        if not movies:
            continue
        # Prefer the MP4 and drop the GIF rather than embed both.
        #
        # The two are the same animation in two containers. A GIF has 256
        # colours, no interframe compression and one palette for the whole
        # sequence, so a full-resolution one runs to about 20 MB; the same
        # frames as H.264 are perhaps a tenth of that, and make_movie writes
        # the MP4 at full resolution because its scale argument only ever
        # applied to the GIF. Choosing the MP4 is how the page stays both
        # crisp and small, and browsers have played H.264 in a <video> tag for
        # fifteen years.
        mp4 = [m for m in movies if m.lower().endswith(".mp4")]
        if mp4:
            if len(mp4) != len(movies):
                log("  animation: using the MP4 and leaving the GIF on disk "
                    "(same frames, a fraction of the size)")
            movies = mp4
            sec["movies"] = movies

        _mb, _fits_ok = _fits(movies, spare)
        if _fits_ok:
            log("  animation %.1f MB, fits" % (_mb / 1e6))
        elif mp4:
            # An MP4 over budget is not worth degrading: it is already the
            # efficient encoding, and the overshoot is small enough to prefer a
            # slightly larger page to a worse animation.
            log("  animation %.1f MB against %.1f MB left after the figures; "
                "kept at full size (--max-mb to raise the budget)"
                % (_mb / 1e6, spare / 1e6))
        else:
            # No ffmpeg on the machine that made this, so a GIF is all there
            # is, and the budget and the resolution genuinely conflict. The
            # budget wins, because it was asked for and because a page too
            # large to open helps nobody, but never below half size.
            frames = sec.get("_movie_frames") or []
            remade = False
            if frames:
                for scale in (0.85, 0.7, 0.6, 0.5):
                    try:
                        new_m = viz3d.make_movie(
                            frames, os.path.splitext(movies[0])[0], fps=2,
                            scale=scale) or []
                    except Exception as exc:
                        log("  ! animation re-encode failed: %s" % exc)
                        break
                    new_m = [m for m in new_m if not m.lower().endswith(".mp4")] or new_m
                    _mb, fits = _fits(new_m, spare)
                    log("  animation, GIF only, re-encoded at %.0f%% -> %.1f MB%s"
                        % (100 * scale, _mb / 1e6, "" if fits else ", still over"))
                    if fits:
                        sec["movies"] = new_m
                        movies = new_m
                        remade = True
                        break
            if not remade:
                log("  animation %.1f MB and cannot be made to fit %.1f MB; "
                    "kept anyway rather than dropped. Install ffmpeg and it "
                    "becomes an MP4 a tenth the size."
                    % (_mb / 1e6, spare / 1e6))

    def img(path, caption, wide=False, force=False):
        if not path or not os.path.exists(path):
            return ""
        src = data_uri(path, budget, force=force) or os.path.basename(path)
        cls = ' class="wide"' if wide else ""
        return ('<figure%s><img src="%s" alt="%s" loading="lazy">'
                '<figcaption>%s</figcaption></figure>'
                % (cls, src, caption, caption))

    H = [html_head("OPM Flow vs NVRS-CUDA")]
    A = H.append
    A("<h1>OPM Flow vs NVRS-CUDA</h1>")
    A("<p>Head-to-head verification of the mixed-precision CUDA reservoir "
      "simulator against OPM Flow on identical decks. Generated %s.</p>"
      % generated)
    A('<div class="note">Both simulators are read through the same ECLIPSE '
      'reader (<code>scripts/ecl.py</code>), from the EGRID/UNRST/UNSMRY each '
      'writes. Oil saturation is formed as '
      '<code>So = 1 &minus; Sw &minus; Sg</code> on both sides. The two decks '
      'are emitted from one parameter set by '
      '<code>benchmark/make_decks.py</code>, which documents exactly which '
      'terms were matched and which could not be.</div>')

    for sec in sections:
        A("<h2>%s</h2>" % sec["title"])
        if sec.get("blurb"):
            A('<div class="note%s">%s</div>'
              % (" warn" if sec.get("warn") else "", sec["blurb"]))

        if sec.get("notes"):
            A("<h3>Reservoir and insights</h3>")
            A("<dl>")
            for head, body in sec["notes"]:
                A("<dt>%s</dt><dd>%s</dd>" % (head, body))
            A("</dl>")

        if sec.get("rock") and sec["rock"].get("figs"):
            A("<h3>Rock</h3>")
            A('<div class="note">Static properties, read from the INIT file. '
              'Both decks are generated from one permeability field, so these '
              'are the same on both sides &mdash; the agreement figure above '
              'says by how much.</div>')
            A('<div class="grid">')
            A(img(sec["rock"]["figs"].get("perm"),
                  "Permeability, mD"))
            A(img(sec["rock"]["figs"].get("poro"), "Porosity"))
            A("</div>")

        if sec.get("timing"):
            A("<h3>Wall clock</h3>")
            A("<table><tr><th>Run</th><th>Wall clock (s)</th>"
              "<th>Linear solve (s)</th><th>of which AMG (s)</th>"
              "<th>Algorithm</th><th>Hardware</th><th>Ranks</th></tr>")
            for r in sec["timing"]:
                bd = dict(r.get("breakdown") or {})
                # NVRS names them assembly_seconds/linear_seconds/...; Flow's
                # log parses to assembly/linear/....  The chart normalises
                # these in _phase_split and the TABLE did not, so both NVRS
                # rows showed "-" for the linear solve and the AMG split while
                # the chart beside them drew the segments.
                for _k in list(bd):
                    if _k.endswith("_seconds"):
                        bd.setdefault(_k[: -len("_seconds")], bd[_k])
                lin = bd.get("linear")
                amg = (bd.get("amg_setup") or 0.0) + (bd.get("amg_solve") or 0.0)
                A("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                  "<td>%s</td><td>%s</td><td>%s</td></tr>"
                  % (r.get("name", "-"),
                     ("%.2f" % r["seconds"]) if r.get("seconds") else "-",
                     ("%.2f (%.0f%%)" % (lin, 100.0 * lin / r["seconds"]))
                     if lin and r.get("seconds") else "-",
                     ("%.2f (%.0f%% of solve)" % (amg, 100.0 * amg / lin))
                     if amg and lin else "-",
                     r.get("solver", "-"),
                     r.get("hardware", "-"), r.get("ranks", "-")))
            A("</table>")
            # Reuse changes what "AMG setup" in the row above is counting, so
            # say so next to the number rather than leaving a reader to wonder
            # why a rebuilt-every-iteration preconditioner looks so cheap.
            for r in sec["timing"]:
                bd = r.get("breakdown") or {}
                reuses = bd.get("amg_reuses") or 0
                setups = bd.get("amg_setups") or 0
                if not (reuses and setups):
                    continue
                A('<div class="note">%s rebuilt its AMG hierarchy %d times '
                  'across %d Newton iterations, reusing it for the other %d. '
                  'The hierarchy is abandoned early whenever the reused one '
                  'starts costing noticeably more Krylov iterations than the '
                  'fresh one did, so this trades setup time for iterations '
                  'only while that trade is winning. Flow does the same thing '
                  'for its own CPR setup.</div>'
                  % (r.get("name", "NVRS-CUDA"), int(setups),
                     int(setups) + int(reuses), int(reuses)))
            if sec.get("runtime_fig"):
                A(img(sec["runtime_fig"],
                      "Wall clock by phase; the speed-up is marked on the "
                      "slower run"))
            if sec.get("speedup"):
                A("<p><strong>%s</strong></p>" % sec["speedup"])
            if sec.get("backend_table") and \
                    len(sec["backend_table"]["cols"]) > 1:
                A("<h3>Which backend is best: wall clock and well accuracy</h3>")
                A(render_backend_verdict(sec["backend_table"]))

        # --- Nsight Systems / Nsight Compute -------------------------------
        #
        # Optional: profile_case.sh writes these and profile_plots.py draws
        # them.  Absent when nobody has profiled, and the report simply does
        # not mention it rather than showing an empty heading.
        # sec["_figdir"] is the run directory this case's figures came from,
        # which is where profile_case.sh put its output too.
        _fd = sec.get("_figdir") or ""
        prof_dir = os.path.join(_fd, "profile") if _fd else ""
        prof_figs = [
            ("profile_cpr.png",
             "Inside the linear solve: CPR setup, AMG V-cycles, stage-2, "
             "operator SpMV and CGS2, for both multigrid backends"),
            ("profile_timeline.png",
             "Nsight Systems timeline: the whole profiled run above, and the "
             "busiest window below with the individual kernel launches"),
            ("profile_time.png",
             "Nsight Systems: measured share of GPU time, by phase and by "
             "kernel"),
            ("profile_roofline.png",
             "Nsight Compute: arithmetic intensity against achieved "
             "throughput, with the GB200 FP64 ridge"),
            ("profile_limiters.png",
             "Nsight Compute: DRAM against SM throughput, so what limits each "
             "kernel can be read off directly"),
        ]
        # Each figure may exist plain, or once per backend as
        # profile_time_amgx.png beside profile_time_hypre.png.  Both are
        # collected: a report that showed only one of two profiled backends
        # would be worse than one that showed neither, because nothing on the
        # page would say the other had been left out.
        _expanded = []
        for _f, _cap in prof_figs:
            _stem, _ext = os.path.splitext(_f)
            if os.path.exists(os.path.join(prof_dir, _f)):
                _expanded.append((_f, _cap))
            try:
                _sib = sorted(x for x in os.listdir(prof_dir)
                              if x.startswith(_stem + "_") and x.endswith(_ext))
            except OSError:
                _sib = []
            for _s in _sib:
                _back = _s[len(_stem) + 1:-len(_ext)]
                _expanded.append((_s, "%s &mdash; %s backend" % (_cap, _back)))
        prof_figs = _expanded
        have_prof = [f for f, _ in prof_figs
                     if os.path.exists(os.path.join(prof_dir, f))]
        if have_prof:
            A("<h3>Profile</h3>")
            psum = {}
            pj = os.path.join(prof_dir, "profile_summary.json")
            if os.path.exists(pj):
                try:
                    with open(pj, encoding="utf-8") as fh:
                        psum = json.load(fh)
                except Exception:
                    psum = {}
            pmeta = psum.get("meta", {})
            # Only three of the four figures come from Nsight.  profile_cpr is
            # the solver timing itself, and it is drawn whether or not a
            # profiler ever ran -- so claiming "Nsight measured this" on the
            # strength of that figure alone states something false about where
            # the numbers came from.  Say only what is actually on the page.
            _nsight = [f for f in have_prof if not f.startswith("profile_cpr")]
            A('<div class="note">The wall-clock table above is what the run '
              'timed itself as.'
              + (' This is what Nsight measured independently.' if _nsight
                 else '')
              + ' The first figure needs no profiler at all: the solver times '
              'its own phases and writes them out, which is the only way to '
              'tell a multigrid V-cycle from the operator apply, because both '
              'are the same sparse kernel and a trace cannot separate them.'
              + ('' if _nsight else
                 ' No Nsight figure is present in this report: nsys produced '
                 'no kernel summary for this run, so the timeline and the '
                 'kernel-level breakdown are missing. Re-run with '
                 'NVRS_PROFILE=1, or ./profile_case.sh, on a machine where '
                 'nsys is on PATH.')
              + (" Profiled on %s, %s backend, shortened to TEND=%s days "
                 "&mdash; the kernel mix is the benchmark's; the wall-clock "
                 "total is not, and is not quoted as one."
                 % (pmeta.get("gpu", "?"), pmeta.get("backend", "?"),
                    pmeta.get("tend_days", pmeta.get("report_steps", "?")))
                 if pmeta else "")
              + '</div>')
            if psum.get("ai_median"):
                A("<table><tr><th>Measured</th><th>Value</th></tr>")
                A("<tr><td>Median arithmetic intensity</td><td>%.3f FLOP/byte"
                  "</td></tr>" % psum["ai_median"])
                A("<tr><td>GB200 FP64 ridge</td><td>%.1f FLOP/byte</td></tr>"
                  % psum.get("ridge_flop_per_byte", float("nan")))
                A("<tr><td>Distance below the ridge</td><td>%.0f&times;</td>"
                  "</tr>" % psum.get("below_ridge_factor", float("nan")))
                if psum.get("dram_pct_median") is not None:
                    A("<tr><td>Median DRAM throughput</td><td>%.0f%% of peak"
                      "</td></tr>" % psum["dram_pct_median"])
                if psum.get("sm_pct_median") is not None:
                    A("<tr><td>Median SM throughput</td><td>%.0f%% of peak"
                      "</td></tr>" % psum["sm_pct_median"])
                if psum.get("fp32_share_of_flops") is not None:
                    A("<tr><td>FP32 share of all flops</td><td>%.0f%%</td>"
                      "</tr>" % (100 * psum["fp32_share_of_flops"]))
                A("</table>")
                A('<div class="note">Every kernel below the ridge is the whole '
                  'mixed-precision argument, measured rather than estimated: '
                  'the solve is bandwidth bound, so narrowing the '
                  'preconditioner to FP32 buys time and emulating FP64 costs '
                  'none.</div>')
            A('<div class="grid">')
            for fn, cap in prof_figs:
                fp = os.path.join(prof_dir, fn)
                if os.path.exists(fp):
                    A(img(fp, cap, wide=True))
            A("</div>")

        if sec.get("scaling"):
            sc = sec["scaling"]
            A("<h3>NVRS-CUDA strong scaling</h3>")
            A('<div class="note">The same deck on 1 to N GPUs, run straight '
              'after the head-to-head so the machine is in the same state. '
              'Parallel efficiency is the measured speed-up divided by the '
              'ideal linear one. The last two columns compare each run\'s '
              'final field against the 1-GPU result: a decomposition that '
              'drops a halo layer still converges and still reports plausible '
              'rates, so timings alone cannot tell you the answer survived '
              'being split up.</div>')
            A("<table><tr><th>GPUs</th><th>Cells / GPU</th>"
              "<th>Wall clock (s)</th>"
              "<th>Speed-up</th><th>Efficiency</th>"
              "<th>Linear solve (s)</th><th>AMG setup (s)</th>"
              "<th>AMG rebuilds</th>"
              "<th>Newton its</th><th>Krylov its</th>"
              "<th>max |&Delta;p| vs 1 GPU</th>"
              "<th>max |&Delta;Sw| vs 1 GPU</th></tr>")
            inv = {r[0]: r for r in (sc.get("invariance") or [])}
            for r in sc["table"]:
                iv = inv.get(r["ranks"])
                # Flag the rank counts that are below the point where dividing
                # the grid can pay, so the efficiency column is read as a
                # property of the deck and not of the solver.
                thin = (r.get("cells_per_gpu") and r["eff"] < 60.0
                        and r["cells_per_gpu"] < DECOMP_FLOOR)
                cpg = ("{:,}".format(r["cells_per_gpu"])
                       if r.get("cells_per_gpu") else "-")
                if thin:
                    cpg += " &#9888;"
                setups = r.get("setups") or 0
                reuses = r.get("reuses") or 0
                rebuilt = ("%d of %d" % (setups, setups + reuses)) \
                    if (setups and reuses) else (str(setups) if setups else "-")
                A("<tr><td>%d</td><td>%s</td>"
                  "<td>%.2f</td><td>%.2fx</td><td>%.0f%%</td>"
                  "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                  "<td>%s</td><td>%s</td></tr>"
                  % (r["ranks"], cpg, r["seconds"], r["speedup"], r["eff"],
                     ("%.2f" % r["linear"]) if r["linear"] else "-",
                     ("%.2f" % r["amg_setup"]) if r["amg_setup"] else "-",
                     rebuilt, r["newton"] or "-", r["krylov"] or "-",
                     ("%.3g psi" % iv[1]) if iv else "-",
                     ("%.3g" % iv[2]) if iv else "-"))
            A("</table>")
            if sc.get("verdict"):
                A('<div class="note"><strong>Reading this table.</strong> %s'
                  '</div>' % sc["verdict"])
            A('<div class="note">If the Krylov count climbs with rank count, '
              'the preconditioner itself changed &mdash; BoomerAMG coarsens '
              'differently on N ranks, so each iteration is cheaper but more '
              'are needed. If instead AMG setup grows while the counts hold '
              'steady, the cost is parallel coarsening, which is '
              'communication-bound and barely divides. Those need different '
              'fixes, which is why both columns are here. &#9888; marks a rank '
              'count holding fewer than %s cells per GPU, which is below the '
              'size at which splitting this problem can pay for itself.'
              '</div>' % "{:,}".format(DECOMP_FLOOR))
            A(img(sc["path"], "NVRS-CUDA strong scaling"))

        def _qty_of(path):
            """WOPR out of well_CASE_WOPR.png or well_CASE_WOPR_2.png."""
            stem = os.path.splitext(os.path.basename(path))[0]
            bits = [b for b in stem.split("_") if b.isupper() and len(b) >= 3]
            return bits[-1] if bits else stem

        if sec.get("field_injection"):
            A("<h3>Field injection</h3>")
            A('<div class="note">FWIR and FGIR are the rates actually accepted '
              'after the injector BHP ceiling, not merely the schedule request. '
              'This separates a group-allocation error from an injectivity or '
              'reference-depth limitation before any relative-permeability '
              'parameter is changed.</div>')
            A(img(sec["field_injection"],
                  "Field water and gas injection accepted", wide=True))

        if sec.get("wells"):
            A("<h3>Well response</h3>")
            A('<div class="note">One panel per producer, with shared x/y '
              'limits for each metric so absolute well magnitudes remain '
              'comparable. Every simulator is a line in every panel: OPM '
              'Flow (solid red) and each NVRS-CUDA backend that ran, in its '
              'own colour and dash (legend under each page). '
              'WBHP is reported as zero whenever a well is shut, matching '
              'the Flow summary convention used by these cases.</div>')
            A('<div class="grid">')
            for p in sec["wells"]:
                A(img(p, "%s — one panel per producer" % _qty_of(p), wide=True))
            A("</div>")

        if sec.get("well_r2"):
            A("<h3>Well accuracy against OPM Flow</h3>")
            A("<div class=\"note\">R&sup2; per producer, computed on "
              "OPM&rsquo;s report times with each NVRS series "
              "interpolated onto them.  1.0 is exact; 0 is no better than "
              "a flat line through OPM&rsquo;s mean; negative is worse "
              "than that, and is shown rather than clipped.</div>")
            A('<div class="grid">')
            for p in sec["well_r2"]:
                A(img(p, "%s — R² against OPM Flow" % _qty_of(p), wide=True))
            A("</div>")

        if sec.get("well_diagnostics"):
            d = sec["well_diagnostics"]
            low = d.get("low") or []
            all_rows = d.get("rows") or []
            A("<h3>Active-period well diagnostics</h3>")
            A('<div class="note">Raw R&sup2; is retained above. This table '
              'lists rows below 0.80 using only OPM-active report times, with '
              'a scale-aware NRMSE, cumulative-rate error and opening/closing '
              'date shifts. Where OPM&rsquo;s value is constant over the '
              'active period (a producer held at a fixed bottom-hole '
              'pressure, a water cut that stays at zero), R&sup2; is measured '
              'against 5%% of that constant level instead of its zero '
              'variance, and is marked "constant reference". '
              '%d of %d well/quantity/backend rows are below the threshold.'
              '</div>' % (len(low), len(all_rows)))
            A("<table><tr><th>Backend</th><th>Well</th><th>Quantity</th>"
              "<th>Active points</th><th>Active R&sup2;</th>"
              "<th>NRMSE</th><th>Cumulative error</th>"
              "<th>Opening shift</th><th>Closing shift</th></tr>")
            for r in low:
                r2 = r["r2"]
                A("<tr><td>%s</td><td>%s</td><td>%s</td><td>%d</td>"
                  "<td>%s</td><td>%.2f%%</td><td>%s</td>"
                  "<td>%s</td><td>%s</td></tr>"
                  % (r["source"], r["well"], r["quantity"], r["active"],
                     (("%.4f" % r2) + (" (constant reference)"
                                        if r.get("constant_reference") else ""))
                     if np.isfinite(r2) else "n/a",
                     r["nrmse"],
                     ("%.2f%%" % r["cumulative"])
                     if r["cumulative"] is not None else "n/a",
                     ("%+.1f d" % r["open_shift"])
                     if r["open_shift"] is not None else "n/a",
                     ("%+.1f d" % r["close_shift"])
                     if r["close_shift"] is not None else "n/a"))
            A("</table>")
        if sec.get("balance"):
            b = sec["balance"]
            A("<h3>Water balance</h3>")
            A('<div class="note">Surface water in place applies the run&rsquo;s '
              'rock compressibility and water FVF to each cell&rsquo;s '
              '<code>PORV &times; Sw</code>. Production uses the exact FWPT '
              'summary cumulative when available; injection is integrated '
              'from the reported injector rates. Thus every column has one '
              'stock-tank volume basis.</div>')
            A("<table><tr><th>Run</th><th>Water in place t=0</th>"
              "<th>at t=%g d</th><th>Gained</th><th>Produced</th>"
              "<th>Implied injected</th><th>Reported injected</th>"
              "<th>Closure error</th></tr>" % b["days"])
            for r in b["table"]:
                bad = abs(r["err"]) > 5.0 if np.isfinite(r["err"]) else False
                A("<tr><td>%s</td><td>%.4g</td><td>%.4g</td><td>%.4g</td>"
                  "<td>%.4g</td><td>%.4g</td><td>%.4g</td><td>%s</td></tr>"
                  % (r["name"], r["w0"], r["wend"], r["gain"], r["produced"],
                     r["implied"], r["injected"],
                     ("<b>%+.1f%%</b>" % r["err"]) if bad
                     else ("%+.1f%%" % r["err"]) if np.isfinite(r["err"])
                     else "-"))
            A("</table>")
            A("<p style=\"font-size:.88rem\">All volumes are million STB. "
              "<code>Implied injected = gained + produced</code>. Closure "
              "error compares this with integrated reported injection; a run "
              "more than 5% off is not conserving water.</p>")
            A(img(b["path"], "Water in place over time, both simulators"))

        if sec.get("metrics"):
            A("<h3>Field agreement</h3>")
            names = sec["metric_names"]
            A("<table><tr><th>Step</th><th>Time (d)</th>")
            for f in names:
                A("<th>%s RMS</th><th>%s max</th><th>%s RMS/range</th>"
                  % (f, f, f))
            A("</tr>")
            for r in sec["metrics"]:
                A("<tr><td>%d</td><td>%g</td>" % (r["step"], r["time"]))
                for f in names:
                    rms, linf, rel = r[f]
                    A("<td>%.4g</td><td>%.4g</td><td>%s</td>"
                      % (rms, linf, "n/a" if not np.isfinite(rel)
                         else "%.3g%%" % (100.0 * rel)))
                A("</tr>")
            A("</table>")

        if sec.get("movies"):
            A("<h3>Evolution</h3>")
            for p in sec["movies"]:
                if not os.path.exists(p):
                    continue
                if p.lower().endswith(".mp4"):
                    src = data_uri(p, budget, force=True) or os.path.basename(p)
                    A('<figure class="wide"><video controls loop muted '
                      'src="%s"></video><figcaption>All report steps in '
                      'sequence</figcaption></figure>' % src)
                else:
                    A(img(p, "All report steps in sequence", wide=True,
                          force=True))

        if sec.get("panels"):
            A("<h3>Report steps</h3>")
            A('<div class="note">First and last report step only. The '
              'animation above covers every step.</div>')
            for p, t in sec["panels"]:
                A(img(p, "t = %g days — columns: OPM Flow, NVRS-CUDA, "
                         "difference; rows: pressure and phase saturations"
                      % t, wide=True))

    A('<hr><p style="font-size:.8rem;color:#888">Generated by '
      'benchmark/compare.py &middot; NVRS-CUDA &middot; '
      'Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko, Nick Luiken, '
      'Issam Said</p>')
    A("</body></html>")

    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write(relabel("\n".join(H)))
    log("report: %.1f MB embedded, %d figure(s) too large to inline"
        % (budget["used"] / 1e6, budget["skipped"]))
    return outpath


# ===========================================================================
# Driver
# ===========================================================================
# The solver arms, in the order the verdict table shows them: timing.json key,
# the well-curve label compute_well_diagnostics files rows under, and the
# column heading.
BACKEND_ARMS = [
    ("nvrs_amgx", "NVRS-CUDA (AmgX)", "AmgX FP32"),
    ("nvrs_hypre", "NVRS-CUDA (HYPRE)", "HYPRE FP32"),
    ("nvrs_ginkgo", "NVRS-CUDA (Ginkgo FP16)", "Ginkgo FP16"),
    ("nvrs_ginkgo32", "NVRS-CUDA (Ginkgo FP32)", "Ginkgo FP32"),
    ("nvrs_ginkgo_abj", "NVRS-CUDA (Ginkgo FP16 + adaptive bJ)",
     "Ginkgo FP16 + adaptive bJ"),
    ("nvrs_amgx_fp64", "NVRS-CUDA (AmgX, FP64 operator)",
     "AmgX FP32, FP64 operator"),
]


def backend_verdict(timing_json, well_diag, well_r2=None, tol=0.01):
    """Wall clock and mean well R2 per backend, and which one is best.

    Mean R2 of a quantity is the whole-history R2 of each producer against
    OPM Flow -- the number printed in that well's panel of the R2 charts --
    summed over the producers and divided by their number (22 on Norne).
    The all-quantities value is the mean of those six.  "Best" is the fastest
    backend whose all-quantities mean is within `tol` of the most accurate
    backend: speed only counts among answers that are as good as the best.

    This used to average the active-period R2 of the well diagnostics instead
    (only the report times a well is flowing), clipped to [-1, 1]: a different
    number from the charts, and far lower (Norne AmgX 0.51 against 0.79).
    That is still the fallback when no chart values are passed.
    """
    rows = (well_diag or {}).get("rows", []) or []
    cols = []
    for key, label, short in BACKEND_ARMS:
        e = (timing_json or {}).get(key) or {}
        secs = e.get("seconds")
        r2 = {}
        for kw, _l, _d in WELL_QTY:
            if well_r2:
                v = [float(val) for (src, q, _w), val in well_r2.items()
                     if src == label and q == kw and np.isfinite(val)]
            else:
                v = [max(-1.0, min(1.0, float(r["r2"]))) for r in rows
                     if r.get("source") == label and r.get("quantity") == kw
                     and r.get("r2") is not None and np.isfinite(r["r2"])]
            r2[kw] = (float(np.mean(v)) if v else None, len(v))
        means = [m for m, _n in r2.values() if m is not None]
        if not secs and not means:
            continue
        cols.append(dict(key=key, name=short, seconds=secs, r2=r2,
                         overall=float(np.mean(means)) if means else None))
    if not cols:
        return None
    best_acc = max((c["overall"] for c in cols if c["overall"] is not None),
                   default=None)
    eligible = [c for c in cols if c["seconds"] and
                (best_acc is None or (c["overall"] is not None and
                                      c["overall"] >= best_acc - tol))]
    best = min(eligible, key=lambda c: c["seconds"]) if eligible else None
    return dict(cols=cols, best=best["name"] if best else None, tol=tol,
                best_acc=best_acc)


def render_backend_verdict(bt):
    """The verdict table as HTML: rows are metrics, columns backends, the best
    backend's column boxed in red, the best value of each row in bold."""
    cols = bt["cols"]
    red = "3px solid #d0021b"
    th = ('<th style="text-align:left;padding:6px 10px">Metric</th>')
    for c in cols:
        st = "padding:6px 10px;text-align:right"
        if c["name"] == bt["best"]:
            st += (";border-left:%s;border-right:%s;border-top:%s;"
                   "color:#d0021b;border-radius:10px 10px 0 0" % (red, red, red))
        th += '<th style="%s">%s</th>' % (st, c["name"])
    lines = ['<table style="border-collapse:separate;border-spacing:0">',
             "<thead><tr>%s</tr></thead><tbody>" % th]
    metric_rows = [("Wall clock (s)", [c["seconds"] for c in cols], "%.2f",
                    "min")]
    for kw, label, _d in WELL_QTY:
        metric_rows.append(("Mean R&sup2; %s" % kw,
                            [c["r2"][kw][0] for c in cols], "%.3f", "max"))
    metric_rows.append(("Mean R&sup2;, all quantities",
                        [c["overall"] for c in cols], "%.3f", "max"))
    for ri, (name, vals, fmt_, sense) in enumerate(metric_rows):
        finite = [v for v in vals if v is not None]
        target = (min(finite) if sense == "min" else max(finite)) \
            if finite else None
        tds = '<td style="padding:5px 10px">%s</td>' % name
        for c, v in zip(cols, vals):
            st = "padding:5px 10px;text-align:right"
            if v is not None and target is not None and abs(v - target) < 1e-12:
                st += ";font-weight:700"
            if c["name"] == bt["best"]:
                st += ";border-left:%s;border-right:%s" % (red, red)
                if ri == len(metric_rows) - 1:
                    st += (";border-bottom:%s;border-radius:0 0 10px 10px"
                           % red)
            tds += '<td style="%s">%s</td>' % (
                st, (fmt_ % v) if v is not None else "&mdash;")
        lines.append("<tr>%s</tr>" % tds)
    lines.append("</tbody></table>")
    if bt["best"]:
        lines.append(
            '<p><strong style="color:#d0021b">Best backend: %s.</strong> '
            "The fastest backend whose mean R&sup2; over all well quantities "
            "is within %.2f of the most accurate backend (%.3f). Mean "
            "R&sup2; of a quantity is each producer&rsquo;s whole-history "
            "R&sup2; against OPM Flow (the value printed in its panel of the "
            "R&sup2; charts below), summed over the producers and divided by "
            "their number; the all-quantities row is the mean of those "
            "rows.</p>" % (bt["best"], bt["tol"], bt["best_acc"]
                           if bt["best_acc"] is not None else float("nan")))
    return "".join(lines)


def process_run(rundir, workers, keep_panels):
    case = os.path.basename(os.path.normpath(rundir))
    opm_base, nvrs_base = find_bases(rundir)
    if not opm_base or not nvrs_base:
        log("%s: missing %s output; skipping"
            % (case, "OPM" if not opm_base else "NVRS"))
        return None
    log("%s: OPM=%s  NVRS=%s" % (case, os.path.basename(opm_base),
                                 os.path.basename(nvrs_base)))

    A = load_case(opm_base)
    B = load_case(nvrs_base)
    if A["dims"] != B["dims"]:
        log("  ! grids differ: %s vs %s; skipping" % (A["dims"], B["dims"]))
        return None

    pairs = align_times(A["times"], B["times"])
    if not pairs:
        log("  ! no report times in common; skipping")
        return None
    log("  %d report step(s) matched of %d (OPM) and %d (NVRS)"
        % (len(pairs), len(A["times"]), len(B["times"])))

    # Well locations for the 3-D overlay come from the deck the run used.
    wells, producers, q_inj_total = read_wells(rundir)

    rock = rock_figures(rundir, case, A, B, wells)
    # The corner-point geometry the NVRS run wrote, if it wrote one.  With it
    # the panels draw the field's real dipping, faulted surface, the way the
    # single-run report already does; without it they fall back to the index
    # box, which is right for a Cartesian case and wrong for this one.
    cg = sorted(glob.glob(os.path.join(rundir, "**", "cgrid.bin"),
                          recursive=True))
    cgrid_path = cg[0] if cg else None
    if cgrid_path:
        log("  corner-point geometry: %s"
            % os.path.relpath(cgrid_path, rundir))
    # NVRS report steps after Flow's last one join the movie, NVRS only.
    t_last = float(pairs[-1][2])
    tail = [(ib, float(tb)) for ib, tb in enumerate(B["times"])
            if float(tb) > t_last * (1 + 1e-6) + 1e-9]
    if tail and os.environ.get("NVRS_MOVIE_TAIL", "1") != "0":
        log("  movie: %d further NVRS step(s) after Flow's last report (%g d)"
            % (len(tail), t_last))
    else:
        tail = []
    panels, names = render_panels(rundir, case, A, B, pairs, wells, workers,
                                  cgrid_path)
    # Every NVRS backend, not just the one the field panels use: a well
    # curve is a scalar in time and three lines fit on one panel.
    # Last net: if neither the deck nor meta.json named any producer, take
    # them from OPM's own summary.  A well with an oil-rate vector in the
    # reference IS a producer, and having the curves is worth more than
    # insisting on the tidier source.
    if not producers:
        _, _vec = summary_of_base(opm_base)
        if _vec:
            seen = []
            for key in _vec:
                if ":" not in key:
                    continue
                kw, _, w = key.partition(":")
                if kw in ("WOPR", "WWPR", "WGPR") and w and w not in seen:
                    seen.append(w)
            if seen:
                producers = seen
                log("  wells: %d producer(s) from OPM's summary" % len(seen))

    nvrs_srcs = find_nvrs_bases(rundir)
    field_injection = render_field_injection(rundir, case, opm_base,
                                             nvrs_srcs)
    # Base paths, not loaded cases: the well vectors are read straight from
    # the summary, so these figures survive a run whose EGRID a reader will
    # not accept -- which is exactly the state a partly-broken run is in.
    wellfigs = render_well_curves(rundir, case, opm_base, nvrs_srcs,
                                  producers) or []
    # And how well each backend reproduces OPM, well by well.
    wellr2_values = {}
    wellr2 = render_well_r2(rundir, case, opm_base, nvrs_srcs, producers,
                            collect=wellr2_values) or []
    well_diag = compute_well_diagnostics(opm_base, nvrs_srcs, producers)
    # The per-well table in machine-readable form: NRMSE against the robust
    # reference range, cumulative-rate error, opening/closing shifts and the
    # water-breakthrough offset, for every backend.  The report only draws the
    # worst rows.
    if well_diag:
        try:
            with open(os.path.join(rundir, "well_diagnostics.json"), "w") as f:
                json.dump(well_diag, f, indent=1, default=float)
        except Exception as exc:                              # noqa: BLE001
            log("  ! well_diagnostics.json not written: %s" % exc)
    mets = field_metrics(A, B, pairs, names)
    accfig = accuracy_chart(rundir, case, mets, names)
    balance = material_balance(rundir, case, A, B, pairs, producers,
                               q_inj_total)
    notes = insights(A, B, rock) or []

    movies = []
    try:
        movies = viz3d.make_movie(panels,
                                  os.path.join(rundir, "panels_" + case),
                                  fps=2) or []
    except Exception as exc:
        log("  ! animation skipped: %s" % exc)

    t = read_timing(rundir)
    timing, speed = [], None
    # Three runs, not two: OPM Flow, and NVRS with each stage-one multigrid.
    # AmgX and HYPRE solve the same pressure system by different hierarchies,
    # and which one wins on a faulted field is a question about the field, not
    # a setting to assume.  A directory that is not there is simply skipped, so
    # a two-run comparison still works unchanged.
    # Name the stage-one multigrid, always.  "NVRS-CUDA" on its own is not a
    # configuration: AmgX and HYPRE solve the same pressure system by different
    # hierarchies and can differ by more than the gap to Flow, so a bar without
    # the preconditioner on it cannot be compared with anything.  A directory
    # that is not there is skipped, so a two-run comparison still works.
    seen_names = set()
    for key, nm in (("opm", "OPM Flow"),
                    ("nvrs_amgx", "NVRS-CUDA (AMGX)"),
                    ("nvrs_hypre", "NVRS-CUDA (HYPRE)"),
                    ("nvrs_ginkgo", "NVRS-CUDA (GINKGO FP16)"),
                    ("nvrs_ginkgo32", "NVRS-CUDA (GINKGO FP32)"),
                    ("nvrs_ginkgo_abj", "NVRS-CUDA (GINKGO FP16 + ADAPTIVE BJ)"),
                    ("nvrs_amgx_fp64", "NVRS-CUDA (AMGX, FP64 OPERATOR)"),
                    # A single unnamed run: its solver is whatever the deck
                    # selected, so it is not labelled as either backend.
                    ("nvrs", "NVRS-CUDA")):
        if nm in seen_names:
            continue
        e = t.get(key)
        if e:
            seen_names.add(nm)
            timing.append(dict(name=nm, seconds=e.get("seconds"),
                               hardware=e.get("hardware", "-"),
                               ranks=e.get("ranks", "-"),
                               solver=e.get("solver", "-"),
                               breakdown=e.get("breakdown") or {},
                               steps=len(pairs)))
    # The headline sentence, against OPM Flow, for however many NVRS runs there
    # are.  Naming each by its stage-one multigrid is the point of running both.
    ref = next((x for x in timing
                if x["name"] == "OPM Flow" and x["seconds"]), None)
    others = [x for x in timing if x is not ref and x["seconds"]]
    if ref and others:
        bits = []
        for x in others:
            r = ref["seconds"] / x["seconds"]
            bits.append("%s is %.2fx %s (%.2f s vs %.2f s)"
                        % (x["name"], r if r >= 1 else 1.0 / r,
                           "faster" if r >= 1 else "slower",
                           x["seconds"], ref["seconds"]))
        speed = "Against OPM Flow on this case: " + "; ".join(bits) + "." 

    runtime_fig = runtime_chart(rundir, case, timing)
    scaling = scaling_study(rundir, case, wells)
    three = "gas" in names
    blurb = ("Two-phase oil/water. Both runs use the same grid, rock, PVT, "
             "relative-permeability tables, initialization, wells, controls, "
             "and report schedule. Differences below therefore isolate the "
             "simulator implementations, linear solvers, and time stepping."
             if not three else
             "Full three-phase with mobile gas. NVRS and Flow use the same "
             "ECLIPSE-default saturation-weighted three-phase oil relative "
             "permeability convention. Remaining differences below come from "
             "the reservoir state, well controls and nonlinear time stepping.")

    backend_table = backend_verdict(t, well_diag, wellr2_values)
    if backend_table and backend_table.get("best"):
        log("  best backend: %s" % backend_table["best"])
    sec = dict(case=case,
               backend_table=backend_table,
               title="%s &mdash; %s" % (case, "three-phase" if three
                                        else "two-phase"),
               blurb=blurb, warn=three,
               timing=timing, speedup=speed, runtime_fig=runtime_fig,
               field_injection=field_injection,
               wells=wellfigs, well_r2=wellr2,
               well_diagnostics=well_diag,
               metrics=mets, metric_names=names, movies=movies,
               balance=balance, rock=rock, notes=notes, scaling=scaling,
               # Kept so write_report can re-encode the animation smaller if
               # the page would otherwise blow its budget.
               _movie_frames=panels,
               panels=_endpoints(list(zip(panels,
                                          [p[2] for p in pairs]))))
    # Where every figure for this case landed, so main() can clear them once
    # they are inside the page.  rundir, not outdir: inside process_run the
    # case directory is `rundir`, and every figure writer above joins onto it.
    sec["_figdir"] = rundir
    if not keep_panels:
        sec["_cleanup"] = panels
    return sec


def _endpoints(items):
    """First and last of a sequence, without duplicating a single element.

    The animation still covers every report step; the static panels are only
    the two a reader actually compares.
    """
    if len(items) <= 2:
        return items
    return [items[0], items[-1]]


def read_wells(rundir):
    """Well head locations for the 3-D overlay, from the NVRS deck.

    Parsed from the deck rather than the output because neither EGRID nor
    UNRST carries the i,j of a well, and the deck is the thing both runs were
    generated from.
    """
    wells, producers, q_inj = [], [], [0.0]

    # A FIELD deck has no WELLS block.
    #
    # This function reads the generated benchmark decks, where the wells are
    # written out as a WELLS section.  Norne's wells come from the field
    # deck's schedule -- WELSPECS, COMPDAT, WCONHIST -- and none of that is in
    # the run deck, so the parse below found nothing, `producers` came back
    # empty, and every well figure was skipped without a word: no well
    # response section and no accuracy section on the page at all.
    #
    # The run writes what it resolved into meta.json, which is the same list
    # after the schedule has been applied, so prefer it where it exists.
    for meta_path in sorted(glob.glob(os.path.join(rundir, "**", "meta.json"),
                                      recursive=True)):
        try:
            with open(meta_path, encoding="utf-8", errors="replace") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        for w in (meta.get("wells") or []):
            name = w.get("name")
            if not name:
                continue
            kind = str(w.get("type", "prod"))
            # "prod", or "winj"/"ginj"/"inj" for the injectors.
            wells.append(dict(i=int(w.get("i", 0)), j=int(w.get("j", 0)),
                              name=name,
                              type="prod" if kind == "prod" else "inj"))
            if kind == "prod":
                producers.append(name)
        if wells:
            log("  wells: %d from %s (%d producer(s))"
                % (len(wells), os.path.relpath(meta_path, rundir),
                   len(producers)))
            break

    hits = sorted(glob.glob(os.path.join(rundir, "*.DATA")))
    hits += sorted(glob.glob(os.path.join(rundir, "nvrs", "*.DATA")))
    deck = next((h for h in hits if not h.upper().endswith("_OPM.DATA")), None)
    if wells or not deck:
        return wells, producers, q_inj[0]
    with open(deck, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    inside = False
    for ln in lines:
        s = ln.split("--")[0].strip()
        if not s:
            continue
        if s.upper().startswith("WELLS"):
            inside = True
            continue
        if inside:
            if s.startswith("/"):
                break
            tok = s.split()
            if len(tok) < 9:
                continue
            try:
                i, j, typ = int(tok[0]), int(tok[1]), int(tok[2])
            except ValueError:
                continue
            name = tok[-1]
            wells.append(dict(i=i, j=j, name=name,
                              type="inj" if typ > 0 else "prod"))
            if typ < 0:
                producers.append(name)
            else:
                # Column 4 is the injector's target surface rate.  Reading it
                # here lets the water balance be checked against what the deck
                # actually asked for, rather than only against itself.
                try:
                    q_inj[0] += float(tok[3])
                except (IndexError, ValueError):
                    pass
    return wells, producers, q_inj[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"),
                    help="directory holding one subdirectory per case")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--case", action="append", metavar="NAME",
                    help="only this case (repeatable).  Rebuilding one page "
                         "from output already on disk needs no simulator run, "
                         "and there is no reason to re-read every other case "
                         "to do it.")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel renderers (0 = auto)")
    ap.add_argument("--max-mb", type=float, default=0.0,
                    help="size budget per report file in MB; 0 (the default) "
                         "means none: every figure and the animation are "
                         "embedded at full resolution, never re-encoded "
                         "smaller or left out.  A positive value restores the "
                         "budget, which may shrink a GIF animation to fit")
    ap.add_argument("--keep-panels", action="store_true",
                    help="keep the per-step PNGs after embedding them")
    ap.add_argument("--combine", default=None, metavar="PATH",
                    help="write every case into one page instead of one "
                         "page each -- for comparing two runs of the same "
                         "case side by side")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.runs):
        log("no runs directory: %s" % args.runs)
        log("run benchmark/run_comparison.sh first")
        return 1
    rundirs = [d for d in sorted(glob.glob(os.path.join(args.runs, "*")))
               if os.path.isdir(d)]
    if args.case:
        want = {c.upper() for c in args.case}
        kept = [d for d in rundirs if os.path.basename(d).upper() in want]
        if not kept:
            log("no case matching %s under %s"
                % (", ".join(sorted(want)), args.runs))
            log("available: %s"
                % ", ".join(os.path.basename(d) for d in rundirs))
            return 1
        rundirs = kept
    if not rundirs:
        log("no case directories under %s" % args.runs)
        return 1

    workers = args.workers or min(os.cpu_count() or 1, 12)
    t0 = time.time()
    sections = []
    for d in rundirs:
        sec = process_run(d, workers, args.keep_panels)
        if sec:
            sections.append(sec)
    if not sections:
        log("nothing to report")
        return 1

    # One page per case by default: the cases are read separately, and a
    # combined file is the sum of their sizes, which is what once pushed it
    # past what a mail client will carry.
    #
    # --combine overrides that, because there is one case where a single page
    # is the point: the same case solved twice by different linear solvers.
    # Reading those as two documents makes the reader hold one set of curves
    # in their head while scrolling the other, which is exactly the comparison
    # the page is supposed to be making.  The embed budget is shared across
    # the sections, so a combined page drops animations before it overruns.
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # No budget unless one is asked for: figures are never compressed or
    # dropped to fit a size (BENCH3P's animation was being re-encoded again
    # and again to squeeze into what the static figures left).
    limit = (int(args.max_mb * 1024 * 1024) if args.max_mb > 0
             else 1 << 62)
    written = []
    if args.combine:
        d = os.path.dirname(os.path.abspath(args.combine))
        if d:
            os.makedirs(d, exist_ok=True)
        write_report(args.combine, sections, stamp, limit)
        written.append(args.combine)
    else:
        for sec in sections:
            case = sec.get("case") or "case"
            out = os.path.join(args.runs, "compare_%s.html" % case)
            write_report(out, [sec], stamp, limit)
            written.append(out)
    if args.out and len(written) == 1:
        try:
            os.replace(written[0], args.out)
            written = [args.out]
        except OSError:
            pass
    # Same rule as scripts/plot_results.py: the figures are embedded in the
    # page and rolled into the animations by now, so on disk they are only
    # fixed-name bait for the next run's glob.  Clear them -- but only if a
    # report was actually written, or a failed run would lose both.
    for sec in sections:
        for p in sec.get("_cleanup", []):
            try:
                os.remove(p)
            except OSError:
                pass
    if written and not os.environ.get("NVRS_KEEP_PNG"):
        gone = 0
        for d in {s.get("_figdir") for s in sections if s.get("_figdir")}:
            for f in (glob.glob(os.path.join(d, "*.png"))
                      + glob.glob(os.path.join(d, "*.PNG"))):
                try:
                    os.remove(f)
                    gone += 1
                except OSError:
                    pass
        if gone:
            log("cleared %d loose PNG(s); the report holds them "
                "(NVRS_KEEP_PNG=1 to keep)" % gone)
    log("done in %.1f s" % (time.time() - t0))
    for w in written:
        log("  %-52s %5.1f MB" % (w, os.path.getsize(w) / 1e6))
    log("open one with:  xdg-open %s" % written[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
