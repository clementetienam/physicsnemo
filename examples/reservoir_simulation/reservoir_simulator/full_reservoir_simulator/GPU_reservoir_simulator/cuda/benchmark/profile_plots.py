#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
"""Figures from the Nsight Systems and Nsight Compute profiles.

Three plots, one question each.  Nothing here is decorative: a profile produces
thousands of rows and almost all of them say the same thing, so the job is to
show the few that decide an argument.

  1. Where the time goes.       nsys, per-kernel totals rolled into the phases
                                the wall-clock table already names, so the two
                                can be compared.  If the report says the linear
                                solve is 78% of the run, the trace has to agree.

  2. The roofline, measured.    ncu, arithmetic intensity against achieved
                                throughput, with the GB200 ridge drawn.  This is
                                the whole mixed-precision argument: the kernels
                                sit far below the ridge, so they are bandwidth
                                bound, so narrowing the preconditioner buys time
                                and emulating FP64 costs none.  Until this plot
                                that was an estimate from a byte count.

  3. What limits each kernel.   ncu, memory throughput against SM throughput as
                                percentages of peak.  A kernel high on one axis
                                and low on the other is bound by that one; a
                                kernel low on both is bound by latency or
                                occupancy, which is a different fix.

Run profile_case.sh <CASE> first (profile_norne.sh is the Norne wrapper).
Missing inputs are reported and skipped rather
than faked, so a partial profile still yields the figures it can support.

    python3 profile_plots.py [--case NORNE] [--dir <dir>] [--outdir <same>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NV_GREEN = "#76b900"
GREY = "#9e9e9e"

# Output format and typography.  The HTML wants PNG in the browser's sans; the
# paper wants vector PDF in its serif.  Same figures either way -- a second
# copy of a plot is a copy that will disagree with the first one eventually.
FMT = "png"

# Appended to every figure name, so two backends can be drawn into one
# directory without the second erasing the first.  Empty for a flat
# profile directory, which keeps the old names for the old layout.
SUFFIX = ""


def figpath(outdir, stem):
    return os.path.join(outdir, stem + SUFFIX + "." + FMT)


def use_paper_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.linewidth": 0.8})

# GB200 (sm_100) per-GPU peaks.  Only the ratio matters for the ridge, but the
# absolute values let the plot say how far from peak the kernels actually run.
# Overridable, because a profile taken on other hardware should not be drawn
# against Blackwell's numbers.
PEAK_BW_TBS = float(os.environ.get("PEAK_BW_TBS", "8.0"))      # TB/s, HBM3e
# Same resolution as compare.py's figures, since both end up in one page.
REPORT_DPI = int(os.environ.get("NVRS_REPORT_DPI", "200"))
PEAK_FP64_TFS = float(os.environ.get("PEAK_FP64_TFS", "40.0"))  # TFLOP/s, FP64

# Which phase a kernel belongs to.  Ordered: the first match wins, so the more
# specific patterns come first.  These names are the ones the run's own
# wall-clock table uses, which is the point -- the two must be comparable.
PHASES = [
    # gko:: first: every Ginkgo kernel in this simulator is inside the
    # stage-1 multigrid (the outer Krylov uses cuBLAS, the operator cuSPARSE),
    # and without it Ginkgo's CSR SpMV would be counted as the operator apply.
    # k_narrow16 / k_widen16 / k_scale_f* / k_rhs are the FP16 scaling around it.
    ("AMG (multigrid)",   r"gko::|ginkgo|pgm|multigrid|k_narrow16|k_widen|k_scale_f|k_rhs|k_div32|amgx|hypre|csrmv|jacobi(?!an)|smooth|relax|coarse|prolong|restrict|galerkin|rap"),
    ("Krylov / BLAS",     r"gemv|axpy|dot|nrm2|scal|reduce|blas|givens|ortho|gemm"),
    ("SpMV / operator",   r"spmv|csr|sparse|matvec|apply_op"),
    ("Assembly (physics)", r"assemble|residual|jacobian|k_bo|blackoil|pvt|relperm|well|flux|aim"),
    ("Halo / exchange",   r"halo|pack|unpack|gather|scatter|exchange"),
    ("Memory ops",        r"memcpy|memset|Memcpy|Memset"),
]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Reading what the tools wrote
# ---------------------------------------------------------------------------
def read_csv_rows(path):
    """nsys and ncu both emit CSV with a variable number of preamble lines.

    ncu in particular prints its banner and any warnings before the header, so
    seeking the header rather than assuming line 1 is what makes this survive a
    tool upgrade.
    """
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    # Seek the header rather than assume line 1: nsys prints a report title,
    # and an export notice when a .sqlite already exists, above it.
    #
    # "start (ns)" and "duration" are here because cuda_gpu_trace's header
    # contains neither "time" nor "name" -- it is
    #
    #     Start (ns),Duration (ns),CorrId,GrdX,...
    #
    # so the older tests all missed it, every row was read against a header of
    # None, and the timeline came out empty.  It went unnoticed because the
    # fixtures put the header on line 1, where the default start already
    # pointed, and a default that happens to be right is not a passing test.
    start = None
    for i, ln in enumerate(lines[:120]):
        low = ln.lower()
        if "," not in ln:
            continue
        if ("time" in low or "kernel name" in low or '"id"' in low
                or low.startswith("id,") or "start (" in low
                or "duration" in low):
            start = i
            break
    if start is None:
        return []
    if start >= len(lines):
        return []
    return list(csv.DictReader(lines[start:]))


def numeric(v):
    """Tool CSVs carry thousands separators, percent signs and n/a."""
    if v is None:
        return float("nan")
    s = str(v).strip().replace(",", "").replace("%", "")
    if s in ("", "n/a", "N/A", "-", "nan"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def col(row, *names):
    """First column whose name contains one of `names`, case-insensitively."""
    for want in names:
        for k in row:
            if k and want.lower() in k.lower():
                return row[k]
    return None


def classify(name):
    n = name or ""
    for phase, pat in PHASES:
        if re.search(pat, n, re.I):
            return phase
    return "Other"


def short_kernel(name, width=34):
    """A demangled C++ kernel name is unreadable on an axis; keep the identity.

    Template arguments nest, so they are stripped innermost first; a single
    non-greedy pass left fragments like "reduced_row_major, int>".  Ginkgo
    runs most of its work through generic_kernel_* templates whose identity is
    in the lambda's enclosing function, so for gko:: kernels the last
    kernels::<executor>::<module>::<function> path in the full name is used.
    """
    raw = name or ""
    if "gko::" in raw:
        hits = re.findall(r"kernels::(?:cuda|components|cuda_detail)?:?:?"
                          r"([A-Za-z_][A-Za-z_0-9]*)::([A-Za-z_][A-Za-z_0-9]*)",
                          raw)
        hits = [h for h in hits if h[1] not in ("generic_kernel_1d",
                                                "generic_kernel_2d",
                                                "generic_kernel_reduction_1d",
                                                "generic_kernel_reduction_2d")]
        if hits:
            n = "gko %s::%s" % hits[-1]
            return n if len(n) <= width else n[: width - 1] + "…"
    n = raw
    while "<" in n and re.search(r"<[^<>]*>", n):
        n = re.sub(r"<[^<>]*>", "", n)
    n = n.split("(")[0].strip()
    n = n.replace("void ", "").strip()
    n = n.split("::")[-1]
    return n if len(n) <= width else n[: width - 1] + "…"


# ---------------------------------------------------------------------------
# 1. Where the time goes
# ---------------------------------------------------------------------------
def plot_time_breakdown(d, outdir, meta):
    rows = read_csv_rows(os.path.join(d, "nsys_gpukernsum.csv"))
    if not rows:
        log("  [skip] no nsys_gpukernsum.csv -- run ./profile_case.sh <CASE> nsys")
        return None

    per_phase, per_kernel = {}, {}
    for r in rows:
        name = col(r, "Name", "Kernel")
        t = numeric(col(r, "Total Time", "Time (ns)", "Duration"))
        if not name or not np.isfinite(t):
            continue
        per_phase[classify(name)] = per_phase.get(classify(name), 0.0) + t
        per_kernel[name] = per_kernel.get(name, 0.0) + t

    mem = read_csv_rows(os.path.join(d, "nsys_gpumemtimesum.csv"))
    for r in mem:
        t = numeric(col(r, "Total Time", "Time (ns)", "Duration"))
        if np.isfinite(t):
            per_phase["Memory ops"] = per_phase.get("Memory ops", 0.0) + t

    if not per_phase:
        log("  [skip] gpukernsum had no usable rows")
        return None

    total = sum(per_phase.values())
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14.0, 4.9),
                                   gridspec_kw={"width_ratios": [1, 1.35],
                                                "wspace": 0.75})

    # Left: the phase split, as a horizontal bar rather than a pie -- a pie
    # cannot be read to a percent and this is a plot people will quote from.
    order = sorted(per_phase.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in order]
    vals = np.array([v for _, v in order]) / total * 100.0
    shades = [NV_GREEN if i == 0 else plt.cm.Greens(0.75 - 0.09 * i)
              for i in range(len(vals))]
    y = np.arange(len(vals))[::-1]
    axl.barh(y, vals, color=shades, edgecolor="white", height=0.72)
    for yi, v in zip(y, vals):
        axl.text(v + max(vals) * 0.02, yi, "%.1f%%" % v, va="center",
                 fontsize=9)
    axl.set_yticks(y)
    axl.set_yticklabels(labels, fontsize=9)
    axl.set_xlabel("share of GPU time (%)")
    axl.set_xlim(0, max(vals) * 1.30)
    axl.set_title("Where the GPU time goes", fontsize=11, fontweight="bold")
    axl.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)

    # Right: the individual kernels, because the phase split hides whether one
    # kernel dominates a phase or twenty share it -- and those need different
    # work.
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:12]
    ky = np.arange(len(top))[::-1]
    kv = np.array([v for _, v in top]) / total * 100.0
    kc = [NV_GREEN if classify(n) == labels[0] else GREY for n, _ in top]
    axr.barh(ky, kv, color=kc, edgecolor="white", height=0.72)
    for yi, v in zip(ky, kv):
        axr.text(v + max(kv) * 0.015, yi, "%.1f%%" % v, va="center",
                 fontsize=8)
    axr.set_yticks(ky)
    axr.set_yticklabels([short_kernel(n) for n, _ in top], fontsize=8)
    axr.set_xlabel("share of GPU time (%)")
    axr.set_xlim(0, max(kv) * 1.25)
    axr.set_title("The twelve costliest kernels", fontsize=11, fontweight="bold")
    axr.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)

    fig.suptitle("Nsight Systems: %s backend, %s simulated days"
                 % (meta.get("backend", "?"), meta.get("tend_days", "?")),
                 fontsize=12, fontweight="bold", y=1.02)
    path = figpath(outdir, "profile_time")
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    log("  wrote %s" % path)
    return path



# ---------------------------------------------------------------------------
# 1c. The trace itself, as a timeline
# ---------------------------------------------------------------------------
PHASE_COLOURS = {
    "AMG (multigrid)":    "#76b900",
    "Krylov / BLAS":      "#1f77b4",
    "SpMV / operator":    "#d62728",
    "Assembly (physics)": "#9467bd",
    "Halo / exchange":    "#ff7f0e",
    "Memory ops":         "#8c564b",
    "Other":              "#9e9e9e",
}


# The same bound profile_run.sh writes the CSV with.  Two caps that can
# disagree are a cap nobody can raise: setting PROF_TRACE_ROWS higher for a long
# history would have written the larger file and then thrown the extra away.
TRACE_ROWS = int(os.environ.get("PROF_TRACE_ROWS", "400000"))


def read_gpu_trace(d, cap=TRACE_ROWS):
    """One row per launch: (start_ns, duration_ns, phase).

    Capped, because a trace of a long run has millions of rows and the shape of
    the run is settled long before the last of them.  The cap is reported when
    it bites, since a truncated timeline that says so is useful and one that
    does not is misleading.
    """
    path = os.path.join(d, "nsys_gputrace.csv")
    rows = read_csv_rows(path)
    if not rows:
        return [], False
    out = []
    for r in rows[:cap]:
        st = numeric(col(r, "Start"))
        du = numeric(col(r, "Duration"))
        nm = col(r, "Name") or ""
        if st != st or du != du:
            continue
        out.append((st, du, classify(nm)))
    out.sort(key=lambda t: t[0])
    return out, len(rows) > cap


def plot_timeline(d, outdir, meta):
    ev, truncated = read_gpu_trace(d)
    if not ev:
        log("  [skip] no nsys_gputrace.csv -- run ./profile_case.sh <CASE> nsys")
        return None

    t0 = ev[0][0]
    t_end = max(s + u for s, u, _ in ev)
    span_ms = (t_end - t0) / 1e6
    if span_ms <= 0:
        log("  [skip] the trace has no duration")
        return None

    phases = [p for p in PHASE_COLOURS if any(e[2] == p for e in ev)]

    # The stride is stated wherever it is not 1.  A sampled timeline is an
    # honest picture of a whole run and a dishonest count of its launches, so
    # the count is qualified rather than the sampling hidden.
    stride = 1
    try:
        stride = max(1, int(meta.get("trace_stride", 1)))
    except (TypeError, ValueError):
        stride = 1

    fig, (axo, axz) = plt.subplots(
        2, 1, figsize=(11.5, 1.1 + 0.42 * len(phases) * 2),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.55})

    # --- overview: busy fraction per bin ---------------------------------
    #
    # Not bars.  At this scale a launch is far narrower than a pixel, so
    # drawing them individually produces a solid block that hides exactly the
    # structure the panel exists to show.  Shading by occupancy keeps the gaps
    # visible, which is where a launch-bound solve shows itself.
    nbin = 1400
    bw = (t_end - t0) / nbin
    occ = {p: np.zeros(nbin) for p in phases}
    for st, du, ph in ev:
        if ph not in occ:
            continue
        a = (st - t0) / bw
        b = (st + du - t0) / bw
        i0, i1 = int(a), min(int(b), nbin - 1)
        if i0 == i1:
            occ[ph][i0] += (b - a)
        else:
            occ[ph][i0] += (i0 + 1 - a)
            if i1 > i0 + 1:
                occ[ph][i0 + 1:i1] += 1.0
            occ[ph][i1] += (b - i1)
    # Each row is scaled to its own busiest bin, not to a full bin.
    #
    # Against an absolute scale this panel is nearly blank.  A bin here spans
    # milliseconds and the kernels inside it are microseconds, so a phase that
    # is genuinely active occupies a few percent of one and renders white.  Only
    # multigrid, which dominates, showed at all -- so the panel answered "which
    # phase is biggest", which is what the next figure is for, instead of "when
    # is each phase running", which is the only thing it can answer.
    #
    # Scaled per row, every phase uses the full colour range and the structure
    # is visible in all of them.  The cost is that colours are no longer
    # comparable between rows, which the title says.
    for y, ph in enumerate(phases):
        v = occ[ph].astype(float)
        peak = v.max()
        if peak > 0:
            v = v / peak
        # A gentle gamma lifts the low end, and any activity at all is floored
        # well clear of white: a bin holding one launch should be visible as a
        # mark rather than argued about.
        v = np.power(np.clip(v, 0, 1), 0.55)
        v = np.where(v > 0, 0.22 + 0.78 * v, 0.0)[None, :]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "c", ["#ffffff", PHASE_COLOURS[ph]])
        axo.imshow(v, extent=[0, span_ms, y - 0.42, y + 0.42], aspect="auto",
                   cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                   zorder=2)
    axo.set_xlim(0, span_ms)
    axo.set_ylim(-0.6, len(phases) - 0.4)
    axo.set_yticks(range(len(phases)))
    axo.set_yticklabels(phases, fontsize=8)
    axo.invert_yaxis()
    axo.set_xlabel("time (ms)", fontsize=9)
    axo.set_title("The whole profiled run: when each phase is active "
                  "(each row shaded against its own busiest moment)",
                  fontsize=10, fontweight="bold", loc="left")
    axo.grid(axis="x", alpha=0.25, zorder=0)

    # --- zoom: the densest window, drawn as launches ---------------------
    #
    # Chosen rather than fixed.  A window at the front lands in setup and a
    # window at a fixed offset lands wherever the run happens to be; the
    # busiest window is the steady state, which is the part worth seeing.
    win = min(max(span_ms / 400.0, 0.4), 12.0)
    counts = np.zeros(nbin)
    for st, _, _ in ev:
        counts[min(int((st - t0) / bw), nbin - 1)] += 1
    kbin = max(1, int(win * 1e6 / bw))
    if kbin < nbin:
        csum = np.convolve(counts, np.ones(kbin), mode="valid")
        b0 = int(np.argmax(csum))
    else:
        b0 = 0
    z0 = t0 + b0 * bw
    z1 = z0 + win * 1e6

    drawn = 0
    for y, ph in enumerate(phases):
        segs = [((st - z0) / 1e6, max(du / 1e6, win * 5e-4))
                for st, du, p in ev if p == ph and z0 <= st <= z1]
        drawn += len(segs)
        if segs:
            axz.broken_barh(segs, (y - 0.36, 0.72),
                            facecolors=PHASE_COLOURS[ph], edgecolor="none")
    axz.set_xlim(0, win)
    axz.set_ylim(-0.6, len(phases) - 0.4)
    axz.set_yticks(range(len(phases)))
    axz.set_yticklabels(phases, fontsize=8)
    axz.invert_yaxis()
    axz.set_xlabel("time (ms) within the window", fontsize=9)
    # Named for what it draws.  Under sampling these are not all the launches
    # in the window, and calling them individual launches would say they were.
    if stride > 1:
        zt = ("Zoomed to the busiest %.2f ms: %d launches, 1 in %d sampled "
              "(profile a few steps for every launch)" % (win, drawn, stride))
    else:
        zt = ("Zoomed to the busiest %.2f ms: %d individual launches"
              % (win, drawn))
    axz.set_title(zt, fontsize=10, fontweight="bold", loc="left")
    axz.grid(axis="x", alpha=0.25)

    for ax in (axo, axz):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    if stride > 1:
        note = ("Nsight Systems trace: %s backend, %s simulated days, "
                "%d launches sampled 1 in %d across the whole run"
                % (meta.get("backend", "?"), meta.get("tend_days", "?"),
                   len(ev), stride))
    else:
        note = ("Nsight Systems trace: %s backend, %s simulated days, "
                "%d launches"
                % (meta.get("backend", "?"), meta.get("tend_days", "?"),
                   len(ev)))
    if truncated:
        note += " (cut at %d rows)" % TRACE_ROWS
    fig.suptitle(note, fontsize=12, fontweight="bold", y=1.0)

    path = figpath(outdir, "profile_timeline")
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    log("  wrote %s" % path)
    return path

# ---------------------------------------------------------------------------
# 1b. Inside the linear solve: CPR setup against CPR solve
# ---------------------------------------------------------------------------
# This one needs no profiler.  The solver times its own phases in CprTimings
# and writes them to meta.json, so the breakdown is available for every run
# ever made rather than only the profiled one -- and it is SEMANTIC where a
# kernel trace is not.  nsys can tell you a csrmv took 38% of the GPU; only the
# solver knows whether that csrmv was a V-cycle inside the preconditioner or
# the operator apply in the Krylov loop.
#
# The two backends are timed by the same struct, so AmgX and BoomerAMG land on
# the same axes and can be read against each other directly.
CPR_TERMS = [
    ("AMG setup",        "amg_setup_seconds", "#1b5e20"),
    ("AMG V-cycles",     "amg_solve_seconds", NV_GREEN),
    ("Stage-2 (bJacobi)", "stage2_seconds",   "#8bc34a"),
    ("Stage-2 setup",    "stage2_setup_seconds", "#dcedc8"),
    ("Operator SpMV",    "spmv_seconds",      "#4a7c00"),
    ("CGS2 ortho",       "ortho_seconds",     "#c5e1a5"),
    # HYPRE's IJ vector hand-off, or Ginkgo's FP16 scaling and narrowing.
    ("Marshalling",       "ijvec_seconds",    "#cfd8dc"),
]


def plot_cpr_breakdown(timing, outdir):
    """Wall clock -> assembly / linear, then linear -> the six CPR terms."""
    runs = [(k, v) for k, v in (timing or {}).items()
            if isinstance(v, dict) and k != "opm"
            and isinstance(v.get("breakdown"), dict)
            and v["breakdown"].get("linear_seconds")]
    if not runs:
        log("  [skip] no NVRS breakdown in timing.json (re-run run_norne.sh)")
        return None

    fig, axes = plt.subplots(1, len(runs), figsize=(7.6 * len(runs), 6.0),
                             squeeze=False)
    for ax, (name, rec) in zip(axes[0], runs):
        b = rec["breakdown"]
        wall = float(rec.get("seconds") or b.get("wall_seconds") or 0.0)
        lin = float(b.get("linear_seconds") or 0.0)
        asm = float(b.get("assembly_seconds") or 0.0)

        parts, colours = [], []
        named = 0.0
        for label, key, c in CPR_TERMS:
            v = b.get(key)
            if v is None:
                continue
            v = float(v)
            if v <= 0:
                continue
            parts.append((label, v)); colours.append(c); named += v
        rest = max(lin - named, 0.0)
        if rest > 1e-9:
            parts.append(("unattributed", rest)); colours.append("#eceff1")

        # A stacked bar for the whole run, then the linear solve exploded
        # beside it, with a bracket joining the two: the reader can see both
        # that the solve dominates the run and what dominates the solve,
        # without holding two separate charts in their head.
        ax.bar([0], [asm], width=0.52, color="#90a4ae", edgecolor="white",
               label="assembly")
        ax.bar([0], [lin], bottom=[asm], width=0.52, color=NV_GREEN,
               edgecolor="white", label="linear solve")
        other = max(wall - asm - lin, 0.0)
        if other > 0:
            ax.bar([0], [other], bottom=[asm + lin], width=0.52,
                   color="#eceff1", edgecolor="white", label="everything else")

        bottom = 0.0
        ytop = max(wall, lin) * 1.16
        side = []                      # (centre, text) for labels moved out
        for (label, v), c in zip(parts, colours):
            ax.bar([1], [v], bottom=[bottom], width=0.52, color=c,
                   edgecolor="white")
            txt = "%s  %.1f s (%.0f%%)" % (label, v, 100 * v / lin)
            if v / ytop > 0.085:
                ax.text(1, bottom + v / 2, "%s\n%.1f s  (%.0f%%)"
                        % (label, v, 100 * v / lin), ha="center", va="center",
                        fontsize=8, color="#212121")
            else:
                side.append((bottom + v / 2, txt))
            bottom += v
        # Spread the moved labels upward so consecutive ones keep a gap.
        gap = ytop * 0.045
        placed = []
        for yc, txt in side:
            yl = max(yc, placed[-1] + gap) if placed else yc
            placed.append(yl)
            ax.annotate(txt, xy=(1.26, yc), xytext=(1.42, yl),
                        ha="left", va="center", fontsize=7.5,
                        color="#212121",
                        arrowprops=dict(arrowstyle="-", color="#90a4ae",
                                        lw=0.7))

        ax.plot([0.27, 0.73], [asm + lin, lin], color="#607d8b", lw=1.0,
                ls="--", zorder=0)
        ax.plot([0.27, 0.73], [asm, 0.0], color="#607d8b", lw=1.0,
                ls="--", zorder=0)

        ax.set_xlim(-0.45, 2.55 if side else 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["whole run\n%.1f s" % wall,
                            "the linear solve\n%.1f s  (%.0f%% of the run)"
                            % (lin, 100 * lin / max(wall, 1e-9))], fontsize=9)
        ax.set_ylabel("seconds")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, loc="upper left", frameon=True,
                  framealpha=0.9, edgecolor="none")
        # Headroom, so the legend never sits on top of the bar it describes.
        ax.set_ylim(0, max(wall, lin) * 1.16)

        # The counts turn a time into a rate, which is what tells you whether
        # AMG setup is expensive or merely frequent.
        ns, na = b.get("amg_setups"), b.get("amg_applications")
        bits = []
        if ns:
            bits.append("%d setups (%.0f ms each)"
                        % (ns, 1e3 * float(b.get("amg_setup_seconds", 0)) / ns))
        if na:
            bits.append("%d V-cycle applies (%.2f ms each)"
                        % (na, 1e3 * float(b.get("amg_solve_seconds", 0)) / na))
        if b.get("amg_reuses"):
            bits.append("%d setups skipped by reuse" % b["amg_reuses"])
        if bits:
            # Wrapped and kept inside this axes: two panels side by side put
            # their footnotes on the same baseline, and a single long line from
            # each ran into the other's.
            ax.text(0.5, -0.115, "\n".join(bits), transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color="#555555",
                    linespacing=1.5)

    fig.suptitle("Inside the linear solve: where CPR spends its time",
                 fontsize=13, fontweight="bold", y=1.02)
    path = os.path.join(outdir, "profile_cpr." + FMT)
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    log("  wrote %s" % path)
    return path


# ---------------------------------------------------------------------------
# ncu rows -> flops, bytes, seconds
# ---------------------------------------------------------------------------
def ncu_kernels(d):
    rows = read_csv_rows(os.path.join(d, "ncu_kernels.csv"))
    if not rows:
        return []

    # ncu's long format puts one metric per row; the wide format one kernel per
    # row.  Detect which we have rather than assuming, because --print-summary
    # changes it.
    long_form = any(col(r, "Metric Name") for r in rows[:5])
    per = {}
    if long_form:
        for r in rows:
            kid = (col(r, "ID") or "") + "|" + (col(r, "Kernel Name") or "")
            m = (col(r, "Metric Name") or "").strip()
            v = numeric(col(r, "Metric Value"))
            per.setdefault(kid, {"name": col(r, "Kernel Name") or "", "m": {}})
            per[kid]["m"][m] = v
    else:
        for i, r in enumerate(rows):
            name = col(r, "Kernel Name", "Name") or ""
            per["%d|%s" % (i, name)] = {
                "name": name,
                "m": {k: numeric(v) for k, v in r.items() if k},
            }

    out = []
    for rec in per.values():
        m = rec["m"]

        def g(*names):
            for n in names:
                for k, v in m.items():
                    if n.lower() in k.lower() and np.isfinite(v):
                        return v
            return float("nan")

        dur_ns = g("gpu__time_duration.sum", "Duration")
        byts = g("dram__bytes.sum")
        # Two flops for an FMA, one each for add and multiply -- the standard
        # count, and the one the peak numbers are quoted against.
        f64 = (g("op_dadd_pred_on.sum") + g("op_dmul_pred_on.sum")
               + 2.0 * g("op_dfma_pred_on.sum"))
        f32 = (g("op_fadd_pred_on.sum") + g("op_fmul_pred_on.sum")
               + 2.0 * g("op_ffma_pred_on.sum"))
        dram_pct = g("dram__throughput.avg.pct_of_peak")
        sm_pct = g("sm__throughput.avg.pct_of_peak")
        occ = g("sm__warps_active.avg.pct_of_peak")

        flops = np.nansum([f64, f32])
        if not (np.isfinite(dur_ns) and dur_ns > 0):
            continue
        secs = dur_ns * 1e-9
        rec2 = {
            "name": rec["name"],
            "phase": classify(rec["name"]),
            "secs": secs,
            "bytes": byts,
            "flops": flops if np.isfinite(flops) else float("nan"),
            "f64": f64,
            "f32": f32,
            "dram_pct": dram_pct,
            "sm_pct": sm_pct,
            "occ": occ,
        }
        if np.isfinite(byts) and byts > 0 and np.isfinite(flops) and flops > 0:
            rec2["ai"] = flops / byts
            rec2["gflops"] = flops / secs / 1e9
        out.append(rec2)
    return out


# ---------------------------------------------------------------------------
# 2. The roofline, measured
# ---------------------------------------------------------------------------
def plot_roofline(ks, outdir, meta):
    pts = [k for k in ks if "ai" in k and np.isfinite(k["ai"]) and k["ai"] > 0]
    if not pts:
        log("  [skip] no kernel had both a byte count and a flop count")
        return None

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    bw = PEAK_BW_TBS * 1e3                      # GB/s
    peak = PEAK_FP64_TFS * 1e3                  # GFLOP/s
    ridge = peak / bw

    # Frame the data.  A fixed 1e-3..1e3 window puts every kernel in a tenth of
    # the axis and six decades of it empty, which hides the very spread the
    # plot exists to show.  Keep the ridge in view, since the distance to it is
    # the point, and give the points room either side.
    lo_ai = min(p["ai"] for p in pts)
    hi_ai = max(p["ai"] for p in pts)
    x0 = 10 ** (np.floor(np.log10(lo_ai)) - 0.35)
    x1 = max(ridge * 6.0, hi_ai * 3.0)
    ai = np.logspace(np.log10(x0), np.log10(x1), 400)
    roof = np.minimum(ai * bw, peak)

    ax.loglog(ai, roof, color="#333333", lw=2.0, zorder=3,
              label="GB200 FP64 roofline")
    ax.axvline(ridge, color=GREY, ls=":", lw=1.4, zorder=2)
    ax.text(ridge * 1.12, peak * 0.055,
            "ridge\n%.1f FLOP/byte" % ridge, fontsize=8, color="#555555")
    # The shaded half is the statement the paper makes: everything left of the
    # ridge is bandwidth bound, and every kernel here lands in it.
    ax.axvspan(x0, ridge, color=NV_GREEN, alpha=0.055, zorder=1)
    ax.text(x0 * 1.5, peak * 0.55, "bandwidth bound", fontsize=9,
            color="#4a7c00", fontweight="bold")

    phases = sorted({p["phase"] for p in pts})
    cmap = plt.cm.viridis(np.linspace(0.12, 0.88, max(len(phases), 1)))
    sizes = np.array([p["secs"] for p in pts])
    sizes = 40 + 340 * (sizes / sizes.max() if sizes.max() > 0 else sizes)
    for c, ph in zip(cmap, phases):
        sel = [i for i, p in enumerate(pts) if p["phase"] == ph]
        ax.scatter([pts[i]["ai"] for i in sel], [pts[i]["gflops"] for i in sel],
                   s=[sizes[i] for i in sel], color=c, alpha=0.82,
                   edgecolor="white", linewidth=0.7, zorder=4, label=ph)

    hot = max(pts, key=lambda p: p["secs"])
    ax.annotate(short_kernel(hot["name"], 26),
                (hot["ai"], hot["gflops"]), textcoords="offset points",
                xytext=(10, -14), fontsize=8, color="#333333",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#888888"))

    ax.set_xlim(x0, x1)
    ax.set_xlabel("arithmetic intensity (FLOP / byte of HBM traffic)")
    ax.set_ylabel("achieved throughput (GFLOP/s)")
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=8, loc="lower right", frameon=False,
              title="marker area = time", title_fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    med = float(np.median([p["ai"] for p in pts]))
    ax.set_title("Nsight Compute: every kernel sits below the ridge\n"
                 "median intensity %.2f FLOP/byte, %.0f× short of it"
                 % (med, ridge / med if med > 0 else float("nan")),
                 fontsize=12, fontweight="bold")

    path = figpath(outdir, "profile_roofline")
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    log("  wrote %s" % path)
    return path


# ---------------------------------------------------------------------------
# 3. What limits each kernel
# ---------------------------------------------------------------------------
def plot_limiters(ks, outdir, meta):
    pts = [k for k in ks
           if np.isfinite(k.get("dram_pct", float("nan")))
           and np.isfinite(k.get("sm_pct", float("nan")))]
    if not pts:
        log("  [skip] no throughput percentages in the ncu output")
        return None

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.plot([0, 100], [0, 100], color=GREY, ls="--", lw=1.1, zorder=1)
    ax.fill_between([0, 100], [0, 100], 100, color="#1f77b4", alpha=0.05)
    ax.fill_between([0, 100], 0, [0, 100], color=NV_GREEN, alpha=0.06)
    # In the corners: the data lands mid-right, and a label under a marker is
    # worse than no label.
    ax.text(97, 3.5, "memory bound", fontsize=9.5, color="#4a7c00",
            fontweight="bold", ha="right")
    ax.text(3, 96, "compute bound", fontsize=9.5, color="#1f5c99",
            fontweight="bold", ha="left", va="top")

    sizes = np.array([p["secs"] for p in pts])
    sizes = 40 + 340 * (sizes / sizes.max() if sizes.max() > 0 else sizes)
    occ = np.array([p.get("occ", np.nan) for p in pts], dtype=float)
    sc = ax.scatter([p["dram_pct"] for p in pts], [p["sm_pct"] for p in pts],
                    s=sizes, c=np.where(np.isfinite(occ), occ, 0.0),
                    cmap="viridis", vmin=0, vmax=100, alpha=0.85,
                    edgecolor="white", linewidth=0.7, zorder=4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("achieved occupancy (% of peak)", fontsize=9)

    # Label the three costliest only, alternating above and below so two close
    # markers do not print over each other, and flipping to right-aligned near
    # the frame so no name runs off it.
    for i, p in enumerate(sorted(pts, key=lambda q: -q["secs"])[:3]):
        right = p["dram_pct"] > 72
        ax.annotate(short_kernel(p["name"], 20), (p["dram_pct"], p["sm_pct"]),
                    textcoords="offset points",
                    xytext=(-9 if right else 9, 9 if i % 2 == 0 else -13),
                    ha="right" if right else "left",
                    fontsize=7.6, color="#333333")

    ax.set_xlabel("DRAM throughput (% of peak)")
    ax.set_ylabel("SM throughput (% of peak)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("What limits each kernel", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.02, 0.955,
            "marker area = time;  low on both axes means latency or occupancy,\n"
            "which is a different fix from either bound",
            transform=ax.transAxes, fontsize=8, color="#666666", va="top")

    path = figpath(outdir, "profile_limiters")
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    log("  wrote %s" % path)
    return path


# ---------------------------------------------------------------------------
def write_summary(d, ks, meta):
    """A small JSON the HTML and the paper both read, so neither retypes a number."""
    out = {"meta": meta, "kernels_profiled": len(ks)}
    pts = [k for k in ks if "ai" in k and np.isfinite(k["ai"])]
    if pts:
        ai = np.array([p["ai"] for p in pts])
        out["ai_median"] = float(np.median(ai))
        out["ai_min"] = float(ai.min())
        out["ai_max"] = float(ai.max())
        out["ridge_flop_per_byte"] = PEAK_FP64_TFS * 1e3 / (PEAK_BW_TBS * 1e3)
        out["below_ridge_factor"] = out["ridge_flop_per_byte"] / out["ai_median"]
        f64 = float(np.nansum([p["f64"] for p in pts]))
        f32 = float(np.nansum([p["f32"] for p in pts]))
        if f64 + f32 > 0:
            out["fp32_share_of_flops"] = f32 / (f64 + f32)
    dr = [k["dram_pct"] for k in ks if np.isfinite(k.get("dram_pct", np.nan))]
    sm = [k["sm_pct"] for k in ks if np.isfinite(k.get("sm_pct", np.nan))]
    if dr:
        out["dram_pct_median"] = float(np.median(dr))
    if sm:
        out["sm_pct_median"] = float(np.median(sm))

    rows = read_csv_rows(os.path.join(d, "nsys_gpukernsum.csv"))
    if rows:
        per = {}
        for r in rows:
            t = numeric(col(r, "Total Time", "Time (ns)", "Duration"))
            n = col(r, "Name", "Kernel")
            if n and np.isfinite(t):
                per[classify(n)] = per.get(classify(n), 0.0) + t
        tot = sum(per.values())
        if tot > 0:
            out["time_share"] = {k: v / tot for k, v in
                                 sorted(per.items(), key=lambda kv: -kv[1])}
    p = os.path.join(d, "profile_summary.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log("  wrote %s" % p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--case", default="NORNE",
                    help="case name; --dir is runs/<CASE>/profile unless given")
    ap.add_argument("--dir", default=None,
                    help="where profile_case.sh wrote its output")
    ap.add_argument("--outdir", default=None, help="where to put the figures")
    ap.add_argument("--format", default="png", choices=("png", "pdf"),
                    help="png for the HTML report, pdf for the paper")
    ap.add_argument("--paper-style", action="store_true",
                    help="serif and thinner rules, to match the paper")
    a = ap.parse_args()

    if a.dir is None:
        a.dir = os.path.join(here, "runs", a.case.upper(), "profile")

    global FMT
    FMT = a.format
    if a.paper_style:
        use_paper_style()

    root = a.dir
    outdir = a.outdir or root
    if not os.path.isdir(root):
        log("no profile directory at %s" % root)
        log("run ./profile_case.sh %s first" % a.case.upper())
        return 1
    os.makedirs(outdir, exist_ok=True)

    # One source directory per backend, or the directory itself when it holds
    # the profile directly.  The second form is the old layout and is still
    # what profile_run.sh produces when it is pointed somewhere by hand.
    def holds_profile(p):
        return any(os.path.exists(os.path.join(p, f)) for f in
                   ("profile_meta.json", "nsys_gpukernsum.csv",
                    "ncu_kernels.csv"))

    subs = sorted(os.path.join(root, x) for x in os.listdir(root)
                  if os.path.isdir(os.path.join(root, x))
                  and holds_profile(os.path.join(root, x)))
    sources = subs if subs else [root]

    # The CPR split is drawn once, not once per backend: it comes from the
    # run's own timings and already shows both backends side by side.
    # runs/<CASE>/timing.json, beside the profile directory.  This named
    # NORNE outright and then "fell back" to the identical path by another
    # route, so every other case silently lost its CPR figure.
    tj = os.path.join(os.path.dirname(root), "timing.json")
    timing = {}
    if os.path.exists(tj):
        with open(tj, encoding="utf-8") as fh:
            timing = json.load(fh)
    else:
        log("  [note] no timing.json beside the profile; CPR split skipped")

    global SUFFIX
    made = [plot_cpr_breakdown(timing, outdir)]

    for d in sources:
        meta = {}
        mp = os.path.join(d, "profile_meta.json")
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                meta = {}
        # Named after the directory rather than the metadata: the directory is
        # what guarantees uniqueness, and a missing or stale profile_meta.json
        # would otherwise collide two backends back onto one filename.
        SUFFIX = "" if d == root else "_" + os.path.basename(d)

        log("profile figures from %s" % d)
        made.append(plot_time_breakdown(d, outdir, meta))
        made.append(plot_timeline(d, outdir, meta))

        ks = ncu_kernels(d)
        if ks:
            log("  %d kernel launches with counters" % len(ks))
            made.append(plot_roofline(ks, outdir, meta))
            made.append(plot_limiters(ks, outdir, meta))
        else:
            log("  [skip] no ncu_kernels.csv -- run ./profile_case.sh <CASE> ncu")
        write_summary(d, ks, meta)

    SUFFIX = ""
    n = len([m for m in made if m])
    log("%d figure(s) written" % n)
    if n == 0:
        log("nothing to plot; did the profile run succeed?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
