#!/usr/bin/env python3
"""Tables and plots for a precision_fp64_vs_mixed.sh run directory.

    python3 precision_report.py runs/precision_<stamp>

Reads results.csv, each run's newton.csv / linear.csv (NVRS_RESLOG),
RSM_FVM.csv and state_*.bin, and writes RESULTS.txt, fields.csv and plots/:

    wall_clock.png        wall time, mixed against FP64, one GPU, every case
    scaling.png           MILLION ladder on 1, 2, 4 GPUs, both arms
    residual_<run>.png    Newton residual, linear iterations and linear residual

fields.csv holds |mixed - FP64| for pressure and saturations at every report
step.  The field and well figures are compare.py's (compare/compare_<CASE>.html).

It runs again on an unpacked archive (without the field plots, since the
archive has no state_*.bin).
"""
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = os.path.abspath(sys.argv[1])
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

C_MIX, C_F64 = "#76b900", "#1f4e9c"
LBL_MIX, LBL_F64 = "Mixed (FP32 preconditioner)", "All FP64"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "legend.frameon": False, "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  plot: plots/{name}")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# timing table
# --------------------------------------------------------------------------
rows = list(csv.DictReader(open(os.path.join(OUT, "results.csv"))))
# PREC_SKIP="NORNE ..." leaves cases out of every table and plot.
_skip = set(os.environ.get("PREC_SKIP", "").upper().split())
rows = [r for r in rows if r["case"].upper() not in _skip]
for r in rows:
    r.setdefault("gpus", "1")


def med(cs, arm, g, k):
    v = [fnum(r[k]) for r in rows
         if r["case"] == cs and r["arm"] == arm and str(r["gpus"]) == str(g)]
    v = [x for x in v if x == x]
    return statistics.median(v) if v else float("nan")


cases = list(dict.fromkeys(r["case"] for r in rows))
keys = list(dict.fromkeys((r["case"], int(r["gpus"])) for r in rows))
lines = ["Mixed precision (FP32 preconditioner) against all-FP64, same time steps", ""]
lines.append(f"{'case':12s} {'GPUs':>4s} {'arm':6s} {'wall s':>9s} {'Newton':>8s} "
             f"{'Krylov':>9s} {'V-cycle ms':>11s}")
for cs, g in keys:
    for arm in ("mixed", "fp64"):
        lines.append(f"{cs:12s} {g:4d} {arm:6s} {med(cs, arm, g, 'wall_s'):9.2f} "
                     f"{med(cs, arm, g, 'newton'):8.0f} {med(cs, arm, g, 'krylov'):9.0f} "
                     f"{med(cs, arm, g, 'vcycle_ms_each'):11.3f}")
    w = med(cs, "fp64", g, "wall_s") / med(cs, "mixed", g, "wall_s")
    v = med(cs, "fp64", g, "vcycle_ms_each") / med(cs, "mixed", g, "vcycle_ms_each")
    lines.append(f"{cs:12s} {g:4d} FP64 / mixed: wall {w:.2f}x, V-cycle {v:.2f}x")
    lines.append("")


# --------------------------------------------------------------------------
# wall clock, one GPU
# --------------------------------------------------------------------------
one = [cs for cs, g in keys if g == 1]
if one:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(max(8, 1.5 * len(one) + 4), 4.2),
                                  gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(one))
    wm = [med(c, "mixed", 1, "wall_s") for c in one]
    wf = [med(c, "fp64", 1, "wall_s") for c in one]
    b1 = ax.bar(x - 0.2, wm, 0.4, color=C_MIX, label=LBL_MIX)
    b2 = ax.bar(x + 0.2, wf, 0.4, color=C_F64, label=LBL_F64)
    for xi, a, b in zip(x, wm, wf):
        ax.text(xi + 0.2, b, f"{b / a:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("MILLION_", "") for c in one])
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time on one GPU")
    ax.legend(loc="upper left")
    vm = [med(c, "mixed", 1, "vcycle_ms_each") for c in one]
    vf = [med(c, "fp64", 1, "vcycle_ms_each") for c in one]
    ax2.bar(x - 0.2, vm, 0.4, color=C_MIX)
    ax2.bar(x + 0.2, vf, 0.4, color=C_F64)
    for xi, a, b in zip(x, vm, vf):
        if a == a and b == b:
            ax2.text(xi + 0.2, b, f"{b / a:.2f}x", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels([c.replace("MILLION_", "") for c in one])
    ax2.set_ylabel("ms per V-cycle")
    ax2.set_title("AMG V-cycle cost")
    fig.suptitle("Mixed precision against all-FP64 (labels: FP64 / mixed)", y=1.02)
    fig.tight_layout()
    save(fig, "wall_clock.png")


# --------------------------------------------------------------------------
# MILLION scaling
# --------------------------------------------------------------------------
mil = [c for c in cases if c.startswith("MILLION_")]
if mil:
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(mil):
        gs = sorted({g for cc, g in keys if cc == c})
        col = cmap(i / max(1, len(mil) - 1) * 0.85)
        wm = [med(c, "mixed", g, "wall_s") for g in gs]
        wf = [med(c, "fp64", g, "wall_s") for g in gs]
        lab = c.replace("MILLION_", "")
        axs[0].plot(gs, wm, "-o", color=col, label=f"{lab} mixed")
        axs[0].plot(gs, wf, "--s", color=col, label=f"{lab} FP64", mfc="white")
        axs[1].plot(gs, [wm[0] / v for v in wm], "-o", color=col, label=f"{lab} mixed")
        axs[1].plot(gs, [wf[0] / v for v in wf], "--s", color=col, mfc="white",
                    label=f"{lab} FP64")
        axs[2].plot(gs, [b / a for a, b in zip(wm, wf)], "-o", color=col, label=lab)
    gmax = max(g for _, g in keys)
    axs[1].plot([1, gmax], [1, gmax], ":", color="grey", lw=1, label="ideal")
    axs[2].axhline(1, color="grey", lw=1, ls=":")
    for a in axs:
        a.set_xlabel("GPUs")
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel("Wall time (s)"); axs[0].set_title("Strong scaling, wall time")
    axs[1].set_ylabel("Speed-up over one GPU"); axs[1].set_title("Speed-up")
    axs[2].set_ylabel("FP64 / mixed wall time"); axs[2].set_title("Cost of all-FP64")
    axs[0].legend(fontsize=7, ncol=2)
    axs[2].legend(fontsize=8)
    fig.suptitle("MILLION family: solid = mixed, dashed = all FP64", y=1.02)
    fig.tight_layout()
    save(fig, "scaling.png")


# --------------------------------------------------------------------------
# residuals
# --------------------------------------------------------------------------
def read_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        r = list(csv.DictReader(f))
    return r or None


def arm_dirs(cs):
    """(tag, mixed rep1 dir, fp64 rep1 dir) for each GPU count of a case."""
    out = []
    for m in sorted(glob.glob(os.path.join(OUT, cs, "mixed*"))):
        tag = os.path.basename(m)[len("mixed"):]
        f = os.path.join(OUT, cs, "fp64" + tag)
        if os.path.isdir(f):
            out.append((tag, os.path.join(m, "rep1"), os.path.join(f, "rep1")))
    return out


resid_lines = []
for cs in cases:
    for tag, dm, df in arm_dirs(cs):
        nm, nf = read_csv(os.path.join(dm, "newton.csv")), read_csv(os.path.join(df, "newton.csv"))
        lm, lf = read_csv(os.path.join(dm, "linear.csv")), read_csv(os.path.join(df, "linear.csv"))
        if not (nm and nf and lm and lf):
            continue
        fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
        # (a) Newton residual per Newton evaluation
        a = axs[0, 0]
        for n, col, lab in ((nm, C_MIX, LBL_MIX), (nf, C_F64, LBL_F64)):
            y = np.array([fnum(r["rnorm"]) for r in n])
            a.semilogy(np.arange(len(y)), y, color=col, lw=0.6, alpha=0.85, label=lab)
        a.set_xlabel("Newton evaluation (cumulative)")
        a.set_ylabel(r"$\|R\|_2$")
        a.set_title("Nonlinear residual")
        a.legend(loc="upper right")
        # (b) residual history within the first few hardest steps: per-step
        # Newton iteration counts for both arms
        a = axs[0, 1]
        for n, col, lab, mk in ((nm, C_MIX, LBL_MIX, "o"), (nf, C_F64, LBL_F64, "s")):
            per = {}
            for r in n:
                k = (int(r["step"]), int(r["sub"]))
                per[k] = max(per.get(k, 0), int(r["newton"]))
            ks = sorted(per)
            a.plot(np.arange(len(ks)), np.cumsum([per[k] for k in ks]), color=col,
                   lw=2.2 if mk == "o" else 1.3, ls="-" if mk == "o" else "--",
                   label=lab)
        a.set_xlabel("Substep")
        a.set_ylabel("Newton iterations (cumulative)")
        a.set_title("Newton iterations over the run")
        a.legend(loc="upper left")
        # (c) linear iterations per solve, running mean
        a = axs[1, 0]
        for l, col, lab in ((lm, C_MIX, LBL_MIX), (lf, C_F64, LBL_F64)):
            it = np.array([fnum(r["lin_iters"]) for r in l])
            w = max(1, len(it) // 100)
            rm = np.convolve(it, np.ones(w) / w, mode="valid")
            a.plot(np.arange(len(it)), it, color=col, lw=0.3, alpha=0.25)
            a.plot(np.arange(len(rm)) + w // 2, rm, color=col, lw=1.6,
                   label=f"{lab} (mean {it.mean():.1f}, total {int(it.sum())})")
        a.set_xlabel("Linear solve (cumulative)")
        a.set_ylabel("FGMRES iterations")
        a.set_title("Krylov iterations per linear solve")
        a.legend(loc="upper right", fontsize=8)
        # (d) final relative linear residual distribution
        a = axs[1, 1]
        allv = []
        for l in (lm, lf):
            allv += [fnum(r["lin_rel"]) for r in l if fnum(r["lin_rel"]) > 0]
        if allv:
            bins = np.logspace(np.log10(min(allv)), np.log10(max(allv)), 50)
            for l, col, lab in ((lm, C_MIX, LBL_MIX), (lf, C_F64, LBL_F64)):
                v = [fnum(r["lin_rel"]) for r in l if fnum(r["lin_rel"]) > 0]
                a.hist(v, bins=bins, color=col, alpha=0.55, label=lab)
            a.set_xscale("log")
        a.set_xlabel("Final relative linear residual")
        a.set_ylabel("Linear solves")
        a.set_title("Linear solve accuracy reached")
        a.legend(loc="upper left", fontsize=8)
        gp = tag.strip("_") or "1gpu"
        fig.suptitle(f"{cs} ({gp}): residuals, mixed against all-FP64, same time steps", y=1.01)
        fig.tight_layout()
        save(fig, f"residual_{cs}{tag}.png")
        um = [fnum(r["lin_iters"]) for r in lm]
        uf = [fnum(r["lin_iters"]) for r in lf]
        nc = lambda l: sum(1 for r in l if r["converged"] != "1")
        resid_lines.append(
            f"{cs:12s} {gp:5s} Newton evals {len(nm):6d} / {len(nf):6d}   "
            f"linear solves {len(lm):6d} / {len(lf):6d}   mean FGMRES {np.mean(um):6.2f} / "
            f"{np.mean(uf):6.2f}   unconverged {nc(lm)} / {nc(lf)}")


# --------------------------------------------------------------------------
# pressure and saturation
# --------------------------------------------------------------------------
def meta_of(d):
    p = os.path.join(d, "OUTPUT", "meta.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def state(d, s, nbox):
    a = np.fromfile(os.path.join(d, "OUTPUT", "state_%04d.bin" % s), dtype=np.float32)
    return a[:4 * nbox].reshape(4, nbox).astype(np.float64)


field_rows = []
for cs in cases:
    if cs.startswith("MILLION_"):
        continue
    for tag, dm, df in arm_dirs(cs):
        if tag:
            continue
        mm, mf = meta_of(dm), meta_of(df)
        sm = sorted(glob.glob(os.path.join(dm, "OUTPUT", "state_*.bin")))
        sf = sorted(glob.glob(os.path.join(df, "OUTPUT", "state_*.bin")))
        if not (mm and mf and sm and sf):
            continue
        nx, ny, nz = mm["nx"], mm["ny"], mm["nz"]
        nbox = nx * ny * nz
        times = mm.get("times") or list(range(len(sm)))
        nsn = min(len(sm), len(sf))
        hist = []
        for s in range(nsn):
            A, B = state(dm, s, nbox), state(df, s, nbox)
            # Corner-point runs (Norne) write NaN in inactive cells.
            act = (np.isfinite(A[0]) & np.isfinite(B[0]) &
                   ((A[0] != 0) | (B[0] != 0)))
            if not act.any():
                continue
            dP = np.abs(A[0] - B[0])[act]
            Pn = np.abs(B[0][act])
            dS = [np.abs(A[k] - B[k])[act] for k in (1, 2, 3)]
            row = dict(case=cs, snapshot=s, time_days=fnum(times[s]) if s < len(times) else s,
                       max_dP_psi=dP.max(), mean_dP_psi=dP.mean(),
                       rel_l2_P=np.linalg.norm(dP) / max(np.linalg.norm(Pn), 1e-300),
                       max_dSw=dS[0].max(), rmse_dSw=np.sqrt((dS[0] ** 2).mean()),
                       max_dSo=dS[1].max(), max_dSg=dS[2].max(),
                       rmse_dSg=np.sqrt((dS[2] ** 2).mean()))
            field_rows.append(row)
            hist.append(row)
if field_rows:
    with open(os.path.join(OUT, "fields.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(field_rows[0]))
        w.writeheader(); w.writerows(field_rows)

# --------------------------------------------------------------------------
# RESULTS.txt
# --------------------------------------------------------------------------
if resid_lines:
    lines += ["Residual logs (mixed / FP64)"] + resid_lines + [""]
if field_rows:
    lines.append("Field difference at the last snapshot (|mixed - FP64| over active cells)")
    lines.append(f"{'case':12s} {'day':>7s} {'max dP psi':>11s} {'rel L2 P':>10s} "
                 f"{'max dSw':>9s} {'RMS dSw':>9s} {'max dSg':>9s}")
    for cs in dict.fromkeys(r["case"] for r in field_rows):
        r = [x for x in field_rows if x["case"] == cs][-1]
        lines.append(f"{cs:12s} {r['time_days']:7.0f} {r['max_dP_psi']:11.3e} "
                     f"{r['rel_l2_P']:10.2e} {r['max_dSw']:9.2e} {r['rmse_dSw']:9.2e} "
                     f"{r['max_dSg']:9.2e}")
open(os.path.join(OUT, "RESULTS.txt"), "w").write("\n".join(lines) + "\n")
