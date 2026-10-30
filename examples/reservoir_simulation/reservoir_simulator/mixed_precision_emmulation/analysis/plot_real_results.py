#!/usr/bin/env python3
"""
plot_real_results.py
====================
Build the MEASURED-DATA figures for the paper from a real GB200 comparison JSON
(fp64 / mixed / mixed-emu). Writes into ../document:

  real_convergence.png        residual vs iteration, 3 variants, one panel/grid
  real_final_residual_bars.png final relative residual, grouped bars per grid
  real_solvetime_bars.png     solve time, grouped bars per grid, speedup vs FP64
                              annotated, fastest variant highlighted in RED

Usage:
  python plot_real_results.py [path/to/compare.json]
  (defaults to the newest runs_hypre_cusolver/*compare*.json)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "document"

VARIANTS = ["fp64", "mixed", "mixed-emu"]
VLABEL = {"fp64": "FP64", "mixed": "mixed", "mixed-emu": "mixed-emu (df64)"}
VCOL = {"fp64": "#264653", "mixed": "#e76f51", "mixed-emu": "#2a9d8f"}
RED = "#d81e2c"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 170, "font.size": 11,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load(path):
    d = json.loads(Path(path).read_text())
    tol = float(d.get("tol", 1e-8))
    by = {}
    for r in d["results"]:
        by.setdefault(r["n_side"], {})[r["variant"]] = r
    return dict(sorted(by.items())), tol


# --------------------------------------------------------------------------
def plot_convergence(by, tol, out):
    sides = list(by)
    n = len(sides)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.5 * nrow),
                             squeeze=False)
    styles = {"fp64": dict(ls="-", lw=2.2), "mixed": dict(ls="--", lw=1.9),
              "mixed-emu": dict(ls=":", lw=2.6)}
    for k, side in enumerate(sides):
        ax = axes[k // ncol][k % ncol]
        for var in VARIANTS:
            r = by[side].get(var)
            if not r:
                continue
            h = [v for v in r.get("residual_history", []) if v and v > 0]
            ax.semilogy(range(len(h)), h, color=VCOL[var], label=VLABEL[var],
                        **styles[var])
        ax.axhline(tol, color="#888", ls=":", lw=1)
        conv = all(by[side].get(v, {}).get("converged") for v in VARIANTS if by[side].get(v))
        ax.set_title(f"${side}^3$  ({'converged' if conv else 'capped at max-iters'})",
                     fontsize=11)
        ax.set_xlabel("FGMRES iteration")
        ax.set_ylabel(r"$\|b-Ax\|/\|b\|$")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(fontsize=9, loc="upper right")
    for k in range(len(sides), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Measured convergence: FP64 vs mixed vs mixed-emu (df64) on GB200",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out); plt.close(fig)
    print("wrote", out.name)


def plot_final_residual(by, tol, out):
    sides = list(by)
    x = np.arange(len(sides))
    w = 0.26
    offs = (np.arange(3) - 1) * w
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for k, var in enumerate(VARIANTS):
        vals = [by[s].get(var, {}).get("final_residual", np.nan) for s in sides]
        bars = ax.bar(x + offs[k], vals, w, color=VCOL[var], edgecolor="white",
                      linewidth=0.8, label=VLABEL[var], zorder=3)
        for rect, v in zip(bars, vals):
            if v and np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2, v * 1.15,
                        f"{v:.1e}", ha="center", va="bottom", fontsize=7.6,
                        rotation=90, color="#222")
    ax.axhline(tol, color=RED, ls="--", lw=1.3)
    ax.text(len(sides) - 0.5, tol * 1.3, f"tol = {tol:.0e}", color=RED,
            ha="right", fontsize=9)
    ax.set_yscale("log")
    allv = [by[s].get(v, {}).get("final_residual", np.nan) for s in sides for v in VARIANTS]
    allv = [v for v in allv if v and np.isfinite(v)]
    ax.set_ylim(min(min(allv), tol) * 0.25, max(allv) * 60)
    ax.set_xticks(x); ax.set_xticklabels([f"${s}^3$" for s in sides])
    ax.set_xlabel("grid size")
    ax.set_ylabel(r"final relative residual $\|b-Ax\|/\|b\|$")
    ax.set_title("Final residual per variant and grid (lower = more accurate)")
    ax.grid(True, axis="y", which="both", alpha=0.25, zorder=0)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    print("wrote", out.name)


def plot_solvetime(by, out):
    sides = list(by)
    x = np.arange(len(sides))
    w = 0.26
    offs = (np.arange(3) - 1) * w
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ymax = max(by[s].get(v, {}).get("solve_ms", 0) for s in sides for v in VARIANTS)
    bar_handles = {}
    for k, var in enumerate(VARIANTS):
        vals = [by[s].get(var, {}).get("solve_ms", np.nan) for s in sides]
        bars = ax.bar(x + offs[k], vals, w, color=VCOL[var], edgecolor="white",
                      linewidth=0.8, label=VLABEL[var], zorder=3)
        bar_handles[var] = bars
        for rect, v in zip(bars, vals):
            if v and np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2, v + ymax * 0.012,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8.2, color="#222")

    # per grid: speedup vs FP64 + highlight the fastest variant in RED
    for i, s in enumerate(sides):
        times = {v: by[s][v]["solve_ms"] for v in VARIANTS if by[s].get(v)}
        fp = times.get("fp64")
        fastest = min(times, key=times.get)
        # red outline + star on the fastest bar
        rect = bar_handles[fastest][i]
        rect.set_edgecolor(RED); rect.set_linewidth(2.6); rect.set_zorder(5)
        ax.text(rect.get_x() + rect.get_width() / 2, times[fastest] * 0.5,
                "★", ha="center", va="center", fontsize=15, color=RED, zorder=6)
        # speedup annotations vs FP64
        lines = [f"vs FP64:"]
        for v in ("mixed", "mixed-emu"):
            if fp and times.get(v):
                lines.append(f"{VLABEL[v].split(' ')[0]} {fp/times[v]:.2f}×")
        ax.text(x[i], ymax * 1.16, "\n".join(lines), ha="center", va="bottom",
                fontsize=8.4, color="#222", linespacing=1.3)

    ax.set_ylim(0, ymax * 1.42)
    ax.set_xticks(x); ax.set_xticklabels([f"${s}^3$" for s in sides])
    ax.set_xlabel("grid size")
    ax.set_ylabel("solve-stage time [ms]  (setup excluded)")
    ax.set_title("Solve time: FP64 vs mixed vs mixed-emu  "
                 "(★ red = fastest per grid)")
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    handles = [bar_handles[v] for v in VARIANTS] + [
        Patch(facecolor="none", edgecolor=RED, linewidth=2.4, label="fastest (★)")]
    ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    print("wrote", out.name)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        c = sorted(glob.glob(str(ROOT / "runs_hypre_cusolver" / "*compare*.json")))
        if not c:
            raise SystemExit("no compare JSON found")
        path = c[-1]
    print("reading", path)
    by, tol = load(path)
    # print speedup table
    print(f"\n{'side':>5} {'fp64_ms':>9} {'mixed_ms':>9} {'emu_ms':>9} "
          f"{'mix_x':>7} {'emu_x':>7} {'fastest':>10}")
    for s in by:
        t = {v: by[s].get(v, {}).get("solve_ms", np.nan) for v in VARIANTS}
        fp = t["fp64"]
        fastest = min((v for v in VARIANTS if by[s].get(v)), key=lambda v: t[v])
        print(f"{s:>5} {t['fp64']:>9.1f} {t['mixed']:>9.1f} {t['mixed-emu']:>9.1f} "
              f"{fp/t['mixed']:>7.3f} {fp/t['mixed-emu']:>7.3f} {fastest:>10}")
    plot_convergence(by, tol, DOC / "real_convergence.png")
    plot_final_residual(by, tol, DOC / "real_final_residual_bars.png")
    plot_solvetime(by, DOC / "real_solvetime_bars.png")


if __name__ == "__main__":
    main()
