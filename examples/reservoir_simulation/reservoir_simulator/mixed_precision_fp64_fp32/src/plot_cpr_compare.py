#!/usr/bin/env python3
"""
Elegant plots for the HYPRE+cuSolver mixed-precision CPR benchmark.

Reads one or more JSON files produced by run_hypre_cusolver_mixed_cpr.py and
draws:

  1. Convergence: relative residual vs. iteration, one line per grid size for
     each of fp64 (solid) and mixed (dashed). Colour encodes problem size,
     line style encodes precision.
  2. Solve-only speedup: FP64 solve time / mixed solve time vs. grid size
     (setup time is deliberately excluded), with the absolute solve times
     alongside for context.

Usage:
  python plot_cpr_compare.py                          # newest JSON in ./runs_hypre_cusolver
  python plot_cpr_compare.py path/to/compare.json
  python plot_cpr_compare.py a.json b.json --out figs/run

Each run dict is expected to carry: variant ("fp64"/"mixed"), n_side, unknowns,
solve_ms, build_setup_ms, final_residual, converged, residual_history.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator


# ----------------------------------------------------------------------------- style
INK = "#1b1b1b"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.9,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelpad": 7,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.solid_capstyle": "round",
    })


def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=4, width=0.8)


def size_colors(n: int):
    # perceptually-uniform, truncated to avoid the pale yellow end of viridis
    cmap = plt.get_cmap("viridis")
    if n == 1:
        return [cmap(0.35)]
    return [cmap(t) for t in np.linspace(0.12, 0.80, n)]


# ----------------------------------------------------------------------------- data
def load_runs(paths: list[str]) -> list[dict]:
    runs: list[dict] = []
    for p in paths:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        runs.extend(payload.get("results", []))
    # de-dup on (variant, n_side): keep the last one seen
    keyed = {(r["variant"], r["n_side"]): r for r in runs}
    return list(keyed.values())


def by_size(runs: list[dict]) -> dict[int, dict[str, dict]]:
    out: dict[int, dict[str, dict]] = {}
    for r in runs:
        out.setdefault(r["n_side"], {})[r["variant"]] = r
    return dict(sorted(out.items()))


def tol_from(paths: list[str]) -> float | None:
    for p in paths:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        if "tol" in payload:
            return float(payload["tol"])
    return None


# ----------------------------------------------------------------------------- plots
def plot_convergence(grouped, tol, out_base: Path) -> Path | None:
    sizes = list(grouped.keys())
    if not sizes:
        return None
    colors = dict(zip(sizes, size_colors(len(sizes))))

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    style = {"fp64": dict(ls="-", lw=1.9), "mixed": dict(ls="--", lw=1.7)}
    markers = {"fp64": "o", "mixed": "s"}

    any_data = False
    for side in sizes:
        for variant in ("fp64", "mixed"):
            run = grouped[side].get(variant)
            if not run:
                continue
            h = run.get("residual_history") or []
            h = [v for v in h if v is not None and np.isfinite(v) and v > 0]
            if len(h) < 1:
                continue
            any_data = True
            it = np.arange(len(h))
            me = max(1, len(h) // 12)
            ax.plot(it, h, color=colors[side], marker=markers[variant],
                    ms=3.6, markevery=me, markeredgecolor="white",
                    markeredgewidth=0.4, alpha=0.95, **style[variant])
    if not any_data:
        plt.close(fig)
        return None

    if tol:
        ax.axhline(tol, color=MUTED, ls=":", lw=1.1, zorder=0)
        ax.text(ax.get_xlim()[1], tol, f"  tol {tol:.0e}", color=MUTED,
                va="center", ha="left", fontsize=9)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"relative residual  $\|b - Ax\|\,/\,\|b\|$")
    ax.set_title("CPR convergence: FP64 vs mixed precision")
    ax.grid(True, which="major", color=GRID, lw=0.7, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    despine(ax)

    # two legends: colour = size, style = precision
    size_handles = [Line2D([0], [0], color=colors[s], lw=2.4,
                           label=f"${s}^3$") for s in sizes]
    prec_handles = [
        Line2D([0], [0], color=INK, lw=1.9, ls="-", marker="o", ms=4,
               markeredgecolor="white", label="FP64"),
        Line2D([0], [0], color=INK, lw=1.7, ls="--", marker="s", ms=4,
               markeredgecolor="white", label="mixed"),
    ]
    leg1 = ax.legend(handles=size_handles, title="grid size", loc="upper right",
                     ncol=1, handlelength=1.8, borderaxespad=0.8)
    leg1.get_title().set_fontsize(9.5)
    leg1.get_title().set_color(MUTED)
    ax.add_artist(leg1)
    ax.legend(handles=prec_handles, title="precision", loc="lower left",
              handlelength=2.4, borderaxespad=0.8).get_title().set_color(MUTED)

    fig.tight_layout()
    png = out_base.with_name(out_base.name + "_convergence.png")
    pdf = out_base.with_name(out_base.name + "_convergence.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_speedup(grouped, out_base: Path) -> Path | None:
    rows = []
    for side, v in grouped.items():
        f, m = v.get("fp64"), v.get("mixed")
        if not f or not m:
            continue
        fs, ms = f.get("solve_ms"), m.get("solve_ms")
        if not fs or not ms:
            continue
        rows.append((side, fs, ms))
    if not rows:
        return None
    rows.sort()
    sides = [r[0] for r in rows]
    fp64_ms = [r[1] for r in rows]
    mixed_ms = [r[2] for r in rows]
    # speedup is ALWAYS FP64 / mixed (mixed is the baseline -> "mixed to FP64")
    speed = [f / m for _, f, m in rows]

    cf, cm = "#264653", "#e76f51"
    x = np.arange(len(sides))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    b_fp = ax.bar(x - w / 2, fp64_ms, w, color=cf, edgecolor="white",
                  linewidth=0.8, label="FP64", zorder=3)
    b_mx = ax.bar(x + w / 2, mixed_ms, w, color=cm, edgecolor="white",
                  linewidth=0.8, label="mixed", zorder=3)

    ymax = max(max(fp64_ms), max(mixed_ms))
    ax.set_ylim(0, ymax * 1.34)
    ax.set_xlim(-0.7, len(sides) - 0.3)

    # value labels on each bar
    for bars, vals in ((b_fp, fp64_ms), (b_mx, mixed_ms)):
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, val + ymax * 0.012,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9, color=INK)

    # asterisk over the faster (shorter solve time) bar + per-group sentence
    for i in range(len(sides)):
        fp64_faster = fp64_ms[i] <= mixed_ms[i]
        fx = x[i] - w / 2 if fp64_faster else x[i] + w / 2
        fy = (fp64_ms[i] if fp64_faster else mixed_ms[i])
        ax.text(fx, fy + ymax * 0.055, "\u2605", ha="center", va="bottom",
                fontsize=13, color="#e9b949", zorder=4)
        ax.text(x[i], ymax * 1.17,
                f"mixed was {speed[i]:.2f}\u00d7\nfaster than FP64",
                ha="center", va="bottom", fontsize=9, color=INK,
                linespacing=1.25)

    ax.set_xticks(x)
    ax.set_xticklabels([f"${s}^3$" for s in sides])
    ax.set_xlabel("grid size")
    ax.set_ylabel("solve time [ms]  (setup excluded)")
    ax.set_title("Solve time: FP64 vs mixed")
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    handles = [b_fp, b_mx,
               Line2D([0], [0], marker="*", color="none",
                      markerfacecolor="#e9b949", markeredgecolor="#e9b949",
                      markersize=11, label="faster")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, handlelength=1.6, borderaxespad=0.0)

    fig.tight_layout()
    png = out_base.with_name(out_base.name + "_solvetime_bars.png")
    pdf = out_base.with_name(out_base.name + "_solvetime_bars.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def print_summary(grouped, tol) -> None:
    print(f"\n{'side':>6} {'variant':>7} {'iters':>6} {'solve_ms':>10} "
          f"{'final_res':>11} {'converged':>10}")
    for side, v in grouped.items():
        for variant in ("fp64", "mixed"):
            r = v.get(variant)
            if not r:
                continue
            niter = len(r.get("residual_history") or []) - 1
            print(f"{side:>6} {variant:>7} {niter:>6} "
                  f"{r.get('solve_ms', float('nan')):>10.2f} "
                  f"{r.get('final_residual', float('nan')):>11.2e} "
                  f"{str(r.get('converged')):>10}")
    if tol:
        stalled = [(s, vv) for s, v in grouped.items()
                   for vv, r in v.items() if not r.get("converged", False)]
        if stalled:
            print(f"\n[warn] {len(stalled)} run(s) did NOT reach tol={tol:.0e}; "
                  "speedup compares non-converged solves. See convergence plot.")


def newest_default() -> list[str]:
    cands = sorted(glob.glob("./runs_hypre_cusolver/*compare*.json"),
                   key=os.path.getmtime)
    return cands[-1:] if cands else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="*", help="compare JSON file(s)")
    ap.add_argument("--out", default=None, help="output path base (no extension)")
    args = ap.parse_args()

    paths = args.json or newest_default()
    if not paths:
        raise SystemExit("no JSON given and none found in ./runs_hypre_cusolver")

    set_style()
    runs = load_runs(paths)
    grouped = by_size(runs)
    tol = tol_from(paths)

    out_base = Path(args.out) if args.out else Path(paths[0]).with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    c = plot_convergence(grouped, tol, out_base)
    s = plot_speedup(grouped, out_base)
    print_summary(grouped, tol)
    if c:
        print(f"\nwrote convergence: {c}  (+ .pdf)")
    if s:
        print(f"wrote speedup:     {s}  (+ .pdf)")


if __name__ == "__main__":
    main()