#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# backend_figs.py -- the backend study's two headline charts as PNGs.
#
# The study's real report is benchmark/backend_study/index.html, but GitLab
# renders an HTML file as raw source in the browser, so a README cannot show
# it inline.  These two PNGs are the same data in embeddable form: they are
# generated from the committed scaling.json files -- never typed in -- and
# written to benchmark/docs/, which the READMEs already serve images from.
#
#   python3 scripts/backend_figs.py            # from cuda/
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA = os.path.dirname(HERE)
STUDY = os.path.join(CUDA, "benchmark", "backend_study")
OUT = os.path.join(CUDA, "benchmark", "docs")

C_HYPRE, C_AMGX, C_OPM = "#5B8FF9", "#2e9e6b", "#8f7fd6"


def load(study, backend):
    p = os.path.join(STUDY, study, backend, "scaling.json")
    rows = json.load(open(p, encoding="utf-8"))
    return {r["gpus"]: r for r in rows}


def opm_seconds():
    p = os.path.join(STUDY, "bench3p", "opm", "seconds.txt")
    return float(open(p, encoding="utf-8").read().strip())


# ---------------------------------------------------------------------------
# 1. The head-to-head bar: Flow vs both backends, one GB200 each.
# ---------------------------------------------------------------------------
def speed_png():
    flow = opm_seconds()
    hy = load("bench3p", "hypre")[1]["wall"]
    ax_ = load("bench3p", "amgx")[1]["wall"]

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    names = ["OPM Flow 2024.10\n(gpuistl GPU solver)",
             "NVRS-CUDA\n(LINSOLVER HYPRE)",
             "NVRS-CUDA\n(LINSOLVER AMGX, default)"]
    vals = [flow, hy, ax_]
    cols = [C_OPM, C_HYPRE, C_AMGX]
    top = max(vals)
    cells = load("bench3p", "amgx")[1].get("cells", 0)
    bars = ax.barh(range(3)[::-1], vals, color=cols, height=0.62)
    for y, v in zip(range(3)[::-1], vals):
        ax.text(v + 0.012 * top, y, "%.2f s" % v, va="center", fontsize=10,
                fontweight="bold")
    # The ratios are read off the same numbers as the bars, never typed in.
    ax.text(vals[2] + 0.14 * top, 0, "%.1fx faster than Flow\n%.1fx faster "
            "than HYPRE" % (flow / ax_, hy / ax_),
            va="center", fontsize=9, color=C_AMGX, fontweight="bold")
    ax.set_yticks(range(3)[::-1])
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Wall clock [s], BENCH3P, {:,} cells, 20 report steps, "
                  "one GB200 each".format(int(cells)), fontsize=8.5)
    ax.set_xlim(0, 1.18 * top)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "backend_speed.png"), dpi=170)
    plt.close(fig)
    print("backend_speed.png")


# ---------------------------------------------------------------------------
# 2. The scaling story: the two backends move in opposite directions.
# ---------------------------------------------------------------------------
def scaling_png():
    sizes = [("million", "1M"), ("2m", "2M"), ("4m", "4M"), ("5m", "5M")]
    gpus = [1, 2, 4]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.1))

    # Left: wall clock against ranks on the 5M case -- one line falls, one
    # rises, which is the whole result in a single pair of curves.
    for be, c, lab in (("hypre", C_HYPRE, "HYPRE BoomerAMG"),
                       ("amgx", C_AMGX, "AmgX (default)")):
        d = load("5m", be)
        a1.plot(gpus, [d[g]["wall"] for g in gpus], "-o", color=c, label=lab,
                lw=2.2, ms=5)
        for g in gpus:
            a1.annotate("%.0f" % d[g]["wall"], (g, d[g]["wall"]),
                        textcoords="offset points",
                        xytext=(0, 7 if be == "hypre" else -13),
                        ha="center", fontsize=7.5, color=c)
    a1.set_xticks(gpus)
    a1.set_xlabel("GPUs", fontsize=9)
    a1.set_ylabel("Wall clock [s]", fontsize=9)
    a1.set_title("5,000,000 cells, full schedule", fontsize=9.5)
    a1.legend(fontsize=8, frameon=False)
    a1.grid(alpha=0.25)
    a1.set_ylim(0, 1.2 * max(max(load("5m", b)[g]["wall"] for g in gpus)
                             for b in ("hypre", "amgx")))

    # Right: 4-GPU scaling across the ladder, with break-even marked.
    xs = range(len(sizes))
    for be, c, lab in (("hypre", C_HYPRE, "HYPRE BoomerAMG"),
                       ("amgx", C_AMGX, "AmgX (default)")):
        ys = []
        for st, _ in sizes:
            d = load(st, be)
            ys.append(d[1]["wall"] / d[4]["wall"])
        a2.plot(list(xs), ys, "-o", color=c, label=lab, lw=2.2, ms=5)
        for x, y in zip(xs, ys):
            a2.annotate("%.2f" % y, (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=c)
    a2.axhline(1.0, color="#999999", lw=1, ls="--")
    a2.text(0.04, 1.015, "break-even vs 1 GPU", fontsize=7.5, color="#777777")
    a2.set_xticks(list(xs))
    a2.set_xticklabels([n for _, n in sizes], fontsize=9)
    a2.set_xlabel("Grid size [cells]", fontsize=9)
    a2.set_ylabel(u"Speed-up on 4 GPUs [×]", fontsize=9)
    a2.set_title("Strong scaling at 4 GPUs, each vs its own 1 GPU",
                 fontsize=9.5)
    a2.legend(fontsize=8, frameon=False, loc="upper left")
    a2.grid(alpha=0.25)
    a2.set_ylim(0.4, max(1.55, 0.15 + max(
        load(st, "amgx")[1]["wall"] / load(st, "amgx")[4]["wall"]
        for st, _ in sizes)))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "backend_scaling.png"), dpi=170)
    plt.close(fig)
    print("backend_scaling.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    speed_png()
    scaling_png()
    sys.exit(0)
