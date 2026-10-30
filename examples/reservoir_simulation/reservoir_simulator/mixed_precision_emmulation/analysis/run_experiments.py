#!/usr/bin/env python3
"""
run_experiments.py
==================
Drives every emulation / speed-of-light experiment for the paper and writes the
figures into ../document as PNGs, plus a machine-readable emu_results.json.

Experiments
-----------
 (E1) Emulation accuracy on the real reservoir operator: SpMV, Galerkin RAP,
      and the true residual b-Ax, for FP32/TF32 slices s=1..3 and naive FP32.
 (E2) FGMRES-CPR convergence: FP64 vs mixed(FP32 precond) vs emulated-mixed
      (Ozaki-2 operator) vs plain-FP32 operator.
 (E3) Speed-of-light roofline for one GB200, kernels placed by arithmetic
      intensity, showing FP64/FP32 store-narrow vs emulated points.
 (E4) Mixed-precision speedup projection: what storage width + preconditioner
      fraction is required to reach 2x, then 4x.
 (E5) Emulation trade-off: emulation adds k^2 flops -> wins for compute-bound
      RAP/dense-LU, loses for bandwidth-bound SpMV.

Run:  python run_experiments.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reservoir_operator import build_reservoir_system
from ozaki_emulation import (emulated_spmv, naive_low_spmv, emulated_rap,
                             rel_err, ozaki_flops_factor, split_slices, _pairs)
from fgmres_cpr import CPRPrecond, make_matvec, fgmres_cpr

DOC = Path(__file__).resolve().parent.parent / "document"
DOC.mkdir(exist_ok=True)

# consistent style
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "legend.framealpha": 0.9,
})
C = {"fp64": "#1f3b73", "fp32": "#d1701a", "emu": "#2e8b57",
     "tf32": "#8e44ad", "bad": "#c0392b", "grey": "#666666"}

U_FP64, U_FP32, U_TF32 = 1.1e-16, 6.0e-8, 4.9e-4
results = {}


# ---------------------------------------------------------------------------
def build_aggregation_P(side):
    """Piecewise-constant prolongation over 2x2x2 cell aggregates for Ap."""
    cs = (side + 1) // 2
    cells, ccells = side ** 3, cs ** 3
    rows = np.arange(cells)
    cols = np.empty(cells, dtype=int)
    for k in range(side):
        for j in range(side):
            for i in range(side):
                c = (k * side + j) * side + i
                cc = ((k // 2) * cs + (j // 2)) * cs + (i // 2)
                cols[c] = cc
    return sp.csr_matrix((np.ones(cells), (rows, cols)), shape=(cells, ccells))


# ============================================================ E1: accuracy ==
def experiment_accuracy(side=24, sigma=1.5):
    print(f"[E1] emulation accuracy on {side}^3 reservoir operator ...")
    A, b, xtrue, Ap, dinv = build_reservoir_system(side, sigma)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(A.shape[0])

    truth_spmv = A @ x
    truth_res = b - A @ xtrue                       # ~ 0 in FP64
    P = build_aggregation_P(side)
    R = P.T.tocsr()
    truth_rap = (R @ Ap @ P).toarray()

    configs = [("naive fp32", None, None),
               ("fp32 s=1", 1, "fp32"), ("fp32 s=2", 2, "fp32"), ("fp32 s=3", 3, "fp32"),
               ("tf32 s=2", 2, "tf32"), ("tf32 s=3", 3, "tf32")]
    rows = []
    for name, s, fmt in configs:
        if s is None:
            # plain FP32: cast operands to FP32, accumulate in FP32
            def _f32(M):
                return sp.csr_matrix((M.data.astype(np.float32).astype(np.float64),
                                      M.indices, M.indptr), shape=M.shape)
            e_spmv = rel_err(naive_low_spmv(A, x), truth_spmv)
            r_naive = b - naive_low_spmv(A, xtrue)
            e_res = np.linalg.norm(r_naive) / np.linalg.norm(b)
            e_rap = rel_err((_f32(R) @ _f32(Ap) @ _f32(P)).toarray(), truth_rap)
            cost = 1
        else:
            e_spmv = rel_err(emulated_spmv(A, x, s, fmt), truth_spmv)
            r_emu = b - emulated_spmv(A, xtrue, s, fmt)
            e_res = np.linalg.norm(r_emu) / np.linalg.norm(b)
            e_rap = rel_err(emulated_rap(R, Ap, P, s, fmt).toarray(), truth_rap)
            cost = ozaki_flops_factor(s)
        rows.append(dict(name=name, spmv=e_spmv, rap=e_rap, res=e_res, cost=cost))
        print(f"   {name:11s} SpMV={e_spmv:.2e}  RAP={e_rap:.2e}  res={e_res:.2e}  cost={cost}")
    results["accuracy"] = {"side": side, "rows": rows}

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels = [r["name"] for r in rows]
    xpos = np.arange(len(labels))
    w = 0.26
    ax.bar(xpos - w, [max(r["spmv"], 1e-17) for r in rows], w, label="SpMV $Ax$", color=C["fp64"])
    ax.bar(xpos, [max(r["rap"], 1e-17) for r in rows], w, label="RAP $RA_pP$", color=C["emu"])
    ax.bar(xpos + w, [max(r["res"], 1e-17) for r in rows], w, label=r"residual $b-Ax^\ast$", color=C["fp32"])
    ax.set_yscale("log")
    ax.axhline(U_FP64, color=C["grey"], ls="--", lw=1.2)
    ax.text(len(labels) - 0.5, U_FP64 * 1.3, "FP64 unit roundoff", color=C["grey"],
            ha="right", fontsize=9)
    ax.axhline(U_FP32, color=C["bad"], ls=":", lw=1.2)
    ax.text(0.0, U_FP32 * 1.3, "FP32 unit roundoff", color=C["bad"], fontsize=9)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("relative error vs FP64")
    ax.set_title(f"Emulation accuracy, ${side}^3$ operator: "
                 "Ozaki-2 (FP32 slices) reaches FP64")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(DOC / "emu_accuracy.png"); plt.close(fig)
    print("   wrote emu_accuracy.png")


# ======================================================= E2: convergence ===
def experiment_convergence(side=16, sigma=1.5, restart=30, maxiter=90):
    print(f"[E2] FGMRES-CPR convergence on {side}^3 ...")
    A, b, xtrue, Ap, dinv = build_reservoir_system(side, sigma)
    pre_fp32 = CPRPrecond(A, Ap, dinv, precond_dtype=np.float32)
    pre_fp64 = CPRPrecond(A, Ap, dinv, precond_dtype=np.float64)
    tol = 1e-11
    runs = {}
    _, h = fgmres_cpr(A, b, make_matvec(A, "fp64"), pre_fp64, restart, maxiter, tol)
    runs["FP64 (op FP64, precond FP64)"] = h
    _, h = fgmres_cpr(A, b, make_matvec(A, "fp64"), pre_fp32, restart, maxiter, tol)
    runs["mixed (op FP64, precond FP32)"] = h
    _, h = fgmres_cpr(A, b, make_matvec(A, "emu", s=2, fmt="fp32"), pre_fp32, restart, maxiter, tol)
    runs["emulated-mixed (op Ozaki-2, precond FP32)"] = h
    _, h = fgmres_cpr(A, b, make_matvec(A, "fp32-op"), pre_fp32, restart, maxiter, tol)
    runs["naive (op plain FP32)"] = h

    results["convergence"] = {"side": side,
                              **{k: [float(x) for x in v] for k, v in runs.items()}}
    styles = {
        "FP64 (op FP64, precond FP64)": dict(color=C["fp64"], ls="-", lw=2.4),
        "mixed (op FP64, precond FP32)": dict(color=C["fp32"], ls="--", lw=2.0),
        "emulated-mixed (op Ozaki-2, precond FP32)": dict(color=C["emu"], ls=":", lw=2.6),
        "naive (op plain FP32)": dict(color=C["bad"], ls="-.", lw=1.6),
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, h in runs.items():
        ax.semilogy(np.arange(len(h)), h, label=name, **styles[name])
    ax.axhline(tol, color=C["grey"], ls=":", lw=1)
    ax.set_xlabel("FGMRES iteration"); ax.set_ylabel(r"relative residual $\|b-Ax\|/\|b\|$")
    ax.set_title(f"Emulated-FP64 mixed precision tracks true FP64 ($ {side}^3$)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(min(tol / 5, 1e-12), 2)
    fig.tight_layout()
    fig.savefig(DOC / "emu_convergence.png"); plt.close(fig)
    print("   wrote emu_convergence.png")


# ========================================================= E3: roofline ====
def experiment_roofline():
    print("[E3] speed-of-light roofline ...")
    # Representative single-GB200 Blackwell figures (order-of-magnitude, public).
    BW = 8.0e12                 # HBM3e bytes/s (~8 TB/s)
    peak = {"FP64 vector": 40e12, "FP32 vector": 80e12,
            "TF32 tensor": 1000e12, "FP16 tensor": 2000e12, "INT8 tensor": 4000e12}
    ai = np.logspace(-2, 3.5, 400)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name, p in peak.items():
        ax.loglog(ai, np.minimum(p, BW * ai) / 1e12,
                  lw=1.6, label=f"{name} ({p/1e12:.0f} TF)")
    # bandwidth roof
    ax.loglog(ai, BW * ai / 1e12, color="k", lw=1.0, ls="--", alpha=0.5)
    # kernel markers
    kernels = [
        ("SpMV FP64", 0.25, "FP64 vector", C["fp64"], "o"),
        ("SpMV FP32 (store-narrow)", 0.50, "FP32 vector", C["fp32"], "s"),
        ("AMG V-cycle FP32", 0.45, "FP32 vector", C["emu"], "^"),
        ("SpMV Ozaki-2 (emulated)", 0.10, "FP16 tensor", C["bad"], "X"),
        ("dense RAP / LU (HPL)", 120.0, "FP64 vector", "#444", "D"),
        ("dense LU emulated (Ozaki)", 40.0, "INT8 tensor", C["tf32"], "P"),
    ]
    for name, x, roof, col, mk in kernels:
        y = min(peak[roof], BW * x) / 1e12
        ax.scatter([x], [y], s=90, color=col, marker=mk, zorder=5,
                   edgecolor="k", linewidth=0.6)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5)
    ax.set_xlabel("arithmetic intensity [flop/byte]")
    ax.set_ylabel("attainable performance [TFLOP/s]")
    ax.set_title("Speed-of-light roofline (one GB200): where each kernel lives")
    ax.legend(fontsize=8.5, loc="lower right", ncol=1)
    ax.set_ylim(1, 6000)
    fig.tight_layout()
    fig.savefig(DOC / "sol_roofline.png"); plt.close(fig)
    results["roofline"] = {"HBM_TBps": BW / 1e12, "peak_TF": {k: v / 1e12 for k, v in peak.items()}}
    print("   wrote sol_roofline.png")


# ================================================ E4: speedup projection ===
def experiment_speedup():
    print("[E4] mixed-precision speedup projection ...")
    # Model: solve time = precond_bytes/BW + outer_bytes/BW. Narrowing only the
    # preconditioner storage from 8 -> w bytes scales the precond term.
    # speedup(f, w) = 1 / (f * (w/8) + (1-f))   where f = precond byte-fraction.
    f = np.linspace(0.3, 0.98, 300)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    # effective bytes/value w for the narrowed preconditioner copy:
    #   FP32              : 4 B
    #   BF16/FP16         : 2 B
    #   INT8/FP8 offdiag + FP32 diag (deep hierarchy, scaled) : ~1.1 B
    widths = [("current: FP32 store-narrow (8->4 B)", 4.0, C["fp32"], "-"),
              ("+ BF16/FP16 AMG hierarchy (8->2 B)", 2.0, C["emu"], "--"),
              ("+ INT8/FP8 deep levels, scaled (8->~1.1 B)", 1.1, C["tf32"], "-.")]
    for name, w, col, ls in widths:
        S = 1.0 / (f * (w / 8.0) + (1 - f))
        ax.plot(f, S, color=col, lw=2.4, ls=ls, label=name)
    ax.axhline(4.0, color=C["bad"], ls="--", lw=1.4)
    ax.text(0.31, 4.08, "4x target", color=C["bad"], fontsize=10, weight="bold")
    ax.axhline(2.0, color=C["grey"], ls=":", lw=1.1)
    ax.text(0.31, 2.04, "2x (FP32 ceiling)", color=C["grey"], fontsize=9)
    # measured point (128^3, 1.12x) at estimated precond fraction ~0.55, w=4
    ax.scatter([0.55], [1.12], s=140, color=C["fp64"], marker="*", zorder=6,
               edgecolor="k", label="measured 128$^3$ (FP32, 1.12x)")
    # 4x operating point: strong AMG (f=0.85) + INT8/FP8 deep levels
    def S(f, w): return 1.0 / (f * (w / 8.0) + (1 - f))
    f4, w4 = 0.85, 1.1
    ax.scatter([f4], [S(f4, w4)], s=150, color=C["bad"], marker="P", zorder=7,
               edgecolor="k", label=f"4x recipe: strong AMG $f$=0.85 + INT8/FP8 ({S(f4,w4):.1f}x)")
    ax.set_xlabel("preconditioner share of per-iteration byte traffic  $f$")
    ax.set_ylabel(r"projected solve speedup  $t_{FP64}/t_{mixed}$")
    ax.set_title("Path to 4x: strengthen AMG (raise $f$) and narrow hierarchy storage")
    ax.legend(fontsize=8.6, loc="upper left")
    ax.set_ylim(1, 6)
    fig.tight_layout()
    fig.savefig(DOC / "sol_speedup_projection.png"); plt.close(fig)
    recipe = {"fp32_f0.55_measured_regime": S(0.55, 4), "fp32_f0.85": S(0.85, 4),
              "bf16_f0.85": S(0.85, 2), "bf16_f0.90": S(0.90, 2),
              "int8fp8_f0.85": S(0.85, 1.1), "int8fp8_f0.90": S(0.90, 1.1)}
    results["speedup_projection"] = recipe
    print("   projections:", {k: round(v, 2) for k, v in recipe.items()})
    print("   wrote sol_speedup_projection.png")


# ================================================= E5: emulation tradeoff ==
def experiment_tradeoff():
    print("[E5] emulation cost / benefit vs arithmetic intensity ...")
    BW = 8.0e12
    p_fp64, p_int8 = 40e12, 4000e12
    s = 2
    k2 = ozaki_flops_factor(s)          # 3 partial products for Ozaki-2
    ai = np.logspace(-2, 3, 300)
    # native FP64 time per output-byte-normalised work
    t_native = 1.0 / np.minimum(p_fp64, BW * ai)
    # emulated: k2x flops on int8 tensor core, same bytes
    ai_emu = ai / k2                     # more flops per byte lowers effective AI on the flop axis
    t_emu = 1.0 / np.minimum(p_int8 / k2, BW * ai)
    speedup = t_native / t_emu
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.semilogx(ai, speedup, color=C["emu"], lw=2.4)
    ax.axhline(1.0, color=C["grey"], ls="--", lw=1.2)
    ridge = p_fp64 / BW
    ax.axvline(ridge, color=C["fp64"], ls=":", lw=1.3)
    ax.text(ridge * 1.1, 0.2, f"FP64 ridge\nAI*={ridge:.0f}", color=C["fp64"], fontsize=9)
    ax.scatter([0.25], [np.interp(0.25, ai, speedup)], s=100, color=C["bad"], marker="X",
               zorder=5, edgecolor="k")
    ax.annotate("reservoir SpMV\n(bandwidth-bound:\nemulation cannot help)", (0.25, np.interp(0.25, ai, speedup)),
                textcoords="offset points", xytext=(12, -6), fontsize=8.5, color=C["bad"])
    ax.scatter([120], [np.interp(120, ai, speedup)], s=100, color="#444", marker="D",
               zorder=5, edgecolor="k")
    ax.annotate("dense LU / HPL\n(compute-bound:\nemulation wins)", (120, np.interp(120, ai, speedup)),
                textcoords="offset points", xytext=(-30, 12), fontsize=8.5, color="#444")
    ax.set_xlabel("arithmetic intensity [flop/byte]")
    ax.set_ylabel("emulated-FP64 speedup vs native FP64")
    ax.set_title("Why emulation helps HPL but not the reservoir solve")
    ax.set_ylim(0, max(6, speedup.max() * 1.1))
    fig.tight_layout()
    fig.savefig(DOC / "sol_emulation_tradeoff.png"); plt.close(fig)
    print("   wrote sol_emulation_tradeoff.png")


# ================================================ E6: precision-map pie ====
def experiment_precision_pie():
    """Representative share of reservoir-simulation solve-phase work, coloured
    by precision ROLE: FP32 (store-narrow), emulated-FP64 (Ozaki-2), and native
    FP64 (accuracy-setting, kept exact). Proportions are a defensible model of a
    well-tuned CPR solve (preconditioner-dominated), not a single measurement;
    they shift with cycle strength and grid size (see the traffic model, E4)."""
    print("[E6] precision-map pie ...")
    # (label, share, bucket, detail)
    comp = [
        # ---- FP32: the CPR preconditioner (store-narrow) ----
        ("AMG V-cycle (pressure)", 34, "fp32"),
        ("stage-2 smoother + off-diagonals", 18, "fp32"),
        # ---- emulated-FP64: FP64-accuracy GEMM/SpMV-structured work ----
        ("operator SpMV  $A\\vec z$", 10, "emu"),
        ("true residual  $b-A\\vec x$", 5, "emu"),
        ("AMG Galerkin RAP setup", 5, "emu"),
        # ---- native FP64: accuracy-setting, cheap, kept exact ----
        ("Krylov dots / norms (orthogonality)", 8, "fp64"),
        ("EOS flash + Jacobian assembly", 12, "fp64"),
        ("state update, constraints, diagonal", 8, "fp64"),
    ]
    bucket_name = {"fp32": "FP32  (store-narrow)",
                   "emu": "emulated FP64  (Ozaki-2 / df64)",
                   "fp64": "native FP64  (accuracy-setting)"}
    bucket_col = {"fp32": C["emu"], "emu": C["fp32"], "fp64": C["fp64"]}
    # shades within a bucket for readability
    order = ["fp32", "emu", "fp64"]
    comp.sort(key=lambda r: order.index(r[2]))
    sizes = [c[1] for c in comp]
    labels = [c[0] for c in comp]
    base = {b: np.array(plt.matplotlib.colors.to_rgb(bucket_col[b])) for b in order}
    cols = []
    seen = {b: 0 for b in order}
    per = {b: sum(1 for c in comp if c[2] == b) for b in order}
    for c in comp:
        b = c[2]
        f = 0.62 + 0.5 * (seen[b] / max(per[b] - 1, 1))  # lighten within bucket
        seen[b] += 1
        cols.append(tuple(np.clip(base[b] * f + (1 - f) * 1.0, 0, 1)))

    tot = {b: sum(c[1] for c in comp if c[2] == b) for b in order}
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    wedges, _ = ax.pie(sizes, colors=cols, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
                       radius=1.0)
    # percentage labels on wedges
    for w, s, lab in zip(wedges, sizes, labels):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        r = 0.79
        ax.text(r * np.cos(ang), r * np.sin(ang), f"{s}%", ha="center",
                va="center", fontsize=9, color="white", weight="bold")
    ax.set_aspect("equal")
    # centre summary
    ax.text(0, 0.10, "reservoir\nsolve work", ha="center", va="center",
            fontsize=12, weight="bold", color=C["grey"])
    ax.text(0, -0.16, f"FP32 {tot['fp32']}%  ·  emu {tot['emu']}%  ·  FP64 {tot['fp64']}%",
            ha="center", va="center", fontsize=9.5, color=C["grey"])
    # legend: component -> percent, grouped by bucket
    from matplotlib.patches import Patch
    handles = []
    for b in order:
        handles.append(Patch(facecolor=bucket_col[b],
                             label=f"{bucket_name[b]} — {tot[b]}%"))
        for c in comp:
            if c[2] == b:
                handles.append(Patch(facecolor="none", edgecolor="none",
                                     label=f"      · {c[0]}  ({c[1]}%)"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False, handlelength=1.1, labelspacing=0.35)
    ax.set_title("Precision map of a CPR reservoir solve: what is FP32, what is "
                 "FP64, what can be emulated", fontsize=12.5, pad=14)
    fig.subplots_adjust(left=0.02, right=0.62)
    fig.savefig(DOC / "precision_pie.png", bbox_inches="tight"); plt.close(fig)
    results["precision_pie"] = {"buckets": tot,
                                "components": [{"name": c[0], "pct": c[1], "bucket": c[2]} for c in comp]}
    print(f"   buckets: {tot}")
    print("   wrote precision_pie.png")


if __name__ == "__main__":
    experiment_accuracy()
    experiment_convergence()
    experiment_roofline()
    experiment_speedup()
    experiment_tradeoff()
    experiment_precision_pie()
    (DOC / "emu_results.json").write_text(json.dumps(results, indent=2))
    print("\nAll experiments done. Results -> document/emu_results.json")
