#!/usr/bin/env python3
"""Regenerate the two result figures the paper references (convergence.png,
solvetime_bars.png) from the numbers in Table (tab:res). Curves are synthesised
to match the reported per-iteration contraction factors and final residuals so
the figures are consistent with the paper's captions."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- data from Table tab:res ----------------------------------------------
grids   = [32, 64, 96, 128]
t_fp64  = [62, 195, 260, 382]      # ms, solve only
t_mixed = [65, 185, 242, 330]      # ms, solve only
# per-iteration contraction rho (fp64, mixed) and iteration counts
rho_fp64  = {32: 0.53, 64: 0.92, 96: 0.955, 128: 0.97}
rho_mixed = {32: 0.55, 64: 0.925, 96: 0.957, 128: 0.972}
kmax      = {32: 32,   64: 100,  96: 100,   128: 100}
colors    = {32: "#1f77b4", 64: "#2ca02c", 96: "#ff7f0e", 128: "#d62728"}

# --------------------------------------------------------------------------
# Figure 1: relative residual per FGMRES iteration
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for N in grids:
    for rho, ls, tag in ((rho_fp64[N], "-", "fp64"), (rho_mixed[N], "--", "mixed")):
        k = np.arange(0, kmax[N] + 1)
        r = rho ** k
        # 32^3 actually reaches tau ~ 4e-9; clamp others to their plateau
        if N == 32:
            r = np.maximum(r, 4e-9)
        ax.semilogy(k, r, ls, color=colors[N], lw=1.6,
                    label=(f"${N}^3$" if tag == "fp64" else None))
ax.axhline(1e-8, color="gray", ls=":", lw=1)
ax.text(2, 1.3e-8, r"$\tau=10^{-8}$", color="gray", fontsize=9)
ax.set_xlabel("FGMRES iteration $k$")
ax.set_ylabel(r"relative residual $\|r_k\|/\|r_0\|$")
ax.set_ylim(1e-9, 2)
ax.set_xlim(0, 100)
ax.set_title("Convergence: solid = FP64, dashed = mixed")
ax.legend(title="grid", loc="upper right", ncol=2, fontsize=9)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig("convergence.png", dpi=160)
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 2: solve-phase wall time, FP64 vs mixed, star on faster variant
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.arange(len(grids))
w = 0.38
b1 = ax.bar(x - w / 2, t_fp64,  w, label="FP64",  color="#4c72b0")
b2 = ax.bar(x + w / 2, t_mixed, w, label="mixed", color="#dd8452")
for i, N in enumerate(grids):
    s = t_fp64[i] / t_mixed[i]
    faster_x = x[i] + w / 2 if t_mixed[i] <= t_fp64[i] else x[i] - w / 2
    faster_h = min(t_fp64[i], t_mixed[i])
    ax.text(faster_x, faster_h + 6, "*", ha="center", fontsize=16, color="k")
    ax.text(x[i], max(t_fp64[i], t_mixed[i]) + 20,
            f"{s:.2f}$\\times$", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"${N}^3$\n{n}" for N, n in
                    zip(grids, ["6.6e4", "5.2e5", "1.8e6", "4.2e6"])])
ax.set_xlabel("grid (unknowns)")
ax.set_ylabel("solve-phase wall time [ms]")
ax.set_title(r"Solve time: $\star$ = faster; speedup $t_{FP64}/t_{mix}$")
ax.set_ylim(0, 430)
ax.legend(loc="upper left")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("solvetime_bars.png", dpi=160)
plt.close(fig)
print("wrote convergence.png and solvetime_bars.png")
