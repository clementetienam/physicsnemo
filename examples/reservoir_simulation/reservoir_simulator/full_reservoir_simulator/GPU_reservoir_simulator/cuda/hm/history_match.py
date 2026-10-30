# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# history_match.py — Bayesian history matching driver (α-REKI) around the
# accelerated NVRS-CUDA forward (or a trained surrogate).  Produces a posterior
# permeability ensemble with P10/P50/P90, a data-match plot and a convergence
# trace.
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import areki
import ensemble as ens
from forward import CudaForward, SurrogateForward, parse_rsm


def well_masks_from_deck(deck_path, nx, ny, nz):
    """Parse the deck's WELLS block into injector/producer cell masks."""
    inj = np.zeros(nx * ny * nz)
    prod = np.zeros(nx * ny * nz)
    in_wells = False
    with open(deck_path) as f:
        for raw in f:
            line = raw.split("--")[0].strip()
            if not line:
                continue
            if line.upper().startswith("WELLS"):
                in_wells = True
                continue
            if in_wells:
                if line == "/":
                    break
                tk = line.split()
                if len(tk) < 4:
                    continue
                i, j, typ = int(tk[0]) - 1, int(tk[1]) - 1, int(tk[2])
                for k in range(nz):
                    c = (k * ny + j) * nx + i
                    (inj if typ > 0 else prod)[c] = 1.0
    return inj, prod


def build_forward(cfg, obs_len):
    grid = cfg["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    if cfg["forward"] == "surrogate":
        inj, prod = well_masks_from_deck(cfg["deck"], nx, ny, nz)
        return SurrogateForward(
            cfg["surrogate_model"], nx, ny, nz,
            cfg["kmin"], cfg["kmax"], cfg["poro_min"], cfg["poro_max"],
            inj, prod, device=cfg.get("device", "cuda"))
    return CudaForward(
        cfg["exe"], cfg["deck"], cfg["workdir"], nx, ny, nz,
        cfg["kmin"], cfg["kmax"], tuple(cfg["meas_types"]),
        n_gpus=cfg.get("n_gpus", 1), ranks_per_member=cfg.get("ranks_per_member", 1),
        hypre_lib=cfg.get("hypre_lib"))


def run(cfg):
    grid = cfg["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    outdir = cfg.get("outdir", "HM_RESULTS")
    os.makedirs(outdir, exist_ok=True)

    # --- observed data + measurement covariance ---
    obs_vec, cols, ntimes = parse_rsm(cfg["truth_csv"], tuple(cfg["meas_types"]))
    obs = obs_vec.reshape(-1, 1)
    rel = cfg.get("noise_rel", 0.05)
    floor = cfg.get("noise_floor", 1.0)
    cdd_diag = (rel * np.abs(obs_vec)) ** 2 + floor ** 2
    cdd_inv = 1.0 / cdd_diag
    nd = obs.shape[0]
    print(f"[HM] observed data: {nd} measurements "
          f"({len(cols)} well series x {ntimes} times)")

    forward = build_forward(cfg, nd)

    # --- prior ensemble ---
    ne = cfg["Ne"]
    lnk = ens.generate_lnk_ensemble(
        nx, ny, nz, ne, cfg["ln_mean"], cfg["ln_std"], cfg["corr_len"],
        seed=cfg.get("seed", 0))
    lo, hi = np.log(cfg["kmin"]), np.log(cfg["kmax"])
    lnk = np.clip(lnk, lo, hi)

    rng = np.random.default_rng(cfg.get("seed", 0) + 777)
    termm = cfg.get("Termm", 10)
    converged, alphas, rmse_hist = 0.0, [], []

    for it in range(termm):
        sim = forward(lnk)                                   # (Nd, Ne)
        alpha, chok = areki.adaptive_alpha(sim, obs, cdd_inv, converged)
        pert = areki.sample_perturbations(cdd_diag, ne, rng)
        upd = areki.kalman_update(lnk, sim, obs, cdd_diag, alpha, pert)
        lnk = np.clip(lnk + upd, lo, hi)

        r = areki.rmse(sim.mean(axis=1, keepdims=True), obs)
        rmse_hist.append(r)
        alphas.append(alpha)
        converged += 1.0 / alpha
        print(f"[HM] iter {it+1}/{termm}  alpha={alpha:9.3f}  "
              f"sum(1/alpha)={converged:.3f}  data-RMSE={r:.4g}")
        if converged >= 1.0 - 1e-6:
            print("[HM] regularising tempering complete (Σ 1/α ≈ 1)")
            break

    # --- posterior forward + stats ---
    sim_post = forward(lnk)
    perm = np.exp(lnk)
    pct = ens.percentiles(perm, (10, 50, 90))
    np.savez(os.path.join(outdir, "posterior.npz"),
             lnk=lnk, perm=perm, P10=pct[10], P50=pct[50], P90=pct[90],
             alphas=np.array(alphas), rmse=np.array(rmse_hist))

    # --- plots ---
    _plot_convergence(rmse_hist, alphas, outdir)
    _plot_datamatch(obs_vec, sim_post, cols, ntimes, outdir)
    _plot_perm_percentiles(pct, nx, ny, nz, outdir)
    print(f"[HM] done → {outdir} (posterior.npz + plots)")
    return lnk, sim_post


def _plot_convergence(rmse_hist, alphas, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(1, len(rmse_hist) + 1), rmse_hist, "-o")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("data-space RMSE")
    ax[0].set_title("α-REKI convergence"); ax[0].set_yscale("log")
    ax[1].plot(range(1, len(alphas) + 1), alphas, "-s", color="C1")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("inflation α")
    ax[1].set_title("adaptive α"); ax[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "convergence.png"), dpi=120)
    plt.close(fig)


def _plot_datamatch(obs_vec, sim, cols, ntimes, outdir):
    n = min(6, len(cols))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    obs2 = obs_vec.reshape(ntimes, -1, order="F")
    sim2 = sim.reshape(ntimes, -1, sim.shape[1], order="F")
    t = np.arange(ntimes)
    for c in range(n):
        ax = axes.flat[c]
        ax.plot(t, sim2[:, c, :], color="0.7", lw=0.6)
        ax.plot(t, sim2[:, c, :].mean(1), "C0", lw=1.8, label="posterior mean")
        ax.plot(t, obs2[:, c], "k--", lw=1.8, label="observed")
        ax.set_title(cols[c], fontsize=8)
        if c == 0:
            ax.legend(fontsize=7)
    fig.suptitle("History match — posterior vs observed")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "data_match.png"), dpi=120)
    plt.close(fig)


def _plot_perm_percentiles(pct, nx, ny, nz, outdir):
    k = nz // 2
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, q in zip(ax, (10, 50, 90)):
        field = pct[q].reshape(nz, ny, nx)[k]
        im = a.imshow(field, origin="lower", cmap="jet", aspect="auto")
        a.set_title(f"Posterior P{q} perm (layer {k})")
        fig.colorbar(im, ax=a, label="mD")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "perm_percentiles.png"), dpi=120)
    plt.close(fig)
