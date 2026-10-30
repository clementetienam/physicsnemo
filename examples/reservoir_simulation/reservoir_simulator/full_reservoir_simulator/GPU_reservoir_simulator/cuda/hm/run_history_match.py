#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# run_history_match.py — CLI entry for α-REKI Bayesian history matching around
# the NVRS-CUDA forward (or a trained surrogate).
#
#   # 1. synthesise an observed dataset from a "true" reservoir (once)
#   python hm/run_history_match.py --config hm/config.yaml --make-truth
#   # 2. history-match with the CUDA executable as the forward
#   python hm/run_history_match.py --config hm/config.yaml --forward cuda
#   # 3. ... or with a trained surrogate as the forward
#   python hm/run_history_match.py --config hm/config.yaml --forward surrogate
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import yaml

import ensemble as ens
import history_match as hm
from forward import CudaForward


def make_truth(cfg):
    """Run NVRS-CUDA once on a 'true' geostatistical field → observed RSM CSV."""
    grid = cfg["grid"]; nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    lnk = ens.generate_lnk_ensemble(
        nx, ny, nz, 1, cfg["ln_mean"], cfg["ln_std"], cfg["corr_len"],
        seed=cfg.get("truth_seed", 4242))
    lo, hi = np.log(cfg["kmin"]), np.log(cfg["kmax"])
    lnk = np.clip(lnk, lo, hi)
    fwd = CudaForward(
        cfg["exe"], cfg["deck"], os.path.join(cfg["workdir"], "truth"),
        nx, ny, nz, cfg["kmin"], cfg["kmax"], tuple(cfg["meas_types"]),
        n_gpus=cfg.get("n_gpus", 1), ranks_per_member=cfg.get("ranks_per_member", 1),
        hypre_lib=cfg.get("hypre_lib"))
    fwd(lnk)                                     # writes mem_0000/OUT/RSM_FVM.csv
    src = os.path.join(cfg["workdir"], "truth", "mem_0000", "OUT", "RSM_FVM.csv")
    dst = cfg["truth_csv"]
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy(src, dst)
    # keep the true perm for reference / plotting
    np.save(os.path.splitext(dst)[0] + "_perm.npy", np.exp(lnk[:, 0]))
    print(f"[make-truth] observed data → {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="hm/config.yaml")
    ap.add_argument("--forward", choices=["cuda", "surrogate"], default=None,
                    help="override the forward model in the config")
    ap.add_argument("--make-truth", action="store_true",
                    help="generate the observed dataset then exit")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.forward:
        cfg["forward"] = args.forward

    if args.make_truth:
        make_truth(cfg)
        return
    hm.run(cfg)


if __name__ == "__main__":
    main()
