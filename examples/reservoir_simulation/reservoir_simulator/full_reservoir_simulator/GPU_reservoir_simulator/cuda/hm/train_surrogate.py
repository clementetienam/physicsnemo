# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# train_surrogate.py — build a PyTorch FNO forward surrogate for the reservoir.
#
# Pipeline (the method of the reference surrogate/train.py, self-contained):
#   1. Sample a training ensemble of geostatistical ln-permeability fields.
#   2. Run the accelerated NVRS-CUDA executable on each (fanning out over GPUs)
#      to get the well-rate data vector — i.e. the labels come from the *same*
#      accelerated simulator the history matcher would otherwise call.
#   3. Assemble (parameter-image → data-vector) pairs, normalise, and train the
#      FNO (hm/surrogate_model.FNOForward).
#   4. Save a checkpoint {arch, state, norm} that hm/forward.SurrogateForward
#      loads to serve as the fast, batched, differentiable HM forward.
#
# Launched by scripts/history_match.sh (stage: train-surrogate) or directly:
#   python hm/train_surrogate.py --config hm/config.yaml
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

import ensemble as ens
from forward import CudaForward
from history_match import well_masks_from_deck


def generate_dataset(cfg):
    grid = cfg["grid"]; nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    ntrain = cfg["surrogate"]["n_train"]
    lnk = ens.generate_lnk_ensemble(
        nx, ny, nz, ntrain, cfg["ln_mean"], cfg["ln_std"], cfg["corr_len"],
        seed=cfg.get("seed", 0) + 12345)
    lo, hi = np.log(cfg["kmin"]), np.log(cfg["kmax"])
    lnk = np.clip(lnk, lo, hi)

    fwd = CudaForward(
        cfg["exe"], cfg["deck"], os.path.join(cfg["workdir"], "train"),
        nx, ny, nz, cfg["kmin"], cfg["kmax"], tuple(cfg["meas_types"]),
        n_gpus=cfg.get("n_gpus", 1), ranks_per_member=cfg.get("ranks_per_member", 1),
        hypre_lib=cfg.get("hypre_lib"))
    print(f"[train] running NVRS-CUDA on {ntrain} members to label the dataset ...")
    sim = fwd(lnk)                       # (Nd, ntrain)
    return lnk, sim


def build_tensors(cfg, lnk, sim):
    from surrogate_model import param_features
    grid = cfg["grid"]; nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    inj, prod = well_masks_from_deck(cfg["deck"], nx, ny, nz)
    inj2d = inj.reshape(nz, ny, nx).sum(0)
    prod2d = prod.reshape(nz, ny, nx).sum(0)
    X = np.stack([
        param_features(lnk[:, m], nx, ny, nz, cfg["kmin"], cfg["kmax"],
                       cfg["poro_min"], cfg["poro_max"], inj2d, prod2d)
        for m in range(lnk.shape[1])])
    Y = sim.T                            # (ntrain, Nd)
    return X.astype(np.float32), Y.astype(np.float32)


def train(cfg):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from surrogate_model import FNOForward, n_input_channels

    device = cfg.get("device", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    lnk, sim = generate_dataset(cfg)
    X, Y = build_tensors(cfg, lnk, sim)

    x_mean = X.mean((0, 2, 3), keepdims=True)
    x_std = X.std((0, 2, 3), keepdims=True) + 1e-6
    y_mean = Y.mean(0, keepdims=True); y_std = Y.std(0, keepdims=True) + 1e-6
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    ds = TensorDataset(torch.as_tensor(Xn), torch.as_tensor(Yn))
    n_val = max(1, int(0.1 * len(ds)))
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    dl = DataLoader(tr, batch_size=cfg["surrogate"].get("batch", 8), shuffle=True)
    vdl = DataLoader(va, batch_size=cfg["surrogate"].get("batch", 8))

    nz = cfg["grid"]["nz"]
    arch = dict(in_channels=n_input_channels(nz), out_dim=Y.shape[1],
                width=cfg["surrogate"].get("width", 32),
                modes=cfg["surrogate"].get("modes", 12),
                n_layers=cfg["surrogate"].get("layers", 4))
    model = FNOForward(**arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["surrogate"].get("lr", 1e-3))
    lossf = torch.nn.MSELoss()
    epochs = cfg["surrogate"].get("epochs", 200)

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        if (ep + 1) % max(1, epochs // 20) == 0:
            model.eval()
            with torch.no_grad():
                vloss = np.mean([lossf(model(xb.to(device)), yb.to(device)).item()
                                 for xb, yb in vdl])
            print(f"[train] epoch {ep+1}/{epochs}  train={loss.item():.4e}  val={vloss:.4e}")

    out = cfg["surrogate"]["model_out"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({
        "arch": arch, "state": model.state_dict(),
        "norm": {"x_mean": torch.as_tensor(x_mean, device=device),
                 "x_std": torch.as_tensor(x_std, device=device),
                 "y_mean": torch.as_tensor(y_mean, device=device),
                 "y_std": torch.as_tensor(y_std, device=device)},
    }, out)
    print(f"[train] surrogate saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="hm/config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
