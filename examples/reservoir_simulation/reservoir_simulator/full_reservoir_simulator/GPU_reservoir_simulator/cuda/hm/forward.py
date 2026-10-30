# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# forward.py — pluggable forward operators for history matching.
#
# Both implement the same contract  G : (Ns, Ne) ln-perm ensemble → (Nd, Ne)
# simulated-data ensemble, so the α-REKI driver is agnostic to which one runs:
#
#   * CudaForward      — runs the accelerated NVRS-CUDA executable per ensemble
#                        member (writes a PERMFILE + deck, launches nvrs_cuda,
#                        parses RSM_FVM.csv).  Members fan out across GPUs.
#   * SurrogateForward — evaluates a trained PyTorch surrogate (hm/surrogate_model)
#                        that maps perm/poro fields → well-rate time series.
#
# This realises the goal: an accelerated simulator plugged into a Bayesian
# history-matching loop, with the forward being *either* the CUDA executable
# *or* the neural surrogate — same interface, same observed-data layout.
from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# --------------------------------------------------------------------------
# RSM (well-rate) parsing — shared observed-data layout for obs and sim.
# --------------------------------------------------------------------------
def parse_rsm(csv_path, meas_types):
    """Read RSM_FVM.csv and return (data_vector, columns, ntimes).

    Selects every column whose header contains one of `meas_types`
    (e.g. 'WOPR', 'WWPR') and flattens column-major (all times of column 0,
    then column 1, ...) so the layout is identical for observed and simulated.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    sel = [j for j, h in enumerate(header)
           if any(mt in h for mt in meas_types)]
    arr = np.array([[float(r[j]) for j in sel] for r in data], dtype=np.float64)
    # arr shape (ntimes, ncols) → flatten column-major
    vec = arr.reshape(-1, order="F")
    return vec, [header[j] for j in sel], arr.shape[0]


class ForwardModel:
    """Abstract forward operator."""
    def __call__(self, lnk_ensemble: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------
# CUDA executable forward
# --------------------------------------------------------------------------
class CudaForward(ForwardModel):
    def __init__(self, exe, base_deck, workdir, nx, ny, nz, kmin, kmax,
                 meas_types=("WOPR", "WWPR"), n_gpus=1, ranks_per_member=1,
                 hypre_lib=None):
        self.exe = os.path.abspath(exe)
        with open(base_deck) as f:
            self.base_text = f.read()
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.kmin, self.kmax = kmin, kmax
        self.meas_types = meas_types
        self.n_gpus = max(1, n_gpus)
        self.ranks = ranks_per_member
        self.hypre_lib = hypre_lib
        self.nd = None       # inferred on first run

    def _deck_for(self, permfile, outdir):
        # Strip any existing PERMFILE/OUTDIR/AUTOPLOT/MAKE* then append overrides.
        keep = []
        drop = ("PERMFILE", "OUTDIR", "AUTOPLOT", "MAKEVTK", "MAKEGASSMANN",
                "MAKESUMMARY")
        for line in self.base_text.splitlines():
            tok = line.strip().split()
            if tok and tok[0] in drop:
                continue
            keep.append(line)
        keep += [
            f"PERMFILE {permfile}",
            f"OUTDIR {outdir}",
            "AUTOPLOT 0",            # no per-member plotting in the HM loop
            "MAKEVTK 0",
            "MAKEGASSMANN 0",
            "MAKESUMMARY 0",
            "MAKECSV 1",
        ]
        return "\n".join(keep) + "\n"

    def _run_member(self, m, lnk_col):
        gpu = m % self.n_gpus
        mdir = os.path.join(self.workdir, f"mem_{m:04d}")
        os.makedirs(mdir, exist_ok=True)
        permfile = os.path.join(mdir, "perm.in")
        outdir = os.path.join(mdir, "OUT")
        perm = np.clip(np.exp(lnk_col), self.kmin, self.kmax)
        np.savetxt(permfile, perm, fmt="%.6e")
        deckfile = os.path.join(mdir, "run.DATA")
        with open(deckfile, "w") as f:
            f.write(self._deck_for(permfile, outdir))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if self.hypre_lib:
            env["LD_LIBRARY_PATH"] = self.hypre_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        if self.ranks > 1:
            cmd = ["mpirun", "-np", str(self.ranks), self.exe, deckfile]
        else:
            cmd = [self.exe, deckfile]
        subprocess.run(cmd, cwd=mdir, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vec, _, _ = parse_rsm(os.path.join(outdir, "RSM_FVM.csv"), self.meas_types)
        return m, vec

    def __call__(self, lnk_ensemble):
        ne = lnk_ensemble.shape[1]
        results = [None] * ne
        with ThreadPoolExecutor(max_workers=self.n_gpus) as ex:
            futs = [ex.submit(self._run_member, m, lnk_ensemble[:, m])
                    for m in range(ne)]
            for fut in futs:
                m, vec = fut.result()
                results[m] = vec
        self.nd = results[0].shape[0]
        return np.stack(results, axis=1)     # (Nd, Ne)


# --------------------------------------------------------------------------
# Surrogate forward (PyTorch)
# --------------------------------------------------------------------------
class SurrogateForward(ForwardModel):
    def __init__(self, model_path, nx, ny, nz, kmin, kmax, poro_min, poro_max,
                 inj_mask, prod_mask, device="cuda"):
        import torch
        from surrogate_model import FNOForward   # local module
        self.torch = torch
        ckpt = torch.load(model_path, map_location=device)
        self.model = FNOForward(**ckpt["arch"]).to(device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.norm = ckpt["norm"]
        self.device = device
        self.nx, self.ny, self.nz = nx, ny, nz
        self.kmin, self.kmax = kmin, kmax
        self.pmin, self.pmax = poro_min, poro_max
        self.inj = inj_mask.reshape(nz, ny, nx)
        self.prod = prod_mask.reshape(nz, ny, nx)

    def _features(self, lnk_col):
        from surrogate_model import param_features
        return param_features(
            lnk_col, self.nx, self.ny, self.nz, self.kmin, self.kmax,
            self.pmin, self.pmax, self.inj.sum(axis=0), self.prod.sum(axis=0))

    def __call__(self, lnk_ensemble):
        torch = self.torch
        ne = lnk_ensemble.shape[1]
        feats = np.stack([self._features(lnk_ensemble[:, m]) for m in range(ne)])
        xt = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
        xt = (xt - self.norm["x_mean"]) / self.norm["x_std"]
        with torch.no_grad():
            out = self.model(xt).cpu().numpy()          # (Ne, Nd) normalised
        out = out * self.norm["y_std"] + self.norm["y_mean"]
        return out.T                                    # (Nd, Ne)
