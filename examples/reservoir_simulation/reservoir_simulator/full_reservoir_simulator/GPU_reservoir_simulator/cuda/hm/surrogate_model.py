# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# surrogate_model.py — a compact Fourier Neural Operator (FNO) forward surrogate,
# the method used by the reference training pipeline (surrogate/train.py, an
# FNO/PINO black-oil operator) distilled to a self-contained PyTorch module.
#
# It maps the reservoir parameter field (per-layer ln-permeability + porosity +
# well masks, as image channels) to the observed well-rate data vector — exactly
# the quantity α-REKI assimilates — so a trained surrogate is a drop-in forward
# for the history-matching loop, replacing the CUDA executable when a fast,
# differentiable, batched forward is wanted.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def param_features(lnk_col, nx, ny, nz, kmin, kmax, pmin, pmax, inj2d, prod2d):
    """Build the FNO input image (C, ny, nx) for one member: per-layer
    ln-permeability + porosity channels, plus injector/producer masks.  Shared
    by training (hm/train_surrogate) and inference (hm/forward.SurrogateForward)
    so the channel layout can never drift between them."""
    vol = lnk_col.reshape(nz, ny, nx)
    poro = pmin + (pmax - pmin) * (
        (np.exp(vol) - kmin) / (kmax - kmin + 1e-30))
    chans = [vol[k] for k in range(nz)] + [poro[k] for k in range(nz)]
    chans.append(inj2d)
    chans.append(prod2d)
    return np.stack(chans, axis=0)      # (2*nz + 2, ny, nx)


def n_input_channels(nz):
    return 2 * nz + 2


class SpectralConv2d(nn.Module):
    """2-D spectral convolution: pointwise-linear on the lowest Fourier modes."""
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        m1 = min(self.m1, h);  m2 = min(self.m2, w // 2 + 1)
        out[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(h, w))


class FNOForward(nn.Module):
    """FNO encoder over the parameter image + global head → data vector."""
    def __init__(self, in_channels, out_dim, width=32, modes=12, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.specs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.act = nn.GELU()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_dim))

    def forward(self, x):
        x = self.lift(x)
        for sp, w in zip(self.specs, self.ws):
            x = self.act(sp(x) + w(x))
        return self.head(x)
