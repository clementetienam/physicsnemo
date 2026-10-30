# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# ensemble.py — geostatistical prior ensemble generation for history matching.
# Draws log-normal permeability fields as stationary Gaussian random fields via
# an FFT moving-average (spectral) construction — the same family as the
# reference `fast_gaussian` layer sampler, kept dependency-light (NumPy only).
from __future__ import annotations

import numpy as np


def _grf_2d(nx, ny, corr_len, rng):
    """One unit-variance, zero-mean Gaussian random field on an nx×ny grid with
    Gaussian covariance of correlation length `corr_len` (cells), via FFT."""
    # Covariance kernel on the periodic grid, centred, then its power spectrum.
    ix = np.fft.fftfreq(nx) * nx
    iy = np.fft.fftfreq(ny) * ny
    X, Y = np.meshgrid(ix, iy, indexing="ij")
    cov = np.exp(-(X**2 + Y**2) / (2.0 * corr_len**2))
    psd = np.abs(np.fft.fft2(cov))
    amp = np.sqrt(np.maximum(psd, 0.0))
    white = rng.standard_normal((nx, ny))
    field = np.fft.ifft2(np.fft.fft2(white) * amp).real
    field -= field.mean()
    s = field.std()
    return field / (s + 1e-30)


def generate_lnk_ensemble(nx, ny, nz, ne, ln_mean, ln_std, corr_len, seed=0):
    """Prior ensemble of ln-permeability fields.

    Returns
    -------
    (Ns, Ne) array, Ns = nx*ny*nz, values = ln(k) in natural (C) cell order
    g = (k*ny + j)*nx + i — matching the simulator's PERMFILE ordering.
    """
    rng = np.random.default_rng(seed)
    ns = nx * ny * nz
    ens = np.empty((ns, ne), dtype=np.float64)
    for m in range(ne):
        vol = np.empty((nz, ny, nx))
        for k in range(nz):
            # decorrelate layers a little by jittering the correlation length
            cl = corr_len * (0.8 + 0.4 * rng.random())
            vol[k] = ln_mean + ln_std * _grf_2d(nx, ny, cl, rng).T
        ens[:, m] = vol.reshape(-1)     # C-order (k,j,i) → g
    return ens


def lnk_to_perm(lnk, kmin, kmax):
    """Exponentiate and clip ln-permeability to physical [kmin,kmax] (mD)."""
    k = np.exp(lnk)
    return np.clip(k, kmin, kmax)


def percentiles(ensemble, qs=(10, 50, 90)):
    """P10/P50/P90 (or any percentiles) across the ensemble axis (=1)."""
    return {q: np.percentile(ensemble, q, axis=1) for q in qs}
