#!/usr/bin/env python3
"""
reservoir_operator.py
=====================
Port of the synthetic n_u = 2 (pressure-saturation) reservoir operator built by
`src/hypre_cusolver_mixed_cpr.cu :: build_host_system`, in pure numpy/scipy so
the emulation and precision experiments here operate on *exactly* the same
matrix the GB200 solver benchmarks.

The operator mimics a fully-implicit two-field discretisation on a side^3 grid:
per cell c there is a pressure unknown (row 2c) and a saturation unknown (row
2c+1). Off-diagonal couplings come from a 6-point (2 per axis) stencil with a
heterogeneous transmissibility t_ij; the 2x2 cell block carries the elliptic
pressure diagonal and the local saturation dynamics. This reproduces the
ill-conditioned, block-structured A of the paper.

Everything is FP64 here; the emulation module downcasts on demand.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _cid(i, j, k, s):
    return (k * s + j) * s + i


def _perm_factor(i, j, k, sigma):
    x = np.sin(0.173 * (i + 1)) + np.cos(0.119 * (j + 2)) + np.sin(0.071 * (k + 3))
    return np.exp(0.25 * sigma * x)


def build_reservoir_system(side: int, sigma: float = 1.5):
    """Return (A, b, xtrue, Ap) mirroring build_host_system in the .cu.

    A    : (2*cells) x (2*cells) CSR FP64 full two-field operator
    b    : RHS s.t. A xtrue = b
    xtrue: manufactured solution
    Ap   : cells x cells CSR pressure operator (stage-1 AMG target, R W A C)
    """
    cells = side ** 3
    n = 2 * cells
    di = (1, -1, 0, 0, 0, 0)
    dj = (0, 0, 1, -1, 0, 0)
    dk = (0, 0, 0, 0, 1, -1)

    rows, cols, vals = [], [], []      # full operator triplets
    prows, pcols, pvals = [], [], []   # pressure operator triplets
    # dinv (2x2 block inverse) per cell for the stage-2 smoother, if needed
    dinv = np.zeros((cells, 4), dtype=np.float64)

    for k in range(side):
        for j in range(side):
            for i in range(side):
                c = _cid(i, j, k, side)
                p, q = 2 * c, 2 * c + 1
                pdiag, sdiag, ps, sp_ = 1e-3, 2.0, 0.05, -0.02
                for m in range(6):
                    ni, nj, nk = i + di[m], j + dj[m], k + dk[m]
                    if ni < 0 or nj < 0 or nk < 0 or ni >= side or nj >= side or nk >= side:
                        continue
                    cn = _cid(ni, nj, nk, side)
                    t = 0.5 * (_perm_factor(i, j, k, sigma) + _perm_factor(ni, nj, nk, sigma))
                    pdiag += t
                    sdiag += 0.04 * t
                    rows.append(p); cols.append(2 * cn); vals.append(-t)
                    rows.append(q); cols.append(2 * cn + 1); vals.append(-0.04 * t)
                    prows.append(c); pcols.append(cn); pvals.append(-t)
                # cell block
                rows.append(p); cols.append(p); vals.append(pdiag)
                rows.append(p); cols.append(q); vals.append(ps)
                rows.append(q); cols.append(p); vals.append(sp_)
                rows.append(q); cols.append(q); vals.append(sdiag)
                prows.append(c); pcols.append(c); pvals.append(pdiag)
                det = pdiag * sdiag - ps * sp_
                dinv[c] = (sdiag / det, -ps / det, -sp_ / det, pdiag / det)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)).astype(np.float64)
    Ap = sp.csr_matrix((pvals, (prows, pcols)), shape=(cells, cells)).astype(np.float64)

    xtrue = np.empty(n, dtype=np.float64)
    cidx = np.arange(cells)
    xtrue[0::2] = 1.0 + 0.01 * np.sin(0.01 * cidx)
    xtrue[1::2] = 0.2 + 0.01 * np.cos(0.013 * cidx)
    b = A @ xtrue
    return A, b, xtrue, Ap, dinv


def condition_estimate(A, k=6):
    """Cheap 2-norm condition-number *estimate* via extremal singular values
    (ARPACK). Returns (sigma_max, sigma_min, cond). Intended for small grids."""
    from scipy.sparse.linalg import svds
    smax = svds(A, k=1, which="LM", return_singular_vectors=False)[0]
    try:
        smin = svds(A, k=1, which="SM", return_singular_vectors=False)[0]
    except Exception:
        smin = np.nan
    return smax, smin, smax / smin if smin else np.nan


if __name__ == "__main__":
    for side in (8, 16, 32):
        A, b, xtrue, Ap, dinv = build_reservoir_system(side)
        print(f"side={side:3d}  n={A.shape[0]:8d}  nnz(A)={A.nnz:9d}  nnz(Ap)={Ap.nnz:9d}")
