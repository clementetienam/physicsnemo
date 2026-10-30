#!/usr/bin/env python3
"""
fgmres_cpr.py
=============
Self-contained flexible GMRES with a two-stage CPR preconditioner, matching the
structure of `src/hypre_cusolver_mixed_cpr.cu`:

  * outer FGMRES with incremental Givens-QR and CGS2 orthogonalisation (FP64),
  * stage-1: pressure solve on Ap (here a direct factorisation, a *strong* stand
    -in for one AMG V-cycle) applied in a chosen precision,
  * stage-2: analytic 2x2 block-Jacobi correction.

The operator application  w = A z  and the true residual  r = b - A x  are routed
through a pluggable `matvec`, so we can compare:

  * fp64      : operator in FP64                 (baseline)
  * mixed     : operator FP64, preconditioner FP32
  * emu-mixed : operator EMULATED FP64 (Ozaki-2), preconditioner FP32
  * fp32-op   : operator in plain FP32           (shows why naive narrowing fails)

This isolates exactly the question: does emulating the FP64 operator (SpMV /
residual) from FP32 slices reproduce true-FP64 mixed-precision convergence?
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ozaki_emulation import _slice_sparse, emulated_spmv, to_fp32, _pairs


class CPRPrecond:
    """Two-stage CPR: FP32 (or FP64) pressure direct solve + 2x2 block-Jacobi."""

    def __init__(self, A, Ap, dinv, precond_dtype=np.float32):
        self.A = A.tocsr()
        self.cells = Ap.shape[0]
        self.dinv = dinv.astype(precond_dtype)
        self.pdt = precond_dtype
        # factor the pressure operator in the preconditioner precision
        Ap_p = Ap.astype(precond_dtype if precond_dtype == np.float64 else np.float64)
        # scipy splu needs fp64/fp32 CSC; emulate fp32 by casting values to fp32 then back
        if precond_dtype == np.float32:
            Ap_p = sp.csc_matrix(Ap).astype(np.float32).astype(np.float64)
        self.lu = spla.splu(sp.csc_matrix(Ap_p))

    def apply(self, r):
        # stage 1: pressure solve, scatter into z
        rp = r[0::2].astype(self.pdt)
        dp = self.lu.solve(rp.astype(np.float64)).astype(self.pdt)
        z = np.zeros_like(r)
        z[0::2] = dp.astype(np.float64)
        # stage 2: block-Jacobi correction on r - A z
        Az = self.A @ z
        r0 = (r[0::2] - Az[0::2]).astype(self.pdt)
        r1 = (r[1::2] - Az[1::2]).astype(self.pdt)
        d = self.dinv
        z[0::2] += (d[:, 0] * r0 + d[:, 1] * r1).astype(np.float64)
        z[1::2] += (d[:, 2] * r0 + d[:, 3] * r1).astype(np.float64)
        return z


def make_matvec(A, mode, s=2, fmt="fp32"):
    """Return a callable y=matvec(x) applying A in the requested precision."""
    Acsr = A.tocsr()
    if mode == "fp64":
        return lambda x: Acsr @ x
    if mode == "fp32-op":
        Al = sp.csr_matrix((Acsr.data.astype(np.float32), Acsr.indices, Acsr.indptr),
                           shape=Acsr.shape)
        return lambda x: (Al @ x.astype(np.float32)).astype(np.float64)
    if mode in ("emu", "emu-mixed", "ozaki"):
        A_slices = _slice_sparse(Acsr, s, fmt)
        return lambda x: emulated_spmv(Acsr, x, s, fmt, A_slices=A_slices)
    raise ValueError(mode)


def fgmres_cpr(A, b, matvec, precond, restart=30, maxiter=120, tol=1e-10, x0=None):
    """Flexible GMRES with Givens-QR + CGS2. Returns (x, rel_history)."""
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        bnorm = 1.0
    hist = []
    total = 0
    r = b - matvec(x)
    rel = np.linalg.norm(r) / bnorm
    hist.append(rel)
    while total < maxiter and rel > tol:
        m = min(restart, maxiter - total)
        V = np.zeros((n, m + 1))
        Z = np.zeros((n, m))
        H = np.zeros((m + 1, m))
        cs = np.zeros(m); sn = np.zeros(m)
        g = np.zeros(m + 1)
        beta = np.linalg.norm(r)
        V[:, 0] = r / beta
        g[0] = beta
        used = 0
        for j in range(m):
            Z[:, j] = precond.apply(V[:, j])
            w = matvec(Z[:, j])
            # CGS2 (two classical GS passes)
            for _ in range(2):
                h = V[:, :j + 1].T @ w
                w = w - V[:, :j + 1] @ h
                H[:j + 1, j] += h
            hn = np.linalg.norm(w)
            H[j + 1, j] = hn
            if hn > 1e-300:
                V[:, j + 1] = w / hn
            # apply previous Givens
            for i in range(j):
                t0 = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t0
            denom = np.hypot(H[j, j], H[j + 1, j])
            denom = denom if denom > 1e-300 else 1e-300
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom
            H[j, j] = denom
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            rel = abs(g[j + 1]) / bnorm
            used = j + 1
            total += 1
            hist.append(rel)
            if rel <= tol or total >= maxiter or hn <= 1e-300:
                break
        # back-substitution
        y = np.zeros(used)
        for i in range(used - 1, -1, -1):
            ssum = g[i] - H[i, i + 1:used] @ y[i + 1:used]
            y[i] = ssum / H[i, i] if abs(H[i, i]) > 1e-300 else 0.0
        x = x + Z[:, :used] @ y
        r = b - matvec(x)         # true residual restart (routed through matvec)
        rel = np.linalg.norm(r) / bnorm
    return x, np.array(hist)
