#!/usr/bin/env python3
"""
ozaki_emulation.py
==================
A faithful *numerical model* of FP64 emulation from low-precision slices --- the
Ozaki scheme (Ozaki, Ogita, Rump, Oishi 2012) and its integer/tensor-core
realisation (Ootomo, Ozaki, Yokota 2024) --- used to answer the question posed
in the task:

    "Emulate all the FP64 compute (RAP, residual, SpMV). Is emulated-FP64
     mixed precision as good as true FP64 + FP32 mixed precision?"

Model of the scheme
-------------------
Split each FP64 operand `a` into `s` low-precision slices a = a0 + a1 + ... :

    r <- a
    for m in 0..s-1:  a_m <- fl_low(r);  r <- r - a_m

Each slice a_m is representable in the low format (FP32 or TF32). A product a*b
is the sum of slice products  a_i * b_j.  The essential Ozaki fact is an
*error-free transformation*: the LOW-precision units perform the multiplies, but
the partial products are accumulated in a WIDE accumulator (int32 on the integer
tensor core, or FP32/FP64) so that the accumulation itself is (near) exact. The
result accuracy is therefore set by how many bits the *slices* collectively
capture, not by the accumulator: s FP32 slices capture ~ s*24 bits.

    s = 1  ->  ~24 bits  (== plain FP32)
    s = 2  ->  ~48 bits  ("Ozaki-2"; near FP64, whose mantissa is 53 bits)
    s = 3  ->  ~72 bits  (beyond FP64)

Cost model: keeping the triangular set {(i,j): i+j <= s-1} uses s(s+1)/2
low-precision GEMMs, i.e. 1 (s=1), 3 (s=2), 6 (s=3) partial products per output.

This module provides emulated GEMM, SpMV, the Galerkin RAP triple product, and
the true residual b - A x, plus accuracy measurements against the FP64 truth.
The accumulation of slice products is done in FP64 to model the exact wide
accumulate of the integer tensor core.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# --- low-precision formats -------------------------------------------------
def to_fp32(x):
    return x.astype(np.float32).astype(np.float64)


def to_tf32(x):
    """Round to TF32 (8-bit exp, 10-bit mantissa) by masking the FP32 mantissa
    to 10 bits. Value kept in an FP64 container for arithmetic."""
    f = x.astype(np.float32)
    u = f.view(np.uint32)
    # TF32 keeps 10 mantissa bits -> clear the low 13 of FP32's 23, round-to-nearest
    u = (u + np.uint32(0x1000)) & np.uint32(0xFFFFE000)
    return u.view(np.float32).astype(np.float64)


_ROUND = {"fp32": to_fp32, "tf32": to_tf32}


def split_slices(a, s, fmt="fp32"):
    """Split FP64 array `a` into `s` low-precision slices (residual splitting).
    Returns a list of FP64-container arrays, each representable in `fmt`."""
    rnd = _ROUND[fmt]
    r = a.astype(np.float64).copy()
    out = []
    for _ in range(s):
        hi = rnd(r)
        out.append(hi)
        r = r - hi
    return out


def _pairs(s):
    """Triangular set of slice-index pairs kept by an s-slice Ozaki product."""
    return [(i, j) for i in range(s) for j in range(s) if i + j <= s - 1]


# --- dense GEMM (for the RAP / Galerkin triple product demonstration) ------
def emulated_gemm(A, B, s, fmt="fp32"):
    """Emulate C = A @ B (dense) with s-slice Ozaki. Partial products summed in
    FP64 (models the wide accumulator)."""
    As = split_slices(A.ravel(), s, fmt)
    Bs = split_slices(B.ravel(), s, fmt)
    As = [a.reshape(A.shape) for a in As]
    Bs = [b.reshape(B.shape) for b in Bs]
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    for (i, j) in _pairs(s):
        C += As[i] @ Bs[j]
    return C


# --- sparse SpMV -----------------------------------------------------------
def _slice_sparse(A_csr, s, fmt):
    """Split a CSR matrix's values into s low-precision slices, sharing structure."""
    slices = split_slices(A_csr.data, s, fmt)
    return [sp.csr_matrix((d, A_csr.indices, A_csr.indptr), shape=A_csr.shape) for d in slices]


def emulated_spmv(A_csr, x, s, fmt="fp32", A_slices=None):
    """Emulate y = A @ x with s-slice Ozaki. A_slices may be precomputed."""
    if A_slices is None:
        A_slices = _slice_sparse(A_csr, s, fmt)
    xs = split_slices(x, s, fmt)
    y = np.zeros(A_csr.shape[0], dtype=np.float64)
    for (i, j) in _pairs(s):
        y += A_slices[i] @ xs[j]
    return y


def naive_low_spmv(A_csr, x, fmt="fp32"):
    """Plain low-precision SpMV: cast A and x to `fmt`, multiply, accumulate in
    that same low format (the *wrong* way -- shown for contrast)."""
    rnd = _ROUND[fmt]
    Al = sp.csr_matrix((rnd(A_csr.data).astype(np.float32),
                        A_csr.indices, A_csr.indptr), shape=A_csr.shape)
    xl = rnd(x).astype(np.float32)
    return (Al @ xl).astype(np.float64)


# --- Galerkin RAP triple product (sparse) ----------------------------------
def emulated_rap(R, A, P, s, fmt="fp32"):
    """Emulate the AMG Galerkin coarse operator A_c = R A P (all sparse) with
    s-slice Ozaki on each of the two sparse products. Accumulate in FP64."""
    Rs = _slice_sparse(R.tocsr(), s, fmt)
    As = _slice_sparse(A.tocsr(), s, fmt)
    Ps = _slice_sparse(P.tocsr(), s, fmt)
    # AP first
    AP = None
    for (i, j) in _pairs(s):
        term = As[i] @ Ps[j]
        AP = term if AP is None else AP + term
    APs = _slice_sparse(AP.tocsr(), s, fmt)
    Ac = None
    for (i, j) in _pairs(s):
        term = Rs[i] @ APs[j]
        Ac = term if Ac is None else Ac + term
    return Ac


# --- accuracy helpers ------------------------------------------------------
def rel_err(approx, truth):
    d = np.linalg.norm(approx - truth)
    n = np.linalg.norm(truth)
    return d / n if n else d


def ozaki_flops_factor(s):
    """Relative FLOP cost of an s-slice Ozaki product vs one native product."""
    return len(_pairs(s))


if __name__ == "__main__":
    # quick self-test on a random ill-scaled matrix
    rng = np.random.default_rng(0)
    n = 400
    A = rng.standard_normal((n, n)) * 10.0 ** rng.integers(-4, 4, (n, n))
    x = rng.standard_normal(n)
    truth = A @ x
    Acsr = sp.csr_matrix(A)
    print("format  slices  rel-err vs FP64      cost(GEMMs)")
    for fmt in ("fp32", "tf32"):
        for s in (1, 2, 3):
            y = emulated_spmv(Acsr, x, s, fmt)
            print(f"{fmt:5s}   s={s}     {rel_err(y, truth):.3e}        {ozaki_flops_factor(s)}")
    print("naive fp32 (low accumulate):", f"{rel_err(naive_low_spmv(Acsr, x), truth):.3e}")
