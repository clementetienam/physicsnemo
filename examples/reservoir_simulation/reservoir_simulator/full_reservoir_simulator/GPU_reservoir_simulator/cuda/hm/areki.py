# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# areki.py — Adaptive Regularised Ensemble Kalman Inversion (α-REKI).
#
# This is the self-contained numerical core of the history-matching module,
# faithfully adapted from the reference NVIDIA PhysicsNeMo pipeline
# (surrogate/inverse/history_matching.py, inversion_operation_misc.py) but with
# no PhysicsNeMo / Hydra / torch dependency — pure NumPy so it runs anywhere and
# plugs into either forward model (the NVRS-CUDA executable or a PyTorch
# surrogate).
#
# Method (Iglesias 2016; Emerick & Reynolds ES-MDA with adaptive tempering):
#   The Bayesian inverse problem  d = G(m) + e,  e ~ N(0, C_D)  is solved by a
#   sequence of ensemble-Kalman analysis steps with inflation factors α_i chosen
#   adaptively so that  Σ_i 1/α_i = 1  (the regularising tempering constraint).
#   At each step the data mismatch statistics set α:
#       Φ_j = ½ (d - G(m_j))ᵀ C_D⁻¹ (d - G(m_j))
#       α = 1 / min( max( N_d/(2 <Φ>),  sqrt(N_d/(2 Var Φ)) ),  1 - Σ 1/α )
#   and the state is updated by the whitened ensemble-Kalman formula
#       m_j ← m_j + C_MD (C_DD + α C_D)⁻¹ ( d + sqrt(α) ξ_j - G(m_j) ),
#   with ξ_j ~ N(0, C_D) the ES-MDA data perturbations.
from __future__ import annotations

import numpy as np


def data_mismatch(sim: np.ndarray, obs: np.ndarray, cdd_inv_diag: np.ndarray):
    """Per-member objective Φ_j = ½ rᵀ C_D⁻¹ r.  sim (Nd,Ne), obs (Nd,1)."""
    r = obs - sim                                   # (Nd, Ne)
    return 0.5 * np.sum(r * r * cdd_inv_diag[:, None], axis=0)   # (Ne,)


def adaptive_alpha(sim, obs, cdd_inv_diag, converged_sum):
    """Adaptive inflation α and its reciprocal (Iglesias adaptive REKI rule)."""
    phi = data_mismatch(sim, obs, cdd_inv_diag)     # (Ne,)
    nd = obs.shape[0]
    phi_mean = float(np.mean(phi))
    phi_var = float(np.var(phi))
    leftt = nd / (2.0 * phi_mean + 1e-30)
    rightt = np.sqrt(nd / (2.0 * phi_var + 1e-30))
    # cap so the running sum Σ 1/α never overshoots 1 (regularising constraint)
    chok = min(max(leftt, rightt), max(1.0 - converged_sum, 1e-8))
    chok = max(chok, 1e-8)
    return 1.0 / chok, chok


def kalman_update(Y, sim, obs, cdd_diag, alpha, perturb):
    """Whitened ensemble-Kalman analysis increment for the state ensemble Y.

    Parameters
    ----------
    Y        : (Ns, Ne)  state ensemble (e.g. ln-permeability per cell)
    sim      : (Nd, Ne)  simulated data for each member, G(m_j)
    obs      : (Nd, 1)   observed data d
    cdd_diag : (Nd,)     measurement-error variances (diagonal C_D)
    alpha    : float     current inflation factor
    perturb  : (Nd, Ne)  data perturbations ξ_j ~ N(0, C_D)

    Returns
    -------
    (Ns, Ne) additive update to Y.
    """
    ne = sim.shape[1]
    s1 = np.sqrt(ne - 1.0)
    Sd = (sim - sim.mean(axis=1, keepdims=True)) / s1     # (Nd, Ne) data anomalies
    Sy = (Y - Y.mean(axis=1, keepdims=True)) / s1         # (Ns, Ne) state anomalies

    inv_sqrt = 1.0 / np.sqrt(cdd_diag)                    # (Nd,) C_D^{-1/2} diag
    # whitened data anomalies  G = Sdᵀ C_D^{-1/2}   (Ne, Nd)
    GDT = Sd.T * inv_sqrt[None, :]
    Cdd_w = GDT.T @ GDT                                   # (Nd, Nd)
    Cyd = Sy @ GDT                                        # (Ns, Nd) cross-cov (whitened)

    X = np.linalg.pinv(Cdd_w + alpha * np.eye(cdd_diag.shape[0]))
    factor = (obs + np.sqrt(alpha) * perturb) - sim       # (Nd, Ne)
    update = Cyd @ X @ (inv_sqrt[:, None] * factor)       # (Ns, Ne)
    return np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)


def sample_perturbations(cdd_diag, ne, rng):
    """Draw ES-MDA data perturbations ξ ~ N(0, C_D), C_D diagonal."""
    return np.sqrt(cdd_diag)[:, None] * rng.standard_normal((cdd_diag.shape[0], ne))


def rmse(sim_mean, obs):
    """Data-space RMSE of the ensemble mean prediction."""
    return float(np.sqrt(np.mean((sim_mean - obs) ** 2)))
