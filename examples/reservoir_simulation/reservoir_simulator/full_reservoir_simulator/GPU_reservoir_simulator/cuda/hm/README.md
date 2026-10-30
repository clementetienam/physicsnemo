# `hm/`: Bayesian History Matching around NVRS-CUDA (α-REKI)

> Install, run commands, measured results and the paper are in the [top-level README](../../../../README.md).

Plugs the accelerated simulator into a Bayesian inverse workflow: estimate the
reservoir permeability field (and its uncertainty) from observed well data,
using **Adaptive Regularised Ensemble Kalman Inversion (α-REKI)**: faithfully
adapted from the reference NVIDIA PhysicsNeMo pipeline (`surrogate/inverse/`),
but self-contained (NumPy core; PyTorch only for the surrogate).

The forward model is **pluggable**: the exact goal you asked for:

```
                          ┌─────────────────────────────┐
   prior ensemble  ──────▶│   forward G(m)              │──▶ simulated well data
   (perm fields)          │   • CudaForward  (nvrs_cuda)│
        ▲                 │   • SurrogateForward (FNO)  │
        │                 └─────────────────────────────┘
        │   α-REKI update (adaptive α, whitened ensemble Kalman)   │
        └─────────────────────◀───────────────────────────────────┘
                          observed well data + C_D
```

## Files
| file | role |
|---|---|
| `areki.py` | adaptive-α + whitened ensemble-Kalman update (pure NumPy) |
| `ensemble.py` | geostatistical log-normal permeability prior (FFT Gaussian fields) |
| `forward.py` | `CudaForward` (runs `nvrs_cuda` per member) and `SurrogateForward` (PyTorch) |
| `surrogate_model.py` | compact FNO forward surrogate + shared feature builder |
| `train_surrogate.py` | label data with NVRS-CUDA → train the FNO |
| `history_match.py` | α-REKI driver: prior → posterior, P10/P50/P90, plots |
| `run_history_match.py` | CLI (`--make-truth`, `--forward cuda|surrogate`) |
| `config.yaml` | all settings |

## Quick start
```bash
# forward = the CUDA executable
./scripts/history_match.sh cuda

# forward = a trained surrogate (train it first from NVRS-CUDA runs)
./scripts/history_match.sh surrogate --train
```
Or step by step:
```bash
python hm/run_history_match.py --config hm/config.yaml --make-truth      # observed data
python hm/train_surrogate.py   --config hm/config.yaml                   # (surrogate only)
python hm/run_history_match.py --config hm/config.yaml --forward cuda    # assimilate
```

## Method (see `docs/` §14 for the full derivation)
Solve the Bayesian inverse problem `d = G(m) + e`, `e ~ N(0, C_D)` by a sequence
of ensemble-Kalman analyses with inflation factors `α_i` chosen adaptively so
that `Σ_i 1/α_i = 1` (regularising tempering). Each step:
1. **α** from the data-mismatch statistics `Φ_j = ½(d−G(m_j))ᵀC_D⁻¹(d−G(m_j))`:
   `α = 1/min(max(N_d/2⟨Φ⟩, √(N_d/2·VarΦ)), 1−Σ1/α)`.
2. **Update** `m_j ← m_j + C_MD(C_DD+αC_D)⁻¹(d + √α·ξ_j − G(m_j))`, `ξ_j~N(0,C_D)`.

Outputs land in `HM_RESULTS/`: `posterior.npz` (perm + P10/P50/P90), plus
`convergence.png`, `data_match.png`, `perm_percentiles.png`.

## Requirements
`numpy`, `scipy`, `matplotlib`, `pyyaml` for the core + CUDA forward; add
`torch` for the surrogate. The CUDA forward needs a built `build/nvrs_cuda` and
(for `--make-truth`/labels) at least one GPU; members fan out across `n_gpus`.

Build that binary as `../README.md` describes. The stage-1 multigrid defaults
to AmgX, which matters here more than elsewhere: an ensemble runs the forward
model once per member per iteration, so the ~4.5× the solver gains is ~4.5× off
the whole history match. Build AmgX with `../benchmark/install_amgx.sh` and
source its `amgx_env.sh` before `make`. Without it the forward model still runs,
on HYPRE, and the banner says which one it used.

## Forward-model baseline used by history matching

As of the Norne forward model uses OPM-style schedule interval
semantics, history RESV conversion, active zero-rate shutdown, metric gas
conversion, PVT regions, Carlson gas hysteresis and phase-specific
`GCONINJE RATE` ceilings. The AmgX and HYPRE forward trajectories overlay;
backend choice changes cost, not the physical target presented to alpha-REKI.
See [`../docs/manual.pdf`](../../../../docs/manual.pdf), Chapter 10, before generating a
new truth or observation set. Do not mix legacy 45-second Norne outputs with
the current fidelity configuration.

