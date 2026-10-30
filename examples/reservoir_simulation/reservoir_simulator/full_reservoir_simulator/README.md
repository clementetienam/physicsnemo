# NVRS-CUDA: A GPU-Native Reservoir Simulator

> Install, run commands, measured results and the paper are in the [top-level README](../README.md).

### From rock physics to Bayesian inversion, with a mixed-precision CPR solver

**Clement Etienam**¹ · **Tsubasa Onishi**¹ · **Oleg Ovcharenko**¹ · **Nick Luiken**¹ · **Issam Said**¹

¹ NVIDIA

---

A CUDA C++ black-oil and compositional reservoir simulator built around a
**mixed-precision CPR** linear solver that scales from a single GPU to
multi-node clusters, wrapped in an **α-REKI Bayesian history-matching** layer
whose forward map is either the simulator itself or a neural surrogate of it.
The multigrid inside CPR comes from **AmgX** (default) or **HYPRE BoomerAMG**,
chosen with one deck keyword, see §5.4.

This document develops the mathematics from first principles and maps every
equation onto the code that implements it.

> **Just want to run it?** See [Quickstart](#quickstart) immediately below.
> **Reviewing the numerics?** Start at [§0](#0-the-one-idea) and read straight through.

---

## Table of contents

| | |
|---|---|
| [0. The one idea](#0-the-one-idea) | Why every design decision follows from one observation |
| [1. Repository map](#1-repository-map) | What lives where |
| [2. The continuum model](#2-the-continuum-model) | Conservation, Darcy, black oil, compositional EOS |
| [3. Discretisation](#3-discretisation-tpfa--fully-implicit-newton) | TPFA, upstream weighting, FIM, Newton, **adaptive implicit** |
| [4. Wells](#4-wells-the-peaceman-model) | Peaceman inflow and the equivalent radius |
| [5. The linear solver](#5-the-linear-solver-cpr) | Krylov, AMG, CPR, two stages, **automatic hierarchy choice** |
| [6. Mixed precision](#6-mixed-precision-and-df64-emulation) | FP32 preconditioning and df64 error-free transforms |
| [7. Parallel decomposition](#7-parallel-decomposition-and-the-halo) | K-slabs, ghost cells, rank invariance |
| [9. Running](#9-running) | Single GPU, multi-GPU, multi-node |
| [10. Deck reference](#10-deck-reference) | Every keyword |
| [11. Output and plotting](#11-output-and-plotting) | ECLIPSE/OPM files, 3-D views, slices, movies |
| [12. Bayesian history matching](#12-bayesian-history-matching-α-reki) | α-REKI and the surrogate forward map |
| [13. Verification](#13-verification-and-testing) | What is tested and how |
| [SPE10 model 2](#spe10-model-2) | The standard heterogeneity benchmark, and how 861 s became 342 s |
| [14. Status](#14-status-validation-and-provenance) | What was repaired, what OPM Flow says, what is in scope |

---

## 0. The one idea

Multiphase flow in porous media couples two problems of **opposite mathematical
character**:

- a **near-elliptic** pressure field: global, diffusive, stiff, with a condition
  number growing like $`\mathcal{O}(h^{-2})`$ under mesh refinement;
- **near-hyperbolic** saturation and composition fronts: local, advective, cheap
  per cell but sharp, and prone to spurious oscillation if discretised naively.

No single-character linear solver handles both well. Algebraic multigrid is
superb on the elliptic part and useless on the hyperbolic part; a local smoother
is the reverse. **Every** numerical decision in this repository (the CPR
preconditioner, the mixed-precision split, the upstream weighting, even the
domain decomposition) exists to attack each character with the tool that suits
it. Keep this split in mind; the rest is detail.

---

## 1. Repository map

```
full_reservoir_simulator/
├── README.md                          ← this document
├── quickstart.sh                      ★ build everything + run one simulation
├── run_all.sh                         one script: build + run everything
├── simulate.sh                        thin driver for a single deck
├── GPU_reservoir_simulator/
│   ├── NVRS/                          original CuPy reference (physics ground truth)
│   ├── data/                          real permeability fields (PERMFILE input)
│   │   ├── iglesias2.out              1089 values  → 33×33×1,  50–500 mD
│   │   ├── iglesias3.out              4800 values  → 40×40×3,  60–377 mD
│   │   └── test.out                   12000 values → 40×60×5,  0.0003–1 D
│   ├── RESULTS/                       reference outputs for regression
│   │   ├── 33331/  40403/  40605/     one folder per reference case
│   └── cuda/                          ★ the CUDA C++ simulator
│       ├── include/
│       │   ├── common.hpp             scalar types, error macros, capacity limits
│       │   ├── grid.hpp               K-slab decomposition + ghost index space
│       │   ├── halo.hpp               the one-layer K halo exchange
│       │   ├── deck.hpp               OPM/Eclipse-style deck
│       │   ├── fluid.hpp              rel-perm + analytic mobility derivatives
│       │   ├── pvt.hpp                black-oil PVT correlations
│       │   ├── physics.hpp            reservoir state + assembly contract
│       │   ├── linear_solver.hpp      the CPR-preconditioned Krylov solver
│       │   ├── ecl_io.hpp             ECLIPSE/OPM output (EGRID/INIT/UNRST/RSM…)
│       │   └── io.hpp                 snapshots, well rates, summary
│       ├── src/
│       │   ├── main.cu                driver: deck → decompose → Newton+CPR → output
│       │   ├── deck.cpp               keyword parser
│       │   ├── halo.cu                MPI ghost-layer exchange
│       │   ├── physics_host.cu        setup, sparsity/topology, Newton update, PVT
│       │   ├── assemble_blackoil.cu   FIM residual + analytic & numerical Jacobians
│       │   ├── assemble_compositional.cu  nc-component flash + residual/Jacobian
│       │   ├── cpr_solver.cu          CPR + FGMRES/BiCGSTAB, df64 SpMV, HYPRE
│       │   ├── amgx_backend.cu        AmgX stage-1 multigrid (optional)
│       │   ├── ecl_io.cpp             ECLIPSE unformatted writer
│       │   └── io.cu                  snapshots, well rates, summary
│       ├── decks/                     FIELD.DATA (start here), RESERVOIR.DATA,
│       │                              COMPOSITIONAL.DATA, SMOKE.DATA
│       ├── hm/                        α-REKI Bayesian history matching + FNO surrogate
│       ├── scripts/
│       │   ├── plot_results.py        automatic post-processor (driver)
│       │   ├── viz3d.py               3-D voxel views, slice panels, movies
│       │   ├── loaders.py             .mat / .npy / ASCII field readers
│       │   ├── ecl.py                 ECLIPSE/OPM file reader
│       │   ├── run.sh, slurm_multinode.sh, ci_smoke_test.sh, history_match.sh
│       └── docs/                      the full mathematical report
├── Mixed_precision/                   reference mixed-precision CPR + HYPRE build script
└── surrogate/                         PhysicsNeMo inverse pipeline (reference)
```

---

## 2. The continuum model

### 2.1 Conservation and Darcy's law

For each phase $`\alpha \in \{w, o, g\}`$ with saturation $`S_\alpha`$
(with $`\sum_\alpha S_\alpha = 1`$), density $`\rho_\alpha`$ and volumetric flux
$`\vec{u}_\alpha`$, mass conservation on the domain $`\Omega`$ is

```math
\frac{\partial}{\partial t}\big(\phi\,\rho_\alpha S_\alpha\big)
+ \nabla\!\cdot\!(\rho_\alpha \vec{u}_\alpha) \;=\; \rho_\alpha q_\alpha ,
```

with $`q_\alpha`$ the well source/sink and $`\phi`$ the porosity. The multiphase
extension of Darcy's law is

```math
\vec{u}_\alpha \;=\; -\frac{k_{r\alpha}(S)}{\mu_\alpha}\,
\mathbf{K}\,\big(\nabla p_\alpha - \rho_\alpha g\,\nabla z\big) ,
```

where $`k_{r\alpha}`$ is relative permeability, $`\mu_\alpha`$ viscosity,
$`\mathbf{K}`$ the absolute permeability tensor, and $`z`$ depth. Define the
phase **mobility** and **potential**

```math
\lambda_\alpha = \frac{k_{r\alpha}}{\mu_\alpha},
\qquad
\Phi_\alpha = p_\alpha - \rho_\alpha g z .
```

### 2.2 Relative permeability

Two-phase uses the Brooks–Corey closure on the normalised saturation
$`\bar S_w = (S_w - S_{wi})/(1 - S_{wi} - S_{wr})`$:

```math
k_{rw} = \bar S_w^{\,2}, \qquad k_{ro} = (1 - \bar S_w)^2 .
```

Three-phase oil relative permeability is genuinely two-dimensional in
$`(S_w, S_g)`$ and cannot be read from a single table. The maintained release
uses the ECLIPSE/OPM saturation-weighted default, interpolating between the
oil/water and oil/gas two-phase curves:

```math
k_{ro} =
\frac{S_g k_{rog}(S_g)+(S_w-S_{wco})k_{row}(S_w)}
     {S_g+S_w-S_{wco}} .
```

Its analytic quotient-rule partials with respect to both transported
saturations are used by the implicit Jacobian. The earlier Baker-product path
was removed from the maintained parity configuration; archived results made
before the three-phase correction release must be labelled accordingly. Implemented in
[`fluid.hpp`](GPU_reservoir_simulator/cuda/include/fluid.hpp) as `relperm2`
and `relperm3`, which return mobilities **and** derivatives in one pass.

### 2.3 Black-oil PVT

The black-oil model lumps hydrocarbons into surface "oil" and "gas"
pseudo-components. Reservoir volumes relate to stock-tank volumes through
formation volume factors $`B_\alpha(p)`$ and the solution gas–oil ratio
$`R_s(p)`$. With $`b_\alpha = 1/B_\alpha`$, conservation becomes the
surface-volume system

```math
\begin{aligned}
\partial_t(\phi\, b_w S_w) &+ \nabla\!\cdot\!(b_w \vec{u}_w) = b_w q_w, \\
\partial_t(\phi\, b_o S_o) &+ \nabla\!\cdot\!(b_o \vec{u}_o) = b_o q_o, \\
\partial_t\big(\phi[b_g S_g + R_s b_o S_o]\big) &+ \nabla\!\cdot\!(b_g \vec{u}_g + R_s b_o \vec{u}_o)
   = b_g q_g + R_s b_o q_o .
\end{aligned}
```

The implemented correlations, with $`\Delta p = p_{\text{atm}} - \min(p, p_b)`$:

```math
R_s(p) = C\Big(\tfrac{p}{p_b}\Big)^{1.3}, \quad
B_g = e^{-1.7\times10^{-3}\Delta p}, \quad
B_o = e^{\,8\times10^{-5}(p - p_{\text{atm}})}, \quad
\mu_g = 3\!\times\!10^{-10}p^2 + 10^{-6}p + 0.0133 .
```

Above the bubble point $`R_s = C`$ is constant and $`B_o`$ decreases with the oil
compressibility $`c_o`$. See [`pvt.hpp`](GPU_reservoir_simulator/cuda/include/pvt.hpp).

### 2.4 Compositional flow

When phase behaviour is composition-dependent (gas condensate, volatile oil, gas
injection/EOR), the black-oil closure fails and we track $`n_c`$ components with
overall molar densities $`N_i`$. Per cell we solve a two-phase vapour–liquid
flash: Wilson $`K`$-values

```math
K_i = \frac{p_{c,i}}{p}\exp\!\Big[5.373\,(1 + \omega_i)\Big(1 - \frac{T_{c,i}}{T}\Big)\Big]
```

followed by Rachford–Rice for the vapour mole fraction $`\beta`$,

```math
\sum_{i=1}^{n_c} \frac{z_i (K_i - 1)}{1 + \beta(K_i - 1)} = 0 ,
\qquad
x_i = \frac{z_i}{1 + \beta(K_i-1)}, \quad y_i = K_i x_i .
```

The pressure equation is the **volume balance**: the flashed fluid must exactly
fill the pore volume,

```math
\mathcal{R}_p \;=\; \underbrace{\frac{(1-\beta)N_t}{\rho_L}}_{V_o}
\;+\; \underbrace{\frac{\beta N_t}{\rho_V}}_{V_g} \;-\; 1 \;=\; 0 ,
```

with $`\rho_V = p/(Z R T)`$ and $`\rho_L = \big(\sum_i x_i v_{m,i}\big)^{-1}`$.

> **Note.** This closure must use the **unnormalised** phase volumes
> $`V_o, V_g`$. Written in terms of normalised saturations $`S_o + S_g - 1`$ it
> is identically zero, the pressure row of the Jacobian is structurally zero,
> and the CPR pressure block is singular. This was a real defect in the code and
> is [fixed](#14-status-validation-and-provenance).

---

## 3. Discretisation: TPFA + fully-implicit Newton

### 3.1 Two-point flux approximation

Integrating conservation over a cell and applying the divergence theorem gives a
balance of face fluxes. For a face between cells $`i, j`$ the geometric
**transmissibility** is the *harmonic* (series) combination

```math
T_{ij} = \Big(\tfrac{1}{T^{i}_{ij}} + \tfrac{1}{T^{j}_{ij}}\Big)^{-1},
\qquad T^{i}_{ij} = C\,\frac{k_i A_{ij}}{d_i} ,
```

with $`C = 1.127\times10^{-3}`$ the Darcy field-unit constant. This is not a
modelling choice: writing the one-sided Darcy fluxes to a shared face pressure
$`p_{ij}`$, $`q = T^i_{ij}(p_i - p_{ij}) = T^j_{ij}(p_{ij} - p_j)`$, and
eliminating $`p_{ij}`$ yields exactly the harmonic form. It is *exact* for
piecewise-constant $`\mathbf{K}`$ across a planar interface.

The discrete phase flux uses **single-point upstream weighting**:

```math
F^\alpha_{ij} = T_{ij}\,\lambda_\alpha^{\uparrow}\,\big(\Phi_{\alpha,j} - \Phi_{\alpha,i}\big),
\qquad
\lambda_\alpha^{\uparrow} =
\begin{cases}\lambda_\alpha(S_j) & \Phi_{\alpha,j} > \Phi_{\alpha,i}\\
             \lambda_\alpha(S_i) & \text{otherwise.}\end{cases}
```

**Proposition (monotonicity).** *For the scalar transport step with these fluxes
and a nondecreasing fractional flow $`f(S) = \lambda_w/(\lambda_w+\lambda_o)`$,
if*

```math
\Delta t \le \frac{V_i\phi_i}{\sum_j (T_{ij}\,\partial_S f)^+ |\Delta\Phi_{ij}|}
```

*then $`S_i^{n+1} \in [\min_k S_k^n, \max_k S_k^n]`$; the update is a convex
combination of neighbouring states and introduces no new extrema.*

Upstream weighting makes $`\partial S_i^{n+1}/\partial S_k^n \ge 0`$ for
$`k \neq i`$; the CFL bound keeps the self-coefficient nonnegative. All
coefficients nonnegative and summing to one is exactly a convex combination, so
the discrete maximum principle holds. A centred weighting loses the sign
condition and admits oscillations.

### 3.2 The fully-implicit residual

Collecting accumulation, fluxes and wells, the discrete residual for component
$`c`$ in cell $`i`$ over a step $`\Delta t`$ is

```math
R^c_i(\mathbf{u}) = \frac{V_i\phi_i}{\Delta t}\big(m^c_i - m^{c,n}_i\big)
 - \sum_{j\in\mathcal{N}(i)} b_c^{\uparrow} F^{c}_{ij}(\mathbf{u}) - Q^c_i(\mathbf{u}) .
```

Per-cell unknowns are $`(p, S_w)`$ for two-phase, $`(p, S_w, S_g)`$ for
three-phase, or $`(p, N_1, \dots, N_{n_c})`$ for compositional, a block size
$`b_s \in \{2, 3, n_c{+}1\}`$.

### 3.3 Newton's method

```math
\mathbf{J}(\mathbf{u}^{k})\,\delta\mathbf{u} = -\mathbf{R}(\mathbf{u}^{k}),
\qquad
\mathbf{u}^{k+1} = \mathbf{u}^{k} + \theta\,\delta\mathbf{u},
\qquad
\mathbf{J} = \frac{\partial \mathbf{R}}{\partial \mathbf{u}} .
```

Newton converges quadratically near the root: if $`\mathbf{J}`$ is Lipschitz with
constant $`\gamma`$ and $`\|\mathbf{J}(\mathbf{u}_\star)^{-1}\| \le \beta`$, then

```math
\|\mathbf{u}^{k+1} - \mathbf{u}_\star\| \le \tfrac12\beta\gamma\|\mathbf{u}^{k} - \mathbf{u}_\star\|^2 .
```

Because $`R^c_i`$ depends only on cell $`i`$ and its face neighbours,
$`\mathbf{J}`$ has a **7-point block-sparse** structure of $`b_s \times b_s`$
blocks.

**Two assembly paths, and why both exist.**

1. **Numerical** (`JACOBIAN NUMERICAL`): each thread copies its $`\le 7`$-member,
   $`b_s`$-unknown stencil into registers, evaluates the residual, then perturbs
   one unknown at a time,
   $`J_{ab} \approx [R_a(\mathbf{u} + \varepsilon e_b) - R_a(\mathbf{u})]/\varepsilon`$.
   This reproduces the exact sparsity with no hand-coded derivatives (a
   derivative-bug-free reference) at $`\mathcal{O}(7 b_s)`$ residual evaluations
   per cell.
2. **Analytic** (`JACOBIAN ANALYTIC`): exact derivatives for two- and three-phase
   black oil, differentiating the *same* residual. Each phase flux is
   differentiated with respect to its own potential-upstream cell, so saturation
   derivatives route to self or neighbour independently per phase.

The two must agree to $`\mathcal{O}(\varepsilon)`$. That is not a hope, it is a
**built-in test**: `./scripts/run.sh deck.DATA --jac-check` assembles both and reports
the maximum relative difference.

### 3.4 Adaptive implicit (`AIM`, on by default)

In a fully implicit step every cell carries an implicit saturation unknown,
including the great majority in which almost nothing moves. **Adaptive
implicit** keeps those cells implicit in *pressure only*: their face
mobilities are frozen at the start of the step, and their saturation
couplings leave the Jacobian. Wells and cells at a front stay fully implicit.

The accumulation term stays implicit **everywhere**, so every row remains well
posed, the sparsity pattern never changes, and the Newton loop, the residual
test and the CPR solve are untouched. Nothing downstream of the assembly knows
AIM is on.

**The switching rule** is the classical CFL condition written in reservoir
variables, flux per pore volume *is* the reciprocal residence time, so this is
$`v\,\Delta t/\Delta x`$:

```math
\mathrm{CFL}_c \;=\; \frac{\Delta t}{V\phi}\sum_{f}T_f\,\bigl|\Delta\Phi_p\bigr|\,
                     \left|\frac{\partial\lambda_p}{\partial S}\right| \;<\; \texttt{AIMCFL}
```

Below the threshold the cell is explicit; above it, implicit. Two details in
that formula were each found by measurement, and both matter:

1. **The sensitivity `∂λ/∂S`, not the mobility `λ`.** The stability limit of an
   explicit saturation update is set by how fast the flux *changes* with
   saturation. Near a Corey front the derivative is several times the mobility,
   so a mobility-weighted rule marks front-adjacent cells stable when they are
   not, and the timestep controller then collapses to whatever those cells
   tolerate (measured: dt pinned at 0.39 days, Newton count more than doubled).
2. **Both flow directions count, each with its donor's mobility.** Counting
   only outflow marks the cell just *downstream* of a front explicit (its own
   water is still immobile) while an implicit neighbour keeps filling it at a
   rate frozen for the substep. The receiver overfills past its physical
   bounds, the saturation clamp blocks the residual from closing, and Newton
   stalls at every dt.

**The rates come from a converged state.** They cannot be evaluated at the top
of a substep, where the pressure field is whatever the previous solve left, on
the very first substep that is the *uniform initial condition*, every potential
difference is zero, the throughput looks like zero everywhere, and the entire
field is flagged explicit including the cells about to be filled by the
injector. Until a converged measurement exists the run is fully implicit, so
the first substep of each report step is a plain FIM step; a timestep chop
resets to implicit, because the rates it measured describe a state that was
rolled back.

**What it buys.** On SPE10 model 2 over 2000 days on one GB200:

| | Wall clock | Newton | Krylov |
|---|---|---|---|
| Fully implicit | 402.6 s | 1984 | 203,414 |
| **Adaptive implicit** | **342.1 s** | **1789** | **175,728** |

The explicit fraction falls from 86% early in the run to 54% at the end, as the
timestep grows and more cells exceed the threshold.

The gain is real but smaller than adaptive implicitness classically delivers,
and the reason is worth stating: **CPR has already captured most of it.** AIM's
textbook benefit is a smaller coupled system, but stage 1 solves a
*pressure-only* system whose size AIM does not change, and the saturation
couplings it removes were being handled by a cheap block-Jacobi stage 2. What
remains is the assembly and operator saving, which is what the measurement
shows.

`AIM 0` restores fully implicit. It is the right first move if a result differs
from an older run and you want to know whether the scheme is responsible. AIM
requires `JACOBIAN ANALYTIC` (it freezes donor mobilities *inside the
residual*, and the numerical Jacobian would differentiate a different one) and
does not apply to compositional runs; both fall back to FIM automatically.

---

## 4. Wells: the Peaceman model

A grid cell is vastly larger than a wellbore radius, so the radial pressure
funnel cannot be resolved. Peaceman's model supplies the analytic correction:

```math
q_\alpha = \frac{2\pi\,C\,k\,h\,\lambda_\alpha}{\ln(r_e/r_w) + s}\,(p - p_{\text{wf}}),
\qquad r_e = 0.2\,\Delta x ,
```

with $`p_{\text{wf}}`$ the flowing bottom-hole pressure and $`s`$ the skin. The
**well index** $`\mathrm{WI} = 2\pi C k h/[\ln(r_e/r_w)+s]`$ is precomputed per
perforated cell.

*Where $`r_e = 0.2\Delta x`$ comes from.* In the well cell the true flow is
radial, $`p(r) = p_{\text{wf}} + \frac{q\mu}{2\pi k h}\ln(r/r_w)`$, but the
simulator carries only the cell-average $`p_i`$. Peaceman solved the discrete
five-point pressure equation for a single well in an infinite uniform grid and
matched it to the analytic radial solution: $`p_i`$ equals the steady radial
pressure at the **equivalent radius** where the numerical and analytic profiles
cross, $`r_e \approx 0.198\,\Delta x`$ for an isotropic square grid. The constant
is exposed as the deck keyword `REFACTOR`, so the well model is never hard-wired.

Injectors are rate-controlled; producers are BHP-controlled and their rates
therefore depend on $`p`$ and are differentiated inside the Newton Jacobian.

---

## 5. The linear solver: CPR

### 5.1 Block structure

Write the coupled Jacobian in pressure/saturation blocks:

```math
\mathbf{J} = \begin{pmatrix}\mathbf{A}_{pp} & \mathbf{A}_{ps}\\
                            \mathbf{A}_{sp} & \mathbf{A}_{ss}\end{pmatrix},
```

where $`\mathbf{A}_{pp}`$ is the elliptic pressure operator. **Constrained
Pressure Residual** (CPR) is a two-stage preconditioner tailored to exactly this
split. In additive-correction form, $`\mathbf{z} = \mathbf{M}^{-1}\mathbf{r}`$ is

```math
\begin{aligned}
\textbf{Stage 1:}\quad &\mathbf{z} \leftarrow \mathbf{R}_p^{\top}\,
   \mathrm{AMG}\big(\mathbf{R}_p\,\mathbf{r}\big), \\
\textbf{Stage 2:}\quad &\mathbf{z} \leftarrow \mathbf{z} + \mathbf{D}^{-1}\big(\mathbf{r} - \mathbf{J}\mathbf{z}\big),
   \qquad \mathbf{D} = \mathrm{blockdiag}(\mathbf{J}) .
\end{aligned}
```

Stage 1 removes the global elliptic mode with one AMG V-cycle; stage 2 is a
cell-local $`b_s \times b_s`$ block-Jacobi correction that cleans the hyperbolic
coupling.

**Proposition (why the two stages compose).** *Suppose (i) the AMG stage is
spectrally equivalent to $`\mathbf{A}_{pp}^{-1}`$ on the pressure block, and
(ii) block-Jacobi is exact on the cell-diagonal blocks. Write
$`\mathbf{A} = \mathbf{D} + \mathbf{O}`$. Stage 1 removes the pressure-range
component up to the AMG defect $`\varepsilon_{\mathrm{amg}} \ll 1`$; stage 2
inverts $`\mathbf{D}`$ exactly. What survives is $`\mathbf{D}^{-1}\mathbf{O}`$
acting on the AMG defect, so*

```math
\rho\big(\mathbf{I} - \mathbf{M}_{\mathrm{CPR}}^{-1}\mathbf{A}\big)
\;\lesssim\; \|\mathbf{D}^{-1}\mathbf{O}\|\;\varepsilon_{\mathrm{amg}} .
```

*For reservoir Jacobians $`\|\mathbf{D}^{-1}\mathbf{O}\| = \mathcal{O}(1)`$
(local coupling) and $`\varepsilon_{\mathrm{amg}}`$ is mesh-independent, so
$`\rho < 1`$ uniformly in $`h`$.* This is the reason outer iteration counts stay
nearly flat under mesh refinement.

### 5.2 The apply, as implemented

| Step | Operation | Kernel | Precision |
|---|---|---|---|
| 1 | $`\mathbf{r}_p \leftarrow \mathbf{R}_p\mathbf{r}`$ (gather, stride $`b_s`$) | `k_extract_p` | FP32 |
| 2 | $`\mathbf{z} \leftarrow \mathbf{R}_p^\top\mathrm{AMG}(\mathbf{r}_p)`$ | HYPRE BoomerAMG + `k_scatter_p` | FP32 |
| 3 | $`\mathbf{t} \leftarrow \mathbf{A}\mathbf{z}`$ | cuSPARSE SpMV or `k_spmv_df64` | FP64 / df64 |
| 4 | $`\mathbf{z} \leftarrow \mathbf{z} + \mathbf{D}^{-1}(\mathbf{r}-\mathbf{t})`$ | `k_stage2` | FP32 |

The block inverses $`\mathbf{D}^{-1}`$ are assembled in closed form for
$`b_s \le 3`$ (cofactor) and by Gauss–Jordan for compositional.

**Proposition (CPR is bandwidth-dominated).** One apply costs one coupled SpMV
($`\sim 7b_s^2`$ nnz), one AMG V-cycle on a scalar operator, and
$`\mathcal{O}(b_s^2)`$ block-smoother flops. All three are memory-bound and the
dominant traffic is reading the matrix values, which is precisely the lever
mixed precision pulls.

### 5.3 Outer Krylov

CPR is *always* the preconditioner; the outer accelerator is selectable.

- **FGMRES** (default): flexible GMRES with Givens QR and CGS2 reorthogonalisation.
  Flexible is required: the preconditioner is nonlinear across iterations because
  AMG is not a fixed linear operator to full precision.
- **BiCGSTAB**: shorter recurrence, constant memory, no restart parameter.

Both run in FP64 with MPI-reduced inner products.

### 5.4 Which multigrid: `LINSOLVER`, and which hierarchy inside it

Stage 1 needs a multigrid for $`\mathbf{A}_p`$, and the simulator carries two
libraries:

```
LINSOLVER AMGX     -- default: NVIDIA AmgX
LINSOLVER HYPRE    -- HYPRE BoomerAMG, classical PMIS coarsening
```

**Within AmgX the hierarchy is chosen automatically, by rank count**, and the
run banner states which and why:

| Ranks | Hierarchy | Why |
|---|---|---|
| **1** | classical PMIS + extended (D2) interpolation | The stronger preconditioner on heterogeneous rock. On SPE10 model 2 it halves the Krylov count against aggregation and halves the run: **117 s against 237 s** over four report steps. |
| **2+** | aggregation, `SIZE_8` | The hierarchy that survives a decomposition. |

The split is a property of the library, not a preference. AmgX's **distributed
classical** setup exhausts device memory while building its communication maps;
`markCoarseFinePoints` → `CommsMPIHostBufferStream::do_setup` →
`cudaMallocAsync` fails: within seconds of the first setup, at 2, 3 and 4
ranks alike, with either interpolator (D1 or D2), *on a problem each rank holds
less of than the single-rank run that succeeds*. Aggregation is unaffected and
scales. So single-GPU runs get the fast hierarchy and distributed runs get the
one that works, without anyone having to know this.

Overrides, in order of precedence:

```bash
NVRS_AMGX_CONFIG=<file>            # replace the configuration entirely
NVRS_AMGX_ALGO=CLASSICAL           # force classical (even on several ranks)
NVRS_AMGX_ALGO=AGGREGATION         # force aggregation (even on one)
```

Classical-on-many-ranks is deliberately left reachable so that a future AmgX
can be retested with one environment variable and no rebuild. Note also that a
scaling study should pin **one** hierarchy with `NVRS_AMGX_ALGO`, since a
speed-up column that compares classical at one rank against aggregation at four
is measuring two things at once.

It is worth being exact about what this does **not** change, because otherwise
the choice looks like it moves the precision story. It does not. The outer
Krylov is the same FP64 FGMRES; the operator it applies is the same
df64-emulated product of §6; the quasi-IMPES weights that build $`\mathbf{A}_p`$
are the same and still FP64; and stage 2 is the same $`b_s \times b_s`$
block-Jacobi correction. AmgX runs in `dFFI` mode (device, FP32 matrix, FP32
vector) which is exactly where the FP32 build of HYPRE sits. Both are
configured as a single V-cycle per application, iteration limit 1 and tolerance
0, so a comparison between them is a comparison of two multigrids and not of two
stopping rules.

They differ in kind. BoomerAMG coarsens with PMIS, an *iterative* parallel
maximal-independent-set computation whose passes exchange boundary state with
neighbours; the number of passes depends on the graph rather than on how much of
it each rank holds, so the setup cost barely divides when the grid does. AmgX
builds aggregates locally and its setup runs entirely on the device.

Measured on the 4,800-cell `BENCH3P` deck over the full twenty-step
schedule, one GB200, the one case where both arms take a comparable nonlinear
path:

| | HYPRE BoomerAMG | AmgX |
|---|---:|---:|
| Wall clock | 3.21 s | **0.88 s** |
| Newton iterations | 264 | 239 |
| Krylov iterations | 6,829 | **1,279** |
| AMG setup, total | 0.29 s | **0.13 s** |
| Hierarchy rebuilds | 48 | **24** |

A comparable nonlinear problem (264 and 239 Newton iterations) reached in a
fifth of the Krylov iterations, so AmgX is a better preconditioner and not
merely a cheaper one to build. On the Norne field deck the same ordering holds:
71.51 s against 44.95 s, 1,506 against 1,497 Newton iterations, with Ginkgo's best arm between them at 70.42 s.

The more consequential difference is what happens past one GPU. The full
measured ladder, every run the deck's whole schedule, wall-clock seconds on
GB200 and on VR72:

| Cells | Backend | GB200 1 GPU | 2 GPUs | 4 GPUs | 1 to 4 | VR72 1 GPU | 2 GPUs | 4 GPUs | 1 to 4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,800 | HYPRE | 3.00 | 6.22 | 7.91 | 0.38× | 3.44 | 7.98 | 13.29 | 0.26× |
| | AmgX | **0.89** | 2.00 | 2.75 | 0.32× | **0.95** | 2.33 | 3.93 | 0.24× |
| 1,000,000 | HYPRE | 26.23 | 33.85 | 43.27 | 0.61× | 19.94 | 28.88 | 49.61 | 0.40× |
| | AmgX | 21.17 | 18.96 | **18.26** | 1.16× | 14.87 | **14.02** | 16.17 | 0.92× |
| 2,000,000 | HYPRE | 49.94 | 52.84 | 65.32 | 0.76× | 38.23 | 43.83 | 72.11 | 0.53× |
| | AmgX | 39.38 | 31.05 | **30.84** | 1.28× | 27.17 | **21.24** | 23.62 | 1.16× |
| 4,000,000 | HYPRE | 81.87 | 80.00 | 86.67 | 0.94× | 60.97 | 60.96 | 85.21 | 0.72× |
| | AmgX | 77.67 | 58.01 | **51.78** | 1.50× | 52.33 | 36.58 | **41.14** | 1.27× |
| 5,000,000 | HYPRE | 102.32 | 96.16 | 96.64 | 1.06× | 76.65 | 77.49 | 93.30 | 0.81× |
| | AmgX | 95.61 | 71.14 | **61.71** | 1.55× | 63.77 | 43.91 | **42.26** | 1.51× |

Two readings. Across the ladder AmgX is faster at every size and every rank
count: 3.6× on the small case, 1.09–1.26× on one GB200 from one to five million cells, and 1.64–2.37× at four GB200 GPUs (2.24–3.10× on VR72). And the two
backends scale in opposite directions: HYPRE never gets past its own single GPU on GB200 and loses ground on VR72, while AmgX gains from the
one-million-cell case on and its scaling grows with problem size (1.28× at 2M, 1.50× at 4M, 1.55× at 5M on GB200; 1.51× at 5M on VR72). The mechanism is the
hierarchy setup: BoomerAMG's grows from 9.2 s to 23.1 s across one to four GB200 ranks on the 5M case while AmgX's falls from 10.7 s to 1.7 s. The phase
that refused to divide is the phase aggregation coarsening makes cheap enough
not to matter.

![Head to head](paper/figs/headline.png)

![Backend scaling](paper/figs/scaling.png)

`scripts/backend_study.sh` runs both across 1, 2 and 4 GPUs on a ladder of grid
sizes and writes the comparison, including the well curves and saturation panels
that show the two agree. AmgX is optional at build time: without it
`src/amgx_backend.cu` compiles to nothing, a deck asking for `LINSOLVER AMGX`
gets HYPRE **and a warning saying so**, and no log can silently mislabel which
backend produced it.

---

## 6. Mixed precision and df64 emulation

IEEE-754 FP64/FP32 carry 52/23 mantissa bits, unit roundoff
$`u = 2^{-53}, 2^{-24}`$. The arithmetic model is
$`\mathrm{fl}(a \odot b) = (a \odot b)(1+\delta)`$, $`|\delta| \le u`$.

### 6.1 Mixed CPR

The outer Krylov and the true residual run in **FP64**; the CPR preconditioner (BoomerAMG and the stage-2 blocks) runs in **FP32**.

**Proposition (backward error is unaffected).** *Let FGMRES run with the exact
operator but a perturbed preconditioner
$`\widehat{\mathbf{M}}^{-1} = \mathbf{M}^{-1} + \mathbf{E}`$. The computed
iterate satisfies the same stopping test
$`\|\mathbf{b} - \mathbf{A}\mathbf{x}_m\| \le \tau\|\mathbf{b}\|`$ in the
operator precision; only the iteration count changes.*

FGMRES minimises the **true** residual over the flexible Krylov space regardless
of which $`\mathbf{z}_j = \widehat{\mathbf{M}}^{-1}\mathbf{v}_j`$ built that
space. The preconditioner chooses search directions, not the residual functional.
So $`\mathbf{E}`$ moves the *rate*, never the *limit*.

### 6.2 Double–single (df64) emulation

On hardware where the FP64:FP32 throughput ratio is punishing (1:32 on consumer
and some datacentre parts), the FP64 **operator** itself can be emulated in
double–single arithmetic: represent $`x`$ as a pair of FP32 values
$`(hi, lo)`$ with $`hi + lo \approx x`$. Using error-free transformations,

```math
\mathrm{TwoSum}(a,b) = (s,e),\; s = \mathrm{fl}(a+b),\; e = b - (s-a);
\qquad
\mathrm{TwoProd}(a,b) = (p,e),\; e = \mathrm{fl}_{\mathrm{FMA}}(ab - p),
```

both add and multiply become pure FP32 sequences.

**Lemma (df64 accuracy).** *For $`hi = \mathrm{fl}_{32}(x)`$,
$`lo = \mathrm{fl}_{32}(x - hi)`$,*

```math
|(hi + lo) - x| \le u_{32}^2\,|x| \approx 2^{-48}|x| .
```

*Proof.* $`hi = x(1+\delta_1)`$ with $`|\delta_1| \le u_{32}`$, so
$`r := x - hi = -x\delta_1`$ with $`|r| \le u_{32}|x|`$. Then
$`lo = \mathrm{fl}_{32}(r) = r(1+\delta_2)`$, so $`hi + lo = x + r\delta_2`$ and
$`|(hi+lo) - x| = |r||\delta_2| \le u_{32}^2|x|`$. ∎

So df64 carries $`\approx 48`$ correct bits, comfortably enough to preserve
Krylov convergence to reservoir tolerances of $`10^{-6}`$, while every
instruction issues on the FP32 pipeline. Implemented in `k_spmv_df64`; the
operator is split into $`(hi, lo)`$ FP32 slices after each assembly.

Setting `EMULATE 0` selects a native-FP64 operator, which lets you verify
*emulated-FP64 mixed* $`\equiv`$ *true-FP64 mixed*.

---

## 7. Parallel decomposition and the halo

The global grid is sliced into contiguous slabs along the $`K`$ axis, one slab
per MPI rank, one GPU per rank. Global natural ordering
$`g = (k \cdot n_y + j)\cdot n_x + i`$ means each rank owns a **contiguous global
row range** $`[\text{row}_0, \text{row}_0 + n_{\text{local}})`$, exactly the
layout HYPRE's IJ interface wants, so the pressure AMG runs distributed with zero
renumbering.

### 7.1 Why the halo is not optional

The TPFA stencil reaches one cell across the slab boundary. Each rank therefore
stores a one-layer **ghost** region appended after its interior cells:

```
local index   0 .. N-1                    interior (owned)
              N .. N+L-1                  ghost layer at global k₀-1
              N+L .. N+L+L'-1             ghost layer at global k₀+nz_local
```

The coupled operator is consequently **rectangular in the local index space**:
$`n`$ owned rows by $`n_{\text{ext}}`$ columns. Every vector handed to the SpMV
must carry the ghost tail, and `halo_exchange` refreshes it before every assembly
and every SpMV.

Without this, the off-rank $`K`$ faces are silently no-flow and an $`N`$-rank run
becomes $`N`$ **independent reservoirs stacked on top of each other**: the
answer then depends on the rank count. This is invisible on one rank, which is
what makes it dangerous. The guard is a regression test:

```bash
./scripts/ci_smoke_test.sh   # stage 7 compares 1-rank vs 2-rank final state
```

Cell-centre depth is likewise measured from the **global** datum via a per-cell
global-$`K`$ array, otherwise each rank would apply gravity from its own slab
origin.

### 7.2 Scaling modes

| Configuration | Ranks | Launch |
|---|---|---|
| 1 node, 1 GPU | 1 | `NGPU=1 ./scripts/run.sh deck.DATA` |
| 1 node, G GPUs | G | `./scripts/run.sh deck.DATA` |
| N nodes × G GPUs | N·G | `sbatch scripts/slurm_multinode.sh deck.DATA` |

By default halo layers are staged through pinned host buffers, which works with
any MPI. Export `NVRS_CUDA_AWARE_MPI=1` to pass device pointers straight to MPI
(UCX/HPC-X, MPICH+GDRCopy) and skip the staging copies.

---

## 9. Running

### 9.1 Basic invocation

Run from `GPU_reservoir_simulator/cuda/`, so the deck's relative `PERMFILE`
path resolves:

```bash
NGPU=1 ./scripts/run.sh decks/FIELD.DATA
```

or through the launcher, which detects GPUs and sets up `LD_LIBRARY_PATH`:

```bash
# single GPU
NGPU=1 ./scripts/run.sh decks/FIELD.DATA

# every GPU on the node
./scripts/run.sh decks/FIELD.DATA

# the large synthetic case
./scripts/run.sh decks/RESERVOIR.DATA

# compositional case (BiCGSTAB outer Krylov, selected in the deck)
./scripts/run.sh decks/COMPOSITIONAL.DATA
```

### 9.2 Multi-node

```bash
sbatch scripts/slurm_multinode.sh decks/RESERVOIR.DATA
```

or directly:

```bash
mpirun -np 16 --map-by ppr:4:node ./build/nvrs_cuda decks/RESERVOIR.DATA
```

One rank per GPU. The grid is K-slab decomposed automatically, so the rank count
must not exceed `NZ`.

### 9.3 Verification mode

```bash
NGPU=1 ./scripts/run.sh decks/RESERVOIR.DATA --jac-check
```

Assembles the analytic and numerical Jacobians on the initial state and reports
the maximum absolute and relative difference. A relative difference below
$`10^{-4}`$ is a pass.

### 9.4 Environment variables

| Variable | Effect |
|---|---|
| `NVRS_CUDA_AWARE_MPI=1` | Pass device pointers to MPI; skip host staging in the halo exchange |
| `NVRS_PLOT_SCRIPT=<path>` | Override the location of `plot_results.py` |
| `NVRS_PYTHON=<exe>` | Python interpreter used for automatic plotting |
| `CUDA_VISIBLE_DEVICES` | Restrict which GPUs are used |

## 10. Deck reference

The grammar follows OPM Flow / Eclipse convention: uppercase keywords introduce
sections, one record per line, `/` terminates a record list, `--` starts a
comment. Unknown scalars are ignored for forward compatibility.

### Model and grid

| Keyword | Default | Meaning |
|---|---|---|
| `TITLE` | - | Run title |
| `MODEL` | `BLACKOIL` | `BLACKOIL` or `COMPOSITIONAL` |
| `NCOMP` | 3 | Black oil: 2 = oil/water, 3 = oil/water/gas. Compositional: number of components (≤ 8). **Must precede `MODEL`.** |
| `OUTDIR` | `OUTPUT` | Output directory |
| `NX NY NZ` | 1 | Grid dimensions |
| `DX DY DZ` | 50/50/20 | Cell size [ft] |
| `KZMULT` | 0.2 | $`K_z = \text{KZMULT} \cdot K_x`$ |
| `PERMMIN PERMMAX` | 5 / 500 | [mD] range for the synthetic permeability field |
| `POROMIN POROMAX` | 0.05 / 0.30 | Porosity range |
| `PERMFILE` | - | External permeability file (one value per global cell, natural order) |

### Fluid properties

| Keyword | Default | Meaning |
|---|---|---|
| `UW UO UG` | 1.0 / 2.5 / 0.02 | Viscosities [cP] |
| `BW BO BG` | 1.0 / 1.2 / 0.005 | Formation volume factors |
| `RHOW RHOO RHOG` | 1.0 / 0.8 / 0.01 | Densities [g/cc] |
| `CFW CFO CT` | 1e-5 / 1e-5 / 2e-5 | Compressibilities [1/psi] |
| `SWI SWR` | 0.1 / 0.1 | Irreducible water / residual saturations |
| `PINIT PBUB PATM` | 3000 / 3000 / 14.6959 | Initial, bubble-point, atmospheric pressure [psia] |
| `SWINIT SGINIT SOINIT` | 0.2 / 0.02 / 0.78 | Initial saturations |
| `GRAVITY` | 1 | 1 = on, 0 = off |
| `SWOF` / `SGOF` | - | Rel-perm tables: `(Sw, Krow, Krw)` / `(Sg, Krog, Krg)`, `/`-terminated |

### Compositional (EOS)

| Keyword | Meaning |
|---|---|
| `ZMF` | Overall feed mole fractions (`NCOMP` values) |
| `TCRIT` | Critical temperatures [R] |
| `PCRIT` | Critical pressures [psia] |
| `ACF` | Acentric factors |
| `VMOL` | Liquid molar volumes [ft³/lbmol] |
| `TEMP` | Reservoir temperature [R] |
| `ZFACTOR` | Gas compressibility factor |

### Solver

| Keyword | Default | Meaning |
|---|---|---|
| `KRYLOV` | `FGMRES` | Outer accelerator: `FGMRES` or `BICGSTAB`. CPR is always the preconditioner. |
| `LINTOL` | 1e-6 | Linear relative tolerance |
| `LINMAXIT` | 100 | Maximum Krylov iterations |
| `RESTART` | 30 | FGMRES restart length |
| `MIXED` | 1 | 1 = FP32 AMG (mixed), 0 = FP64 AMG |
| `EMULATE` | 1 | 1 = double-word (df64) FP64 operator, 0 = native FP64 |
| `JACOBIAN` | `NUMERICAL` | `NUMERICAL` (all models) or `ANALYTIC` (black oil 2- and 3-phase). Required by `AIM`. |
| `LINSOLVER` | `AMGX` | Stage-1 multigrid library: `AMGX` or `HYPRE`. The hierarchy *within* AmgX is chosen automatically, see §5.4. |
| `STAGE2` | `JACOBI` | CPR second stage: `JACOBI` (per-cell block) or `ZLINE` (block tridiagonal along each vertical column) |
| `AIM` | **1** | **Adaptive implicit**: see §3.4. `AIM 0` (or `IMPLICIT FIM`) for fully implicit. |
| `AIMCFL` | 0.5 | Throughput CFL below which a cell runs explicit in saturation |
| `APPLEYARD` | 0 | Truncate saturation updates that cross a relperm endpoint (measured slower on SPE10) |
| `FIELDSATFUNC` | **1** | Take `SWOF`/`SGOF` from the deck named by `FIELDDECK` rather than the run deck. Converts ECLIPSE's `Sw krw krow` order to this format's oil-curve-second; prints the four curve corners at startup so a swap cannot pass silently. |
| `PROPSARRAYS` | 0 | Read per-cell arrays from `PROPS`/`REGIONS`/`SOLUTION`, not just `GRID`. Makes ENDSCALE's endpoints visible, and also gives equilibration the deck's `SWL` and `EQLNUM`, which changes the initial state. Runs with 0 and 1 are not comparable. |
| `USEENDSCALE` | 0 | Scale each cell's relperm onto its own endpoints (needs `PROPSARRAYS 1`). Two-point, or three-point where the field deck sets `SCALECRS YES`. |

### Physical constants (deck-overridable: nothing is hard-wired)

| Keyword | Default | Meaning |
|---|---|---|
| `DARCY` | 1.127e-3 | Field-unit transmissibility constant |
| `GRAVGRAD` | 0.4335 | psi/ft per (g/cc) fluid density |
| `REFACTOR` | 0.2 | Peaceman equivalent radius / `DX` |
| `FDEPS` | 1e-7 | Numerical-Jacobian finite-difference step scale |

### Time and output

| Keyword | Default | Meaning |
|---|---|---|
| `DTREPORT` | 100 | Days per report step |
| `TEND` | 2000 | Simulated period [days] |
| `NEWTONMAX` | 20 | Maximum Newton iterations per step |
| `NEWTONTOL` | 5e-3 | Newton relative tolerance |
| `MAKECSV MAKEVTK MAKESUMMARY` | 1 | Output toggles |
| `AUTOPLOT` | 1 | Render plots at end of run (set 0 inside history-matching loops) |
| `ECLOUT` | 1 | Write the ECLIPSE/OPM file family (EGRID, INIT, UNRST, SMSPEC, UNSMRY, RSM) |
| `CASENAME` | `NVRS` | ECLIPSE base name for those files |

### Wells

```
WELLS
--  I   J  TYPE  RATE   RW   SKIN  PWF  PERF  NAME
   10  91    1   1000  500   0.0   100  all   I1
    7   9   -1     -1  100   0.0   100  all   P1
/
```

`TYPE` is `1` for a rate-controlled injector, `-1` for a BHP-controlled producer.
`PERF` is `all` layers or `last` (bottom layer only).

---

## 11. Output and plotting

### 11.1 ECLIPSE / OPM Flow compatible output

Every run writes the standard ECLIPSE file family, so results open directly in
**ResInsight**, OPM's `summary` and `compare_ECL_*` tools, Petrel, and anything
else in the usual toolchain, and so a run can be diffed against **OPM Flow** on
the same deck, which is the strongest verification available.

| File | Contents |
|---|---|
| `CASE.EGRID` | Grid geometry: `COORD`/`ZCORN` corner-point pillars + `ACTNUM` |
| `CASE.INIT` | Static properties: `PORO`, `PERMX/Y/Z`, `DX/DY/DZ`, `TOPS`, `DEPTH`, `PORV` |
| `CASE.UNRST` | Unified restart: `PRESSURE`, `SWAT`, `SGAS`, `SOIL` per report step |
| `CASE.SMSPEC` | Summary vector specification (mnemonics, units, well names) |
| `CASE.UNSMRY` | Unified summary data, the time series themselves |
| `CASE.RSM` | The human-readable ASCII summary ECLIPSE prints |

The binary files use the ECLIPSE *unformatted* convention: big-endian Fortran
sequential records, each bracketed by a 4-byte byte-count marker. A named array
is a header record

```
[int32 16]["KEYWORD "][int32 count]["TYPE"][int32 16]
```

followed by data in blocks of at most 1000 elements (105 for `CHAR`), each block
bracketed by its own markers. Types are `INTE`, `REAL`, `DOUB`, `LOGI`, `CHAR`
and `MESS`. Implemented in
[`ecl_io.cpp`](GPU_reservoir_simulator/cuda/src/ecl_io.cpp); controlled by the
deck keywords `ECLOUT` (default 1) and `CASENAME`.

Summary vectors written: `TIME`, per-producer `WOPR`, `WWPR`, `WGPR`, `WWCT`,
`WBHP`, and field-level `FOPR`, `FWPR`, `FOPT`, `FWPT`, `FWCT`, `FPR`, `FPAV`.
Cumulatives use the trapezoidal rule over the report interval, matching
ECLIPSE's `FOPT`/`FWPT`.

A pure-Python reader for the same format,
[`scripts/ecl.py`](GPU_reservoir_simulator/cuda/scripts/ecl.py): reads both this
simulator's output and OPM Flow's:

```bash
python3 scripts/ecl.py OUTPUT/NVRS.UNRST      # dump keyword inventory
```

```python
from ecl import EclipseCase
case = EclipseCase("OUTPUT/NVRS")
nx, ny, nz = case.dims
P = case.restart_field("PRESSURE")     # (nsteps, nz, ny, nx)
t, vectors = case.summary()            # time series by "KEYWORD:WELL"
```

### 11.2 Native output

| File | Contents |
|---|---|
| `meta.json` | Grid, times, solver telemetry, well list |
| `state_%04d.bin` | float32, four planes of $`n_x n_y n_z`$: $`[p, S_w, S_o, S_g]`$ |
| `rock.bin` | float32, two planes of $`n_x n_y n_z`$: $`[k, \phi]`$: the static model, written once |
| `RSM_FVM.csv` | Per-report-step well rates |
| `summary.txt` | Written insight summary |

### 11.3 The post-processor is fully automatic

Run with **no arguments** and it finds the output directories itself, processes
every one, and writes a browsable HTML report:

```bash
python3 scripts/plot_results.py                  # discover and plot everything
python3 scripts/plot_results.py OUTPUT           # one specific directory
python3 scripts/plot_results.py --watch          # re-plot as a long run advances
```

It is also invoked automatically at the end of a run (unless `AUTOPLOT 0`), with
the script path resolved relative to the executable, so the working directory
does not matter.

**It reads four kinds of input**, not just its own:

```bash
# this simulator's native output
python3 scripts/plot_results.py OUTPUT

# any ECLIPSE/OPM case -- including an OPM Flow run, for side-by-side comparison
python3 scripts/plot_results.py --ecl OUTPUT/NVRS

# a MATLAB or NumPy array: 3-D (k,j,i) or 4-D (t,k,j,i), time first
python3 scripts/plot_results.py --file pressure.mat --out figs
python3 scripts/plot_results.py --file fields.npz --key SWAT

# flat ASCII (whitespace or comma separated; ECLIPSE n*value repeats understood)
python3 scripts/plot_results.py --file perm.txt --dims 40 40 5
```

Array names are recognised automatically, `PRESSURE`, `press`, `SWAT`, `Sw`,
`SGAS`, `So`, `PERMX`, `phi` and similar all map onto the right label and colour
map.

### 11.4 What it produces

- **3-D voxel views** with every well drawn as a vertical line running the full
  model height and continuing above the top face, labelled and coloured by type
  (injectors black, producers red): the style of the original CuPy reference
  plotter, in orthographic projection at aspect $`[n_x, n_y, 2n_z]`$. Large
  grids are block-averaged down to a render budget first, so the cost stays
  bounded without distorting the picture.
- **Orthogonal slice panels**: areal plan ($`K`$), and cross-sections along
  $`I`$ and $`J`$, with wells annotated on the plan view.
- **Movies**: GIF always, plus MP4 when `ffmpeg` is available, for both the 3-D
  views and the slice panels.
- **Field maps** per snapshot with well markers and a colour scale shared across
  frames (so an animation shows real change, not autoscale churn).
- **Layer panels**: the final state across every $`K`$ layer.
- **Field evolution**: mean/min/max pressure, saturations, recovery factor.
- **Well response**: rates, water cut, cumulative production.
- **Solver convergence**: Newton and Krylov iterations per report step.
- **`index.html`**: everything tied together in one page.

By design it is **never fatal**: every renderer is individually guarded and the
script exits 0 unless `--strict` is passed, so a plotting problem can never turn
a good simulation into a failed job. A missing `matplotlib` degrades to a printed
text summary rather than a traceback; a missing `Pillow` or `ffmpeg` just means
fewer movie formats.

Modules: [`plot_results.py`](GPU_reservoir_simulator/cuda/scripts/plot_results.py)
(driver), [`viz3d.py`](GPU_reservoir_simulator/cuda/scripts/viz3d.py) (3-D,
slices, movies), [`loaders.py`](GPU_reservoir_simulator/cuda/scripts/loaders.py)
(array input), [`ecl.py`](GPU_reservoir_simulator/cuda/scripts/ecl.py) (ECLIPSE
reader).

---

## 12. Bayesian history matching (α-REKI)

The inverse problem: given observed well data $`\mathbf{d}`$, recover the
permeability/porosity fields $`\mathbf{m}`$ under the forward map
$`\mathcal{G}`$ (the simulator, or a trained surrogate of it). The posterior is

```math
\pi(\mathbf{m}\mid\mathbf{d}) \propto
\exp\!\Big(-\tfrac12\|\mathbf{d}-\mathcal{G}(\mathbf{m})\|^2_{\mathbf{C}_D^{-1}}
           -\tfrac12\|\mathbf{m}-\mathbf{m}_0\|^2_{\mathbf{C}_M^{-1}}\Big) .
```

**Regularising ensemble Kalman inversion** tempers the likelihood across steps
with inflation factors $`\alpha_i`$ satisfying $`\sum_i 1/\alpha_i = 1`$, so the
telescoping product recovers the full likelihood exactly. Each step is a Gaussian
conditional update

```math
\mathbf{m}_j \leftarrow \mathbf{m}_j
 + \mathbf{C}_{MD}\big(\mathbf{C}_{DD} + \alpha_i \mathbf{C}_D\big)^{-1}
   \big(\mathbf{d} + \sqrt{\alpha_i}\,\boldsymbol{\varepsilon}_j - \mathcal{G}(\mathbf{m}_j)\big).
```

**Adaptive inflation (α-REKI).** Rather than fixing the schedule, choose each
$`\alpha_i`$ from the current misfit. With
$`\Phi_j = \tfrac12\|\mathbf{d}-\mathcal{G}(\mathbf{m}_j)\|^2_{\mathbf{C}_D^{-1}}`$,
at the correct noise level $`2\Phi_j`$ is a $`\chi^2_{N_d}`$ variate, so
$`\mathbb{E}[2\Phi] = N_d`$ and $`\mathrm{Var}(2\Phi) = 2N_d`$, the discrepancy
principle. Matching the expected reduction gives $`\alpha \approx N_d/(2\mu_\Phi)`$;
controlling the spread gives $`\alpha \approx \sqrt{N_d/(2\sigma_\Phi^2)}`$.
Taking the more conservative and capping by the remaining budget:

```math
\alpha_i = \Big[\min\Big(\max\Big(\tfrac{N_d}{2\mu_\Phi},\ \sqrt{\tfrac{N_d}{2\sigma_\Phi^2}}\Big),\
  1-\textstyle\sum_{k<i}\tfrac{1}{\alpha_k}\Big)\Big]^{-1} ,
```

so the final step lands exactly on $`\sum_i 1/\alpha_i = 1`$. Large early
$`\alpha_i`$ (small steps) when the misfit is huge; $`\alpha_i \to 1`$ as the
ensemble approaches the data.

```bash
cd GPU_reservoir_simulator/cuda/hm
python3 run_history_match.py --config config.yaml
```

See [`hm/README.md`](GPU_reservoir_simulator/cuda/hm/README.md).

---

## 13. Verification and testing

| Test | Command | What it proves |
|---|---|---|
| Build + smoke | `./scripts/ci_smoke_test.sh` | Toolchain, HYPRE link, a full tiny run, outputs written |
| Jacobian cross-check | `./scripts/run.sh deck.DATA --jac-check` | Analytic derivatives match the finite-difference reference to $`\mathcal{O}(\varepsilon)`$ |
| Rank invariance | stage 7 of the smoke test | The K-slab decomposition and halo exchange are correct: 1-rank and 2-rank results agree |
| Precision equivalence | `EMULATE 0` vs `EMULATE 1` | df64-emulated FP64 reproduces native FP64 |
| Preconditioner equivalence | `MIXED 0` vs `MIXED 1` | FP32 preconditioning changes the rate, not the answer |
| ECLIPSE format round-trip | `python3 scripts/ecl.py OUTPUT/NVRS.EGRID` | The binary output parses back to the values written |
| Cross-simulator | `benchmark/run_comparison.sh` | Fields and well rates agree with OPM Flow on a deck matched term by term, see §13.1 |
| Physics regression | compare against `GPU_reservoir_simulator/RESULTS/` | Fields and well rates match the CuPy reference |

The last one is the one that matters for production use and requires running the
reference cases (33×33×1, 40×40×3, 40×60×5) on real hardware.

### 13.1 Head-to-head against OPM Flow

`GPU_reservoir_simulator/cuda/benchmark/` runs this simulator and
[OPM Flow](https://opm-project.org/) on the same case and writes a **second**
HTML report, `benchmark/runs/compare.html`, separate from the `index.html` a
single run produces:

```bash
cd GPU_reservoir_simulator/cuda/benchmark
./install_opm_gpu.sh        # build Flow 2024.10 with CUDA
./run_comparison.sh         # run both, time both, write runs/compare.html
```

The report carries, per case: a wall-clock table with the hardware and rank
count each simulator used; one figure per well quantity (WOPR, WWPR, WWCT,
WGOR, WBHP) overlaying both simulators for every producer with the per-well
difference beneath; a per-report-step error table; and, for every report step, a
3 × 4 grid of **3-D** renderings: columns OPM Flow / NVRS-CUDA / difference,
rows $`p`$, $`S_w`$, $`S_o`$, $`S_g`$.

Both decks are emitted from one parameter set by `benchmark/make_decks.py`, and
the two formulations are matched term by term: constant PVT via the new
**`PVTCONST`** keyword (the correlations in `pvt.hpp` are a port of the CuPy
reference and are not a dimensionally consistent field-unit property set, so
they cannot be transcribed into an ECLIPSE table), zero compressibility to match
NVRS's $`V\phi/\Delta t`$ accumulation, `RS 0` with no `DISGAS`, densities
chosen so $`\rho_{\text{surf}}/B`$ agrees with the surface density NVRS uses in
the gravity head, and the well index written explicitly into `COMPDAT` item 8.

Two cases are shipped to isolate the phase paths. The former three-phase
relative-permeability difference has been removed: both current decks use the
ECLIPSE/OPM saturation-weighted $`k_{ro}`$ form.

- **`BENCH2P`** (oil/water) is *exact*. NVRS's two-phase kernel uses analytic
  Corey curves and never reads `SWOF`; the ECLIPSE deck tabulates precisely
  those curves to $`6\\times10^{-6}`$. Every term matches, so any difference is
  discretisation, linear solver and time stepping alone. This is the case that
  measures the simulator.
- **`BENCH3P`** (oil/water/gas) exercises the common three-phase path and
  supplies the gas comparison. A current discrepancy belongs to properties,
  controls, nonlinear stepping or the linear solve, not a known $`k_{ro}`$
  convention mismatch.

See `benchmark/README.md` for the full derivation.

---

## SPE10 model 2

The Tenth SPE Comparative Solution Project, model 2 (Christie & Blunt, SPE
72469) is the standard stress test for algebraic multigrid on reservoir
pressure systems: **60 × 220 × 85 = 1,122,000 cells** at 20 × 10 × 2 ft, the
geostatistical Brent-sequence permeability field with a **3 × 10⁷** contrast,
two-phase, one central injector against four corner producers, 2000 days. It
was built as an *upscaling* challenge, most of the original participants could
not run the fine grid at all.

```bash
./scripts/backend_study.sh spe10   # timings, into the study report
./scripts/run_spe10.sh             # full run + benchmark/spe10/spe10.html
./scripts/spe10_tune.sh            # solver A/B, one variable per row
```

SPE10 is the **sixth case of the backend study**, alongside `bench3p` and the
1M/2M/4M/5M ladder, and is included in `backend_study.sh all`. It sweeps one
GPU where the ladder sweeps 1–4, for the reasons under *Why single-GPU* below.

`run_spe10.sh` is the deep dive rather than the timing run: it writes
`benchmark/spe10/spe10.html` (the rock, a flood animation across all twenty
report steps, well responses, timings and the log) plus
`benchmark/spe10/spe10_run.log` with the complete run.

`benchmark/make_spe10.py` downloads the public dataset (in the ECLIPSE include
form redistributed by OPM's `opm-data` under ODbL) into
`cuda/dataset/SPE10/` on first use: 77 MB, git-ignored, downloaded to a
temporary name and renamed only when complete, so an interrupted fetch cannot
masquerade as the dataset. It then writes the deck and the per-cell rock files.

Two things make the case faithful rather than approximate. The spec's quadratic
Corey curves with `Swc = Sor = 0.2` are **analytically identical** to the
two-phase kernel (`krw = S²`, `kro = (1−S)²`) at `SWI = SWR = 0.2`, so no
saturation table is involved on either side. And the generator verifies
`ky == kx` in the published data before writing anything, rather than assuming
the isotropy NVRS relies on. Every simplification it *does* make (dead oil with
constant B, surface-rate rather than voidage injection, adaptive substepping in
place of the reference deck's fixed TSTEP ramp) is listed in its header.

Reading SPE10 required two simulator features that are now defaults:
per-cell **`POROFILE`** (SPE10's porosity is an independent measurement, not a
function of permeability) and **`PERMZFILE`** (its kv/kh is not a scalar).

### The result

Each row below is a measured A/B against the row above it, on one GB200:

| Configuration | Wall clock (2000 days) | Speed-up |
|---|---:|---:|
| Aggregation hierarchy, fully implicit | 861.3 s | 1.00× |
| Classical PMIS hierarchy (§5.4) | 402.6 s | 2.14× |
| **+ adaptive implicit (§3.4)** | **342.1 s** | **2.52×** |

*That ablation was measured earlier on the 2000-day SPE10 configuration. On the matched SPE10 deck (1000 days, CPRW, one Eclipse deck read by both simulators) SPE10 model 2 completes in **53.06 s** on one GB200 and **41.21 s** on one VR72 under AmgX, 18.5x faster
than OPM Flow on the VR72 host, with a mean per-well R^2 of 0.999.*

Both are now defaults, so a plain `mpirun -np 1 ./nvrs_cuda SPE10.DATA`
produces the last row.

### The candidates that lost

Recorded because negative results are results, and each cost a measurement:

| Candidate | Outcome |
|---|---|
| 2 pre/post smoothing sweeps | 235 s vs 287 s baseline: cheaper cycles, unchanged convergence |
| `SIZE_4` aggregates | 197 s, same |
| `MULTICOLOR_DILU` smoother | Converges in far fewer iterations; smoother itself ~3× dearer, net loss |
| Two V-cycles per application | **Diverged**: the timestep controller could not recover |
| `RESTART 60` | CGS2 *more* expensive per iteration at a third the iteration count |
| Appleyard endpoint chop | 143 s vs 120 s: more Newton iterations, no larger timesteps |
| `STAGE2 ZLINE` | Krylov per Newton 61.5 → 45.6, but the kernel costs 3× block-Jacobi: one thread per areal column is only 13,200 threads |

Reproduce any of them with `./scripts/spe10_tune.sh <candidate>`.

### Why single-GPU

SPE10 is reported on one GPU, which is how published numbers for it are quoted.
At 1.12 M cells the field is below the ~5 × 10⁵ cells-per-GPU threshold of §7
where decomposition starts to pay, and multiple ranks would additionally force
the weaker aggregation hierarchy (§5.4). The size ladder in
`benchmark/backend_study/` is where the scaling story is told.

---

## 14. Status, validation and provenance

### What was audited and repaired

This codebase was reviewed and corrected as part of preparing it for release. The
findings are recorded here because they matter to anyone who trusts the results.

**The code had never been compiled.** `linear_solver.hpp` forward-declared
`struct hypre_amg_impl` at global scope while `cpr_solver.cu` defined it inside
`namespace nvrs`, making them distinct types, a hard compile error on the first
build. All eight translation units now compile clean and device-link under
`nvcc -rdc=true`.

**The MPI halo exchange did not exist.** Both `grid.hpp` and `cpr_solver.cu`
documented a one-layer K halo exchanged each timestep. There was no such code:
`build_sparsity` marked every off-rank K neighbour as no-flow. Multi-GPU and
multi-node runs therefore solved $`N`$ decoupled reservoirs and produced
rank-count-dependent answers, silently. This is now implemented
(`halo.hpp`/`halo.cu`, ghost index space, rectangular local operator, global
column ids for HYPRE) and guarded by a rank-invariance regression test.

Also fixed:

| Defect | Consequence |
|---|---|
| Cell depth computed from the *local* K index | Gravity referenced each rank's slab origin, not the global datum |
| Race in `k_scatter_p` | The AMG pressure correction was randomly zeroed depending on block scheduling |
| `k_stage2` fixed `rr[8]` | Stack overrun for compositional runs with $`b_s = n_c + 1 = 9`$ |
| Compositional volume balance written as $`S_o + S_g - 1`$ | Identically zero by construction → structurally singular pressure block |
| Compositional initial state never seeded from `ZMF` | Newton started from an empty reservoir |
| Well-rate post-processing used `PERMMIN` and omitted the Darcy constant | Reported rates wrong by ~900× and inconsistent with the simulated field |
| `NCOMP` unvalidated against the kernels' fixed-size storage | Out-of-bounds access for `NCOMP > 8` |
| `ensure_dir` used `_mkdir` without `<direct.h>` | Did not compile on Windows; could not create nested output paths |
| The pressure matrix was handed to HYPRE from **host** pointers while `HYPRE_MEMORY_DEVICE` was selected | HYPRE dereferenced host addresses as device memory: `CUDA ERROR (code = 1, invalid argument)` in HYPRE's `memory.c`, then a bogus thrust `invalid device ordinal`. Every GPU run died at the first linear solve. |
| GPU binding used the global MPI rank, ignored both CUDA return codes, and never forced context creation | Wrong device on multi-node; a failed bind surfaced far away inside HYPRE instead of at the call |
| A rank throwing an exception unwound alone | The other ranks blocked forever in the next collective, so a clear error looked like a hang |
| `AUTOPLOT` hardcoded `python3 scripts/plot_results.py` | Plotting only worked from one specific working directory |

### Validated against OPM Flow

The open question in earlier revisions of this document (*does it agree with a
reference simulator?*) is now closed. `GPU_reservoir_simulator/cuda/benchmark`
runs NVRS-CUDA and [OPM Flow](https://opm-project.org/) on decks generated from
one parameter set, matched term by term, and reports the difference field by
field and well by well.

On `BENCH3P` at 46 × 112 × 22 (113,344 cells), both simulators on one GB200:

| Measure | OPM Flow | NVRS-CUDA |
|---|---|---|
| Wall clock | 161.60 s | **38.67 s** |
| Water in place at 2000 d | 3.652 Mbbl | 3.590 Mbbl |
| Implied water injected vs the 4 Mbbl the deck asks for | −0.8 % | −0.5 % |
| Oil recovery factor | 41.8 % | 45.0 % |

Water is conserved to under 1 % on both sides, which is the test that says the
two are solving the same system rather than two plausible-looking ones. These
figures predate the three-phase correction and are retained as archived
benchmark provenance, not as the current Norne parity result. `BENCH2P`
isolates the oil/water path; its two-phase agreement is 0.305 % of range in
$`S_w`$ RMS. Current `BENCH3P` runs use the common ECLIPSE/OPM
saturation-weighted $`k_{ro}`$ form.

Reproduce with:

```bash
cd GPU_reservoir_simulator/cuda/benchmark
BENCH_GRID="46 112 22" ./run_comparison.sh BENCH3P
```

See [benchmark/README.md](GPU_reservoir_simulator/cuda/benchmark/README.md) for
what is matched, what cannot be, and how to read the wall clock honestly.

`./scripts/ci_smoke_test.sh` remains the fast gate: it runs the
analytic-vs-numerical Jacobian cross-check and the 1-rank vs 2-rank invariance
test, and it is what to run first on any new node.

### Scope of the model

These are not defects; they are the boundaries of what is implemented, and they
belong next to any number taken from this code.

1. **The compositional EOS is a reference closure**: Wilson $`K`$-values,
   ideal-gas vapour density, constant liquid molar volume. The data path is
   complete (deck EOS keywords, $`n_c`$-component transport, wells, and CPR at
   $`b_s = n_c + 1`$); swap in a full Peng–Robinson flash for quantitative
   compositional accuracy.
2. **Compositional Jacobians are numerical.** Analytic derivatives are
   implemented for two- and three-phase black oil and cross-checked against the
   finite-difference path by `--jac-check`.
3. **Well controls are static.** There is no rate/BHP switching: an injector
   holds its rate and a producer holds its BHP for the whole run, whatever the
   field pressure does. Check that the deck's `PWF` gives a sensible voidage
   balance, see the comments in `decks/FIELD.DATA`.
4. **Decomposition is 1-D (K-slabs).** The rank count cannot exceed `NZ`, and
   the surface-to-volume ratio degrades at high rank counts on thin models.
   Separately, and more often the binding constraint in practice: below roughly
   500,000 cells per GPU the parallel BoomerAMG setup costs more than the split
   saves, so a small model belongs on one GPU regardless of `NZ`.
5. **`PERMFILE` reading is serial on every rank**: each rank reads the whole
   global file and slices its own slab.

---

## Norne/OPM parity release (version 2 results)

The CUDA C++ simulator now carries an explicit OPM-parity path for the metric
Norne history case. The release preserves the central mixed-precision design:
the nonlinear residual and outer Krylov convergence are evaluated in FP64,
while AmgX or HYPRE supplies the FP32 CPR pressure preconditioner. No parity
correction changes that precision boundary.

### Discrete conservation and connection flow

For phase $\alpha\in\{w,o,g\}$ in cell $i$, backward Euler gives

```math
R_{\alpha,i}^{n+1}=
\frac{V_i}{\Delta t}\left(M_{\alpha,i}^{n+1}-M_{\alpha,i}^{n}\right)
+\sum_{j\in N(i)}F_{\alpha,ij}^{n+1}-Q_{\alpha,i}^{n+1}=0.
```

The TPFA connection flux uses phase-potential upwinding,

```math
F_{\alpha,ij}=T_{ij}\lambda_{\alpha,\mathrm{up}}
\left[(p_{\alpha,i}-p_{\alpha,j})-\rho_{\alpha,\mathrm{up}}g(z_i-z_j)\right],
```

with $\lambda_\alpha=k_{r\alpha}/\mu_\alpha$. The ECLIPSE/OPM default
three-phase oil curve is the saturation-weighted interpolation between the two
oil two-phase curves; it replaces the earlier Baker-product interpretation.
Endpoint scaling applies its coordinate transform and chain rule consistently
to residual and Jacobian. Carlson model-0 hysteresis stores the maximum gas
saturation and shifts the non-wetting gas curve continuously on reversal.

For completion $c$, the signed Peaceman phase rate is

```math
q_{\alpha,c}=WI_c\lambda_{\alpha,c}
\left[p_c-p_{bh}-\rho_{\alpha,c}g(z_c-z_{ref})\right].
```

Summing live completions before enforcing the scalar target is essential: a
well target belongs to the whole well, not independently to every completion
or MPI rank. A history target of exactly zero is an active shut condition, not
a missing control. WWCT and WGOR are reported from the same signed surface
phase rates used by the well ledger,

```math
WWCT=\frac{q_w}{q_o+q_w},\qquad WGOR=\frac{q_g}{q_o},
```

only where their denominators are physically valid; shut/near-zero points are
masked rather than turned into artificial steps or spikes.

### OPM-style history RESV conversion

`WCONHIST RESV` supplies observed surface oil, water and gas rates. It is not a
ready-made reservoir-volume rate. For a PVT region evaluated at its
hydrocarbon-pore-volume-weighted average state, the reservoir voidage target is

```math
q_{RESV}=B_w q_w+B_o q_o+B_g\max(q_g-R_s q_o,0),
```

with the corresponding live-oil/wet-gas coupling used when vaporised oil is
active. The regional state is computed once per report step and cached; doing
this work inside each Newton iteration both changes the control while solving
it and caused the severe slowdown seen in intermediate experiments.

For a producer controlled to scalar total $Q_t$, define a signed connection
coefficient $C_c=WI_c\sum_\alpha\lambda_{\alpha,c}$ and phase/gravity offset
$H_c$. The closed-form well pressure used by the schedule path is

```math
p_{bh}=\frac{\sum_c(C_c p_c-H_c)-Q_t}{\sum_c C_c}.
```

This preserves the target across active completions without adding a new CUDA
Jacobian kernel. `GCONINJE RATE` is then applied as a phase-specific ceiling:
if the stated injector targets in group $G$ sum above $Q_G$, each is multiplied
by $\gamma_G=Q_G/\sum_{i\in G}Q_i$. `RESV`, `REIN` and `VREP` group-injection
modes are parsed but are not yet enforced.

### Units, measured result and remaining error

Norne is metric. Liquid surface rates are converted from sm³/day using
$1\ \mathrm{m^3}=6.28981077\ \mathrm{stb}$; gas rates use the standard-gas
volume conversion rather than the liquid factor. Pressures and completion
hydrostatic offsets are likewise converted exactly once at the parser boundary.

| 247-step run, one GB200 | Wall time | Linear time | Fraction linear |
|---|---:|---:|---:|
| OPM Flow | 488.51 s | 97.22 s | 21% |
| NVRS-AmgX | **44.95 s** | 23.60 s | 54% |
| NVRS-Ginkgo FP16 | **80.55 s** | 56.30 s | 70% |
| NVRS-HYPRE | **71.51 s** | 49.91 s | 70% |

The physical curves of the solver arms are essentially coincident, so the
differences between them are the multigrid alone. At 3260 days the implied water
injection is 664.8 M STB in Flow and 656.7 M STB in NVRS; cumulative oil is 425.07 M STB against Flow's 422.78 (0.54 %), and the field pressure RMS difference is 327 psi at 3260 days. 19 of the 22 producers match Flow at R^2 >= 0.70 on all four rate quantities, 21 on water cut and all 22 on gas rate; the
per-producer table is at the top of this page.

Per-well R² is retained but is not interpreted as a percentage. A nearly
constant or mostly shut OPM trace has negligible variance and can yield a huge
negative R² for a small absolute error. Release decisions therefore combine
R² with active-time NRMSE, cumulative-volume error, event-date error and a
denominator-validity flag. The full derivation, per-well triage table,
experiment matrix and regression checklist are in the 100-page
[`docs/manual.pdf`](../docs/manual.pdf),
Chapter 10.

---

