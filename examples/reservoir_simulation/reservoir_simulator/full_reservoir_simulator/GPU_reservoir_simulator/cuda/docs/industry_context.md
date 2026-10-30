# Industry context for mixed-precision GPU reservoir simulation

Last updated: 15 September 2026

## Purpose and positioning

This work is not intended to propose NVRS-CUDA as a replacement for commercial
reservoir simulators such as the SLB INTERSECT simulator or Stone Ridge
Technology's ECHELON software. Those products have broad physics, field
management, workflow integration, quality assurance and commercial support
built over many releases.

The purpose of this project is to promote and investigate **GPU-accelerated
reservoir simulation**, with particular emphasis on:

- mixed FP64/FP32 numerical methods;
- FP64-quality operator evaluation on accelerators with limited native FP64;
- CPR-AMG implementations that exploit GPU memory bandwidth;
- multi-GPU execution; and
- the performance characteristics of new architectures, including NVIDIA Vera
  Rubin-class systems.

INTERSECT and ECHELON are therefore industry context and evidence that
full-GPU reservoir simulation is an important direction. They are not targets
that this research code claims to replace.

## Executive assessment

NVRS-CUDA implements an industry-recognizable numerical core: fully implicit
Newton, adaptive implicitness, CPR, AMG, flexible GMRES, GPU-resident kernels
and MPI decomposition. Its distinguishing research contribution is the
explicit treatment of precision: FP32 is used where bandwidth dominates, while
the operator, true residual, Krylov reductions and convergence-defining work
retain FP64 quality, either natively or through the df64 emulation path.

The current Norne comparison supports a much stronger accuracy statement than
older repository summaries did. It should now be described as a credible
field-scale verification result, not as a simulator with a large production
forecast discrepancy. The proper qualification is that agreement is strong at
the field and material-balance level while some individual well histories and
local cell extrema remain less closely matched.

## Relationship to commercial GPU simulators

| Topic | NVRS-CUDA research focus | INTERSECT context | ECHELON context |
|---|---|---|---|
| GPU execution | CUDA assembly and solve, GPU-resident state, MPI multi-GPU | Full-GPU black-oil implementation and multi-GPU execution | GPU-first implementation, multi-GPU and multi-node execution |
| Nonlinear methods | Fully implicit Newton and adaptive implicit method | Commercial formulations including fully implicit and multiscale SFI | Fully implicit and adaptive implicit methods |
| Linear solver | FGMRES/BiCGSTAB with CPR; FP32 AmgX or HYPRE AMG | Bespoke GPU linear solver; public precision details are limited | GMRES with CPR-AMG and automatically selected second stage |
| Precision question | Explicit native-FP64, mixed FP64/FP32 and df64-emulated studies | Internal precision strategy is not described publicly | Internal precision strategy is not described publicly |
| Main contribution | Reproducible research into accuracy, bandwidth and new GPU architectures | Broad commercial reservoir and field-management platform | Mature GPU-native commercial reservoir simulator |

The shared use of Newton, Krylov methods, CPR-AMG and adaptive implicitness
shows that this project's numerical direction is aligned with industrial
practice. The commercial tools validate the importance of GPU acceleration;
this project adds an open, inspectable study of *where precision can safely be
reduced and where FP64-quality arithmetic must be retained*.

## Current Norne evidence

The current report was generated on 15 September 2026 from OPM Flow 2025.10
and NVRS-CUDA outputs using the same ECLIPSE reader and matched rock properties.
The complete evidence is in [the Norne comparison report](../../../../results/GB200/compare_NORNE.html).

### Field and recovery agreement

- Cumulative oil: **422.71 MMSTB** for OPM Flow and **425.64 MMSTB** for
  NVRS-CUDA, a difference of about **0.69%** relative to Flow.
- Recovery factor: **41.8%** for Flow and **41.5%** for NVRS-CUDA.
- Remaining oil inferred from the saturation field: **550.83** versus
  **548.33 million reservoir bbl**.
- At the final aligned report time (3260 days), RMS differences normalized by
  the contemporaneous field range are **5.14%** for pressure, **1.60%** for
  water saturation, **2.02%** for oil saturation and **1.85%** for gas
  saturation.
- The regenerated report gives water-inventory closure errors of **+3.6%** for Flow
  and **+6.8%** for NVRS-CUDA. The merged reporting and volume-accounting definition
  changed both values, so this is not a like-for-like change in simulator accuracy;
  the NVRS diagnostic is now above the report's 5% warning threshold.

These numbers supersede the old README statement that compared 286.35 MMSTB
with 422.38 MMSTB. That statement described an earlier implementation before
the recent schedule, well, fluid-property and accounting corrections.

### Well-level qualification

The field totals and material balance now agree closely, but the result should
not be described as bitwise or per-well identical. In the current active-period
diagnostic, 64 of 132 well/quantity rows fall below an R-squared threshold of
0.80. Several are short-lived or nearly constant histories for which raw
R-squared is unstable; the report therefore also supplies NRMSE, cumulative
rate error and opening/closing shifts. These diagnostics identify remaining
model-semantics work without invalidating the field-scale accuracy result.

## Current performance evidence

For the same current Norne report:

- OPM Flow 2025.10: **205.28 s** on one CPU rank with two OpenMP threads.
- NVRS-CUDA with AmgX: **32.58 s** on one GPU.
- Observed case speed-up: **6.30x**.
- NVRS linear solve: **16.42 s**, of which **6.92 s** is AMG.

This demonstrates the potential of the GPU implementation on this case. It is
not an INTERSECT-versus-NVRS or ECHELON-versus-NVRS benchmark. Cross-product
performance claims would require the same deck, accepted physics, hardware,
convergence criteria, timestep path and output frequency.

The repository's generated scaling cases reach five million cells and show
2.81x strong scaling from one to four GB200 GPUs with AmgX. These controlled
cases study the solver and communication path; they are not substitutes for
the asset-model validation and very-large-model demonstrations published by
commercial vendors.

## What the commercial products establish

SLB documents full-GPU black-oil execution in INTERSECT, including GPU property
calculation, equation linearization and a bespoke solver. Its published
2.3-million-active-cell example ran three times faster on one A100 than on 48
CPU cores while producing the same CPU/GPU result. Current INTERSECT releases
also support multi-GPU execution. This shows that GPU acceleration is
compatible with a mature commercial simulation workflow.

Stone Ridge Technology describes ECHELON as GPU-first and reports the same
broad solver family used here: GMRES, CPR and AMG, with both FIM and AIM.
Published ECHELON studies demonstrate multi-GPU black-oil and compositional
simulation. Its expansion to current NVIDIA and AMD accelerators reinforces
the project's thesis that reservoir-simulation performance is strongly tied to
accelerator memory bandwidth and architecture.

Vendor figures are useful context but are not directly comparable with NVRS
because the models, physics, hardware and convergence paths differ.

## Recommended claim

> NVRS-CUDA is a research platform for demonstrating accurate,
> bandwidth-efficient reservoir simulation on modern GPUs. It combines an
> industry-standard fully implicit CPR-AMG formulation with mixed-precision and
> FP64-emulation strategies designed for present and next-generation
> accelerators. Current Norne results reproduce field recovery and material
> balance closely while running substantially faster than the CPU reference on
> the measured system. Commercial GPU simulators such as INTERSECT and ECHELON
> demonstrate the industrial importance of this direction; NVRS-CUDA
> complements them by exposing and quantifying the numerical role of precision.

Avoid claims that NVRS-CUDA is a drop-in replacement for a commercial
simulator, that it is faster than INTERSECT or ECHELON, or that every cell and
well response is identical. None is required to make the research case.

## Strategic research value

The strongest contribution is the connection among algorithm, arithmetic and
hardware:

1. Reservoir Jacobians and AMG cycles are often bandwidth-bound.
2. Narrower storage and arithmetic can reduce that bandwidth cost.
3. Flexible Krylov methods permit an inexact low-precision preconditioner.
4. FP64-quality true residuals prevent false convergence.
5. Emulated FP64 provides a path for architectures whose low-precision
   throughput grows faster than native FP64 throughput.

That makes Vera Rubin and later accelerator generations scientifically
interesting even when current native-FP64 systems show only modest gains from
precision reduction in the isolated linear solve.

## Sources

Local evidence:

- [Current Norne comparison](../../../../results/GB200/compare_NORNE.html)
- [NVRS-CUDA user manual](manual.pdf)
