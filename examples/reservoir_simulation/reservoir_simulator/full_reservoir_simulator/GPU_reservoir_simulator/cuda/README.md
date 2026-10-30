# NVRS-CUDA: GPU Black-Oil / Compositional Reservoir Simulator (CUDA C++)

> Install, run commands, measured results and the paper are in the [top-level README](../../../README.md).

A CUDA C++ translation of the CuPy `GPU_reservoir_simulator` (NVRS), built around
a **mixed-precision CPR** linear solver that reuses the exact algorithm validated
in `../../Mixed_precision`. The multigrid inside CPR comes from **AmgX**
(default) or **HYPRE BoomerAMG**, chosen with one deck keyword. It runs from a
single GPU to multi-GPU nodes to multi-node clusters, reads an OPM/Eclipse-style
deck, and renders plots and an insight summary automatically.

> Author: **Clement Etienam**. Physics, discretisation and solver mirror the
> Python reference one-to-one; only the execution layer changed (CuPy → CUDA
> C++ + AmgX/HYPRE + MPI).

---

## What it does

| Capability | How |
|---|---|
| Physics | Two-phase (oil/water) & three-phase (oil/water/gas) **black oil**, plus a new **compositional** (nc-component, Wilson-K + Rachford-Rice flash) model. `MODEL BLACKOIL` / `MODEL COMPOSITIONAL` in the deck selects it (OPM style). |
| Discretisation | Structured Cartesian TPFA finite volume, phase-potential upstream weighting, fully-implicit (FIM) Newton. |
| Linear solver | **CPR is always the preconditioner**: multigrid on the pressure block + a `bs×bs` block-Jacobi second stage. The **multigrid** is `LINSOLVER AMGX` (default, aggregation AMG), `LINSOLVER HYPRE` (BoomerAMG) or `LINSOLVER GINKGO` (Ginkgo PGM aggregation AMG in **FP16**, one GPU; `GINKGOPREC 16` or `32`). The **outer Krylov** is `KRYLOV FGMRES` (default) or `KRYLOV BICGSTAB`. |
| Mixed precision (default) | **FP32** multigrid + stage-2 (`MIXED 1`) with the **FP64 operator evaluated in double-word (df64) arithmetic** on FP32 ALUs (`EMULATE 1`). Set `EMULATE 0` for a native-FP64 operator. The precision boundary does not move with `LINSOLVER`: AmgX runs in `dFFI` mode and the HYPRE build is FP32, so both are FP32 inside an FP64 Krylov. Adapted from `hypre_cusolver_mixed_cpr.cu`. |
| No hard-coding | Every physical constant is deck-driven with a documented default: `DARCY`, `GRAVGRAD`, `REFACTOR`, `FDEPS`, `ZFACTOR`, reservoir `TEMP`, and the compositional EOS tables `TCRIT`/`PCRIT`/`ACF`/`VMOL`. Rock fields load from `PERMFILE` when supplied. |
| Parallelism | MPI 3-D `px×py×pz` domain decomposition, one GPU per rank: 1 GPU · single-node multi-GPU · multi-node multi-GPU. Halo exchanges pass device pointers straight to MPI, verified at start-up rather than assumed. |
| Output | Field snapshots (`state_*.bin` + `meta.json`), well rates (`RSM_FVM.csv`), auto-rendered PNG/GIF plots and a written `summary.txt` with insights. |

## Layout

```
cuda/
  include/            headers (deck, grid, pvt, fluid, physics, linear_solver, io)
  src/
    main.cu                 driver: deck → decompose → Newton+CPR → output
    deck.cpp                OPM/Eclipse-style keyword parser
    physics_host.cu         reservoir setup, sparsity/topology, Newton update, PVT
    assemble_blackoil.cu    FIM black-oil residual + numerical Jacobian
    assemble_compositional.cu  nc-component flash + residual/Jacobian
    cpr_solver.cu           CPR + FGMRES/BiCGSTAB (mixed precision), HYPRE path
    amgx_backend.cu         AmgX stage-1 multigrid (compiles out without AmgX)
    io.cu                   snapshots, RSM CSV, summary
  decks/              RESERVOIR.DATA (black oil), COMPOSITIONAL.DATA
  scripts/            run.sh, slurm_multinode.sh, plot_results.py
  CMakeLists.txt      GEOS-style HYPRE link
  Makefile            single-command nvcc build
```

## Build

`./nvrs.sh install` in `benchmark/` installs the libraries and builds the simulator.
To rebuild only the simulator afterwards, source the environment files it wrote
(`../../Mixed_precision/hypre_env.sh`, `../../amgx-install/amgx_env.sh`,
`../../ginkgo-install/ginkgo_env.sh`), then:

```bash
make -j                                       # nvrs_cuda, mixed precision
make -j FP64_PRECOND=1                        # nvrs_cuda_fp64, fully FP64 preconditioner
make -j TENSORCORE=1 CUTLASS_DIR=<checkout>   # nvrs_cuda_tc, adds JACOBIAN TENSORCORE
```

The CUDA architecture is detected from your GPU; `CUDA_ARCH=100` pins it. The scripts
run the binary `make` writes in this folder. Read the AmgX line the build prints:
`AmgX: not found -- HYPRE only` means a run labelled AmgX would measure BoomerAMG.

## Run

> **Always launch through `scripts/run.sh`, never the bare binary.** It calls
> `mpirun` even for a single rank. A singleton `MPI_Init`, what you get by
> running `nvrs_cuda` directly, breaks UCX on a cluster with
> `"UCP version is incompatible"`, and inside a Slurm allocation it can take
> the whole step down with it. `run.sh` forwards any extra flags, so
> `--jac-check` and friends work unchanged.

```bash
# single GPU
NGPU=1 ./scripts/run.sh decks/RESERVOIR.DATA
# all GPUs on the node
./scripts/run.sh decks/RESERVOIR.DATA
# compositional case, BiCGSTAB outer Krylov
./scripts/run.sh decks/COMPOSITIONAL.DATA
# multi-node (4 nodes × 4 GB200)
sbatch scripts/slurm_multinode.sh decks/RESERVOIR.DATA
```

Always launch through `scripts/run.sh`. The simulator is an MPI program;
started directly it initialises no transport and exits with a message from UCX
that says nothing about the deck.

Plots and `summary.txt` are written to the deck's `OUTDIR` and rendered
automatically by `scripts/plot_results.py` at the end of the run.

### Field decks (corner-point)

A corner-point deck (the format the industry's simulators share) is read
directly, unmodified. Name it from a run deck with `FIELDDECK` and launch it
like any other case. Norne is the reference model:

```bash
./benchmark/get_norne.sh                    # 61 files, ~26 MB, on demand
NGPU=1 ./scripts/run.sh decks/NORNE.DATA
```

That gives corner-point geometry, 61 faults with the non-neighbour connections
their throws create, region arrays, equilibration from each region's datum, and
36 wells whose controls change across 247 dated report steps. In version 2 on one GB200, OPM Flow took **488.51 s**, NVRS-AmgX **44.95 s** (10.87× faster; 46.75 s and 10.45× as the median of three runs in a second session), NVRS-Ginkgo FP16 **80.55 s** (6.06×) and NVRS-HYPRE **71.51 s** (6.83× faster); on one VR72, 243.88 s, 35.73 s, 50.02 s and 73.54 s. Every producer's R^2 against Flow is in
the table at the top of this page and in the manual.

**The field deck's saturation functions are used, not the run deck's.**
`FIELDSATFUNC` defaults to 1 for a reason worth stating plainly: a run deck
that names a field deck is describing that field, so its own `SWOF`/`SGOF` are
at best a placeholder, and benchmarking against a simulator that reads the real
tables would compare two different rocks. The column order differs between the
two formats, ECLIPSE writes `Sw krw krow`, this deck format puts the oil curve
second, and both orders look like plausible curves, so the run prints the four
corner values at startup and every one must be the large end of its curve:

```
[satfunc] krw(Sw_max) …, krow(Sw_min) …, krg(Sg_max) …, krog(Sg_min) …
```

Three switches control how much of the field deck is taken, and two of them
default off because they change the model rather than just refine it:

| Keyword | Default | Effect |
|---|---|---|
| `FIELDSATFUNC` | **1** | Take `SWOF`/`SGOF` from the field deck. `0` keeps the run deck's. |
| `PROPSARRAYS` | 0 | Also read per-cell arrays from `PROPS`/`REGIONS`/`SOLUTION`. This is what makes the ENDSCALE endpoints visible, and it also hands equilibration the deck's `SWL` and `EQLNUM`, which on Norne moves ~3000 cells onto a gas cap they did not have. Results across the two settings are not comparable. |
| `USEENDSCALE` | 0 | Scale each cell's relperm curves onto its own endpoints. Needs `PROPSARRAYS 1`. Honours the deck's `SCALECRS`, two-point, or three-point when it says `YES`, as Norne does. |

Capillary pressure (`SWOF`/`SGOF` column four), PVT regions, Carlson non-wetting
gas hysteresis (`EHYSTR` model 0), producer group control
(`GRUPTREE`/`GCONPROD`, shared by productivity), and phase-specific
`GCONINJE RATE` ceilings are implemented. `GCONINJE RESV`, `REIN` and `VREP`
are parsed but not enforced; other hysteresis models and oil/water scanning
curves remain outside scope.

To check what a deck reads as, without touching a GPU:

```bash
./scripts/build_deck_tests.sh dataset/NORNE/NORNE_ATW2013.DATA
./build/deck/deckinfo dataset/NORNE/NORNE_ATW2013.DATA --gradients 0.0706,0.1001,0.0186
```

`deckinfo` reports what each reader understood **and every keyword it did not**,
which is the fastest way to separate a deck problem from a solver problem.
Both build with a plain host compiler: no CUDA, no MPI.

## Documentation

**[`docs/manual.pdf`](../../../docs/manual.pdf) is the user reference manual**: 100
pages covering every deck keyword with its default, the numerical methods,
adaptive implicit, parallel execution, output formats, the benchmarks, a
complete Norne/OPM parity derivation and a troubleshooting chapter. Start
there.

## Deck keywords (quick reference)

`FIELDDECK GRADOIL GRADWATER GRADGAS SWCONNATE USESCHEDULE CONNLIST` (corner-point path) ·
`MODEL` `NCOMP` · `NX NY NZ DX DY DZ KZMULT` · `PERMFILE POROFILE PERMZFILE` ·
`UW UO UG BW BO BG RHO* CF* SWI SWR
PINIT PBUB PATM SW/SG/SOINIT GRAVITY` · `SWOF` / `SGOF` tables · `ZMF` (feed) ·
`AIM AIMCFL` · `LINSOLVER {AMGX|HYPRE|GINKGO}` `GINKGOPREC {16|32}` `KRYLOV {FGMRES|BICGSTAB}` `LINTOL
LINMAXIT RESTART MIXED EMULATE STAGE2 {JACOBI|ZLINE}` ·
`AMGFAST AMGREUSE AMGREUSEGROWTH AMGMAXCOARSE AMGCOARSERELAX` ·
`DTMAX DTGROW MAXCHOP DPMAX DSMAX` ·
`DTREPORT TEND NEWTONMAX NEWTONTOL` · `JACOBIAN {NUMERICAL|ANALYTIC|TENSORCORE}` `TCPRECISION {TF32|FP32}` `TCVALIDATE` ·
`FIELDSATFUNC PROPSARRAYS USEENDSCALE` (how much of the field deck to take) ·
`WELLS` (`I J TYPE RATE RW SKIN PWF PERF NAME`).

Full reference with defaults: [`docs/manual.pdf`](../../../docs/manual.pdf) chapter 4.

**Adaptive implicit (`AIM`, on by default).** A cell whose throughput CFL over
the coming substep is below `AIMCFL` (0.5) runs *explicit in saturation*: its
face mobilities are frozen at the start of the step and its saturation
couplings leave the Jacobian. Wells and cells at a front stay fully implicit,
and the accumulation term is implicit everywhere, so the matrix structure, the
Newton loop and the CPR solve are unchanged. On SPE10 model 2 over 2000 days
this takes the run from **402.6 s to 342.1 s** on one GB200, with fewer Newton
(1984 → 1789) and fewer Krylov (203k → 176k) iterations. `AIM 0` restores the
fully implicit scheme; the banner always states which is in use.

The switching rule is the familiar `v·dt/dx` in reservoir variables,

```
CFL = dt/(Vφ) · Σ_faces T_f |ΔΦ_p| |∂λ_p/∂S|   <   AIMCFL
```

with two details that were found by measurement: the *sensitivity* `∂λ/∂S`
sets the stability limit rather than the mobility itself (near a Corey front
the derivative is several times larger), and both flow directions count with
the donor cell's mobility, because the cell just downstream of a front has
little outflow but is being filled fast. The rates are measured from a
**converged** state, at the top of a substep the pressure field is not yet
solved, and every cell would look explicit.

**Multigrid hierarchy (automatic).** Stage 1 picks its hierarchy from the rank
count and says so in the banner:

| Ranks | Hierarchy | Why |
|---|---|---|
| 1 | classical PMIS + extended interpolation | Stronger on heterogeneous fields: on SPE10 it halves the Krylov count and halves the run (117 s against 237 s over four report steps) |
| 2+ | aggregation, SIZE_8 | The hierarchy that survives a decomposition: AmgX's distributed classical setup exhausts device memory building its communication maps, at 2, 3 and 4 ranks alike, with either interpolator |

`NVRS_AMGX_ALGO=CLASSICAL|AGGREGATION` forces either (classical on several
ranks is deliberately still reachable, so a future AmgX can be retested with
one variable and no rebuild); `NVRS_AMGX_CONFIG=<file>` replaces the
configuration entirely.

**Linear solver.** `LINSOLVER AMGX` is the default. It replaces only the
stage-1 pressure multigrid: the outer FP64 FGMRES, the df64-emulated operator,
the FP64 quasi-IMPES weights and the `bs×bs` second stage are the same either
way, and both multigrids run FP32, so the mixed-precision arrangement does not
change with the backend. On the Norne field deck AmgX is **1.60× faster than BoomerAMG on one GB200** (44.95 s against 71.51 s), and on the controlled
4,800-cell `BENCH3P` it needs **1,279 Krylov iterations against 6,829** over a
comparable nonlinear path (239 and 264 Newton iterations), so it is a better
preconditioner and not merely a faster one. See
`scripts/backend_study.sh` and `benchmark/backend_study/index.html`.

**Ginkgo, the FP16 arm.** `LINSOLVER GINKGO` puts Ginkgo's parallel-graph-
matching aggregation AMG in stage 1 and runs the whole V-cycle (hierarchy,
Jacobi smoothing, coarse solve and vectors) in IEEE half precision. The
pressure matrix is symmetrically Jacobi-scaled and the right-hand side
normalised before narrowing, because field-unit coefficients exceed FP16's
65,504; the correction is scaled back exactly. `GINKGOPREC 32` builds the same
hierarchy in FP32, so the two isolate the precision. It is single-GPU in this
build. `run_norne.sh` and `backend_study.sh` run it as the
third arm automatically when the binary has Ginkgo; `NVRS_GINKGO_SWEEPS`,
`_RELAX`, `_COARSE_ITERS`, `_MAXLEVELS`, `_MINCOARSE` and `_CYCLE` tune it
without a rebuild.

`STAGE2 GINKGO` replaces the stage-2 smoother with Ginkgo's adaptive-precision
block-Jacobi on the same cell blocks: each block inverse is stored in FP64,
FP32 or FP16 according to its conditioning, and the run prints the split at the
first build. The ablation therefore has five arms, defined once in
`benchmark/backend_arms.sh`:

| Arm | Deck lines | What it isolates |
|---|---|---|
| `amgx` | `LINSOLVER AMGX` | FP32 AMG, the default |
| `hypre` | `LINSOLVER HYPRE` | FP32 BoomerAMG |
| `ginkgo` | `LINSOLVER GINKGO`, `GINKGOPREC 16` | FP16 AMG |
| `ginkgo32` | `LINSOLVER GINKGO`, `GINKGOPREC 32` | against `ginkgo`: the AMG precision alone |
| `ginkgo_abj` | `LINSOLVER GINKGO`, `GINKGOPREC 16`, `STAGE2 GINKGO` | against `ginkgo`: the stage-2 smoother alone |

`run_norne.sh`, `backend_study.sh` and `nvrs.sh` run all
five when the binary has Ginkgo; the Ginkgo arms always run on one GPU.

![Head to head](benchmark/docs/backend_speed.png)

![Backend scaling](benchmark/docs/backend_scaling.png)

**Jacobian.** `JACOBIAN NUMERICAL` (default) uses exact-structure
finite-difference derivatives for every model. `JACOBIAN ANALYTIC` uses
hand-derived exact derivatives for **both** two-phase and three-phase black oil
(`NCOMP 2` and `NCOMP 3`): faster, and the reference for validating the
numerical path (they must agree to `O(ε)`). Compositional analytic derivatives
are the next increment; that block size transparently falls back to numerical.

## SPE10 model 2

The standard heterogeneity stress test: 60×220×85 = 1,122,000 cells at
20×10×2 ft, permeability contrast 3×10⁷, two-phase, 2000 days. It is the
sixth case of the backend study, and it also has its own deep-dive report:

```bash
./scripts/backend_study.sh spe10   # timings, into the study report
./scripts/run_spe10.sh             # full run + benchmark/spe10/spe10.html
```

`run_spe10.sh` produces a self-contained page covering the permeability field
by layer on a log scale with its distribution and vertical profile, the
independent porosity field, the Kz/Kx ratio that motivates `PERMZFILE`, water
saturation at three times **plus an animation over all twenty report steps**,
pressure, the well responses, a timing table and the tail of the run log.

SPE10 runs on **one GPU** in the study while the ladder sweeps 1–4. At
1.12M cells the case is below the threshold where decomposition pays, every
published SPE10 number is single-rank, and more than one rank would force the
weaker aggregation hierarchy. `SPE10_GPUS="1 2 3 4"` overrides this if a
scaling probe is wanted.

`benchmark/make_spe10.py` downloads the public Christie & Blunt dataset into
`cuda/dataset/SPE10/` on first use (77 MB, git-ignored, atomic rename) and
generates the deck. The spec's quadratic Corey curves with Swc = Sor = 0.2 are
*analytically identical* to the two-phase kernel at `SWI = SWR = 0.2`, so no
saturation table is needed; the generator's header records every simplification
it makes and verifies `ky == kx` in the dataset before writing anything.

Measured on one GB200, each step an A/B against the one before it:

| Configuration | Wall clock (2000 days) | Speed-up |
|---|---|---|
| Aggregation hierarchy, fully implicit | 861.3 s | 1.00× |
| Classical PMIS hierarchy | 402.6 s | 2.14× |
| **Classical + adaptive implicit** | **342.1 s** | **2.52×** |

*That ablation was measured earlier on the 2000-day SPE10 configuration. On the matched SPE10 deck (1000 days, CPRW, one Eclipse deck read by both simulators) SPE10 model 2 completes in **53.06 s** on one GB200 and **41.21 s** on one VR72 under AmgX, 18.5x faster
than OPM Flow on the VR72 host, with a mean per-well R^2 of 0.999.*

Both improvements are now defaults, so a plain single-GPU run gets the last
row. The A/B tables behind them: including the candidates that *lost*
(two V-cycles per application, `MULTICOLOR_DILU`, the Appleyard chop, `ZLINE`):
are reproducible with `./scripts/spe10_tune.sh`.

SPE10 is reported single-GPU, which is how published numbers for it are quoted;
the field is small enough that decomposing it loses, and multiple ranks would
force the weaker aggregation hierarchy.

## Status & validation

Built and run on NVIDIA GB200 (`sm_100`) with both backends, at 1, 2 and 4
GPUs, on grids from 4,800 to 5M cells, plus SPE10 model 2 at 1.12M cells, and
on one VR72 GPU. Against OPM Flow 2024.10 using its own GPU linear solver on the
same device and the same schedule, NVRS is **10.45× faster** on the Norne field deck (46.75 s, median of three runs, against 488.51 s), cumulative oil agrees to 0.54 %, and 19 of the 22 producers match Flow at R^2 >= 0.70 on all four rate quantities. On SPE10 model 2 one VR72 is 18.5× faster than Flow on its host CPU, at a mean per-well
R^2 of 0.999.

Still open:

1. The compositional flash/phase-density closures are a documented **reference
   EOS** (Wilson K + ideal-gas vapour + constant liquid molar volume); swap in a
   full Peng-Robinson flash for quantitative compositional accuracy.
2. Compositional analytic derivatives; that block size falls back to the
   numerical Jacobian today.
3. Further align three-phase well controls and nonlinear time stepping with
   OPM Flow for difficult field-scale schedules.

See `docs/manual.pdf` for the methods in full.

