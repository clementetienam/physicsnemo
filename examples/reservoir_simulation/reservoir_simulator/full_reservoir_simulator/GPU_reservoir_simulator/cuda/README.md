# NVRS-CUDA: GPU Black-Oil / Compositional Reservoir Simulator (CUDA C++)

## Headline

The published **Norne field deck**, 44,431 active cells over 3312 days and 247
report steps, one GPU and one rank, with six solver arms run back to back on each
machine. All numbers are **version 2** of the code: the coupled operator stored
as 3x3 blocks with one index per block (`BLOCKSPMV 1`) and the Krylov tail
replayed from CUDA graphs (`CUDAGRAPH 1`). Version 1 is the same code with scalar
CSR and direct launches.

| Run | GB200 | VR72 | Over Flow, GB200 | Over Flow, VR72 |
|---|---|---|---|---|
| OPM Flow 2024.10 | 488.51 s | 243.88 s | | |
| NVRS-CUDA (HYPRE BoomerAMG, FP32) | 71.51 s | 73.54 s | 6.83x | 3.32x |
| NVRS-CUDA (Ginkgo, FP16) | 80.55 s | 50.02 s | 6.06x | 4.88x |
| NVRS-CUDA (Ginkgo, FP32) | 70.42 s | 49.31 s | 6.94x | 4.95x |
| NVRS-CUDA (Ginkgo FP16 + adaptive block-Jacobi) | 76.61 s | 57.27 s | 6.38x | 4.26x |
| NVRS-CUDA (AmgX, FP32, native FP64 operator) | 43.19 s | 34.91 s | 11.31x | 6.99x |
| **NVRS-CUDA (AmgX, FP32, default)** | **44.95 s** | **35.73 s** | **10.87x** | **6.83x** |

These are single runs from one session (the comparison session), used to compare the backends with
one another. Run three times in a second session, the default configuration takes **46.75 s**
[46.45, 46.99] on GB200 and 35.76 s on VR72, on the same Newton path (1,497 Newton, 41,114 Krylov);
the 4 % GB200 difference between sessions sits in the launch-bound phases. Wall clocks are compared
only within a session, and every factor against Flow below uses the repeated session.

Against Flow on one rank with two host threads NVRS is **10.45x** faster, and most of that is host work Flow does on two threads.
Given the whole node, Flow is fastest with its CPU CPR solver on 16 ranks, and
**the factor is 3.5x on GB200 and 2.5x on VR72** (NVRS median of three runs).

| Flow 2024.10 configuration | GB200 | NVRS faster by | VR72 | NVRS faster by |
|---|---|---|---|---|
| GPU ILU(0), 1 rank x 2 threads | 488.51 s | 10.45x | 243.88 s | 6.82x |
| GPU ILU(0), every host thread | 348.95 s | 7.46x | 543.49 s | 15.2x |
| **CPU CPR, 16 ranks (fastest)** | **165.62 s** | **3.54x** | **88.48 s** | **2.47x** |

NVRS runs a looser material-balance tolerance than Flow (MB 3.5e-5 against
1e-7); its exact discrete water balance closes to -0.68 %. At Flow's MB 1e-7 the NVRS Newton
iteration stagnates (one VR72 run took 2,901 s, 33x Flow's CPU CPR), so every factor here holds
only at the NVRS tolerance; this is the main limitation, and a trust-region saturation update is
the planned fix.

On GB200 the AmgX run takes 1,497 Newton iterations against Flow's 1,985, and
each costs 30.0 ms against 246.1 ms. It rebuilds the AMG hierarchy 172 times at
10.2 ms each and retries 3 of the 247 report steps; BoomerAMG takes 1,506 Newton
iterations. On VR72 the AmgX run takes 1,571 Newton iterations at 22.7 ms each.

Where the time goes on GB200, in seconds:

| Run | Total | Linear solve | of which AMG | Outside the solve |
|---|---|---|---|---|
| OPM Flow | 488.51 | 97.22 | n/a | 391.29 |
| NVRS-CUDA (AmgX) | **44.95** | 23.60 | 14.56 | 21.36 |
| NVRS-CUDA (HYPRE) | 71.51 | 49.91 | 34.12 | 21.61 |
| NVRS-CUDA (Ginkgo FP16) | 80.55 | 56.30 | 41.52 | 24.24 |

Under AmgX the run splits 38.2 % native FP32, 2.2 % emulated FP64 (the operator
apply alone) and 59.6 % native FP64 (54.3 / 2.5 / 43.1 % under BoomerAMG).

![Precision split of the Norne run](../../paper/figs/precision_pie.png)

### Every ablation and study

Norne on one GPU unless a row says otherwise; each row changes one thing
from the default. GB200 walls are medians of three runs with [min, max]
unless marked as a single run; Newton and Krylov counts are GB200.

| Study | Change from the default | GB200 wall, s | Newton / Krylov | VR72 wall, s | Finding |
|---|---|---|---|---|---|
| **Default** | | | | | |
| Default configuration | block CSR, captured Krylov tail, df64 operator, quasi-IMPES, block-Jacobi stage 2, AMG reused up to 10 Newton | 46.75 [46.45, 46.99] | 1,497 / 41,114 | 35.76 [35.64, 36.62] | reference point; 11.6 s of it is output |
| **Storage and launch (2x2)** | | | | | |
| Scalar CSR, direct launches | version 1 operator | 51.24 [51.22, 51.55] | 2,004 / 45,125 | 40.08 | operator 64.1 us per Krylov |
| Block CSR, direct launches | no graph capture | 45.25 [44.74, 45.47] | 1,497 / 41,114 | 34.89 | 52.0 us: storage alone 1.23x (model 1.31x); fastest |
| Scalar CSR, captured tail | no block storage | 52.55 [52.44, 52.56] | 2,004 / 45,125 | 40.86 | capture costs 2.9 s; not repaid |
| Block CSR, captured, reuse guard | CUDAGRAPHMINREUSE 2 | 45.21 [44.91, 45.33] | 1,497 / 41,114 | 34.91 | guard stops after 192 captures (0.22 reuses each) |
| **Operator arithmetic** | | | | | |
| Native FP64 operator | EMULATE 0 | 44.04 [43.82, 44.21] | 1,400 / 38,869 | 34.56 | 11.37 us per apply; path differs, not arithmetic |
| FP32-stored operator, FP64 master | EMULATE 2 | 47.62 [47.52, 47.81] | 1,654 / 42,608 | 34.36 | apply 7% cheaper, run 8% slower; 9 of 1,663 solves continued on the true residual |
| **Preconditioner** | | | | | |
| Rebuild AMG every Newton | AMGREUSE 0 (single run) | 54.90 | 1,532 / 31,479 | 59.73 | reuse is worth 1.17x (1.67x on VR72) |
| Dynamic row-sum decoupling | CPRDECOUPLE DRS (single run) | 45.21 | 1,440 / 40,149 | 37.11 | Krylov per Newton 27.9 vs 27.5 |
| Plain row-sum decoupling | CPRDECOUPLE SUM (single run) | 47.41 | 1,596 / 42,864 | 37.41 | 26.9 per Newton: decoupling within 5% |
| Vertical line stage 2 | STAGE2 ZLINE | n/a | n/a | n/a | needs the Cartesian neighbour map; falls back to block-Jacobi on Norne |
| BoomerAMG, coarse grid 20,000 rows | AMGMAXCOARSE 20000, vs BoomerAMG 71.51 s (single run) | 87.35 | 1,723 / 136,274 | 73.39 | V-cycle 0.24 vs 0.44 ms but 1.87x Krylov |
| **Time integration** | | | | | |
| Flow's accepted substeps imposed | DTSEQFILE | 47.68 [47.23, 47.82] | 1,580 / 42,867 | 32.35 | Newton ratio to Flow 1.26 (1.33 unconstrained) |
| **Reference (Flow 2024.10)** | | | | | |
| (P) GPU ILU(0), 1 rank x 2 threads | two-thread configuration | 488.51 | 1,985 / 30,072 | 243.88 | NVRS 10.45x faster (VR72 6.82x) |
| (G) GPU ILU(0), every host thread | 1 x 144 (VR72 1 x 352) | 348.95 | 1,985 | 543.49 | 7.46x; threads slow VR72 down |
| (C) CPU CPR, best of 8/16/32 ranks | 16 ranks on both | 165.62 | 1,263 | 88.48 | 3.54x (VR72 2.47x) |
| **Scaling, 1 to 4 GPUs** | | | | | |
| 1 million cells | aggregation, NCCL, one imposed step sequence | 24.92 -> 27.22 | 683 -> 687 | 16.84 -> 25.67 | 0.92x (VR72 0.66x): too small to scale |
| 5 million cells | as above | 104.68 -> 80.60 | 683 -> 685 | 69.16 -> 56.73 | 1.30x, 33% efficiency (VR72 1.22x) |
| **Kernels in isolation** | | | | | |
| Block FP64 vs cuSPARSE CSR / BSR | Norne-sized box, us per apply | 14.4 vs 34.8 / 39.0 |  | 11.1 vs 29.7 / 32.3 | 2.42x / 2.71x (VR72 2.69x / 2.92x) |
| df64 vs native FP64 | same box | 18.5 vs 14.4 |  | 13.3 vs 11.1 | emulation costs 1.28x (VR72 1.21x) |
| Graph capture break-even r* | capture cost / saving per replay | 2.3 |  | 30 | Norne reuses each graph 1.16 times |
| **Accuracy** | | | | | |
| Pressure-matrix conditioning | cond2(A_p) = 5.2e3 | u32: 3e-4, u16: 2.5 |  | same | FP32 hierarchy safe; FP16 outside Proposition 1 |
| Exact discrete water ledger | NVRS_SUBSTEP_BALANCE=1 | -0.68% (trapezoid 1.75%) |  | -0.70% (1.15%) | bias due to the MB tolerance; the remainder is integration error in the balance |

### Version 2: block storage and CUDA graphs

| Norne, GB200, AmgX | Version 1 | Version 2 |
|---|---|---|
| Operator apply, per Krylov iteration | 64.0 us | **24.1 us** |
| Orthogonalisation (CGS2), per Krylov iteration | 91.0 us | **44.8 us** |
| AMG V-cycle | 0.31 ms | 0.31 ms |
| Kernel launches in the Nsight trace | 173,083 | **129,499** |
| Wall clock, GB200 | 46.52 s | **44.95 s** |
| Wall clock, VR72 | 44.65 s | **35.73 s** |

The two changes were introduced together; the 2x2 ablation below separates
them. Block storage makes the operator 1.23x cheaper per Krylov iteration, close
to the 1.31x the traffic model predicts. Graph capture halves the operator time
the solver's timer sees, but on Norne each graph is reused only 1.16 times
against a measured break-even of 2.3, the captures cost 2.9 s, and the run is
3 % slower than with direct launches. The reuse guard (`CUDAGRAPHMINREUSE`)
stops capturing after 192 captures and recovers the direct-launch time. On
SPE10 each graph is reused 159 times and capture stays on.

### Emulated against native FP64 in the operator

A sixth arm runs AmgX with `EMULATE 0`: the operator apply in native FP64,
everything else, the FP32 multigrid included, unchanged. It is not a comparison
of FP64 against FP32; both arms are equally mixed precision.

| AmgX, one GPU | Wall | Newton | Operator | Per apply |
|---|---|---|---|---|
| Norne, GB200, df64 | 44.95 s | 1,497 | 0.99 s | 11.8 us |
| Norne, GB200, native FP64 | 43.19 s | 1,400 | 0.90 s | 11.3 us |
| Norne, VR72, df64 | 35.73 s | 1,571 | 0.99 s | 11.6 us |
| Norne, VR72, native FP64 | 34.91 s | 1,526 | 0.96 s | 11.6 us |

Inside the solver the cost per apply is the same within 4 %, and the operator
is 2 % of the run, so the wall-clock differences are the Newton path, not the
arithmetic. Measured in isolation on a fixed input, the df64 apply costs 1.28x
the native one on GB200 and 1.21x on VR72: a df64 pair is the same 8 bytes as
FP64, but it does ten to twenty times the arithmetic. Emulation buys FP64
semantics, not speed. A third arm stores the operator in FP32 with an FP64
master for every residual that decides convergence (`EMULATE 2`): the apply is
7 % cheaper and the run 8 % slower on GB200, because the perturbed operator
lengthens the nonlinear path. On SPE10, whose deck already uses the native
operator, the two runs agree to 0.5 %.

### Ginkgo: a third multigrid, down to half precision

Ginkgo's parallel-graph-matching AMG (Anzt et al., ACM TOMS 2022) runs the whole
V-cycle in IEEE FP16 (or FP32 for a control), optionally with Ginkgo's
adaptive-precision block-Jacobi (Flegar et al., ACM TOMS 2021) as the second CPR
stage. It **does not beat AmgX**: its best arm is 1.57x slower on GB200 and 1.38x
on VR72, while beating BoomerAMG on both (1.02x and 1.49x).

| Norne, one GPU | V-cycle GB200 | V-cycle VR72 | Mean R^2 GB200 | Mean R^2 VR72 |
|---|---|---|---|---|
| **AmgX FP32** | **0.31 ms** | **0.23 ms** | 0.901 | 0.910 |
| AmgX FP32, native FP64 operator | 0.31 ms | 0.23 ms | 0.908 | 0.759* |
| BoomerAMG FP32 | 0.44 ms | 0.44 ms | 0.902 | 0.910 |
| Ginkgo FP16 | 0.65 ms | 0.36 ms | 0.851* | 0.903 |
| Ginkgo FP32 | 0.65 ms | 0.37 ms | 0.905 | 0.898 |
| Ginkgo FP16 + adaptive bJ | 0.64 ms | 0.36 ms | 0.893 | 0.911 |

\* One producer, D-4AH, moved its water breakthrough in that run; the other 21
wells match the other arms. The agreement measures used in the arXiv paper
(range-normalised error, cumulative error, breakthrough offset) are reported
there.

Half precision keeps the answer but not a speed-up. The measured conditioning
of the Norne pressure matrix puts FP16 outside the guarantee: cond(A_p) x u32 is
3e-4, well inside the condition of the paper's Proposition 1, but cond(A_p) x
u16 is 2.5, outside it. Per Newton iteration, however, FP16 Ginkgo needs no more
Krylov iterations than FP32 Ginkgo (32.0 against 33.5 on GB200, 35.5 against
34.6 on VR72); its 1.21x larger total comes from a longer nonlinear path (1,826
against 1,437 Newton), which `REVIEW_PARTS=fp16steps` tests on one imposed
substep sequence. On traffic: a CSR matrix with 32-bit
column indices moves 4 + w bytes per nonzero, so halving the value width saves at
most a quarter of the traffic, and the Nsight traces show the Norne kernels bound
by launch latency, with about 15 % of Ginkgo's GPU time in per-application solver
bookkeeping.

### Norne, every producer scored against Flow

Each of the 22 producers is scored separately by the R^2 of its whole history
against OPM Flow, on six quantities (GB200, AmgX). 🟢 R^2 >= 0.80,
🟡 0.70 to 0.80, 🔴 below 0.70. Where Flow's value is constant over the history
(a well held at a pressure limit), the error is measured against 5 % of that
level, so shut-in periods do not give negative scores.

| Well | WOPR | WWPR | WWCT | WGPR | WGOR | WBHP |
|---|---:|---:|---:|---:|---:|---:|
| B-1BH | 🟢 0.99 | 🟡 0.76 | 🟢 0.84 | 🟢 0.97 | 🟢 0.98 | 🟢 0.98 |
| B-1H | 🟢 0.95 | 🟢 0.83 | 🟢 0.93 | 🟢 0.97 | 🟢 0.85 | 🟢 0.92 |
| B-2H | 🟢 0.96 | 🟢 0.92 | 🟢 0.94 | 🟢 0.92 | 🟢 0.83 | 🔴 **0.28** |
| B-3H | 🟢 0.93 | 🟢 0.94 | 🟢 0.94 | 🟢 0.93 | 🟢 0.93 | 🟢 0.92 |
| B-4BH | 🟢 0.98 | 🟢 0.99 | 🟢 0.99 | 🟢 0.96 | 🟢 0.93 | 🟢 0.99 |
| B-4DH | 🟢 0.95 | 🟢 0.92 | 🟢 0.95 | 🟢 0.97 | 🟢 0.95 | 🟢 0.95 |
| B-4H | 🟢 0.98 | 🟢 0.98 | 🟢 0.96 | 🟢 0.98 | 🟢 0.97 | 🟢 1.00 |
| D-1CH | 🟢 0.99 | 🟢 0.98 | 🟢 1.00 | 🟢 0.97 | 🟢 0.97 | 🟢 0.96 |
| D-1H | 🟢 0.98 | 🟢 0.97 | 🟢 0.86 | 🟢 0.96 | 🟢 0.97 | 🟢 0.95 |
| D-2H | 🟢 0.97 | 🟢 0.90 | 🟢 0.93 | 🟢 0.95 | 🟢 0.80 | 🔴 **0.53** |
| D-3AH | 🟢 0.98 | 🟢 0.97 | 🟢 0.97 | 🟢 0.98 | 🟢 0.97 | 🟢 0.97 |
| D-3BH | 🟢 0.99 | 🟡 0.78 | 🟢 0.86 | 🟢 0.99 | 🟢 0.98 | 🟢 0.90 |
| D-4AH | 🔴 **0.67** | 🔴 **-0.25** | 🔴 **0.53** | 🟢 0.92 | 🟡 0.72 | 🟢 0.97 |
| D-4H | 🟢 0.98 | 🟢 0.95 | 🟢 0.89 | 🟢 0.96 | 🟢 0.97 | 🟢 0.99 |
| E-1H | 🟢 0.97 | 🟢 0.98 | 🟢 0.98 | 🟢 0.94 | 🟢 0.93 | 🟢 0.91 |
| E-2AH | 🟢 0.86 | 🟢 0.96 | 🟢 0.90 | 🟢 0.81 | 🟡 0.74 | 🟢 0.99 |
| E-2H | 🟢 0.98 | 🟢 0.96 | 🟢 0.99 | 🟢 0.94 | 🟢 0.96 | 🟢 0.94 |
| E-3AH | 🟢 0.93 | 🔴 **0.68** | 🟢 0.84 | 🟢 0.86 | 🔴 **0.69** | 🟢 0.99 |
| E-3CH | 🟢 0.84 | 🟢 0.97 | 🟢 0.99 | 🟢 0.82 | 🟢 0.97 | 🟡 0.78 |
| E-3H | 🟢 0.98 | 🟢 0.98 | 🟢 0.98 | 🟢 0.97 | 🟢 0.98 | 🟢 0.98 |
| E-4AH | 🟢 0.90 | 🟢 0.88 | 🟢 0.92 | 🟡 0.77 | 🔴 **0.52** | 🟢 1.00 |
| K-3H | 🔴 **0.64** | 🔴 **0.65** | 🟢 0.93 | 🟢 0.97 | 🔴 **0.67** | 🟢 0.95 |
| **wells >= 0.70** | **20/22** | **19/22** | **21/22** | **22/22** | **19/22** | **20/22** |

**20 of the 22 producers match Flow at R^2 >= 0.70 on oil rate, 21 on water cut,
all 22 on gas rate, and 19 on all four rate quantities at once** (all but D-4AH,
E-3AH and K-3H). The mean over producers and quantities is 0.901. The VR72 run
gives 20 of 22 on all four rates and a mean of 0.910.

![R^2 per producer, one panel per quantity](../../paper/figs/norne_well_r2.png)

Field recovery agrees to a fraction of a percent:

| | OPM Flow | NVRS-CUDA |
|---|---|---|
| Peak field oil rate | 227.52 k STB/day | 232.20 k STB/day |
| Best single well (E-2H) | 57.75 k STB/day | 54.33 k STB/day |
| Cumulative oil, 3312 days | 422.78 M STB | **425.07 M STB** |
| **Recovery factor** | **41.8 %** | **41.5 %** |
| Implied B_o | 1.303 | 1.289 |

The oil balance closes: what the saturation field gives up accounts for
**100 %** of what the well rates report. Water gained over the history is
523.0 M STB against Flow's 518.4; reconciled against reported injection the
water balance closes to +6.2 % (GB200) and +6.6 % (VR72) against Flow's +3.8 %.

![Norne oil rate per producer against Flow](../../paper/figs/norne_wopr.png)

![Norne pressure and water saturation at three moments](../../paper/figs/norne_evolution.png)

### SPE10 model 2 against OPM Flow

1,122,000 cells, 1000 days, one Eclipse deck (`decks/SPE10_MATCHED.DATA`) read
by both simulators, CPRW, `LINTOL 3e-2`, native FP64 operator, one GPU. OPM Flow
runs on the VR72 host (its GPU bridge needs block size 3 and SPE10 is two-phase).

![SPE10 permeability, mD](../../paper/figs/spe10_perm.png)

![SPE10 porosity](../../paper/figs/spe10_poro.png)

| Run | GB200 | VR72 | Over Flow | Mean R^2 |
|---|---|---|---|---|
| OPM Flow 2024.10 (CPU) | n/r | 766.16 s | | reference |
| NVRS-CUDA (HYPRE) | 99.22 s | 91.74 s | 8.37x | 0.999 |
| **NVRS-CUDA (AmgX)** | **53.06 s** | **41.21 s** | **18.59x** | **0.999** |

Recovery factor 21.1 % in both codes and the water balance closes exactly.
AmgX needs 39.2 Krylov iterations per Newton step against BoomerAMG's 69.9.

![SPE10 flood, OPM Flow against NVRS-CUDA on VR72](../../paper/figs/spe10_movie.gif)

### Every model, every backend, both machines, one GPU

![Wall clock by model, backend and GPU generation](../../paper/figs/headline.png)

| Model | Cells | Flow GB200 | HYPRE GB200 | Ginkgo GB200 | AmgX GB200 | Flow VR72 | HYPRE VR72 | Ginkgo VR72 | AmgX VR72 |
|---|---|---|---|---|---|---|---|---|---|
| Norne, field | 44,431 | 488.51 | 71.51 | 80.55 | **44.95** | 243.88 | 73.54 | 50.02 | **35.73** |
| Bench3P | 4,800 | 9.25 | 3.00 | n/r | **0.89** | 5.32 | 3.44 | n/r | **0.95** |
| SPE10 model 2 | 1,122,000 | n/r | 99.22 | n/r | **53.06** | 766.16 (CPU) | 91.74 | n/r | **41.21** |
| Million | 1,000,000 | n/r | 26.23 | n/r | **21.17** | n/r | 19.94 | n/r | **14.87** |
| Million-2M | 2,000,000 | n/r | 49.94 | n/r | **39.38** | n/r | 38.23 | n/r | **27.17** |
| Million-4M | 4,000,000 | n/r | 81.87 | n/r | **77.67** | n/r | 60.97 | n/r | **52.33** |
| Million-5M | 5,000,000 | n/r | 102.32 | n/r | **95.61** | n/r | 76.65 | n/r | **63.77** |

Seconds, one GPU. AmgX is the fastest library on every model on both machines.
Solid bars are GB200, hatched bars VR72; green AmgX, blue BoomerAMG, on Norne
orange, violet and teal for the three Ginkgo arms (FP16 shown in the table), and
dark olive for AmgX with a native FP64 operator. `n/r` marks an arm that was not
run.

### Scaling across GPUs: AmgX against BoomerAMG, GB200 and VR72

![GB200: wall clock by rank count and backend](../../paper/figs/scaling.png)

![VR72: wall clock by rank count and backend](../../paper/figs/scaling_vr72.png)

At two and four GPUs the two arms take the same Newton path to within four
iterations, and each backend is also measured against **its own** single-GPU
rate:

| Cells | Backend | GB200 1 / 2 / 4 GPUs | GB200 1 to 4 | VR72 1 / 2 / 4 GPUs | VR72 1 to 4 |
|---|---|---|---|---|---|
| 4,800 | BoomerAMG | 3.00 / 6.22 / 7.91 s | 0.38x | 3.44 / 7.98 / 13.29 s | 0.26x |
| | AmgX | 0.89 / 2.00 / 2.75 s | 0.32x | 0.95 / 2.33 / 3.93 s | 0.24x |
| 1,000,000 | BoomerAMG | 26.23 / 33.85 / 43.27 s | 0.61x | 19.94 / 28.88 / 49.61 s | 0.40x |
| | AmgX | 21.17 / 18.96 / **18.26** s | **1.16x** | 14.87 / **14.02** / 16.17 s | 0.92x |
| 2,000,000 | BoomerAMG | 49.94 / 52.84 / 65.32 s | 0.76x | 38.23 / 43.83 / 72.11 s | 0.53x |
| | AmgX | 39.38 / 31.05 / **30.84** s | **1.28x** | 27.17 / **21.24** / 23.62 s | **1.16x** |
| 4,000,000 | BoomerAMG | 81.87 / 80.00 / 86.67 s | 0.94x | 60.97 / 60.96 / 85.21 s | 0.72x |
| | AmgX | 77.67 / 58.01 / **51.78** s | **1.50x** | 52.33 / 36.58 / **41.14** s | **1.27x** |
| 5,000,000 | BoomerAMG | 102.32 / 96.16 / 96.64 s | 1.06x | 76.65 / 77.49 / 93.30 s | 0.81x |
| | AmgX | 95.61 / 71.14 / **61.71** s | **1.55x** | 63.77 / 43.91 / **42.26** s | **1.51x** |

BoomerAMG does not scale on this problem: its hierarchy setup grows with ranks
(5.2 s to 17.6 s at a million cells on GB200) while AmgX's shrinks. At four GPUs
AmgX is faster in absolute terms by 1.64x to 2.37x on GB200 and 2.24x to 3.10x
on VR72. On one GPU from a million cells up AmgX takes about 694 Newton
iterations to BoomerAMG's 362 (its single-rank classical hierarchy; at two ranks
it builds aggregates and the counts agree), and is still 2.1x to 2.7x cheaper per
Newton iteration.

**VR72 against GB200.** AmgX gains 1.30x to 1.41x across the million-cell
family, 1.29x on SPE10 and 1.26x on Norne; Bench3P, at 4,800 cells, is too small
to gain. Keep the CUDA JIT cache enabled on VR72: the device reports a compute
capability no released toolkit targets, so every kernel is compiled from PTX at
load.

**Time integration follows OPM Flow's**: a continuous timestep carried across
report steps and rescaled from the Newton count of the last converged sub-step,
with Flow's pore-volume-weighted convergence criterion (`DTCONTROL`, `OPMCONV`,
`NEWTONSTALLMAX`).

Reproduce a case with
`cd full_reservoir_simulator/GPU_reservoir_simulator/cuda/benchmark && ./nvrs.sh all NORNE`
(or `SPE10`, or `BENCH3P`).

## Start here: the container

Everything in this repository was measured inside one image, and it is the
supported way to build and run any of it.  From this folder:

```bash
../../../build_container.sh
```

That builds the image if it does not exist and drops you into a shell already
in `/workspace/project`, which **is** this repository, mounted rather than
copied, so
an edit on the host is an edit in the container, and rebuilding the simulator
never means rebuilding the image.

Then, inside the container:

```bash
cd full_reservoir_simulator/GPU_reservoir_simulator/cuda && make -j
```

Before trusting any multi-GPU timing, prove the image is what it claims:

```bash
../../../build_container.sh check
```

A container can carry CUDA and an MPI and still not have **CUDA-aware** MPI,
and a successful build says nothing either way.  `check` compiles and runs a
two-rank `MPI_Send`/`MPI_Recv` on a `cudaMalloc`'d pointer and reports what
actually happened.  It matters: NVRS ranks exchange *device* buffers directly,
and a stock OpenMPI stages them through host memory. It still runs, and it is
quietly several times slower, which is the worst kind of difference because
nothing fails.

[`Dockerfile`](../../../Dockerfile) is CUDA 12.8 on Ubuntu 22.04 with UCX
1.18.1, PMIx 4.2.8 and OpenMPI 4.1.5 built from source (`--with-cuda
--with-ucx --with-pmix --with-slurm`).  UCX's own test programs are not built: they are not used here, and pinning
them to one CUDA version is what broke the first build of this image.  The base image is multi-architecture,
so it builds on x86_64 and on aarch64 (GB200) unchanged.  The first build
compiles three libraries from source and takes tens of minutes; later builds
come from the layer cache.

[`build_container.sh`](../../../build_container.sh) also takes `build`,
`rebuild`, `shell`, `run <cmd>` and `help`.  It uses `podman` when `docker` is
absent, and where neither is permitted (common on a shared cluster) it prints
the Apptainer equivalent.

---

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

## Then: one command

`nvrs.sh` is the whole workflow behind one entry point. It does not replace any
of the scripts below; it calls them in order and knows which one a given case
needs, because that is where the steps were being lost: sourcing an environment
file between two installers, or knowing that Norne is downloaded
while SPE10 is generated and the BENCH cases are computed, or that the figures
need a second command after the profile.

```bash
cd full_reservoir_simulator/GPU_reservoir_simulator/cuda/benchmark

./nvrs.sh                  # what is installed, what is ready, what to type
./nvrs.sh install          # HYPRE, AmgX, OPM Flow, Nsight, and nvrs_cuda
./nvrs.sh all NORNE        # fetch it, run it, profile it, report it
```

That last line ends with `runs/compare_NORNE.html`: NVRS against OPM Flow, the
field images, the animation, and the Nsight Systems figures for both multigrid
backends, in one page.

Any case works the same way:

```bash
./nvrs.sh cases            # every case, and whether its deck is ready
./nvrs.sh all SPE10
./nvrs.sh all BENCH3P
NVRS_DECK=/path/to/MY.DATA ./nvrs.sh all MYFIELD
```

Or one step at a time, when a stage needs re-running on its own:

| Command | What it does |
|---|---|
| `./nvrs.sh` | the menu: commands, options, and what is ready on this machine |
| `./nvrs.sh install` | HYPRE, AmgX, OPM Flow, Nsight and `nvrs_cuda` |
| `./nvrs.sh cases` | every case, and whether its deck is ready |
| `./nvrs.sh get <CASE>` | fetch or generate the deck and its data |
| `./nvrs.sh run <CASE>` | the backends, against OPM Flow |
| `./nvrs.sh profile <CASE>` | Nsight Systems, once per backend |
| `./nvrs.sh report <CASE>` | the figures, folded into the HTML |
| `./nvrs.sh all <CASE>` | all four of the above, in order |

Options are flags, and every command prints the ones it takes when it starts,
so there is nothing to memorise:

```
  in effect now         --backend amgx --no-opm
  'profile' also takes  --backend  --fraction  --tend  --steps  --ncu  --deck
  full list             ./nvrs.sh --help
```

| Option | Applies to | Meaning |
|---|---|---|
| `--no-opm` | run, all | NVRS only; do not run OPM Flow |
| `--backend <b>` | run, profile | `amgx`, `hypre`, or `both` (default both) |
| `--ranks <n>` | run, profile | MPI ranks for NVRS |
| `--no-profile` | all | skip the profiling step |
| `--no-scaling` | run, all | one GPU only; skip the 2/4-GPU strong-scaling sweep |
| `--scaling` | run, all | force the sweep on (SPE10 defaults to off) |
| `--ncu` | profile, all | add Nsight Compute counters (~100x slower) |
| `--fraction <f>` | profile, all | history to profile: `0.25`, or `all` (default `0.1`) |
| `--tend <days>` | profile, all | profile exactly this many days instead |
| `--steps <n>` | profile, all | **NORNE only**: report steps, or `all` |
| `--heartbeat <s>` | profile, all | progress interval, `0` to silence |
| `--deck <path>` | get, run, all | your own `.DATA`, for a case not under `decks/` |

`--no-scaling` matters for the large cases. By default a run is repeated on 2,
4 and however many GPUs are present, which is the point for the BENCH family
and wasted time for SPE10: 1.12 million cells that exist to stress the solver,
where each extra rank repeats a long run to answer a question the case was not
chosen to ask. SPE10 therefore defaults to a single GPU, and `--scaling`
overrides that.

`--steps` is Norne-only on purpose: a report step is a fortnight there and
something else in every other deck, so it is refused rather than quietly
misapplied. Use `--fraction` or `--tend` elsewhere.

Some worked examples:

```bash
./nvrs.sh all NORNE --no-opm              # skip Flow, NVRS only
./nvrs.sh all NORNE --backend amgx        # one backend, faster
./nvrs.sh all NORNE --steps all           # the whole 247-step history
./nvrs.sh all SPE10 --fraction 0.5        # half of SPE10's history
./nvrs.sh all MYFIELD --deck /path.DATA   # a deck of your own
./nvrs.sh profile NORNE --ncu             # add the kernel counters
./nvrs.sh report NORNE                    # rebuild the HTML only
```

The older environment variables (`SKIP_OPM`, `NVRS_BACKENDS`, `PROF_FRACTION`
and the rest) still work; the flags override them.

Each install step is skipped when its output is already present, so
`./nvrs.sh install` is safe to re-run after a failure and resumes rather than
starting over. Nothing in it needs root.

## Installing everything, in one command

```bash
cd full_reservoir_simulator/GPU_reservoir_simulator/cuda/benchmark && ./install_all.sh
```

That runs, in dependency order: HYPRE, AmgX, OPM Flow (with DUNE, Boost and
SuiteSparse), Nsight Systems, and finally `nvrs_cuda` itself. Each step is
skipped if its output is already there, so re-running after a failure resumes
rather than starting over, and a step that fails does not stop the others: a
missing AmgX still leaves a working HYPRE build.

`SKIP_HYPRE=1`, `SKIP_AMGX=1`, `SKIP_OPM=1`, `SKIP_NSYS=1` and `SKIP_BUILD=1`
turn off individual steps. `FORCE_HYPRE=1` and friends rebuild something that
is already present.

Then:

```bash
cd full_reservoir_simulator/GPU_reservoir_simulator/cuda/benchmark && NVRS_PROFILE=1 ./run_norne.sh
```

## When the GPU is newer than the CUDA that has to build for it

`nvidia-smi` reporting a compute capability no toolkit lists is not an error to
work around. Vera Rubin reports 10.7, and CUDA 13.1 knows 100, 103, 110, 120
and 121, with nothing at 107. Nothing can compile *to* sm_107.

The answer is PTX. Every install script picks the newest architecture **not
newer than** the device and emits PTX for it, which the driver JIT-compiles at
load, targeting the real hardware. So the code that finally executes is native
sm_107 produced by the driver; the `103` is the version of the intermediate
form, not what the kernels end up as.

The direction matters. PTX JITs forward onto newer hardware and never back onto
older, and a cubin runs at an equal or higher *minor* revision within one major
capability, never a lower one. So `compute_103` PTX is right for a 10.7 device
and `compute_121` PTX would not run on it at all. Taking the toolkit's newest
architecture is the obvious wrong answer.

This is automatic. `CUDA_ARCH=100` pins it if you need to cross-build for a
machine you are not sitting on; anything else lets the scripts decide.

Two consequences worth knowing:

* **The first run is slow.** Every kernel is compiled from PTX on first launch.
  Norne's report step 1 took 63.5 s that way against 1.5 s for the average
  step. The driver caches the result, so it is a one-off, and the benchmark
  scripts point `CUDA_CACHE_PATH` at `runs/.jitcache` with a 4 GiB
  `CUDA_CACHE_MAXSIZE` so the cache lives on the mounted filesystem and
  survives the container. If your environment sets `CUDA_CACHE_DISABLE=1`, the
  banner says so and every run pays the cost again.
* **Do not compare timings across machines.** JIT-compiled kernels on one GPU
  and native cubins on another are not the same measurement. The kernel *mix*
  in a profile is still valid; the wall clock is not.

`./check_hypre.sh` reports which architectures are inside the HYPRE that NVRS
actually loads, and whether the GPU can run any of them.

## Installing dependencies without root

Inside a Slurm container the user is typically not root: `dpkg` refuses with
*requested operation requires superuser privilege*, and `sudo` is not setuid.
`apt-get download` needs no privilege, though, and `dpkg-deb -x` unpacks into
any directory you own. So `install_opm_gpu.sh` fetches what is missing (Boost,
SuiteSparse, ZOLTAN, METIS, Scotch and the rest) into a prefix under
the build tree and points CMake at it. DUNE is built from source into the same
prefix OPM installs to, since the packaged version is older than OPM wants.

`install_nsys.sh` does the same for Nsight Systems, which several images omit
while shipping Nsight Compute.

One rule that comes out of this: **when a prefix is added to
`LD_LIBRARY_PATH`, append it.** It exists to supply libraries nothing else
provides, so last place is the correct place. Prepending it once put Ubuntu's
Open MPI ahead of the container's source-built one, and `orterun` then rejected
every argument it has, including `--version`, which looks exactly like a
launcher with strange syntax and is not. The MPI stack, the fabric libraries
and the process managers are never unpacked at all now, whatever the
dependency resolver says about them.

## MPI launcher

`--oversubscribe` and `--allow-run-as-root` are Open MPI spellings, and the
first `mpirun` on `PATH` is not necessarily the MPI the binaries were linked
against, since a module system can put its own ahead of the container's. The
benchmark scripts prefer `$OMPI_HOME/bin/mpirun`, and add those flags only
after confirming the launcher really is Open MPI. `MPIRUN=/path/to/mpirun`
overrides the choice.

`nvrs_cuda` is always launched under MPI, profiling included. A bare launch
does not merely fail: a singleton `MPI_Init` under Slurm's PMIx takes the whole
job step down.

## Profiling

```bash
NVRS_PROFILE=1 ./run_norne.sh        # nsys: where the time goes
NVRS_PROFILE=both ./run_norne.sh     # add Nsight Compute's kernel counters
```

nsys costs about 1.2x runtime and can ride along with any benchmark. ncu
replays every kernel it profiles and is roughly 100x slower per kernel, so it
is always asked for explicitly. Both are found where they are actually
installed (`/opt/nvidia/nsight-systems/*/target-linux-*/`, the CUDA toolkit's
`bin`, the HPC SDK), not only on `PATH`; `NSYS=` and `NCU=` name
them directly.

Four figures reach the HTML report. The first needs no profiler at all: the
solver times its own phases, which is the only way to separate a multigrid
V-cycle from the operator apply, since both are the same sparse kernel and a
trace cannot tell them apart. The other three come from nsys and ncu, and the
report says plainly when they are absent rather than implying Nsight measured
what it did not.


### When the profilers do not work

Both tools depend on the driver and on what the container exposes, so either
can fail on a machine where the simulator itself runs perfectly. The failures
are quiet by nature, and each looks like something it is not.

**nsys runs and collects no CUDA.** The report is written, nsys exits 0, and
the only sign is in its own diagnostics:

```
CUDA profiling might have not been started correctly.
No CUDA events collected. Does the process use CUDA?
```

Two causes. The profiled process may have exited before reaching the GPU: check
the session duration; a run that lasted two seconds did not simulate
anything. Or nsys is older than the GPU: an nsys that does not know the
architecture cannot trace CUDA on it, and says nothing clearer than the above.
`nsys --version` against the age of the hardware settles which.

`profile_run.sh` reports an empty kernel summary rather than passing it on, so
this does not reach the report as a missing figure with no explanation.

**ncu fails to initialise.**

```
Failed to initialize the profiler: LibraryNotLoaded.
Check that a compatible driver library is loaded.
```

This is not a permissions problem; that one says `ERR_NVGPUCTRPERM` and is
lifted by an administrator setting `NVreg_RestrictProfilingToAdminUsers=0`.
`LibraryNotLoaded` means the driver's profiling components are not reachable
inside the container, which is a matter of what the container runtime was told
to expose: `NVIDIA_DRIVER_CAPABILITIES` has to include `compute,utility`, and
profiling generally wants `CAP_SYS_ADMIN`.

To tell "ncu is broken here" from "ncu is broken on this application", profile
something trivial:

```bash
printf '__global__ void k(float*a){a[threadIdx.x]=1;}\nint main(){float*d;cudaMalloc(&d,1024);k<<<1,64>>>(d);cudaDeviceSynchronize();}\n' > t.cu
nvcc t.cu -o t && ncu --metrics gpu__time_duration.sum ./t
```

If that fails too, nothing in this repository can help; if it succeeds, the
problem is in how the application is being launched.

Note that ncu writes its CSV to **stdout**, so its error messages land in
`ncu_kernels.csv` rather than `ncu.log`, which holds only whatever MPI said
afterwards. `profile_run.sh` prints both when a profile comes back too small,
because reading only the log points at the wrong cause.

`NSYS=/path/to/nsys` and `NCU=/path/to/ncu` select a particular copy, useful
when the one bundled with the CUDA toolkit differs from a standalone install.

**A profile is only comparable to a run on the same hardware.** On a machine
where everything is JIT-compiled from PTX, the kernel mix is still valid and
the wall clock is not. Timings meant to be compared against a baseline should
be taken where the binaries are native.

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

For a clean cluster checkout containing only the maintained development branch:

```bash
git lfs install
git clone --branch feature/norne-examples --single-branch \
  https://github.com/clementetienam/physicsnemo.git
cd physicsnemo/examples/reservoir_simulation/reservoir_simulator
```

Do not place a personal access token in the command or shell history; enter it
at the credential prompt or use the cluster credential helper. `git lfs pull`
downloads tracked large files such as the compiled manual but does not build
the simulator.

1. Build HYPRE (FP32 for mixed precision, FP64 for the reference) with the
   existing helper:

   ```bash
   cd ../../Mixed_precision
   CUDA_ARCH=100 ./scripts/docker/install_hypre_sm100.sh   # → hypre-install/{fp32,fp64}
   source hypre_env.sh                                     # sets HYPRE_FP32_DIR / HYPRE_FP64_DIR
   ```

2. Build **AmgX**, which supplies the default `LINSOLVER AMGX`:

   ```bash
   cd ../GPU_reservoir_simulator/cuda
   ./benchmark/install_amgx.sh            # clone, build, install
   source ../../amgx-install/amgx_env.sh  # sets AMGX_DIR and LD_LIBRARY_PATH
   ```

   The script does three things that are easy to get wrong by hand: it
   provisions a CMake older than 4 (AmgX does not build under 4), it takes the
   SM from `nvidia-smi` rather than a default list that omits GB200, and it
   stages the public headers itself because AmgX's install target has not always
   shipped them. Each of those otherwise fails quietly and shows up much later.

   Optionally build **Ginkgo**, which supplies `LINSOLVER GINKGO`, the FP16
   multigrid:

   ```bash
   ./benchmark/install_ginkgo.sh              # clone v1.11.0, CUDA + half precision
   source ../../ginkgo-install/ginkgo_env.sh  # sets GINKGO_DIR and LD_LIBRARY_PATH
   ```

   The script switches off Ginkgo's tests, examples and benchmarks (they
   download dependencies, which fails on a node without internet) and refuses
   to finish unless the install has the CUDA module and half precision enabled.
   Without Ginkgo, `src/ginkgo_backend.cpp` compiles to stubs and a deck asking
   for `LINSOLVER GINKGO` runs AmgX with a warning.

   **AmgX is optional.** Without it everything still builds and runs on HYPRE:
   `src/amgx_backend.cu` compiles to nothing, and a deck asking for `LINSOLVER
   AMGX` gets HYPRE plus a warning saying so, rather than a silent substitution
   that would mislabel every benchmark.

3. Build the simulator. **The CUDA architecture is auto-detected**: `native`
   when a GPU is present on the build node, `all-major` (Ampere→Hopper→Blackwell
   fat binary) otherwise. Pin it only if you want to.

   CMake:

   ```bash
   cmake -S . -B build      -DHYPRE_DIR=$HYPRE_FP32_DIR      -DAMGX_DIR=$AMGX_DIR      -DCMAKE_CUDA_ARCHITECTURES=100      -DNVRS_MIXED_PRECISION=ON
   cmake --build build -j
   ```

   The second command is the compilation step. After a later
   `git pull --ff-only origin ir-develop`, rerun only `cmake --build build -j`
   for an incremental CMake build (or `make -j` if the original build was made
   with the Makefile).

   **Which binary do the benchmarks run?** The Makefile one. They are different
   files and only one of them is used:

   | Built with | Produces | Used by the benchmarks |
   |---|---|---|
   | `make -j` | `cuda/nvrs_cuda` | **yes** |
   | `cmake --build build` | `cuda/build/nvrs_cuda` | no |

   Every script here runs `cuda/nvrs_cuda`, and prints
   `note : ignoring older .../cuda/build/nvrs_cuda` when the other exists. So a
   CMake-only build leaves the harness measuring whatever `make` produced last,
   or nothing at all. The two also find AmgX differently: CMake is handed
   `-DAMGX_DIR` outright, while the Makefile reads it from the environment, so
   one build can have AmgX and the other not from the same source tree. If you
   build both, build with `make` last.

   `-DAMGX_DIR` may be omitted if `AMGX_DIR` is already in the environment;
   configure prints which AmgX it found, or `not found -- building HYPRE-only`.

   Or the direct nvcc path (mirrors the reference link line; `-arch=native` by
   default). It reads `HYPRE_FP32_DIR` and `AMGX_DIR` straight from the
   environment, so with both `env.sh` files sourced it is one word:

   ```bash
   make -j                    # auto arch
   make -j CUDA_ARCH=100      # or pin sm_100
   ```

   `make` prints the AmgX prefix it found before linking, and that line is
   worth reading every time:

   ```
     AmgX: /path/to/amgx-install          <- good
     AmgX: not found -- HYPRE only        <- stop here
   ```

   AmgX missing is not an error. The build carries on, the run prints one
   warning, and the benchmark then measures BoomerAMG under a label that says
   AmgX. On SPE10 that is a 100-day timestep against 571 Newton iterations. The
   usual cause is a prefix holding `lib/libamgxsh.so` without
   `include/amgx_c.h`: AmgX's own install step has historically skipped the
   headers, which is why `install_amgx.sh` copies them itself and refuses to
   finish without them. A prefix in that state was never completed, and
   `FORCE_AMGX=1 ./benchmark/install_all.sh` rebuilds it. The Makefile also
   fails with the exact command to run if `HYPRE_DIR` is unset, rather than
   expanding it to `-I/include` and reporting three "cannot open source file
   HYPRE.h" errors that name nothing.

   Confirm the backend from the run banner:

   ```
   *      Mixed-precision CPR: AmgX aggregation AMG + block-Jacobi on CUDA      *
     Linear   : stage-1 multigrid from AmgX
     AmgX     : dFFI (FP32), aggregation AMG, 1 V-cycle per apply
   ```

4. Smoke test (build + tiny run, CI-friendly):

   ```bash
   ./scripts/ci_smoke_test.sh   # checks toolchain+HYPRE, builds, runs decks/SMOKE.DATA
   ```
   It exits non-zero on any failure and degrades to *build-only* when no GPU is
   visible (e.g. a CI login node). A GitHub Actions template that drives it on a
   self-hosted GPU runner is in `ci/ci.yml`.

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

**[`docs/manual.pdf`](docs/manual.pdf) is the user reference manual**: 100
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

Full reference with defaults: [`docs/manual.pdf`](docs/manual.pdf) chapter 4.

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

## Running the benchmarks

Every command below is run from `cuda/`, after sourcing the two environment
scripts and building (`make -j`).

```bash
# The whole suite: bench3p (46×112×22) + the 1M/2M/4M/5M ladder + SPE10,
# both backends, 1–4 GPUs, with OPM Flow once on bench3p as the reference.
GPUS="1 2 3 4" ./scripts/backend_study.sh all
```

```bash
# One case at a time -- bench3p | million | 2m | 4m | 5m | spe10
GPUS="1 2 3 4" ./scripts/backend_study.sh million
```

```bash
# SPE10 in depth: full 2000-day run, then a detailed report with the rock,
# a flood animation, well curves, timings and the log.
./scripts/run_spe10.sh
```

```bash
# Solver A/B on SPE10, one variable per row, on a short 400-day slice
./scripts/spe10_tune.sh
```

**Where the output goes.** All paths relative to `cuda/`:

| Path | Contents |
|---|---|
| `benchmark/backend_study/index.html` | Performance report, all cases |
| `benchmark/backend_study/study.log` | **One log for the whole study session**, in order |
| `benchmark/backend_study/<case>/<backend>/scaling.json` | Timings, Newton and Krylov counts per rank count |
| `benchmark/backend_study/<case>/<backend>/console.log` | That arm's transcript |
| `benchmark/backend_study/bench3p/opm/flow.log` | The OPM Flow reference run |
| `benchmark/backend_study/bench3p/physics.html` | Well curves and fields vs Flow |
| `benchmark/spe10/spe10.html` | **The detailed SPE10 report** |
| `benchmark/spe10/spe10_run.log` | The SPE10 run, start to finish |
| `runs_scaling/<timestamp>/<n>gpu/run.log` | Individual run output |

Useful environment variables: `GPUS` (rank counts), `STEPS` (shorten a case),
`GRID` (resize without editing the deck), `BACKENDS` (`"amgx"` to skip HYPRE),
`RUN_OPM=0` (skip the Flow arm), `SPE10_GPUS` (SPE10 rank counts, default 1).

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

## Profiling: Nsight Systems (and Nsight Compute if you want it)

Run these from `benchmark`
(and `scripts/backend_study.sh` from `..`).

`nsys` answers *where the time goes*, independently of the wall-clock split the
run claims for itself. That independence is the point: a phase boundary drawn
in the wrong place inside the code produces a self-consistent table that is
wrong, and only an outside measurement catches it.

Any benchmark profiles itself if asked. `NVRS_PROFILE=1` adds **Nsight
Systems** only:

```bash
NVRS_PROFILE=1 ./run_norne.sh                      # Norne
NVRS_PROFILE=1 ./run_comparison.sh BENCH2P BENCH3P # the generated cases
NVRS_PROFILE=1 ./scripts/backend_study.sh all      # every arm of the study
```

Or standalone, on **any deck**, without re-running a benchmark:

```bash
./profile_case.sh SPE1                 # nsys, first 10% of that deck's history
./profile_case.sh SPE10 both           # add the kernel counters
./profile_case.sh NORNE                # any case under decks/
./profile_case.sh                      # lists the decks it can see
NVRS_DECK=/path/to/MY.DATA ./profile_case.sh MYFIELD
```

The case name picks the deck (`decks/<CASE>.DATA`, unless `NVRS_DECK` names
one) and decides where the output goes. Nothing else differs between cases:
one implementation, so profiling a new field cannot quietly behave differently
from profiling Norne.

How much of the history to cover:

```bash
PROF_FRACTION=0.25 ./profile_case.sh SPE1  # a quarter of the deck's TEND
PROF_FRACTION=all  ./profile_case.sh SPE1  # the deck as written
PROF_TEND=140      ./profile_case.sh SPE1  # exactly 140 days
```

A **fraction**, not a step count, because a step count means nothing without
knowing how long a step is and that varies by deck. Norne's are a fortnight;
`profile_norne.sh` is a thin wrapper that keeps `PROF_STEPS` for the one deck
where the answer is known:

```bash
./profile_norne.sh                     # nsys, 24 report steps
PROF_STEPS=10 ./profile_norne.sh       # shorter
PROF_STEPS=all ./profile_norne.sh      # all 247 steps
```

Both backends, kept side by side rather than one overwriting the other:

```bash
./profile_case.sh SPE1                        # amgx
NVRS_BACKEND=hypre ./profile_case.sh SPE1     # hypre, alongside it
```

Then the figures and the report:

```bash
python3 profile_plots.py --case SPE1  # draws them
python3 compare.py --case SPE1        # folds them into compare_SPE1.html
```

Everything lands in `runs/<CASE>/profile/<backend>/`, and the figures go into
the **same** `runs/compare_<CASE>.html` as the field images and the animation.
There is no separate profiling report. The field images come from the full
unprofiled benchmark runs; only the profile figures come from here.

With more than one backend profiled, each figure is drawn once per backend and
named for it, `profile_timeline_amgx.png` beside `profile_timeline_hypre.png`
, and the report captions say which is which. `profile_cpr.png` is drawn once,
because it comes from the run's own timers and already shows both together.

### What each figure needs

| Figure | Needs | Answers |
|---|---|---|
| `profile_cpr.png` | **no profiler** | how the linear solve divides: AMG setup, V-cycles, stage-2, operator SpMV, CGS2 |
| `profile_timeline.png` | nsys | *when* each phase runs, the whole run above, the busiest window below |
| `profile_time.png` | nsys | *how much* GPU time each phase and kernel takes |
| `profile_roofline.png` | ncu | where each kernel sits against the roofline |
| `profile_limiters.png` | ncu | what limits each kernel: bandwidth, compute, or latency |

The first needs no profiler because the solver times its own CPR phases and
writes them to `meta.json`. That is the only way to tell a multigrid V-cycle
from the operator apply in the Krylov loop: they are the same sparse kernel, and
no trace can separate them from outside.

In `profile_timeline.png`, each row of the upper panel is shaded against **its
own** busiest moment, so colours show when a phase is active and are **not**
comparable between rows. Use `profile_time.png` for relative size.

### Nsight Compute, when you want it

Off by default and asked for explicitly, because it serialises and replays every
kernel and is roughly **100x slower per kernel**:

```bash
./profile_case.sh NORNE both         # timeline + counters
./profile_case.sh NORNE ncu          # counters only
NCU_COUNT=40 ./profile_case.sh NORNE ncu   # fewer, faster
```

Bounded by `NCU_SKIP` (launches to skip past cold start-up, default 200) and
`NCU_COUNT` (launches to profile, default 120) rather than let loose on a run.
`NCU_CLOCK` defaults to `--clock-control none`: pinning the GPU to base clocks
needs a privilege a container usually lacks, and refusing to profile is worse
than absolute rates that depend on where the clocks were. Nothing here quotes an
absolute rate, the counters give ratios, the wall clock comes from unprofiled
runs.

### Watching it, and what it is doing

The run's own output goes to the terminal and to `nsys_run.log`, and a heartbeat
prints elapsed time and GPU utilisation every 20 seconds (`NVRS_HEARTBEAT=0` to
silence, or a different interval):

```
  [profile] profiling ...   40s   util 87 %, device mem 3565 MiB in use
```

A GPU at 90% is working. A GPU at 0% for minutes is worth interrupting. The
memory figure is the **whole device**, CUDA context, library workspaces, every
process on it, not this run's arrays.

The heartbeat also covers the phase after the run, where `nsys stats` builds a
SQLite database from the report in silence. On a long trace that takes as long
as the run did.

When it finishes, the heaviest kernels are printed without opening anything:

```
  [profile] top 3 kernels by GPU time (of 46 distinct), over 140 days:
  [profile]   1.  41.2%    12345.7 ms     48231 x  amgx::spmv_kernel
```

`NVRS_TOPK` changes how many.

### The profiler options

The defaults are copied from a configuration that produces working reports on
this hardware, not derived:

```
--trace=cuda,nvtx --cuda-memory-usage=true --wait=all --mpi-impl=openmpi
```

`--mpi-impl` is passed only when the launcher was detected as Open MPI, and both
it and `--wait` are checked against `nsys profile --help` first, so an older
build is never handed an option it will reject. `NSYS_OPTS` replaces the set
entirely, useful for narrowing a profiler problem by hand, and setting it also
suppresses the automatic retry described below.

A profiled deck sets `AUTOPLOT 0`. At the end of a run the simulator shells out
to `plot_results.py`, and under a profiler that Python process is a traced
child; with `--wait=all` nsys will not finalise until it exits, so a four-second
simulation ends up held open by matplotlib for minutes. It reads as a hang, and
interrupting it aborts nsys before it flushes, which is how a run that launched
thousands of kernels produces a report containing none. `PROF_PLOTS=1` draws
them afterwards instead, with the profiler detached.

### The trace, and its size

`profile_timeline.png` is drawn from `cuda_gpu_trace`, one row per kernel launch.
A full 247-step Norne run launches on the order of ten million, so the trace is
**sampled, not truncated**: every Nth launch, with N derived from the launch
count in the kernel summary so that it fits `PROF_TRACE_ROWS` (default 400 000).
Truncating would keep the opening minutes and draw them as though they were the
history. The stride is stated in the figure whenever it is not 1.

```bash
PROF_TRACE_ROWS=2000000 ./profile_case.sh SPE1    # finer, larger file
PROF_TIMELINE=0 ./profile_case.sh SPE1            # skip the trace entirely
```

`PROF_TRACE_ROWS` applies to `profile_plots.py` as well, so set it on both when
you raise it.

### If it captures no kernels

The script prints `nsys_run.log` and the program's own output at the point of
failure, then retries once with a minimal option set. An empty kernel table
means nsys wrote no kernel rows, which happens both when a process launches none
and when the profiler cannot attach or the process dies before the buffers
flush, so read the logs rather than assuming either.

A profiled run is deliberately **shortened**. The kernel *mix* is the
benchmark's, since the same assembly, V-cycle and SpMV run at every step; the
wall-clock *total* is not, and neither the report nor the paper quotes it as one.

## References

The three interchangeable stage-1 multigrids, the reference simulator, and the
methods and cases behind every number above.

**Solver libraries**

- **AmgX** (default stage-1 multigrid, `LINSOLVER AMGX`). M. Naumov, M. Arsaev,
  P. Castonguay, J. Cohen, J. Demouth, J. Eaton, S. Layton, N. Markovskiy,
  I. Reguly, N. Sakharnykh, V. Sellappan and R. Strzodka, *AmgX: a library for
  GPU accelerated algebraic multigrid and preconditioned iterative methods*,
  SIAM J. Sci. Comput. 37(5), S602–S626, 2015.
  <https://github.com/NVIDIA/AMGX>
- **HYPRE BoomerAMG** (`LINSOLVER HYPRE`). V. E. Henson and U. M. Yang,
  *BoomerAMG: a parallel algebraic multigrid solver and preconditioner*, Appl.
  Numer. Math. 41(1), 155–177, 2002; R. D. Falgout and U. M. Yang, *hypre: a
  library of high performance preconditioners*, ICCS, 632–641, 2002.
  <https://github.com/hypre-space/hypre>
- **Ginkgo** (`LINSOLVER GINKGO`, FP16 or FP32 hierarchy). H. Anzt, T. Cojean,
  G. Flegar, F. Göbel, T. Grützmacher, P. Nayak, T. Ribizel, Y.-H. M. Tsai and
  E. S. Quintana-Ortí, *Ginkgo: a modern linear operator algebra framework for
  high performance computing*, ACM TOMS 48(1), 1–33, 2022.
  <https://github.com/ginkgo-project/ginkgo>
- **Adaptive-precision block-Jacobi** (`STAGE2 GINKGO`). G. Flegar, H. Anzt,
  T. Cojean and E. S. Quintana-Ortí, *Adaptive precision block-Jacobi for high
  performance preconditioning in the Ginkgo linear algebra software*, ACM TOMS
  47(2), 1–28, 2021.
- **OPM Flow** (the reference simulator). A. F. Rasmussen et al., *The Open
  Porous Media Flow reservoir simulator*, Comput. Math. Appl. 81, 159–185,
  2021. <https://opm-project.org>

**Method**

- J. R. Wallis, *Incomplete Gaussian elimination as a preconditioning for
  generalized conjugate gradient acceleration*, SPE 12265, 1983; H. Cao,
  H. A. Tchelepi, J. R. Wallis and H. E. Yardumian, *Parallel scalable
  unstructured CPR-type linear solver for reservoir simulation*, SPE 96809,
  2005.
- J. W. Ruge and K. Stüben, *Algebraic multigrid*, in Multigrid Methods, SIAM,
  73–130, 1987.
- Y. Saad, *A flexible inner-outer preconditioned GMRES algorithm*, SIAM
  J. Sci. Comput. 14(2), 461–469, 1993.
- S. C. Eisenstat and H. F. Walker, *Choosing the forcing terms in an inexact
  Newton method*, SIAM J. Sci. Comput. 17(1), 16–32, 1996.
- D. W. Peaceman, *Interpretation of well-block pressures in numerical
  reservoir simulation*, SPE J. 18(3), 183–194, 1978.

**Mixed precision and emulation**

- T. J. Dekker, *A floating-point technique for extending the available
  precision*, Numer. Math. 18, 224–242, 1971.
- K. Ozaki, T. Ogita, S. Oishi and S. M. Rump, *Error-free transformations of
  matrix multiplication by using fast routines of matrix multiplication and its
  applications*, Numer. Algorithms 59, 95–118, 2012; H. Ootomo, K. Ozaki and
  R. Yokota, *DGEMM on integer matrix multiplication unit*, Int. J. HPC Appl.,
  2024.
- N. J. Higham and T. Mary, *Mixed precision algorithms in numerical linear
  algebra*, Acta Numerica 31, 347–414, 2022.
- S. Williams, A. Waterman and D. Patterson, *Roofline: an insightful visual
  performance model for multicore architectures*, Comm. ACM 52(4), 65–76, 2009.

**Cases, inversion and surrogates**

- M. A. Christie and M. J. Blunt, *Tenth SPE comparative solution project*, SPE
  Reservoir Eval. Eng. 4(4), 308–317, 2001 (SPE10 model 2).
- Open Porous Media initiative, *The Norne field data set*, 2012.
- M. A. Iglesias, K. J. H. Law and A. M. Stuart, *Ensemble Kalman methods for
  inverse problems*, Inverse Problems 29(4), 045001, 2013.
- Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart
  and A. Anandkumar, *Fourier neural operator for parametric partial
  differential equations*, ICLR, 2021.

The full bibliography, with the derivations that use it, is in the user manual
(`full_reservoir_simulator/GPU_reservoir_simulator/cuda/docs/manual.pdf`,
Chapter "References").
