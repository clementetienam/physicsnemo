# Tests

## The forward model these use (version 2)

The simulator under this layer is the one measured in `../README.md`: on the
published Norne field deck, one GB200, 247 report steps over 3312 days,
**44.95 s** under AmgX (46.75 s as the median of three runs) against OPM Flow's **488.51 s** (10.87x; 10.45x on the median), **71.51 s** under HYPRE BoomerAMG and **80.55 s** under Ginkgo's FP16 multigrid; on one VR72, 35.73 s, 73.54 s and 50.02 s against Flow's 243.88 s. On the matched SPE10 deck, 1,122,000 cells over 1000 days, one VR72 finishes in **41.21 s** against **766.16 s** for Flow on the host, at a mean per-well R^2 of 0.999. On Norne 19 of the 22 producers match Flow at R^2 >= 0.70 on all four rate quantities, with a mean over producers and quantities of 0.901.

## Start here: the container

Everything in this repository was measured inside one image, and it is the
supported way to build and run any of it.  From this folder:

```bash
../../../../build_container.sh
```

That builds the image if it does not exist and drops you into a shell already
in `/workspace/project`, which **is** this repository, mounted rather than
copied, so
an edit on the host is an edit in the container, and rebuilding the simulator
never means rebuilding the image.

Then, inside the container:

```bash
cd full_reservoir_simulator/GPU_reservoir_simulator/cuda
./scripts/build_deck_tests.sh dataset/NORNE/NORNE_ATW2013.DATA
```

Before trusting any multi-GPU timing, prove the image is what it claims:

```bash
../../../../build_container.sh check
```

A container can carry CUDA and an MPI and still not have **CUDA-aware** MPI,
and a successful build says nothing either way.  `check` compiles and runs a
two-rank `MPI_Send`/`MPI_Recv` on a `cudaMalloc`'d pointer and reports what
actually happened.  It matters: NVRS ranks exchange *device* buffers directly,
and a stock OpenMPI stages them through host memory. It still runs, and it is
quietly several times slower, which is the worst kind of difference because
nothing fails.

[`Dockerfile`](../../../../Dockerfile) is CUDA 12.8 on Ubuntu 22.04 with UCX
1.18.1, PMIx 4.2.8 and OpenMPI 4.1.5 built from source (`--with-cuda
--with-ucx --with-pmix --with-slurm`).  UCX's own test programs are not built: they are not used here, and pinning
them to one CUDA version is what broke the first build of this image.  The base image is multi-architecture,
so it builds on x86_64 and on aarch64 (GB200) unchanged.  The first build
compiles three libraries from source and takes tens of minutes; later builds
come from the layer cache.

[`build_container.sh`](../../../../build_container.sh) also takes `build`,
`rebuild`, `shell`, `run <cmd>` and `help`.  It uses `podman` when `docker` is
absent, and where neither is permitted (common on a shared cluster) it prints
the Apptainer equivalent.

---

Host-only, no GPU and no MPI. They check the parts of the 3-D decomposition
that fail *silently*; a wrong global row or a dropped connection does not
crash, it produces a plausible answer that is wrong.

```bash
g++ -std=c++17 -I../include -o t1 test_grid3d.cpp     && ./t1
g++ -std=c++17 -I../include -o t2 test_sparsity3d.cpp && ./t2
```

`test_grid3d`: every cell owned exactly once, rows a contiguous partition of
[0,N), all ranks agreeing on each cell's global row, ghost slabs owned by the
right neighbour, and `NVRS_DECOMP="1 1 P"` reproducing the old K-slab numbering
exactly so results from before the change stay comparable.

`test_sparsity3d`; the distributed coupling matches a serial assembly row for
row, the CSR columns come out ascending, and the structure is symmetric, which
is what catches a connection swallowed at a rank boundary.

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

## Saturation functions

`test_endscale`: ENDSCALE, the per-cell relative permeability endpoints.
Needs `nvcc` because `fluid.hpp` is device code, but runs on the host:

```bash
nvcc -std=c++17 -arch=sm_70 -I../include test_endscale.cu -o t && ./t
```

Three properties, and the third is the one that bites:

- **Identity.** A cell whose endpoints are the table's own must reproduce the
  unscaled curve to rounding. Any error in the map shows up here first,
  before any question of whether the physics improved.
- **The shift happens.** Water below a raised `SWCR` is immobile.
- **Derivatives.** Scaling multiplies every slope by the map's own; miss that
  chain rule and the Jacobian silently disagrees with the residual, Newton
  stalls, and nothing says why. Checked against central differences, for both
  the two-point and three-point (`SCALECRS YES`) paths: and a fourth check
  asserts the two methods actually differ, so a change that reads `SCALECRS`
  and ignores it cannot pass quietly.
- **Carlson hysteresis.** Both OPM two-phase subsystems are checked: gas/oil
  against historical maximum gas saturation and oil/water against historical
  maximum oil saturation. Each scanning curve is continuous at reversal and
  has reduced non-wetting mobility on imbibition.

The sampling steps deliberately off the table nodes. `kr` is piecewise linear,
so a central difference straddling a node averages two different one-sided
slopes and disagrees with the analytic value by O(1) however correct the code
is; the first run of this test "failed" for exactly that reason.

## Units

`test_units`: METRIC run decks.

```bash
nvcc -std=c++17 -I../include test_units.cpp ../src/deck.cpp -o t && ./t
```

It writes the same reservoir twice, once in metric and once in field units,
parses both, and requires every converted value to agree. That round trip is
the point: a conversion that is merely self-consistent is not enough, because
every wrong factor is also self-consistent; it simulates a different field,
quietly, and the run converges and reports success. 12811 sm³/day read as
12811 stb/day is a sixth of the intended injection and nothing downstream can
tell.

Two cases it pins that are easy to get wrong: the `-1` rate sentinel in a
`WELLS` record must survive unscaled (scaling it makes a producer into an
injector of −6.29 stb/day), and the conversion must be idempotent.

## Everything at once, without a GPU

```bash
CUDA_ARCH=70 ./scripts/syntax_check.sh
```

Compiles all 20 CUDA and C++ sources against stubbed MPI and HYPRE. It does
not produce a runnable binary; what it catches is type errors, missing
members, wrong signatures and the arithmetic of a kernel body; the things
that otherwise cost a queue wait to discover.

## Schedule and Norne parity regression

`test_schedule.cpp` is the host-only gate for dated report intervals, well and
completion state inheritance, metric rate conversion, active zero targets,
history RESV classification, group hierarchy, and `GCONINJE` parsing. Build it
with the schedule/deck sources and run it after any schedule or well change:

```bash
cd ..
./scripts/build_deck_tests.sh
# To include the Norne-specific assertions:
./scripts/build_deck_tests.sh dataset/NORNE/NORNE_ATW2013.DATA
```

The maintained suite reports **362 passing checks** across the six deck tests
(`test_cpgrid` 49, `test_schedule` 94, `test_equil` 42, `test_partition` 110,
`test_field_units` 34, `test_well_control` 41). `GCONINJE RATE` is enforced by the field-model
application path; `RESV`, `REIN` and `VREP` are deliberately parsed but not
enforced and must remain visible as limitations. For a release, also run both
Norne backends and verify that their physical curves overlay; backend
disagreement would indicate a precision/linear-path regression rather than an
OPM schedule-semantics gap.

## The field deck's own units

`test_field_units` covers METRIC *field* decks, which is where Norne's numbers
actually live.

```bash
./scripts/build_deck_tests.sh          # builds and runs it with the rest
```

`test_units` above round-trips the RUN deck. That deck is a handful of solver
settings; the grid, the wells and every control driving them come from the
corner-point deck `FIELDDECK` names, and Norne's is METRIC throughout: rates in
sm³/day, pressures in bar, depths in metres, connection factors in
cP·rm³/(day·bar). Those are the numbers the well model is handed, and none of
them were covered by a test before.

The property is the same one and the only one that catches a wrong factor: write
the same field twice, once in metric and once in field units, load both through
`load_field_model()`, and require every quantity the solver goes on to use to
agree. That is bulk and pore volume, depth, transmissibility, the WELSPECS datum, the
COMPDAT connection factor, WCONHIST's ORAT/WRAT/GRAT, WCONINJE's water and gas
rates and its BHP ceiling, WCONPROD's RESV and BHP, the GCONINJE ceiling, the
equilibrated initial pressure, and idempotence. A conversion that is merely
self-consistent passes no test at all: every wrong factor is self-consistent
too, and simulates a different field quietly.

Two things this test made possible or found:

- `to_field_units()` and `load_field_model()` moved out of `field_model.cu` into
  **`src/units_field.cpp`**, a host-only translation unit. Nothing about loading
  a deck is CUDA, and a conversion that can only be compiled by `nvcc` cannot be
  tested on a laptop, which is why it had no test. `halo_conn.hpp` now declares
  `Real` directly when it is not being compiled by `nvcc`, the same convention
  `cpgrid.hpp` and `pvt_tables.hpp` already use, and `tests/stub_halo_conn.cpp`
  supplies the one CUDA symbol `~FieldModel` reaches.
- A **FIELD** corner-point deck's cell and pore volumes came out in ft³ where a
  METRIC one came out in reservoir barrels: 5.61× too large, silently, in the
  accumulation term, the MINPV cut and the pore-volume ledger alike. Fixed in
  `cpgrid.cpp` at the one place geometry becomes a volume. Norne is METRIC and
  is unaffected; any field-unit corner-point deck was not.

## Well controls

`test_well_control` covers the flowing-pressure solve for a rate-controlled producer
when no completion is allowed to flow backwards. Pure arithmetic on
`include/well_control.hpp`, so it needs no deck, no CUDA and no GPU:

```bash
g++ -O2 -std=c++17 -I../include test_well_control.cpp -o t && ./t
```

The solve replaces one division over every completion,

    p_wf = (SUM A_k theta_k - q_target) / SUM A_k

with the constrained root of

    SUM_k A_k max(0, theta_k - p_wf) = q_target

because the first counts a completion's *reverse* flow against the total, and
lets one wellbore produce at one end while injecting at the other. OPM never
does this: `StandardWell::computePerfRate` either zeroes a reverse-flowing
connection or gives it total mobility distributed by the wellbore mixture
`cmix_s`, so its reverse flow always carries the well's own composition.

The first case in the test is the one that matters most, and it is a
**no-change** guarantee: where no completion would have flowed backwards, the
constrained solve must return the same pressure as the division it replaces. If
that stops holding, every well in every deck moves and not just the broken ones.

The rest pin the properties that make it usable inside a Newton loop:
monotonicity in the target (pressure falls, the active set only grows, so no
latching), the bottom-hole floor binding and reporting the miss rather than the
request, ties in `theta`, dead and negative mobilities, a zero target, and a
21-completion Norne-shaped well with a swept toe. That last one is the headline
number: the unconstrained solve moves **1.05 million** units of fluid to report
60,000, and the constrained one moves exactly 60,000.

Note the direction, because the natural guess is wrong: dropping completions
makes the well need **less** drawdown, not more. It was over-producing the good
completions to net out against its own re-injection. The test asserts this, and
caught it being asserted backwards the first time.

## Profiling: Nsight Systems (and Nsight Compute if you want it)

Run these from `../benchmark`
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
([`docs/manual.pdf`](../../docs/manual.pdf), chapter "References").
