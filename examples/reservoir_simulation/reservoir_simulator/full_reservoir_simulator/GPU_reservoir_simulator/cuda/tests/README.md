# Tests

> Install, run commands, measured results and the paper are in the [top-level README](../../../../README.md).

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

## `test_tc_jacobian.cu`: the Tensor Core Jacobian against the analytic kernels

Runs each analytic kernel and its factorised counterpart on a small box, with random
state, PVT slopes, capillary pressure, AIM flags and THPRES, in five cases: two-phase
and three-phase on the connection list and the fixed stencil, and three-phase with
Sg/Rs/Rv switching. It compares the self blocks in FP64, then the neighbour blocks,
self blocks, pressure row and stage-2 inverse through the product. Needs a GPU.

```bash
cd full_reservoir_simulator/GPU_reservoir_simulator/cuda
make TENSORCORE=1 test_tc_jacobian
./test_tc_jacobian        # FP32 product, any GPU
./test_tc_jacobian tf32   # TF32 GEMM, needs CUTLASS and compute capability 8.0
```

