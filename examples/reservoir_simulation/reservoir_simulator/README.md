# Reservoir Simulator

A GPU-resident, fully implicit reservoir simulator in CUDA C++, and an
experimental Tensor Core path for forming its Jacobian.

The simulator solves three-phase black oil with dissolved gas and vaporised
oil, and two-phase oil-water, on corner-point and Cartesian grids. Property
evaluation, analytic Jacobian assembly, algebraic multigrid and the well
calculations all execute on the device. Each Newton step solves

```text
J dx = -R
```

by flexible GMRES with a two-stage constrained-pressure-residual
preconditioner. The state, the residual and the Krylov recurrences stay in
double precision; the preconditioner is where reduced precision is allowed to
live, because that is the only place it changes the rate of convergence rather
than the answer.

## Layout

```text
reservoir_simulator/
  simulator/        sources, headers, decks and tests
  gui/              browser interface: build a deck, read a run
  docs/manual.pdf   the user manual
```

## What you need

| | |
|---|---|
| CUDA | 12.x, and a GPU of compute capability 7.0 or newer |
| Build | CMake 3.20+ or GNU make |
| Multigrid | one of AmgX, HYPRE or Ginkgo |
| MPI | any, for more than one rank |
| CUTLASS | optional, header only, for the TF32 Tensor Core path |

## Build

```bash
cmake -S simulator -B build \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DAMGX_DIR=/path/to/amgx-install \
      -DCUTLASS_DIR=/path/to/cutlass        # optional
cmake --build build -j
```

Configure prints which optional libraries it found. Anything missing is
reported and the corresponding deck option falls back and says so, rather than
running something other than what you asked for and reporting it under the
wrong name.

The Makefile is equivalent:

```bash
make -C simulator -j AMGX_DIR=/path/to/amgx-install CUTLASS_DIR=/path/to/cutlass
```

## Run

```bash
mpirun -n 1 ./build/nvrs_cuda simulator/decks/NORNE.DATA
```

Launch through `mpirun` even on one GPU. Starting the binary directly fails
under some HPC-X installations.

Smaller decks are in `simulator/decks/`: `SMOKE.DATA` runs in seconds and is
the one to try first, `SPE10_MATCHED.DATA` and `NORNE.DATA` are the two real
models.

## Deck keywords

The deck is ECLIPSE-style. Beyond the standard keywords, the ones that select
how the solve is done:

| Keyword | Values | Meaning |
|---|---|---|
| `LINSOLVER` | `AMGX`, `HYPRE`, `GINKGO` | which multigrid sits inside stage 1 of CPR |
| `JACOBIAN` | `NUMERICAL`, `ANALYTIC`, `TENSORCORE` | how the Jacobian is formed |
| `MIXED` | `0`, `1` | reduced precision in the preconditioner |
| `EMULATE` | `0`, `1` | double-word arithmetic on FP32 pairs |
| `BLOCKSPMV` | `0`, `1` | block-sparse operator, one column index per block |
| `CUDAGRAPH` | `0`, `1` | capture and replay the Krylov tail |
| `TCVALIDATE` | `0`, `1` | assemble both ways and report the difference |
| `TCPRECISION` | `TF32`, `FP32` | arithmetic for the operator product |

A keyword with a value it does not recognise is an error. A benchmark labelled
`GINKGO FP16` that silently ran FP32, or AmgX, is worse than no benchmark.

## The Tensor Core Jacobian

`JACOBIAN TENSORCORE` forms the local connection Jacobians as a chain-rule
matrix product rather than with scalar arithmetic inside the assembly kernel.
For an oriented connection the primaries are

```text
bs = 2:  [p_i, Sw_i, p_j, Sw_j]
bs = 3:  the same with the third phase, six per connection
```

and two padded operators are built,

```text
L_props = d(features)/d(primaries)      L_flux = d(residual)/d(features)
```

so that

```text
J_local = L_flux * L_props
```

is one strided-batched GEMM over every connection. Padding each tile to a
native Tensor Core shape is deliberate: it trades redundant FLOPs on the
padding lanes for regular memory access and dense arithmetic the Tensor Cores
can issue.

Both grid paths are covered, the connection list used by corner-point geometry
and the fixed six-neighbour stencil used by Cartesian grids, for both block
sizes.

The implementation is `simulator/src/tc_jacobian.cu` with its interface in
`simulator/include/tc_jacobian.hpp`. CUTLASS is optional: built with
`-DCUTLASS_DIR=...` the batch is issued as a TF32 Tensor Core GEMM on compute
capability 8.0 and newer; without it, or on older devices, the same batch runs
as an FP32 product on CUDA cores. The configure step reports which.

`TCVALIDATE 1` assembles the connection blocks both ways, by the operator
product and by the analytic path, and reports the block-wise difference.

## The browser interface

`gui/index.html` builds a run configuration, shows the deck it produces, and
reads a finished run's `summary.json` or well-rate CSV. Open it directly in a
browser; it needs no server and uploads nothing.

## Licence

Apache 2.0.
