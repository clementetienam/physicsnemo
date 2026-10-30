# Mixed-Precision CPR: scripts and sources

> Install, run commands, measured results and the paper are in the [top-level README](../../README.md).

The study itself is written up in [`mixed_precision_fp64_fp32/README.md`](../../mixed_precision_fp64_fp32/README.md).

## Use in the full Norne simulator

This directory validates the mixed-precision CPR building block. The full
Norne parity run lives in
[`../GPU_reservoir_simulator/cuda/`](../GPU_reservoir_simulator/cuda/README.md)
and is documented in its manual. In version 2 on one GB200, AmgX (43.54 s), Ginkgo FP16 (77.00 s) and HYPRE (69.76 s) produce coincident physical curves against OPM Flow's 463.59 s, with the FP64-Krylov/reduced-precision
AMG boundary unchanged even when the hierarchy is FP16, and 19 of the 22 producers match Flow at R^2 >= 0.70 on all four rate quantities.
