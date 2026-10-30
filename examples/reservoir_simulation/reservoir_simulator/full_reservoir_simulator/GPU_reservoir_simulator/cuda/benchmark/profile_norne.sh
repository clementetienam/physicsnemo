#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# profile_norne.sh -- profile the Norne run.
#
# A thin wrapper over profile_run.sh, which does the work for every case.  One
# implementation, so the Norne profile and the BENCH/SPE10 ones cannot drift.
#
#   ./profile_norne.sh                 # nsys only: the timeline, ~1.2x runtime
#   ./profile_norne.sh both            # add Nsight Compute's kernel counters
#   ./profile_norne.sh ncu             # counters only
#   PROF_STEPS=12 ./profile_norne.sh   # shorter still; default 24 report steps
#   PROF_STEPS=all ./profile_norne.sh  # the whole 247-step history
#   NVRS_BACKEND=hypre ./profile_norne.sh
#
# nsys is the default because it is cheap enough to be routine.  ncu replays
# every kernel it profiles and is roughly 100x slower per kernel, so it is
# always asked for explicitly, here and everywhere else in this repository.
#
# Then:  python3 profile_plots.py
#        python3 compare.py --case NORNE     # folds the figures into the HTML
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
RUNS="${RUNS:-$HERE/runs}"

MODE="${1:-nsys}"
STEPS="${PROF_STEPS:-24}"

# Norne's 247 report steps span 3312 days, so a step is about a fortnight.
# That conversion is the whole reason this wrapper exists: PROF_STEPS is a
# useful thing to say about a deck whose step length is known, and a
# meaningless one about a deck whose is not, so the generic script takes a
# fraction of the history instead and this one turns steps into days first.
#
# PROF_STEPS=all profiles the deck as written, all 247 of them.
case "${STEPS,,}" in
  all|full|0)
    export PROF_FRACTION="all"
    ;;
  *)
    export PROF_TEND="${PROF_TEND:-$((STEPS * 14))}"
    ;;
esac

exec "$HERE/profile_case.sh" NORNE "$MODE"
