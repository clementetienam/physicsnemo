#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
#
# run_scaling.sh — NVRS-CUDA strong scaling, on its own.
#
# run_comparison.sh runs OPM Flow first, and Flow takes 50 minutes on a million
# cells against NVRS's three.  When the question is "why is 2 GPUs slower than
# 1", Flow contributes nothing and costs almost all of the wall clock.  This
# runs NVRS only, on 1, 2, 4, ... GPUs, and prints where the time went at each
# rank count.
#
#   ./run_scaling.sh                        # decks/MILLION.DATA, 1 2 4 ... GPUs
#   ./run_scaling.sh decks/RESERVOIR.DATA   # a different deck
#   GPUS="1 2" ./run_scaling.sh             # only these rank counts
#   STEPS=3 ./run_scaling.sh                # shorten TEND for a quick look
#
# Writes runs_scaling/<stamp>/ with one output directory per rank count, the
# per-run logs, and scaling.json.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
# There are two ways to build this: CMake writes build/nvrs_cuda, the Makefile
# writes ./nvrs_cuda.  This script used to hardcode the CMake one, so `make &&
# ./scripts/run_scaling.sh` silently measured whatever binary CMake had left
# behind -- a fix could be built, not run, and appear not to work.
#
# Take the newer of the two, and say which, so the run is always attributable to
# a build.  NVRS_EXE still overrides.
EXE="${NVRS_EXE:-}"
if [[ -z "$EXE" ]]; then
  for c in "$CUDA_DIR/build/nvrs_cuda" "$CUDA_DIR/nvrs_cuda"; do
    [[ -x "$c" ]] || continue
    if [[ -z "$EXE" || "$c" -nt "$EXE" ]]; then EXE="$c"; fi
  done
fi
DECK="${1:-$CUDA_DIR/decks/MILLION.DATA}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$CUDA_DIR/runs_scaling/$STAMP"

if [[ -z "$EXE" || ! -x "$EXE" ]]; then
  echo "ERROR: no nvrs_cuda binary found." >&2
  echo "  cmake --build $CUDA_DIR/build -j     (writes build/nvrs_cuda)" >&2
  echo "  or: make -j                          (writes ./nvrs_cuda)" >&2
  exit 1
fi
# Name the loser too, so a binary that is present but ignored cannot be mistaken
# for the one under test.
for c in "$CUDA_DIR/build/nvrs_cuda" "$CUDA_DIR/nvrs_cuda"; do
  [[ -x "$c" && "$c" != "$EXE" ]] && \
    echo "note    : ignoring older $c" >&2
done
if [[ ! -f "$DECK" ]]; then
  echo "ERROR: no deck at $DECK" >&2
  exit 1
fi

# --- how many GPUs are there, and which rank counts to try -----------------
ndev="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
[[ -z "$ndev" || "$ndev" -lt 1 ]] && ndev=1
# The decomposition has been 3-D since the px x py x pz rewrite, so NZ is no
# longer a ceiling on the rank count -- ranks can split in I and J as well.
# The comment here still claimed a 1-D K decomposition and capped accordingly,
# which would silently drop rank counts on a thin grid.
cap=$ndev

if [[ -n "${GPUS:-}" ]]; then
  LIST=($GPUS)
else
  LIST=()
  g=1
  while [[ $g -le $cap ]]; do LIST+=("$g"); g=$((g * 2)); done
  # Include the full count when it is not a power of two.
  [[ "${LIST[-1]}" -ne "$cap" ]] && LIST+=("$cap")
fi

mkdir -p "$OUT"
echo "=== NVRS-CUDA strong scaling ==="
echo "exe     : $EXE  (built $(date -r "$EXE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null))"
[[ -n "${EXTRA:-}" ]] && echo "extra   : ${EXTRA}"
echo "deck    : $DECK"
echo "grid    : $(awk '/^[[:space:]]*N[XYZ][[:space:]]/ {printf "%s ", $2}' "$DECK")"
echo "GPUs    : ${LIST[*]}   (node has $ndev)"
[[ -n "${GRID:-}" ]] && echo "grid    : $GRID  (overriding the deck)"
echo "output  : $OUT"
echo

# shellcheck source=../benchmark/backend_arms.sh
. "$CUDA_DIR/benchmark/backend_arms.sh"

for g in "${LIST[@]}"; do
  RUNDIR="$OUT/${g}gpu"
  mkdir -p "$RUNDIR"
  # The deck names its own output directory, so give each rank count its own
  # copy rather than letting the runs overwrite each other.
  sed -E "s|^([[:space:]]*OUTDIR[[:space:]]+).*|\1$RUNDIR/OUTPUT|" "$DECK" \
    > "$RUNDIR/deck.DATA"
  grep -qE '^[[:space:]]*OUTDIR' "$RUNDIR/deck.DATA" || \
    printf '\nOUTDIR %s\n' "$RUNDIR/OUTPUT" >> "$RUNDIR/deck.DATA"
  if [[ -n "${STEPS:-}" ]]; then
    dtr="$(awk '/^[[:space:]]*DTREPORT[[:space:]]/ {print $2; exit}' "$RUNDIR/deck.DATA")"
    [[ -n "$dtr" ]] && sed -i -E \
      "s|^([[:space:]]*TEND[[:space:]]+).*|\1$(awk "BEGIN{print $dtr*$STEPS}")|" \
      "$RUNDIR/deck.DATA"
  fi
  # Plotting is a per-run Python call that has nothing to do with scaling.
  sed -i -E 's|^([[:space:]]*AUTOPLOT[[:space:]]+).*|\10|' "$RUNDIR/deck.DATA"

  # GRID="nx ny nz" resizes the problem without editing the deck.  Strong
  # scaling at a fixed size answers "is this problem faster on more GPUs"; it
  # does not answer "is there enough work per GPU to be worth it", and at 1M
  # cells over two GB200s there may simply not be.  That needs a bigger grid.
  if [[ -n "${GRID:-}" ]]; then
    read -r gx gy gz <<<"$GRID"
    sed -i -E "s|^([[:space:]]*NX[[:space:]]+).*|\1$gx|; \
               s|^([[:space:]]*NY[[:space:]]+).*|\1$gy|; \
               s|^([[:space:]]*NZ[[:space:]]+).*|\1$gz|" "$RUNDIR/deck.DATA"
  fi

  # Extra keywords, appended last so they win over whatever the deck said.
  # This is what makes an A/B a change of command line rather than a change of
  # deck, e.g.  EXTRA="AMGFAST 1"  or  EXTRA="PRECOND MGR"; separate several
  # with semicolons.
  if [[ -n "${EXTRA:-}" ]]; then
    printf '\n-- appended by run_scaling.sh --\n' >> "$RUNDIR/deck.DATA"
    printf '%s\n' "${EXTRA//;/$'\n'}" >> "$RUNDIR/deck.DATA"
  fi

  # Ginkgo is single-GPU: a multi-rank run would only reach main.cu's abort.
  if [[ "$g" -gt 1 ]] && deck_single_gpu "$RUNDIR/deck.DATA"; then
    echo "-- $g GPU(s): skipped, the deck selects Ginkgo, which runs on one GPU --"
    rm -rf "$RUNDIR"
    continue
  fi
  echo "-- $g GPU(s) --"
  t0=$(date +%s.%N)
  mpirun -np "$g" "$(nvrs_pick_exe "$EXE" "$RUNDIR/deck.DATA")" "$RUNDIR/deck.DATA" 2>&1 | tee "$RUNDIR/run.log"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s.%N)
  echo "$(awk "BEGIN{printf \"%.3f\", $t1-$t0}")" > "$RUNDIR/wall.txt"
  echo "   rc=$rc  wall=$(cat "$RUNDIR/wall.txt")s"
  echo
  [[ $rc -ne 0 ]] && { echo "run on $g GPU(s) failed; see $RUNDIR/run.log" >&2; }
done

python3 "$HERE/scaling_report.py" "$OUT"
echo
echo "Artifacts in $OUT"
