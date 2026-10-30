#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# run.sh — launch NVRS-CUDA on a single node.  One MPI rank per GPU; the grid
# is K-slab decomposed across ranks automatically.
#
#   ./run.sh decks/FIELD.DATA          # use all visible GPUs
#   NGPU=1 ./run.sh decks/FIELD.DATA   # single GPU
#   NGPU=4 ./run.sh decks/FIELD.DATA   # 4 GPUs on this node
#   NGPU=1 ./run.sh decks/NORNE.DATA   # a corner-point field deck
#
# Environment:
#   NGPU=<n>              number of ranks (default: number of *visible* GPUs)
#   NVRS_DEVICE=<ordinal> pin every rank to one device (debugging)
#   NVRS_CUDA_AWARE_MPI=1 pass device pointers to MPI in the halo exchange
set -euo pipefail

# Open MPI's startup probes hwloc's GL/OpenCL components to find which
# display each GPU is attached to.  On a machine whose X server does not
# answer that probe, it blocks forever -- mpirun never reaches its own
# startup banner, no error, no timeout, just a hang that looks like the
# simulator never started.  Neither component does anything this script
# needs, so they are off by default; a caller that wants them back can still
# set HWLOC_COMPONENTS before invoking this script.
export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl,-opencl}"

cd "$(dirname "$0")/.."          # → cuda/ project root

DECK="${1:-decks/FIELD.DATA}"
shift || true
EXTRA=("$@")          # anything else is forwarded, e.g. --jac-check
# Which binary to run.
#
# There are two build systems and they write to different places: CMake puts
# the binary in build/nvrs_cuda, the Makefile puts it in ./nvrs_cuda.  Hard-
# coding either one is a trap -- `make -j` followed by this script would build
# one binary and run the other, and a stale build/nvrs_cuda left over from an
# earlier CMake run is indistinguishable from a working one until the results
# make no sense.  That failure is silent and expensive: it presents as the
# simulator ignoring every change you just made.
#
# So take whichever exists, and if both do, the newer.
if [[ -z "${EXE:-}" ]]; then
  for cand in build/nvrs_cuda nvrs_cuda; do
    [[ -x "$cand" ]] || continue
    if [[ -z "${EXE:-}" || "$cand" -nt "$EXE" ]]; then EXE="$cand"; fi
  done
fi
EXE="${EXE:-build/nvrs_cuda}"

[[ -f "$DECK" ]] || { echo "ERROR: deck not found: $DECK" >&2; exit 1; }
[[ -x "$EXE"  ]] || { echo "ERROR: executable not found: $EXE (build first)" >&2; exit 1; }

# --- how many GPUs can this process actually use? --------------------------
# CUDA_VISIBLE_DEVICES, not nvidia-smi, decides what the process can see.
# Counting with `nvidia-smi -L` reports every GPU on the node and will happily
# launch more ranks than there are usable devices.
count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Empty string means "no GPUs"; otherwise it is a comma-separated list.
    [[ -z "${CUDA_VISIBLE_DEVICES// /}" ]] && { echo 0; return; }
    awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 0
    return
  fi
  echo 0
}

if [[ -z "${NGPU:-}" ]]; then
  NGPU="$(count_visible_gpus)"
  if [[ "$NGPU" -lt 1 ]]; then NGPU=1; fi
fi
# The Ginkgo backend is single-GPU; main.cu aborts a multi-rank run of it.
# shellcheck source=../benchmark/backend_arms.sh
. benchmark/backend_arms.sh
if [[ "$NGPU" -gt 1 ]] && deck_single_gpu "$DECK"; then
  echo "==> deck selects Ginkgo, which runs on one GPU; ranks $NGPU -> 1"
  NGPU=1
fi

# --- a field deck brings its own grid --------------------------------------
# FIELDDECK names a corner-point deck, and then the NX/NY/NZ in this deck are
# placeholders the parser wants rather than a description of anything.  The
# rank cap below must not be applied to them: NZ is typically 1 there, which
# would silently pin every field run to a single GPU.  An unstructured grid is
# not K-slab decomposed in any case -- it is cut by active-cell weight in all
# three index directions (see field_partition.hpp).
FIELDDECK="$(awk '/^[[:space:]]*FIELDDECK[[:space:]]/ {print $2; exit}' "$DECK" 2>/dev/null || true)"

if [[ -n "${FIELDDECK:-}" ]]; then
  if [[ ! -f "$FIELDDECK" ]]; then
    echo "ERROR: FIELDDECK names a deck that is not there:" >&2
    echo "         $FIELDDECK" >&2
    echo "       For Norne, fetch it first:  ./benchmark/get_norne.sh" >&2
    exit 1
  fi
  echo "==> field deck: $FIELDDECK"
else
  # --- the structured decomposition slices K, so ranks cannot exceed NZ ----
  NZ="$(awk '/^[[:space:]]*NZ[[:space:]]/ {print $2; exit}' "$DECK" 2>/dev/null || true)"
  if [[ -n "${NZ:-}" && "$NZ" =~ ^[0-9]+$ && "$NGPU" -gt "$NZ" ]]; then
    echo "==> deck has NZ=$NZ layers; reducing ranks from $NGPU to $NZ"
    echo "    (the grid is decomposed into K-slabs, so a rank per layer is the limit)"
    NGPU="$NZ"
  fi
fi

# Make HYPRE's shared lib discoverable (installed by install_hypre_sm100.sh).
if [[ -n "${HYPRE_FP32_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$HYPRE_FP32_DIR/lib:$HYPRE_FP32_DIR/lib64:${LD_LIBRARY_PATH:-}"
fi

# --- log every run -------------------------------------------------------
# Keep a timestamped transcript next to the deck's output, and a stable
# `latest.log` symlink, so a run that scrolled past or was interrupted can
# still be read back.  NVRS_LOG=<path> overrides, NVRS_LOG=0 disables.
LOGDIR="${NVRS_LOGDIR:-logs}"
LOG="${NVRS_LOG-}"
if [[ "$LOG" != "0" ]]; then
  if [[ -z "$LOG" ]]; then
    mkdir -p "$LOGDIR"
    LOG="$LOGDIR/$(basename "${DECK%.*}")_$(date +%Y%m%d_%H%M%S).log"
  fi
  mkdir -p "$(dirname "$LOG")"
fi

echo "==> NVRS-CUDA  deck=$DECK  ranks/GPUs=$NGPU${EXTRA[*]:+  args: ${EXTRA[*]}}"
# Name the binary and its age.  With two build systems writing to two places,
# "which one did I just run" is the first question worth answering when the
# results do not reflect a change that was definitely made.
echo "    exe=$EXE  built $(date -r "$EXE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo unknown)"
echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, all visible>}"
# Guarded with `if`, not `&&`: under `set -e` a false `&&` would abort here.
if [[ "$LOG" != "0" ]]; then echo "    log=$LOG"; fi

# The deck's MIXED keyword picks the binary: MIXED 0 runs nvrs_cuda_fp64, the
# all-FP64 preconditioner build (backend_arms.sh, nvrs_pick_exe).
EXE="$(nvrs_pick_exe "$EXE" "$DECK")" || exit 1

# Always launch through mpirun, even for a single rank.  Starting an
# MPI_Init'ing binary directly is a "singleton" init, which makes the MPI
# runtime pick its own transports and can fail against a mismatched UCX in the
# container ("UCP version is incompatible, required: 1.18, actual: 1.12").
# mpirun sets that environment up properly.
if [[ "$LOG" == "0" ]]; then
  exec mpirun -np "$NGPU" "$EXE" "$DECK" ${EXTRA[@]+"${EXTRA[@]}"}
fi

# Merge stderr into stdout so warnings interleave with the step table in the
# transcript, and preserve mpirun's exit status through the pipe rather than
# reporting tee's.
set -o pipefail
mpirun -np "$NGPU" "$EXE" "$DECK" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
ln -sf "$(basename "$LOG")" "$(dirname "$LOG")/latest.log" 2>/dev/null || true
echo "==> log written to $LOG"
exit "$status"
