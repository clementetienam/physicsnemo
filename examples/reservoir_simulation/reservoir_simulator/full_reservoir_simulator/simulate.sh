#!/usr/bin/env bash
# =============================================================================
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# simulate.sh — run ONE reservoir simulation from the top of the tree.
# You never cd anywhere: run this from the folder that holds the three
# sub-folders (GPU_reservoir_simulator/, Mixed_precision/, surrogate/).
#
# You give just the deck NAME — the script finds it in the project's decks
# folder (GPU_reservoir_simulator/cuda/decks/). No full path needed.
#
# Usage (from the repository root):
#   ./simulate.sh                         # runs RESERVOIR.DATA on all GPUs
#   ./simulate.sh COMPOSITIONAL.DATA      # another bundled deck (found by name)
#   ./simulate.sh --list                  # list the available decks
#   ./simulate.sh /path/to/your.DATA      # your own deck (full path)
#   NGPU=1 ./simulate.sh                  # force a single GPU
#
# If the executable does not exist yet, build it first with:
#   ./run_all.sh --skip-smoke --skip-jaccheck --skip-hm
# =============================================================================
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$ROOT_DIR/GPU_reservoir_simulator/cuda"
EXE="$CUDA_DIR/build/nvrs_cuda"
[[ -x "$EXE" ]] || EXE="$CUDA_DIR/nvrs_cuda"      # Makefile build fallback

# ── list decks and exit, if asked ────────────────────────────────────────────
if [[ "${1:-}" == "--list" ]]; then
  echo "Available decks (pass the NAME, no path):"
  for d in "$CUDA_DIR"/decks/*.DATA; do echo "    $(basename "$d")"; done
  exit 0
fi

# ── resolve the deck: bare name → decks/<name>, else use as given ────────────
DECK="${1:-RESERVOIR.DATA}"
if [[ "$DECK" != /* && ! -f "$DECK" ]]; then DECK="$CUDA_DIR/decks/$DECK"; fi

# ── checks ───────────────────────────────────────────────────────────────────
if [[ ! -x "$EXE" ]]; then
  echo "The simulator is not built yet."
  echo "Build it once (from here, no cd) with:"
  echo "    ./run_all.sh --skip-smoke --skip-jaccheck --skip-hm"
  exit 1
fi
if [[ ! -f "$DECK" ]]; then
  echo "deck not found: $1"
  echo "Available decks (pass the NAME, no path):"
  for d in "$CUDA_DIR"/decks/*.DATA; do echo "    $(basename "$d")"; done
  exit 1
fi

# ── HYPRE shared library on the loader path ──────────────────────────────────
if [[ -f "$ROOT_DIR/Mixed_precision/hypre_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/Mixed_precision/hypre_env.sh"
fi
if [[ -n "${HYPRE_FP32_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$HYPRE_FP32_DIR/lib:$HYPRE_FP32_DIR/lib64:${LD_LIBRARY_PATH:-}"
fi

# ── how many GPUs ────────────────────────────────────────────────────────────
if [[ -z "${NGPU:-}" ]]; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [[ "${NGPU:-0}" -lt 1 ]] && NGPU=1
fi

# ── run (inside the cuda dir so OUTPUT/ + auto-plots land there) ──────────────
echo "Running: $(basename "$DECK")  on  $NGPU GPU(s)"
( cd "$CUDA_DIR" && mpirun -np "$NGPU" "$EXE" "$DECK" )

# ── report where the output went ─────────────────────────────────────────────
OUT=$(grep -E '^\s*OUTDIR' "$DECK" | awk '{print $2}' | tail -1); OUT="${OUT:-OUTPUT}"
echo
echo "Done. Results are in:  GPU_reservoir_simulator/cuda/$OUT/"
echo "  (fields, RSM_FVM.csv, plots, summary.txt)"
