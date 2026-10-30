#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# quickstart.sh — build everything and run one reservoir simulation.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
#   ./quickstart.sh
#
# Four stages, each skipped if already done:
#   1. build HYPRE (FP32 + FP64) with CUDA support        ~10-20 min, once
#   2. build AmgX, the default stage-1 multigrid          ~15-30 min, once
#   3. build the simulator                                 ~2 min
#   4. run decks/FIELD.DATA and render the figures         ~1 min
#
# Environment overrides:
#   CUDA_ARCH=90        GPU architecture (70 V100, 80 A100, 90 H100, 100 GB200)
#   DECK=decks/X.DATA   run a different deck
#   NGPU=4              number of GPUs / MPI ranks
#   SKIP_HYPRE=1        assume HYPRE_FP32_DIR is already set
#   SKIP_AMGX=1         build without AmgX; the simulator then runs on HYPRE
#                       BoomerAMG and says so at start-up
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_ARCH="${CUDA_ARCH:-}"
DECK="${DECK:-decks/FIELD.DATA}"

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 0. Prerequisites
# --------------------------------------------------------------------------
say "Checking prerequisites"
for t in nvcc cmake mpicxx git; do
  command -v "$t" >/dev/null 2>&1 || die "$t not found in PATH"
  printf '    %-8s %s\n' "$t" "$(command -v "$t")"
done
command -v python3 >/dev/null 2>&1 || warn "python3 not found — plotting will be skipped"

# Detect the GPU architecture if the caller did not pin one.
if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
  if [[ -n "$cc" ]]; then
    CUDA_ARCH="${cc//./}"
    printf '    %-8s sm_%s (detected)\n' "gpu" "$CUDA_ARCH"
  fi
fi
[[ -z "$CUDA_ARCH" ]] && { CUDA_ARCH=90; warn "no GPU detected; assuming sm_${CUDA_ARCH} (override with CUDA_ARCH=)"; }

# --------------------------------------------------------------------------
# 1. HYPRE
# --------------------------------------------------------------------------
if [[ "${SKIP_HYPRE:-0}" != "1" ]]; then
  if [[ -f "$ROOT/Mixed_precision/hypre_env.sh" ]]; then
    say "Reusing the existing HYPRE build"
    # shellcheck disable=SC1091
    source "$ROOT/Mixed_precision/hypre_env.sh"
  else
    say "Building HYPRE with CUDA (once; this is the slow step)"
    # Invoked through `bash` rather than executed directly: a checkout made on a
    # filesystem without POSIX permissions (Windows, some CI runners, a zip
    # download) loses the executable bit, and the failure is a confusing
    # "Permission denied" rather than anything to do with the build.
    ( cd "$ROOT/Mixed_precision" \
      && CUDA_ARCH="$CUDA_ARCH" bash ./scripts/docker/install_hypre_sm100.sh )
    # shellcheck disable=SC1091
    source "$ROOT/Mixed_precision/hypre_env.sh"
  fi
fi
[[ -n "${HYPRE_FP32_DIR:-}" ]] || die "HYPRE_FP32_DIR is not set (source Mixed_precision/hypre_env.sh)"
printf '    HYPRE_FP32_DIR=%s\n' "$HYPRE_FP32_DIR"

# --------------------------------------------------------------------------
# 2. Build the simulator
# --------------------------------------------------------------------------
CUDA_DIR="$ROOT/GPU_reservoir_simulator/cuda"

# --------------------------------------------------------------------------
# AmgX -- the default stage-1 multigrid
# --------------------------------------------------------------------------
# Optional.  Without it the simulator builds and runs on HYPRE BoomerAMG and
# says so at start-up, rather than silently substituting a backend the deck did
# not ask for.  Worth the wait: on 46x112x22 AmgX is 4.5x faster than BoomerAMG,
# and it is the backend that keeps scaling past one GPU.
if [[ "${SKIP_AMGX:-0}" != "1" ]]; then
  if [[ -f "$ROOT/amgx-install/amgx_env.sh" ]]; then
    say "Reusing the existing AmgX build"
  else
    say "Building AmgX (once; the second slow step)"
    CUDA_ARCH="$CUDA_ARCH" bash "$CUDA_DIR/benchmark/install_amgx.sh" \
      || say "AmgX build failed; continuing with HYPRE only"
  fi
  if [[ -f "$ROOT/amgx-install/amgx_env.sh" ]]; then
    source "$ROOT/amgx-install/amgx_env.sh"
  fi
fi
if [[ -n "${AMGX_DIR:-}" ]]; then
  printf '    AMGX_DIR=%s\n' "$AMGX_DIR"
else
  printf '    AmgX not built -- the simulator will use HYPRE BoomerAMG\n'
fi

say "Building the simulator (sm_$CUDA_ARCH)"
cmake -S "$CUDA_DIR" -B "$CUDA_DIR/build" \
      -DHYPRE_DIR="$HYPRE_FP32_DIR" \
      ${AMGX_DIR:+-DAMGX_DIR="$AMGX_DIR"} \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DNVRS_MIXED_PRECISION=ON >/dev/null
cmake --build "$CUDA_DIR/build" -j
[[ -x "$CUDA_DIR/build/nvrs_cuda" ]] || die "build produced no executable"

# --------------------------------------------------------------------------
# 3. Run
# --------------------------------------------------------------------------
# Run from cuda/ so the deck's relative PERMFILE path resolves.
cd "$CUDA_DIR"
NGPU="${NGPU:-1}"
say "Running $DECK on $NGPU GPU(s)"
export LD_LIBRARY_PATH="$HYPRE_FP32_DIR/lib:$HYPRE_FP32_DIR/lib64:${LD_LIBRARY_PATH:-}"
if [[ "$NGPU" -gt 1 ]]; then
  mpirun -np "$NGPU" ./build/nvrs_cuda "$DECK"
else
  ./build/nvrs_cuda "$DECK"
fi

OUT="$(awk '/^OUTDIR/ {print $2}' "$DECK" | tail -1)"
OUT="${OUT:-OUTPUT}"
say "Done"
cat <<EOF
    Results:        $CUDA_DIR/$OUT
    Figures/report: $CUDA_DIR/$OUT/index.html
    ECLIPSE files:  $CUDA_DIR/$OUT/*.{EGRID,INIT,UNRST,SMSPEC,UNSMRY,RSM}

    Re-plot at any time:
      python3 scripts/plot_results.py $OUT
EOF
