#!/usr/bin/env bash
# =============================================================================
#  One-shot: compile + run + plot the FP64 / mixed / mixed-emu CPR benchmark.
# -----------------------------------------------------------------------------
#  Produces a 3-way solve-time BAR CHART and a 3-way convergence plot comparing:
#     fp64       : FP64 outer + FP64 AMG
#     mixed      : FP64 outer + FP32 AMG  (the shipped mixed-precision solver)
#     mixed-emu  : same, but the FP64 operator/residual are EMULATED in df64
#                  (Ozaki-2 double-single) -- runs on FP32 arithmetic
#
#  Prereqs (done once by ./scripts/docker/run_cpr_pipeline.sh):
#     * PhysicsNeMo venv built, HYPRE built for sm_100 in FP64 AND FP32,
#       and ./hypre_env.sh present (exports HYPRE_FP64_DIR / HYPRE_FP32_DIR).
#  If HYPRE is not built yet, run the full pipeline once, then use this script
#  for fast re-runs.  Set FULL=1 to chain the full pipeline first.
#
#  Usage (from the repo root):
#     chmod +x run_bench.sh
#     ./run_bench.sh                       # default sweep, all 3 variants
#     SIDES=32,64 ./run_bench.sh           # smaller/faster sweep
#     FULL=1 ./run_bench.sh                # build HYPRE first, then bench
# =============================================================================
set -Eeo pipefail

# ---- config (override via environment) --------------------------------------
VENV_ACTIVATE="${VENV_ACTIVATE:-physicsnemo_venv/bin/activate}"
HYPRE_ENV="${HYPRE_ENV:-./hypre_env.sh}"
RUNNER="${RUNNER:-./src/run_hypre_cusolver_mixed_cpr.py}"
PLOTTER="${PLOTTER:-./src/plot_cpr_compare.py}"
LOGDIR="${LOGDIR:-./runs_hypre_cusolver}"

SIDES="${SIDES:-32,64,96,128}"
TOL="${TOL:-1e-8}"
RESTART="${RESTART:-30}"
MAX_ITERS="${MAX_ITERS:-80}"
LAUNCHER="${LAUNCHER:-mpirun}"
# strengthened, GPU-tuned AMG (see src/.cu AMGConfig); override any of these
COARSEN="${COARSEN:-8}"                 # PMIS
INTERP="${INTERP:-6}"                   # ext+i
RELAX="${RELAX:-18}"                    # l1-Jacobi
AMG_SWEEPS="${AMG_SWEEPS:-2}"
AMG_AGG="${AMG_AGG:-1}"
AMG_STRONG="${AMG_STRONG:-0.5}"

# ---- pretty logging ---------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then GB=$'\e[1;32m'; R=$'\e[31m'; X=$'\e[0m'; else GB=""; R=""; X=""; fi
step(){ printf '\n%s>> %s%s\n' "$GB" "$1" "$X"; }
die(){ printf '\n%s!! %s%s\n' "$R" "$1" "$X" >&2; exit 1; }
trap 'die "failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

[[ -f "$RUNNER" ]] || die "run from the repo root (cannot find $RUNNER)"

# ---- 0) optional: build everything first ------------------------------------
if [[ "${FULL:-0}" == "1" ]]; then
  step "FULL=1: running the full build pipeline first"
  ./scripts/docker/run_cpr_pipeline.sh
fi

# ---- 1) activate venv (if present) ------------------------------------------
if [[ -f "$VENV_ACTIVATE" ]]; then
  step "activating venv: $VENV_ACTIVATE"
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

# ---- 2) source HYPRE env -----------------------------------------------------
step "sourcing HYPRE env: $HYPRE_ENV"
[[ -f "$HYPRE_ENV" ]] || die "$HYPRE_ENV not found. Build HYPRE first: FULL=1 ./run_bench.sh"
# shellcheck disable=SC1090
source "$HYPRE_ENV"
[[ -n "${HYPRE_FP64_DIR:-}" && -n "${HYPRE_FP32_DIR:-}" ]] \
  || die "HYPRE_FP64_DIR / HYPRE_FP32_DIR not set after sourcing $HYPRE_ENV"

# ---- 3) compile (x3) + run all 3 variants (the runner does this) ------------
# --emulate compiles a third binary with CPR_EMULATE_FP64=1 and runs mixed-emu.
step "compiling + running fp64 / mixed / mixed-emu  (sides=$SIDES)"
python "$RUNNER" \
  --logdir "$LOGDIR" \
  --sides "$SIDES" \
  --tol "$TOL" \
  --restart "$RESTART" \
  --max-iters "$MAX_ITERS" \
  --coarsen-type "$COARSEN" \
  --interp-type "$INTERP" \
  --relax-type "$RELAX" \
  --amg-sweeps "$AMG_SWEEPS" \
  --amg-agg-levels "$AMG_AGG" \
  --amg-strong-threshold "$AMG_STRONG" \
  --launcher "$LAUNCHER" \
  --emulate \
  --progress

# ---- 4) plot the 3-way bar chart + convergence ------------------------------
step "plotting 3-way comparison"
LATEST_JSON="$(ls -t "$LOGDIR"/*compare*.json 2>/dev/null | head -n1 || true)"
[[ -n "$LATEST_JSON" ]] || die "no *compare*.json found in $LOGDIR"
python "$PLOTTER" "$LATEST_JSON"

BASE="${LATEST_JSON%.json}"
printf '\n%s== done ==%s\n' "$GB" "$X"
printf '  bar chart   : %s_solvetime_bars.png  (+ .pdf)\n' "$BASE"
printf '  convergence : %s_convergence.png     (+ .pdf)\n' "$BASE"
printf '  raw json    : %s\n' "$LATEST_JSON"
