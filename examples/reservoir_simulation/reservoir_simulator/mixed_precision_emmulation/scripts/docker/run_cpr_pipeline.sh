#!/usr/bin/env bash
# =============================================================================
#  HYPRE + cuSolver mixed-precision CPR  —  build & run pipeline
# -----------------------------------------------------------------------------
#  Steps, in order (all in this one shell so venv + env vars persist):
#    1. install PhysicsNeMo            (skippable: SKIP_PHYNEMO=1)
#    2. source the venv
#    3. ensure CMake >= 3.23           (auto-upgrade; Umpire/HYPRE need it)
#    4. build HYPRE (sm_100, FP64+FP32) after wiping stale build dirs
#    5. source ./hypre_env.sh
#    6. run the mixed-CPR sweep
#    7. plot the results
#
#  Usage:
#    chmod +x run_cpr_pipeline.sh
#    ./run_cpr_pipeline.sh
#
#  Resume after a mid-pipeline failure (skip the slow PhysicsNeMo install):
#    SKIP_PHYNEMO=1 ./run_cpr_pipeline.sh
#
#  Run from the repository root (the dir containing ./scripts and ./src).
#  Override any path/param via environment, e.g.  SIDES=32,64,96 ./run_cpr_pipeline.sh
# =============================================================================

# Fail fast; keep error traps through functions. `-u` is left OFF because
# third-party activate/env scripts often reference unset variables.
set -Eeo pipefail

# -----------------------------------------------------------------------------
# Green aesthetics (auto-disabled when not a TTY or when NO_COLOR is set)
# -----------------------------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  G=$'\e[32m'; GB=$'\e[1;32m'; DIM=$'\e[2;32m'; R=$'\e[31m'; X=$'\e[0m'
else
  G=""; GB=""; DIM=""; R=""; X=""
fi

STEP=0
banner() {
  printf '%s\n' "${GB}"
  printf '%s\n' "  ┌──────────────────────────────────────────────────────────┐"
  printf '%s\n' "  │      HYPRE + cuSolver mixed-CPR  •  build & run            │"
  printf '%s\n' "  └──────────────────────────────────────────────────────────┘${X}"
}
step() { STEP=$((STEP + 1)); printf '\n%s[%d] %s%s\n' "${GB}" "${STEP}" "$1" "${X}"; }
log()  { printf '%s    ↳ %s%s\n' "${DIM}" "$1" "${X}"; }
ok()   { printf '%s    ✓ %s%s\n' "${G}" "$1" "${X}"; }
die()  { printf '\n%s    ✗ %s%s\n' "${R}" "$1" "${X}" >&2; exit 1; }

trap 'die "failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

# Run a command, echo it in dim green, and time it.
run() {
  log "\$ $*"
  local t0=${SECONDS}
  "$@"
  ok "done in $((SECONDS - t0))s"
}

# -----------------------------------------------------------------------------
# Configurable paths / parameters (override via environment)
# -----------------------------------------------------------------------------
INSTALL_PHYNEMO="${INSTALL_PHYNEMO:-./scripts/docker/install_phynemo.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-physicsnemo_venv/bin/activate}"
INSTALL_HYPRE="${INSTALL_HYPRE:-./scripts/docker/install_hypre_sm100.sh}"
HYPRE_ENV="${HYPRE_ENV:-./hypre_env.sh}"
RUNNER="${RUNNER:-./src/run_hypre_cusolver_mixed_cpr.py}"
PLOTTER="${PLOTTER:-./src/plot_cpr_compare.py}"

LOGDIR="${LOGDIR:-./runs_hypre_cusolver}"
SIDES="${SIDES:-32,64,96,128}"
TOL="${TOL:-1e-8}"
RESTART="${RESTART:-30}"
MAX_ITERS="${MAX_ITERS:-80}"
AMG_PRINT_LEVEL="${AMG_PRINT_LEVEL:-0}"
LAUNCHER="${LAUNCHER:-mpirun}"

# CMake bootstrap knobs
CMAKE_MIN="${CMAKE_MIN:-3.23}"                 # Umpire 2025.x requires >= 3.23
CMAKE_VERSION="${CMAKE_VERSION:-3.30.5}"       # version fetched if pip route fails
# Stale build dirs to wipe so the new CMake is actually used (cache pins it)
HYPRE_BUILD_DIRS="${HYPRE_BUILD_DIRS:-hypre/build-fp64 hypre/build-fp32}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"                # set 0 to keep existing build dirs

# Skip switches (for resuming)
SKIP_PHYNEMO="${SKIP_PHYNEMO:-0}"
SKIP_HYPRE="${SKIP_HYPRE:-0}"

# helper: run a script directly if executable, else via bash
exec_script() {
  local s="$1"
  [[ -f "$s" ]] || die "script not found: $s (are you in the repo root?)"
  if [[ -x "$s" ]]; then run "$s"; else run bash "$s"; fi
}

# version_ge A B  -> true if A >= B  (semantic version compare)
version_ge() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]; }

cmake_ver() { command -v cmake >/dev/null 2>&1 && cmake --version | head -n1 | awk '{print $3}'; }

# -----------------------------------------------------------------------------
# Ensure CMake >= CMAKE_MIN. Tries: (a) existing cmake, (b) pip into the active
# venv, (c) Kitware prebuilt tarball for this architecture.
# -----------------------------------------------------------------------------
ensure_cmake() {
  local cur; cur="$(cmake_ver || true)"
  if [[ -n "$cur" ]] && version_ge "$cur" "$CMAKE_MIN"; then
    ok "cmake $cur (>= $CMAKE_MIN) at $(command -v cmake)"; return 0
  fi
  log "cmake ${cur:-not found} < ${CMAKE_MIN}; attempting upgrade"

  # (b) pip into the active venv
  if command -v python >/dev/null 2>&1; then
    log "trying: python -m pip install --upgrade 'cmake>=${CMAKE_MIN}'"
    if python -m pip install --upgrade "cmake>=${CMAKE_MIN}" >/dev/null 2>&1; then
      hash -r; cur="$(cmake_ver || true)"
    fi
  fi

  # (c) Kitware prebuilt tarball fallback
  if ! { [[ -n "$cur" ]] && version_ge "$cur" "$CMAKE_MIN"; }; then
    local arch url dest
    arch="$(uname -m)"   # aarch64 (GB200) or x86_64
    url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${arch}.tar.gz"
    dest="${CMAKE_PREFIX_LOCAL:-$PWD/.cmake-${CMAKE_VERSION}}"
    log "pip route insufficient; fetching ${url}"
    rm -rf "$dest"; mkdir -p "$dest"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "$url" | tar xz -C "$dest" --strip-components=1
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- "$url" | tar xz -C "$dest" --strip-components=1
    else
      die "need curl or wget to download CMake ${CMAKE_VERSION}"
    fi
    export PATH="$dest/bin:$PATH"
    hash -r; cur="$(cmake_ver || true)"
  fi

  version_ge "${cur:-0}" "$CMAKE_MIN" \
    || die "CMake upgrade failed (have ${cur:-none}, need >= ${CMAKE_MIN})"
  # Pin CMAKE for build scripts that honour it, in case one hardcodes a path.
  export CMAKE; CMAKE="$(command -v cmake)"
  ok "cmake $cur at ${CMAKE}"
}

# Wipe half-configured HYPRE build trees so the upgraded CMake is picked up.
clean_hypre_builds() {
  if [[ "${CLEAN_BUILD}" != "1" ]]; then log "CLEAN_BUILD=0; keeping build dirs"; return 0; fi
  local d
  for d in ${HYPRE_BUILD_DIRS}; do
    if [[ -d "$d" ]]; then log "removing stale build dir: $d"; rm -rf "$d"; fi
  done
}

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
banner
PIPE_T0=${SECONDS}
log "working directory: $(pwd)"

# 1) Install / build PhysicsNeMo (skippable when resuming)
if [[ "${SKIP_PHYNEMO}" == "1" ]]; then
  step "Installing PhysicsNeMo (skipped: SKIP_PHYNEMO=1)"
else
  step "Installing PhysicsNeMo"
  exec_script "${INSTALL_PHYNEMO}"
fi

# 2) Activate the PhysicsNeMo virtual environment (must be sourced)
step "Activating venv: ${VENV_ACTIVATE}"
[[ -f "${VENV_ACTIVATE}" ]] || die "venv activate not found: ${VENV_ACTIVATE}"
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
ok "venv active: ${VIRTUAL_ENV:-unknown}  ($(command -v python))"

# 3) Ensure a new-enough CMake (Umpire/HYPRE require >= 3.23)
step "Ensuring CMake >= ${CMAKE_MIN}"
ensure_cmake

# 4) Build HYPRE for sm_100 (FP64 + FP32), after clearing stale build dirs
if [[ "${SKIP_HYPRE}" == "1" ]]; then
  step "Building HYPRE (skipped: SKIP_HYPRE=1)"
else
  step "Building HYPRE (sm_100)"
  clean_hypre_builds
  exec_script "${INSTALL_HYPRE}"
fi

# 5) Export HYPRE environment (HYPRE_FP64_DIR / HYPRE_FP32_DIR) — sourced
step "Sourcing HYPRE env: ${HYPRE_ENV}"
[[ -f "${HYPRE_ENV}" ]] || die "HYPRE env script not found: ${HYPRE_ENV}"
# shellcheck disable=SC1090
source "${HYPRE_ENV}"
[[ -n "${HYPRE_FP64_DIR:-}" ]] || die "HYPRE_FP64_DIR not set after sourcing ${HYPRE_ENV}"
[[ -n "${HYPRE_FP32_DIR:-}" ]] || die "HYPRE_FP32_DIR not set after sourcing ${HYPRE_ENV}"
ok "HYPRE_FP64_DIR=${HYPRE_FP64_DIR}"
ok "HYPRE_FP32_DIR=${HYPRE_FP32_DIR}"

# 6) Run the mixed-precision CPR sweep
step "Running mixed-CPR benchmark"
[[ -f "${RUNNER}" ]] || die "runner not found: ${RUNNER}"
run python "${RUNNER}" \
  --logdir "${LOGDIR}" \
  --sides "${SIDES}" \
  --tol "${TOL}" \
  --restart "${RESTART}" \
  --max-iters "${MAX_ITERS}" \
  --amg-print-level "${AMG_PRINT_LEVEL}" \
  --progress \
  --launcher "${LAUNCHER}"

# 7) Plot the results (convergence + solve-time bars) from the newest JSON
step "Plotting results"
[[ -f "${PLOTTER}" ]] || die "plotter not found: ${PLOTTER}"
LATEST_JSON="$(ls -t "${LOGDIR}"/*compare*.json 2>/dev/null | head -n1 || true)"
[[ -n "${LATEST_JSON}" ]] || die "no *compare*.json found in ${LOGDIR}"
log "plotting: ${LATEST_JSON}"
run python "${PLOTTER}" "${LATEST_JSON}"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
printf '\n%s════════════════════════════════════════════════════════════%s\n' "${GB}" "${X}"
ok "pipeline finished in $((SECONDS - PIPE_T0))s"
log "results written under: ${LOGDIR}"
log "plots: ${LATEST_JSON%.json}_convergence.png  +  _solvetime_bars.png  (+ .pdf)"
printf '%s════════════════════════════════════════════════════════════%s\n' "${GB}" "${X}"