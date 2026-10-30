#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# install_all.sh -- everything the Norne comparison needs, in one command.
#
#   ./install_all.sh
#
# The order is not arbitrary, and neither is the sourcing between the steps.
# Each installer writes an environment file naming the prefix it just created,
# and the step after it needs those variables:
#
#   1. HYPRE     ->  writes Mixed_precision/hypre_env.sh
#      source it ->  HYPRE_FP32_DIR, HYPRE_FP64_DIR, and the MPI on PATH
#   2. AmgX      ->  writes <amgx prefix>/amgx_env.sh
#      source it ->  AMGX_DIR
#   3. Ginkgo    ->  writes <ginkgo prefix>/ginkgo_env.sh (CUDA + FP16)
#      source it ->  GINKGO_DIR
#   4. OPM Flow  ->  the reference simulator, with DUNE, Boost and SuiteSparse
#   5. Nsight    ->  nsys, for images that ship only Nsight Compute
#   6. nvrs_cuda ->  links against HYPRE, AmgX and Ginkgo, so it goes
#                    last and needs all of the above in the environment
#
# Doing this by hand is the sequence in the READMEs:
#
#   ./install_hypre_sm100.sh && source .../hypre_env.sh
#   ./install_amgx.sh        && source .../amgx_env.sh
#   make -j
#
# AmgX is built after sourcing HYPRE's file on purpose: that file puts the
# right mpicxx on PATH, and AmgX is configured against it.
#
# Each step is skipped when its output is already present, so re-running after
# a failure resumes rather than starting over -- but the environment files are
# sourced either way, so a resumed run has the same variables as a fresh one.
# A step that fails is reported and the rest continue: a missing AmgX still
# leaves a working HYPRE build, and a missing nsys still leaves a working
# benchmark.
#
# Nothing here needs root.  Where a dependency is missing it is unpacked into a
# prefix under this tree rather than installed, because dpkg needs to be root
# and inside a Slurm container nobody is.
#
# Environment:
#   SKIP_HYPRE=1 SKIP_AMGX=1 SKIP_GINKGO=1 SKIP_OPM=1 SKIP_NSYS=1
#   SKIP_BUILD=1
#   TENSORCORE=1                               also fetch CUTLASS and build
#                                              nvrs_cuda_tc (JACOBIAN TENSORCORE)
#   FORCE_HYPRE=1 FORCE_AMGX=1 FORCE_GINKGO=1 FORCE_OPM=1
#                                              rebuild one
#   AMGX_PREFIX=/somewhere                     where AmgX lives
#   GINKGO_PREFIX=/somewhere                   where Ginkgo lives
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CUDA_DIR/../.." && pwd)"          # full_reservoir_simulator
MIXED="$ROOT/Mixed_precision"

HYPRE_ENV="$MIXED/hypre_env.sh"
AMGX_PREFIX="${AMGX_PREFIX:-$ROOT/amgx-install}"
AMGX_ENV="$AMGX_PREFIX/amgx_env.sh"
GINKGO_PREFIX="${GINKGO_PREFIX:-$ROOT/ginkgo-install}"
GINKGO_ENV="$GINKGO_PREFIX/ginkgo_env.sh"

FAILED=()
DONE=()

hr() { printf '=%.0s' {1..78}; echo; }
step() { hr; echo " $1"; hr; }

run_step() {
  local name="$1"; shift
  if "$@"; then
    DONE+=("$name")
    return 0
  fi
  FAILED+=("$name")
  echo "  [install_all] $name did not complete; continuing with the rest." >&2
  return 1
}

# Sourced into THIS shell, not a subshell, which is the whole point: the
# variables have to reach the steps that follow.
source_env() {
  local f="$1" label="$2"
  if [[ ! -f "$f" ]]; then
    echo "  [install_all] no $label environment file at $f" >&2
    echo "  [install_all] the build will fall back to whatever it can find." >&2
    return 1
  fi
  # shellcheck disable=SC1090
  if . "$f"; then
    echo "  [install_all] sourced $label: $f"
    return 0
  fi
  echo "  [install_all] failed to source $f" >&2
  return 1
}

# --- 1. HYPRE ---------------------------------------------------------------
do_hypre() {
  local script="$MIXED/scripts/docker/install_hypre_sm100.sh"
  [[ -x "$script" ]] || { echo "  no $script" >&2; return 1; }
  # GPU-aware MPI is compiled into libHYPRE.  An install made without it while
  # this MPI can take device buffers stages every multi-GPU AMG exchange
  # through host memory, so it is rebuilt with it, automatically.
  local cfg="$MIXED/hypre-install/fp32/include/HYPRE_config.h"
  if [[ -f "$cfg" && -z "${FORCE_HYPRE:-}" ]] &&      ! grep -q 'define HYPRE_USING_GPU_AWARE_MPI' "$cfg" && command -v ompi_info >/dev/null 2>&1 &&      { ompi_info --parsable --all 2>/dev/null | grep -q 'mpi_built_with_cuda_support:value:true' ||        ompi_info 2>/dev/null | grep -qi ucx; }; then
    echo "  HYPRE was built without GPU-aware MPI, and this MPI supports it: rebuilding"
    FORCE_HYPRE=1
    export HYPRE_GPU_AWARE_MPI=ON
  fi
  if [[ -f "$MIXED/hypre-install/fp32/lib/libHYPRE.so" && -z "${FORCE_HYPRE:-}" ]]; then
    echo "  already at $MIXED/hypre-install/fp32 (FORCE_HYPRE=1 to rebuild)"
    return 0
  fi
  ( cd "$MIXED" && "$script" )
}

# --- 2. AmgX ----------------------------------------------------------------
do_amgx() {
  [[ -x "$HERE/install_amgx.sh" ]] || { echo "  no install_amgx.sh" >&2; return 1; }
  # A library without a header is not an install.  The build needs amgx_c.h to
  # compile against, and testing only for libamgxsh.so declared this step done
  # over a prefix the compiler could not use: AMGX_DIR was set, the library was
  # there, and every object still compiled without AmgX.  Both are required
  # before this reports success.
  _amgx_hdr=""
  for _h in "$AMGX_PREFIX/include/amgx_c.h" "$AMGX_PREFIX/base/include/amgx_c.h"             "$AMGX_PREFIX"/include/*/amgx_c.h; do
    [[ -f "$_h" ]] && { _amgx_hdr="$_h"; break; }
  done
  if [[ -e "$AMGX_PREFIX/lib/libamgxsh.so" && -z "$_amgx_hdr" && -z "${FORCE_AMGX:-}" ]]; then
    echo "  library at $AMGX_PREFIX/lib but no amgx_c.h under it" >&2
    echo "  that is an incomplete install and the build cannot use it; rebuilding" >&2
    FORCE_AMGX=1
  fi
  if [[ -e "$AMGX_PREFIX/lib/libamgxsh.so" && -n "$_amgx_hdr" && -z "${FORCE_AMGX:-}" ]]; then
    echo "  already at $AMGX_PREFIX (FORCE_AMGX=1 to rebuild)"
    echo "  header: $_amgx_hdr"
    # An install without its environment file is still an install.  The file is
    # written by install_amgx.sh, so a prefix built by an older version of that
    # script, or copied in, has the library and not the file -- and the build
    # then reported "AmgX: not found" over a working AmgX, linked HYPRE only,
    # and failed against objects compiled when AmgX was present.  Write it.
    if [[ ! -f "$AMGX_ENV" ]]; then
      {
        echo "# Written by install_all.sh: the install was present, this was not."
        echo "export AMGX_DIR=\"$AMGX_PREFIX\""
        echo "export LD_LIBRARY_PATH=\"$AMGX_PREFIX/lib:\${LD_LIBRARY_PATH:-}\""
      } > "$AMGX_ENV" 2>/dev/null &&         echo "  wrote the missing $AMGX_ENV" ||         echo "  could not write $AMGX_ENV; exporting AMGX_DIR for this run only" >&2
    fi
    export AMGX_DIR="$AMGX_PREFIX"
    return 0
  fi
  AMGX_PREFIX="$AMGX_PREFIX" "$HERE/install_amgx.sh"
}

# --- 3. Ginkgo --------------------------------------------------------------
# The third stage-1 multigrid, and the one that runs in half precision.  Done
# when the header, the CUDA module and half precision are all there; a Ginkgo
# without half precision is rebuilt, because LINSOLVER GINKGO defaults to FP16.
do_ginkgo() {
  [[ -x "$HERE/install_ginkgo.sh" ]] || { echo "  no install_ginkgo.sh" >&2; return 1; }
  local lib=""
  for _d in "$GINKGO_PREFIX/lib" "$GINKGO_PREFIX/lib64"; do
    [[ -e "$_d/libginkgo_cuda.so" ]] && { lib="$_d"; break; }
  done
  if [[ -n "$lib" && -f "$GINKGO_PREFIX/include/ginkgo/ginkgo.hpp" && -z "${FORCE_GINKGO:-}" ]] && \
     grep -qE '^#define[[:space:]]+GINKGO_ENABLE_HALF[[:space:]]+1' \
       "$GINKGO_PREFIX/include/ginkgo/config.hpp" 2>/dev/null; then
    echo "  already at $GINKGO_PREFIX (FORCE_GINKGO=1 to rebuild)"
    if [[ ! -f "$GINKGO_ENV" ]]; then
      {
        echo "# Written by install_all.sh: the install was present, this was not."
        echo "export GINKGO_DIR=\"$GINKGO_PREFIX\""
        echo "export LD_LIBRARY_PATH=\"$lib:\${LD_LIBRARY_PATH:-}\""
      } > "$GINKGO_ENV" 2>/dev/null && echo "  wrote the missing $GINKGO_ENV"
    fi
    export GINKGO_DIR="$GINKGO_PREFIX"
    return 0
  fi
  GINKGO_PREFIX="$GINKGO_PREFIX" "$HERE/install_ginkgo.sh"
}

# --- 4. OPM Flow ------------------------------------------------------------
# Also brings DUNE, Boost and SuiteSparse, which is why it is by far the
# longest step.  It does not need the HYPRE or AmgX environment.
do_opm() {
  [[ -x "$HERE/install_opm_gpu.sh" ]] || { echo "  no install_opm_gpu.sh" >&2; return 1; }
  local src="${OPM_SRC:-/workspace/project/opm_2024_10_src}"
  if [[ -x "$src/opm-simulators/build/bin/flow" && -z "${FORCE_OPM:-}" ]]; then
    echo "  already at $src/opm-simulators/build/bin/flow (FORCE_OPM=1 to rebuild)"
    return 0
  fi
  "$HERE/install_opm_gpu.sh"
}

# --- 5. Nsight Systems ------------------------------------------------------
do_nsys() {
  if command -v nsys >/dev/null 2>&1; then
    echo "  already on PATH: $(command -v nsys)"
    return 0
  fi
  shopt -s nullglob
  local found=(/opt/nvidia/nsight-systems*/*/target-linux-*/nsys
               "$CUDA_DIR"/tools/nsight/opt/nvidia/nsight-systems*/*/target-linux-*/nsys)
  shopt -u nullglob
  if [[ ${#found[@]} -gt 0 ]]; then
    echo "  already installed: ${found[0]}"
    return 0
  fi
  [[ -x "$HERE/install_nsys.sh" ]] || { echo "  no install_nsys.sh" >&2; return 1; }
  "$HERE/install_nsys.sh"
}

# --- 6. nvrs_cuda -----------------------------------------------------------
# Both environment files have been sourced by now, so HYPRE_FP32_DIR and
# AMGX_DIR are set and the Makefile does not have to guess.  They are reported
# rather than assumed, because a build that silently found neither still
# succeeds -- with both backends compiled out.
do_build() {
  echo "  HYPRE_FP32_DIR : ${HYPRE_FP32_DIR:-<unset>}"
  echo "  AMGX_DIR       : ${AMGX_DIR:-<unset>}"
  echo "  GINKGO_DIR     : ${GINKGO_DIR:-<unset>}"
  if [[ -z "${HYPRE_FP32_DIR:-}" && -z "${AMGX_DIR:-}" ]]; then
    echo "  [install_all] neither backend is in the environment; the build" >&2
    echo "  [install_all] will fall back to its own search." >&2
  fi
  ( cd "$CUDA_DIR" && make -j"$(nproc)" ) || return $?
  # The all-FP64 comparison binary, nvrs_cuda_fp64: the CPR preconditioner in
  # FP64 (AmgX dDDI) against the FP64 HYPRE, objects in build_fp64/.  It is
  # what the mixed-precision runs are measured against; SKIP_FP64=1 skips it.
  if [[ -n "${SKIP_FP64:-}" ]]; then
    echo "  nvrs_cuda_fp64 : skipped (SKIP_FP64=1)"
  elif [[ -z "${HYPRE_FP64_DIR:-}" ]]; then
    echo "  nvrs_cuda_fp64 : skipped, no FP64 HYPRE (HYPRE_VARIANTS must include fp64)" >&2
  else
    echo "  HYPRE_FP64_DIR : $HYPRE_FP64_DIR"
    ( cd "$CUDA_DIR" && make -j"$(nproc)" FP64_PRECOND=1 ) || return $?
  fi
  # The optional Tensor Core binary, only when asked for.  CUTLASS is header
  # only, so fetching it is all the installing it needs.
  if [[ "${TENSORCORE:-0}" == "1" ]]; then
    local cut="${CUTLASS_DIR:-$CUDA_DIR/../../cutlass}"
    if [[ ! -f "$cut/include/cutlass/cutlass.h" ]]; then
      git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git "$cut" || return $?
    fi
    ( cd "$CUDA_DIR" && make -j"$(nproc)" TENSORCORE=1 CUTLASS_DIR="$cut" ) || return $?
  fi
}

echo
step "Installing everything for the Norne comparison"
echo "  cuda dir    : $CUDA_DIR"
echo "  hypre env   : $HYPRE_ENV"
echo "  amgx prefix : $AMGX_PREFIX"
echo "  ginkgo pref : $GINKGO_PREFIX"
echo "  nproc       : $(nproc)"
echo

if [[ -z "${SKIP_HYPRE:-}" ]]; then
  step "1/6  HYPRE (FP32 stage-1 multigrid)"
  run_step HYPRE do_hypre
fi
# Sourced whether or not the step ran: a resumed run needs the same variables
# as a fresh one, and HYPRE may have been installed on a previous invocation.
source_env "$HYPRE_ENV" HYPRE || true

if [[ -z "${SKIP_AMGX:-}" ]]; then
  step "2/6  AmgX (built against the mpicxx HYPRE's env file selected)"
  run_step AmgX do_amgx
fi
source_env "$AMGX_ENV" AmgX || true

if [[ -z "${SKIP_GINKGO:-}" ]]; then
  step "3/6  Ginkgo (CUDA, half precision: the FP16 stage-1 multigrid)"
  run_step Ginkgo do_ginkgo
fi
source_env "$GINKGO_ENV" Ginkgo || true

[[ -n "${SKIP_OPM:-}"   ]] || { step "4/6  OPM Flow, DUNE, Boost, SuiteSparse"; run_step OPM  do_opm; }
[[ -n "${SKIP_NSYS:-}"  ]] || { step "5/6  Nsight Systems";                     run_step nsys do_nsys; }
[[ -n "${SKIP_BUILD:-}" ]] || { step "6/6  Building nvrs_cuda and nvrs_cuda_fp64"; run_step build do_build; }

echo
hr
echo " Summary"
hr
[[ ${#DONE[@]}   -gt 0 ]] && echo "  ok      : ${DONE[*]}"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  FAILED  : ${FAILED[*]}"
echo "  HYPRE_FP32_DIR : ${HYPRE_FP32_DIR:-<unset>}"
echo "  AMGX_DIR       : ${AMGX_DIR:-<unset>}"
echo "  GINKGO_DIR     : ${GINKGO_DIR:-<unset>}"
# Name both binaries, so the all-FP64 one is visibly there (or not).
for b in nvrs_cuda nvrs_cuda_fp64 nvrs_cuda_tc; do
  if [[ -x "$CUDA_DIR/$b" ]]; then
    printf '  %-14s : %s  (built %s)\n' "$b" "$CUDA_DIR/$b" \
      "$(date -r "$CUDA_DIR/$b" '+%Y-%m-%d %H:%M' 2>/dev/null)"
  else
    printf '  %-14s : not built\n' "$b"
  fi
done
echo

echo " Before building or running by hand in a NEW shell, source them:"
echo "   source $HYPRE_ENV"
echo "   source $AMGX_ENV"
echo "   source $GINKGO_ENV"
echo

if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo " next:"
  echo "   cd $HERE && NVRS_PROFILE=1 ./run_norne.sh"
  exit 0
fi

echo " Some steps failed.  Each is independent -- re-run this script and the"
echo " ones that succeeded are skipped, or run the failing one on its own to"
echo " see its output in full."
exit 1
