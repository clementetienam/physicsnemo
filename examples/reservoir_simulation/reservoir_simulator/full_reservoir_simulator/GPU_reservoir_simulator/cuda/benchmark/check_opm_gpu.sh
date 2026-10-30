#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# check_opm_gpu.sh — does the installed OPM Flow have a working GPU linear
# solver?
#
# Worth asking separately from "did the build succeed", because a Flow built
# without CUDA is not broken: it builds, installs, runs, and produces correct
# answers on the CPU.  The only symptom is that the benchmark quietly compares
# a GPU simulator against a CPU one.  Nothing in the run output says so unless
# you know to look, which is how three benchmark runs went by on the CPU.
#
#   ./check_opm_gpu.sh              # check whatever `flow` is on PATH
#   FLOW_EXE=/path/to/flow ./check_opm_gpu.sh
#
set -uo pipefail

FLOW_EXE="${FLOW_EXE:-$(command -v flow || echo /usr/bin/flow)}"
# --- which mpirun, and which flags -------------------------------------------
# --oversubscribe and --allow-run-as-root are Open MPI options.  Passing them
# to anything else fails immediately:
#
#   mpirun: Error: unknown option "--oversubscribe"
#
# And the launcher has to match the MPI the binaries were LINKED against, which
# is not necessarily the first mpirun on PATH: OPM's configure here reported
# /opt/openmpi/lib/libmpi.so while PATH offered a different implementation.
# So look beside the library first, and add the Open MPI flags only after
# confirming the launcher is Open MPI.
#
#   MPIRUN=/path/to/mpirun   to override the choice entirely
mpi_launcher() {
  local c mr=""
  for c in "${MPIRUN:-}" "${OMPI_HOME:-/opt/openmpi}/bin/mpirun" /opt/openmpi/bin/mpirun "$(command -v mpirun 2>/dev/null || true)"; do
    if [[ -n "$c" && -x "$c" ]]; then mr="$c"; break; fi
  done
  if [[ -z "$mr" ]]; then
    echo "ERROR: no mpirun found.  Set MPIRUN=/path/to/mpirun." >&2
    exit 1
  fi
  MPI_RUN=("$mr")
  if "$mr" --version 2>&1 | grep -qiE 'open[ -]?mpi|openrte'; then
    MPI_RUN+=(--oversubscribe --allow-run-as-root)
  fi
}
mpi_launcher

BUILD_ROOT="${BUILD_ROOT:-/workspace/project/opm_2024_10_src}"

hr() { printf '%s\n' "------------------------------------------------------------"; }
verdict=""

hr
echo " Is OPM Flow built with a GPU linear solver?"
hr
if [[ ! -x "$FLOW_EXE" ]]; then
  echo "flow not found at: $FLOW_EXE" >&2
  echo "Set FLOW_EXE, or run ./install_opm_gpu.sh first." >&2
  exit 1
fi
echo "binary : $FLOW_EXE"
echo "built  : $(stat -c '%y' "$FLOW_EXE" 2>/dev/null || echo unknown)"
echo

# --- 1. Linked libraries ---------------------------------------------------
# The least ambiguous test.  A Flow compiled against CUDA links the runtime
# whatever its command-line options happen to advertise.
echo "1. CUDA libraries linked into the binary and its dependencies"
libs="$(ldd "$FLOW_EXE" 2>/dev/null | grep -iE 'libcudart|libcusparse|libcublas' || true)"
if [[ -n "$libs" ]]; then
  echo "$libs" | sed 's/^/     /'
  echo "   -> CUDA runtime IS linked."
  # The strongest single piece of evidence, and enough on its own unless a
  # later test finds an explicit CUDA complaint.
  verdict="gpu"
else
  echo "     (none)"
  echo "   -> no CUDA runtime linked.  This is a CPU-only build."
  verdict="cpu"
fi
echo

# --- 2. Is the option registered? ------------------------------------------
# --accelerator-mode is an ADVANCED parameter, listed by --help-all and not by
# --help, so checking the short help reports "no GPU" on a build that has one.
echo "2. Is --accelerator-mode registered?"
if "$FLOW_EXE" --help-all 2>&1 | grep -qi -- '--accelerator-mode' \
  || "$FLOW_EXE" --help 2>&1 | grep -qi -- '--accelerator-mode'; then
  echo "   -> yes (this alone does NOT mean the backend works; see 3)"
else
  echo "   -> not listed.  This is weak evidence at best: --help-all is not"
  echo "      accepted by every Flow build, and the parameter is advanced, so"
  echo "      absence here does not mean absence from the binary."
fi
echo

# --- 3. Does the backend actually accept work? -----------------------------
# The decisive test.  Registration is unconditional; the backend is not.  A
# build where CMake did not find CUDA advertises the option and then refuses at
# run time with "Error cusparseSolver was chosen, but CUDA was not found by
# CMake".
# Run it the way the benchmark does.  Flow calls MPI_Init even for --version,
# so a bare launch aborts in MPI_Init on a cluster that does not allow
# singleton init -- which says nothing at all about CUDA.  Read the output for
# the specific CUDA complaint instead of trusting the exit code.
echo "3. Does the cusparse backend accept a run (under mpirun -np 1)?"
out="$("${MPI_RUN[@]}" -np 1 "$FLOW_EXE" \
         --accelerator-mode=cusparse --version 2>&1)"
if grep -qiE 'was not found by CMake|cusparseSolver was chosen, but' <<<"$out"; then
  echo "$out" | grep -iE 'cuda|cusparse' | head -3 | sed 's/^/     /'
  echo "   -> REJECTED for CUDA reasons.  No GPU linear solver in this build."
  verdict="cpu"
elif grep -qiE 'unknown value for parameter|AcceleratorMode' <<<"$out"; then
  echo "$out" | grep -iE 'parameter|Accelerator' | head -2 | sed 's/^/     /'
  echo "   -> the option is not understood by this binary."
  verdict="cpu"
elif grep -qiE 'MPI_Init|MPI_ERRORS_ARE_FATAL|Local abort' <<<"$out"; then
  echo "     (MPI could not initialise even under mpirun; inconclusive)"
  echo "   -> says nothing about CUDA; relying on tests 1 and 2."
else
  echo "   -> ACCEPTED.  Flow will use the GPU."
  [[ "$verdict" != "cpu" ]] && verdict="gpu"
fi
echo

# --- 4. What CMake decided, if the build tree is still around ---------------
cache="$BUILD_ROOT/opm-simulators/build/CMakeCache.txt"
cfg="$BUILD_ROOT/opm-simulators/build/config.h"
if [[ -f "$cache" || -f "$cfg" ]]; then
  echo "4. What CMake decided at configure time"
  [[ -f "$cfg" ]] &&
    grep -iE '^\s*(#define|/\* #undef)\s+HAVE_CUDA' "$cfg" | sed 's/^/     /'
  [[ -f "$cache" ]] &&
    grep -iE '^(CUDA_TOOLKIT_INCLUDE|CUDA_CUDART_LIBRARY|CUDAToolkit_ROOT|USE_BDA_BRIDGE)' \
      "$cache" | sed 's/^/     /'
  echo
fi

hr
if [[ "$verdict" == "gpu" ]]; then
  echo " VERDICT: GPU linear solver available."
  echo " run_comparison.sh will pass --accelerator-mode=cusparse and the"
  echo " report will record it as a GPU-vs-GPU comparison."
else
  echo " VERDICT: CPU-only build."
  echo
  echo " The benchmark still runs and is still correct, but it compares a GPU"
  echo " simulator against a CPU one, and the report will say so.  To fix:"
  echo
  echo "     rm -rf $BUILD_ROOT/*/build"
  echo "     ./install_opm_gpu.sh"
  echo
  echo " Watch its first lines for"
  echo "     CUDA incl  : .../include  (cuda_runtime.h found)"
  echo " If it warns instead, stop: the rebuild will only produce another"
  echo " CPU binary."
fi
hr
[[ "$verdict" == "gpu" ]]
