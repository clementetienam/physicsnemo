#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# run_comparison.sh — run both simulators on the same case and build the
# comparison report.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
#   ./run_comparison.sh                  # both base cases, then compare.html
#   ./run_comparison.sh BENCH2P          # one case
#   ./run_comparison.sh BENCH2P_L        # the same reservoir, 129,600 cells
#   OPM_RANKS=16 NVRS_RANKS=1 ./run_comparison.sh
#   BENCH_GRID="46 112 22" ./run_comparison.sh BENCH3P    # a one-off size
#
# Cases, all the same 2000 x 2000 x 60 ft reservoir and the same rock:
#
#   BENCH2P     BENCH3P        40 x  40 x  3  =     4,800 cells
#   BENCH2P_L   BENCH3P_L     120 x 120 x  9  =   129,600
#   BENCH2P_XL  BENCH3P_XL    200 x 200 x 15  =   600,000
#
# The 2P/3P suffix is the number of PHASES; _L and _XL are the number of CELLS.
# A larger case is the same problem at higher resolution -- same pore volume,
# same wells in the same physical spots -- so only the linear system grows.
#
# Layout produced, one directory per case:
#
#   benchmark/runs/<CASE>/<CASE>.DATA          NVRS deck (well locations)
#   benchmark/runs/<CASE>/<CASE>_OPM.DATA      ECLIPSE deck
#   benchmark/runs/<CASE>/opm/                 Flow's output
#   benchmark/runs/<CASE>/nvrs/                NVRS's output
#   benchmark/runs/<CASE>/timing.json          wall clock for both
#   benchmark/runs/compare.html                the report
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
# deck_single_gpu: Ginkgo decks run on one rank (see backend_arms.sh).
# shellcheck source=backend_arms.sh
. "$HERE/backend_arms.sh"
RUNS="$HERE/runs"
# --- PTX JIT cache -----------------------------------------------------------
# Anything built for an architecture the GPU does not match exactly arrives as
# PTX and is compiled by the driver at first launch.  On Vera Rubin that made
# report step 1 take 63.5 s against 1.5 s for the average step.
#
# The driver caches the compiled code, so it is a one-off -- provided the cache
# persists.  Its default home is ~/.nv/ComputeCache, which inside a container is
# usually not on the mounted filesystem, so it vanishes on exit and every run
# pays the cost again, each time inflating the wall clock by a different amount.
#
# So: put it beside the run outputs, which are on the mount by construction, and
# raise the size limit -- the default is well under what a whole HYPRE library's
# PTX needs, and an overflowing cache evicts and re-compiles forever.  4 GiB is
# the maximum the driver accepts.
#
# On a machine whose binaries carry native cubins nothing is JIT-compiled and
# this costs nothing: the directory simply stays empty.
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNS:-$OUT}/.jitcache}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
mkdir -p "$CUDA_CACHE_PATH" 2>/dev/null || true
if [[ "${CUDA_CACHE_DISABLE:-0}" == "1" ]]; then
  echo " jit cache      : DISABLED by CUDA_CACHE_DISABLE=1 (every run re-compiles)"
else
  echo " jit cache      : $CUDA_CACHE_PATH ($(du -sh "$CUDA_CACHE_PATH" 2>/dev/null | cut -f1))"
fi


# CMake writes build/nvrs_cuda, the Makefile writes ./nvrs_cuda.  Hardcoding
# the CMake path meant "make -j && ./run_comparison.sh" benchmarked whatever
# CMake had left behind -- which, against Flow, would publish a number for a
# binary nobody built.  Take the newer of the two and say which.
NVRS_EXE="${NVRS_EXE:-}"
if [[ -z "$NVRS_EXE" ]]; then
  for c in "$CUDA_DIR/build/nvrs_cuda" "$CUDA_DIR/nvrs_cuda"; do
    [[ -x "$c" ]] || continue
    if [[ -z "$NVRS_EXE" || "$c" -nt "$NVRS_EXE" ]]; then NVRS_EXE="$c"; fi
  done
fi
# A source-tree install/ comes before PATH: /usr/bin/flow is usually an older
# system build, and running that while a freshly built GPU-capable one sits
# in the OPM tree is how a benchmark ends up on the CPU without saying so.
OPM_SRC="${OPM_SRC:-/workspace/project/opm_2024_10_src}"
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

# The unpacked dependency prefix, if install_opm_gpu.sh had to build one.
# flow links libumfpack from there, and libumfpack in turn needs libamd.so.2,
# which the loader will not find on its own:
#
#   flow: error while loading shared libraries: libamd.so.2
#
# RPATH does not solve this.  OPM builds with -Wl,--enable-new-dtags, so the
# linker writes DT_RUNPATH, and DT_RUNPATH applies only to an object's own
# direct dependencies -- not to what those dependencies need in turn.
# LD_LIBRARY_PATH is searched for every object in the process, so it is what
# works.
_deps="$OPM_SRC/deps-userland/usr"
if [[ -d "$_deps/lib" ]]; then
  for _d in "$_deps"/lib "$_deps"/lib/*/; do
    [[ -d "$_d" ]] || continue
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${_d%/}:"*) ;;
      *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${_d%/}" ;;
    esac
  done
fi

if [[ -z "${FLOW_EXE:-}" ]]; then
  for cand in "$OPM_SRC/install/bin/flow" \
              "$OPM_SRC/opm-simulators/build/bin/flow" \
              "$(command -v flow || true)" /usr/bin/flow; do
    if [[ -n "$cand" && -x "$cand" ]]; then FLOW_EXE="$cand"; break; fi
  done
fi
FLOW_EXE="${FLOW_EXE:-/usr/bin/flow}"
# 8 rather than 32: the benchmark grid is 4800 cells, and on 32 ranks the load
# balancer hands some ranks under 100 owned cells with more overlap than
# interior.  That starves Flow's AMG of anything to coarsen and makes the
# wall-clock comparison unfair to it.  Override for a larger model.
OPM_RANKS="${OPM_RANKS:-8}"
OPM_RANKS_SET="${OPM_RANKS+set}"
NVRS_RANKS="${NVRS_RANKS:-1}"
if [[ $# -gt 0 ]]; then CASES=("$@"); else CASES=(BENCH2P BENCH3P); fi

LOGS="$HERE/logs"
mkdir -p "$RUNS" "$LOGS"

# One timestamp for the whole invocation, so a run's transcript and its
# per-phase logs sort together and a second run of the same case does not
# overwrite the first.
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOGS/run_${RUN_STAMP}.log"

# Tee the WHOLE run, not just the two simulators.  The per-phase logs already
# captured Flow and NVRS, but the decisions made around them -- which linear
# solver was selected and why, which rank count, what the deck self-check
# reported, what the water balance came out at -- only ever reached the
# terminal, and those are the lines needed to explain a result afterwards.
#
# stdout and stderr are kept on fds 3 and 4 so `finish` can restore them and
# let tee drain; without that the last lines can be lost when the shell exits
# before the tee subprocess has flushed.
exec 3>&1 4>&2
exec > >(tee -a "$MAIN_LOG") 2>&1
finish() {
  exec 1>&3 2>&4 3>&- 4>&-
  wait 2>/dev/null || true
  echo "Full transcript : $MAIN_LOG"
}
trap finish EXIT

hr() { printf '%s\n' "$(printf '=%.0s' {1..72})"; }

# Flow reports a bad deck as one "Error:" line and then aborts every rank, so
# the message that matters arrives buried under one backtrace per rank.  Pull
# it back out rather than making the reader scroll for it.
why() {
  local log="$1" who="$2"
  echo
  echo "  >>> $who reported:"
  grep -iE 'Error:|Exception|terminate called|^ *Warning:' "$log" 2>/dev/null \
    | grep -viE 'End of error message|backtrace' | sort -u | head -20 \
    | sed 's/^/      /'
  echo "  >>> full log: $log"
  echo
}

hr
echo " NVRS-CUDA vs OPM Flow"
echo " started        : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " nvrs           : $NVRS_EXE  (built $(date -r "$NVRS_EXE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null))"
echo " flow           : $FLOW_EXE"
echo " ranks          : OPM $OPM_RANKS, NVRS $NVRS_RANKS"
echo " cases          : ${CASES[*]}"
echo " transcript     : $MAIN_LOG"
hr

[[ -n "$NVRS_EXE" && -x "$NVRS_EXE" ]] || {
  echo "ERROR: no nvrs_cuda binary found." >&2
  echo "  cmake --build $CUDA_DIR/build -j   (writes build/nvrs_cuda)" >&2
  echo "  or: make -j                        (writes ./nvrs_cuda)" >&2
  exit 1; }
for c in "$CUDA_DIR/build/nvrs_cuda" "$CUDA_DIR/nvrs_cuda"; do
  [[ -x "$c" && "$c" != "$NVRS_EXE" ]] && echo " note           : ignoring older $c"
done
# Only when a case will actually use it.  NEED_FLOW is decided below, so this
# check moved down with it rather than rejecting an NVRS-only run for the
# absence of a binary it was never going to call.

# --- Hardware, recorded so the wall-clock table means something ------------
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_DESC="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  NGPU="$(nvidia-smi --list-gpus | wc -l)"
  GPU_DESC="${NGPU}x ${GPU_DESC}"
else
  GPU_DESC="no GPU detected"
fi
CPU_DESC="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- \
            | sed 's/^ *//' || echo unknown)"
CPU_DESC="$(nproc) cores, ${CPU_DESC}"

# What actually puts Flow's linear solve on the GPU is the solver JSON --
# "solver": "gpubicgstab" with a GPU preconditioner selects the gpuistl path.
# --accelerator-mode selects the older BDA bridge and is a separate mechanism;
# both are passed because they coexist and that combination is the one proven
# on the Aramco runs.
#
# GPU support is settled by asking the binary to accept the option, not by
# looking for it in the help: a build whose CMake never found CUDA still
# registers --accelerator-mode and only refuses when asked to use it.
# Decide whether this Flow can use the GPU.
#
# Three sources of evidence, because each one alone gives a wrong answer:
#
#   * ldd for libcusparse -- necessary, and the hardest to fake.
#   * HAVE_CUDA in the build's config.h -- what CMake actually concluded.
#   * a run under mpirun -- confirmatory ONLY, and read for the specific CUDA
#     complaint rather than for its exit code.  Flow calls MPI_Init even for
#     --version, so launching it bare aborts in MPI_Init on a cluster that does
#     not allow singleton init.  That is a statement about MPI, not about CUDA,
#     and treating it as "no GPU" is how a GPU-capable build gets reported as
#     CPU-only.
opm_has_gpu() {
  local out linked=0 cfg_cuda=0
  ldd "$FLOW_EXE" 2>/dev/null | grep -qi 'libcusparse' && linked=1
  local cfg="$OPM_SRC/opm-simulators/build/config.h"
  [[ -f "$cfg" ]] && grep -qE '^[[:space:]]*#define[[:space:]]+HAVE_CUDA[[:space:]]+1' \
    "$cfg" && cfg_cuda=1

  out="$("${MPI_RUN[@]}" -np 1 "$FLOW_EXE" \
           --accelerator-mode=cusparse --version 2>&1)"
  # Only this message disqualifies a build.  Anything else -- MPI noise, a
  # missing display, a warning -- says nothing about CUDA.
  if grep -qiE 'was not found by CMake|cusparseSolver was chosen, but' <<<"$out"; then
    return 1
  fi
  if grep -qiE 'unknown value for parameter|AcceleratorMode' <<<"$out"; then
    return 1
  fi
  [[ $linked -eq 1 || $cfg_cuda -eq 1 ]] && return 0
  return 1
}

OPM_ACCEL=""
OPM_SOLVER_JSON=""
OPM_SOLVER_DESC="Flow default (CPU)"
OPM_GPU=0
if [[ -n "${OPM_EXTRA:-}" ]]; then
  OPM_ACCEL="$OPM_EXTRA"
  OPM_SOLVER_DESC="$OPM_EXTRA"
elif opm_has_gpu; then
  OPM_GPU=1
  # Two mechanisms are being handed to Flow at once, and that is worth naming
  # rather than papering over.  --accelerator-mode=cusparse selects the older
  # BDA bridge; the solver JSON selects gpuistl with gpubicgstab and
  # OPMGPUILU0.  Which one Flow ends up using when given both is not something
  # this script determines, and the description string used to assert gpuistl
  # regardless, which is how a paper came to describe one path while a
  # configuration table listed the other.
  #
  # OPM_ACCELERATOR=none drops the flag and leaves the JSON to choose.  The
  # default is unchanged, so no published number moves without being asked to.
  if [[ "${OPM_ACCELERATOR:-cusparse}" == "none" ]]; then
    OPM_ACCEL=""
    OPM_SOLVER_DESC="gpuistl via solver JSON: gpubicgstab + OPMGPUILU0, FP32 factors (GPU)"
  else
    OPM_ACCEL="--accelerator-mode=${OPM_ACCELERATOR:-cusparse}"
    OPM_SOLVER_DESC="--accelerator-mode=${OPM_ACCELERATOR:-cusparse} AND solver JSON (gpubicgstab + OPMGPUILU0); Flow picks between them"
  fi
fi
if [[ $OPM_GPU -eq 1 || -n "$OPM_ACCEL" ]]; then
  echo "OPM GPU solver : $OPM_ACCEL + solver JSON (gpubicgstab/OPMGPUILU0)"
  FLOW_DESC="$GPU_DESC + $CPU_DESC"
else
  echo "OPM GPU solver : rejected by this binary -- no CUDA in this build."
  echo "                 The comparison below is GPU (NVRS) vs CPU (Flow),"
  echo "                 and the report says so.  ./check_opm_gpu.sh gives"
  echo "                 the diagnosis; ./install_opm_gpu.sh rebuilds."
  FLOW_DESC="$CPU_DESC (CPU linear solver)"
fi
echo

# --- Decks -----------------------------------------------------------------
echo "== generating decks =="
# Only the cases about to run: BENCH2P_XL is 600,000 cells and formatting its
# arrays is not free.
# BENCH_GRID overrides the grid of every case named on the command line, so a
# one-off resolution can be run without editing make_decks.py.  The reservoir
# keeps its extent and the cells get smaller, which is what keeps the flood --
# pore volume, wells, injected fraction -- identical to the other cases, so the
# only thing that changed is the size of the linear system.
GRID_ARGS=()
if [[ -n "${BENCH_GRID:-}" ]]; then
  # shellcheck disable=SC2206
  GRID_ARGS=(--grid ${BENCH_GRID})
  echo "grid override  : ${BENCH_GRID}"
fi
# make_decks.py generates the BENCH family and nothing else, and it was handed
# every case named.  Asked for SPE10 it printed "unknown case(s)" and the run
# stopped there, after the deck had already been fetched and converted, with a
# message about a generator rather than about what was missing.
#
# This harness needs two decks per case: $CASE.DATA for NVRS and
# ${CASE}_OPM.DATA for Flow, because the comparison is the point.  A case that
# brings its own deck brings only the first, so it is refused here, early, and
# told what it can be used for instead.
_GEN=() _PRE=()
for _c in "${CASES[@]}"; do
  case "$_c" in
    BENCH*) _GEN+=("$_c") ;;
    *)      _PRE+=("$_c") ;;
  esac
done
if [[ ${#_GEN[@]} -gt 0 ]]; then
  python3 "$HERE/make_decks.py" --out "$HERE" "${GRID_ARGS[@]+${GRID_ARGS[@]}}" "${_GEN[@]}" || exit 1
fi

# A case that brings its own deck brings only the NVRS one.  make_decks.py
# writes a matching ECLIPSE deck for the cases it generates; SPE10 and anything
# supplied by hand do not have one, and never will, because they exist to
# exercise the solver rather than to be history-matched against Flow.
#
# Such a case runs NVRS alone.  This is decided per case, so a list mixing
# BENCH3P and SPE10 runs each of them the way it can, and it is announced,
# because a comparison report with one simulator in it should not be a surprise.
NEED_FLOW=0
for _c in "${CASES[@]}"; do
  if [[ -z "${SKIP_OPM:-}" && -f "$HERE/${_c}_OPM.DATA" ]]; then NEED_FLOW=1; fi
done
for _c in "${_PRE[@]+${_PRE[@]}}"; do
  if [[ ! -f "$HERE/${_c}_OPM.DATA" ]]; then
    echo " note           : $_c has no ECLIPSE deck; running NVRS alone"
  fi
done
if [[ $NEED_FLOW -eq 1 && ! -x "$FLOW_EXE" ]]; then
  echo "ERROR: flow not found.  Run ./install_opm_gpu.sh first," >&2
  echo "       or SKIP_OPM=1 to run NVRS alone." >&2
  exit 1
fi
echo

secs() { python3 -c 'import time; print("%.6f" % time.time())'; }

status=0
for CASE in "${CASES[@]}"; do
  hr
  echo " CASE $CASE"
  hr
  RD="$RUNS/$CASE"
  # --no-opm keeps the Flow run already here (and its timing) for the report;
  # wiping it left a rerun of the NVRS arms with nothing to compare against.
  keep_opm=""
  if [[ -n "${SKIP_OPM:-}" && -d "$RD/opm" ]] && \
     compgen -G "$RD/opm/*.EGRID" >/dev/null 2>&1; then
    keep_opm="$(mktemp -d)"
    mv "$RD/opm" "$keep_opm/opm"
    [[ -f "$RD/timing.json" ]] && cp "$RD/timing.json" "$keep_opm/timing.json"
  fi
  rm -rf "$RD"
  mkdir -p "$RD"
  if [[ -n "$keep_opm" ]]; then
    mv "$keep_opm/opm" "$RD/opm"
    [[ -f "$keep_opm/timing.json" ]] && cp "$keep_opm/timing.json" "$RD/timing.prev.json"
    rm -rf "$keep_opm"
    echo "  keeping the earlier OPM Flow run of $CASE for the report (--no-opm)"
  fi
  mkdir -p "$RD/opm"
  # Whether this case has a Flow side at all.
  CASE_NO_OPM=0
  [[ -n "${SKIP_OPM:-}" ]] && CASE_NO_OPM=1
  [[ -f "$HERE/${CASE}_OPM.DATA" ]] || CASE_NO_OPM=1
  cp "$HERE/$CASE.DATA" "$RD/"
  [[ $CASE_NO_OPM -eq 0 ]] && cp "$HERE/${CASE}_OPM.DATA" "$RD/"
  # The GPU and CPU JSONs carry the same tolerance and iteration cap; only
  # the solver and preconditioner differ, so the two sides stay comparable
  # whether or not this Flow has CUDA.
  if [[ $OPM_GPU -eq 1 ]]; then
    OPM_SOLVER_JSON="$HERE/${CASE}_solver_gpu.json"
  else
    OPM_SOLVER_JSON="$HERE/${CASE}_solver_cpu.json"
  fi
  # Whether Flow can use the GPU depends on the DECK as well as the binary.
  # The BDA/cusparse bridge only handles blocksize 3 -- three-phase black oil.
  # Handed a two-phase deck it prints
  #
  #     Warning: BdaSolver only accepts blocksize = 3 at this time
  #
  # falls back to Dune, and then aborts on an assertion, because the solver
  # JSON asked for gpubicgstab and Dune cannot provide it.  So a two-phase case
  # goes to the CPU solver deliberately rather than crashing.
  # Reads the ECLIPSE deck, and only decides how Flow is launched, so it is
  # skipped along with Flow rather than guarded at every use.
  CASE_THREE=0
  if [[ $CASE_NO_OPM -eq 0 ]]; then
    grep -qE '^[[:space:]]*GAS[[:space:]]*$' "$RD/${CASE}_OPM.DATA" && CASE_THREE=1
  fi

  CASE_ACCEL="" CASE_RANKS="$OPM_RANKS" CASE_SOLVER="$OPM_SOLVER_DESC"
  if [[ $OPM_GPU -eq 1 && $CASE_THREE -eq 1 ]]; then
    CASE_ACCEL="$OPM_ACCEL"
    CASE_JSON="$HERE/${CASE}_solver_gpu.json"
    # OPM's gpuistl path has no distributed GPU solve in 2024.10; the Aramco
    # runs launch it with srun -n 1 for the same reason.
    [[ -z "${OPM_RANKS_FORCED:-}" ]] && CASE_RANKS=1
    CASE_SOLVER="gpuistl: gpubicgstab + OPMGPUILU0, FP32 factors (GPU, 1 rank)"
    echo "  OPM linear solver: GPU (blocksize 3), $CASE_RANKS rank"
  elif [[ $OPM_GPU -eq 1 ]]; then
    CASE_JSON="$HERE/${CASE}_solver_cpu.json"
    CASE_SOLVER="Flow default (CPU; GPU bridge needs blocksize 3)"
    echo "  OPM linear solver: CPU -- $CASE is TWO-PHASE (blocksize 2) and"
    echo "                     Flow's cusparse bridge only accepts blocksize 3."
    echo "                     Run BENCH3P for a GPU-vs-GPU comparison."
  else
    CASE_JSON="$HERE/${CASE}_solver_cpu.json"
  fi
  if [[ -f "$CASE_JSON" ]]; then cp "$CASE_JSON" "$RD/"; else CASE_JSON=""; fi

  # --- OPM Flow ----------------------------------------------------------
  if [[ $CASE_NO_OPM -eq 1 ]]; then
    echo "-- OPM Flow: skipped for this case --"
    opm_rc=0
    opm_s="0"
  else
    echo "-- OPM Flow --"
    t0="$(secs)"
    ( cd "$RD/opm" && \
      "${MPI_RUN[@]}" -np "$CASE_RANKS" \
        "$FLOW_EXE" "$RD/${CASE}_OPM.DATA" \
        --parsing-strictness=low \
        --enable-ecl-output=true \
        --output-dir="$RD/opm" \
        --matrix-add-well-contributions=true \
        ${CASE_JSON:+--linear-solver="$CASE_JSON"} \
        $CASE_ACCEL ) 2>&1 | tee "$LOGS/${CASE}_opm_${RUN_STAMP}.log"
    opm_rc=${PIPESTATUS[0]}
    t1="$(secs)"
    opm_s="$(python3 -c "print('%.3f' % ($t1 - $t0))")"
    echo "-- OPM Flow finished rc=$opm_rc in ${opm_s} s --"
    [[ $opm_rc -ne 0 ]] && why "$LOGS/${CASE}_opm_${RUN_STAMP}.log" "OPM Flow"
  fi

  # --- NVRS, once per solver arm --------------------------------------------
  # Each arm runs the case's deck with its own solver lines (backend_arms.sh)
  # into runs/<CASE>/nvrs_<arm>/, as run_norne.sh does, and timing.json names
  # each.  This ran once before, with no LINSOLVER in the generated deck -- so
  # AmgX, the simulator's default -- into runs/<CASE>/nvrs, which the report
  # labelled "NVRS-CUDA (HYPRE)": BENCH3P then showed no AmgX at all.
  # NVRS_ONLY (set by nvrs.sh) picks the arms; the default is AmgX and HYPRE.
  ARMS="${NVRS_ONLY:-amgx hypre}"
  ARMS="${ARMS//,/ }"
  ARM_SECS=""
  nvrs_rc=0
  for be in $ARMS; do
    arm_known "$be" || { echo "  [skip] unknown arm $be" >&2; continue; }
    UP="$(echo "$be" | tr '[:lower:]' '[:upper:]')"
    ADECK="$RD/${CASE}_${UP}.DATA"
    AOUT="OUTPUT_${CASE}_${UP}"
    sed -E "s|^([[:space:]]*OUTDIR[[:space:]]+).*|\1$AOUT|" "$RD/$CASE.DATA" > "$ADECK.tmp"
    grep -qE '^[[:space:]]*OUTDIR[[:space:]]' "$ADECK.tmp" || printf '\nOUTDIR %s\n' "$AOUT" >> "$ADECK.tmp"
    arm_apply_deck "$ADECK.tmp" "$ADECK" "$be"
    rm -f "$ADECK.tmp"
    echo
    echo "-- NVRS-CUDA ($(arm_label "$be")) --"
    rm -rf "$CUDA_DIR/$AOUT"
    alog="$LOGS/${CASE}_nvrs_${be}_${RUN_STAMP}.log"
    t0="$(secs)"
    ( cd "$CUDA_DIR" && \
      "${MPI_RUN[@]}" -np "$(deck_single_gpu "$ADECK" && echo 1 || echo "$NVRS_RANKS")" \
        "$NVRS_EXE" "$ADECK" ) 2>&1 | tee "$alog"
    arc=${PIPESTATUS[0]}
    t1="$(secs)"
    asecs="$(python3 -c "print('%.3f' % ($t1 - $t0))")"
    echo "-- NVRS-CUDA ($(arm_label "$be")) finished rc=$arc in ${asecs} s --"
    if [[ $arc -ne 0 ]]; then
      why "$alog" "NVRS-CUDA ($be)"
      nvrs_rc=$arc
    fi
    mkdir -p "$RD/nvrs_$be"
    if [[ -d "$CUDA_DIR/$AOUT" ]]; then
      mv "$CUDA_DIR/$AOUT" "$RD/nvrs_$be/"
      ARM_SECS="$ARM_SECS $be=$asecs"
    else
      echo "WARNING: $CUDA_DIR/$AOUT not found; the $be run produced no output" >&2
    fi
  done

  python3 - "$RD/timing.json" "$opm_s" "$OPM_RANKS" "$NVRS_RANKS" \
           "$FLOW_DESC" "$GPU_DESC" "$LOGS/${CASE}_opm_${RUN_STAMP}.log" \
           "$CASE_SOLVER" "$CASE_RANKS" "$RD" "$ARM_SECS" <<'PY'
import glob, json, os, re, sys
(out, opm_s, _opm_r, nvrs_r, flow_hw, gpu_hw, opm_log, opm_solver, opm_r,
 rd, arm_secs) = sys.argv[1:12]
SECS = dict(kv.split("=", 1) for kv in arm_secs.split() if "=" in kv)


def opm_breakdown(path):
    """Flow prints its own phase timings in the end-of-simulation block.

    Parsing them is the only way to say how much of Flow's run was the linear
    solve; re-timing it from outside can only give the total.
    """
    want = {"Setup time": "setup", "Assembly time": "assembly",
            "Linear solve time": "linear", "Linear setup": "linear_setup",
            "Props/update time": "props", "Pre/post step": "prepost",
            "Output write time": "output", "Simulation time": "simulation"}
    got = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                for label, key in want.items():
                    m = re.match(r"\s*%s:\s+([0-9.]+)\s*s" % re.escape(label),
                                 line)
                    if m:
                        got[key] = float(m.group(1))
    except OSError:
        pass
    return got


def nvrs_breakdown(arm):
    hits = sorted(glob.glob(os.path.join(rd, "nvrs_" + arm, "**", "meta.json"),
                            recursive=True))
    if not hits:
        return {}
    try:
        with open(hits[0], encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for src, dst in (("assembly_seconds", "assembly"),
                     ("linear_seconds", "linear"),
                     ("wall_seconds", "simulation"),
                     ("amg_setup_seconds", "amg_setup"),
                     ("amg_solve_seconds", "amg_solve"),
                     ("stage2_seconds", "stage2")):
        if isinstance(m.get(src), (int, float)):
            out[dst] = float(m[src])
    for src in ("amg_setups", "amg_applications", "amg_cycles",
                "amg_reuses", "amg_reuse_aborts"):
        if isinstance(m.get(src), (int, float)):
            out[src] = m[src]
    return out


DESC = {
    "amgx": "CPR: FP32 AmgX AMG + block-Jacobi, FP64 FGMRES (GPU)",
    "hypre": "CPR: FP32 HYPRE BoomerAMG + block-Jacobi, FP64 FGMRES (GPU)",
    "ginkgo": "CPR: FP16 Ginkgo PGM AMG + block-Jacobi, FP64 FGMRES (GPU)",
    "ginkgo32": "CPR: FP32 Ginkgo PGM AMG + block-Jacobi, FP64 FGMRES (GPU)",
    "ginkgo_abj": "CPR: FP16 Ginkgo PGM AMG + adaptive block-Jacobi (GPU)",
    "amgx_fp64": "CPR: FP32 AmgX AMG + block-Jacobi, native FP64 operator (GPU)",
}
t = {}
if opm_s and float(opm_s) > 0:
    t["opm"] = {"seconds": float(opm_s), "ranks": int(opm_r),
                "hardware": flow_hw, "solver": opm_solver,
                "breakdown": opm_breakdown(opm_log)}
else:
    # Flow not run this time: the earlier Flow entry, kept with its output.
    prev = os.path.join(rd, "timing.prev.json")
    if os.path.exists(prev):
        try:
            old = json.load(open(prev))
            if "opm" in old:
                t["opm"] = old["opm"]
        except Exception:
            pass
for arm, secs in SECS.items():
    t["nvrs_" + arm] = {"seconds": float(secs),
                        "ranks": 1 if arm.startswith("ginkgo") else int(nvrs_r),
                        "hardware": gpu_hw, "solver": DESC.get(arm, arm),
                        "breakdown": nvrs_breakdown(arm)}
json.dump(t, open(out, "w"), indent=2)
print("  wrote %s: %s" % (out, ", ".join(sorted(t))))
PY


  # --- NVRS strong scaling ---------------------------------------------------
  # Run immediately after the head-to-head, on the same deck, while the machine
  # is in the same state.  One GPU against one GPU answers "is it faster than
  # Flow"; this answers "does it use the hardware it is given", which is the
  # other half of the claim and cannot be read off a single-GPU number.
  #
  # Rank counts are powers of two up to the GPU count, plus the GPU count
  # itself when that is not a power of two.  NVRS decomposes the grid into
  # K-slabs, so a rank can only be given work if a layer remains for it; the
  # list is capped at nz, because beyond that a rank owns nothing and the
  # timing says more about MPI than about the simulator.
  if [[ "${NVRS_SCALING:-1}" == "1" && $nvrs_rc -eq 0 ]] && deck_single_gpu "$RD/$CASE.DATA"; then
    echo "  strong scaling skipped: the deck selects Ginkgo, which runs on one GPU"
  elif [[ "${NVRS_SCALING:-1}" == "1" && $nvrs_rc -eq 0 ]]; then
    NZ="$(awk '$1=="NZ"{print $2; exit}' "$RD/$CASE.DATA")"
    NGPU_MAX="${NVRS_MAX_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
    [[ -z "$NGPU_MAX" || "$NGPU_MAX" -lt 1 ]] && NGPU_MAX=1
    if [[ -n "$NZ" && "$NZ" -lt "$NGPU_MAX" ]]; then
      echo "  capping scaling at nz=$NZ (K-slab decomposition)"
      NGPU_MAX="$NZ"
    fi

    RANKS=()
    n=1
    while [[ $n -le $NGPU_MAX ]]; do
      RANKS+=("$n")
      n=$((n * 2))
    done
    if [[ "${RANKS[${#RANKS[@]}-1]}" != "$NGPU_MAX" ]]; then
      RANKS+=("$NGPU_MAX")
    fi

    if [[ ${#RANKS[@]} -gt 1 ]]; then
      hr
      echo " NVRS-CUDA STRONG SCALING - $CASE on ${RANKS[*]} GPU(s)"
      hr
      mkdir -p "$RD/scaling"
      SCALE_TSV="$RD/scaling/points.tsv"
      : > "$SCALE_TSV"
      for n in "${RANKS[@]}"; do
        echo
        echo "-- NVRS-CUDA on $n GPU(s) --"
        rm -rf "$CUDA_DIR/OUTPUT_$CASE"
        t0="$(secs)"
        ( cd "$CUDA_DIR" && "${MPI_RUN[@]}" -np "$n" "$(nvrs_pick_exe "$NVRS_EXE" "$RD/$CASE.DATA")" "$RD/$CASE.DATA" ) 2>&1 | tee "$LOGS/${CASE}_nvrs_${n}gpu_${RUN_STAMP}.log"
        srank_rc=${PIPESTATUS[0]}
        t1="$(secs)"
        el="$(python3 -c "print('%.3f' % ($t1 - $t0))")"
        echo "-- $n GPU(s): rc=$srank_rc in ${el} s --"
        if [[ $srank_rc -eq 0 && -d "$CUDA_DIR/OUTPUT_$CASE" ]]; then
          mkdir -p "$RD/scaling/${n}gpu"
          mv "$CUDA_DIR/OUTPUT_$CASE" "$RD/scaling/${n}gpu/"
          printf '%s\t%s\n' "$n" "$el" >> "$SCALE_TSV"
        else
          echo "WARNING: NVRS failed on $n GPU(s); that point is dropped" >&2
          why "$LOGS/${CASE}_nvrs_${n}gpu_${RUN_STAMP}.log" "NVRS on $n GPU(s)"
        fi
      done

      python3 - "$SCALE_TSV" "$RD/scaling.json" "$RD/scaling" <<'PYSCALE'
import json, os, sys
tsv, out, root = sys.argv[1:4]
rows = []
with open(tsv) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        n, sec = line.split()
        rows.append({"ranks": int(n), "seconds": float(sec),
                     "outdir": os.path.join(root, "%sgpu" % n)})
rows.sort(key=lambda r: r["ranks"])
json.dump({"nvrs": rows}, open(out, "w"), indent=2)
if rows:
    base = rows[0]
    print("[scale] ranks  seconds  speed-up  efficiency")
    for r in rows:
        sp = base["seconds"] / r["seconds"] if r["seconds"] > 0 else float("nan")
        eff = 100.0 * sp / (r["ranks"] / base["ranks"])
        print("[scale] %5d  %7.2f  %7.2fx  %8.1f%%"
              % (r["ranks"], r["seconds"], sp, eff))
PYSCALE

      # The sweep above runs the case's own deck (AmgX).  NVRS_SCALING_ARMS names
      # further arms to sweep on the same rank counts.  Each lands in
      # scaling_<arm>/ and scaling_<arm>.json.
      SARMS="${NVRS_SCALING_ARMS:-}"
      for a in ${SARMS//,/ }; do
        [[ "$a" == ginkgo* ]] && continue      # one GPU only
        ADECK="$RD/${CASE}_scale_${a}.DATA"
        arm_apply_deck "$RD/$CASE.DATA" "$ADECK" "$a"
        hr
        echo " NVRS-CUDA STRONG SCALING - $CASE, $(arm_label "$a"), on ${RANKS[*]} GPU(s)"
        hr
        mkdir -p "$RD/scaling_$a"
        STSV="$RD/scaling_$a/points.tsv"
        : > "$STSV"
        for n in "${RANKS[@]}"; do
          echo
          echo "-- NVRS-CUDA ($(arm_label "$a")) on $n GPU(s) --"
          rm -rf "$CUDA_DIR/OUTPUT_$CASE"
          t0="$(secs)"
          ( cd "$CUDA_DIR" && "${MPI_RUN[@]}" -np "$n" "$(nvrs_pick_exe "$NVRS_EXE" "$ADECK")" "$ADECK" ) 2>&1 \
            | tee "$LOGS/${CASE}_nvrs_${a}_${n}gpu_${RUN_STAMP}.log"
          srank_rc=${PIPESTATUS[0]}
          t1="$(secs)"
          el="$(python3 -c "print('%.3f' % ($t1 - $t0))")"
          echo "-- $(arm_label "$a"), $n GPU(s): rc=$srank_rc in ${el} s --"
          if [[ $srank_rc -eq 0 && -d "$CUDA_DIR/OUTPUT_$CASE" ]]; then
            rm -rf "$RD/scaling_$a/${n}gpu"; mkdir -p "$RD/scaling_$a/${n}gpu"
            mv "$CUDA_DIR/OUTPUT_$CASE" "$RD/scaling_$a/${n}gpu/"
            printf '%s\t%s\n' "$n" "$el" >> "$STSV"
          else
            echo "WARNING: $(arm_label "$a") failed on $n GPU(s); that point is dropped" >&2
          fi
        done
        echo "[scale] $(arm_label "$a")"
        python3 - "$STSV" "$RD/scaling_$a.json" "$RD/scaling_$a" <<'PYSCALE'
import json, os, sys
tsv, out, root = sys.argv[1:4]
rows = []
with open(tsv) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        n, sec = line.split()
        rows.append({"ranks": int(n), "seconds": float(sec),
                     "outdir": os.path.join(root, "%sgpu" % n)})
rows.sort(key=lambda r: r["ranks"])
json.dump({"nvrs": rows}, open(out, "w"), indent=2)
if rows:
    base = rows[0]
    print("[scale] ranks  seconds  speed-up  efficiency")
    for r in rows:
        sp = base["seconds"] / r["seconds"] if r["seconds"] > 0 else float("nan")
        eff = 100.0 * sp / (r["ranks"] / base["ranks"])
        print("[scale] %5d  %7.2f  %7.2fx  %8.1f%%"
              % (r["ranks"], r["seconds"], sp, eff))
PYSCALE
      done
    fi
  fi

  if [[ $opm_rc -ne 0 || $nvrs_rc -ne 0 ]]; then
    echo "WARNING: $CASE did not complete cleanly (opm=$opm_rc nvrs=$nvrs_rc);" >&2
    echo "         the report will cover whatever report steps both wrote." >&2
    status=1
  fi

  # Optional profile of this case.  OFF unless asked for, because it re-runs
  # the deck: nsys adds about 1.2x, which is affordable, and ncu replays every
  # kernel it profiles and is roughly 100x slower per kernel, which is not.
  #
  #   NVRS_PROFILE=1     nsys only, the timeline
  #   NVRS_PROFILE=both  add Nsight Compute's kernel counters
  #   NVRS_PROFILE=ncu   counters only
  if [[ -n "${NVRS_PROFILE:-}" ]]; then
    pmode="$NVRS_PROFILE"; [[ "$pmode" == "1" ]] && pmode=nsys
    "$HERE/profile_run.sh" "$RD/$CASE.DATA" "$RD/profile" "$CASE" "$pmode" ||       echo "  [profile] $CASE profile did not complete; the run itself is unaffected" >&2
  fi
  echo
done

# --- Report ----------------------------------------------------------------
hr
echo " building the comparison report"
hr
if [[ -n "${NVRS_PROFILE:-}" ]]; then
  for pd in "$RUNS"/*/profile; do
    [[ -d "$pd" ]] && python3 "$HERE/profile_plots.py" --dir "$pd" || true
  done
fi
# nvrs.sh runs the report itself, after profiling, so that the Nsight
# figures reach the page.  Building it here as well means building it
# twice: on BENCH3P that was 113 seconds each time, for one page that the
# second run overwrites.  Standalone, this is still the right place for it.
if [[ -z "${NVRS_SKIP_REPORT:-}" ]]; then
  python3 "$HERE/compare.py" --runs "$RUNS" 2>&1 | tee "$LOGS/compare_${RUN_STAMP}.log"
else
  echo " report         : skipped here; nvrs.sh builds it after profiling"
fi

echo
echo "Reports: $RUNS/compare_<CASE>.html"
echo "Logs   : $LOGS"
echo "  transcript   run_${RUN_STAMP}.log"
echo "  per phase    <CASE>_opm_${RUN_STAMP}.log, <CASE>_nvrs_${RUN_STAMP}.log"
exit $status
