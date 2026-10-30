#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# profile_run.sh -- Nsight Systems and Nsight Compute on ANY deck.
#
# The case-independent half of the profiling.  profile_case.sh wraps this for
# Norne; run_comparison.sh, run_norne.sh, backend_study.sh and run_spe10.sh all
# call it when NVRS_PROFILE=1, so one implementation serves every benchmark and
# there is no second copy to drift.
#
#   ./profile_run.sh <deck> <outdir> [label] [nsys|ncu|both]
#
# The mode defaults to nsys, and that default is the point.  nsys costs about
# 1.2x runtime and can ride along with any benchmark; ncu costs roughly 100x
# per profiled kernel and must be asked for.
#
# Two tools, two questions, and they are not interchangeable:
#
#   nsys  WHERE THE TIME GOES.  Traces the whole run at low overhead, so the
#         wall-clock split the run claims for itself can be checked from
#         outside it.  Costs roughly 1.2x runtime.
#
#   ncu   WHY A KERNEL IS SLOW.  Replays kernels with the hardware counters on
#         and places each one against the roofline.  Roughly 100x slower PER
#         PROFILED KERNEL, because it serialises and replays them -- which is
#         why it is pointed at a bounded number of launches and never at a
#         whole run, and why NVRS_PROFILE is opt-in rather than the default.
#
# Environment:
#   NCU_SKIP=200      launches to skip before counting (past cold start-up)
#   NCU_COUNT=120     launches to profile; the cost knob that matters
#   PROF_TEND=<days>  shorten the deck to this TEND first, for a long history
#   NVRS_RANKS=1      ranks; nsys traces each, ncu profiles rank 0 only
set -uo pipefail

DECK="${1:-}"
OUT="${2:-}"
LABEL="${3:-run}"
MODE="${4:-nsys}"

if [[ -z "$DECK" || -z "$OUT" ]]; then
  sed -n '5,30p' "${BASH_SOURCE[0]}" | sed 's|^# \{0,1\}||'
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
NVRS_EXE="${NVRS_EXE:-$CUDA_DIR/nvrs_cuda}"
[[ -x "$NVRS_EXE" ]] || NVRS_EXE="$CUDA_DIR/build/nvrs_cuda"
NVRS_RANKS="${NVRS_RANKS:-1}"
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
# A profiled run prints nothing for minutes and is indistinguishable from a
# hang, which is worth fixing twice over: once by letting the program's own
# output through (see the tee below), and once here, because the phase after
# the run has no output of its own to let through.  nsys stats builds a
# database in silence and can take as long as the run did.
#
# One line every NVRS_HEARTBEAT seconds, carrying elapsed time and what the GPU
# is doing.  A GPU at 90% is working; a GPU at 0% for minutes is the thing
# worth interrupting.  NVRS_HEARTBEAT=0 turns it off.
#
# It writes to stderr so that it stays on the terminal and out of the log being
# tee'd from stdout, which is parsed elsewhere.
HEARTBEAT_PID=""
heartbeat_start() {
  local every="${NVRS_HEARTBEAT:-20}"
  [[ "$every" -gt 0 ]] 2>/dev/null || return 0
  local what="${1:-running}"
  (
    t=0
    while sleep "$every"; do
      t=$((t + every))
      gpu=""
      if command -v nvidia-smi >/dev/null 2>&1; then
        # Labelled, because memory.used is the whole device: the CUDA
        # context, the library workspaces and every other process on it, not
        # this run's arrays.  Unlabelled it reads as the model's footprint and
        # invites the wrong conclusion.
        gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used                --format=csv,noheader 2>/dev/null | head -1)"
        [[ -n "$gpu" ]] && gpu="util ${gpu%%,*}, device mem ${gpu#*, } in use"
      fi
      printf '  [profile] %s ... %4ds   %s
' "$what" "$t" "${gpu:-}" >&2
    done
  ) &
  HEARTBEAT_PID=$!
}
heartbeat_stop() {
  [[ -n "${HEARTBEAT_PID:-}" ]] || return 0
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  HEARTBEAT_PID=""
}
# Ctrl-C must not leave a heartbeat printing into the next command's prompt.
trap 'heartbeat_stop' EXIT INT TERM

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


# nvrs_cuda is launched under MPI, always, exactly as run_norne.sh and
# run_comparison.sh launch it.  This is not a preference.  A bare launch does
# not merely fail: MPI_Init as a singleton under Slurm's PMIx takes the entire
# job step down with
#
#   pmixp_client_v2.c:225: Error handler invoked: status = -61
#   *** STEP ... CANCELLED ... DUE to SIGNAL Killed ***
#
# run_norne.sh:196 documents this and routes even `flow --help` through mpirun
# for the same reason.  This script did not, and so profiled a process that
# exited after 1.9 s without ever reaching the GPU -- producing a valid, empty
# trace and a report crediting Nsight for a figure Nsight never saw.
#
# NVRS_LAUNCH overrides the whole thing, taken verbatim and word-split:
#
#   NVRS_LAUNCH="srun --mpi=pmix -n 1 --cpu-bind=none" ./profile_case.sh SPE1
#
# Needed because the launcher is not always mpirun.  Under Slurm with Pyxis the
# working arrangement is srun with PMIx, which spells its rank count -n rather
# than -np and takes options mpirun has never heard of, so neither MPIRUN nor
# an appended -np can express it.  This is the launcher the Aramco RFP profiles
# were taken with, on this same machine, so it needs to be reachable here.
#
# The topology is the same either way: launcher outside, nsys inside, one
# report per rank.
if [[ -n "${NVRS_LAUNCH:-}" ]]; then
  # Deliberately unquoted: the point is to split a command line into words.
  # shellcheck disable=SC2206
  MPI_LAUNCH=(${NVRS_LAUNCH})
  echo " launcher       : $NVRS_LAUNCH (from NVRS_LAUNCH)"
else
  mpi_launcher
  MPI_LAUNCH=("${MPI_RUN[@]}" -np "$NVRS_RANKS")
fi

# The benchmarks run from $CUDA_DIR, and the deck's relative paths (OUTDIR
# above all) resolve against the working directory, so match them.  Both paths
# handed to the binary are made absolute first, since we are about to move.
abspath() { case "$1" in /*) printf '%s\n' "$1" ;;
                         *) printf '%s/%s\n' "$(pwd)" "$1" ;; esac; }

mkdir -p "$OUT"
OUT="$(abspath "$OUT")"
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

DECK="$(abspath "$DECK")"
have() { command -v "$1" >/dev/null 2>&1; }
# --- locating the profilers --------------------------------------------------
# Nsight Systems and Nsight Compute are routinely installed without being on
# PATH.  They live under /opt/nvidia/nsight-systems/<version>/target-linux-<arch>
# and /opt/nvidia/nsight-compute/<version>, and the CUDA toolkit carries copies
# in its own bin.  Testing PATH alone skipped the profile on machines that had
# the tool installed, and reported it as missing.
#
# The architecture directory differs: target-linux-sbsa-armv8 on Grace and
# Vera, target-linux-x64 elsewhere, so both are globbed rather than chosen.
#
#   NSYS=/path/to/nsys, NCU=/path/to/ncu   to name them explicitly
find_tool() {
  local name="$1" c
  local override
  case "$name" in
    nsys) override="${NSYS:-}" ;;
    ncu)  override="${NCU:-}" ;;
    *)    override="" ;;
  esac
  if [[ -n "$override" && -x "$override" ]]; then printf '%s\n' "$override"; return 0; fi
  c="$(command -v "$name" 2>/dev/null || true)"
  if [[ -n "$c" ]]; then printf '%s\n' "$c"; return 0; fi
  shopt -s nullglob
  local cands=(
    "${CUDA_HOME:-/usr/local/cuda}/bin/$name"
    /usr/local/cuda/bin/"$name"
    /usr/local/cuda-*/bin/"$name"
    /opt/nvidia/nsight-systems/*/target-linux-*/"$name"
    /opt/nvidia/nsight-systems-cli/*/target-linux-*/"$name"
    /opt/nvidia/nsight-compute/*/"$name"
    /opt/nvidia/hpc_sdk/*/*/profilers/Nsight_Systems/target-linux-*/"$name"
    /opt/nvidia/hpc_sdk/*/*/profilers/Nsight_Compute/"$name"
  )
  shopt -u nullglob
  local f
  for f in "${cands[@]}"; do
    [[ -x "$f" ]] && { printf '%s\n' "$f"; return 0; }
  done
  return 1
}

NSYS_BIN="$(find_tool nsys || true)"

# What nsys is asked to trace.
#
# These are not chosen from first principles.  They are the options from the
# Aramco RFP profiling script, which produces working reports on this machine,
# with this HPC SDK, under mpirun, one nsys per rank:
#
#     nsys profile --trace=cuda,nvtx --cuda-memory-usage=true #                  --wait=all --mpi-impl=openmpi ...
#
# Copied rather than reasoned about, because a known-good set beats a theory.
# Two earlier theories about which option was fatal here -- allocation tracking,
# and device-side event completion tracing -- are both contradicted by that
# script: it enables the first explicitly, leaves the second at its default,
# and works.
#
# What this file used to have and that one does not is --sample=cpu, and
# --trace=osrt,mpi.  All three inject into the process rather than observing
# the GPU, none of them is read by anything downstream -- the figures come from
# cuda_gpu_kern_sum, cuda_gpu_mem_time_sum and cuda_api_sum -- so all three are
# gone.
#
# --wait=all makes nsys wait for descendants before finalising, which matters
# when the thing being profiled is launched under a launcher.
#
# --mpi-impl is only passed when the launcher was detected as Open MPI, since
# naming the wrong implementation is worse than naming none.
#
# NSYS_OPTS replaces the lot, and setting it suppresses the fallback below:
# someone narrowing a profiler problem by hand wants the options they chose.
NSYS_OPTS_SET="${NSYS_OPTS:+yes}"
_nsys_help=""
[[ -n "$NSYS_BIN" ]] && _nsys_help="$("$NSYS_BIN" profile --help 2>/dev/null || true)"
_nsys_has() { [[ "$_nsys_help" == *"$1"* ]]; }

_wait=""
_nsys_has "--wait" && _wait=" --wait=all"

# Open MPI is what mpi_launcher detected if it added --oversubscribe, which is
# an Open MPI-only flag.  An overridden launcher says nothing either way, so
# nothing is claimed.
_mpiimpl=""
if _nsys_has "--mpi-impl" && [[ " ${MPI_LAUNCH[*]} " == *" --oversubscribe "* ]]; then
  _mpiimpl=" --mpi-impl=openmpi"
fi

NSYS_OPTS="${NSYS_OPTS:---trace=cuda,nvtx --cuda-memory-usage=true$_wait$_mpiimpl}"

# The fallback, tried once if the above yields no kernels: one trace domain, no
# allocation hooks, and device-side event completion tracing off.  That last is
# the option nsys warns about itself, in the first lines it prints.  It is down
# here rather than in the default precisely because the working script does not
# use it, so it belongs in the defensive set and not the proven one.
_evt=""
_nsys_has "--cuda-event-trace" && _evt=" --cuda-event-trace=false"
NSYS_OPTS_MIN="--trace=cuda$_wait$_mpiimpl$_evt"

NCU_BIN="$(find_tool ncu || true)"

# --clock-control none, copied from the Aramco RFP script, which profiles
# successfully on this machine.  By default ncu pins the GPU to base clocks so
# that repeated passes are comparable, and pinning clocks needs a privilege a
# container generally does not have.  Refusing to profile at all is the worse
# outcome, so the pinning is dropped.
#
# The cost is that absolute rates now depend on where the clocks happened to
# be.  Nothing here quotes an absolute rate: the counters place kernels against
# a roofline, which is a ratio, and the wall-clock figures come from unprofiled
# runs.  Set NCU_CLOCK= to restore the default on a machine that permits it.
NCU_CLOCK="${NCU_CLOCK---clock-control none}"

if [[ ! -x "$NVRS_EXE" ]]; then
  echo "  [profile] no nvrs_cuda; skipping" >&2; exit 1
fi
if [[ ! -f "$DECK" ]]; then
  echo "  [profile] no deck at $DECK; skipping" >&2; exit 1
fi

# Optionally shorten the deck.  A 247-step history under ncu is days, and under
# nsys it is a trace the GUI will not open.  The kernel MIX does not change over
# a history -- the same assembly, V-cycle and SpMV run at every step -- so a
# couple of dozen steps carries the same mix.  What it does NOT carry is the
# wall-clock total, which is why nothing downstream quotes a profiled runtime.
#
# The copy is made whether or not the deck is being shortened.  The profiled
# deck also has to turn AUTOPLOT off, and that must not depend on whether
# PROF_TEND happened to be set: without the copy there is nowhere to put it,
# and the run then drags a Python process through the profiler again.
RUNDECK="$OUT/$(basename "${DECK%.DATA}")_PROF.DATA"
sed -e "s|^OUTDIR .*|OUTDIR   OUTPUT_PROFILE_${LABEL}|" "$DECK" > "$RUNDECK"
if [[ -n "${PROF_TEND:-}" ]] && grep -q '^TEND' "$RUNDECK"; then
  sed -i "s/^TEND .*/TEND      $PROF_TEND/" "$RUNDECK"
fi
# FIELDDECK is resolved relative to the deck's own directory.
fd="$(sed -n 's/^[[:space:]]*FIELDDECK[[:space:]]\+//p' "$DECK" | head -1 | tr -d '\r')"
if [[ -n "$fd" ]]; then
  case "$fd" in
    /*) abs="$fd" ;;
    *)  abs="$CUDA_DIR/$fd" ;;
  esac
  sed -i "s|^FIELDDECK .*|FIELDDECK  $abs|" "$RUNDECK"
fi
# AUTOPLOT off, which matters more here than it looks.
#
# At the end of a run the simulator shells out to plot_results.py.  Under a
# profiler that Python process is a traced child, and --wait=all means nsys
# will not finalise until it exits, so the profile is held open by matplotlib
# long after the GPU has gone idle -- on Norne, a 3.8 s simulation followed by
# minutes of drawing.  It reads as a hang, and interrupting it aborts nsys
# before it flushes, which is how a run that reached the GPU thousands of
# times produces a report containing no kernels at all.
#
# The plots are not wanted from a profiled run in any case.  It uses a
# shortened deck, so its curves are of a simulation nobody is asking about,
# and the figures that reach the report come from profile_plots.py.
sed -i "/^AUTOPLOT/d" "$RUNDECK"
echo "AUTOPLOT 0" >> "$RUNDECK"

# PROF_PLOTS=1 draws the figures afterwards, with the profiler gone.
#
# This is the answer to wanting the pictures without wanting them traced.  The
# simulator no longer shells out to Python while nsys is attached; the same
# script runs here instead, on the same output directory, unprofiled.
#
# Off by default because a profiled deck is a shortened one, so its curves are
# of a simulation nobody asked about.  The field images and the animation in
# the report come from the full run_norne.sh runs, which are not profiled and
# do plot; only the four Nsight figures come from here.
render_plots() {
  [[ "${PROF_PLOTS:-0}" == "1" ]] || return 0
  local script="$CUDA_DIR/scripts/plot_results.py"
  local odir="$CUDA_DIR/OUTPUT_PROFILE_${LABEL}"
  [[ -f "$script" && -d "$odir" ]] || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  echo "  [profile] rendering plots (outside the profiler)"
  heartbeat_start "plotting"
  ( cd "$CUDA_DIR" && python3 "$script" "$odir" ) >"$OUT/plots.log" 2>&1 || echo "  [profile] plotting failed; see $OUT/plots.log" >&2
  heartbeat_stop
}

echo "  [profile] $LABEL: $(basename "$RUNDECK") -> $OUT"

rc=0
TRACE_STRIDE=1

run_nsys() {
  if [[ -z "$NSYS_BIN" ]]; then
    echo "  [profile] Nsight Systems not found; skipping the timeline." >&2
    echo "  [profile] Looked on PATH, in \$CUDA_HOME/bin, /usr/local/cuda*/bin," >&2
    echo "  [profile] /opt/nvidia/nsight-systems*/ and the HPC SDK." >&2
    echo "  [profile] Set NSYS=/path/to/nsys if it is somewhere else." >&2
    return 1
  fi
  echo "  [profile] nsys: $NSYS_BIN"
  # mpirun on the OUTSIDE, nsys on the inside: one report per rank, which is
  # what nsys documents for MPI and the only arrangement that gives each rank
  # its own trace instead of one process's view of the launcher.  %q{} is
  # expanded by nsys from the environment, so the rank suffix appears only when
  # there is more than one rank and single-rank output keeps its plain name.
  local obase="$OUT/$LABEL"
  if [[ "$NVRS_RANKS" -gt 1 ]]; then
    obase="$OUT/${LABEL}_rank%q{OMPI_COMM_WORLD_RANK}"
  fi

  # Two attempts at most, the second stripped to the minimum.  A profiler that
  # cannot attach fails during injection, which is to say almost immediately,
  # so a second attempt costs seconds; the first failure of this kind cost
  # several full-length runs to diagnose instead, because the script reported
  # it as a broken program rather than as a profiler that could not attach.
  local -a attempts=("$NSYS_OPTS")
  [[ -n "$NSYS_OPTS_SET" ]] || attempts+=("$NSYS_OPTS_MIN")

  local opts attempt=0 app_rc=0 rep=""
  for opts in "${attempts[@]}"; do
    attempt=$((attempt + 1))
    if [[ $attempt -gt 1 ]]; then
      echo "  [profile] no kernels with those options; retrying at the" >&2
      echo "  [profile] minimum: $opts" >&2
    fi
    local t0=$SECONDS
    heartbeat_start "profiling"
    ( cd "$CUDA_DIR" && "${MPI_LAUNCH[@]}"         "$NSYS_BIN" profile           $opts           --force-overwrite=true           --output="$obase"           "$NVRS_EXE" "$RUNDECK" ) 2>&1 | tee "$OUT/nsys_run.log"
    # PIPESTATUS, not $?, which would be tee's.  The run's own status is the
    # one that matters and tee always succeeds.
    app_rc="${PIPESTATUS[0]}"
    heartbeat_stop
    echo "  [profile] profiled run: $((SECONDS - t0))s"
    # With one rank this is $LABEL.nsys-rep; with several, rank 0's.
    rep="$OUT/$LABEL.nsys-rep"
    [[ -f "$rep" ]] || rep="$(ls -1 "$OUT/${LABEL}"*.nsys-rep 2>/dev/null | head -1)"
    rep="${rep:-$OUT/$LABEL.nsys-rep}"
    if [[ ! -f "$rep" ]]; then
      echo "  [profile] nsys produced no report; see $OUT/nsys_run.log" >&2
      continue
    fi
    if [[ $app_rc -ne 0 ]]; then
      echo "  [profile] the profiled run exited $app_rc -- see $OUT/nsys_run.log" >&2
    fi
    # Stale summaries from a previous attempt would otherwise be read as this
    # attempt's, and the check below would stop on them.
    rm -f "$OUT/nsys_gpukernsum.csv" "$OUT/nsys_gpumemtimesum.csv"           "$OUT/nsys_cudaapisum.csv" "$OUT/nsys_gputrace.csv"
    # Announced because it is the phase that looks like a hang.  `nsys stats`
    # builds a SQLite database out of the report before it can answer anything,
    # prints nothing at all while doing so, and on a long trace takes as long as
    # the run did.  A line here is the difference between waiting and wondering
    # whether to interrupt.
    local t1=$SECONDS
    heartbeat_start "summarising"
    echo "  [profile] summarising the trace (nsys builds a database first; no"
    echo "  [profile] output until it finishes, and it can take a while)"
    # Report names moved between nsys versions; try the modern one, fall back
    # to the old, and carry on rather than failing a benchmark over a rename.
    for pair in "cuda_gpu_kern_sum:gpukernsum"                 "cuda_gpu_mem_time_sum:gpumemtimesum"                 "cuda_api_sum:cudaapisum"; do
      local new="${pair%%:*}" old="${pair##*:}"
      "$NSYS_BIN" stats --report "$new" --format csv --output "$OUT/nsys_${old}" "$rep" >/dev/null 2>&1         || "$NSYS_BIN" stats --report "$old" --format csv --output "$OUT/nsys_${old}" "$rep" >/dev/null 2>&1         || true
    done
    for f in "$OUT"/nsys_*_*.csv; do
      [[ -f "$f" ]] || continue
      case "$(basename "$f")" in
        nsys_gpukernsum*)    mv -f "$f" "$OUT/nsys_gpukernsum.csv" ;;
        nsys_gpumemtimesum*) mv -f "$f" "$OUT/nsys_gpumemtimesum.csv" ;;
        nsys_cudaapisum*)    mv -f "$f" "$OUT/nsys_cudaapisum.csv" ;;
      esac
    done
    # The per-launch trace, sampled rather than cut short.
    #
    # This is one row per kernel launch, not a summary, and a long run has
    # millions.  The obvious bound is to stop after N rows, and that is the
    # wrong operation: launches are spread over the whole run, so truncating
    # keeps the opening minutes and throws away the rest, then draws the result
    # as though it were the history.
    #
    # Every Nth launch covers the whole run at lower resolution instead, which
    # is what a timeline binned into a thousand columns wants.  N is derived
    # from the kernel summary written just above -- its Instances column counts
    # every launch -- so no row count has to be guessed.  The head at the end is
    # a backstop for when it cannot be derived, not the mechanism.
    if [[ "${PROF_TIMELINE:-1}" == "1" ]]; then
      rm -f "$OUT/nsys_gputrace.csv"
      TRACE_STRIDE=1
      if command -v python3 >/dev/null 2>&1 && [[ -s "$OUT/nsys_gpukernsum.csv" ]]; then
        TRACE_STRIDE="$(python3 "$HERE/profile_stride.py" "$OUT/nsys_gpukernsum.csv" "${PROF_TRACE_ROWS:-400000}" 2>/dev/null || echo 1)"
      fi
      [[ "$TRACE_STRIDE" =~ ^[0-9]+$ ]] || TRACE_STRIDE=1
      # The first lines are a report title and the header, which are kept
      # whatever the stride; the sampling starts after them.
    # The header is kept by finding it, not by counting lines.  nsys prints a
    # report title, a blank, and a two-line NOTICE when a .sqlite already
    # exists, which puts the header on line 5; sampling from line 4 dropped it,
    # every row was then read against a header of None, and the timeline came
    # out empty from a 107 MB file that was written correctly.
      ( "$NSYS_BIN" stats --report cuda_gpu_trace --format csv "$rep" 2>/dev/null || "$NSYS_BIN" stats --report gputrace --format csv "$rep" 2>/dev/null ) | awk -v s="$TRACE_STRIDE" 'h==0 {print; if ($0 ~ /Start/ && $0 ~ /Duration/) {h=1; n=0}; next} {n++; if (n % s == 0) print}' | head -n "$(( ${PROF_TRACE_ROWS:-400000} + 16 ))" > "$OUT/nsys_gputrace.csv" || true
      if [[ -s "$OUT/nsys_gputrace.csv" ]]; then
        if [[ "$TRACE_STRIDE" -gt 1 ]]; then
          echo "  [profile] trace: $(wc -l < "$OUT/nsys_gputrace.csv") rows, every ${TRACE_STRIDE}th launch ($(du -h "$OUT/nsys_gputrace.csv" | cut -f1))"
        else
          echo "  [profile] trace: $(wc -l < "$OUT/nsys_gputrace.csv") rows, every launch ($(du -h "$OUT/nsys_gputrace.csv" | cut -f1))"
        fi
      else
        rm -f "$OUT/nsys_gputrace.csv"
      fi
    fi

    heartbeat_stop
    echo "  [profile] summarised in $((SECONDS - t1))s"
    # What dominates, printed here rather than left in a CSV.  nsys traces
    # every kernel or none, so this is a read of the summary it already wrote,
    # not a narrower collection: the profile costs the same either way.
    if command -v python3 >/dev/null 2>&1; then
      python3 "$HERE/profile_topk.py" "$OUT/nsys_gpukernsum.csv"               "${NVRS_TOPK:-3}" "over ${PROF_TEND:-the full history} days"               2>/dev/null || true
    fi
    if [[ -s "$OUT/nsys_gpukernsum.csv" ]]; then break; fi
  done

  # A report with no CUDA in it is the failure mode that hides.  nsys exits 0,
  # writes a perfectly valid file, and only says so in its own diagnostics:
  #
  #     CUDA profiling might have not been started correctly.
  #     No CUDA events collected. Does the process use CUDA?
  #
  # `nsys stats` then fails on every report because there is no kernel table,
  # and the `|| true` above -- which exists so that a missing profiler cannot
  # fail a benchmark -- swallows that too.  Everything downstream then reports
  # success while producing nothing, so say it here, where the cause is known.
  if [[ ! -s "$OUT/nsys_gpukernsum.csv" ]]; then
    echo "  [profile] nsys captured no CUDA kernels." >&2
    echo "  [profile]" >&2
    # Deliberately not concluding anything about the GPU here.  An empty kernel
    # table means nsys wrote no kernel rows, which is what you get both when a
    # process really never launches a kernel AND when the profiler's injection
    # fails or the process dies before nsys flushes.  Those need opposite fixes,
    # and the logs below distinguish them, so print the logs instead of a guess.
    if [[ -s "$OUT/nsys_run.log" ]]; then
      echo "  [profile] --- nsys_run.log (last 25 lines) ---------------------" >&2
      tail -n 25 "$OUT/nsys_run.log" | sed 's/^/  [profile] /' >&2
    fi
    # nsys does not pass the child's stdout through; it files it separately.
    shopt -s nullglob
    local streams=(/tmp/nvidia/nsight_systems/*/streams/pid_*_std*.log)
    shopt -u nullglob
    if [[ ${#streams[@]} -gt 0 ]]; then
      echo "  [profile] --- the program's own output (last 25 lines) ---------" >&2
      tail -q -n 25 "${streams[@]}" | sed 's/^/  [profile] /' >&2
    fi
    echo "  [profile] ------------------------------------------------------" >&2
    echo "  [profile] The same deck without the profiler:" >&2
    echo "  [profile]     cd $CUDA_DIR && ${MPI_LAUNCH[*]} $NVRS_EXE $RUNDECK" >&2
    echo "  [profile] (under mpirun -- a bare launch takes the Slurm step down)" >&2
    echo "  [profile] If that run completes, the fault is in profiling it, not" >&2
    echo "  [profile] in the run, and NSYS_OPTS narrows which option causes it:" >&2
    echo "  [profile]     NSYS_OPTS=--trace=cuda  $0 nsys" >&2
    return 1
  fi
  echo "  [profile] nsys: $(ls -1 "$OUT"/nsys_*.csv 2>/dev/null | wc -l) summary file(s)"
}

run_ncu() {
  if [[ -z "$NCU_BIN" ]]; then
    echo "  [profile] Nsight Compute not found; skipping the counters." >&2
    echo "  [profile] Set NCU=/path/to/ncu if it is installed elsewhere." >&2
    return 1
  fi
  echo "  [profile] ncu: $NCU_BIN"
  # One metric set, chosen to answer one question -- where each kernel sits
  # against the roofline -- in the fewest passes.  Every extra metric is another
  # replay of every profiled kernel.
  local M="\
dram__bytes.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
  # ncu replays each kernel several times and is roughly 100x slower per
  # kernel, so this is the longest silence in the script by far.
  heartbeat_start "ncu"
  # Under MPI as well, but pinned to a single rank: ncu writes its CSV to
  # stdout, and several ranks interleaving into one stream produces a file that
  # parses into nonsense.  One rank's kernels are the whole kernel mix anyway.
  ( cd "$CUDA_DIR" && "${MPI_RUN[@]}" -np 1 \
      "$NCU_BIN" --csv --target-processes all --metrics "$M" $NCU_CLOCK \
        --launch-skip "${NCU_SKIP:-200}" --launch-count "${NCU_COUNT:-120}" \
        --print-summary per-kernel \
        "$NVRS_EXE" "$RUNDECK" ) > "$OUT/ncu_kernels.csv"           2> >(tee "$OUT/ncu.log" >&2)
  heartbeat_stop
  if [[ ! -s "$OUT/ncu_kernels.csv" ]]; then
    echo "  [profile] ncu wrote nothing; see $OUT/ncu.log" >&2
    return 1
  fi
  local rows
  rows="$(wc -l < "$OUT/ncu_kernels.csv")"
  echo "  [profile] ncu: $rows rows"
  # A handful of rows is not a small profile, it is a failed one: the request
  # was NCU_COUNT kernels and each contributes several. The usual cause is the
  # GPU performance counters being restricted to root, which ncu reports as
  # ERR_NVGPUCTRPERM and which no argument to ncu can work around.
  if [[ "$rows" -lt 20 ]]; then
    echo "  [profile] that is too few to be a profile -- ${NCU_COUNT:-120} kernels were asked for." >&2
    # BOTH streams, because ncu splits itself across them: the CSV goes to
    # stdout, so ncu's own errors land in ncu_kernels.csv, while stderr often
    # carries nothing but mpirun noticing the exit code afterwards.  Showing
    # only the log once made an MPI message look like the cause.
    echo "  [profile] what ncu wrote to its output file:" >&2
    sed -n '1,12p' "$OUT/ncu_kernels.csv" 2>/dev/null | sed 's/^/  [profile]   /' >&2
    echo "  [profile] and to stderr:" >&2
    sed -n '1,12p' "$OUT/ncu.log" 2>/dev/null | sed 's/^/  [profile]   /' >&2
    echo "  [profile] Common causes, in the order worth checking:" >&2
    echo "  [profile]   ERR_NVGPUCTRPERM  counters restricted to root; needs an admin" >&2
    echo "  [profile]   unsupported GPU   this ncu predates the device" >&2
    echo "  [profile]   killed / exit 9   ran out of memory replaying kernels;" >&2
    echo "  [profile]                     try NCU_COUNT=20 to profile fewer" >&2
    return 1
  fi
}

case "$MODE" in
  nsys) run_nsys || rc=1 ;;
  ncu)  run_ncu  || rc=1 ;;
  both) run_nsys || rc=1; run_ncu || rc=1 ;;
  *)    echo "  [profile] unknown mode: $MODE" >&2; exit 1 ;;
esac

# After the profilers, never during one.
render_plots

cat > "$OUT/profile_meta.json" <<EOF
{
  "label": "$LABEL",
  "backend": "${NVRS_BACKEND:-default}",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)",
  "arch": "$(uname -m)",
  "deck": "$(basename "$RUNDECK")",
  "tend_days": "${PROF_TEND:-full}",
  "shortened": $([[ "$RUNDECK" != "$DECK" ]] && echo true || echo false),
  "trace_stride": ${TRACE_STRIDE:-1},
  "note": "Kernel MIX is the benchmark's; the wall-clock total is not, and must not be quoted as one.",
  "generated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
exit $rc
