#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# run_norne.sh -- OPM Flow vs NVRS-CUDA on Norne, for every solver arm:
# AmgX FP32, HYPRE FP32, Ginkgo FP16, Ginkgo FP32 and Ginkgo FP16 with the
# adaptive-precision block-Jacobi stage 2 (see backend_arms.sh).
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# Why this is not run_comparison.sh
# --------------------------------
# That script GENERATES its cases: make_decks.py writes BENCH2P/BENCH3P from
# parameters, so both simulators can be handed the same synthetic deck.  Norne
# is not generated.  It is a real field deck on disk -- Equinor's own -- and
# the two simulators read different files: Flow reads the ECLIPSE deck itself,
# NVRS reads decks/NORNE.DATA, which names that deck with FIELDDECK and adds
# the fluid and solver settings.  There is no shared deck to write, so the case
# needs its own path rather than a branch inside a generator.
#
# Three runs, not two.  AmgX and HYPRE solve the same pressure system by
# different multigrid hierarchies, and which one wins on a faulted corner-point
# field is a question about the field, not a setting to assume.  Running only
# one of them leaves the more interesting half of the comparison unmeasured.
#
# Produces the layout compare.py reads:
#
#   runs/NORNE/opm/          Flow's EGRID/UNRST/SMSPEC/UNSMRY
#   runs/NORNE/nvrs_amgx/    NVRS with AmgX     (also used for the field maps)
#   runs/NORNE/nvrs_hypre/   NVRS with HYPRE
#
# The Flow half is reused for SPE10 (nvrs.sh):
#   NVRS_CASE=SPE10          runs/SPE10 and SPE10_* log names instead of NORNE
#   NVRS_FLOW_ONLY=1         run OPM Flow only: keep the NVRS runs already in
#                            the case directory and add the "opm" entry to its
#                            timing.json rather than rewriting it
#   runs/NORNE/nvrs_ginkgo/      Ginkgo FP16 AMG          (when built in)
#   runs/NORNE/nvrs_ginkgo32/    Ginkgo FP32 AMG
#   runs/NORNE/nvrs_ginkgo_abj/  Ginkgo FP16 AMG + adaptive block-Jacobi
#   runs/NORNE/nvrs_amgx_fp64/   AmgX, native FP64 operator (EMULATE 0)
#   runs/NORNE/timing.json   wall clock and phase breakdown for all three
#
#   ./run_norne.sh                    # both NVRS backends, then compare.html
#   NVRS_ONLY=amgx ./run_norne.sh     # one backend
#   NVRS_ONLY="amgx ginkgo ginkgo32" ./run_norne.sh   # a list
#   SKIP_OPM=1 ./run_norne.sh         # NVRS only, no Flow
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
RUNS="${RUNS:-$HERE/runs}"
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

CASE_NAME="${NVRS_CASE:-NORNE}"
FLOW_ONLY="${NVRS_FLOW_ONLY:-0}"
RD="$RUNS/$CASE_NAME"
LOGS="${LOGS:-$HERE/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS"

hr() { printf '=%.0s' {1..78}; echo; }
secs() { python3 -c 'import time; print("%.6f" % time.time())'; }

NVRS_EXE="${NVRS_EXE:-$CUDA_DIR/nvrs_cuda}"
NVRS_DECK="${NVRS_DECK:-$CUDA_DIR/decks/NORNE.DATA}"
NVRS_RANKS="${NVRS_RANKS:-1}"

# The field deck NVRS points at, and the deck Flow is given.  They are the same
# model; FIELDDECK is how NVRS names it.
FIELD_DECK="$(sed -n 's/^[[:space:]]*FIELDDECK[[:space:]]\+//p' "$NVRS_DECK" \
              | head -1 | tr -d '\r')"
OPM_DECK="${OPM_DECK:-$CUDA_DIR/$FIELD_DECK}"

hr
if [[ "$FLOW_ONLY" == 1 ]]; then
  echo " $CASE_NAME -- OPM Flow (the NVRS runs are already in $RD)"
else
  echo " $CASE_NAME -- OPM Flow vs NVRS-CUDA, every solver arm"
fi
hr
echo " nvrs deck      : $NVRS_DECK"
echo " field deck     : $OPM_DECK"

[[ -x "$NVRS_EXE" ]] || { echo "ERROR: $NVRS_EXE not built.  Run make." >&2; exit 1; }

# Say which binary this is and when it was built, and refuse to benchmark one
# that is older than the code.
#
# "make: 'nvrs_cuda' is up to date" is not proof: a pull that lands while the
# objects are newer than the checkout leaves make with nothing to do and the
# change unbuilt.  That cost three rounds of "did your changes land" against a
# binary that predated them, and a benchmark run on a stale binary publishes a
# number for code nobody is running.
echo " nvrs exe       : $NVRS_EXE  (built $(date -r "$NVRS_EXE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo unknown))"
newer="$(find "$CUDA_DIR/src" "$CUDA_DIR/include" -newer "$NVRS_EXE"            -name '*.cu' -o -newer "$NVRS_EXE" -name '*.cpp'            -o -newer "$NVRS_EXE" -name '*.hpp' 2>/dev/null | head -5)"
if [[ -n "$newer" ]]; then
  echo >&2
  echo "ERROR: these sources are newer than $NVRS_EXE:" >&2
  sed 's/^/         /' <<<"$newer" >&2
  echo "       Run 'make -j' first.  NVRS_SKIP_STALE_CHECK=1 to benchmark anyway." >&2
  [[ -z "${NVRS_SKIP_STALE_CHECK:-}" ]] && exit 1
fi
[[ -f "$NVRS_DECK" ]] || { echo "ERROR: $NVRS_DECK missing." >&2; exit 1; }
if [[ ! -f "$OPM_DECK" ]]; then
  echo "ERROR: $OPM_DECK missing.  Run ./get_norne.sh to fetch the dataset." >&2
  exit 1
fi

# --- OPM Flow, on the GPU where it can be ----------------------------------
# Detection is run_comparison.sh's, verbatim, because this script inventing its
# own found nothing: it expanded "${OPM_SRC:-}" with no default, so every
# candidate collapsed to "/install/bin/flow" and the run reported
#
#     flow : <not found>
#     -- OPM Flow skipped --
#
# and produced a "comparison" with nothing to compare against.
#
# A source-tree install/ comes before PATH: /usr/bin/flow is usually an older
# system build, and running that while a freshly built GPU-capable one sits in
# the OPM tree is how a benchmark ends up on the CPU without saying so.
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
  # Open MPI's startup probes hwloc's GL/OpenCL components to enumerate which
  # display each GPU is attached to.  On a machine whose X server does not
  # answer that particular query -- this one included -- the probe blocks
  # forever: mpirun never even reaches its own startup banner, no error, no
  # timeout, and the run just sits there (seen as a multi-hour hang before
  # this was tracked down).  Neither component is used for anything this
  # script needs, so turn them off before mpirun can ask.  Respect a value
  # the caller already set.
  export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl,-opencl}"
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

FLOW_EXE="${FLOW_EXE:-}"
if [[ -z "$FLOW_EXE" ]]; then
  for cand in "$OPM_SRC/install/bin/flow"               "$OPM_SRC/opm-simulators/build/bin/flow"               "$(command -v flow || true)" /usr/bin/flow; do
    if [[ -n "$cand" && -x "$cand" ]]; then FLOW_EXE="$cand"; break; fi
  done
fi
FLOW_EXE="${FLOW_EXE:-/usr/bin/flow}"

# Is this build GPU-capable?  Three signals, weighted as run_comparison.sh
# weights them:
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
  [[ -f "$cfg" ]] && grep -qE '^[[:space:]]*#define[[:space:]]+HAVE_CUDA[[:space:]]+1'     "$cfg" && cfg_cuda=1

  out="$("${MPI_RUN[@]}" -np 1 "$FLOW_EXE"            --accelerator-mode=cusparse --version 2>&1)"
  if grep -qiE 'was not found by CMake|cusparseSolver was chosen, but' <<<"$out"; then
    return 1
  fi
  if grep -qiE 'unknown value for parameter|AcceleratorMode' <<<"$out"; then
    return 1
  fi
  [[ $linked -eq 1 || $cfg_cuda -eq 1 ]] && return 0
  return 1
}

OPM_RANKS="${OPM_RANKS:-1}"
OPM_ACCEL="" OPM_GPU=0
OPM_SOLVER_DESC="Flow default (CPU)"
if [[ -z "${SKIP_OPM:-}" ]]; then
  # Fail loudly rather than skipping.  A silent skip is how this produced a
  # one-sided comparison and called it done.
  [[ -x "$FLOW_EXE" ]] || {
    echo "ERROR: flow not found at $FLOW_EXE." >&2
    echo "       Looked under OPM_SRC=$OPM_SRC.  Run ./install_opm_gpu.sh," >&2
    echo "       or set FLOW_EXE=/path/to/flow, or SKIP_OPM=1 to run NVRS alone." >&2
    exit 1
  }
  if [[ -n "${OPM_EXTRA:-}" ]]; then
    OPM_ACCEL="$OPM_EXTRA"
    OPM_SOLVER_DESC="$OPM_EXTRA"
  elif opm_has_gpu; then
    OPM_GPU=1
    OPM_ACCEL="--accelerator-mode=${OPM_ACCELERATOR:-cusparse}"
    OPM_SOLVER_DESC="gpuistl: gpubicgstab + OPMGPUILU0, FP32 factors (GPU)"
  else
    echo "WARNING: $FLOW_EXE is not GPU-capable; Flow will run on the CPU and" >&2
    echo "         the speedup figures will not be a GPU-to-GPU comparison." >&2
  fi
fi
# Which preconditioner Flow builds when the GPU solver hands a system back.
#
# --accelerator-mode=cusparse does not replace the CPU path, it shadows it.
# When the GPU solver cannot converge on a system, Flow falls back to Dune for
# that one and builds the preconditioner named by --linear-solver.  Unset,
# that is this version's default, cprw, and this build does not have it:
#
#   Warning: cusparse did not converge, now trying Dune to solve current
#            linear system...
#   Error: Preconditioner type cprw is not registered in the factory.
#          Available types are: DILU DuneILU GPUDILU ... amg cpr cprt famg kamg
#
# which killed the run at report step 107 of 247, on the first system the GPU
# gave up on.  run_comparison.sh never hit this: its cases are small enough
# that cusparse always converges, so the fallback is never constructed.
#
# Note what that list is: PRECONDITIONER type names.  --linear-solver takes
# PRESET names, a different vocabulary, and anything it does not recognise it
# treats as a JSON filename.  So "cpr" is not the answer even though "cpr"
# appears in the list -- it would be looked for as a file.  The presets that
# map onto registered types here are
#
#   cpr_quasiimpes -> cpr     (closest to cprw: CPR, quasi-IMPES weights)
#   cpr_trueimpes  -> cprt
#   ilu0           -> ParOverILU0
#
# Rather than trust that, ask the binary.  A name Flow does not know fails at
# solver setup -- which on this deck is fifteen minutes in -- so it is worth
# the one --help call to find out at startup instead.
# Did the caller name one?  If so, take it and ask nothing: the probe below
# is another Flow launch, and Flow launches are not free on this cluster.
OPM_LINSOLVER_SET=0
[[ -n "${OPM_LINSOLVER+x}" ]] && OPM_LINSOLVER_SET=1
OPM_LINSOLVER="${OPM_LINSOLVER-cpr_quasiimpes}"
if [[ $OPM_LINSOLVER_SET -eq 0 && -z "${SKIP_OPM:-}" && -n "$OPM_LINSOLVER" && -x "$FLOW_EXE" ]]; then
  # UNDER mpirun, never bare.  Flow calls MPI_Init even for --help, and a
  # singleton init on this cluster does not fail politely: it took the
  # whole Slurm step down with
  #
  #   pmixp_client_v2.c:225: Error handler invoked: status = -61
  #   *** STEP ... CANCELLED ... DUE to SIGNAL Killed ***
  #
  # which is the same reason opm_has_gpu() above launches through mpirun
  # and scripts/run.sh refuses to start nvrs_cuda directly.
  flow_help="$("${MPI_RUN[@]}" -np 1 \
                 "$FLOW_EXE" --help 2>&1 || true)"
  if [[ -n "$flow_help" ]] && ! grep -q -- "$OPM_LINSOLVER" <<<"$flow_help"; then
    for alt in cpr_quasiimpes cpr_trueimpes dilu ilu0; do
      if grep -q -- "$alt" <<<"$flow_help"; then
        echo "NOTE: this Flow does not list --linear-solver=$OPM_LINSOLVER;" >&2
        echo "      using $alt instead (OPM_LINSOLVER= to leave Flow's default)." >&2
        OPM_LINSOLVER="$alt"
        break
      fi
    done
  fi
fi

echo " flow           : ${FLOW_EXE:-<not found>}"
echo " flow solver    : $OPM_SOLVER_DESC"
echo " flow fallback  : ${OPM_LINSOLVER:-<Flow default>}  (CPU path, when the GPU solver gives up)"
# Wells in the matrix: required by Flow's GPU solver, which has no separate
# well operator.  On the CPU they replace CPRW's well operators and cost about
# 20% run time (OPM's linear-solver documentation), so a CPU comparison that
# should use Flow's defaults sets OPM_ADD_WELL_CONTRIB=false.
echo " flow wells     : --matrix-add-well-contributions=${OPM_ADD_WELL_CONTRIB:-true}"
echo

# Clear the run directory -- but not a Flow result we were told to keep.
#
# SKIP_OPM means "I already have OPM's output".  Wiping it here made the flag
# self-defeating: the run it was meant to reuse was deleted before the run
# that was meant to reuse it.  Same for the previous timing.json, whose opm
# entry is the only record of how long that run took.
keep_opm=""
if [[ -n "${SKIP_OPM:-}" && -d "$RD/opm" ]]; then
  keep_opm="$(mktemp -d)"
  mv "$RD/opm" "$keep_opm/opm"
  [[ -f "$RD/timing.json" ]] && cp "$RD/timing.json" "$keep_opm/timing.json"
fi
if [[ "$FLOW_ONLY" == 1 ]]; then
  # The NVRS runs, their scaling sweep and the generated field deck live here.
  rm -rf "$RD/opm"
else
  rm -rf "$RD"
fi
mkdir -p "$RD"
if [[ -n "$keep_opm" ]]; then
  mv "$keep_opm/opm" "$RD/opm"
  [[ -f "$keep_opm/timing.json" ]] && cp "$keep_opm/timing.json" "$RD/timing.prev.json"
  rm -rf "$keep_opm"
  echo "-- keeping the OPM Flow output already in $RD/opm --"
else
  mkdir -p "$RD/opm"
fi

opm_s="" ; opm_log=""
if [[ -z "${SKIP_OPM:-}" ]]; then
  echo "-- OPM Flow --"
  opm_log="$LOGS/${CASE_NAME}_opm_${STAMP}.log"
  t0="$(secs)"
  ( cd "$RD/opm" && \
    "${MPI_RUN[@]}" -np "$OPM_RANKS" \
      "$FLOW_EXE" "$OPM_DECK" \
      --parsing-strictness=low \
      --enable-ecl-output=true \
      --output-dir="$RD/opm" \
      --matrix-add-well-contributions="${OPM_ADD_WELL_CONTRIB:-true}" \
      ${OPM_LINSOLVER:+--linear-solver="$OPM_LINSOLVER"} \
      $OPM_ACCEL ) 2>&1 | tee "$opm_log"
  rc=${PIPESTATUS[0]}
  t1="$(secs)"
  opm_s="$(python3 -c "print('%.3f' % ($t1 - $t0))")"
  echo "-- OPM Flow rc=$rc in ${opm_s} s --"
  echo
else
  echo "-- OPM Flow skipped --"; echo
fi

# --- NVRS, once per stage-one multigrid ------------------------------------
# The deck is copied and LINSOLVER rewritten rather than passed a flag, so the
# two runs differ in exactly one line and that line is recorded beside the
# result.  OUTDIR and CASENAME are renamed too, or the second run would land on
# top of the first.
declare -A NVRS_SECS
# shellcheck source=backend_arms.sh
. "$HERE/backend_arms.sh"
# Every arm the binary can run (all five with Ginkgo linked in).  NVRS_ONLY is
# a space- or comma-separated list of arm names.
read -r -a BACKENDS <<< "$(arm_defaults "$NVRS_EXE")"
[[ -n "${NVRS_ONLY:-}" ]] && read -r -a BACKENDS <<< "${NVRS_ONLY//,/ }"
[[ "$FLOW_ONLY" == 1 ]] && BACKENDS=()
for be in ${BACKENDS[@]+"${BACKENDS[@]}"}; do
  arm_known "$be" || { echo "ERROR: unknown arm '$be' (want: $NVRS_ALL_ARMS)" >&2; exit 2; }
done

for be in ${BACKENDS[@]+"${BACKENDS[@]}"}; do
  UP="$(echo "$be" | tr '[:lower:]' '[:upper:]')"
  DECK="$RD/${CASE_NAME}_${UP}.DATA"
  OUT="OUTPUT_${CASE_NAME}_${UP}"
  sed -e "s/^OUTDIR.*/OUTDIR   $OUT/" \
      -e "s/^CASENAME.*/CASENAME ${CASE_NAME}_${UP}/" "$NVRS_DECK" > "$DECK.tmp"
  arm_apply_deck "$DECK.tmp" "$DECK" "$be"
  rm -f "$DECK.tmp"
  echo "   arm $be ($(arm_label "$be")): $(arm_keywords "$be" | paste -sd' ' -)"

  # Ginkgo arms are single-GPU in this build; the others take NVRS_RANKS.
  RANKS_BE="$NVRS_RANKS"
  arm_single_gpu "$be" && RANKS_BE=1
  echo "-- NVRS-CUDA ($UP) --"
  log="$LOGS/${CASE_NAME}_nvrs_${be}_${STAMP}.log"
  t0="$(secs)"
  ( cd "$CUDA_DIR" && \
    "${MPI_RUN[@]}" -np "$RANKS_BE" \
      "$NVRS_EXE" "$DECK" ) 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  t1="$(secs)"
  NVRS_SECS[$be]="$(python3 -c "print('%.3f' % ($t1 - $t0))")"
  echo "-- NVRS-CUDA ($UP) rc=$rc in ${NVRS_SECS[$be]} s --"
  echo

  mkdir -p "$RD/nvrs_$be"
  if [[ -d "$CUDA_DIR/$OUT" ]]; then
    mv "$CUDA_DIR/$OUT" "$RD/nvrs_$be/"
  else
    echo "WARNING: $CUDA_DIR/$OUT not found; that run produced no output" >&2
  fi
done

# --- timing.json, in the shape compare.py reads ----------------------------
python3 - "$RD/timing.json" "${opm_s:-}" "${opm_log:-}" "$OPM_SOLVER_DESC${OPM_LINSOLVER:+, CPU fallback $OPM_LINSOLVER}" \
         "$OPM_RANKS" "$NVRS_RANKS" "$RD" \
         "$(for _b in "${!NVRS_SECS[@]}"; do printf '%s=%s ' "$_b" "${NVRS_SECS[$_b]}"; done)" <<'PY'
import glob, json, os, re, sys
(out, opm_s, opm_log, opm_solver, opm_r, nvrs_r, rd, arm_secs) = sys.argv[1:9]
SECS = dict(kv.split("=", 1) for kv in arm_secs.split() if "=" in kv)


def opm_breakdown(path):
    """Flow's own end-of-run phase timings.  Parsing them is the only way to
    say how much of ITS run was the linear solve; timing from outside gives
    the total and nothing else."""
    want = {"Setup time": "setup", "Assembly time": "assembly",
            "Linear solve time": "linear", "Linear setup": "linear_setup",
            "Props/update time": "props", "Pre/post step": "prepost",
            "Output write time": "output", "Simulation time": "simulation"}
    got = {}
    if not path or not os.path.exists(path):
        return got
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for label, key in want.items():
                m = re.match(r"\s*%s:\s+([0-9.]+)\s*s" % re.escape(label), line)
                if m:
                    got[key] = float(m.group(1))
    return got


def nvrs_breakdown(sub):
    """NVRS writes its own split into meta.json."""
    hits = sorted(glob.glob(os.path.join(rd, sub, "**", "meta.json"),
                            recursive=True))
    if not hits:
        return {}
    try:
        with open(hits[0]) as fh:
            m = json.load(fh)
    except Exception:
        return {}
    return {k: m.get(k) for k in
            ("assembly_seconds", "linear_seconds", "amg_setup_seconds",
             "amg_solve_seconds", "stage2_seconds", "spmv_seconds",
             "ortho_seconds", "ijvec_seconds", "stage2_setup_seconds",
             "wall_seconds",
             "amg_setups", "amg_applications", "amg_cycles",
             "amg_reuses", "amg_reuse_aborts")
            if m.get(k) is not None}


t = {}
if os.environ.get("NVRS_FLOW_ONLY") == "1" and os.path.exists(out):
    # Flow-only: the NVRS entries were written by the run that made them.
    try:
        with open(out) as fh:
            t = {k: v for k, v in json.load(fh).items() if k != "opm"}
    except Exception:
        t = {}
if opm_s:
    obd = opm_breakdown(opm_log)
    # Same for Flow: its own "Simulation time" where the log gives one, so
    # the two runtimes being compared are both simulation and neither is
    # carrying start-up or file writing the other does not do.
    osim = obd.get("simulation")
    t["opm"] = dict(seconds=float(osim if osim else opm_s), ranks=opm_r,
                    solver=opm_solver, hardware="CPU/GPU per solver",
                    breakdown=obd, wall_including_output=float(opm_s))
else:
    # Flow was skipped, so carry its timing over from the run whose output we
    # are reusing.  Without this the speedup charts have NVRS and nothing to
    # compare it against, on a run that has Flow's fields sitting right there.
    prev = os.path.join(rd, "timing.prev.json")
    if os.path.exists(prev):
        try:
            with open(prev) as fh:
                old = json.load(fh)
            if "opm" in old:
                t["opm"] = old["opm"]
                print("  reused the previous run's OPM timing (Flow was skipped)")
        except Exception:
            pass
DESC = {
    "amgx": "CPR: AmgX AMG (FP32) + block-Jacobi",
    "hypre": "CPR: HYPRE BoomerAMG (FP32) + block-Jacobi",
    "ginkgo": "CPR: Ginkgo PGM AMG (FP16) + block-Jacobi",
    "ginkgo32": "CPR: Ginkgo PGM AMG (FP32) + block-Jacobi",
    "ginkgo_abj": "CPR: Ginkgo PGM AMG (FP16) + Ginkgo adaptive-precision block-Jacobi",
    "amgx_fp64": "CPR: AmgX AMG (FP32) + block-Jacobi, native FP64 operator",
    "amgx_fp64amg": "CPR: AmgX AMG (FP64) + block-Jacobi, native FP64 operator: everything FP64",
    "amgx_tensorcore": "CPR: AmgX AMG (FP32) + block-Jacobi, Tensor Core Jacobian",
}
for arm in ("amgx", "hypre", "ginkgo", "ginkgo32", "ginkgo_abj", "amgx_fp64", "amgx_fp64amg", "amgx_tensorcore"):
    key, secs, desc = "nvrs_" + arm, SECS.get(arm, ""), DESC[arm]
    if secs:
        bd = nvrs_breakdown(key)
        # The simulator's OWN wall clock, not this script's.
        #
        # nvrs_cuda renders its report before exiting, so the time measured
        # around mpirun is the simulation PLUS the plotting -- 612 s against
        # 115 s of solving on Norne, which would have been published as the
        # runtime and compared against Flow, which plots nothing.  meta.json
        # carries what the solver actually took; the outer measurement is
        # kept beside it rather than thrown away, since it is the honest
        # answer to a different question.
        sim = bd.get("wall_seconds")
        t[key] = dict(seconds=float(sim if sim else secs),
                      ranks=("1" if arm.startswith("ginkgo") else nvrs_r),
                      solver=desc, hardware="GPU", breakdown=bd,
                      wall_including_output=float(secs))
with open(out, "w") as fh:
    json.dump(t, fh, indent=2)
print("  wrote %s with %d run(s): %s" % (out, len(t), ", ".join(sorted(t))))
PY

hr
echo " building compare.html"
hr
# Optional profile, off unless asked for.  It re-runs a shortened Norne, so it
# costs another run: nsys adds about 1.2x, ncu replays every kernel it profiles
# and is roughly 100x slower per kernel.
#
#   NVRS_PROFILE=1 ./run_norne.sh      nsys only, the timeline
#   NVRS_PROFILE=both ./run_norne.sh   add the kernel counters
if [[ -n "${NVRS_PROFILE:-}" ]]; then
  pmode="$NVRS_PROFILE"; [[ "$pmode" == "1" ]] && pmode=nsys
  hr
  echo " profiling Norne ($pmode)"
  hr
  "$HERE/profile_norne.sh" "$pmode" ||     echo "  [profile] did not complete; the benchmark above is unaffected" >&2
  python3 "$HERE/profile_plots.py" --dir "$RD/profile" || true
fi

# nvrs.sh runs the report itself, after profiling, so that the Nsight
# figures reach the page.  Building it here as well means building it
# twice: on BENCH3P that was 113 seconds each time, for one page that the
# second run overwrites.  Standalone, this is still the right place for it.
if [[ -z "${NVRS_SKIP_REPORT:-}" ]]; then
  python3 "$HERE/compare.py" --runs "$RUNS" || true
else
  echo " report         : skipped here; nvrs.sh builds it after profiling"
fi
echo
echo "done.  runs under $RD"
