#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# nvrs.sh -- one entry point for the whole thing.
#
#   ./nvrs.sh                          # the menu: commands, cases, options
#   ./nvrs.sh install                  # HYPRE, AmgX, Ginkgo, OPM Flow, Nsight, nvrs_cuda
#   ./nvrs.sh all NORNE                # fetch, run, profile, report
#   ./nvrs.sh all NORNE --no-opm --backend amgx --fraction 0.25
#
# The pieces were all here already and each had to be found, called in the
# right order, and handed the right environment. The step that got missed was
# never the hard one: it was sourcing an environment file between two
# installers, or knowing that Norne is downloaded while SPE10 is downloaded and
# converted and the BENCH cases are generated, or that the figures need a
# second command after the profile.
#
# So this replaces nothing. It calls the same scripts in the same order, and
# knows which one a case needs.
#
# Every option is a flag as well as an environment variable. The variables came
# first and are still honoured, but a variable is invisible: nobody discovers
# SKIP_OPM by typing --help, and --help is what people type.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
RUNS="${RUNS:-$HERE/runs}"
ROOT="$(cd "$CUDA_DIR/../.." && pwd)"

# Source the environment the build was linked against, before anything runs.
#
# install_all.sh sources these itself, which is why the build succeeds, and
# the variables die with that process, which is why the run then fails on
#
#   nvrs_cuda: error while loading shared libraries: libHYPRE.so.302
#
# Its own closing note says to source them by hand in a new shell.  Removing
# that step is most of the reason this script exists, so it happens here and
# every command below inherits it.  Sourcing an already-sourced file is
# harmless: both only prepend to PATH and LD_LIBRARY_PATH.
HYPRE_ENV="${HYPRE_ENV:-$ROOT/Mixed_precision/hypre_env.sh}"
AMGX_ENV="${AMGX_ENV:-${AMGX_PREFIX:-$ROOT/amgx-install}/amgx_env.sh}"
GINKGO_ENV="${GINKGO_ENV:-${GINKGO_PREFIX:-$ROOT/ginkgo-install}/ginkgo_env.sh}"
# shellcheck source=backend_arms.sh
. "$HERE/backend_arms.sh"
NVRS_ENV_SOURCED=""
for _e in "$HYPRE_ENV" "$AMGX_ENV" "$GINKGO_ENV"; do
  if [[ -f "$_e" ]]; then
    # set -u off while sourcing: these files are written for an interactive
    # shell and reference variables that may not exist.  Failing here used to
    # be swallowed by a || true, which is how a broken environment looked
    # exactly like a working one.
    set +u
    # shellcheck disable=SC1090
    source "$_e" >/dev/null 2>&1
    set -u
    NVRS_ENV_SOURCED="${NVRS_ENV_SOURCED}$(basename "$_e") "
  fi
done

# Can the binary actually find its libraries?
#
# hypre_env.sh deliberately keeps HYPRE off LD_LIBRARY_PATH: the FP32 and FP64
# builds ship the same libHYPRE.so soname, so a search path could load the
# wrong precision and return a quietly different answer.  The link uses -rpath
# instead.  When that rpath does not resolve -- a moved tree, a rebuilt
# prefix, a different node -- the failure is
#
#   nvrs_cuda: error while loading shared libraries: libHYPRE.so.302
#
# which arrives after the deck has been fetched and Flow has run, and cost a
# full BENCH3P and SPE10 run each to discover.  So it is checked first, and
# repaired from the prefixes the build was configured with rather than by
# putting a guess on the search path.
# Does the binary have the backend the deck asks for?
#
# Falling back is a warning in the run log and a silent change to what is
# being measured.  On SPE10 the fallback from AmgX to HYPRE turned a 100-day
# step into 571 Newton iterations and 60,262 Krylov iterations, timesteps
# chopped to 0.39 days, and a run that looked broken rather than substituted.
# A benchmark that quietly measures a different solver than it was asked for
# is worse than one that stops.
nvrs_backendcheck() {
  local exe="$CUDA_DIR/nvrs_cuda" deck want
  [[ -x "$exe" ]] || return 0
  command -v ldd >/dev/null 2>&1 || return 0
  # Ginkgo first: a run that asks for it and a binary without it would fall
  # back to AmgX and label AmgX numbers as the FP16 arm.
  local _b
  for _b in $BACKENDS; do
    if [[ "$_b" == amgx_tensorcore && ! -x "${exe}_tc" ]]; then
      echo "ERROR: --backend asks for amgx_tensorcore, and ${exe}_tc is not built." >&2
      echo "       Build it:        TENSORCORE=1 ./nvrs.sh install" >&2
      echo "       Or leave it out: --backend amgx,amgx_fp64amg" >&2
      return 1
    fi
    if arm_needs_ginkgo "$_b" && ! ldd "$exe" 2>/dev/null | grep -qi ginkgo; then
      echo "ERROR: this nvrs_cuda was built without Ginkgo, and --backend asks for it." >&2
      echo "       It would fall back to AmgX and label AmgX as the FP16 arm." >&2
      echo "       Install and rebuild:  ./install_ginkgo.sh && source <prefix>/ginkgo_env.sh" >&2
      echo "                             cd $CUDA_DIR && make clean && make -j" >&2
      echo "       Or leave it out:      --backend amgx,hypre" >&2
      return 1
    fi
  done
  ldd "$exe" 2>/dev/null | grep -qi amgx && return 0   # has it, nothing to say

  # No AmgX in the binary.  That is fine unless something wants it.
  local b needs=""
  for b in $BACKENDS; do [[ "$b" == "amgx" ]] && needs="requested by --backend"; done
  deck="$(deck_for "$CASE" 2>/dev/null || true)"
  if [[ -z "$needs" && -n "$deck" && -f "$deck" ]]; then
    grep -qiE "^[[:space:]]*LINSOLVER[[:space:]]+AMGX" "$deck" && needs="named by $(basename "$deck")"
  fi
  [[ -n "$needs" ]] || return 0

  echo "ERROR: this nvrs_cuda was built without AmgX, and AmgX is $needs." >&2
  echo "       It would fall back to HYPRE and measure a different solver." >&2
  echo >&2
  echo "       Rebuild with AmgX:" >&2
  echo "         cd $CUDA_DIR && make clean && cd benchmark && ./nvrs.sh install" >&2
  echo "       and check the build prints:  AmgX: <a path>, not \"not found\"." >&2
  echo >&2
  echo "       To measure HYPRE deliberately, say so:" >&2
  echo "         ./nvrs.sh $CMD $CASE --backend hypre" >&2
  return 1
}

nvrs_libcheck() {
  local exe="$CUDA_DIR/nvrs_cuda" miss
  [[ -x "$exe" ]] || return 0
  command -v ldd >/dev/null 2>&1 || return 0
  miss="$(ldd "$exe" 2>/dev/null | awk "/not found/ {print \$1}")"
  [[ -n "$miss" ]] || return 0

  local d added=""
  for d in "${HYPRE_FP32_LIB:-}" "${AMGX_DIR:-}/lib" "${AMGX_DIR:-}/lib64" \
           "${GINKGO_DIR:-}/lib" "${GINKGO_DIR:-}/lib64"; do
    if [[ -n "$d" && -d "$d" ]]; then
      export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
      added="$added$d "
    fi
  done
  miss="$(ldd "$exe" 2>/dev/null | awk "/not found/ {print \$1}")"
  if [[ -z "$miss" ]]; then
    echo " libraries      : rpath did not resolve; added $added"
    echo "                  (the FP32 HYPRE the build was configured with)"
    return 0
  fi
  echo "ERROR: nvrs_cuda cannot load:" >&2
  printf "         %s
" $miss >&2
  echo "       Its rpath does not resolve and the prefixes are not where" >&2
  echo "       hypre_env.sh looked.  Rebuild against the current tree:" >&2
  echo "         ./nvrs.sh install" >&2
  echo "       or show me:  ldd $exe | grep -i \"not found\"" >&2
  return 1
}

# Bold only on a terminal.  This output is routinely piped into a log or pasted
# into a message, and escape codes there are noise at best.
if [[ -t 1 ]]; then
  bold() { printf '\033[1m%s\033[0m\n' "$*"; }
  dim()  { printf '\033[2m%s\033[0m\n' "$*"; }
else
  bold() { printf '%s\n' "$*"; }
  dim()  { printf '%s\n' "$*"; }
fi
rule() { printf '=%.0s' {1..78}; echo; }
step() { echo; rule; bold " $*"; rule; }
die()  { echo "ERROR: $*" >&2; echo "       ./nvrs.sh --help" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# Defaults come from the environment so the older variable form keeps working,
# and every flag below overrides it.
# All three stage-1 multigrids when the binary has Ginkgo, the original two
# when it does not -- decided after the environment is sourced, below.
# NVRS_ONLY is the name the runners underneath read (run_norne.sh,
# run_comparison.sh); accepting it here too means "NVRS_ONLY=amgx ./nvrs.sh run
# SPE10" runs AmgX alone rather than silently running every arm first.
BACKENDS="${NVRS_BACKENDS:-${NVRS_ONLY:-}}"
BACKENDS="${BACKENDS//,/ }"
# Whether the arms were named (flag or environment) rather than defaulted:
# SPE10 leaves the Ginkgo arms out of its default set, never out of a named one.
BACKENDS_NAMED=0
[[ -n "$BACKENDS" ]] && BACKENDS_NAMED=1
PROFILE_MODE="nsys"
# Strong scaling: repeating the run on 2, 4, ... GPUs.  Empty means the
# per-case default below; 1 and 0 force it either way.
SCALING=""
DO_PROFILE=1
[[ -n "${SKIP_PROFILE:-}" ]] && DO_PROFILE=0

CMD="" CASE="" STEPS_GIVEN=0
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-opm|--skip-opm)  export SKIP_OPM=1 ;;
    --opm)                unset SKIP_OPM ;;
    --backend)
      [[ $# -ge 2 ]] || die "--backend needs a value: $NVRS_ALL_ARMS, a comma list, both or all"
      BACKENDS_NAMED=1
      case "${2,,}" in
        all)  BACKENDS="$NVRS_ALL_ARMS" ;;
        both) BACKENDS="amgx hypre" ;;
        *)
          BACKENDS=""
          for _b in ${2//,/ }; do
            arm_known "${_b,,}" || die "--backend takes $NVRS_ALL_ARMS, a comma list, both or all (got '$2')"
            BACKENDS="${BACKENDS:+$BACKENDS }${_b,,}"
          done ;;
      esac
      shift ;;
    --no-profile)         DO_PROFILE=0 ;;
    --scaling)            SCALING=1 ;;
    --no-scaling)         SCALING=0 ;;
    --ncu)                PROFILE_MODE="both" ;;
    --ncu-only)           PROFILE_MODE="ncu" ;;
    --fraction)
      [[ $# -ge 2 ]] || die "--fraction needs a value: 0.25, or all"
      export PROF_FRACTION="$2"; shift ;;
    --tend)
      [[ $# -ge 2 ]] || die "--tend needs a number of days"
      export PROF_TEND="$2"; shift ;;
    --steps)
      [[ $# -ge 2 ]] || die "--steps needs a number, or all"
      export PROF_STEPS="$2"; STEPS_GIVEN=1; shift ;;
    --deck)
      [[ $# -ge 2 ]] || die "--deck needs a path to a .DATA file"
      [[ -f "$2" ]] || die "no deck at $2"
      export NVRS_DECK="$2"; shift ;;
    --ranks)
      [[ $# -ge 2 ]] || die "--ranks needs a number"
      export NVRS_RANKS="$2"; shift ;;
    --heartbeat)
      [[ $# -ge 2 ]] || die "--heartbeat needs seconds, or 0 to silence it"
      export NVRS_HEARTBEAT="$2"; shift ;;
    -h|--help)            CMD="help" ;;
    -*)                   die "unknown option: $1" ;;
    *)                    POSITIONAL+=("$1") ;;
  esac
  shift
done

# The default backend set.  Ginkgo joins when the binary was linked with it, so
# a machine that has not installed it keeps working exactly as before and one
# that has gets the third arm without asking.
DEFAULT_BACKENDS="$(arm_defaults "$CUDA_DIR/nvrs_cuda")"
BACKENDS="${BACKENDS:-$DEFAULT_BACKENDS}"
for _b in $BACKENDS; do
  arm_known "${_b,,}" || die "unknown backend arm '$_b' (from --backend, NVRS_BACKENDS or NVRS_ONLY); want: $NVRS_ALL_ARMS"
done

# Several cases at once.  "install then run everything" is what people
# actually want, and naming them one command at a time is the sort of friction
# that gets a step forgotten.
CASES=()
if [[ "$CMD" != "help" ]]; then
  CMD="${POSITIONAL[0]:-}"
  for _c in "${POSITIONAL[@]:1}"; do
    # "--backend amgx ginkgo" (a space where a comma was meant) leaves the
    # second arm here as a case name, and an unknown case once ran a 1-cell
    # default model.  A word that names a backend arm is taken as one.
    if arm_known "${_c,,}"; then
      echo "  note: '$_c' is a backend, not a case; running it as --backend ${_c,,}" >&2
      [[ " $BACKENDS " == *" ${_c,,} "* ]] || BACKENDS="${BACKENDS:+$BACKENDS }${_c,,}"
      BACKENDS_NAMED=1
      continue
    fi
    CASES+=("$(echo "$_c" | tr '[:lower:]' '[:upper:]')")
  done
fi
CASE="${CASES[0]:-}"

if [[ $STEPS_GIVEN -eq 1 && -n "$CASE" && "$CASE" != "NORNE" ]]; then
  die "--steps is Norne only: a report step is a fortnight there and something
       else in every other deck.  Use --fraction 0.25 or --tend <days> for $CASE."
fi

# ---------------------------------------------------------------------------
# What each command reads
# ---------------------------------------------------------------------------
# A global option list makes the reader work out which of them apply to the
# command they are about to type.  Every command says it instead, at the point
# of running, together with what is already in effect -- so an option that was
# set and is being ignored by this command is visible rather than mysterious.
opts_for() {
  case "$1" in
    install) echo "(none)" ;;
    cases)   echo "(none)" ;;
    get)     echo "--deck" ;;
    run)     echo "--no-opm  --backend  --ranks  --scaling  --no-scaling" ;;
    profile) echo "--backend  --fraction  --tend  --steps  --ncu  --deck  --ranks  --heartbeat  --scaling  --no-scaling" ;;
    report)  echo "(none)" ;;
    all)     echo "--no-opm  --backend  --no-profile  --ncu  --fraction  --tend  --steps  --deck  --ranks  --heartbeat" ;;
    *)       echo "(none)" ;;
  esac
}

active_opts() {
  local a=()
  [[ -n "${SKIP_OPM:-}" ]]        && a+=("--no-opm")
  [[ "$BACKENDS" != "$DEFAULT_BACKENDS" ]] && a+=("--backend $BACKENDS")
  [[ $DO_PROFILE -eq 0 ]]         && a+=("--no-profile")
  [[ "$PROFILE_MODE" != "nsys" ]] && a+=("--ncu[$PROFILE_MODE]")
  [[ -n "${PROF_FRACTION:-}" ]]   && a+=("--fraction $PROF_FRACTION")
  [[ -n "${PROF_TEND:-}" ]]       && a+=("--tend $PROF_TEND")
  [[ -n "${PROF_STEPS:-}" ]]      && a+=("--steps $PROF_STEPS")
  [[ -n "${NVRS_DECK:-}" ]]       && a+=("--deck $NVRS_DECK")
  [[ -n "${NVRS_RANKS:-}" ]]      && a+=("--ranks $NVRS_RANKS")
  [[ -n "${NVRS_HEARTBEAT:-}" ]]  && a+=("--heartbeat $NVRS_HEARTBEAT")
  [[ "$SCALING" == "1" ]]         && a+=("--scaling")
  [[ "$SCALING" == "0" ]]         && a+=("--no-scaling")
  if [[ ${#a[@]} -eq 0 ]]; then echo "(all defaults)"; else echo "${a[*]}"; fi
}

show_options() {
  printf '  %-21s %s\n' "in effect now" "$(active_opts)"
  printf '  %-21s %s\n' "'$1' also takes" "$(opts_for "$1")"
  printf '  %-21s %s\n' "full list" "./nvrs.sh --help"
  echo
}

# ---------------------------------------------------------------------------
# Which preparation a case needs
# ---------------------------------------------------------------------------
# Three kinds, and the difference is the thing most easily got wrong: Norne is
# downloaded, SPE10 is downloaded and then converted, and the BENCH cases are
# generated from parameters.  A case already sitting in decks/ needs nothing.
prep_for() {
  case "$1" in
    NORNE)   echo "fetch" ;;
    SPE10*)  echo "spe10" ;;
    BENCH*)  echo "generate" ;;
    *)       echo "none" ;;
  esac
}

deck_for() {
  if [[ -n "${NVRS_DECK:-}" ]]; then printf '%s\n' "$NVRS_DECK"; return; fi
  # SPE10 is the Flow-matched deck; see scripts/run_spe10_matched.sh.
  if [[ "$1" == SPE10 ]]; then printf '%s\n' "$CUDA_DIR/decks/SPE10_MATCHED.DATA"; return; fi
  # decks/ holds the ones that ship or are downloaded; make_decks.py writes
  # the generated BENCH family into benchmark/ instead, because that is where
  # run_comparison.sh reads them.  Both are real locations for a real deck,
  # and looking in only one reported a generated case as missing while its
  # deck sat a directory away.
  local c
  for c in "$CUDA_DIR/decks/$1.DATA" "$HERE/$1.DATA"; do
    if [[ -f "$c" ]]; then printf '%s\n' "$c"; return; fi
  done
  printf '%s\n' "$CUDA_DIR/decks/$1.DATA"
}

known_cases() {
  {
    for d in "$CUDA_DIR"/decks/*.DATA; do
      [[ -f "$d" ]] && basename "${d%.DATA}"
    done
    echo NORNE; echo SPE10
    echo BENCH2P; echo BENCH3P; echo BENCH2P_L; echo BENCH3P_L; echo BENCH2P_XL
  } | sort -u
}

cmd_cases() {
  step "Cases"
  printf '  %-14s %-8s %s\n' CASE READY "HOW IT IS PREPARED"
  printf '  %-14s %-8s %s\n' "--------------" "-----" "-------------------"
  local c
  for c in $(known_cases); do
    local ready="no" how
    [[ -f "$CUDA_DIR/decks/$c.DATA" ]] && ready="yes"
    case "$(prep_for "$c")" in
      fetch)    how="downloaded by get_norne.sh" ;;
      spe10)    how="downloaded and converted by make_spe10.py" ;;
      generate) how="generated by make_decks.py" ;;
      *)        how="ships with the repository" ;;
    esac
    printf '  %-14s %-8s %s\n' "$c" "$ready" "$how"
  done
  echo
  echo "  ./nvrs.sh all <CASE>              does everything for one of these"
  echo "  ./nvrs.sh all MYFIELD --deck ...  uses a deck of your own"
}

cmd_install() {
  step "Installing HYPRE, AmgX, Ginkgo, OPM Flow, Nsight, nvrs_cuda and nvrs_cuda_fp64"
  show_options install
  echo " Each step is skipped when its output is already there, so this is safe"
  echo " to re-run after a failure.  Nothing here needs root."
  echo
  "$HERE/install_all.sh"
  local rc=$?
  # install_all.sh ends with a "next:" of its own, naming run_norne.sh.  That
  # was right before this script existed and is now one instruction too many,
  # so the last word here is the one that fits the workflow the reader is in.
  echo
  rule
  if [[ $rc -eq 0 ]]; then
    bold " Installed.  Nothing has been run yet -- that is the next command:"
    echo
    echo "   ./nvrs.sh all NORNE        everything for one case, end to end"
    echo "   ./nvrs.sh all NORNE SPE10  several cases, in order"
    echo "   ./nvrs.sh cases            other cases, and whether they are ready"
    echo "   ./precision_fp64_vs_mixed.sh   mixed precision against all-FP64, Norne and SPE10"
    echo "   ./nvrs.sh --help           every command and option"
  else
    bold " Install reported a problem (exit $rc)."
    echo
    echo "   Re-run ./nvrs.sh install -- finished steps are skipped, so it"
    echo "   resumes rather than starting over."
  fi
  rule
  return $rc
}

cmd_get() {
  local deck
  deck="$(deck_for "$CASE")"
  if [[ -f "$deck" ]]; then
    echo "  [get] $CASE: $deck is already here"
    # Norne's deck ships but its dataset does not, so the deck alone is not
    # proof the case can run.
    if [[ "$CASE" == "NORNE" && ! -d "$CUDA_DIR/dataset/NORNE" ]]; then
      echo "  [get] but the field dataset is missing; fetching it"
      "$HERE/get_norne.sh" || die "get_norne.sh failed"
    fi
    return 0
  fi
  step "Preparing $CASE"
  case "$(prep_for "$CASE")" in
    fetch)    "$HERE/get_norne.sh" || die "get_norne.sh failed" ;;
    spe10)    ( cd "$HERE" && mkdir -p runs/SPE10 && \
                python3 make_spe10_opm.py runs/SPE10/SPE10_OPM.DATA ) || \
                die "make_spe10_opm.py failed" ;;
    generate) python3 "$HERE/make_decks.py" || die "make_decks.py failed" ;;
    *)        die "no deck at $deck, and $CASE is not one this script can make.
       ./nvrs.sh cases                       lists what it can
       ./nvrs.sh all $CASE --deck /path.DATA  uses your own" ;;
  esac
}

# SPE10 is run with the matched deck and method: decks/SPE10_MATCHED.DATA on one
# GPU, one run per backend, through scripts/run_spe10_matched.sh.  The runs
# land in runs/SPE10/nvrs_<arm>/ with a timing.json beside them in the same
# form run_norne.sh writes.
# The arms a case runs, and profiles.  Only Norne runs the Ginkgo arms (all
# five arms, each under Nsight); every other case -- SPE10, BENCH3P and the
# size ladder -- compares AmgX and HYPRE alone, because Ginkgo there was far
# slower and its Nsight profiles slower still.  Naming the arms (--backend,
# NVRS_BACKENDS, NVRS_ONLY) still runs exactly what was named.
case_backends() {
  if [[ "$CASE" != NORNE && $BACKENDS_NAMED -eq 0 ]]; then
    local b out=""
    for b in $BACKENDS; do
      [[ "$b" == ginkgo* ]] && continue
      # The native-FP64-operator arm runs on Norne and SPE10 only.
      [[ "$b" == amgx_fp64 && "$CASE" != SPE10 ]] && continue
      # So does the all-FP64 arm (FP64 multigrid as well as operator).
      [[ "$b" == amgx_fp64amg && "$CASE" != SPE10 ]] && continue
      # And the Tensor Core arm.
      [[ "$b" == amgx_tensorcore && "$CASE" != SPE10 ]] && continue
      out="${out:+$out }$b"
    done
    echo "$out"
  else
    echo "$BACKENDS"
  fi
}

run_spe10_matched() {
  local rd="$RUNS/SPE10" arm rc=0 arms
  arms="$(case_backends)"
  mkdir -p "$rd"
  # Runs and profiles of arms not in this set are from an earlier run; left
  # in place they would reach the report as if they belonged to this one.
  # Except when the arms were named: that ADDS arms to the earlier runs, and
  # deleting the others to run one alone lost them.
  local d
  export SPE10_KEEP_OTHERS="$BACKENDS_NAMED"
  for d in "$rd"/nvrs_* "$rd"/profile/*; do
    [[ -d "$d" ]] || continue
    [[ "$BACKENDS_NAMED" == 1 ]] && continue
    local a="${d##*/}"; a="${a#nvrs_}"
    [[ " $arms " == *" $a "* ]] || { rm -rf "$d"; echo " SPE10   : removed stale $d"; }
  done
  # Every arm: AmgX (the deck as committed), HYPRE and, when built in, the three
  # Ginkgo arms, each on the matched deck with only the solver lines changed, all on
  # one GPU.  Multi-GPU SPE10 is off: on two ranks the matched deck hangs after
  # the AmgX setup (CPRW's setup is not yet rank-safe).  SPE10_GPUS="1 2 4"
  # still asks for the sweep, for whoever fixes that.
  echo " SPE10   : the matched deck (decks/SPE10_MATCHED.DATA), 1 GPU, mpirun; arms: $arms"
  # SPE10 does not run OPM Flow by default: on 1.12 million cells it takes
  # far longer than every NVRS arm together.  A Flow run already in
  # runs/SPE10/opm is kept, and compare_SPE10.html is built against it;
  # without one there is no comparison report, only the NVRS runs, timings
  # and profiles.  SPE10_RUN_OPM=1 runs Flow after the NVRS arms.
  local run_flow=0
  # Leftovers that are not this method's runs: runs/SPE10/nvrs is the older
  # comparison script's layout (the 2000-day make_spe10.py deck), and a Flow
  # directory without SPE10_RUN_OPM's output is a Flow run of something else
  # (it stopped at day 100 on 8 ranks).  compare.py would pick both up.
  if [[ -d "$rd/nvrs" ]]; then rm -rf "$rd/nvrs"; echo " SPE10   : removed stale $rd/nvrs"; fi
  if [[ -d "$rd/opm" ]] && ! compgen -G "$rd/opm/SPE10_RUN_OPM.EGRID" >/dev/null 2>&1; then
    rm -rf "$rd/opm"; echo " SPE10   : removed stale $rd/opm (not the matched Flow run)"
  fi
  if [[ "${SPE10_RUN_OPM:-0}" == 1 ]]; then
    run_flow=1
    echo " OPM Flow: runs after the NVRS arm(s) above (SPE10_RUN_OPM=1; CPU, slow)"
  elif compgen -G "$rd/opm/*.EGRID" >/dev/null 2>&1; then
    echo " OPM Flow: not run (SPE10_RUN_OPM=1 runs it); keeping the earlier Flow run in $rd/opm for the report"
  else
    echo " OPM Flow: not run (SPE10_RUN_OPM=1 runs it); the report is report_SPE10.html (NVRS only)"
  fi
  for arm in $arms; do
    rm -rf "$rd/nvrs_$arm"
    GPUS="${SPE10_GPUS:-1}" SPE10_OUT="$rd/nvrs_$arm" NVRS_ARM="$arm" \
      "$CUDA_DIR/scripts/run_spe10_matched.sh" || rc=$?
  done
  python3 - "$rd" $arms <<'PY'
import glob, json, os, sys
rd, arms = sys.argv[1], sys.argv[2:]
out = os.path.join(rd, "timing.json")
t = {}
# Keep Flow's entry from an earlier run (--no-opm reuses its output), and,
# when the arms were named, every other arm's entry too.
if os.path.exists(out):
    try:
        old = json.load(open(out))
        if os.environ.get("SPE10_KEEP_OTHERS") == "1":
            t.update(old)
        elif "opm" in old:
            t["opm"] = old["opm"]
    except Exception:
        pass
KEYS = ("assembly_seconds", "linear_seconds", "amg_setup_seconds",
        "amg_solve_seconds", "stage2_seconds", "spmv_seconds", "ortho_seconds",
        "ijvec_seconds", "stage2_setup_seconds", "wall_seconds", "amg_setups",
        "amg_applications", "amg_cycles", "amg_reuses", "amg_reuse_aborts")
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
for arm in arms:
    metas = glob.glob(os.path.join(rd, "nvrs_" + arm, "1gpu", "OUTPUT", "**", "meta.json"),
                      recursive=True)
    if not metas:
        continue
    m = json.load(open(metas[0]))
    wall = open(os.path.join(rd, "nvrs_" + arm, "1gpu", "wall.txt")).read().strip()
    t["nvrs_" + arm] = dict(seconds=float(m.get("wall_seconds") or wall), ranks="1",
                           solver=DESC.get(arm, arm), hardware="GPU",
                           deck="decks/SPE10_MATCHED.DATA",
                           breakdown={k: m[k] for k in KEYS if m.get(k) is not None},
                           wall_including_output=float(wall))
json.dump(t, open(out, "w"), indent=2)
print("  wrote %s: %s" % (os.path.join(rd, "timing.json"),
      ", ".join("%s %.2f s" % (k, v["seconds"]) for k, v in sorted(t.items()))))
PY
  # OPM Flow on the same problem, through run_norne.sh's Flow path (the same
  # binary detection, solver and log parsing as Norne), for the speed-up and
  # the field, well and balance comparison.  Flow reads the Eclipse twin cut
  # to the NVRS deck's TEND: make_spe10_opm.py writes 2000 days of schedule
  # while SPE10_MATCHED.DATA stops at 1000, and a Flow run twice as long
  # would halve the speed-up.
  if [[ $run_flow != 1 ]]; then
    :
  else
    local field="$HERE/runs/SPE10/SPE10_OPM.DATA" flowdeck="$rd/SPE10_RUN_OPM.DATA"
    python3 - "$CUDA_DIR/decks/SPE10_MATCHED.DATA" "$field" "$flowdeck" <<'PY' || rc=1
import re, sys
nvrs, src, dst = sys.argv[1:4]
tend = None
for ln in open(nvrs):
    s = ln.split("--")[0].split()
    if len(s) >= 2 and s[0].upper() == "TEND":
        tend = float(s[1])
lines = open(src).read().split("\n")
out, i, cum = [], 0, 0.0
while i < len(lines):
    if lines[i].strip().upper() == "TSTEP":
        vals, j = [], i + 1
        while True:
            body = lines[j].split("--")[0]
            done = "/" in body
            for tok in body.replace("/", " ").split():
                n, _, v = tok.rpartition("*")
                vals += [float(v)] * (int(n) if n else 1)
            j += 1
            if done:
                break
        keep = []
        for v in vals:
            if tend is not None and cum + v > tend + 1e-9:
                break
            keep.append(v)
            cum += v
        if keep:
            out.append("TSTEP")
            out.append(" " + " ".join("%g" % v for v in keep) + " /")
        i = j
        continue
    out.append(lines[i])
    i += 1
open(dst, "w").write("\n".join(out))
print("  Flow deck : %s (schedule cut to %g days)" % (dst, cum))
PY
    # Flow as the matched comparison runs it: CPU, its default
    # (CPRW true-IMPES) linear solver with its own well operators, one MPI
    # rank, two threads.  Flow's GPU bridge takes blocksize 3 only and aborts
    # on this two-phase deck, so the accelerator is not offered, and without it
    # adding the wells to the matrix only slows CPRW down (about 20%).
    # OPM_EXTRA / OPM_LINSOLVER / OPM_ADD_WELL_CONTRIB still override.
    SKIP_OPM= NVRS_CASE=SPE10 NVRS_FLOW_ONLY=1 NVRS_SKIP_REPORT=1 \
      NVRS_DECK="$CUDA_DIR/decks/SPE10_MATCHED.DATA" OPM_DECK="$flowdeck" \
      OPM_RANKS="${OPM_RANKS:-1}" \
      OPM_EXTRA="${OPM_EXTRA:---threads-per-process=2}" \
      OPM_LINSOLVER="${OPM_LINSOLVER-}" \
      OPM_ADD_WELL_CONTRIB="${OPM_ADD_WELL_CONTRIB:-false}" \
      "$HERE/run_norne.sh" || rc=$?
  fi
  return $rc
}

cmd_run() {
  step "Running $CASE"
  show_options run
  if [[ "$CASE" == SPE10* ]]; then
    echo " OPM Flow: $([[ "${SPE10_RUN_OPM:-0}" == 1 ]] && echo 'yes, after the NVRS arms (SPE10_RUN_OPM=1)' || echo 'no, too slow on SPE10 (SPE10_RUN_OPM=1 to run it)')"
  elif [[ -n "${SKIP_OPM:-}" ]]; then
    echo " OPM Flow: skipped (--no-opm)"
  else
    echo " OPM Flow: yes, for comparison (--no-opm to leave it out)"
  fi
  echo " backends: $(case_backends)"
  # Strong scaling repeats the whole run on 2, 4 and however many GPUs are
  # present.  That is the right default for the BENCH family, which is small
  # and exists partly to measure decomposition.  It is the wrong one for
  # SPE10: 1.12 million cells that exist to stress the solver, where each
  # extra rank repeats a long run to produce a number the case was never
  # chosen to answer.  --scaling forces it on, --no-scaling off, for any case.
  if [[ -n "$SCALING" ]]; then
    export NVRS_SCALING="$SCALING"
    echo " scaling : $([[ "$SCALING" == 1 ]] && echo on || echo off) (from the command line)"
  else
    case "$CASE" in
      SPE10*)
        export NVRS_SCALING=0
        echo " scaling : off for SPE10, every arm on 1 GPU (SPE10_GPUS overrides)" ;;
      *)
        echo " scaling : on, up to the GPUs present (--no-scaling to skip)" ;;
    esac
  fi
  nvrs_libcheck || return 1
  nvrs_backendcheck || return 1
  echo
  # The runners take the backend list in NVRS_ONLY (a space-separated list).
  export NVRS_ONLY="$(case_backends)"
  if [[ "$CASE" == "NORNE" ]]; then
    "$HERE/run_norne.sh"
  elif [[ "$CASE" == SPE10* ]]; then
    run_spe10_matched
  else
    "$HERE/run_comparison.sh" "$CASE"
  fi
}

cmd_profile() {
  local b rc=0
  step "Profiling $CASE"
  show_options profile
  nvrs_libcheck || return 1
  nvrs_backendcheck || return 1
  # Profiles of arms this case no longer runs are from an earlier run; left in
  # place, profile_plots.py and the report would still show them.
  if [[ "$CASE" != NORNE && $BACKENDS_NAMED -eq 0 ]]; then
    local d
    for d in "$RUNS/$CASE"/profile/ginkgo*; do
      [[ -d "$d" ]] && { rm -rf "$d"; echo "  [profile] removed stale $d"; }
    done
  fi
  for b in $(case_backends); do
    step "Profiling $CASE with $b ($PROFILE_MODE)"
    # Norne goes through its own wrapper, which is where --steps is turned
    # into days.  profile_case.sh takes a fraction of the history and knows
    # nothing about report steps, so calling it directly here would accept
    # --steps and silently ignore it.
    if [[ "$CASE" == "NORNE" ]]; then
      NVRS_BACKEND="$b" "$HERE/profile_norne.sh" "$PROFILE_MODE" || rc=1
    else
      NVRS_BACKEND="$b" "$HERE/profile_case.sh" "$CASE" "$PROFILE_MODE" || rc=1
    fi
  done
  # A failed profile is not a failed benchmark.  The run's own numbers and the
  # field figures are already written, and the report is worth having without
  # the Nsight figures, so this is reported rather than fatal.
  [[ $rc -eq 0 ]] || \
    echo "  [profile] at least one profile failed; the report will omit its figures" >&2
  return 0
}

cmd_report() {
  step "Figures and report for $CASE"
  show_options report
  python3 "$HERE/profile_plots.py" --case "$CASE" || \
    echo "  [report] no profile figures; carrying on" >&2
  # compare.py builds a comparison, and a comparison needs two simulators. For
  # a case with no ECLIPSE deck there is only ever one, so "nothing to report"
  # is the correct answer rather than a failure, and failing the whole run on
  # it discards the profile figures that were just drawn.
  if [[ "$CASE" == SPE10 ]] && \
     ! compgen -G "$RUNS/SPE10/opm/SPE10_RUN_OPM.EGRID" >/dev/null 2>&1; then
    # No matched Flow run (SPE10 does not run Flow by default): the one-run
    # page instead -- rock, 3-D pressure and saturation with the animation,
    # wells, recovery and balance, timings and the log -- for the AmgX run.
    # A compare_SPE10.html left from an earlier, unmatched Flow run is removed
    # so it cannot be mistaken for this run's report.
    rm -f "$RUNS/compare_SPE10.html"
    local arm="${SPE10_REPORT_ARM:-amgx}" rdir="$RUNS/SPE10/nvrs_${SPE10_REPORT_ARM:-amgx}/1gpu"
    if [[ -d "$rdir" ]]; then
      python3 "$CUDA_DIR/scripts/spe10_report.py" "$rdir" \
        --out "$RUNS/report_SPE10.html" --log "$rdir/run.log" && {
        echo
        echo "  report: $RUNS/report_SPE10.html  (NVRS-CUDA $arm; no OPM Flow run)"
      }
    else
      echo "  [report] no $rdir; run ./nvrs.sh run SPE10 first" >&2
    fi
  elif python3 "$HERE/compare.py" --case "$CASE"; then
    echo
    echo "  report: $RUNS/compare_${CASE}.html"
  elif [[ ! -f "$HERE/${CASE}_OPM.DATA" ]]; then
    echo
    echo "  no comparison report for $CASE: it runs NVRS alone, and"
    echo "  compare.py puts two simulators side by side.  What it did produce:"
    echo "    profile figures : $RUNS/$CASE/profile/"
    echo "    run output      : $CUDA_DIR/OUTPUT_$CASE/"
    echo "    solver timings  : $RUNS/$CASE/timing.json"
  else
    die "compare.py failed"
  fi
}

cmd_all() {
  step "$CASE: get, run, profile, report"
  show_options all
  # A run that could not start makes every later stage meaningless: the profile
  # traces a process that exits at the dynamic linker, and the report is built
  # from nothing.  Both happened before this check existed.
  nvrs_libcheck || return 1
  nvrs_backendcheck || return 1
  # The runner builds the report too, and standalone that is right.  Inside
  # `all` it is the same page built twice -- once before the profile exists and
  # once after -- and the first is thrown away.  On BENCH3P that was 113
  # seconds of work whose only output was overwritten.
  export NVRS_SKIP_REPORT=1
  cmd_get
  cmd_run
  if [[ $DO_PROFILE -eq 1 ]]; then
    cmd_profile
  else
    echo; echo "  [all] profiling skipped (--no-profile)"
  fi
  cmd_report
  step "Done: $CASE"
  echo "  open $RUNS/compare_${CASE}.html"
}

usage() {
  rule
  bold " NVRS-CUDA  --  GPU reservoir simulation, against OPM Flow"
  rule
  echo
  bold "  USAGE"
  echo "    ./nvrs.sh <command> [CASE] [options]"
  echo
  bold "  COMMANDS"
  printf '    %-22s %s\n' "install"        "HYPRE, AmgX, Ginkgo, OPM Flow, Nsight and nvrs_cuda"
  printf '    %-22s %s\n' "cases"          "every case, and whether its deck is ready"
  printf '    %-22s %s\n' "all <CASE>"     "get + run + profile + report, in one command"
  printf '    %-22s %s\n' "get <CASE>"     "fetch or generate the deck and its data"
  printf '    %-22s %s\n' "run <CASE>"     "the backends, against OPM Flow"
  printf '    %-22s %s\n' "profile <CASE>" "Nsight Systems, once per backend"
  printf '    %-22s %s\n' "report <CASE>"  "the figures, folded into the HTML"
  echo
  bold "  OPTIONS"
  printf '    %-18s %-16s %s\n' "OPTION" "APPLIES TO" "MEANING"
  printf '    %-18s %-16s %s\n' "------" "----------" "-------"
  printf '    %-18s %-16s %s\n' "--no-opm" "run, all" "NVRS only; do not run OPM Flow"
  printf '    %-18s %-16s %s\n' "--backend <b>" "run, profile" "amgx hypre ginkgo ginkgo32 ginkgo_abj amgx_fp64 amgx_fp64amg (all FP64) amgx_tensorcore, a,b list, all (default: every one built)"
  printf '    %-18s %-16s %s\n' "--ranks <n>" "run, profile" "MPI ranks for NVRS"
  printf '    %-18s %-16s %s\n' "--no-profile" "all" "skip the profiling step"
  printf '    %-18s %-16s %s\n' "--no-scaling"    "run, all"      "one GPU only; skip the 2/4-GPU sweep"
  printf '    %-18s %-16s %s\n' "--scaling"       "run, all"      "force the sweep on (SPE10 is off)"
  printf '    %-18s %-16s %s\n' "--ncu" "profile, all" "add Nsight Compute counters (~100x slower)"
  printf '    %-18s %-16s %s\n' "--fraction <f>" "profile, all" "history to profile: 0.25, or all (default 0.1)"
  printf '    %-18s %-16s %s\n' "--tend <days>" "profile, all" "profile exactly this many days instead"
  printf '    %-18s %-16s %s\n' "--steps <n>" "profile, all" "NORNE only: report steps, or all"
  printf '    %-18s %-16s %s\n' "--heartbeat <s>" "profile, all" "progress interval, 0 to silence"
  printf '    %-18s %-16s %s\n' "--deck <path>" "get, run, all" "your own .DATA, for a case not under decks/"
  printf '    %-18s %-16s %s\n' "-h, --help" "anywhere" "this"
  echo
  dim "    Every command prints the options it takes when it starts, so this"
  dim "    list is a reference rather than something to memorise."
  echo
  bold "  FROM NOTHING TO A REPORT"
  echo "    ./nvrs.sh install"
  echo "    ./nvrs.sh all NORNE"
  echo "    ./nvrs.sh all NORNE SPE10 BENCH3P    (several, in order)"
  echo
  bold "  COMMON VARIATIONS"
  echo "    ./nvrs.sh all NORNE --no-opm              skip Flow, NVRS only"
  echo "    ./nvrs.sh all NORNE --backend amgx        one backend, faster"
  echo "    ./nvrs.sh all NORNE --backend ginkgo      the FP16 Ginkgo multigrid only"
  echo "    ./nvrs.sh all NORNE --backend ginkgo,ginkgo32,ginkgo_abj   the three Ginkgo arms"
  echo "    ./nvrs.sh all NORNE --steps all           the whole 247-step history"
  echo "    ./nvrs.sh all SPE10 --fraction 0.5        half of SPE10's history"
  echo "    ./nvrs.sh all MYFIELD --deck /path.DATA   a deck of your own"
  echo "    ./nvrs.sh profile NORNE --ncu             add the kernel counters"
  echo "    ./nvrs.sh report NORNE                    rebuild the HTML only"
  echo
  rule
  bold "  THIS MACHINE"
  local exe="$CUDA_DIR/nvrs_cuda"
  if [[ -x "$exe" ]]; then
    printf '    %-22s %s\n' "nvrs_cuda" "built"
  else
    printf '    %-22s %s\n' "nvrs_cuda" "NOT built  ->  ./nvrs.sh install"
  fi
  if [[ -x "$CUDA_DIR/nvrs_cuda_fp64" ]]; then
    printf '    %-22s %s\n' "nvrs_cuda_fp64" "built (all-FP64 preconditioner, for the comparison)"
  else
    printf '    %-22s %s\n' "nvrs_cuda_fp64" "NOT built  ->  ./nvrs.sh install"
  fi
  local ready=0 total=0 c
  for c in $(known_cases); do
    total=$((total + 1))
    [[ -f "$CUDA_DIR/decks/$c.DATA" ]] && ready=$((ready + 1))
  done
  printf '    %-22s %s\n' "decks" "$ready of $total ready  ->  ./nvrs.sh cases"
  if [[ -n "$NVRS_ENV_SOURCED" ]]; then
  printf '    %-22s %s\n' "environment" "sourced: $NVRS_ENV_SOURCED"
  else
  printf '    %-22s %s\n' "environment" "none found  ->  ./nvrs.sh install"
  fi
  local html
  html="$(ls -1t "$RUNS"/compare_*.html 2>/dev/null | head -1)"
  if [[ -n "$html" ]]; then
    printf '    %-22s %s\n' "latest report" "$html"
  fi
  rule
}

need_case() {
  [[ -n "$CASE" ]] || die "which case?  e.g. ./nvrs.sh $CMD NORNE
       ./nvrs.sh cases    lists them"
  local c
  for c in "${CASES[@]}"; do
    known_cases | grep -qx "$c" || die "unknown case '$c' (backends go after --backend, comma-separated)
       ./nvrs.sh cases    lists the cases"
  done
}

# Every per-case command runs once for each case named.  A failure on one is
# reported and the rest still run: with three cases queued, stopping at the
# first is rarely what was wanted, and the summary at the end says what
# happened to each.
run_over_cases() {
  local fn="$1" c failed=()
  for c in "${CASES[@]}"; do
    CASE="$c"
    "$fn" || failed+=("$c")
  done
  if [[ ${#CASES[@]} -gt 1 ]]; then
    step "Summary"
    for c in "${CASES[@]}"; do
      if [[ " ${failed[*]:-} " == *" $c "* ]]; then
        printf '  %-14s %s
' "$c" "FAILED"
      else
        printf '  %-14s %s
' "$c" "ok"
      fi
    done
  fi
  [[ ${#failed[@]} -eq 0 ]]
}

# `./nvrs.sh install all NORNE` is a reasonable thing to type and it used to
# install and then silently drop the rest.  Discarding arguments without
# saying so is worse than refusing them, and chaining is what was meant, so
# the trailing words are run as a second command once the install succeeds.
if [[ "$CMD" == "install" ]]; then
  cmd_install || exit $?
  NEXT="${POSITIONAL[1]:-}"
  case "$NEXT" in
    "")
      exit 0 ;;
    all|get|run|profile|report|cases|list)
      CMD="$NEXT"
      CASES=()
      for _c in "${POSITIONAL[@]:2}"; do
        CASES+=("$(echo "$_c" | tr '[:lower:]' '[:upper:]')")
      done
      CASE="${CASES[0]:-}"
      echo
      rule
      bold " Continuing: $CMD ${CASES[*]:-}"
      rule
      ;;
    *)
      die "install does not take '$NEXT'.
       To install and then run:   ./nvrs.sh install all NORNE
       To just run:               ./nvrs.sh all NORNE" ;;
  esac
fi

# The fastest working GPU-to-GPU transport for every backend (fast_comm.sh):
# NCCL where the library has it, AmgX's MPI_DIRECT where a 2-GPU probe passes.
if [[ "$CMD" == run || "$CMD" == all || "$CMD" == profile ]]; then
  # shellcheck source=fast_comm.sh
  . "$HERE/fast_comm.sh"
  nvrs_fast_comm "$CUDA_DIR/nvrs_cuda" "$RUNS" "$CUDA_DIR"
fi

case "$CMD" in
  install)           cmd_install ;;
  cases|list)        cmd_cases ;;
  get)               need_case; run_over_cases cmd_get ;;
  run)               need_case; run_over_cases cmd_run ;;
  profile)           need_case; run_over_cases cmd_profile ;;
  report)            need_case; run_over_cases cmd_report ;;
  all)               need_case; run_over_cases cmd_all ;;
  ""|help)           usage ;;
  *)                 echo "unknown command: $CMD" >&2; echo >&2; usage >&2; exit 2 ;;
esac
