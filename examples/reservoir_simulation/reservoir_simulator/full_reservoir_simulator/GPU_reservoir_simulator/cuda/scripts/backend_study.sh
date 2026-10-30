#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# backend_study.sh — HYPRE BoomerAMG against AmgX at 1, 2 and 4 GPUs, and
# Ginkgo's FP16 multigrid on one GPU as the third arm.
#
# Only the stage-1 pressure multigrid changes between the arms.  The outer
# FP64 FGMRES, the df64-emulated operator, the quasi-IMPES weights and the
# bs x bs second stage are identical.  HYPRE and AmgX run their multigrid in
# FP32; the Ginkgo arm runs the whole V-cycle in FP16 (GINKGOPREC 16), which
# is the precision ablation.  Ginkgo is single-GPU in this build, so its arm
# always runs on one rank whatever GPUS says.
#
#   ./scripts/backend_study.sh bench3p    # 46x112x22, vs OPM Flow, + physics
#   ./scripts/backend_study.sh million    # 200x200x25   =  1M cells
#   ./scripts/backend_study.sh 2m         # 400x200x25   =  2M
#   ./scripts/backend_study.sh 4m         # 400x400x25   =  4M
#   ./scripts/backend_study.sh 5m         # 500x400x25   =  5M
#   ./scripts/backend_study.sh spe10      # SPE10 model 2, 60x220x85 = 1.12M
#                                         #   (the Flow-matched deck and
#                                         #    method; every arm on ONE
#                                         #    GPU; see
#                                         #    run_spe10_matched.sh)
#   ./scripts/backend_study.sh all
#
# Every study runs its deck's whole schedule.  bench3p additionally runs OPM
# Flow once and renders the well curves, balances and panel series.
#
# Env:
#   GPUS="1 2 4"   rank counts to sweep
#   BACKENDS="amgx"  which arms to run (default "amgx hypre": the ladder,
#                  BENCH3P and SPE10 compare AmgX and HYPRE only; the Ginkgo
#                  arms run on Norne alone, see nvrs.sh).  One backend updates
#                  only its own scaling.json; the other arm's last results
#                  stay in place and the report keeps showing them.
#   OPM_SECONDS    Flow's simulation time for the bench3p chart.  Read from
#                  the newest benchmark/logs/BENCH3P_opm_*.log when unset,
#                  so the bar is traceable to a run rather than typed in.
#
# Results land in benchmark/backend_study/<study>/<backend>/scaling.json, which
# scripts/backend_report.py turns into the HTML.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
BENCH="$CUDA_DIR/benchmark"
OUT="$BENCH/backend_study"
GPUS="${GPUS:-1 2 4}"
# shellcheck source=../benchmark/backend_arms.sh
. "$BENCH/backend_arms.sh"
# AmgX and HYPRE.  The Ginkgo arms are Norne-only (far slower here, and
# single-GPU besides); BACKENDS="... ginkgo" still runs them when asked.
BACKENDS_NAMED=0
[[ -n "${BACKENDS:-}" ]] && BACKENDS_NAMED=1
BACKENDS="${BACKENDS:-amgx hypre}"
BACKENDS="${BACKENDS//,/ }"
# Ginkgo results from earlier runs would otherwise stay in the study tree and
# the report would keep drawing them.
if [[ $BACKENDS_NAMED -eq 0 ]]; then
  for _d in "$OUT"/*/ginkgo*; do
    [[ -d "$_d" ]] && { rm -rf "$_d"; echo "removed stale $_d"; }
  done
fi
# Where install_opm_gpu.sh leaves Flow, matching run_comparison.sh.
OPM_SRC="${OPM_SRC:-/workspace/project/opm_2024_10_src}"
# The unpacked dependency prefix, if install_opm_gpu.sh had to build one.
# flow links libumfpack from there, and libumfpack in turn needs libamd.so.2,
# which the loader will not find on its own:
#
#   flow: error while loading shared libraries: libamd.so.2
#
# RPATH does not solve this: OPM links with --enable-new-dtags, so the runpath
# covers flow's direct dependencies only, not what libumfpack needs in turn.
# LD_LIBRARY_PATH is searched for every object, so it is what works.  Same
# block as run_comparison.sh and run_norne.sh; this script was missing it, so
# the backend studies' Flow arm failed on any node where SuiteSparse had been
# unpacked without root.
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
# RUN_OPM=0 skips the Flow arm when only the two backends are wanted.
RUN_OPM="${RUN_OPM:-1}"

studies="${*:-all}"
[[ "$studies" == "all" ]] && studies="bench3p million 2m 4m 5m spe10"

# --- one log for the whole session ----------------------------------------
# Each arm already tees its own console.log, but a study spans many arms, an
# OPM Flow run and the report build, and the interesting question afterwards
# is usually "what happened, in order".  Re-run once under tee so that every
# line printed by anything below lands in a single file as well as on the
# screen.  STUDY_LOG overrides the location; NVRS_STUDY_TEE guards against
# recursing.
STUDY_LOG="${STUDY_LOG:-$OUT/study.log}"
if [[ -z "${NVRS_STUDY_TEE:-}" ]]; then
  mkdir -p "$(dirname "$STUDY_LOG")"
  export NVRS_STUDY_TEE=1
  {
    echo "=============================================================="
    echo " backend study: $studies"
    echo " started  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo " host     : $(hostname)"
    echo " GPUS     : $GPUS      BACKENDS: $BACKENDS"
    echo " command  : $0 $*"
    echo "=============================================================="
  } > "$STUDY_LOG"
  "$0" "$@" 2>&1 | tee -a "$STUDY_LOG"
  rc=${PIPESTATUS[0]}
  {
    echo
    echo "=============================================================="
    echo " finished : $(date '+%Y-%m-%d %H:%M:%S %Z')   exit status $rc"
    echo "=============================================================="
  } >> "$STUDY_LOG"
  echo
  echo "Full session log: $STUDY_LOG"
  exit "$rc"
fi

# --- OPM Flow, once, on one GPU -------------------------------------------
# The study is about two multigrids inside NVRS, so Flow runs once and only
# for bench3p: it is the reference the chart is measured against, not an arm
# that varies with rank count.  One rank because that is what the GPU linear
# solver supports, which is also how run_comparison.sh runs it.
#
# Same binary search, same flags and same accelerator probe as
# run_comparison.sh, so the number here and the number there mean the same
# thing.  Skipped without complaint if Flow is not installed -- the chart
# simply loses its reference bar.
run_opm() {                 # run_opm <deck-dir>
  # Announce before anything that could block, so a stall is attributable.
  echo
  echo "== OPM Flow: locating the binary and probing for a GPU solver =="
  local rd="$1" dest="$OUT/bench3p/opm"
  local deck="$rd/BENCH3P_OPM.DATA"
  mkdir -p "$dest/out"

  if [[ -z "${FLOW_EXE:-}" ]]; then
    local cand
    for cand in "$OPM_SRC/install/bin/flow" \
                "$OPM_SRC/opm-simulators/build/bin/flow" \
                "$(command -v flow || true)" /usr/bin/flow; do
      if [[ -n "$cand" && -x "$cand" ]]; then FLOW_EXE="$cand"; break; fi
    done
  fi
  if [[ -z "${FLOW_EXE:-}" || ! -x "${FLOW_EXE:-}" ]]; then
    echo "  !! OPM Flow not found; the chart will have no reference bar." >&2
    echo "     Build it with ./benchmark/install_opm_gpu.sh, or set FLOW_EXE." >&2
    return 0
  fi
  [[ -f "$deck" ]] || { echo "  !! no $deck" >&2; return 0; }

  # Does this Flow have a GPU linear solver?
  #
  # NOT by running "flow --version" directly.  Flow calls MPI_Init even for
  # --version, and on a cluster that does not allow singleton init that call
  # blocks in PMIx instead of returning -- which hangs this script before it has
  # printed a single line, with nothing to suggest why.  run_comparison.sh
  # already carries that warning in a comment and I walked into it anyway; this
  # is the detection it uses.
  #
  # ldd and config.h are the real evidence.  The probe run is confirmatory only,
  # goes through mpirun, is read for a specific complaint rather than for its
  # exit code, and is bounded by timeout so it can never hang the study again.
  local accel="" json="$BENCH/BENCH3P_solver_gpu.json" gpu=0 out=""
  ldd "$FLOW_EXE" 2>/dev/null | grep -qi libcusparse && gpu=1
  if [[ -f "$OPM_SRC/opm-simulators/build/config.h" ]] && \
     grep -qE "^[[:space:]]*#define[[:space:]]+HAVE_CUDA[[:space:]]+1" \
       "$OPM_SRC/opm-simulators/build/config.h"; then gpu=1; fi
  if [[ $gpu -eq 1 ]]; then
    out="$(timeout 120 mpirun --oversubscribe --allow-run-as-root -np 1 \
             "$FLOW_EXE" --accelerator-mode=cusparse --version \
             </dev/null 2>&1 || true)"
    grep -qiE "was not found by CMake|cusparseSolver was chosen, but|unknown value for parameter|AcceleratorMode" \
      <<<"$out" && gpu=0
  fi
  if [[ $gpu -eq 1 ]]; then
    accel="--accelerator-mode=${OPM_ACCELERATOR:-cusparse}"
  else
    json="$BENCH/BENCH3P_solver_cpu.json"
  fi
  [[ -f "$json" ]] || json=""

  echo
  echo "=============================================================="
  echo " bench3p / opm flow  (1 rank${accel:+, GPU linear solver})"
  echo "=============================================================="
  ( cd "$dest/out" && \
    mpirun --oversubscribe --allow-run-as-root -np 1 \
      "$FLOW_EXE" "$deck" \
      --parsing-strictness=low \
      --enable-ecl-output=true \
      --output-dir="$dest/out" \
      --matrix-add-well-contributions=true \
      ${json:+--linear-solver="$json"} \
      $accel ) 2>&1 | tee "$dest/flow.log"

  # Flow's own "Simulation time", not the wall clock, because NVRS's wall
  # column in this study is its simulation loop.  Startup and teardown are
  # excluded on both sides or the comparison is not like for like.
  local secs
  secs="$(awk '/Simulation time:/ {print $3; exit}' "$dest/flow.log")"
  if [[ -n "$secs" ]]; then
    printf '%s\n' "$secs" > "$dest/seconds.txt"
    echo "  -> OPM Flow: ${secs}s"
  else
    echo "  !! could not read a simulation time from $dest/flow.log" >&2
  fi
}

# --- the physics report ----------------------------------------------------
# The performance page answers "which is faster".  This answers "and does it
# give the same answer", which is the half that makes a speed-up worth quoting:
# well curves, water balance, and the per-step pressure and saturation panels,
# for each backend against OPM Flow on the same grid and schedule.
#
# compare.py already renders all of it.  It wants one directory per case
# holding the deck, an opm/ tree and an nvrs/ tree, each with an EGRID -- so
# this only has to arrange that layout around output the runs already wrote.
# Flow's output is shared by both cases through a symlink rather than copied:
# it is the same run, and a second copy is several hundred megabytes.
build_physics() {
  local src="$BENCH/runs/BENCH3P" phys="$OUT/bench3p/physics"
  local opm_out="$OUT/bench3p/opm/out"
  local any=0 be dir

  for be in $NVRS_ALL_ARMS; do
    [[ -d "$OUT/bench3p/$be/field" ]] || continue
    dir="$phys/BENCH3P_${be^^}"
    rm -rf "$dir"; mkdir -p "$dir/nvrs"
    cp -f "$src/BENCH3P.DATA" "$dir/" 2>/dev/null || true
    cp -rf "$OUT/bench3p/$be/field" "$dir/nvrs/OUTPUT_BENCH3P"
    if [[ -d "$opm_out" ]]; then ln -sfn "$opm_out" "$dir/opm"; fi
    any=1
  done

  if [[ $any -eq 0 ]]; then
    echo "  !! no NVRS field output captured; skipping the physics report" >&2
    return 0
  fi
  if [[ ! -d "$opm_out" ]]; then
    echo "  !! no OPM Flow output; the physics report needs both sides" >&2
    return 0
  fi

  echo
  echo "== rendering well curves, balances and panel series =="
  # One page, both backends.  Two files would make the reader hold one set of
  # curves in their head while scrolling the other, which is the comparison the
  # page exists to make.  No size budget: every figure and animation is
  # embedded at full resolution.
  # Optional profile of each arm, off unless asked for.  nsys costs about 1.2x
  # runtime; ncu replays every kernel it profiles and is roughly 100x slower per
  # kernel, so NVRS_PROFILE=1 gives the timeline and =both adds the counters.
  if [[ -n "${NVRS_PROFILE:-}" ]]; then
    pmode="$NVRS_PROFILE"; [[ "$pmode" == "1" ]] && pmode=nsys
    for pd in "$phys"/*/; do
      dk="$(ls "$pd"*.DATA 2>/dev/null | head -1)"
      if [[ -n "$dk" ]]; then
        "$BENCH/profile_run.sh" "$dk" "${pd%/}/profile" \
            "$(basename "${pd%/}")" "$pmode" || true
        python3 "$BENCH/profile_plots.py" --dir "${pd%/}/profile" || true
      fi
    done
  fi

  python3 "$BENCH/compare.py" --runs "$phys" \
    --combine "$OUT/bench3p/physics.html" 2>&1 | tee "$phys/compare.log"
}

# One arm: a deck, a backend, and whatever STEPS/GRID the study wants.
# run_scaling.sh already knows how to sweep rank counts and parse the logs, so
# this only has to call it with the right EXTRA and file the result.
run_arm() {           # run_arm <study> <backend> <deck> [extra deck keywords]
  local study="$1" backend="$2" deck="$3"; shift 3
  local dest="$OUT/$study/$backend"
  mkdir -p "$dest"
  echo
  echo "=============================================================="
  echo " $study / $backend"
  echo "=============================================================="
  # The arm's keywords last so they win over whatever the deck says.
  local extra="$*" gpus="$GPUS"
  arm_known "$backend" || { echo "unknown arm $backend" >&2; return 1; }
  extra="${extra:+$extra; }$(arm_extra "$backend")"
  # One GPU for the Ginkgo arms: the backend refuses several ranks rather than
  # run them wrongly.
  arm_single_gpu "$backend" && gpus="1"
  EXTRA="$extra" GPUS="$gpus" STEPS="${STEPS:-}" NVRS_ARM="$backend" \
    "${RUNNER:-$HERE/run_scaling.sh}" "$deck" 2>&1 | tee "$dest/console.log"
  # run_scaling.sh writes into its own timestamped directory; copy the summary
  # out so the report has one predictable path per arm.
  local newest
  newest="$(ls -1dt "$CUDA_DIR"/runs_scaling/*/ 2>/dev/null | head -1)"
  if [[ -n "$newest" && -f "$newest/scaling.json" ]]; then
    cp "$newest/scaling.json" "$dest/scaling.json"
    echo "  -> $dest/scaling.json"
  else
    echo "  !! no scaling.json produced for $study/$backend" >&2
  fi

  # Keep the 1-GPU field output: it is what the physics report is built from.
  # One GPU rather than the fastest rank count, because the answer must not
  # depend on how the grid was divided and the single-rank run is the one with
  # no domain decomposition to argue about.
  if [[ "$study" == "bench3p" && -n "$newest" && -d "$newest/1gpu/OUTPUT" ]]; then
    rm -rf "$dest/field"
    cp -r "$newest/1gpu/OUTPUT" "$dest/field"
    echo "  -> $dest/field  (1 GPU, for the physics report)"
  fi
}

for study in $studies; do
  case "$study" in
    bench3p)
      # The case that has an OPM Flow twin, so the three-way chart is possible.
      deck="$BENCH/runs/BENCH3P/BENCH3P.DATA"
      if [[ ! -f "$deck" ]]; then
        echo "== generating BENCH3P at 46 112 22 =="
        mkdir -p "$BENCH/runs/BENCH3P"
        ( cd "$BENCH" && python3 make_decks.py --out "$BENCH/runs/BENCH3P" \
            --grid 46 112 22 BENCH3P ) || exit 1
      fi
      unset STEPS
      [[ "$RUN_OPM" == "1" ]] && run_opm "$BENCH/runs/BENCH3P"
      for be in $BACKENDS; do run_arm bench3p "$be" "$deck"; done
      build_physics
      ;;
    million)
      # The deck's full schedule -- TEND 1000 at DTREPORT 100, so ten report
      # steps.  This used to stop after the first step to isolate the solver
      # from the timestep controller, but one step is not a simulation: it is
      # the step with all the chopping in it and none of the settled behaviour
      # that follows, so it flatters whichever backend happens to survive the
      # opening transient better.
      deck="$CUDA_DIR/decks/MILLION.DATA"
      unset STEPS
      for be in $BACKENDS; do run_arm million "$be" "$deck"; done
      ;;
    2m)
      deck="$CUDA_DIR/decks/MILLION.DATA"
      unset STEPS
      for be in $BACKENDS; do GRID="400 200 25" run_arm 2m "$be" "$deck"; done
      ;;
    4m)
      # A million rows per GPU at four ranks -- the size at which
      # distributing this problem starts to make sense at all, and the first
      # one where AmgX scales up rather than merely losing more slowly.
      deck="$CUDA_DIR/decks/MILLION.DATA"
      unset STEPS
      for be in $BACKENDS; do GRID="400 400 25" run_arm 4m "$be" "$deck"; done
      ;;
    5m)
      # Five million cells: 15M unknowns, and 1.25M rows per GPU at four
      # ranks.  The top of the ladder, and the point that says whether the
      # scaling AmgX finds at 4M keeps improving or levels off.
      deck="$CUDA_DIR/decks/MILLION.DATA"
      unset STEPS
      for be in $BACKENDS; do GRID="500 400 25" run_arm 5m "$be" "$deck"; done
      ;;
    spe10)
      # SPE10 model 2 with the matched deck and method: decks/SPE10_MATCHED.DATA,
      # the Flow-matched twin (97.7 s for 1000 days in the engineering log),
      # one GPU, launched directly.  scripts/run_spe10_matched.sh holds the
      # method and prints every change it makes to the deck (OUTDIR, the
      # absolute FIELDDECK, and the solver line for a non-AmgX arm).  The
      # older make_spe10.py deck is a different, slower problem and is no
      # longer what this study measures.
      deck="$CUDA_DIR/decks/SPE10_MATCHED.DATA"
      # AmgX and HYPRE, as every study here.  SPE10_BACKENDS overrides.
      spe10_default="$BACKENDS"
      # One GPU for every arm: on two ranks the matched deck hangs after the
      # AmgX setup (CPRW's setup is not yet rank-safe).  SPE10_GPUS overrides.
      spe10_gpus="${SPE10_GPUS:-1}"
      ndev="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
      for be in ${SPE10_BACKENDS:-$spe10_default}; do
        # SPE10_REUSE=1 takes runs already made this way through nvrs.sh rather than repeating them, provided
        # they cover every GPU count this study wants.
        src="$BENCH/runs/SPE10/nvrs_$be"
        want="$spe10_gpus"; arm_single_gpu "$be" && want="1"
        covered=0
        if [[ -f "$src/scaling.json" ]]; then
          python3 - "$src/scaling.json" "${ndev:-1}" $want <<'PY' && covered=1
import json, sys
have = {e["gpus"] for e in json.load(open(sys.argv[1]))}
nd = max(int(sys.argv[2] or 1), 1)
want = {int(g) for g in sys.argv[3:] if int(g) <= nd}
sys.exit(0 if want <= have else 1)
PY
        fi
        if [[ "${SPE10_REUSE:-0}" == 1 && $covered == 1 ]]; then
          mkdir -p "$OUT/spe10/$be"
          cp "$src/scaling.json" "$OUT/spe10/$be/scaling.json"
          echo "  spe10 / $be: reusing $src/scaling.json"
          continue
        fi
        RUNNER="$HERE/run_spe10_matched.sh" GPUS="$spe10_gpus" run_arm spe10 "$be" "$deck"
      done
      ;;
    *)
      echo "unknown study: $study" >&2
      echo "  want one of: bench3p million 2m 4m 5m spe10 all" >&2
      exit 1
      ;;
  esac
done

echo
echo "== building the report =="

# OPM Flow is not run here -- run_comparison.sh does that -- so its bar on
# the bench3p chart has to come from a previous comparison.  Read it from
# the newest Flow log rather than having someone type a number in, which is
# how a stale figure ends up in a chart with nothing to trace it to.
#
# "Simulation time" and not the wall clock: NVRS's wall column here is its
# simulation loop, so this is the like-for-like half of Flow's report.
if [[ -z "${OPM_SECONDS:-}" && -s "$OUT/bench3p/opm/seconds.txt" ]]; then
  OPM_SECONDS="$(cat "$OUT/bench3p/opm/seconds.txt")"
  echo "opm flow  : ${OPM_SECONDS}s, measured by this study"
fi
if [[ -z "${OPM_SECONDS:-}" ]]; then
  newest_opm="$(ls -1t "$BENCH"/logs/BENCH3P_opm_*.log 2>/dev/null | head -1)"
  if [[ -n "$newest_opm" ]]; then
    OPM_SECONDS="$(awk '/Simulation time:/ {print $3; exit}' "$newest_opm")"
    if [[ -n "$OPM_SECONDS" ]]; then
      echo "opm flow  : ${OPM_SECONDS}s, from $(basename "$newest_opm")"
    fi
  else
    echo "opm flow  : no BENCH3P_opm_*.log found; the Flow bar is omitted."
    echo "            run ./benchmark/run_comparison.sh BENCH3P first, or"
    echo "            pass OPM_SECONDS=<seconds> explicitly."
  fi
fi
python3 "$HERE/backend_report.py" "$OUT" ${OPM_SECONDS:+--opm "$OPM_SECONDS"}
