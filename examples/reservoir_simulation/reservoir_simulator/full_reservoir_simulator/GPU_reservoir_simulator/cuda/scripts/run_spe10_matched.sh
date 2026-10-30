#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# run_spe10_matched.sh -- SPE10 model 2 with the Flow-matched deck and method.
#
# The deck is decks/SPE10_MATCHED.DATA, unmodified: the Flow-matched two-phase
# twin (SWOF tables, fixed-BHP hydrostatics, CPRW, LINTOL 3e-2, LINMAXIT 300,
# RESTART 60, ORTHOPASSES 1, AMGREUSE 0, EMULATE 0, AIM 0, the learned
# timestep ceiling, 1000 days).  That is the 97.7 s run in
# docs/NORNE_ENGINEERING_LOG.md ("Report-boundary timestep state").  The older
# benchmark/runs/SPE10/SPE10.DATA from make_spe10.py is a different problem
# (Corey curves, 2000 days, LINTOL 1e-2, RESTART 30, AMGREUSE 10, EMULATE 1)
# and is several times slower; it is not what this script runs.
#
# What this script changes in the deck, and nothing else:
#   FIELDDECK  made absolute, so the run directory can be anywhere
#   OUTDIR     this run's own directory
#   solver     for a non-AmgX arm only: LINSOLVER (and GINKGOPREC / STAGE2),
#              through backend_arms.sh.  The AmgX arm is the deck verbatim.
#   HYPRE      HYPRE arm only: AMGFAST 0 and AMGREUSE 10 (NVRS_HYPRE_AMGFAST /
#              NVRS_HYPRE_AMGREUSE override, as for every HYPRE run).  Both deck
#              values were chosen with AmgX, which ignores AMGFAST.  For
#              BoomerAMG, AMGFAST 1 means aggressive coarsening and a relaxed
#              6000-row coarse grid, which on SPE10 is a weak preconditioner.
#              scripts/spe10_hypre_screen.sh, first 100 days, one GB200:
#                AmgX (deck)                 16.2 s  Krylov  4626  Newton 173
#                HYPRE AMGFAST 1 AMGREUSE 0  49.5 s  Krylov 14532  Newton 257
#                HYPRE AMGFAST 1 AMGREUSE 10 56.1 s  Krylov 20944  Newton 343
#                HYPRE AMGFAST 0 AMGREUSE 0  28.6 s  Krylov  3603  Newton 193
#                HYPRE AMGFAST 0 AMGREUSE 10 23.4 s  Krylov  5656  Newton 171
# The difference against the committed deck is printed before the run.
#
# Launch: from the cuda directory, always through mpirun (-np 1 included).
# GPUS="1 2 4" runs the strong-scaling sweep; Ginkgo arms stay on one GPU.
#
#   ./scripts/run_spe10_matched.sh              # AmgX, the deck as committed
#   NVRS_ARM=ginkgo ./scripts/run_spe10_matched.sh
#   SPE10_OUT=/some/dir ./scripts/run_spe10_matched.sh
#   GPUS="1 2 4" NVRS_ARM=hypre ./scripts/run_spe10_matched.sh
#
# Output: <SPE10_OUT>/<g>gpu/{deck.DATA,run.log,wall.txt,OUTPUT/} and
# <SPE10_OUT>/scaling.json, the layout run_scaling.sh writes, so backend_study.sh
# and the paper figures read it unchanged.  SPE10_OUT defaults to
# runs_scaling/<stamp>.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
BENCH="$CUDA_DIR/benchmark"
# shellcheck source=../benchmark/backend_arms.sh
source "$BENCH/backend_arms.sh"

ARM="${NVRS_ARM:-amgx}"
arm_known "$ARM" || { echo "unknown arm: $ARM" >&2; exit 1; }
DECK="$CUDA_DIR/decks/SPE10_MATCHED.DATA"
FIELD="$BENCH/runs/SPE10/SPE10_OPM.DATA"
OUT="${SPE10_OUT:-$CUDA_DIR/runs_scaling/$(date +%Y%m%d_%H%M%S)}"

EXE="${NVRS_EXE:-}"
if [[ -z "$EXE" ]]; then
  for c in "$CUDA_DIR/build/nvrs_cuda" "$CUDA_DIR/nvrs_cuda"; do
    [[ -x "$c" ]] || continue
    if [[ -z "$EXE" || "$c" -nt "$EXE" ]]; then EXE="$c"; fi
  done
fi
[[ -n "$EXE" && -x "$EXE" ]] || { echo "no nvrs_cuda binary; build first" >&2; exit 1; }
[[ -f "$DECK" ]] || { echo "no $DECK; git pull" >&2; exit 1; }

# His FIELDDECK: the Eclipse twin make_spe10_opm.py writes (it downloads the
# SPE10 rock on first use).
if [[ ! -f "$FIELD" ]]; then
  echo "== generating $FIELD (make_spe10_opm.py) =="
  mkdir -p "$(dirname "$FIELD")"
  ( cd "$BENCH" && python3 make_spe10_opm.py "$FIELD" ) || exit 1
fi

[[ -n "${EXTRA:-}" ]] && ! [[ "$EXTRA" == "$(arm_extra "$ARM")" ]] && \
  echo "note    : ignoring EXTRA=\"$EXTRA\"; SPE10 runs the matched deck as committed" >&2

# GPU counts.  GPUS="1 2 4" is the strong-scaling sweep the MILLION family
# gets; the default is one GPU.  Ginkgo arms always run on one: the backend
# refuses several ranks.  Counts above the GPUs present are skipped.
LIST=(${GPUS:-1})
arm_single_gpu "$ARM" && LIST=(1)
ndev="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
[[ -z "$ndev" || "$ndev" -lt 1 ]] && ndev=1

echo "=============================================================="
echo " SPE10 matched (decks/SPE10_MATCHED.DATA) -- $(arm_label "$ARM"), GPUs: ${LIST[*]}"
echo " exe     : $EXE"
echo " deck    : $DECK"
echo " output  : $OUT"
echo "=============================================================="

rc_all=0
for g in "${LIST[@]}"; do
  if [[ "$g" -gt "$ndev" ]]; then
    echo " skipping $g GPU(s): this node has $ndev" >&2
    continue
  fi
  RD="$OUT/${g}gpu"
  mkdir -p "$RD"
  tmp="$RD/deck.base"
  sed -E -e "s|^([[:space:]]*FIELDDECK[[:space:]]+).*|\1$FIELD|" \
         -e "s|^([[:space:]]*OUTDIR[[:space:]]+).*|\1$RD/OUTPUT|" \
         "$DECK" > "$tmp"
  if [[ "$ARM" == amgx ]]; then
    mv "$tmp" "$RD/deck.DATA"
  else
    arm_apply_deck "$tmp" "$RD/deck.DATA" "$ARM"
    rm -f "$tmp"
  fi
  if [[ "$ARM" == hypre ]]; then
    sed -i -E -e "s|^([[:space:]]*AMGREUSE[[:space:]]+).*|\1${NVRS_HYPRE_AMGREUSE:-10}|" \
              -e "s|^([[:space:]]*AMGFAST[[:space:]]+).*|\1${NVRS_HYPRE_AMGFAST:-0}|" \
      "$RD/deck.DATA"
  fi

  echo
  echo "-- $g GPU(s) --"
  echo " settings: $(grep -hE '^[[:space:]]*(LINSOLVER|LINTOL|LINMAXIT|RESTART|ORTHOPASSES|CPRW|AMGREUSE|EMULATE|MIXED|AIM|STAGE2|NEWTONTOL|CNVTOL|MBTOL|TEND)[[:space:]]' "$RD/deck.DATA" | awk '{printf "%s %s  ", $1, $2}')"
  echo " changes against the committed deck:"
  diff "$DECK" "$RD/deck.DATA" | sed 's/^/   /'

  rm -rf "$RD/OUTPUT"
  t0=$(date +%s.%N)
  # Always mpirun, one rank included.  A direct launch initialises MPI as a
  # singleton, which HPC-X in the cluster container cannot do:
  # "opal_init:startup:internal-failure ... MPI_Init on a NULL communicator".
  ( cd "$CUDA_DIR" && HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}" \
      "${MPIRUN:-mpirun}" -np "$g" "$(nvrs_pick_exe "$EXE" "$RD/deck.DATA")" "$RD/deck.DATA" ) 2>&1 | tee "$RD/run.log"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s.%N)
  awk "BEGIN{printf \"%.3f\n\", $t1-$t0}" > "$RD/wall.txt"
  echo " rc=$rc  wall=$(cat "$RD/wall.txt")s"
  if [[ $rc -ne 0 ]]; then
    echo "SPE10 ($ARM, $g GPU) failed; see $RD/run.log" >&2
    rc_all=$rc
  fi
done

python3 "$HERE/scaling_report.py" "$OUT"
exit $rc_all
