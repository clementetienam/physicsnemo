#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# run_spe10.sh -- the full SPE10 model 2 case, end to end.
#
# Generates the deck if it is not already there, runs the whole 2000-day
# schedule on one GPU, captures everything printed in one log, and renders
# benchmark/spe10/spe10.html: the rock it was given, how the flood developed,
# what the wells produced, where the time went, and the log itself.
#
#   ./scripts/run_spe10.sh                 # full run, 20 report steps
#   STEPS=4 ./scripts/run_spe10.sh         # a short look, 400 days
#   GPUS=1 ./scripts/run_spe10.sh          # rank count (1 is the default and
#                                          #   the configuration SPE10 is
#                                          #   reported on; see the manual)
#
# The run itself takes about six minutes on one GB200 for the full schedule.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
BENCH="$CUDA_DIR/benchmark"
DECKDIR="$BENCH/runs/SPE10"
OUT="$BENCH/spe10"
GPUS="${GPUS:-1}"

mkdir -p "$OUT"
LOG="$OUT/spe10_run.log"

# --- the binary, newest of the two build systems ---------------------------
EXE="${NVRS_EXE:-}"
if [[ -z "$EXE" ]]; then
  for c in "$CUDA_DIR/build/nvrs_cuda" "$CUDA_DIR/nvrs_cuda"; do
    [[ -x "$c" ]] || continue
    if [[ -z "$EXE" || "$c" -nt "$EXE" ]]; then EXE="$c"; fi
  done
fi
if [[ -z "$EXE" || ! -x "$EXE" ]]; then
  echo "ERROR: no nvrs_cuda binary found; build first." >&2
  exit 1
fi

# --- deck ------------------------------------------------------------------
if [[ ! -f "$DECKDIR/SPE10.DATA" ]]; then
  echo "== generating the SPE10 deck (downloads the dataset on first use) =="
  mkdir -p "$DECKDIR"
  ( cd "$BENCH" && python3 make_spe10.py --out "$DECKDIR" ) || exit 1
fi

# A shortened look is useful while iterating; the deck is copied rather than
# edited so the committed one always describes the full case.
RUNDECK="$OUT/SPE10_run.DATA"
sed -E "s|^([[:space:]]*OUTDIR[[:space:]]+).*|\1$OUT/OUTPUT_SPE10|" \
  "$DECKDIR/SPE10.DATA" > "$RUNDECK"

# The report reads the fields from the ECLIPSE output, so force it on here
# rather than relying on the deck: a deck generated before this was the
# default would otherwise run correctly and produce a report with no
# permeability, saturation or pressure figures in it.
if grep -qE '^[[:space:]]*ECLOUT' "$RUNDECK"; then
  sed -i -E 's|^([[:space:]]*ECLOUT[[:space:]]+).*|\11|' "$RUNDECK"
else
  printf '\nECLOUT 1\n' >> "$RUNDECK"
fi
# Well curves come from the CSV, and the timing table from meta.json.
for kw in MAKECSV MAKESUMMARY; do
  if grep -qE "^[[:space:]]*$kw" "$RUNDECK"; then
    sed -i -E "s|^([[:space:]]*$kw[[:space:]]+).*|\11|" "$RUNDECK"
  else
    printf '%s 1\n' "$kw" >> "$RUNDECK"
  fi
done
if [[ -n "${STEPS:-}" ]]; then
  dtr="$(awk '/^[[:space:]]*DTREPORT[[:space:]]/ {print $2; exit}' "$RUNDECK")"
  if [[ -n "$dtr" ]]; then
    new_end="$(awk "BEGIN{print $dtr*$STEPS}")"
    sed -i -E "s|^([[:space:]]*TEND[[:space:]]+).*|\1$new_end|" "$RUNDECK"
    echo "note: shortened to $STEPS report steps (TEND $new_end days)"
  fi
fi

rm -rf "$OUT/OUTPUT_SPE10"

{
  echo "=============================================================="
  echo " SPE10 model 2 -- full run"
  echo " started : $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo " host    : $(hostname)"
  echo " exe     : $EXE"
  echo " deck    : $RUNDECK"
  echo " ranks   : $GPUS"
  echo "=============================================================="
} | tee "$LOG"

t0=$(date +%s.%N)
# Always mpirun, one rank included.  A direct launch initialises MPI as a
# singleton, which HPC-X in the cluster container cannot do ("MPI_Init on a
# NULL communicator").
. "$CUDA_DIR/benchmark/backend_arms.sh"
if [[ "$GPUS" -gt 1 ]] && deck_single_gpu "$RUNDECK"; then
  echo "deck selects Ginkgo, which runs on one GPU; ranks $GPUS -> 1" | tee -a "$LOG"
  GPUS=1
fi
HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}" \
  mpirun -np "$GPUS" "$(nvrs_pick_exe "$EXE" "$RUNDECK")" "$RUNDECK" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
t1=$(date +%s.%N)
wall="$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")"

{
  echo
  echo "=============================================================="
  echo " finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo " exit    : $rc"
  echo " wall    : ${wall}s (including start-up and output)"
  echo "=============================================================="
} | tee -a "$LOG"

if [[ $rc -ne 0 ]]; then
  echo "run failed; see $LOG" >&2
  exit "$rc"
fi

# --- the report ------------------------------------------------------------
echo
echo "== rendering the report =="
python3 "$HERE/spe10_report.py" "$OUT" --out "$OUT/spe10.html" --log "$LOG"

echo
echo "Report : $OUT/spe10.html"
echo "Log    : $LOG"
echo "Fields : $OUT/OUTPUT_SPE10"
