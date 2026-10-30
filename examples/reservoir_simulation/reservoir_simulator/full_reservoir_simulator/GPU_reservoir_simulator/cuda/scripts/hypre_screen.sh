#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# hypre_screen.sh -- HYPRE's AMGFAST x AMGREUSE on a field case, against AmgX.
#
#   CASE=SPE10 ./scripts/hypre_screen.sh     # decks/SPE10_MATCHED.DATA, TEND 100
#   CASE=NORNE ./scripts/hypre_screen.sh     # decks/NORNE.DATA,         TEND 336
#   CASE=MILLION ./scripts/hypre_screen.sh   # decks/MILLION.DATA (1M),  TEND 300
#   GRID="500 400 25" CASE=MILLION ./scripts/hypre_screen.sh   # the 5M rung
#   CASE=NORNE ./scripts/hypre_screen.sh amgx fast0_reuse10   # just these
#   TEND=660 CASE=NORNE ./scripts/hypre_screen.sh             # a longer slice
#
# Results, one GB200, 16 Sep 2026 (wall s / Krylov / Newton):
#
#   NORNE, 336 d   amgx 2.1/1971/59   fast1_reuse0 4.1/2500/59
#                  fast1_reuse10 4.0/3656/59 (deck)   fast0_reuse0 4.8/1167/59
#                  fast0_reuse10 4.6/1870/59          -> Norne keeps AMGFAST 1
#   MILLION, 300 d amgx 5.3/1599/351   fast1_reuse0 11.0/2644/135
#                  fast1_reuse10 6.3/2495/135 (deck)  fast0_reuse0 25.2/2408/351
#                  fast0_reuse10 9.1/2418/351         -> the ladder keeps AMGFAST 1
#   SPE10, 100 d   amgx 16.2/4626/173  fast1_reuse0 49.5/14532/257 (deck)
#                  fast1_reuse10 56.1/20944/343       fast0_reuse0 28.6/3603/193
#                  fast0_reuse10 23.4/5656/171        -> SPE10 HYPRE: AMGFAST 0
#
# The candidates set their own values, so NVRS_HYPRE_AMGFAST/_AMGREUSE must be
# unset while screening.
#
# Both decks set AMGFAST 1, which AmgX ignores and which gives BoomerAMG
# aggressive coarsening and a relaxed 6000-row coarse grid.  On SPE10's first
# 100 days that tripled HYPRE's Krylov count (49.5 s against 23.4 s without
# it).  Each candidate runs the deck's first TEND days on one GPU through
# mpirun, differing only in AMGFAST and AMGREUSE (appended last, so they win),
# and the table comes from each run's own meta.json.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
BENCH="$CUDA_DIR/benchmark"
. "$BENCH/backend_arms.sh"
CASE="$(echo "${CASE:-SPE10}" | tr '[:lower:]' '[:upper:]')"
EXE="${NVRS_EXE:-$CUDA_DIR/nvrs_cuda}"
[[ -x "$EXE" ]] || { echo "no $EXE; build first" >&2; exit 1; }

case "$CASE" in
  SPE10)
    DECK="$CUDA_DIR/decks/SPE10_MATCHED.DATA"
    TEND="${TEND:-100}"
    FIELD="$BENCH/runs/SPE10/SPE10_OPM.DATA"
    if [[ ! -f "$FIELD" ]]; then
      mkdir -p "$(dirname "$FIELD")"
      ( cd "$BENCH" && python3 make_spe10_opm.py "$FIELD" ) || exit 1
    fi
    ;;
  MILLION)
    # The size ladder's deck: 200 x 200 x 25 = 1M, or GRID="nx ny nz" for the
    # 2M/4M/5M rungs, as backend_study.sh sizes them.
    DECK="$CUDA_DIR/decks/MILLION.DATA"
    TEND="${TEND:-300}"
    ;;
  NORNE)
    DECK="$CUDA_DIR/decks/NORNE.DATA"
    TEND="${TEND:-336}"
    [[ -d "$CUDA_DIR/dataset/NORNE" ]] || \
      { echo "no dataset/NORNE; ./benchmark/nvrs.sh get NORNE" >&2; exit 1; }
    ;;
  *) echo "CASE must be SPE10, NORNE or MILLION" >&2; exit 2 ;;
esac
FD="$(sed -n 's/^[[:space:]]*FIELDDECK[[:space:]]\+//p' "$DECK" | head -1 | tr -d '\r' | awk '{print $1}')"
case "$FD" in /*) FDABS="$FD" ;; *) FDABS="$CUDA_DIR/$FD" ;; esac

TAG="$CASE"
[[ -n "${GRID:-}" ]] && TAG="${CASE}_$(echo "$GRID" | tr ' ' 'x')"
OUT="$CUDA_DIR/runs_hypre_screen/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
echo "=== $CASE${GRID:+ (grid $GRID)}, first $TEND days, one GPU ===" | tee "$SUMMARY"
printf '%-14s %-6s %8s %8s %16s %16s %8s %7s %6s\n' \
  CANDIDATE ARM WALL_s LINEAR_s "SETUP s (n)" "VCYCLE s (n)" KRYLOV NEWTON CHOPS \
  | tee -a "$SUMMARY"

run_one() {            # run_one <name> <arm> [KEYWORD value ...]
  local name="$1" arm="$2"; shift 2
  local rd="$OUT/$name"
  mkdir -p "$rd"
  sed -E "s|^([[:space:]]*FIELDDECK[[:space:]]+).*|\1$FDABS|" "$DECK" > "$rd/base.DATA"
  if [[ -n "${GRID:-}" ]]; then
    local gx gy gz
    read -r gx gy gz <<<"$GRID"
    sed -i -E "s|^([[:space:]]*NX[[:space:]]+).*|\1$gx|; s|^([[:space:]]*NY[[:space:]]+).*|\1$gy|; s|^([[:space:]]*NZ[[:space:]]+).*|\1$gz|" "$rd/base.DATA"
  fi
  arm_apply_deck "$rd/base.DATA" "$rd/deck.DATA" "$arm"
  rm -f "$rd/base.DATA"
  {
    printf '\n-- appended by hypre_screen.sh (last value wins) --\n'
    printf 'OUTDIR %s\nTEND %s\nECLOUT 0\nMAKECSV 0\nMAKESUMMARY 0\nAUTOPLOT 0\n' \
      "$rd/OUTPUT" "$TEND"
    local kv
    for kv in "$@"; do printf '%s\n' "$kv"; done
  } >> "$rd/deck.DATA"
  echo "-- $name ($arm): ${*:-deck as committed} --"
  ( cd "$CUDA_DIR" && "${MPIRUN:-mpirun}" -np 1 "$EXE" "$rd/deck.DATA" ) \
    > "$rd/run.log" 2>&1
  local meta
  meta="$(find "$rd/OUTPUT" -name meta.json 2>/dev/null | head -1)"
  python3 - "$name" "$arm" "${meta:-/nonexistent}" "$rd/run.log" <<'PY' | tee -a "$SUMMARY"
import json, sys
name, arm, path, log = sys.argv[1:5]
try:
    m = json.load(open(path))
except Exception:
    print("%-14s %-6s FAILED (no meta.json; see %s)" % (name, arm, log)); sys.exit(0)
def g(k):
    v = m.get(k) or 0
    return float(sum(v)) if isinstance(v, list) else float(v)
chops = sum(1 for ln in open(log, errors="replace") if "[chop]" in ln)
print("%-14s %-6s %8.1f %8.1f %9.1f (%5d) %9.1f (%5d) %8d %7d %6d" % (
    name, arm, g("wall_seconds"), g("linear_seconds"),
    g("amg_setup_seconds"), g("amg_setups"),
    g("amg_solve_seconds"), g("amg_applications"),
    g("linear_iters"), g("newton_iters"), chops))
PY
}

cands="${*:-amgx fast1_reuse0 fast1_reuse10 fast0_reuse0 fast0_reuse10}"
for c in $cands; do
  case "$c" in
    amgx)          run_one amgx          amgx ;;
    fast1_reuse0)  run_one fast1_reuse0  hypre "AMGFAST 1" "AMGREUSE 0" ;;
    fast1_reuse10) run_one fast1_reuse10 hypre "AMGFAST 1" "AMGREUSE 10" ;;
    fast0_reuse0)  run_one fast0_reuse0  hypre "AMGFAST 0" "AMGREUSE 0" ;;
    fast0_reuse10) run_one fast0_reuse10 hypre "AMGFAST 0" "AMGREUSE 10" ;;
    *) echo "unknown candidate: $c" >&2 ;;
  esac
done

echo
echo "=== $SUMMARY ==="
cat "$SUMMARY"
