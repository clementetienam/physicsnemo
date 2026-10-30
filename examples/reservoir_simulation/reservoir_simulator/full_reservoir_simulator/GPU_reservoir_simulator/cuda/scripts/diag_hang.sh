#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# diag_hang.sh — find out where a multi-rank run is actually stuck.
#
# A run that stops producing output could be in MPI, in HYPRE, in a CUDA call,
# or simply doing arithmetic very slowly, and those need completely different
# fixes.  Guessing from the last line printed is how you end up fixing the
# wrong thing twice.  This launches the run, waits for its output to go quiet,
# then attaches gdb to every rank and dumps a backtrace.
#
#   ./diag_hang.sh                       # 2 ranks, decks/MILLION.DATA
#   ./diag_hang.sh 4 decks/MILLION.DATA  # 4 ranks
#   QUIET=60 ./diag_hang.sh              # call it stuck after 60 s of silence
#
# Reads nothing, writes only into a fresh directory under /tmp.
set -uo pipefail

RANKS="${1:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
DECK="${2:-$CUDA_DIR/decks/MILLION.DATA}"
EXE="${NVRS_EXE:-$CUDA_DIR/build/nvrs_cuda}"
QUIET="${QUIET:-90}"          # seconds of no new output before we call it stuck
OUT="$(mktemp -d /tmp/nvrs_hang.XXXXXX)"

command -v gdb >/dev/null 2>&1 || {
  echo "ERROR: gdb is not installed; nothing useful can be reported." >&2
  exit 1
}

echo "ranks   : $RANKS"
echo "deck    : $DECK"
echo "quiet   : ${QUIET}s of silence means stuck"
echo "output  : $OUT"
echo

sed -E "s|^([[:space:]]*OUTDIR[[:space:]]+).*|\1$OUT/OUTPUT|" "$DECK" \
  > "$OUT/deck.DATA"
sed -i -E 's|^([[:space:]]*AUTOPLOT[[:space:]]+).*|\10|' "$OUT/deck.DATA"
# shellcheck source=../benchmark/backend_arms.sh
. "$CUDA_DIR/benchmark/backend_arms.sh"
if [[ "$RANKS" -gt 1 ]] && deck_single_gpu "$OUT/deck.DATA"; then
  echo "deck selects Ginkgo, which runs on one GPU; ranks $RANKS -> 1"
  RANKS=1
fi

mpirun -np "$RANKS" "$EXE" "$OUT/deck.DATA" > "$OUT/run.log" 2>&1 &
MPIRUN_PID=$!

# --- wait for the output to stop changing ----------------------------------
last_size=-1
same_for=0
while kill -0 "$MPIRUN_PID" 2>/dev/null; do
  sleep 5
  size=$(stat -c %s "$OUT/run.log" 2>/dev/null || echo 0)
  if [[ "$size" == "$last_size" ]]; then
    same_for=$((same_for + 5))
  else
    same_for=0
    last_size=$size
  fi
  if [[ $same_for -ge $QUIET ]]; then
    echo "no output for ${same_for}s -- collecting backtraces"
    break
  fi
  printf '\r  running, log %s bytes, quiet %ss   ' "$size" "$same_for"
done
echo

if ! kill -0 "$MPIRUN_PID" 2>/dev/null; then
  echo "The run finished on its own; nothing was stuck."
  tail -20 "$OUT/run.log"
  exit 0
fi

# --- backtrace every rank --------------------------------------------------
PIDS="$(pgrep -f "$(basename "$EXE")" | tr '\n' ' ')"
echo "ranks found: $PIDS"
for p in $PIDS; do
  echo
  echo "================ pid $p ================"
  # thread apply all bt: the interesting frame is often not on the main thread,
  # since MPI progress and CUDA both use helpers.
  timeout 60 gdb -p "$p" -batch \
      -ex "set pagination off" \
      -ex "thread apply all bt 25" 2>/dev/null \
    | tee "$OUT/bt_$p.txt" \
    | grep -E "^(Thread|#[0-9]+)" | head -40
done

echo
echo "=============== summary ==============="
# Name the library each rank is sitting in; that is the answer we are after.
for f in "$OUT"/bt_*.txt; do
  [[ -f "$f" ]] || continue
  printf "%s: " "$(basename "$f")"
  if   grep -qiE "hypre_|HYPRE_"        "$f"; then echo "inside HYPRE"
  elif grep -qiE "mca_|opal_|MPI[A-Z_]" "$f"; then echo "inside MPI"
  elif grep -qiE "cuda|cu[A-Z]"         "$f"; then echo "inside CUDA"
  else echo "see $f"
  fi
done

kill -TERM "$MPIRUN_PID" 2>/dev/null
sleep 2
pkill -f "$(basename "$EXE")" 2>/dev/null
echo
echo "Backtraces and log in $OUT"
