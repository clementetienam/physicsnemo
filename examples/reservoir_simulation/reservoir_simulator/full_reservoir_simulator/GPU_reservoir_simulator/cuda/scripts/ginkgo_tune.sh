#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# ginkgo_tune.sh -- find the fastest Ginkgo multigrid settings on Norne.
#
#   ./scripts/ginkgo_tune.sh                      # every candidate, FP16 and FP32
#   ./scripts/ginkgo_tune.sh base mc2000 ci5      # just these
#   PRECS=16 ./scripts/ginkgo_tune.sh             # FP16 only
#   TEND=all ./scripts/ginkgo_tune.sh             # the whole history (default 336 days)
#   REFS=0 ./scripts/ginkgo_tune.sh               # skip the AmgX reference rows
#
# "base" is the backend's default (pairwise PGM aggregation, relaxation 0.9,
# 10 Jacobi sweeps on the coarsest level, coarsening down to 32 rows, the
# configuration of the 71 s full-Norne run); every other candidate changes
# one thing from it.
#
# Each candidate is one run of the same shortened Norne deck, on one GPU,
# differing only in NVRS_GINKGO_* environment variables (no rebuild), so the
# rows compare Ginkgo configurations and nothing else.  AmgX rows (its
# default classical hierarchy, and aggregation) are added as references.
#
# The table is read from each run's own meta.json: wall clock, linear solve,
# AMG setup (count and ms each), V-cycles (count and ms each), Krylov and
# Newton totals.  Promote a winner by exporting its variables before
# ./nvrs.sh run, or tell us and it becomes the default.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
BENCH="$CUDA_DIR/benchmark"
TEND="${TEND:-336}"
PRECS="${PRECS:-16 32}"
REFS="${REFS:-1}"
EXE="${NVRS_EXE:-$CUDA_DIR/nvrs_cuda}"
MPIRUN="${MPIRUN:-mpirun}"
[[ -x "$EXE" ]] || { echo "no $EXE; build first" >&2; exit 1; }
ldd "$EXE" 2>/dev/null | grep -qi ginkgo || { echo "$EXE has no Ginkgo; install_ginkgo.sh and rebuild" >&2; exit 1; }

DECK="$CUDA_DIR/decks/NORNE.DATA"
[[ -f "$DECK" ]] || { echo "no $DECK; ./benchmark/nvrs.sh get NORNE" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$CUDA_DIR/runs_ginkgo_tune/$STAMP"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
printf '%-12s %-5s %9s %9s %18s %20s %8s %7s\n' \
  CANDIDATE PREC WALL_s LINEAR_s "SETUP s (n, ms)" "VCYCLE s (n, ms)" KRYLOV NEWTON \
  | tee "$SUMMARY"

row_from_meta() {           # row_from_meta <name> <prec> <meta.json>
  python3 - "$@" <<'PY'
import json, sys
name, prec, path = sys.argv[1:4]
try:
    m = json.load(open(path))
except Exception:
    print("%-12s %-5s FAILED (no meta.json)" % (name, prec)); sys.exit(0)
def g(k):
    v = m.get(k) or 0.0
    # Per-step lists (linear_iters, newton_iters) are totalled.
    return float(sum(v)) if isinstance(v, list) else float(v)
ns, na = g("amg_setups"), g("amg_applications")
print("%-12s %-5s %9.2f %9.2f %8.2f (%4d,%5.1f) %9.2f (%5d,%5.2f) %8d %7d" % (
    name, prec, g("wall_seconds"), g("linear_seconds"),
    g("amg_setup_seconds"), ns, 1e3 * g("amg_setup_seconds") / max(ns, 1),
    g("amg_solve_seconds"), na, 1e3 * g("amg_solve_seconds") / max(na, 1),
    int(g("linear_iters")), int(g("newton_iters"))))
PY
}

run_one() {                 # run_one <name> <prec|-> <linsolver> [VAR=value ...]
  local name="$1" prec="$2" ls="$3"; shift 3
  local rd="$OUT/${name}_${prec}"
  mkdir -p "$rd"
  local fd
  fd="$(sed -n 's/^[[:space:]]*FIELDDECK[[:space:]]\+//p' "$DECK" | head -1 | tr -d '\r')"
  sed -E -e '/^[[:space:]]*(OUTDIR|LINSOLVER|GINKGOPREC|AUTOPLOT|MAKECSV|MAKESUMMARY|ECLOUT)([[:space:]]|$)/d' \
         "$DECK" > "$rd/deck.DATA"
  if [[ -n "$fd" && "${fd:0:1}" != "/" ]]; then
    sed -i "s|^[[:space:]]*FIELDDECK .*|FIELDDECK  $CUDA_DIR/$fd|" "$rd/deck.DATA"
  fi
  {
    printf '\n-- appended by ginkgo_tune.sh --\n'
    [[ "$TEND" != "all" ]] && printf 'TEND %s\n' "$TEND"
    printf 'OUTDIR %s\nAUTOPLOT 0\nMAKECSV 0\nMAKESUMMARY 0\nECLOUT 0\nLINSOLVER %s\n' \
      "$rd/OUTPUT" "$ls"
    [[ "$prec" != "-" ]] && printf 'GINKGOPREC %s\n' "$prec"
  } >> "$rd/deck.DATA"
  echo "-- $name ${prec} : $* --"
  ( cd "$CUDA_DIR" && env "$@" "$MPIRUN" -np 1 "$EXE" "$rd/deck.DATA" ) \
    > "$rd/run.log" 2>&1
  local meta
  meta="$(find "$rd/OUTPUT" -name meta.json 2>/dev/null | head -1)"
  row_from_meta "$name" "$prec" "${meta:-/nonexistent}" | tee -a "$SUMMARY"
  # Every Ginkgo line, including a fallback from the direct coarsest solve,
  # which changes what the row measured.
  grep "Ginkgo   :" "$rd/run.log" | sed 's/^/     /'
}

cands="${*:-base mc500 mc2000 ci5 ci20 sw2 relax07}"
for c in $cands; do
  for p in $PRECS; do
    case "$c" in
      base)     run_one base     "$p" GINKGO ;;
      mc500)    run_one mc500    "$p" GINKGO NVRS_GINKGO_MINCOARSE=500 ;;
      mc2000)   run_one mc2000   "$p" GINKGO NVRS_GINKGO_MINCOARSE=2000 ;;
      ci5)      run_one ci5      "$p" GINKGO NVRS_GINKGO_COARSE_ITERS=5 ;;
      ci20)     run_one ci20     "$p" GINKGO NVRS_GINKGO_COARSE_ITERS=20 ;;
      sw2)      run_one sw2      "$p" GINKGO NVRS_GINKGO_SWEEPS=2 ;;
      relax07)  run_one relax07  "$p" GINKGO NVRS_GINKGO_RELAX=0.7 ;;
      *) echo "unknown candidate: $c" >&2 ;;
    esac
  done
done
if [[ "$REFS" == 1 ]]; then
  run_one amgx     - AMGX
  run_one amgx_agg - AMGX NVRS_AMGX_ALGO=AGGREGATION
fi

echo
echo "=== $SUMMARY ==="
cat "$SUMMARY"
