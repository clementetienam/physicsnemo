#!/usr/bin/env bash
# Mixed precision against all-FP64: Norne, SPE10, BENCH3P and the MILLION family.
#
#   mixed  nvrs_cuda       the default: FP64 state, Jacobian, FGMRES and Newton,
#                          the CPR preconditioner (AmgX hierarchy and V-cycle,
#                          stage 2) in FP32 (AmgX mode dFFI)
#   fp64   nvrs_cuda_fp64  the same solver with the preconditioner in FP64 too
#                          (AmgX dDDI) and the native FP64 operator (EMULATE 0)
#
# The mixed run goes first and writes its accepted substeps (dt_sequence.txt);
# the FP64 run imposes them with DTSEQFILE, so both take the same time steps
# and the difference is precision, not a different nonlinear path.  Both
# binaries come from ./nvrs.sh install (or: make; make FP64_PRECOND=1).
#
# NORNE, SPE10 and BENCH3P run on one GPU.  MILLION runs the size ladder
# (1M, 2M, 4M, 5M cells, MILLION.DATA resized) on 1, 2 and 4 GPUs, both arms.
# Every run logs its Newton and linear residuals (NVRS_RESLOG), and
# precision_report.py draws the wall clock, scaling, residual and
# pressure/saturation comparisons.
#
#   ./precision_fp64_vs_mixed.sh                       # everything
#   ./precision_fp64_vs_mixed.sh NORNE SPE10           # only these
#   SIZES="1m 2m" GPUS="1 2" ./precision_fp64_vs_mixed.sh MILLION
#   REPEATS=3 ./precision_fp64_vs_mixed.sh NORNE       # median of three
#
# Results: runs/precision_<stamp>/RESULTS.txt, results.csv, fields.csv and
# plots/*.png, and runs/precision_<stamp>.tar.gz with those, the logs, decks,
# summaries and residual logs (no field output), the one file to send back.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CUDA_DIR/../.." && pwd)"
CASES=("$@"); [[ ${#CASES[@]} -eq 0 ]] && CASES=(NORNE SPE10 BENCH3P MILLION)
REPEATS="${REPEATS:-1}"
SIZES="${SIZES:-1m 2m 4m 5m}"
GPUS="${GPUS:-1 2 4}"
OUT="$HERE/runs/precision_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

for e in "$ROOT/Mixed_precision/hypre_env.sh" "$ROOT/amgx-install/amgx_env.sh"; do
  [[ -f "$e" ]] && . "$e" >/dev/null
done
MIXED_EXE="$CUDA_DIR/nvrs_cuda"
FP64_EXE="$CUDA_DIR/nvrs_cuda_fp64"
# Rebuild both: make is a no-op when they are current, and the residual log
# needs a build that has it.
( cd "$CUDA_DIR" && make -j"$(nproc)" >/dev/null && make -j"$(nproc)" FP64_PRECOND=1 >/dev/null ) ||
  { echo "ERROR: build failed (see ./nvrs.sh install)" >&2; exit 1; }
for x in "$MIXED_EXE" "$FP64_EXE"; do
  [[ -x "$x" ]] || { echo "ERROR: $x was not built (see ./nvrs.sh install)" >&2; exit 1; }
done

# NVRS always goes through mpirun, even on one GPU: a direct start fails under HPC-X.
MPIRUN=""
for c in "${MPIRUN_EXE:-}" "${OMPI_HOME:-/opt/openmpi}/bin/mpirun" "$(command -v mpirun 2>/dev/null || true)"; do
  [[ -n "$c" && -x "$c" ]] && { MPIRUN="$c"; break; }
done
[[ -n "$MPIRUN" ]] || { echo "ERROR: no mpirun on PATH" >&2; exit 1; }
RMPI=("$MPIRUN")
"$MPIRUN" --version 2>&1 | grep -qiE 'open[ -]?mpi|openrte' &&
  RMPI+=(--oversubscribe --allow-run-as-root -x NVRS_RESLOG -x NVRS_AUTOPLOT)
export NVRS_AUTOPLOT=0
NDEV="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
[[ -z "$NDEV" || "$NDEV" -lt 1 ]] && NDEV=1

deck_of() {
  local c
  case "$1" in
    NORNE)     echo "$CUDA_DIR/decks/NORNE.DATA"; return ;;
    SPE10)     echo "$CUDA_DIR/decks/SPE10_MATCHED.DATA"; return ;;
    MILLION_*) echo "$CUDA_DIR/decks/MILLION.DATA"; return ;;
  esac
  for c in "$CUDA_DIR/decks/$1.DATA" "$HERE/$1.DATA"; do
    [[ -f "$c" ]] && { echo "$c"; return; }
  done
  echo "$CUDA_DIR/decks/$1.DATA"
}
grid_of() { case "$1" in
  1m) echo "200 200 25" ;; 2m) echo "400 200 25" ;;
  4m) echo "400 400 25" ;; 5m) echo "500 400 25" ;;
esac; }
# set_kw DECK KEY VALUE: one line "KEY VALUE", replacing any earlier one.
set_kw() {
  local d="$1" k="$2"; shift 2
  sed -i -E "/^[[:space:]]*${k}([[:space:]]|$)/d" "$d"
  printf '%s %s\n' "$k" "$*" >> "$d"
}
# run CASE ARMDIR NP REP EXE [KEY=VALUE ...]
run() {
  local cs="$1" arm="$2" np="$3" rep="$4" exe="$5"; shift 5
  local d="$OUT/$cs/$arm/rep$rep" kv rc
  mkdir -p "$d"; cp "$(deck_of "$cs")" "$d/deck.DATA"
  set_kw "$d/deck.DATA" OUTDIR "$d/OUTPUT"
  set_kw "$d/deck.DATA" CASENAME "${cs}_${arm^^}"
  set_kw "$d/deck.DATA" AUTOPLOT 0
  set_kw "$d/deck.DATA" MAKESUMMARY 1
  set_kw "$d/deck.DATA" LINSOLVER AMGX
  for kv in "$@"; do set_kw "$d/deck.DATA" "${kv%%=*}" "${kv#*=}"; done
  ( cd "$CUDA_DIR" && NVRS_RESLOG="$d" "${RMPI[@]}" -np "$np" "$exe" "$d/deck.DATA" ) > "$d/run.log" 2>&1
  rc=$?
  echo "  $cs $arm rep$rep: rc=$rc, $(grep -oE 'simulation finished in [0-9.]+ s' "$d/run.log" | tail -1)"
}
# one line of results from a run directory
row() {
  local d="$1" log="$1/run.log" sum="$1/OUTPUT/summary.txt"
  local wall newt kry vc vcms
  wall=$(grep -oE 'simulation finished in [0-9.]+' "$log" | awk '{print $4}' | tail -1)
  newt=$(grep -oE 'Total Newton iters : [0-9]+' "$sum" 2>/dev/null | awk '{print $NF}')
  kry=$(grep -oE 'Total linear iters : [0-9]+' "$sum" 2>/dev/null | awk '{print $NF}')
  vc=$(grep -oE 'AMG V-cycles [0-9.]+ s' "$log" | awk '{print $3}' | tail -1)
  vcms=$(grep -oE '[0-9.]+ ms each' "$log" | awk '{print $1}' | tail -1)
  echo "${wall:-NA},${newt:-NA},${kry:-NA},${vc:-NA},${vcms:-NA}"
}
# pair CASE NP [KEY=VALUE ...]: the mixed run, then FP64 on its time steps
pair() {
  local cs="$1" np="$2"; shift 2
  local tag="" r seq_file
  [[ "$np" -gt 1 || "$cs" == MILLION_* ]] && tag="_${np}gpu"
  for r in $(seq 1 "$REPEATS"); do
    run "$cs" "mixed$tag" "$np" "$r" "$MIXED_EXE" MIXED=1 "$@"
    seq_file="$OUT/$cs/mixed$tag/rep1/OUTPUT/dt_sequence.txt"
    if [[ -f "$seq_file" ]]; then
      run "$cs" "fp64$tag" "$np" "$r" "$FP64_EXE" MIXED=0 EMULATE=0 DTSEQFILE="$seq_file" "$@"
    else
      echo "  !! $cs: no dt_sequence.txt from the mixed run; FP64 runs with its own time steps"
      run "$cs" "fp64$tag" "$np" "$r" "$FP64_EXE" MIXED=0 EMULATE=0 "$@"
    fi
    for a in mixed fp64; do
      echo "$cs,$a,$np,$r,$(row "$OUT/$cs/$a$tag/rep$r")" >> "$OUT/results.csv"
      # The ladder is about time, and 5M cells of snapshots is ~80 MB a step.
      [[ "$cs" == MILLION_* ]] && rm -f "$OUT/$cs/$a$tag/rep$r"/OUTPUT/state_*.bin
    done
  done
}

echo "precision comparison -> $OUT"
echo "  mixed: $MIXED_EXE"
echo "  fp64 : $FP64_EXE"
echo "  GPUs on this node: $NDEV"
echo "case,arm,gpus,rep,wall_s,newton,krylov,vcycle_total_s,vcycle_ms_each" > "$OUT/results.csv"
for cs in "${CASES[@]}"; do
  if [[ "$cs" == MILLION ]]; then
    for s in $SIZES; do
      read -r gx gy gz <<<"$(grid_of "$s")"
      [[ -n "$gx" ]] || { echo "  !! unknown size $s"; continue; }
      for g in $GPUS; do
        [[ "$g" -le "$NDEV" ]] || { echo "  !! MILLION_${s^^} on $g GPUs: node has $NDEV, skipped"; continue; }
        pair "MILLION_${s^^}" "$g" NX="$gx" NY="$gy" NZ="$gz"
      done
    done
    continue
  fi
  if [[ ! -f "$(deck_of "$cs")" && "$cs" == BENCH* ]]; then
    echo "  generating the BENCH decks (make_decks.py)"
    ( cd "$HERE" && python3 make_decks.py >/dev/null ) || true
  fi
  [[ -f "$(deck_of "$cs")" ]] || { echo "  !! no deck for $cs"; continue; }
  # ECLIPSE output for the field and well report (compare.py reads it)
  pair "$cs" 1 ECLOUT=1
done

# Tables, residual and scaling plots
python3 "$HERE/precision_report.py" "$OUT" || echo "  !! precision_report.py failed; the runs are still in $OUT"

# Field and well report: the same compare.py report as the OPM
# comparisons (3-D pressure and saturation panels, difference, movie, well
# curves and R2), with the all-FP64 run in the reference slot.
CMP="$OUT/compare"
for cs in "${CASES[@]}"; do
  [[ "$cs" == MILLION ]] && continue
  m="$OUT/$cs/mixed/rep1/OUTPUT"; f="$OUT/$cs/fp64/rep1/OUTPUT"
  [[ -d "$m" && -d "$f" ]] || continue
  mkdir -p "$CMP/$cs"
  ln -sfn "$f" "$CMP/$cs/opm"
  ln -sfn "$m" "$CMP/$cs/nvrs_amgx"
done
if [[ -d "$CMP" ]]; then
  NVRS_REF_LABEL="All FP64" NVRS_CMP_LABEL="Mixed precision"     python3 "$HERE/compare.py" --runs "$CMP" > "$OUT/compare.log" 2>&1 ||
    echo "  !! compare.py failed; see $OUT/compare.log"
fi
cat "$OUT/RESULTS.txt" 2>/dev/null

# One small archive to send: results, plots, every run's log, deck, summary,
# accepted substeps and residual logs; not the field output (state_*.bin).
( cd "$(dirname "$OUT")" && find "$(basename "$OUT")" -type f \
    \( -name RESULTS.txt -o -name results.csv -o -name fields.csv -o -name '*.png' \
       -o -name run.log -o -name deck.DATA -o -name summary.txt -o -name dt_sequence.txt \
       -o -name RSM_FVM.csv -o -name newton.csv -o -name linear.csv -o -name meta.json -o -name 'compare_*.html' -o -name compare.log \) -print0 |
  tar -czf "$(basename "$OUT").tar.gz" --null -T - )
echo
echo "results : $OUT/RESULTS.txt"
echo "plots   : $OUT/plots/"
echo "fields  : $OUT/compare/compare_<CASE>.html"
echo "archive : $OUT.tar.gz   <- send this one for the slides"
