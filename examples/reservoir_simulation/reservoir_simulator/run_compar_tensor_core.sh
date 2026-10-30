#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# run_compar_tensor_core.sh -- one deck, AmgX only, three ways: mixed precision,
# fully FP64, and the Tensor Core Jacobian.  Starts the container if needed, builds
# whatever is missing, runs all three and prints a table.
#
#   ./run_compar_tensor_core.sh                 # RESERVOIR, Cartesian, three-phase
#   ./run_compar_tensor_core.sh NORNE           # corner-point, three-phase
#   ./run_compar_tensor_core.sh SPE10_MATCHED   # corner-point path, two-phase
#   REPEATS=3 ./run_compar_tensor_core.sh       # each arm three times, median wall
#
# Logs and output: full_reservoir_simulator/GPU_reservoir_simulator/cuda/benchmark/runs/amgx_compare/
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(basename "${BASH_SOURCE[0]}")"
DECK="${1:-RESERVOIR}"
REPEATS="${REPEATS:-1}"

# Into the container first, unless already there or there is none on this machine.
if [[ -z "${NVRS_IN_CONTAINER:-}" ]] && { command -v docker >/dev/null 2>&1 || command -v podman >/dev/null 2>&1; }; then
  exec "$HERE/build_container.sh" run "REPEATS=$REPEATS ./$SELF $DECK"
fi

ROOT="$HERE/full_reservoir_simulator"
CUDA="$ROOT/GPU_reservoir_simulator/cuda"
BENCH="$CUDA/benchmark"
OUT="$BENCH/runs/amgx_compare"

# The container runs as root; give what it wrote back to whoever owns this folder.
give_back() {
  if [[ -n "${NVRS_IN_CONTAINER:-}" && "$(id -u)" == "0" ]]; then
    chown -R "$(stat -c %u:%g "$HERE/README.md")" "$HERE" 2>/dev/null || true
  fi
}
trap give_back EXIT

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$CUDA/decks/$DECK.DATA" ]] || die "no deck decks/$DECK.DATA"

# --- build what is missing ----------------------------------------------------
if [[ ! -x "$CUDA/nvrs_cuda" || ! -x "$CUDA/nvrs_cuda_tc" ]]; then
  echo "=== Building: TENSORCORE=1 ./nvrs.sh install"
  ( cd "$BENCH" && TENSORCORE=1 ./nvrs.sh install ) || die "the install failed; see above"
fi
[[ -x "$CUDA/nvrs_cuda" ]]    || die "nvrs_cuda did not build; see the install output"
[[ -x "$CUDA/nvrs_cuda_tc" ]] || die "nvrs_cuda_tc did not build; see the install output"

# --- decks with their own data ------------------------------------------------
case "$DECK" in
  NORNE)         ( cd "$BENCH" && ./nvrs.sh get NORNE ) || die "could not fetch Norne" ;;
  SPE10_MATCHED) ( cd "$BENCH" && ./nvrs.sh get SPE10 ) || die "could not generate SPE10" ;;
esac

for e in "$ROOT/Mixed_precision/hypre_env.sh" "$ROOT/amgx-install/amgx_env.sh" \
         "$ROOT/ginkgo-install/ginkgo_env.sh"; do
  # shellcheck disable=SC1090
  if [[ -f "$e" ]]; then source "$e" >/dev/null 2>&1; fi
done

# TF32 needs compute capability 8.0.  An older GPU runs the same product in FP32.
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
TCPREC=TF32
if [[ -n "$CAP" && "$CAP" -lt 80 ]]; then TCPREC=FP32; fi

mkdir -p "$OUT"
cd "$CUDA" || exit 1

median() { sort -n | awk '{a[NR]=$1} END{if (NR) print a[int((NR+1)/2)]}'; }

# --- the three runs -----------------------------------------------------------
declare -A WALL RATE
for arm in mixed fp64 tc; do
  case $arm in
    mixed) exe=nvrs_cuda;      extra=$'MIXED 1\n' ;;
    fp64)  exe=nvrs_cuda_fp64; extra=$'MIXED 0\nEMULATE 0\n' ;;
    tc)    exe=nvrs_cuda_tc;   extra=$'MIXED 1\nJACOBIAN TENSORCORE\nTCPRECISION '"$TCPREC"$'\nTCVALIDATE 0\n' ;;
  esac
  if [[ ! -x "./$exe" ]]; then
    echo "=== $arm: skipped, $exe is not built"
    continue
  fi
  walls=""
  for ((rep = 1; rep <= REPEATS; rep++)); do
    tag="${DECK}_${arm}_$rep"
    d="$OUT/$tag.DATA"
    log="$OUT/$tag.log"
    cp "decks/$DECK.DATA" "$d"
    printf '\nOUTDIR %s/OUTPUT_%s\nLINSOLVER AMGX\n%s' "$OUT" "$tag" "$extra" >> "$d"
    echo "=== $arm, run $rep of $REPEATS: mpirun -n 1 ./$exe $d"
    if ! mpirun -n 1 "./$exe" "$d" > "$log" 2>&1; then
      echo "    FAILED. Last lines of $log:"
      tail -15 "$log" | sed 's/^/    /'
      continue
    fi
    w="$(sed -n 's/.*simulation finished in \([0-9.]*\) s.*/\1/p' "$log" | tail -1)"
    if [[ -n "$w" ]]; then walls+="$w"$'\n'; fi
    RATE[$arm]="$(sed -n 's/.*Rates *: *\([0-9.]*\) Newton per accepted substep.*/\1/p' "$log" | tail -1)"
  done
  WALL[$arm]="$(printf '%s' "$walls" | median)"
done

# --- the table ----------------------------------------------------------------
BASE="${WALL[mixed]:-}"
{
  echo
  echo "AmgX only, deck $DECK, median of $REPEATS run(s), one GPU"
  printf '  %-34s %12s %10s %12s\n' "Run" "Wall, s" "vs mixed" "Newton/step"
  for arm in mixed fp64 tc; do
    case $arm in
      mixed) name="Mixed precision (FP32 AMG)" ;;
      fp64)  name="Fully FP64" ;;
      tc)    name="Tensor Core Jacobian ($TCPREC)" ;;
    esac
    w="${WALL[$arm]:-}"
    if [[ -z "$w" ]]; then
      printf '  %-34s %12s\n' "$name" "no result"
      continue
    fi
    ratio="-"
    if [[ -n "$BASE" ]]; then ratio="$(awk -v a="$w" -v b="$BASE" 'BEGIN{printf "%.2fx", a/b}')"; fi
    printf '  %-34s %12s %10s %12s\n' "$name" "$w" "$ratio" "${RATE[$arm]:-}"
  done
  tclog="$OUT/${DECK}_tc_$REPEATS.log"
  if [[ -f "$tclog" ]] && grep -q '\[tc\] seconds' "$tclog"; then
    echo
    echo "Tensor Core breakdown, last run. The analytic kernel still runs first, so the"
    echo "total includes it: fill + product + scatter is what the path adds."
    grep -E '\[tc\]' "$tclog" | sed 's/^ */  /'
  fi
  echo
  echo "Logs and output: $OUT"
} | tee "$OUT/summary_$DECK.txt"
