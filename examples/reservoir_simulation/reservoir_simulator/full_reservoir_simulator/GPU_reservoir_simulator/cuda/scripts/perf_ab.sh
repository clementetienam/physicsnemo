#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# perf_ab.sh -- measure BLOCKSPMV and CUDAGRAPH on one deck, one GPU.
#
#   ./scripts/perf_ab.sh                       # Norne, all four variants
#   ./scripts/perf_ab.sh decks/NORNE.DATA      # any deck
#   VARIANTS="10 11" ./scripts/perf_ab.sh      # only these (BLOCKSPMV CUDAGRAPH)
#
# Each variant is the same binary and deck with the two switches set through
# NVRS_BLOCKSPMV and NVRS_CUDA_GRAPH, run back to back through run.sh (which
# launches with mpirun), with plotting off (NVRS_AUTOPLOT=0): about a minute
# per Norne run on GB200 instead of eight.  SKIP_BENCH=1 skips the
# micro-benchmark.  Also builds and runs tests/bspmv_bench.cu, the
# isolated kernel and graph measurement, so the report carries both the
# micro-benchmark and the end-to-end effect.
#
# Output: perf_ab/<deck>_<stamp>/{bench.txt, bs<B>_graph<G>.log, summary.txt}
set -euo pipefail
cd "$(dirname "$0")/.."

DECK="${1:-decks/NORNE.DATA}"
VARIANTS="${VARIANTS:-00 10 01 11}"
name=$(basename "$DECK" .DATA)
out="perf_ab/${name}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"

echo "== kernel micro-benchmark (tests/bspmv_bench.cu)"
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [[ "${SKIP_BENCH:-0}" == 1 ]]; then
  echo "   (skipped: SKIP_BENCH=1)"
elif nvcc -O3 -std=c++17 ${arch:+-arch=sm_$arch} -Iinclude -o "$out/bspmv_bench" \
     tests/bspmv_bench.cu -lcusparse -lcublas -w; then
  "$out/bspmv_bench" | tee "$out/bench.txt"
  "$out/bspmv_bench" 200 200 25 | tee -a "$out/bench.txt"
  "$out/bspmv_bench" 40 40 3 | tee -a "$out/bench.txt"
else
  echo "   (benchmark did not build; continuing with the end-to-end runs)"
fi

for v in $VARIANTS; do
  b=${v:0:1}; g=${v:1:1}
  log="$out/bs${b}_graph${g}.log"
  echo "== BLOCKSPMV $b, CUDAGRAPH $g  ->  $log"
  NGPU=1 NVRS_AUTOPLOT=0 NVRS_BLOCKSPMV=$b NVRS_CUDA_GRAPH=$g ./scripts/run.sh "$DECK" > "$log" 2>&1 || \
    echo "   run failed; see $log"
done

{
  printf "%-22s %12s %10s %10s %10s %10s %s\n" variant wall_s newton spmv_s stage2_s cgs_s operator
  for v in $VARIANTS; do
    b=${v:0:1}; g=${v:1:1}
    log="$out/bs${b}_graph${g}.log"
    wall=$(grep -o "simulation finished in [0-9.]*" "$log" | tail -1 | awk '{print $4}')
    newton=$(grep -o "of [0-9]* rebuilt" "$log" | tail -1 | awk '{print $2}')
    lin=$(grep "Linear   :" "$log" | tail -1)
    spmv=$(echo "$lin" | grep -o "operator+halo [0-9.]*" | awk '{print $2}')
    st2=$(echo "$lin" | grep -o "stage-2 [0-9.]*" | awk '{print $2}')
    cgs=$(echo "$lin" | grep -o "CGS[12] [0-9.]*" | awk '{print $2}')
    op=$(grep "Operator :" "$log" | tail -1 | sed 's/^ *Operator : //')
    printf "%-22s %12s %10s %10s %10s %10s %s\n" "BLOCKSPMV $b CUDAGRAPH $g" \
      "${wall:-?}" "${newton:-?}" "${spmv:-?}" "${st2:-?}" "${cgs:-?}" "$op"
  done
} | tee "$out/summary.txt"
echo "wrote $out/summary.txt"
