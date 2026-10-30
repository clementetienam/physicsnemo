#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# run_tensorcore.sh -- one deck two ways: JACOBIAN ANALYTIC, then JACOBIAN
# TENSORCORE with TCVALIDATE 1, each with nvrs_cuda_tc.  Prints the comparison
# the simulator itself reports.  Build first: TENSORCORE=1 ./nvrs.sh install
#
#   ./run_tensorcore.sh                 # SMOKE (two-phase) and RESERVOIR (three-phase)
#   ./run_tensorcore.sh NORNE           # any deck in decks/, by name
#   TCPRECISION=FP32 ./run_tensorcore.sh    # the product on CUDA cores instead of TF32
#
# SMOKE and RESERVOIR are Cartesian.  NORNE (three-phase) and SPE10_MATCHED
# (two-phase) are corner-point or field decks on the connection list, and need
# ./nvrs.sh get NORNE or ./nvrs.sh get SPE10 first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CUDA_DIR/../.." && pwd)"
EXE="$CUDA_DIR/nvrs_cuda_tc"
PREC="${TCPRECISION:-TF32}"

if [[ ! -x "$EXE" ]]; then
  echo "ERROR: $EXE is not built. Run: TENSORCORE=1 ./nvrs.sh install" >&2
  exit 1
fi
for e in "$ROOT/Mixed_precision/hypre_env.sh" "$ROOT/amgx-install/amgx_env.sh" \
         "$ROOT/ginkgo-install/ginkgo_env.sh"; do
  if [[ -f "$e" ]]; then
    set +u; source "$e" >/dev/null 2>&1; set -u
  fi
done

OUT="$HERE/runs/tensorcore"
mkdir -p "$OUT"
cases=("$@")
[[ ${#cases[@]} -gt 0 ]] || cases=(SMOKE RESERVOIR)

cd "$CUDA_DIR"
for name in "${cases[@]}"; do
  src="decks/$name.DATA"
  [[ -f "$src" ]] || { echo "ERROR: no deck $src" >&2; exit 1; }
  for mode in ANALYTIC TENSORCORE; do
    deck="$OUT/${name}_$mode.DATA"
    log="$OUT/${name}_$mode.log"
    cp "$src" "$deck"
    {
      echo
      echo "OUTDIR $OUT/OUTPUT_${name}_$mode"
      echo "JACOBIAN $mode"
      if [[ "$mode" == TENSORCORE ]]; then
        echo "TCVALIDATE 1"
        echo "TCPRECISION $PREC"
      fi
    } >> "$deck"
    echo "=== $name, JACOBIAN $mode"
    mpirun -n 1 "$EXE" "$deck" > "$log" 2>&1 || {
      echo "ERROR: the run failed. Last lines of $log:" >&2
      tail -20 "$log" >&2
      exit 1
    }
    grep -E 'Rates  |[Ww]all|\[tc\]|neighbour blocks|self blocks' "$log" || true
  done
done
echo
echo "Logs and output: $OUT"
