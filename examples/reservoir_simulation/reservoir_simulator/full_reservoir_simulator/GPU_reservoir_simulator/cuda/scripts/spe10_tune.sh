#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# spe10_tune.sh — measured A/B tuning of the SPE10 run, one knob at a time.
#
# The full SPE10 model 2 run costs 861 s on one GB200 under the default
# configuration, and the phase breakdown says all of it is linear solver:
# ~130 Krylov iterations per Newton (vs ~15 on the synthetic ladder) and
# ~1,900 substeps (vs ECHELON's published 68).  Both symptoms point at the
# stage-1 multigrid being weak for THIS rock: SPE10 cells are 20 x 10 x 2 ft,
# so vertical transmissibility is ~100x horizontal, and plain aggregation
# AMG with a Jacobi smoother is the textbook loser on strong anisotropy.
# That is a hypothesis, and this script exists to test it rather than trust
# it: every candidate below changes exactly one thing against the baseline,
# runs the same 400-day slice (the hardest four report steps, including the
# dt collapse at step 4) on one GPU, and lands in one table.
#
#   ./scripts/spe10_tune.sh                # run the standard candidates
#   ./scripts/spe10_tune.sh base classical # just these two
#   TEND=200 ./scripts/spe10_tune.sh       # a faster, rougher slice
#
# Candidates (deck is identical unless stated):
#   base       the shipped configuration, as the reference row
#   ua_2x2     AmgX aggregation, 2 pre + 2 post smoothing sweeps
#   ua_size4   AmgX aggregation, SIZE_4 aggregates (slower coarsening)
#   ua_dilu    AmgX aggregation, MULTICOLOR_DILU smoother
#   classical  AmgX classical PMIS + D2 interpolation, strength 0.5
#              (the BoomerAMG-style hierarchy, built by AmgX on the device)
#   newton     deck only: NEWTONMAX 40, DSMAX 0.25, RESTART 60 — lets big
#              timesteps converge instead of chopping, and cheapens CGS2
#
# Results in runs_spe10_tune/<stamp>/<name>/run.log and a summary table.
# The AmgX configuration files live in benchmark/amgx_configs/ as plain
# key=value files (the same format as the compiled-in default; this AmgX
# build rejects the JSON form), loaded through NVRS_AMGX_CONFIG without a
# rebuild.  Each run streams to the console and is logged.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
BENCH="$CUDA_DIR/benchmark"
CFGS="$BENCH/amgx_configs"
TEND="${TEND:-400}"

EXE="${NVRS_EXE:-}"
if [[ -z "$EXE" ]]; then
  for c in "$CUDA_DIR/build/nvrs_cuda" "$CUDA_DIR/nvrs_cuda"; do
    [[ -x "$c" ]] || continue
    if [[ -z "$EXE" || "$c" -nt "$EXE" ]]; then EXE="$c"; fi
  done
fi
[[ -n "$EXE" && -x "$EXE" ]] || { echo "no nvrs_cuda binary; build first" >&2; exit 1; }

DECK="$BENCH/runs/SPE10/SPE10.DATA"
if [[ ! -f "$DECK" ]]; then
  mkdir -p "$BENCH/runs/SPE10"
  ( cd "$BENCH" && python3 make_spe10.py --out "$BENCH/runs/SPE10" ) || exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$CUDA_DIR/runs_spe10_tune/$STAMP"
mkdir -p "$OUT"
echo "=== SPE10 tuning: 1 GPU, TEND $TEND days, baseline-relative A/B ==="
echo "exe   : $EXE"
echo "out   : $OUT"
echo

run_one() {          # run_one <name> <amgx-config-or--> [extra keyword lines...]
  local name="$1" cfg="$2"; shift 2
  local rd="$OUT/$name"
  mkdir -p "$rd"
  sed -E "s|^([[:space:]]*OUTDIR[[:space:]]+).*|\1$rd/OUTPUT|" "$DECK" \
    > "$rd/deck.DATA"
  {
    printf '\n-- appended by spe10_tune.sh --\n'
    printf 'TEND %s\nAUTOPLOT 0\nMAKECSV 0\nMAKESUMMARY 0\nECLOUT 0\n' "$TEND"
    local kwline
    for kwline in "$@"; do printf '%s\n' "$kwline"; done
    printf 'LINSOLVER AMGX\n'
  } >> "$rd/deck.DATA"

  echo "-- $name --"
  # Streamed to the console AND logged: a five-minute run with its output
  # hidden in a file is indistinguishable from a hang, and gets killed.
  if [[ "$cfg" != "-" ]]; then
    [[ -f "$CFGS/$cfg" ]] || { echo "   !! no config $CFGS/$cfg" >&2; return; }
    NVRS_AMGX_CONFIG="$CFGS/$cfg" \
      mpirun -np 1 "$EXE" "$rd/deck.DATA" 2>&1 | tee "$rd/run.log"
  else
    mpirun -np 1 "$EXE" "$rd/deck.DATA" 2>&1 | tee "$rd/run.log"
  fi
  local rc=${PIPESTATUS[0]}

  # One row from the log: wall, Newton and Krylov totals, V-cycle cost.
  local wall newt kry vms row
  wall="$(awk '/simulation finished in/ {print $4}' "$rd/run.log")"
  read -r newt kry <<<"$(awk '
    /^[[:space:]]+[0-9]+[[:space:]]+[0-9.]+[[:space:]]+[0-9.e+-]+/ \
      { n += $6; k += $7 } END { print n+0, k+0 }' "$rd/run.log")"
  vms="$(sed -nE 's/.*AMG V-cycles.*\(([0-9]+) applies, ([0-9.]+) ms each\).*/\2/p' \
    "$rd/run.log" | head -1)"
  if [[ $rc -ne 0 || -z "$wall" ]]; then
    row="$(printf '%-10s  FAILED (rc=%s) — see %s/run.log' "$name" "$rc" "$name")"
    grep -im1 "amgx.*error\|AMGX_RC\|terminate\|what()" "$rd/run.log" | head -2
  else
    row="$(awk -v n="$name" -v w="$wall" -v nt="$newt" -v k="$kry" -v v="${vms:-?}" \
      'BEGIN { printf "%-10s  %8.1f s   %5d Newton   %7d Krylov   %5.1f Kry/Newt   V-cyc %s ms", \
               n, w, nt, k, (nt>0 ? k/nt : 0), v }')"
  fi
  echo "   $row"
  printf '%s\n' "$row" >> "$OUT/summary.txt"
}

want="${*:-base ua_2x2 ua_size4 ua_dilu classical newton}"
for c in $want; do
  case "$c" in
    base)      run_one base      -              ;;
    ua_2x2)    run_one ua_2x2    ua_2x2.cfg     ;;
    ua_size4)  run_one ua_size4  ua_size4.cfg   ;;
    ua_dilu)   run_one ua_dilu   ua_dilu.cfg    ;;
    classical) run_one classical classical.cfg  ;;
    newton)    run_one newton    -  "NEWTONMAX 40" "DSMAX 0.25" "RESTART 60" ;;
    # Round 2, composed from round 1's table: classical won at 2.4x (57 vs
    # 120 Kry/Newton); the newton row cut Newton count but RESTART 60 made
    # CGS2 more expensive per iteration, so the combo keeps RESTART 30.
    str025)    run_one str025    classical_str025.cfg ;;
    combo)     run_one combo     classical.cfg "NEWTONMAX 40" "DSMAX 0.25" ;;
    combo25)   run_one combo25   classical_str025.cfg "NEWTONMAX 40" "DSMAX 0.25" ;;
    # Two V-cycles per CPR application: doubles the stage-1 work per Krylov
    # iteration, and pays whenever it better-than-halves the iteration count.
    classical2v) run_one classical2v classical_2v.cfg ;;
    combo2v)     run_one combo2v     classical_2v.cfg "NEWTONMAX 40" "DSMAX 0.25" ;;
    # Round 3: the ZLINE stage 2 (vertical block-tridiagonal line relaxation,
    # a code feature, needs a binary built from this commit or later) against
    # the round-2 winner `combo`, one knob apart.
    zline)   run_one zline   classical.cfg "NEWTONMAX 40" "DSMAX 0.25" "STAGE2 ZLINE" ;;
    # Round 4: the Appleyard chop (code feature), one knob from `combo`.
    # The chop only pays if it lets larger timesteps converge, so `appds`
    # additionally loosens the per-iteration magnitude clip that otherwise
    # still caps how fast a front can move through a cell.
    app)     run_one app     classical.cfg "NEWTONMAX 40" "DSMAX 0.25" "APPLEYARD 1" ;;
    appds)   run_one appds   classical.cfg "NEWTONMAX 40" "DSMAX 0.5"  "APPLEYARD 1" ;;
    # Round 5: adaptive implicit (code feature, needs a binary from this
    # commit or later), one knob from the promoted `classical` winner.  The
    # [aim] lines print the explicit fraction per attempt; aim05/aim09 probe
    # the CFL threshold.
    aim)     run_one aim     classical.cfg "AIM 1" ;;
    aim09)   run_one aim09   classical.cfg "AIM 1" "AIMCFL 0.9" ;;
    aim03)   run_one aim03   classical.cfg "AIM 1" "AIMCFL 0.3" ;;
    *) echo "unknown candidate: $c" >&2 ;;
  esac
done

echo
echo "=== summary ($OUT/summary.txt) ==="
cat "$OUT/summary.txt" 2>/dev/null
echo
echo "Combine the winners into one run by editing the deck or adding a case"
echo "here; promote a winning AmgX config by pointing NVRS_AMGX_CONFIG at it"
echo "in scripts/backend_study.sh, or by making it the compiled default."
