#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# emulate_ablation.sh -- what the df64 operator costs.
#
#   ./emulate_ablation.sh NORNE
#   ./emulate_ablation.sh SPE10 BENCH3P
#   NVRS_BACKEND=hypre ./emulate_ablation.sh NORNE
#   ABL_TEND=300 ./emulate_ablation.sh NORNE   # 300 days, not 3312
#
# Every result in the paper was produced with EMULATE 1, the emulated FP64
# operator, which accounts for about a third of wall clock. The paper argues
# that emulation cannot help a bandwidth-bound kernel, and then never reports
# the run that would test it. This runs both.
#
# Three outcomes, and each says something different:
#
#   df64 ~ native   the contribution is the FP32 preconditioner inside an FP64
#                   Krylov method, which is established practice, and emulation
#                   is a portability property rather than a performance one
#   df64 faster     the roofline argument is too strong as stated: the operator
#                   was closer to the FP64 unit than to memory
#   df64 slower     emulation is being paid for, and the reason to keep it is
#                   hardware with a narrow or absent FP64 vector unit
#
# Nothing here interprets the result. It runs the pair and prints them side by
# side with the counts needed to tell a faster solve from a shorter one: a run
# that takes fewer Newton iterations is not evidence about the operator.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
OUT="${OUT:-$HERE/runs/ablation}"
BACKEND="${NVRS_BACKEND:-amgx}"

# The same launcher every other script here uses.  A bare launch takes the
# Slurm step down; see profile_run.sh.
MPI_RUN=(mpirun --oversubscribe --allow-run-as-root)
command -v mpirun >/dev/null 2>&1 || { echo "no mpirun on PATH" >&2; exit 1; }

[[ $# -gt 0 ]] || { echo "usage: $0 <CASE> [CASE...]" >&2; exit 2; }

NVRS_EXE="$CUDA_DIR/nvrs_cuda"
[[ -x "$NVRS_EXE" ]] || { echo "no nvrs_cuda; run ./nvrs.sh install" >&2; exit 1; }

mkdir -p "$OUT"
printf '=%.0s' {1..78}; echo
echo " Emulated FP64 ablation: EMULATE 1 against EMULATE 0"
echo "   backend : $BACKEND"
echo "   cases   : $*"
echo "   out     : $OUT"
printf '=%.0s' {1..78}; echo

# One row per (case, setting).  Written as TSV so the paper's table can be
# generated rather than transcribed.
TSV="$OUT/ablation.tsv"
printf 'case\temulate\tseconds\tnewton\tkrylov\tamg_solve_s\toperator_s\n' > "$TSV"

for CASE in "$@"; do
  CASE="$(echo "$CASE" | tr '[:lower:]' '[:upper:]')"
  DECK=""
  for c in "$CUDA_DIR/decks/$CASE.DATA" "$HERE/$CASE.DATA"; do
    [[ -f "$c" ]] && { DECK="$c"; break; }
  done
  if [[ -z "$DECK" ]]; then
    echo "  [skip] no deck for $CASE" >&2
    continue
  fi

  for EMU in 1 0; do
    RD="$OUT/$CASE/emulate$EMU"
    mkdir -p "$RD"
    WORK="$RD/${CASE}_EMU$EMU.DATA"

    # EMULATE and LINSOLVER are both substituted when present and appended when
    # not.  A substitution that silently matches nothing is how a deck without
    # the keyword ends up running the default under the other setting's name.
    cp "$DECK" "$WORK"
    # NVRS_HYPRE_AMGFAST / NVRS_HYPRE_AMGREUSE reach a HYPRE run here too, as
    # in every other runner (see backend_arms.sh).
    kvs=("EMULATE $EMU" "LINSOLVER $(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')")
    if [[ "$BACKEND" == hypre ]]; then
      [[ -n "${NVRS_HYPRE_AMGFAST:-}" ]]  && kvs+=("AMGFAST $NVRS_HYPRE_AMGFAST")
      [[ -n "${NVRS_HYPRE_AMGREUSE:-}" ]] && kvs+=("AMGREUSE $NVRS_HYPRE_AMGREUSE")
    fi
    for kv in "${kvs[@]}"; do
      k="${kv%% *}"
      if grep -qE "^[[:space:]]*$k[[:space:]]" "$WORK"; then
        sed -i "s|^[[:space:]]*$k[[:space:]].*|$kv|" "$WORK"
      else
        printf '%s\n' "$kv" >> "$WORK"
      fi
      grep -qE "^[[:space:]]*$kv([[:space:]]|$)" "$WORK" || {
        echo "ERROR: could not set '$kv' in $WORK" >&2; exit 1; }
    done
    # Output elsewhere, so the two settings cannot overwrite each other.
    sed -i "s|^OUTDIR .*|OUTDIR   OUTPUT_${CASE}_EMU$EMU|" "$WORK"

    # ABL_TEND shortens both runs by the same amount.  The question here is the
    # cost of one operator against another, and that shows in a fraction of the
    # history as clearly as in all of it; Norne in full is about five minutes
    # per setting and there is no reason to pay it twice while deciding whether
    # the effect exists at all.  Both settings get the same TEND, so the
    # comparison stays like for like.
    if [[ -n "${ABL_TEND:-}" ]] && grep -qE "^[[:space:]]*TEND[[:space:]]" "$WORK"; then
      sed -i "s|^[[:space:]]*TEND[[:space:]].*|TEND      $ABL_TEND|" "$WORK"
      echo "   (shortened to TEND $ABL_TEND by ABL_TEND)"
    fi

    echo
    echo "-- $CASE, EMULATE $EMU --"
    LOG="$RD/run.log"
    ( cd "$CUDA_DIR" && "${MPI_RUN[@]}" -np 1 "$NVRS_EXE" "$WORK" ) 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
      echo "  [warn] $CASE EMULATE $EMU exited $rc; row omitted" >&2
      continue
    fi

    # Read the numbers out of the run's own summary rather than timing the
    # process: a shorter run that took fewer Newton iterations says nothing
    # about the operator, and the counts are what separate the two.
    python3 - "$LOG" "$CASE" "$EMU" "$TSV" <<'PY'
import re, sys
log, case, emu, tsv = sys.argv[1:5]
t = open(log, encoding="utf-8", errors="replace").read()

def find(pat, cast=float, default=""):
    m = re.search(pat, t)
    return cast(m.group(1).replace(",", "")) if m else default

secs   = find(r"simulation finished in\s+([0-9.]+)\s*s")
# The per-step table's last row carries the cumulative Newton and Krylov counts.
rows   = re.findall(r"^\s*\d+/\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\d+)\s+(\d+)\s", t, re.M)
newton = sum(int(a) for a, _ in rows) if rows else ""
krylov = sum(int(b) for _, b in rows) if rows else ""
amg    = find(r"AMG V-cycles\s+([0-9.]+)\s*s")
op     = find(r"operator\+halo\s+([0-9.]+)\s*s")
with open(tsv, "a", encoding="utf-8") as fh:
    fh.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (case, emu, secs, newton, krylov, amg, op))
print("   recorded: %s s, %s Newton, %s Krylov" % (secs, newton, krylov))
PY
  done
done

echo
printf '=%.0s' {1..78}; echo
echo " Result"
printf '=%.0s' {1..78}; echo
python3 - "$TSV" <<'PY'
import sys, collections
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8")][1:]
by = collections.defaultdict(dict)
for r in rows:
    if len(r) >= 7:
        by[r[0]][r[1]] = r
if not by:
    print("  no completed pairs"); sys.exit(0)
print("  %-10s %-9s %10s %8s %9s %11s" %
      ("case", "EMULATE", "seconds", "newton", "krylov", "operator s"))
for case, d in sorted(by.items()):
    for emu in ("1", "0"):
        if emu in d:
            r = d[emu]
            print("  %-10s %-9s %10s %8s %9s %11s" % (case, emu, r[2], r[3], r[4], r[6]))
    if "1" in d and "0" in d:
        try:
            a, b = float(d["1"][2]), float(d["0"][2])
            n1, n0 = int(d["1"][3]), int(d["0"][3])
        except ValueError:
            continue
        # Only comparable when the nonlinear problem was the same.  A different
        # Newton count means the two runs solved different sequences of linear
        # systems, and the wall clocks are not a measurement of the operator.
        note = "" if n1 == n0 else "   (Newton counts differ; not a like-for-like operator comparison)"
        print("  %-10s df64 / native wall clock: %.3f%s" % (case, a / b if b else float("nan"), note))
print()
print("  rows: %s" % sys.argv[1])
PY
