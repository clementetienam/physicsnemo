#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# profile_case.sh -- profile any deck, not just Norne.
#
#   ./profile_case.sh SPE1                  # nsys, first 10% of the history
#   ./profile_case.sh SPE10 both            # add the kernel counters
#   ./profile_case.sh NORNE nsys            # the same as ./profile_norne.sh
#   NVRS_DECK=/path/to/MY.DATA ./profile_case.sh MYFIELD
#   NVRS_BACKEND=hypre ./profile_case.sh SPE1
#
# The case name decides two things: which deck is used, when NVRS_DECK does not
# name one, and where the output goes.  Everything else is the same machinery
# every case uses, which is the point -- one implementation, so a profile of a
# new field cannot quietly behave differently from a profile of Norne.
#
# How much of the history to profile:
#
#   PROF_TEND=<days>       explicit, and the only form that does not depend on
#                          the deck being readable
#   PROF_FRACTION=0.25     a quarter of the deck's own TEND (default 0.1)
#   PROF_FRACTION=all      the deck as written, however long that is
#
# A fraction rather than a step count, because a step count means nothing
# without knowing how long a step is, and that varies by deck: Norne's are a
# fortnight, SPE1's are not.  profile_norne.sh keeps PROF_STEPS for the deck
# where the answer is known.
#
# Then:  python3 profile_plots.py --case <CASE>
#        python3 compare.py --case <CASE>
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
RUNS="${RUNS:-$HERE/runs}"

CASE="${1:-}"
MODE="${2:-nsys}"

if [[ -z "$CASE" ]]; then
  echo "usage: $0 <CASE> [nsys|ncu|both]" >&2
  echo "       CASE names a deck at decks/<CASE>.DATA, or set NVRS_DECK" >&2
  echo >&2
  echo "available decks:" >&2
  for d in "$CUDA_DIR"/decks/*.DATA; do
    [[ -f "$d" ]] && echo "  $(basename "${d%.DATA}")" >&2
  done
  exit 2
fi

CASE="$(echo "$CASE" | tr '[:lower:]' '[:upper:]')"
LABEL="$(echo "$CASE" | tr '[:upper:]' '[:lower:]')"

# NVRS_ONLY is what run_norne.sh calls this, and people reasonably expect the
# same name here.
BACKEND="${NVRS_BACKEND:-${NVRS_ONLY:-amgx}}"
# NVRS_ONLY may be a list ("amgx hypre ginkgo"); a profile is one backend.
BACKEND="${BACKEND%% *}"
BACKEND="${BACKEND%%,*}"
# shellcheck source=backend_arms.sh
. "$HERE/backend_arms.sh"
arm_known "$BACKEND" || { echo "ERROR: unknown backend arm '$BACKEND' (want: $NVRS_ALL_ARMS)" >&2; exit 2; }
if arm_single_gpu "$BACKEND"; then
  # Single GPU in this build, whatever --ranks said; and let Ginkgo annotate
  # its own generate/apply calls so the timeline shows the hierarchy levels.
  export NVRS_RANKS=1
  export NVRS_GINKGO_NVTX="${NVRS_GINKGO_NVTX:-1}"
fi
# The deck is not always under decks/.  make_decks.py writes the BENCH family
# into benchmark/ beside run_comparison.sh, which is where that script reads
# them from, so a case can be perfectly runnable and still absent from decks/.
# Looking in both is the difference between profiling a generated case and
# telling the user their deck does not exist while it sits one directory away.
if [[ -n "${NVRS_DECK:-}" ]]; then
  DECK="$NVRS_DECK"
elif [[ "$CASE" == SPE10 ]]; then
  # The matched deck, as every SPE10 run uses (run_spe10_matched.sh).
  DECK="$CUDA_DIR/decks/SPE10_MATCHED.DATA"
else
  DECK=""
  for _cand in "$CUDA_DIR/decks/$CASE.DATA" "$HERE/$CASE.DATA"; do
    if [[ -f "$_cand" ]]; then DECK="$_cand"; break; fi
  done
  DECK="${DECK:-$CUDA_DIR/decks/$CASE.DATA}"
fi
# One directory per backend.  Shared, the second profile silently replaced the
# first, so two backends could never be compared from one report.
OUT="$RUNS/$CASE/profile/$BACKEND"

if [[ ! -f "$DECK" ]]; then
  echo "ERROR: no deck for $CASE.  Looked in:" >&2
  echo "         $CUDA_DIR/decks/$CASE.DATA" >&2
  echo "         $HERE/$CASE.DATA" >&2
  echo "       A generated case needs generating first:" >&2
  echo "         ./nvrs.sh get $CASE" >&2
  echo "       or name the deck outright: NVRS_DECK=/path/to/deck.DATA" >&2
  exit 1
fi

# --- how much of the history ------------------------------------------------
#
# PROF_TEND wins if it is set, because it says exactly what it means and needs
# nothing read out of the deck.  Otherwise a fraction of the deck's own TEND,
# which is the only deck-independent way to say "enough to see the kernel mix,
# not the whole run".
FRACTION="${PROF_FRACTION:-0.1}"
DECK_TEND="$(sed -n 's/^TEND[[:space:]]\+\([0-9.]*\).*/\1/p' "$DECK" | head -1)"

if [[ -n "${PROF_TEND:-}" ]]; then
  :
else
  case "${FRACTION,,}" in
    all|full|1|1.0)
      export PROF_TEND=""
      ;;
    *)
      if [[ -n "$DECK_TEND" ]]; then
        export PROF_TEND="$(awk -v t="$DECK_TEND" -v f="$FRACTION" \
          'BEGIN { v = t * f; if (v < 1) v = 1; printf "%d", v }')"
      else
        # No TEND in the deck means nothing to take a fraction of.  Profiling
        # the whole thing is the honest fallback: a made-up duration would be
        # silently wrong for every deck it did not fit.
        echo "  [profile] no TEND in $(basename "$DECK"); profiling it whole" >&2
        export PROF_TEND=""
      fi
      ;;
  esac
fi
export NVRS_BACKEND="$BACKEND"

printf '=%.0s' {1..78}; echo
echo " $CASE profile: ${MODE}"
echo "   backend : $BACKEND"
echo "   deck    : $DECK"
if [[ -z "${PROF_TEND:-}" ]]; then
  echo "   history : the deck's own, in full${DECK_TEND:+ (TEND $DECK_TEND)}"
else
  echo "   history : TEND $PROF_TEND days${DECK_TEND:+ of $DECK_TEND}" \
       "(PROF_FRACTION or PROF_TEND to change)"
fi
echo "   out     : $OUT"
printf '=%.0s' {1..78}; echo

# The backend is a deck keyword, so select it before profile_run.sh shortens
# the deck rather than editing the shortened copy afterwards.
WORKDECK="$DECK"
if [[ "$BACKEND" != "amgx" ]] || ! grep -qE "^[[:space:]]*LINSOLVER[[:space:]]+AMGX" "$DECK"; then
  mkdir -p "$OUT"
  WORKDECK="$OUT/${CASE}_$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]').DATA"
  # Drop the deck's own lines for these keywords and append the arm's.  A
  # plain substitution silently did nothing on a deck without LINSOLVER, and
  # SPE10's "hypre" profile then ran AmgX under the HYPRE label.
  arm_apply_deck "$DECK" "$WORKDECK" "$BACKEND"
  _LS="$(arm_keywords "$BACKEND" | awk '$1=="LINSOLVER"{print $2}')"
  # And confirm it took: this failing quietly produced two identical profiles
  # under two different labels.
  if ! grep -qE "^[[:space:]]*LINSOLVER[[:space:]]+$_LS" "$WORKDECK"; then
    echo "ERROR: could not select $_LS in $WORKDECK" >&2
    exit 1
  fi
  echo "   arm     : $(arm_label "$BACKEND"): $(arm_keywords "$BACKEND" | paste -sd' ' -)"
  # SPE10's HYPRE arm: AMGFAST 0, AMGREUSE 10, as scripts/run_spe10_matched.sh
  # runs it (measured by scripts/spe10_hypre_screen.sh).  The profile must
  # measure the same run.
  if [[ "$CASE" == SPE10 && "$BACKEND" == hypre ]]; then
    sed -i -E -e "s|^([[:space:]]*AMGREUSE[[:space:]]+).*|\1${NVRS_HYPRE_AMGREUSE:-10}|" \
              -e "s|^([[:space:]]*AMGFAST[[:space:]]+).*|\1${NVRS_HYPRE_AMGFAST:-0}|" "$WORKDECK"
    echo "   hypre   : AMGFAST ${NVRS_HYPRE_AMGFAST:-0}, AMGREUSE ${NVRS_HYPRE_AMGREUSE:-10} (the deck's values are AmgX's)"
  fi
  # FIELDDECK is relative to the deck's own directory, and the copy is not in
  # that directory.
  fd="$(sed -n 's/^[[:space:]]*FIELDDECK[[:space:]]\+//p' "$DECK" | head -1 | tr -d '\r')"
  if [[ -n "$fd" ]]; then
    case "$fd" in
      /*) abs="$fd" ;;
      *)  abs="$CUDA_DIR/$fd" ;;
    esac
    sed -i "s|^FIELDDECK .*|FIELDDECK  $abs|" "$WORKDECK"
  fi
fi

"$HERE/profile_run.sh" "$WORKDECK" "$OUT" "$LABEL" "$MODE"
rc=$?

echo
echo " next:"
echo "   python3 $HERE/profile_plots.py --case $CASE"
echo "   python3 $HERE/compare.py --case $CASE"
[[ "$MODE" == "nsys" ]] && echo "   (add the kernel counters with: $0 $CASE both)"
exit $rc
