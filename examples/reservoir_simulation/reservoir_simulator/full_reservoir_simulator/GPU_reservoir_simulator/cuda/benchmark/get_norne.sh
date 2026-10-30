#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# get_norne.sh -- fetch the Norne field model on demand.
#
# Norne is the Norwegian Sea field Equinor released as an open benchmark: a
# 46 x 112 x 22 corner-point grid with 44,927 active cells, 61 named faults,
# five equilibration regions, and nine years of well history over 247 dated
# report steps.  It is the reference case for NVRS-CUDA's field-deck path --
# faults, non-neighbour connections, horizontal and vertical wells, and
# time-varying well control.
#
# The full model is about 50 MB across sixty files, which is more than belongs
# in a git repository, so it is downloaded when first needed -- the same
# arrangement SPE10 uses in make_spe10.py.
#
#   ./get_norne.sh                    # into cuda/dataset/NORNE
#   ./get_norne.sh --out /scratch/n   # somewhere else
#   ./get_norne.sh --grid-only        # just what the grid reader needs
#   ./get_norne.sh --check            # verify an existing copy, download nothing
#
# Files already present and non-empty are left alone, so re-running is cheap
# and an interrupted download resumes rather than starting over.  Each file is
# written to a .part name and renamed only once it has arrived, so a killed
# download can never leave a truncated file that looks complete.
set -uo pipefail

BASE="https://raw.githubusercontent.com/OPM/opm-data/master/norne"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../dataset/NORNE"
GRID_ONLY=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)       OUT="$2"; shift 2 ;;
    --grid-only) GRID_ONLY=1; shift ;;
    --check)     CHECK_ONLY=1; shift ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *)           echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# The file list, in the layout the deck's INCLUDE paths expect.
# ---------------------------------------------------------------------------

# Everything the grid, schedule and equilibration readers actually consume.
CORE=(
  "NORNE_ATW2013.DATA"
  "INCLUDE/GRID/IRAP_1005.GRDECL"
  "INCLUDE/GRID/ACTNUM_0704.prop"
  "INCLUDE/PETRO/PORO_0704.prop"
  "INCLUDE/PETRO/PERM_0704.prop"
  "INCLUDE/PETRO/NTG_0704.prop"
  "INCLUDE/PETRO/MULTZ_HM_1.INC"
  "INCLUDE/PETRO/MULTZ_JUN_05_MOD.INC"
  "INCLUDE/PETRO/FLUXNUM_0704.prop"
  "INCLUDE/PETRO/MULTREGT_D_27.prop"
  "INCLUDE/PETRO/EQLNUM_0704.prop"
  "INCLUDE/PETRO/FIPNUM_0704.prop"
  "INCLUDE/PETRO/E3.prop"
  "INCLUDE/PETRO/SWINITIAL.INC"
  "INCLUDE/FAULT/FAULT_JUN_05.INC"
  "INCLUDE/FAULT/FAULTMULT_AUG-2006.INC"
  "INCLUDE/PVT/PVT-WET-GAS.INC"
  "INCLUDE/RELPERM/SCAL_NORNE.INC"
  "INCLUDE/BC0407_HIST01122006.SCH"
)

# Referenced by the deck but not modelled by these readers: tubing tables,
# productivity multipliers, summary requests, extra regions.  Fetched anyway
# so the copy on disk is the whole published model rather than a subset that
# would have to be topped up later.
EXTRA=(
  "INCLUDE/PETRO/EXTRA_REG.inc"
  "INCLUDE/PI/pimultab_low-high_aug-2006.inc"
  "INCLUDE/SUMMARY/summary.data"
  "INCLUDE/SUMMARY/extra.inc"
  "INCLUDE/SUMMARY/gas.inc"
  "INCLUDE/SUMMARY/tracer.data"
  "INCLUDE/SUMMARY/wpave.inc"
  "INCLUDE/VFP/AlmostVertNew.VFP"
  "INCLUDE/VFP/DevNew.VFP"
  "INCLUDE/VFP/E1h.VFP"
  "INCLUDE/VFP/GAS_PD2.VFP"
  "INCLUDE/VFP/GasProd.VFP"
  "INCLUDE/VFP/NEW_D2_GAS_0.00003.VFP"
  "INCLUDE/VFP/pd2.VFP"
  "INCLUDE/VFP/pe2.VFP"
  "INCLUDE/VFP/B1BH.Ecl"
  "INCLUDE/VFP/B2H.Ecl"
  "INCLUDE/VFP/B3H.Ecl"
  "INCLUDE/VFP/B4DH.Ecl"
  "INCLUDE/VFP/C1H.Ecl"
  "INCLUDE/VFP/C2H.Ecl"
  "INCLUDE/VFP/C3H.Ecl"
  "INCLUDE/VFP/C4AH.Ecl"
  "INCLUDE/VFP/C4H.Ecl"
  "INCLUDE/VFP/D1CH.Ecl"
  "INCLUDE/VFP/D2H.Ecl"
  "INCLUDE/VFP/D3BH.Ecl"
  "INCLUDE/VFP/E1H.Ecl"
  "INCLUDE/VFP/E3CH.Ecl"
  "INCLUDE/VFP/F1H.Ecl"
  "INCLUDE/VFP/F2H.Ecl"
  "INCLUDE/VFP/F3H.Ecl"
  "INCLUDE/VFP/F4H.Ecl"
  "INCLUDE/VFP/K3H.Ecl"
  "INCLUDE/VFP/PB1.PIPE.Ecl"
  "INCLUDE/VFP/PB2.PIPE.Ecl"
  "INCLUDE/VFP/PD1.PIPE.Ecl"
  "INCLUDE/VFP/PD2.PIPE.Ecl"
  "INCLUDE/VFP/PE1.PIPE.Ecl"
  "INCLUDE/VFP/PE2.PIPE.Ecl"
  "INCLUDE/VFP/WIC.PIPE.Ecl"
  "INCLUDE/VFP/WIF.PIPE.Ecl"
)

FILES=("${CORE[@]}")
[[ $GRID_ONLY -eq 1 ]] || FILES+=("${EXTRA[@]}")

# ---------------------------------------------------------------------------
if [[ $CHECK_ONLY -eq 1 ]]; then
  echo "checking $OUT"
  missing=0
  for f in "${FILES[@]}"; do
    if [[ -s "$OUT/$f" ]]; then
      printf '  %-46s %10d bytes\n' "$f" "$(wc -c < "$OUT/$f")"
    else
      printf '  %-46s MISSING\n' "$f"
      missing=$((missing + 1))
    fi
  done
  if [[ $missing -eq 0 ]]; then
    echo "all ${#FILES[@]} files present"
    exit 0
  fi
  echo "$missing of ${#FILES[@]} files missing; run without --check to fetch them" >&2
  exit 1
fi

# Pick a downloader.  Both are told to fail on an HTTP error rather than
# cheerfully writing the error page into the file.
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL --retry 3 --retry-delay 2 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q --tries=3 -O "$2" "$1"; }
else
  echo "ERROR: neither curl nor wget is available" >&2
  exit 1
fi

echo "== Norne =="
echo "   source : $BASE"
echo "   target : $OUT"
echo "   files  : ${#FILES[@]}"
echo

mkdir -p "$OUT"

have=0
got=0
failed=()
for f in "${FILES[@]}"; do
  dst="$OUT/$f"
  if [[ -s "$dst" ]]; then
    have=$((have + 1))
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  printf '   %-46s ' "$f"
  # Download to .part and rename on success, so an interrupted run never
  # leaves a truncated file that the next run would mistake for complete.
  if fetch "$BASE/$f" "$dst.part"; then
    mv -f "$dst.part" "$dst"
    printf '%10d bytes\n' "$(wc -c < "$dst")"
    got=$((got + 1))
  else
    rm -f "$dst.part"
    echo "FAILED"
    failed+=("$f")
  fi
done

echo
echo "   $got downloaded, $have already present"

if [[ ${#failed[@]} -gt 0 ]]; then
  echo
  echo "   ${#failed[@]} file(s) could not be fetched:" >&2
  printf '     %s\n' "${failed[@]}" >&2
  exit 1
fi

DECK="$OUT/NORNE_ATW2013.DATA"
echo
echo "Deck: $DECK"
echo
echo "Run it with:"
echo "  NGPU=1 ./scripts/run.sh decks/NORNE.DATA"
echo
echo "Read the deck without touching the GPU:"
echo "  ./scripts/build_deck_tests.sh $DECK"
echo "  ./build/deck/deckinfo $DECK --gradients 0.0706,0.1001,0.0186"
