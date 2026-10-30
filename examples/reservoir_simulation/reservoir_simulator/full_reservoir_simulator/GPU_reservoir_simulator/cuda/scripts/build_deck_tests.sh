#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# build_deck_tests.sh -- build and run the field-deck readers.
#
# These are host-only: corner-point geometry, the schedule, and equilibration
# have no CUDA in them, so they build with a plain C++ compiler and run
# anywhere.  That is deliberate.  A geometry or parsing error found on a laptop
# in two seconds is the same error that would otherwise surface as a wrong
# answer after a multi-GPU run.
#
#   ./scripts/build_deck_tests.sh                    # build and run
#   ./scripts/build_deck_tests.sh path/to/deck.DATA  # also run against a deck
#
# With a deck argument the Norne sections of each test run; without one they
# are skipped and only the analytic cases execute.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
BUILD="${BUILD_DIR:-$CUDA_DIR/build/deck}"
DECK="${1:-}"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter}"

mkdir -p "$BUILD"

SRC=("$CUDA_DIR/src/cpgrid.cpp" "$CUDA_DIR/src/schedule.cpp"
     "$CUDA_DIR/src/equil.cpp" "$CUDA_DIR/src/field_partition.cpp")

# test_field_units builds the whole FieldModel, so it also needs the unit
# conversion and the one CUDA symbol ~FieldModel reaches (see the stub itself).
UNITS_SRC=("$CUDA_DIR/src/units_field.cpp" "$CUDA_DIR/tests/stub_halo_conn.cpp")

echo "== building =="
echo "   compiler: $($CXX --version | head -1)"

fail=0
build_one() {
  local name="$1"; shift
  printf '   %-14s ' "$name"
  if $CXX $CXXFLAGS -I"$CUDA_DIR/include" "$@" "${SRC[@]}" \
       -o "$BUILD/$name" 2> "$BUILD/$name.log"; then
    echo "ok"
  else
    echo "FAILED"
    sed -n '1,25p' "$BUILD/$name.log"
    fail=1
  fi
}

build_one test_cpgrid   "$CUDA_DIR/tests/test_cpgrid.cpp"
build_one test_schedule "$CUDA_DIR/tests/test_schedule.cpp"
build_one test_equil    "$CUDA_DIR/tests/test_equil.cpp"
build_one test_partition "$CUDA_DIR/tests/test_partition.cpp"
build_one test_field_units "$CUDA_DIR/tests/test_field_units.cpp" "${UNITS_SRC[@]}"
# test_well_control is pure arithmetic on a header -- no deck sources at all.
printf '   %-14s ' test_well_control
if $CXX $CXXFLAGS -I"$CUDA_DIR/include" "$CUDA_DIR/tests/test_well_control.cpp" \
     -o "$BUILD/test_well_control" 2> "$BUILD/test_well_control.log"; then
  echo "ok"
else
  echo "FAILED"; sed -n '1,25p' "$BUILD/test_well_control.log"; fail=1
fi
# test_pvt reads and evaluates the tabulated PVT; the only source it needs
# beyond the header is the table reader itself.
printf '   %-14s ' test_pvt
if $CXX $CXXFLAGS -I"$CUDA_DIR/include" "$CUDA_DIR/tests/test_pvt.cpp" \
     "$CUDA_DIR/src/pvt_tables.cpp" \
     -o "$BUILD/test_pvt" 2> "$BUILD/test_pvt.log"; then
  echo "ok"
else
  echo "FAILED"; sed -n '1,25p' "$BUILD/test_pvt.log"; fail=1
fi
# test_solver_keywords parses the NVRS keyword deck (LINSOLVER and the
# per-library knobs); it needs deck.cpp and nothing else.
printf '   %-14s ' test_solver_keywords
if $CXX $CXXFLAGS -I"$CUDA_DIR/include" "$CUDA_DIR/tests/test_solver_keywords.cpp" \
     "$CUDA_DIR/src/deck.cpp" \
     -o "$BUILD/test_solver_keywords" 2> "$BUILD/test_solver_keywords.log"; then
  echo "ok"
else
  echo "FAILED"; sed -n '1,25p' "$BUILD/test_solver_keywords.log"; fail=1
fi
build_one deckinfo      "$CUDA_DIR/tools/deckinfo.cpp"

[[ $fail -eq 0 ]] || { echo "build failed; see $BUILD/*.log" >&2; exit 1; }

echo
echo "== running =="
rc=0
for t in test_cpgrid test_schedule test_equil test_partition test_field_units \
         test_well_control test_pvt test_solver_keywords; do
  echo
  echo "---- $t ----"
  if [[ -n "$DECK" ]]; then
    "$BUILD/$t" "$DECK" || rc=1
  else
    "$BUILD/$t" || rc=1
  fi
done

echo
if [[ $rc -eq 0 ]]; then
  echo "all deck tests passed"
else
  echo "SOME DECK TESTS FAILED" >&2
fi

if [[ -n "$DECK" ]]; then
  echo
  echo "For a full report on this deck:"
  echo "  $BUILD/deckinfo $DECK --gradients 0.0706,0.1001,0.0186"
fi

exit $rc
