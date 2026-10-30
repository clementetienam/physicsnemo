#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# check_hypre.sh — which HYPRE is NVRS actually using, and what was it built
# with?
#
# Worth a dedicated check because there are three different answers and they
# can disagree:
#
#   1. the headers NVRS COMPILED against  -> decides what the banner prints
#   2. the libHYPRE.so it LOADS at run time -> decides what actually happens
#   3. whatever else is lying around under other prefixes
#
# install_hypre_sm100.sh installs into $PWD/hypre-install, so running it from
# two directories leaves two installs, and NVRS links whichever one -DHYPRE_DIR
# pointed at when its build was configured.  Rebuilding HYPRE without
# reconfiguring NVRS changes the library on disk and nothing else; the banner
# keeps reporting the old headers, and a feature like GPU-aware MPI appears not
# to have taken effect when in fact it was never picked up.
#
#   ./check_hypre.sh
#   NVRS_EXE=/path/to/nvrs_cuda ./check_hypre.sh
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
NVRS_EXE="${NVRS_EXE:-$CUDA_DIR/build/nvrs_cuda}"
CACHE="$CUDA_DIR/build/CMakeCache.txt"

MACROS=(HYPRE_USING_CUDA HYPRE_USING_GPU_AWARE_MPI HYPRE_USING_UNIFIED_MEMORY
        HYPRE_SEQUENTIAL HYPRE_SINGLE HYPRE_USING_UMPIRE)

hr() { printf '%s\n' "------------------------------------------------------------"; }

# Which GPU architectures are actually inside a library, and can this GPU run
# any of them?  "Built with CUDA" and "runs here" are different questions, and
# on a machine whose GPU is newer than its CUDA they get different answers.
#
# The rules, so the verdict below is not folklore:
#   * a cubin (sm_NN) runs on the same MAJOR capability at an equal or higher
#     minor revision -- sm_100 runs on a 10.7 device, sm_90 does not;
#   * PTX (compute_NN) is JIT-compiled forward only, never onto older hardware.
arch_report() {
  local lib="$1" pad="${2:-  }"
  if ! command -v cuobjdump >/dev/null 2>&1; then
    echo "${pad}(cuobjdump not on PATH; cannot inspect)"
    return 0
  fi
  local elf ptx dev major minor ok="" err=""
  err="$(cuobjdump --list-elf "$lib" 2>&1 >/dev/null)"
  elf="$(cuobjdump --list-elf "$lib" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')"
  # cuobjdump names PTX entries after the ARCHITECTURE, not with a
  # compute_ prefix:
  #
  #   PTX file    1: libHYPRE.so.3.2.1.sm_103.ptx
  #
  # so both streams are matched on sm_NNN and told apart by which command
  # produced them.  Matching compute_[0-9]+ here found nothing in a library
  # that was entirely PTX, and the empty result then read as "no device
  # code at all".  Renamed to compute_NNN afterwards, because that is what
  # PTX for that architecture is properly called.
  ptx="$(cuobjdump --list-ptx "$lib" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | sed 's/^sm_/compute_/' | tr '\n' ' ')"
  echo "${pad}cubins : ${elf:-<none>}"
  echo "${pad}ptx    : ${ptx:-<none>}"

  # No cubins AND no PTX is not a finding about the library.  A library with no
  # device code in it cannot have run anything, so if this one is in use, the
  # empty lists mean cuobjdump declined to answer -- it does that for files it
  # does not recognise, and for shared objects whose device code it cannot
  # reach.  Report that as missing information, not as a verdict.
  if [[ -z "$elf" && -z "$ptx" ]]; then
    echo "${pad}verdict: cuobjdump reported no device code at all."
    echo "${pad}         That is a statement about cuobjdump, not about the"
    echo "${pad}         library: if this library runs, it has device code."
    [[ -n "$err" ]] && echo "${pad}         it said: $err"
    echo "${pad}         Check by hand with:"
    echo "${pad}           cuobjdump --list-elf $lib | head"
    return 0
  fi

  dev="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  [[ -z "$dev" ]] && { echo "${pad}(no GPU visible; cannot judge)"; return 0; }
  major="${dev%%.*}"; minor="${dev##*.}"
  echo "${pad}device : sm_${major}${minor}"
  local a n am an
  for a in $elf; do
    n="${a#sm_}"; am="${n%?}"; an="${n: -1}"
    if [[ "$am" == "$major" && "$an" -le "$minor" ]]; then ok="cubin $a"; break; fi
  done
  if [[ -z "$ok" ]]; then
    for a in $ptx; do
      n="${a#compute_}"; am="${n%?}"; an="${n: -1}"
      if [[ "$am" -lt "$major" || ( "$am" == "$major" && "$an" -le "$minor" ) ]]; then
        ok="PTX $a (JIT at load)"; break
      fi
    done
  fi
  if [[ -n "$ok" ]]; then
    echo "${pad}verdict: RUNS HERE via $ok"
  else
    echo "${pad}verdict: nothing listed above can run on sm_${major}${minor}."
    echo "${pad}         Rebuild it -- install_hypre_sm100.sh resolves the"
    echo "${pad}         architecture automatically now."
  fi
}


show_config() {
  local cfg="$1" pad="$2"
  if [[ ! -f "$cfg" ]]; then
    echo "${pad}(no HYPRE_config.h)"
    return 1
  fi
  local m
  for m in "${MACROS[@]}"; do
    if grep -qE "^[[:space:]]*#define[[:space:]]+$m" "$cfg"; then
      printf '%s%-32s yes\n' "$pad" "$m"
    else
      printf '%s%-32s no\n' "$pad" "$m"
    fi
  done
  grep -oE '#define HYPRE_RELEASE_VERSION "[^"]*"' "$cfg" 2>/dev/null \
    | sed "s|^|${pad}|"
}

hr
echo " Which HYPRE is NVRS using?"
hr

# --- 1. what NVRS was compiled against ------------------------------------
echo "1. Headers NVRS COMPILED against (this is what the banner reports)"
BUILD_INC=""
if [[ -f "$CACHE" ]]; then
  BUILD_INC="$(grep -E '^HYPRE_INCLUDE_DIR(:[A-Z]+)?=' "$CACHE" \
               | head -1 | cut -d= -f2-)"
  BUILD_LIB="$(grep -E '^HYPRE_LIBRARY(:[A-Z]+)?=' "$CACHE" \
               | head -1 | cut -d= -f2-)"
  echo "     include : ${BUILD_INC:-<not in cache>}"
  echo "     library : ${BUILD_LIB:-<not in cache>}"
  [[ -n "$BUILD_INC" ]] && show_config "$BUILD_INC/HYPRE_config.h" "       "
else
  echo "     no CMakeCache.txt at $CACHE -- NVRS has not been configured here"
fi
echo

# --- 2. what it loads at run time -----------------------------------------
# This is the one that decides behaviour: HYPRE_USING_GPU_AWARE_MPI is compiled
# into libHYPRE.so, so the loaded library is what actually stages halo
# exchanges through host memory or not.
echo "2. Library NVRS LOADS at run time (this is what actually runs)"
RUN_LIB=""
if [[ -x "$NVRS_EXE" ]]; then
  RUN_LIB="$(ldd "$NVRS_EXE" 2>/dev/null | awk '/libHYPRE/{print $3; exit}')"
  if [[ -n "$RUN_LIB" ]]; then
    echo "     $RUN_LIB"
    RUN_PREFIX="$(cd "$(dirname "$RUN_LIB")/.." && pwd)"
    show_config "$RUN_PREFIX/include/HYPRE_config.h" "       "
  else
    echo "     no libHYPRE in ldd output (statically linked, or not found)"
  fi
else
  echo "     $NVRS_EXE not built"
fi
echo

# --- 3. every other install on the box ------------------------------------
echo "3. Other HYPRE installs found nearby"
FOUND=0
while IFS= read -r cfg; do
  [[ -z "$cfg" ]] && continue
  pre="$(cd "$(dirname "$cfg")/.." && pwd)"
  [[ -n "$BUILD_INC" && "$pre/include" == "$BUILD_INC" ]] && continue
  FOUND=$((FOUND + 1))
  echo "     $pre"
  show_config "$cfg" "       "
done < <(find "$CUDA_DIR/../.." -maxdepth 6 -name HYPRE_config.h 2>/dev/null \
         | head -8)
[[ $FOUND -eq 0 ]] && echo "     (none besides the one above)"
echo

hr
# --- the trap: headers and library disagreeing ----------------------------
if [[ -n "$BUILD_INC" && -n "$RUN_LIB" ]]; then
  RUN_PREFIX="$(cd "$(dirname "$RUN_LIB")/.." && pwd)"
  if [[ "$RUN_PREFIX/include" != "$BUILD_INC" ]]; then
    echo " WARNING: NVRS compiled against"
    echo "            $BUILD_INC"
    echo "          but loads"
    echo "            $RUN_LIB"
    echo " Those are different installs.  Behaviour comes from the library,"
    echo " the banner comes from the headers, so the banner can be wrong in"
    echo " either direction.  Fix LD_LIBRARY_PATH, or reconfigure NVRS with"
    echo " -DHYPRE_DIR=$RUN_PREFIX and rebuild." >&2
  else
    echo " Headers and loaded library are the same install."
  fi
fi
echo
echo " If GPU-aware MPI is 'no' above, rebuild HYPRE with it and then"
echo " RECONFIGURE NVRS against that prefix -- rebuilding HYPRE alone leaves"
echo " NVRS compiled against the old headers:"
echo
echo "     HYPRE_GPU_AWARE_MPI=ON bash <repo>/Mixed_precision/scripts/docker/install_hypre_sm100.sh"
echo "     cd $CUDA_DIR && rm -rf build"
echo "     cmake -S . -B build -DHYPRE_DIR=<prefix> -DNVRS_MIXED_PRECISION=ON"
echo "     cmake --build build -j"
hr

hr
echo "GPU architectures in the library NVRS loads"
hr
_lib="${RUN_LIB:-${BUILD_LIB:-}}"
if [[ -n "$_lib" && -f "$_lib" ]]; then
  echo "  $_lib"
  arch_report "$_lib" "  "
else
  echo "  (no libHYPRE located above; nothing to inspect)"
fi
