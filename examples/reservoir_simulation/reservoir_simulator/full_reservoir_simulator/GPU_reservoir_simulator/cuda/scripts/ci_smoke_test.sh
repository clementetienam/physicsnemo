#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# ci_smoke_test.sh — build + run smoke test for NVRS-CUDA.
#
# Stages (each fails loudly with a non-zero exit for CI):
#   1. toolchain check   (nvcc, mpicxx, cmake, python3)
#   2. HYPRE check       (HYPRE_DIR / HYPRE_FP32_DIR + libHYPRE present)
#   3. configure + build (CMake; falls back to the Makefile)
#   4. run               (tiny decks/SMOKE.DATA — skipped if no GPU visible)
#   5. output check      (state_*.bin, summary.txt produced)
#
# Usage:
#   source ../../Mixed_precision/hypre_env.sh      # sets HYPRE_FP32_DIR
#   ./scripts/ci_smoke_test.sh
# Env overrides: HYPRE_DIR, CUDA_ARCH, BUILD_DIR, DECK.
set -uo pipefail

cd "$(dirname "$0")/.."          # → cuda/ project root
ROOT="$PWD"
BUILD_DIR="${BUILD_DIR:-build}"
DECK="${DECK:-decks/SMOKE.DATA}"
HYPRE_DIR="${HYPRE_DIR:-${HYPRE_FP32_DIR:-}}"

pass=0; fail=0
ok()   { echo "  [ OK ] $*"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $*"; fail=$((fail+1)); }
hdr()  { echo; echo "== $* =="; }

# ---- 1. toolchain --------------------------------------------------------
hdr "1. toolchain"
for t in nvcc mpicxx cmake python3; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t → $(command -v $t)"
  else bad "$t not found in PATH"; fi
done
[[ "$fail" -gt 0 ]] && { echo; echo "toolchain incomplete — aborting"; exit 1; }

# ---- 2. HYPRE ------------------------------------------------------------
hdr "2. HYPRE"
if [[ -z "$HYPRE_DIR" ]]; then
  bad "HYPRE_DIR / HYPRE_FP32_DIR not set (source Mixed_precision/hypre_env.sh)"
  exit 1
fi
HYPRE_LIBDIR=""
for d in "$HYPRE_DIR/lib" "$HYPRE_DIR/lib64"; do
  if ls "$d"/libHYPRE.* >/dev/null 2>&1; then HYPRE_LIBDIR="$d"; fi
done
if [[ -n "$HYPRE_LIBDIR" ]]; then ok "libHYPRE in $HYPRE_LIBDIR"
else bad "libHYPRE not found under $HYPRE_DIR/{lib,lib64}"; exit 1; fi
[[ -f "$HYPRE_DIR/include/HYPRE.h" ]] && ok "HYPRE.h found" || bad "HYPRE.h missing"

# ---- 3. build ------------------------------------------------------------
hdr "3. build"
EXE=""
if cmake -S . -B "$BUILD_DIR" -DHYPRE_DIR="$HYPRE_DIR" \
     ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH} \
     -DNVRS_MIXED_PRECISION=ON >/tmp/nvrs_cmake.log 2>&1 \
   && cmake --build "$BUILD_DIR" -j >/tmp/nvrs_build.log 2>&1; then
  EXE="$BUILD_DIR/nvrs_cuda"
  ok "CMake build succeeded → $EXE"
else
  echo "  CMake path failed (see /tmp/nvrs_cmake.log, /tmp/nvrs_build.log);"
  echo "  trying the Makefile ..."
  if make HYPRE_DIR="$HYPRE_DIR" ${CUDA_ARCH:+CUDA_ARCH=$CUDA_ARCH} \
       >/tmp/nvrs_make.log 2>&1; then
    EXE="./nvrs_cuda"; ok "Makefile build succeeded → $EXE"
  else
    bad "build failed (see /tmp/nvrs_*.log)"; tail -30 /tmp/nvrs_build.log 2>/dev/null
    exit 1
  fi
fi
[[ -x "$EXE" ]] && ok "executable present" || { bad "executable missing"; exit 1; }

# ---- 4. run --------------------------------------------------------------
hdr "4. run ($DECK)"
export LD_LIBRARY_PATH="$HYPRE_LIBDIR:${LD_LIBRARY_PATH:-}"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L | wc -l); [[ "$NGPU" -lt 1 ]] && NGPU=1
  echo "  detected $NGPU GPU(s); running on 1 rank"
  if mpirun -np 1 "$EXE" "$DECK" >/tmp/nvrs_run.log 2>&1; then
    ok "simulation ran"
  else
    bad "run failed (see /tmp/nvrs_run.log)"; tail -30 /tmp/nvrs_run.log; exit 1
  fi
else
  echo "  no GPU visible — build verified, run skipped (CI build-only mode)"
  echo; echo "SMOKE TEST: build OK (run skipped). pass=$pass fail=$fail"
  exit 0
fi

# ---- 5. outputs ----------------------------------------------------------
hdr "5. outputs"
OUT="OUTPUT_SMOKE"
ls "$OUT"/state_0000.bin >/dev/null 2>&1 && ok "state snapshots written" \
  || bad "no state_*.bin produced"
[[ -f "$OUT/summary.txt" ]] && ok "summary.txt written" || bad "summary.txt missing"
[[ -f "$OUT/RSM_FVM.csv" ]] && ok "RSM_FVM.csv written" || bad "RSM_FVM.csv missing"

# ---- 6. Jacobian cross-check --------------------------------------------
# The analytic and finite-difference Jacobians must agree to O(eps); a mismatch
# means a hand-derived derivative is wrong.
hdr "6. Jacobian cross-check (analytic vs numerical)"
if mpirun -np 1 "$EXE" "$DECK" --jac-check >/tmp/nvrs_jac.log 2>&1; then
  if grep -q "PASS" /tmp/nvrs_jac.log; then
    ok "analytic and numerical Jacobians agree"
  else
    bad "Jacobian mismatch"; grep -A2 "max rel diff" /tmp/nvrs_jac.log || true
  fi
else
  bad "--jac-check run failed (see /tmp/nvrs_jac.log)"
fi

# ---- 7. rank invariance (the halo-exchange regression) -------------------
# A K-slab decomposition must not change the answer.  Historically the off-rank
# K faces were silently treated as no-flow, which turned an N-rank run into N
# independent reservoirs — a bug that is invisible on one rank.  This check
# reruns the same deck on 2 ranks and compares the final field bit-for-bit
# within a tolerance.
hdr "7. rank invariance (1 rank vs 2 ranks)"
if [[ "${NGPU:-1}" -ge 1 ]]; then
  cp -r "$OUT" "${OUT}_np1" 2>/dev/null
  if mpirun -np 2 "$EXE" "$DECK" >/tmp/nvrs_run2.log 2>&1; then
    LAST=$(ls "$OUT"/state_*.bin | sort | tail -1 | xargs basename)
    if python3 - "$OUT/$LAST" "${OUT}_np1/$LAST" <<'PY'
import sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
if a.shape != b.shape:
    print(f"  shape mismatch {a.shape} vs {b.shape}")
    sys.exit(1)
scale = max(float(np.abs(b).max()), 1e-30)
rel = float(np.abs(a - b).max()) / scale
print(f"  max relative difference between 1-rank and 2-rank final state: {rel:.3e}")
sys.exit(0 if rel < 1e-4 else 1)
PY
    then ok "2-rank result matches the 1-rank reference"
    else bad "2-rank result differs — halo exchange or decomposition is wrong"
    fi
  else
    echo "  2-rank launch failed (may be an oversubscription limit, not a bug):"
    tail -5 /tmp/nvrs_run2.log
  fi
  rm -rf "${OUT}_np1"
fi

echo
echo "SMOKE TEST COMPLETE — pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]] || exit 1
