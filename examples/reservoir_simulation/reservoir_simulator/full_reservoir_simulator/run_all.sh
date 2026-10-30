#!/usr/bin/env bash
# =============================================================================
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# run_all.sh — ONE script, at the top of the tree, to build and run everything.
#
# Repository layout this script drives:
#   ./Mixed_precision/            reference mixed-precision CPR + HYPRE build
#   ./GPU_reservoir_simulator/    the CUDA simulator (in cuda/) + history matching
#   ./surrogate/                  original PhysicsNeMo pipeline (reference)
#
# Stages (each can be skipped):
#   [1] prerequisites   check nvcc / mpicxx / cmake / python3 / nvidia-smi
#   [2] HYPRE           build the FP32 HYPRE (skipped if already available)
#   [3] build           compile GPU_reservoir_simulator/cuda -> nvrs_cuda
#   [4] smoke           quick build+run sanity test (decks/SMOKE.DATA)
#   [5] jac-check       analytic-vs-numerical Jacobian cross-check
#   [6] simulate        run the reservoir deck (single/multi-GPU) + auto plots
#   [7] history-match   Bayesian alpha-REKI with the CUDA forward
#   [8] surrogate       (opt) train an FNO surrogate + alpha-REKI with it
#
# Usage (run from the repository root):
#   ./run_all.sh                     # full pipeline, all GPUs, RESERVOIR.DATA
#   ./run_all.sh --quick             # small ensemble so history matching is fast
#   ./run_all.sh --surrogate         # also do the surrogate train + HM (stage 8)
#   ./run_all.sh --deck GPU_reservoir_simulator/cuda/decks/COMPOSITIONAL.DATA
#   ./run_all.sh --ngpus 2 --arch 100
#   ./run_all.sh --skip-hypre --skip-hm      # e.g. just build + simulate
#
# Skips:  --skip-hypre  --skip-build  --skip-smoke  --skip-jaccheck
#         --skip-sim    --skip-hm
# =============================================================================
set -uo pipefail

# ── locate the tree ──────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$ROOT_DIR/GPU_reservoir_simulator/cuda"
MIXED_DIR="$ROOT_DIR/Mixed_precision"
[[ -d "$CUDA_DIR" ]]  || { echo "cannot find $CUDA_DIR"; exit 1; }
[[ -d "$MIXED_DIR" ]] || { echo "cannot find $MIXED_DIR"; exit 1; }

# ── defaults ─────────────────────────────────────────────────────────────────
DECK="$CUDA_DIR/decks/RESERVOIR.DATA"
NGPU=""              # auto-detect if empty
ARCH=""             # auto-detect if empty (numeric SM, e.g. 100 for GB200)
QUICK=false
DO_SURROGATE=false
SKIP_HYPRE=false; SKIP_BUILD=false; SKIP_SMOKE=false
SKIP_JAC=false;   SKIP_SIM=false;   SKIP_HM=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deck)          DECK="$2"; shift 2;;
    --ngpus)         NGPU="$2"; shift 2;;
    --arch)          ARCH="$2"; shift 2;;
    --quick)         QUICK=true; shift;;
    --surrogate)     DO_SURROGATE=true; shift;;
    --skip-hypre)    SKIP_HYPRE=true; shift;;
    --skip-build)    SKIP_BUILD=true; shift;;
    --skip-smoke)    SKIP_SMOKE=true; shift;;
    --skip-jaccheck) SKIP_JAC=true; shift;;
    --skip-sim)      SKIP_SIM=true; shift;;
    --skip-hm)       SKIP_HM=true; shift;;
    -h|--help)       sed -n '2,45p' "$0"; exit 0;;
    *) echo "unknown option: $1"; exit 1;;
  esac
done
# make DECK absolute (accept a path relative to the repo root or to cuda/)
[[ "$DECK" = /* ]] || { [[ -f "$ROOT_DIR/$DECK" ]] && DECK="$ROOT_DIR/$DECK" || DECK="$CUDA_DIR/$DECK"; }

# ── pretty logging ───────────────────────────────────────────────────────────
c_hdr='\033[1;36m'; c_ok='\033[0;32m'; c_warn='\033[1;33m'; c_err='\033[0;31m'; c_x='\033[0m'
section() { echo -e "\n${c_hdr}════════════════════════════════════════════════════════════════\n  $*\n════════════════════════════════════════════════════════════════${c_x}"; }
ok()   { echo -e "${c_ok}[ OK ]${c_x} $*"; }
warn() { echo -e "${c_warn}[WARN]${c_x} $*"; }
die()  { echo -e "${c_err}[FAIL]${c_x} $*"; exit 1; }

# =============================================================================
section "[1/8] Prerequisites"
for t in nvcc cmake python3; do command -v "$t" >/dev/null || die "$t not found in PATH"; ok "$t : $(command -v $t)"; done
command -v mpicxx  >/dev/null || warn "mpicxx not found (needed to build)"
command -v mpirun  >/dev/null || warn "mpirun not found (needed for multi-GPU)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — a GPU node is required"

if [[ -z "$NGPU" ]]; then NGPU=$(nvidia-smi -L | wc -l); [[ "$NGPU" -lt 1 ]] && NGPU=1; fi
ok "GPUs to use : $NGPU"
if [[ -z "$ARCH" ]]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  ARCH=${cc/./}; [[ -z "$ARCH" ]] && ARCH=80
fi
ok "CUDA SM arch : $ARCH"
python3 -c "import numpy,scipy,matplotlib,yaml,PIL" 2>/dev/null \
  && ok "python deps present (numpy/scipy/matplotlib/pyyaml/pillow)" \
  || warn "missing python deps — run: pip install numpy scipy matplotlib pyyaml pillow"

# =============================================================================
section "[2/8] HYPRE (FP32, for mixed-precision CPR)"
if [[ "$SKIP_HYPRE" == true ]]; then
  warn "skipped (--skip-hypre)"
elif [[ -n "${HYPRE_FP32_DIR:-}" && -d "${HYPRE_FP32_DIR}" ]]; then
  ok "using existing HYPRE_FP32_DIR = $HYPRE_FP32_DIR"
elif [[ -f "$MIXED_DIR/hypre_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$MIXED_DIR/hypre_env.sh"; ok "sourced $MIXED_DIR/hypre_env.sh → HYPRE_FP32_DIR=$HYPRE_FP32_DIR"
else
  warn "building HYPRE from source (one-time, ~10-20 min) ..."
  ( cd "$MIXED_DIR" && CUDA_ARCH="$ARCH" ./scripts/docker/install_hypre_sm100.sh ) \
    || die "HYPRE build failed"
  # shellcheck disable=SC1091
  source "$MIXED_DIR/hypre_env.sh"
  ok "HYPRE built → HYPRE_FP32_DIR=$HYPRE_FP32_DIR"
fi
[[ -n "${HYPRE_FP32_DIR:-}" ]] || die "HYPRE_FP32_DIR is not set — cannot build the simulator"
HYPRE_LIB="$HYPRE_FP32_DIR/lib"; [[ -d "$HYPRE_LIB" ]] || HYPRE_LIB="$HYPRE_FP32_DIR/lib64"
export LD_LIBRARY_PATH="$HYPRE_LIB:${LD_LIBRARY_PATH:-}"

# Everything from here runs inside the simulator directory.
cd "$CUDA_DIR"

# =============================================================================
section "[3/8] Build nvrs_cuda"
EXE="build/nvrs_cuda"
if [[ "$SKIP_BUILD" == true ]]; then
  warn "skipped (--skip-build)"; [[ -x "$EXE" ]] || die "no executable at $CUDA_DIR/$EXE"
else
  if cmake -S . -B build -DHYPRE_DIR="$HYPRE_FP32_DIR" \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DNVRS_MIXED_PRECISION=ON \
     && cmake --build build -j; then
    ok "CMake build → $CUDA_DIR/$EXE"
  else
    warn "CMake failed; trying the Makefile ..."
    make HYPRE_DIR="$HYPRE_FP32_DIR" CUDA_ARCH="$ARCH" && EXE="./nvrs_cuda" \
      || die "build failed (see errors above)"
    ok "Makefile build → $CUDA_DIR/$EXE"
  fi
fi
[[ -x "$EXE" ]] || die "executable not found after build"

# =============================================================================
section "[4/8] Smoke test"
if [[ "$SKIP_SMOKE" == true ]]; then warn "skipped (--skip-smoke)"; else
  EXE="$EXE" ./scripts/ci_smoke_test.sh || die "smoke test failed"
fi

# =============================================================================
section "[5/8] Jacobian cross-check ($DECK)"
if [[ "$SKIP_JAC" == true ]]; then warn "skipped (--skip-jaccheck)"; else
  mpirun -np 1 "$EXE" "$DECK" --jac-check || warn "jac-check reported an issue (see above)"
fi

# =============================================================================
section "[6/8] Simulate ($DECK on $NGPU GPU(s))"
if [[ "$SKIP_SIM" == true ]]; then warn "skipped (--skip-sim)"; else
  mpirun -np "$NGPU" "$EXE" "$DECK" || die "simulation failed"
  ok "simulation done — see the deck's OUTDIR for fields, RSM_FVM.csv, plots, summary.txt"
fi

# =============================================================================
# Keep hm/config.yaml in sync with the deck + hardware so HM 'just works'.
sync_hm_config() {
  python3 - "$DECK" "$EXE" "$NGPU" "$HYPRE_LIB" "$QUICK" <<'PY'
import sys, yaml, re
deck, exe, ngpu, hypre_lib, quick = sys.argv[1:6]
nx=ny=nz=None
for line in open(deck):
    line=line.split('--')[0]
    for key in ('NX','NY','NZ'):
        m=re.match(rf'\s*{key}\s+(\d+)', line)
        if m: globals()[key.lower()]=int(m.group(1))
cfg=yaml.safe_load(open('hm/config.yaml'))
cfg['deck']=deck; cfg['exe']=exe; cfg['n_gpus']=int(ngpu)
cfg['hypre_lib']=hypre_lib
if nx and ny and nz: cfg['grid']={'nx':nx,'ny':ny,'nz':nz}
if quick=='true':
    cfg['Ne']=20; cfg['Termm']=6
    cfg.setdefault('surrogate',{})['n_train']=40
    cfg['surrogate']['epochs']=40
yaml.safe_dump(cfg, open('hm/config.yaml','w'), sort_keys=False)
print(f"  hm/config.yaml synced: grid={cfg.get('grid')} exe={exe} n_gpus={ngpu} Ne={cfg['Ne']}")
PY
}

section "[7/8] History matching — forward = CUDA executable"
if [[ "$SKIP_HM" == true ]]; then warn "skipped (--skip-hm)"; else
  sync_hm_config
  python3 hm/run_history_match.py --config hm/config.yaml --make-truth || die "make-truth failed"
  python3 hm/run_history_match.py --config hm/config.yaml --forward cuda  || die "history matching failed"
  ok "history matching done — see $CUDA_DIR/HM_RESULTS/ (posterior.npz + plots)"
fi

# =============================================================================
section "[8/8] Surrogate (train + HM)"
if [[ "$DO_SURROGATE" == true ]]; then
  python3 -c "import torch" 2>/dev/null || die "torch not installed (pip install torch)"
  sync_hm_config
  python3 hm/run_history_match.py --config hm/config.yaml --make-truth || die "make-truth failed"
  python3 hm/train_surrogate.py   --config hm/config.yaml               || die "surrogate training failed"
  python3 hm/run_history_match.py --config hm/config.yaml --forward surrogate || die "surrogate HM failed"
  ok "surrogate pipeline done — see $CUDA_DIR/HM_RESULTS/"
else
  warn "skipped (pass --surrogate to enable)"
fi

section "ALL DONE"
ok "Executable : $CUDA_DIR/$EXE"
ok "Sim output : $CUDA_DIR/$(python3 -c "print([l.split()[1] for l in open('$DECK') if l.strip().startswith('OUTDIR')][0])" 2>/dev/null || echo OUTPUT)/"
ok "HM output  : $CUDA_DIR/HM_RESULTS/"
