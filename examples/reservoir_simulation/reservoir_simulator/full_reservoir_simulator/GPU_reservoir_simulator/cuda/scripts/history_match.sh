#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# history_match.sh — end-to-end Bayesian history matching around NVRS-CUDA.
#
# Stages:
#   [1] make-truth        synthesise observed well data from a "true" field
#   [2] train-surrogate   (optional) run NVRS-CUDA to label data + train the FNO
#   [3] history-match     α-REKI with the chosen forward (cuda | surrogate)
#
# Usage:
#   ./scripts/history_match.sh cuda                    # HM with the executable
#   ./scripts/history_match.sh surrogate --train       # train then HM w/ surrogate
#   CONFIG=hm/config.yaml ./scripts/history_match.sh cuda
set -euo pipefail

cd "$(dirname "$0")/.."          # → cuda/ project root
CONFIG="${CONFIG:-hm/config.yaml}"
FORWARD="${1:-cuda}"
TRAIN=false
[[ "${2:-}" == "--train" ]] && TRAIN=true

# HYPRE shared lib on the loader path (installed by install_hypre_sm100.sh).
if [[ -n "${HYPRE_FP32_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$HYPRE_FP32_DIR/lib:$HYPRE_FP32_DIR/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "==> [1/3] make-truth (observed data)"
python3 hm/run_history_match.py --config "$CONFIG" --make-truth

if [[ "$FORWARD" == "surrogate" && "$TRAIN" == true ]]; then
  echo "==> [2/3] train surrogate (NVRS-CUDA labels → FNO)"
  python3 hm/train_surrogate.py --config "$CONFIG"
fi

echo "==> [3/3] history match (forward = $FORWARD)"
python3 hm/run_history_match.py --config "$CONFIG" --forward "$FORWARD"

echo "==> done. See HM_RESULTS/ (posterior.npz, convergence.png, data_match.png, perm_percentiles.png)"
