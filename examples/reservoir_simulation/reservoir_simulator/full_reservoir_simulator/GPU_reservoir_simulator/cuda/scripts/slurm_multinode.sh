#!/usr/bin/env bash
#SBATCH --job-name=nvrs-cuda
#SBATCH --nodes=4                 # multi-node
#SBATCH --ntasks-per-node=4       # 4 GPUs (GB200) per node → 4 ranks/node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --exclusive
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# slurm_multinode.sh — NVRS-CUDA across N nodes × G GPUs.  Total ranks =
# nodes*ntasks-per-node; each rank binds one GPU and owns one K-slab of the
# reservoir.  HYPRE's BoomerAMG pressure solve runs distributed over all ranks.
#
#   sbatch scripts/slurm_multinode.sh decks/RESERVOIR.DATA
set -euo pipefail

DECK="${1:-decks/RESERVOIR.DATA}"
EXE="${EXE:-build/nvrs_cuda}"

module load cuda openmpi 2>/dev/null || true
if [[ -n "${HYPRE_FP32_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$HYPRE_FP32_DIR/lib:$HYPRE_FP32_DIR/lib64:${LD_LIBRARY_PATH:-}"
fi

# One rank per GPU; map ranks to GPUs round-robin (handled in main.cu too).
export NCCL_DEBUG=WARN
srun --mpi=pmix --gpu-bind=closest "$EXE" "$DECK"
