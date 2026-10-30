#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# build_container.sh -- build and enter the container these results came from.
#
# The image is CUDA 12.8 on Ubuntu 22.04 with UCX, PMIx and OpenMPI built from
# source, so that MPI is CUDA-aware and Slurm-aware.  That is the whole reason
# it exists rather than a stock image: NVRS is an MPI program whose ranks
# exchange DEVICE buffers, and a stock OpenMPI stages those through host memory.
# It still runs, and it is quietly several times slower -- which is the worst
# kind of difference, because nothing fails and the numbers look plausible.
#
#   ./build_container.sh                 # build (if needed), then open a shell
#   ./build_container.sh build           # build only
#   ./build_container.sh shell           # open a shell in the built image
#   ./build_container.sh run <cmd...>    # run one command inside and exit
#   ./build_container.sh check           # prove MPI is CUDA-aware, then exit
#   ./build_container.sh rebuild         # build from scratch, no layer cache
#
# Environment overrides:
#   IMAGE=name:tag     image name           (default nvrs/gb200:cuda12.8)
#   NAME=container     running container    (default nvrs)
#   GPUS=all|"0,1"     which GPUs to expose (default all)
#   MOUNT=/path        host dir to mount    (default this repository)
#
# The repository is MOUNTED, not copied.  An edit on the host is an edit in the
# container, so rebuilding the simulator never means rebuilding the image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-nvrs/gb200:cuda12.8}"
NAME="${NAME:-nvrs}"
GPUS="${GPUS:-all}"
MOUNT="${MOUNT:-$HERE}"
WORKDIR=/workspace/project

hr() { printf '=%.0s' {1..74}; echo; }

# --- the container runtime -------------------------------------------------
#
# podman is a drop-in for every flag used below and is what a shared cluster is
# more likely to allow, so take it when docker is absent rather than failing.
# Resolved on demand, not at startup: `help` must work on a machine that has
# no container runtime at all, which is exactly where someone reads it first.
DOCKER="${DOCKER:-}"
need_runtime() {
  [ -n "$DOCKER" ] && return 0
  if   command -v docker >/dev/null 2>&1; then DOCKER=docker
  elif command -v podman >/dev/null 2>&1; then DOCKER=podman
  else
    echo "ERROR: neither docker nor podman is on PATH." >&2
    echo "  On a cluster you may want Apptainer/Singularity instead:" >&2
    echo "    apptainer build nvrs.sif docker-daemon://$IMAGE" >&2
    echo "    apptainer shell --nv -B \$PWD:/workspace/project nvrs.sif" >&2
    exit 1
  fi
}

# --- GPU flags -------------------------------------------------------------
#
# Docker 19.03+ takes --gpus; older installs need --runtime=nvidia, and podman
# takes neither and wants a device instead.  Choosing here keeps the run
# command below readable and keeps the failure -- "no GPU in the container" --
# from being discovered halfway through a benchmark.
gpu_flags() {
  if [ "$DOCKER" = "podman" ]; then
    echo "--device nvidia.com/gpu=all --security-opt=label=disable"
  elif $DOCKER run --help 2>/dev/null | grep -q -- '--gpus'; then
    echo "--gpus ${GPUS}"
  else
    echo "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPUS}"
  fi
}

image_exists() { $DOCKER image inspect "$IMAGE" >/dev/null 2>&1; }

do_build() {
  local extra="${1:-}"
  hr
  echo " Building $IMAGE"
  echo "   runtime : $DOCKER"
  echo "   context : $HERE"
  echo "   arch    : $(uname -m)"
  hr
  # UCX, PMIx and OpenMPI are all compiled from source here, so this is tens of
  # minutes on a first build and seconds afterwards from the layer cache.
  # shellcheck disable=SC2086
  $DOCKER build $extra -t "$IMAGE" -f "$HERE/Dockerfile" "$HERE"
  echo
  echo "Built $IMAGE"
}

# The flags the benchmarks were run with, in one place so the shell, the
# one-shot run and the CUDA-aware check cannot drift apart.
#
#   --ipc=host, --ulimit memlock=-1, --ulimit stack=67108864
#       MPI and UCX register pinned host memory and use shared memory between
#       ranks on a node.  Without these a multi-rank run fails at startup, or
#       worse, silently falls back to a slower transport.
#   --cap-add=SYS_PTRACE, --security-opt seccomp=unconfined
#       so that nsys, gdb and strace work inside the container.
run_flags() {
  cat <<FLAGS
--rm
$(gpu_flags)
--ipc=host
--ulimit memlock=-1
--ulimit stack=67108864
--cap-add=SYS_PTRACE
--security-opt seccomp=unconfined
-p 127.0.0.1:${NVRS_GUI_PORT:-8765}:8765
-v ${MOUNT}:${WORKDIR}
-w ${WORKDIR}
-e NVRS_IN_CONTAINER=1
FLAGS
}

do_shell() {
  hr
  echo " $IMAGE"
  echo "   $MOUNT  ->  $WORKDIR   (mounted: edits on the host are live here)"
  echo
  echo " Build the simulator:"
  echo "   cd full_reservoir_simulator && ./quickstart.sh"
  echo " Run Norne against OPM Flow:"
  echo "   cd full_reservoir_simulator/GPU_reservoir_simulator/cuda/benchmark"
  echo "   ./get_norne.sh && ./run_norne.sh"
  hr
  # shellcheck disable=SC2046
  exec $DOCKER run -it --name "$NAME" $(run_flags) "$IMAGE" /bin/bash
}

do_run() {
  # shellcheck disable=SC2046
  exec $DOCKER run -i $(run_flags) "$IMAGE" /bin/bash -lc "$*"
}

# Prove the thing this image exists for, rather than assuming it.
#
# A container can have CUDA and an MPI and still not have CUDA-AWARE MPI, and
# nothing about a successful build says otherwise.  This asks the three
# questions whose answers decide whether a multi-GPU number is meaningful.
do_check() {
  # shellcheck disable=SC2046
  $DOCKER run -i $(run_flags) "$IMAGE" /bin/bash -lc '
    set -e
    printf "\n== GPUs ==\n";        nvidia-smi -L || echo "  no GPUs visible!"
    printf "\n== nvcc ==\n";        nvcc --version | tail -2
    printf "\n== mpirun ==\n";      mpirun --version | head -1
    printf "\n== UCX ==\n";         ucx_info -v | head -2
    printf "\n== CUDA-aware MPI ==\n"
    ompi_info --parsable --all 2>/dev/null \
      | grep -i "mpi_built_with_cuda_support:value" \
      || echo "  could not query ompi_info"
    printf "\n== UCX CUDA transports ==\n"
    ucx_info -d 2>/dev/null | grep -i "Transport: cuda" | sort -u \
      || echo "  no CUDA transports -- MPI will stage through host memory"
    printf "\n== a real CUDA-aware send/recv ==\n"
    cat > /tmp/t.c <<EOF
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);
  int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  int ndev=0; cudaGetDeviceCount(&ndev);
  if(ndev>0) cudaSetDevice(rank%ndev);
  int *d=0; cudaMalloc(&d,sizeof(int));       /* DEVICE buffer, on purpose */
  int host = (rank==0) ? 42 : 0;
  cudaMemcpy(d,&host,sizeof(int),cudaMemcpyHostToDevice);
  if(size>1){
    if(rank==0) MPI_Send(d,1,MPI_INT,1,0,MPI_COMM_WORLD);
    else if(rank==1) MPI_Recv(d,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  cudaMemcpy(&host,d,sizeof(int),cudaMemcpyDeviceToHost);
  printf("  rank %d of %d, %d GPU(s), value %d\n",rank,size,ndev,host);
  cudaFree(d); MPI_Finalize(); return 0;
}
EOF
    mpicc -o /tmp/t /tmp/t.c -L${CUDA_HOME}/lib64 -lcudart
    # --allow-run-as-root because the image has no unprivileged user; -np 2 so
    # the send/recv above actually crosses a rank boundary.
    mpirun --allow-run-as-root -np 2 /tmp/t \
      || mpirun --allow-run-as-root -np 1 /tmp/t
    printf "\nIf a value of 42 arrived at rank 1 from a DEVICE pointer, MPI is CUDA-aware.\n"
  '
}

case "${1:-default}" in
  build)   need_runtime; do_build ;;
  rebuild) need_runtime; do_build --no-cache ;;
  shell)   need_runtime; image_exists || do_build; do_shell ;;
  check)   need_runtime; image_exists || do_build; do_check ;;
  run)     shift; need_runtime; image_exists || do_build; do_run "$@" ;;
  default) need_runtime; image_exists || do_build; do_shell ;;
  -h|--help|help)
    sed -n '5,28p' "${BASH_SOURCE[0]}" | sed 's|^# \{0,1\}||'
    ;;
  *)
    echo "unknown command: $1  (try: build, rebuild, shell, run, check, help)" >&2
    exit 1 ;;
esac
