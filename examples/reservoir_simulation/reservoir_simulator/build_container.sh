#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# build_container.sh -- build and enter the container for this machine.
#
# It reads the CUDA version your driver supports from nvidia-smi and builds on the
# matching NVIDIA HPC SDK image, which brings nvcc, the CUDA libraries and a
# CUDA-aware OpenMPI (HPC-X).  NVRS is an MPI program whose ranks exchange DEVICE
# buffers, and a stock OpenMPI stages those through host memory: it still runs,
# and it is quietly several times slower.
#
#   ./build_container.sh                 # build (if needed), then open a shell
#   ./build_container.sh build           # build only
#   ./build_container.sh shell           # open a shell in the built image
#   ./build_container.sh run <cmd...>    # run one command inside and exit
#   ./build_container.sh check           # prove MPI is CUDA-aware, then exit
#   ./build_container.sh rebuild         # build from scratch, no layer cache
#
# Environment overrides:
#   NVHPC_IMAGE=tag    base image, instead of choosing from nvidia-smi
#   IMAGE=name:tag     image name           (default follows the base image)
#   NAME=container     running container    (default nvrs)
#   GPUS=all|"0,1"     which GPUs to expose (default all)
#   MOUNT=/path        host dir to mount    (default this repository)
#   NVRS_GUI_PORT=n    port on this machine for gui/server.py (default 8765)
#
# The repository is MOUNTED, not copied.  An edit on the host is an edit in the
# container, so rebuilding the simulator never means rebuilding the image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-}"
NVHPC_IMAGE="${NVHPC_IMAGE:-}"
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
  if [ -n "$DOCKER" ]; then
    can_reach_daemon
    return 0
  fi
  if   command -v docker >/dev/null 2>&1; then DOCKER=docker
  elif command -v podman >/dev/null 2>&1; then DOCKER=podman
  else
    echo "ERROR: neither docker nor podman is on PATH." >&2
    echo "  On a cluster you may want Apptainer/Singularity instead:" >&2
    echo "    apptainer build nvrs.sif docker-daemon://$IMAGE" >&2
    echo "    apptainer shell --nv -B \$PWD:/workspace/project nvrs.sif" >&2
    exit 1
  fi
  can_reach_daemon
}

# A daemon that is down, or a user outside the docker group, otherwise surfaces
# as "image not found" followed by a failed build, which names neither cause.
can_reach_daemon() {
  $DOCKER info >/dev/null 2>&1 && return 0
  echo "ERROR: $DOCKER is installed but cannot reach its daemon." >&2
  if [ "$DOCKER" = "docker" ]; then
    echo "  Start it:           sudo systemctl start docker" >&2
    echo "  Or join its group:  sudo usermod -aG docker \$USER   (then log out and in)" >&2
  fi
  exit 1
}

# --- which NVIDIA HPC SDK ----------------------------------------------------
#
# A program built with a CUDA newer than the driver supports may not start, and
# nvidia-smi's header states the newest the driver supports.  So this machine's
# answer selects the SDK.  The newest targeted is CUDA 13.2.  CUDA 13 dropped the
# oldest GPUs, so those stay on CUDA 12.9, the last release that builds for them.
pick_nvhpc() {
  local smi cuda cap major minor n sdk
  if [ -z "$NVHPC_IMAGE" ]; then
    if ! smi="$(nvidia-smi 2>/dev/null)"; then
      echo "ERROR: nvidia-smi did not run, so there is no GPU to build for." >&2
      echo "  Install the NVIDIA driver and check that nvidia-smi works." >&2
      echo "  To build the image anyway, name one:" >&2
      echo "    NVHPC_IMAGE=nvcr.io/nvidia/nvhpc:26.5-devel-cuda13.2-ubuntu22.04 ./build_container.sh build" >&2
      exit 1
    fi
    cuda="$(printf '%s\n' "$smi" | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
    cuda="${cuda%%$'\n'*}"
    if [ -z "$cuda" ]; then
      echo "ERROR: nvidia-smi did not report a CUDA version. Set NVHPC_IMAGE to choose one." >&2
      exit 1
    fi
    major="${cuda%%.*}"; minor="${cuda#*.}"
    n=$((10#$major * 100 + 10#$minor))
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
    cap="${cap%%$'\n'*}"; cap="${cap//[. ]/}"
    if [ -n "$cap" ] && [ "$cap" -lt 75 ] && [ "$n" -ge 1300 ]; then n=1209; fi
    if   [ "$n" -ge 1302 ]; then sdk=26.5-devel-cuda13.2
    elif [ "$n" -ge 1301 ]; then sdk=26.3-devel-cuda13.1
    elif [ "$n" -ge 1300 ]; then sdk=25.11-devel-cuda13.0
    elif [ "$n" -ge 1209 ]; then sdk=25.7-devel-cuda12.9
    elif [ "$n" -ge 1208 ]; then sdk=25.3-devel-cuda12.8
    elif [ "$n" -ge 1206 ]; then sdk=25.1-devel-cuda12.6
    else
      echo "ERROR: this driver supports CUDA $cuda, and the simulator needs 12.6 or newer." >&2
      echo "  Update the NVIDIA driver, then run this again." >&2
      exit 1
    fi
    NVHPC_IMAGE="nvcr.io/nvidia/nvhpc:${sdk}-ubuntu22.04"
    echo "Driver supports CUDA $cuda, so using $NVHPC_IMAGE"
  fi
  [ -n "$IMAGE" ] || IMAGE="nvrs/nvhpc:${NVHPC_IMAGE##*:}"
}

# --- GPU flags -------------------------------------------------------------
#
# Docker 19.03+ takes --gpus; older installs need --runtime=nvidia, and podman
# takes neither and wants a device instead.  Choosing here keeps the run
# command below readable and keeps the failure -- "no GPU in the container" --
# from being discovered halfway through a benchmark.
#
# The help text is captured before it is searched.  Under pipefail, grep -q
# leaving the pipe early can kill the writer with SIGPIPE, which made this test
# fail on a docker that does have --gpus and chose --runtime=nvidia instead.
gpu_flags() {
  local help
  if [ "$DOCKER" = "podman" ]; then
    RUN_FLAGS+=(--device nvidia.com/gpu=all --security-opt=label=disable)
    return 0
  fi
  if ! command -v nvidia-container-cli >/dev/null 2>&1 && \
     ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "WARNING: the NVIDIA Container Toolkit is not on PATH, so the container may not" >&2
    echo "         see the GPU. Install nvidia-container-toolkit if this run fails." >&2
  fi
  help="$($DOCKER run --help 2>/dev/null || true)"
  if [[ "$help" == *--gpus* ]]; then
    RUN_FLAGS+=(--gpus "$GPUS")
  else
    RUN_FLAGS+=(--runtime=nvidia -e "NVIDIA_VISIBLE_DEVICES=$GPUS")
  fi
}

image_exists() { $DOCKER image inspect "$IMAGE" >/dev/null 2>&1; }

do_build() {
  local extra="${1:-}"
  local context
  # The Dockerfile copies nothing, because the repository is mounted.  An empty
  # context keeps the results and any installed libraries from being sent to
  # the daemon, which is several hundred megabytes before the first step.
  context="$(mktemp -d)"
  hr
  echo " Building $IMAGE"
  echo "   base    : $NVHPC_IMAGE"
  echo "   runtime : $DOCKER"
  echo "   arch    : $(uname -m)"
  hr
  # The first build downloads the SDK, about 7 GB, then installs system packages.
  # Later builds come from the layer cache.
  # shellcheck disable=SC2086
  if ! $DOCKER build $extra --build-arg "NVHPC_IMAGE=$NVHPC_IMAGE" -t "$IMAGE" -f "$HERE/Dockerfile" "$context"; then
    rmdir "$context"
    echo "ERROR: the image did not build. The first failing step is above." >&2
    exit 1
  fi
  rmdir "$context"
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
#   OMPI_ALLOW_RUN_AS_ROOT*
#       the image has no unprivileged user, and mpirun otherwise refuses to start.
#   GIT_CONFIG_*
#       git refuses a repository owned by another user, which a mount is here.
#
# An array, so a folder with a space in its name stays one argument.
run_flags() {
  RUN_FLAGS=(--rm)
  gpu_flags
  RUN_FLAGS+=(--ipc=host --ulimit memlock=-1 --ulimit stack=67108864
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
    -v "$MOUNT:$WORKDIR" -w "$WORKDIR"
    -e NVRS_IN_CONTAINER=1
    -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e "GIT_CONFIG_VALUE_0=*")
}

# The browser interface's port.  Published only by the interactive shell, and
# only when free, so that something already using it cannot stop the container.
gui_port_flags() {
  local port="${NVRS_GUI_PORT:-8765}"
  if (: </dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
    echo "note: port $port is in use, so gui/server.py will not be reachable from this" >&2
    echo "      machine. Set NVRS_GUI_PORT to a free port to publish it." >&2
  else
    RUN_FLAGS+=(-p "127.0.0.1:$port:8765")
  fi
}

do_shell() {
  local names inner
  # A second terminal joins the running container rather than failing on its name.
  names="$($DOCKER ps --format '{{.Names}}' 2>/dev/null || true)"
  if printf '%s\n' "$names" | grep -qx -- "$NAME"; then
    echo "Joining the running container $NAME"
    exec $DOCKER exec -it -w "$WORKDIR" "$NAME" /bin/bash
  fi
  names="$($DOCKER ps -a --format '{{.Names}}' 2>/dev/null || true)"
  if printf '%s\n' "$names" | grep -qx -- "$NAME"; then
    $DOCKER rm "$NAME" >/dev/null
  fi
  hr
  echo " $IMAGE"
  echo "   $MOUNT  ->  $WORKDIR   (mounted: edits on the host are live here)"
  echo
  echo " Next:"
  echo "   cd full_reservoir_simulator/GPU_reservoir_simulator/cuda/benchmark"
  echo "   ./nvrs.sh install"
  echo "   ./nvrs.sh all NORNE"
  hr
  run_flags
  gui_port_flags
  # The container runs as root, so what it builds in the mounted folder belongs
  # to root.  Hand it back on the way out, or git pull and rm fail on the host.
  # Rootless podman maps root to the calling user already.
  if [ "$DOCKER" = "docker" ] && [ "$(id -u)" != "0" ]; then
    inner="/bin/bash; s=\$?; chown -R $(id -u):$(id -g) '$WORKDIR' 2>/dev/null; exit \$s"
    exec $DOCKER run -it --name "$NAME" "${RUN_FLAGS[@]}" "$IMAGE" /bin/bash -c "$inner"
  fi
  exec $DOCKER run -it --name "$NAME" "${RUN_FLAGS[@]}" "$IMAGE" /bin/bash
}

do_run() {
  run_flags
  exec $DOCKER run -i "${RUN_FLAGS[@]}" "$IMAGE" /bin/bash -lc "$*"
}

# Prove the thing this image exists for, rather than assuming it.
#
# A container can have CUDA and an MPI and still not have CUDA-AWARE MPI, and
# nothing about a successful build says otherwise.  This asks the three
# questions whose answers decide whether a multi-GPU number is meaningful.
do_check() {
  run_flags
  $DOCKER run -i "${RUN_FLAGS[@]}" "$IMAGE" /bin/bash -lc '
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
    root="$(dirname "$(dirname "$(command -v nvcc)")")"
    mpicc -o /tmp/t /tmp/t.c -I"$root/include" -L"$root/lib64" -lcudart
    # --allow-run-as-root because the image has no unprivileged user; -np 2 so
    # the send/recv above actually crosses a rank boundary.
    mpirun --allow-run-as-root -np 2 /tmp/t \
      || mpirun --allow-run-as-root -np 1 /tmp/t
    printf "\nIf a value of 42 arrived at rank 1 from a DEVICE pointer, MPI is CUDA-aware.\n"
  '
}

case "${1:-default}" in
  build)   need_runtime; pick_nvhpc; do_build ;;
  rebuild) need_runtime; pick_nvhpc; do_build --no-cache ;;
  shell)   need_runtime; pick_nvhpc; image_exists || do_build; do_shell ;;
  check)   need_runtime; pick_nvhpc; image_exists || do_build; do_check ;;
  run)     shift; need_runtime; pick_nvhpc; image_exists || do_build; do_run "$@" ;;
  default) need_runtime; pick_nvhpc; image_exists || do_build; do_shell ;;
  -h|--help|help)
    awk 'NR>=6 && !/^#/{exit} NR>=6{sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}"
    ;;
  *)
    echo "unknown command: $1  (try: build, rebuild, shell, run, check, help)" >&2
    exit 1 ;;
esac
