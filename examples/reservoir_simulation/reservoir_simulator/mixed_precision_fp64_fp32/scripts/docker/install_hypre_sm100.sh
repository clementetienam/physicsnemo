#!/usr/bin/env bash
#
# install_hypre_sm100.sh
# =======================
# Build two HYPRE installs for Blackwell/GB200 experiments:
#   1) FP64 HYPRE  (default scalar type)
#   2) FP32 HYPRE  (single precision scalar type)
#
# This is needed because HYPRE precision is normally chosen at build time, not
# per-solver like AmgX modes. The benchmark can then link once against the FP64
# lib and once against the FP32 lib to compare timings.
#
# Usage on the GPU node/container:
#   ./install_hypre_sm100.sh
#   HYPRE_SRC=/path/hypre HYPRE_PREFIX=/path/hypre-install ./install_hypre_sm100.sh
#
set -euo pipefail

HYPRE_SRC="${HYPRE_SRC:-$PWD/hypre}"
HYPRE_PREFIX="${HYPRE_PREFIX:-$PWD/hypre-install}"
ENV_FILE="${ENV_FILE:-$PWD/hypre_env.sh}"
HYPRE_REF="${HYPRE_REF:-master}"
CUDA_ARCH="${CUDA_ARCH:-auto}"                      # auto = ask the GPU; see resolve_cuda_arch below.
# A number here pins it and skips detection entirely, which is what you
# want when cross-building for a machine you are not sitting on.
# It must NOT default to a number: a default is indistinguishable from a
# deliberate pin, and would silently disable the detection below.

# ---------------------------------------------------------------------------
# Which architecture to build for, when the GPU is newer than the toolkit
# ---------------------------------------------------------------------------
# Hecate's Vera Rubin reports compute capability 10.7, and no CUDA yet ships an
# sm_107 target: even CUDA 13.1 lists 100, 103, 110, 120, 121 and nothing at
# 107.  HYPRE's own CMake runs nvidia-smi, finds 107 and passes
# -gencode=arch=compute_107 straight to nvcc:
#
#   nvcc fatal : Unsupported gpu architecture 'compute_107'
#
# PTX is the way out.  code=compute_NN emits PTX rather than SASS and the driver
# JIT-compiles it at load, which is the supported path for hardware newer than
# the CUDA that built the binary.
#
# It must be the newest architecture NOT NEWER THAN the device.  The driver will
# JIT compute_103 onto an sm_107, but PTX built for compute_121 cannot run on a
# 10.7 device at all, so taking the toolkit's maximum is wrong.
#
# Setting the CUDAARCHS environment variable does NOT work here, and it is worth
# recording why: HYPRE tests CMAKE_CUDA_ARCHITECTURES before it calls
# enable_language(CUDA), and CUDAARCHS is only read by that call.  At the moment
# of the test the variable is still undefined, so HYPRE detects and then
# overwrites with CACHE ... FORCE.  Only a -D on the command line is defined
# early enough.
# sm_100 = Grace Blackwell (GB200/B200), the target this project is built
# for and the answer whenever there is no GPU to ask.
CUDA_ARCH_FALLBACK="${CUDA_ARCH_FALLBACK:-100}"
resolve_cuda_arch() {
  local want="${CUDA_ARCH:-}" nvcc_bin list seen le
  # An explicit CUDA_ARCH is the caller's decision and is honoured as given.
  if [ -n "$want" ] && [ "$want" != "auto" ]; then printf '%s\n' "$want"; return; fi
  nvcc_bin="$(command -v nvcc 2>/dev/null || printf '%s' "${CUDA_HOME:-}/bin/nvcc")"
  list="$("$nvcc_bin" --list-gpu-arch 2>/dev/null | sed 's/compute_//' | sort -n)"
  seen="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
  if [ -z "$list" ]; then printf '%s\n' "${seen:-100}"; return; fi
  if [ -n "$seen" ] && printf '%s\n' "$list" | grep -qx "$seen"; then
    printf '%s\n' "$seen"; return
  fi
  if [ -n "$seen" ]; then
    le="$(printf '%s\n' "$list" | awk -v d="$seen" '$1<=d{m=$1} END{if (m) print m}')"
    if [ -n "$le" ]; then printf '%s-virtual\n' "$le"; return; fi
    # The device is OLDER than anything this toolkit can build for -- an H100,
    # say, against a CUDA that has dropped sm_90.  Nothing here will run: PTX
    # JITs forward onto newer hardware only, never back onto older.  Falling
    # through to the toolkit's newest would hand back a binary that dies at its
    # first kernel launch, a long way from this script, so refuse instead.
    echo "resolve_cuda_arch: this CUDA can target [$(printf '%s ' $list)]," >&2
    echo "  but the GPU reports ${seen}, which is older than all of them." >&2
    echo "  Nothing will run here.  Install a CUDA that still supports" >&2
    echo "  sm_${seen}, or set CUDA_ARCH=<arch> to override this check." >&2
    return 1
  fi
  # No GPU visible: a login node, or a container started without --gpus.  There
  # is nothing to detect, so take the toolkit's newest and say so out loud.
  # This is the one branch that guesses, and the guess is wrong whenever the
  # machine that ends up running the binary is not the newest thing this CUDA
  # knows about.
  # No GPU visible: a login node, or a container started without --gpus.  Fall
  # back to sm_100 -- Grace Blackwell, what this project actually runs on -- and
  # NOT to the toolkit's newest.  The newest would be sm_120 under CUDA 12.8, a
  # consumer part nobody here builds for, and it would silently change what a
  # GPU-less container build produces.  This is the value these scripts used
  # before any of the detection existed, so a build with nothing to detect keeps
  # behaving exactly as it always did.
  echo "resolve_cuda_arch: no GPU visible; falling back to sm_${CUDA_ARCH_FALLBACK}." >&2
  echo "  Set CUDA_ARCH=<arch> if the target machine is something else." >&2
  printf '%s\n' "$CUDA_ARCH_FALLBACK"
}
CUDA_ARCH_RESOLVED="$(resolve_cuda_arch)" || exit 1
# CMAKE_CUDA_ARCHITECTURES reads "103-virtual" as PTX only.  HYPRE_CUDA_SM and
# AmgX want a bare number, so both forms are kept.
CUDA_ARCH_NUM="${CUDA_ARCH_RESOLVED%-virtual}"

# When the device is newer than anything this toolkit can name, build BOTH.
#
# CMake reads "103" as code=sm_103 and "103-virtual" as code=compute_103.  The
# cubin is what removes the JIT: cubins are forward-compatible across MINOR
# revisions within one major capability, so sm_103 machine code loads on a 10.7
# device directly.  The PTX rides along in case this generation's rules do not
# cover the gap, in which case the driver JIT-compiles exactly as before.
#
# Shipping both is strictly better than either alone. It costs binary size and
# nothing else.
case "$CUDA_ARCH_RESOLVED" in
  *-virtual)
    CUDA_ARCH_CMAKE="${CUDA_ARCH_NUM};${CUDA_ARCH_NUM}-virtual"
    echo "==> GPU newer than this CUDA: SASS for sm_${CUDA_ARCH_NUM} (loads directly) plus PTX as a fallback" ;;
  *)
    CUDA_ARCH_CMAKE="$CUDA_ARCH_RESOLVED"
    echo "==> building for sm_${CUDA_ARCH_NUM}" ;;
esac

JOBS="${JOBS:-16}"

# ----------------------- NVIDIA HPC SDK, on whichever architecture ---------
# The SDK installs under /opt/nvidia/hpc_sdk/Linux_<arch>/<version>, and this
# was hardcoded to Linux_aarch64/25.3 because that is the GB200 the results
# came from.  On x86_64 that directory does not exist, so CUDA_HOME was never
# set, the compilers never reached PATH, and the failure arrived much later as
# a missing header -- which is a poor way to learn the script is aarch64-only.
#
# Resolve the architecture from uname and take the newest installed version, so
# the same script works on both.  HPC_SDK_BASE still overrides everything.
_nvhpc_base() {
  local arch root
  case "$(uname -m)" in
    aarch64|arm64) arch=aarch64 ;;
    x86_64|amd64)  arch=x86_64  ;;
    *)             arch="$(uname -m)" ;;
  esac
  root="/opt/nvidia/hpc_sdk/Linux_${arch}"
  [ -d "$root" ] || return 0
  # sort -V so 25.3 beats 25.11 correctly and the newest wins.
  ls -1d "$root"/*/ 2>/dev/null | sed 's:/*$::' | sort -V | tail -1
}
HPC_SDK_BASE="${HPC_SDK_BASE:-$(_nvhpc_base)}"
if [[ -z "${CUDA_HOME:-}" ]]; then
  # The HPC SDK under /opt comes FIRST, and deliberately so.
  #
  # Its CUDA, its compilers and its HPCX are one matched set.  A system nvcc on
  # PATH -- /usr/local/cuda, say -- is very often a different version, and
  # building against that while linking MPI from the SDK is the kind of
  # mismatch that does not fail at link time.  It fails much later, as a
  # wrong answer or a segfault in a halo exchange.
  #
  # nvcc on PATH is the FALLBACK, for a machine with no SDK at all -- which is
  # the container built by build_container.sh, where CUDA 12.8 lives at
  # /usr/local/cuda and there is no HPC SDK whatsoever.
  if [[ -n "${HPC_SDK_BASE:-}" ]]; then
    _c="$(ls -1d "$HPC_SDK_BASE"/cuda/*/ 2>/dev/null | sed 's:/*$::' | sort -V | tail -1)"
    if [[ -n "$_c" ]]; then export CUDA_HOME="$_c"
    elif [[ -d "$HPC_SDK_BASE/cuda" ]]; then export CUDA_HOME="$HPC_SDK_BASE/cuda"; fi
    unset _c
  fi
  if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
fi
[[ -d "$HPC_SDK_BASE/compilers/bin" ]] && export PATH="$HPC_SDK_BASE/compilers/bin:$PATH"
if [[ -n "${CUDA_HOME:-}" ]]; then
  [[ -d "$CUDA_HOME/bin" ]] && export PATH="$CUDA_HOME/bin:$PATH"
  [[ -d "$CUDA_HOME/lib64" ]] && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

command -v git >/dev/null || { echo "ERROR: git not found"; exit 1; }
command -v cmake >/dev/null || { echo "ERROR: cmake not found"; exit 1; }
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found; set CUDA_HOME"; exit 1; }

if [[ -d "$HYPRE_SRC/.git" ]]; then
  echo "==> updating HYPRE checkout: $HYPRE_SRC"
  git -C "$HYPRE_SRC" fetch --all --tags --quiet
  git -C "$HYPRE_SRC" checkout --quiet "$HYPRE_REF"
  git -C "$HYPRE_SRC" pull --quiet --ff-only || true
else
  echo "==> cloning HYPRE ($HYPRE_REF) into $HYPRE_SRC"
  git clone --branch "$HYPRE_REF" https://github.com/hypre-space/hypre.git "$HYPRE_SRC"
fi

build_one() {
  local name="$1"
  local single="$2"
  local build_dir="$HYPRE_SRC/build-$name"
  local prefix="$HYPRE_PREFIX/$name"
  mkdir -p "$build_dir" "$prefix"

  echo "==> configuring HYPRE $name (single=$single, sm_$CUDA_ARCH_NUM)"
  cmake -S "$HYPRE_SRC/src" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DBUILD_SHARED_LIBS=ON \
    -DHYPRE_ENABLE_SHARED=ON \
    -DHYPRE_ENABLE_CUDA=ON \
    -DHYPRE_CUDA_SM="$CUDA_ARCH_NUM" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_CMAKE" \
    -DHYPRE_ENABLE_UMPIRE=ON \
    -DHYPRE_BUILD_UMPIRE=ON \
    -DHYPRE_ENABLE_GPU_AWARE_MPI=OFF \
    -DHYPRE_ENABLE_SINGLE="$single" \
    -DHYPRE_ENABLE_BIGINT=OFF \
    -DHYPRE_ENABLE_MIXEDINT=OFF

  echo "==> building HYPRE $name"
  cmake --build "$build_dir" --parallel "$JOBS"
  cmake --install "$build_dir"
}

build_one fp64 OFF
build_one fp32 ON

MPI_PREFIX=""
if command -v mpicc >/dev/null; then
  MPI_LIB_FLAG="$(mpicc --showme:link 2>/dev/null | tr ' ' '\n' | grep '^-L.*/lib$' | head -1 || true)"
  if [[ -n "$MPI_LIB_FLAG" ]]; then
    MPI_PREFIX="$(dirname "${MPI_LIB_FLAG#-L}")"
  fi
fi
if [[ -z "$MPI_PREFIX" ]]; then
  for d in "$HPC_SDK_BASE"/comm_libs/12.8/hpcx/hpcx-*/ompi; do
    [[ -d "$d/share/openmpi" ]] && MPI_PREFIX="$d" && break
  done
fi

cat > "$ENV_FILE" <<EOF
# source this before running HYPRE benchmarks
export HYPRE_FP64_DIR="$HYPRE_PREFIX/fp64"
export HYPRE_FP32_DIR="$HYPRE_PREFIX/fp32"
${MPI_PREFIX:+export OPAL_PREFIX="$MPI_PREFIX"}
${MPI_PREFIX:+export PATH="$MPI_PREFIX/bin:\${PATH:-}"}
export HYPRE_FP64_LIB="$HYPRE_PREFIX/fp64/lib"
export HYPRE_FP32_LIB="$HYPRE_PREFIX/fp32/lib"
export LD_LIBRARY_PATH="${MPI_PREFIX:+$MPI_PREFIX/lib:}\${LD_LIBRARY_PATH:-}"
EOF

echo "==> wrote $ENV_FILE"
echo "==> next:"
echo "    source $ENV_FILE"
echo "    python run_hypre_and_plot.py --logdir ./runs_hypre --sides 32,48,64,96"
