#!/usr/bin/env bash
#
# install_amgx_sm100.sh  (corrected)
# Build NVIDIA AmgX (single-GPU) for Blackwell datacenter GPUs (GB200 = sm_100),
# no sudo, into a writable prefix. Produces libamgxsh.so for the ctypes binding
# in mixed_precision_cpr_amgx.py.
#
# Defaults: clone + install INTO THE CURRENT DIRECTORY ($PWD), not $HOME.
# Sets up the same CUDA toolchain PhysicsNeMo uses (HPC SDK 25.3 / CUDA 12.8 on
# Grace aarch64) itself, and writes a sourceable amgx_env.sh with the AmgX +
# CUDA runtime lib paths (searching HPC SDK sbsa-linux + math_libs locations).
#
# Run inside your container with the PhysicsNeMo venv activated, on a node of
# the architecture you will run on (GB200 = Grace aarch64). sm_100 needs CUDA>=12.8.
#
# Usage:
#   ./install_amgx_sm100.sh
#   AMGX_SRC=/path/AMGX AMGX_PREFIX=/path/amgx-install CUDA_ARCH=100 JOBS=16 ./install_amgx_sm100.sh
#
set -euo pipefail

# ----------------------------- config -------------------------------------
# Default to the directory you launch from (override with env vars if you like).
AMGX_SRC="${AMGX_SRC:-$PWD/AMGX}"                  # source checkout (this folder)
AMGX_PREFIX="${AMGX_PREFIX:-$PWD/amgx-install}"    # install location (this folder)
ENV_FILE="${ENV_FILE:-$PWD/amgx_env.sh}"           # sourceable env (this folder)
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

AMGX_REF="${AMGX_REF:-main}"                        # branch/tag; main has Blackwell support
JOBS="${JOBS:-16}"                                  # parallel compile jobs (cap to avoid OOM)

# ----------------------- CUDA toolchain (HPC SDK 25.3 / CUDA 12.8) ---------
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
  [[ -d "$CUDA_HOME/bin" ]]   && export PATH="$CUDA_HOME/bin:$PATH"
  [[ -d "$CUDA_HOME/lib64" ]] && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "==> AmgX build config"
echo "    source      = $AMGX_SRC"
echo "    prefix      = $AMGX_PREFIX"
echo "    env file    = $ENV_FILE"
echo "    cuda arch   = sm_$CUDA_ARCH_NUM"
echo "    ref         = $AMGX_REF"
echo "    jobs        = $JOBS"
echo "    CUDA_HOME   = ${CUDA_HOME:-<unset, will use nvcc on PATH>}"

# ----------------------- prerequisite checks ------------------------------
command -v git >/dev/null || { echo "ERROR: git not found"; exit 1; }
command -v gcc >/dev/null || { echo "ERROR: gcc/g++ not found"; exit 1; }

NVCC="$(command -v nvcc || true)"
if [[ -z "$NVCC" && -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  NVCC="$CUDA_HOME/bin/nvcc"
fi
[[ -n "$NVCC" ]] || { echo "ERROR: nvcc not found. Set CUDA_HOME to your CUDA>=12.8."; exit 1; }
echo "==> nvcc: $NVCC"
"$NVCC" --version | sed -n 's/^.*release \([0-9.]*\).*/    CUDA release \1/p'

# sm_100 requires CUDA >= 12.8. `--list-gpu-arch` prints compute_XX; some nvcc
# builds reject `--list-gpu-code`, so capture with `|| true` to avoid pipefail
# poisoning the match. Word boundary so 100 does not match 101/1000.
arch_list="$( { "$NVCC" --list-gpu-arch; "$NVCC" --list-gpu-code; } 2>/dev/null || true )"
if printf '%s\n' "$arch_list" | grep -qE "(sm|compute)_${CUDA_ARCH}\b"; then
  echo "==> toolkit supports arch ${CUDA_ARCH} (compute_${CUDA_ARCH}/sm_${CUDA_ARCH}): OK"
else
  echo "ERROR: this nvcc cannot target arch ${CUDA_ARCH}."
  echo "       sm_100 needs CUDA >= 12.8. Point CUDA_HOME at a newer toolkit."
  echo "       Supported arches by this nvcc:"
  printf '%s\n' "$arch_list" | sed 's/^/         /'
  exit 1
fi

# ---- locate CUDA runtime + math libs (cudart, cusparse) for LD_LIBRARY_PATH ----
# HPC SDK on Grace puts libs under cuda/12.8/targets/sbsa-linux/lib and
# math_libs/12.8/targets/sbsa-linux/lib. Search a broad candidate list.
NVCC_REAL="$(readlink -f "$NVCC" 2>/dev/null || echo "$NVCC")"
CUDA_TK_ROOT="$(dirname "$(dirname "$NVCC_REAL")")"
_cand=(
  "$CUDA_TK_ROOT/lib64" "$CUDA_TK_ROOT/lib"
  "$CUDA_TK_ROOT/targets/sbsa-linux/lib" "$CUDA_TK_ROOT/targets/x86_64-linux/lib"
  "$HPC_SDK_BASE/cuda/12.8/lib64"
  "$HPC_SDK_BASE/cuda/12.8/targets/sbsa-linux/lib"
  "$HPC_SDK_BASE/cuda/12.8/targets/x86_64-linux/lib"
  "$HPC_SDK_BASE/cuda/lib64"
  "$HPC_SDK_BASE/math_libs/12.8/targets/sbsa-linux/lib"
  "$HPC_SDK_BASE/math_libs/12.8/targets/x86_64-linux/lib"
  "$HPC_SDK_BASE/math_libs/12.8/lib64"
  "$HPC_SDK_BASE/math_libs/lib64"
  "${CUDA_HOME:-}/lib64" "${CUDA_HOME:-}/lib"
  "${CUDA_HOME:-}/targets/sbsa-linux/lib" "${CUDA_HOME:-}/targets/x86_64-linux/lib"
  "/usr/local/cuda/lib64" "/usr/local/cuda/targets/sbsa-linux/lib"
)
_find_dir() {                       # $1 = glob; echo first candidate dir containing it
  local pat="$1" d
  for d in "${_cand[@]}"; do
    [[ -n "$d" && -d "$d" ]] || continue
    if compgen -G "$d/$pat" >/dev/null 2>&1; then echo "$d"; return 0; fi
  done
  return 1
}
CUDART_DIR="$(_find_dir 'libcudart.so*' || true)"
CUSPARSE_DIR="$(_find_dir 'libcusparse.so*' || true)"
# Build a colon list of the unique dirs we found.
RT_LIBS=""
for d in "$CUDART_DIR" "$CUSPARSE_DIR"; do
  [[ -n "$d" ]] || continue
  case ":$RT_LIBS:" in *":$d:"*) ;; *) RT_LIBS="${RT_LIBS:+$RT_LIBS:}$d";; esac
done
echo "    cudart dir  = ${CUDART_DIR:-<none>}"
echo "    cusparse dir= ${CUSPARSE_DIR:-<none>}"

# CMake >= 3.24 for clean CMAKE_CUDA_ARCHITECTURES handling of new arches.
need_cmake=1
if command -v cmake >/dev/null; then
  cm_ver="$(cmake --version | sed -n 's/^cmake version \([0-9.]*\).*/\1/p')"
  cm_major="${cm_ver%%.*}"; cm_minor="$(echo "$cm_ver" | cut -d. -f2)"
  if (( cm_major > 3 || (cm_major == 3 && cm_minor >= 24) )); then
    echo "==> cmake $cm_ver: OK"; need_cmake=0
  else
    echo "==> cmake $cm_ver too old (<3.24), will pip-install a newer one"
  fi
fi
if (( need_cmake == 1 )); then
  python -m pip install --upgrade "cmake>=3.28" ninja
  hash -r
  echo "==> cmake $(cmake --version | head -1)"
fi

# ----------------------------- fetch --------------------------------------
if [[ -d "$AMGX_SRC/.git" ]]; then
  echo "==> updating existing checkout at $AMGX_SRC"
  git -C "$AMGX_SRC" fetch --all --tags --quiet
  git -C "$AMGX_SRC" checkout --quiet "$AMGX_REF"
  git -C "$AMGX_SRC" pull --quiet --ff-only || true
else
  echo "==> cloning AmgX ($AMGX_REF) into $AMGX_SRC"
  git clone --branch "$AMGX_REF" https://github.com/NVIDIA/AMGX.git "$AMGX_SRC"
fi

# ----------------------------- configure ----------------------------------
BUILD_DIR="$AMGX_SRC/build"
mkdir -p "$BUILD_DIR"
echo "==> configuring (single-GPU, no MPI, sm_${CUDA_ARCH})"
cmake -S "$AMGX_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_CMAKE" \
  -DCUDA_ARCH="$CUDA_ARCH_NUM" \
  -DCMAKE_NO_MPI=True \
  -DCMAKE_INSTALL_PREFIX="$AMGX_PREFIX"

# ----------------------------- build/install ------------------------------
echo "==> building with $JOBS jobs (this takes a while)"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
echo "==> installing to $AMGX_PREFIX"
cmake --install "$BUILD_DIR"

LIBDIR=""
for d in "$AMGX_PREFIX/lib" "$AMGX_PREFIX/lib64"; do
  [[ -f "$d/libamgxsh.so" ]] && LIBDIR="$d" && break
done
[[ -n "$LIBDIR" ]] || { echo "ERROR: libamgxsh.so not found under $AMGX_PREFIX"; exit 1; }
echo "==> built: $LIBDIR/libamgxsh.so"

# ----------------------------- env + verify -------------------------------
{
  echo "# source this (and your PhysicsNeMo venv) before running the benchmark"
  [[ -n "${CUDA_HOME:-}" ]] && echo "export CUDA_HOME=\"$CUDA_HOME\""
  echo "export AMGX_DIR=\"$AMGX_SRC\""
  echo "export AMGX_BUILD_DIR=\"$BUILD_DIR\""
  if [[ -n "$RT_LIBS" ]]; then
    echo "export LD_LIBRARY_PATH=\"$LIBDIR:$RT_LIBS:\${LD_LIBRARY_PATH:-}\""
  else
    echo "export LD_LIBRARY_PATH=\"$LIBDIR:\${LD_LIBRARY_PATH:-}\""
  fi
} > "$ENV_FILE"
echo "==> wrote env file: $ENV_FILE"

# shellcheck disable=SC1090
source "$ENV_FILE"
echo "==> verifying load via ctypes"
python - <<'PY'
import ctypes, ctypes.util
name = ctypes.util.find_library("amgxsh") or "libamgxsh.so"
lib = ctypes.CDLL(name)
lib.AMGX_initialize(); lib.AMGX_finalize()
print("OK: libamgxsh.so loads and AMGX_initialize/finalize succeed")
PY

cat <<EOF

==============================================================
 AmgX (sm_${CUDA_ARCH}) installed under: $AMGX_PREFIX
 Each NEW session: activate the PhysicsNeMo venv, then:
     source $ENV_FILE
 Then run the experiments + plot:
     python run_and_plot.py --logdir ./runs --sides 24 --pathB          # smoke test
     python run_and_plot.py --logdir ./runs --sides 32,48,64,96 --pathB # full sweep
==============================================================
EOF