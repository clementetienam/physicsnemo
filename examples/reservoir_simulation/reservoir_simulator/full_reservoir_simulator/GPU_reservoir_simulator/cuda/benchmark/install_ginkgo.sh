#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# install_ginkgo.sh -- fetch, build and install Ginkgo with CUDA and half
# precision, then print how to link it.
#
#   ./install_ginkgo.sh                     # clone, build, install beside the source
#   GINKGO_PREFIX=/opt/ginkgo ./install_ginkgo.sh
#   GINKGO_VERSION=v1.11.0 ./install_ginkgo.sh   # a tag or branch (default v1.11.0)
#   CUDA_ARCH=100 ./install_ginkgo.sh       # pin the SM (default: detect, else 100)
#   GINKGO_SRC=/path/to/ginkgo ./install_ginkgo.sh   # reuse a clone you already have
#   JOBS=16 ./install_ginkgo.sh             # parallel compile jobs
#
# On success it writes ginkgo_env.sh next to the install, so a later shell only
# needs:  source <prefix>/ginkgo_env.sh
#
# What the simulator needs from Ginkgo, and what this script therefore insists
# on rather than leaves to defaults:
#
#   * the CUDA executor (GINKGO_BUILD_CUDA=ON) -- the backend runs on the GPU;
#   * half precision (GINKGO_ENABLE_HALF=ON) -- LINSOLVER GINKGO runs its
#     multigrid in FP16 by default.  It is on by default except under MSVC,
#     and the build is checked for it afterwards rather than assumed;
#   * 1.11.0 or newer on CUDA 13 (the first release validated against it);
#     1.9.0 is the oldest with half precision at all.
#
# Tests, examples and benchmarks are switched off: they download googletest,
# gflags and nlohmann-json, which fails on a compute node without internet,
# and they roughly double an already long CUDA build.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$(cd "$CUDA_DIR/../.." && pwd)}"
GINKGO_SRC="${GINKGO_SRC:-$BUILD_ROOT/ginkgo}"
GINKGO_PREFIX="${GINKGO_PREFIX:-$BUILD_ROOT/ginkgo-install}"
GINKGO_VERSION="${GINKGO_VERSION:-v1.11.0}"
CMAKE_FETCH_VERSION="${CMAKE_FETCH_VERSION:-3.31.6}"
# Ginkgo's CUDA translation units are large; a job per core can exhaust host
# memory on a big node.  Cap the default, and let JOBS override it.
_nproc="$(nproc 2>/dev/null || echo 8)"
JOBS="${JOBS:-$(( _nproc < 32 ? _nproc : 32 ))}"

die() { echo "ERROR: $*" >&2; exit 1; }
say() { printf '\n== %s ==\n' "$*"; }
trap 'rc=$?; [[ $rc -ne 0 ]] && echo "install_ginkgo.sh failed at line $LINENO (rc=$rc)" >&2' ERR

# --- CMake ------------------------------------------------------------------
# Ginkgo needs 3.16 or newer.  A system CMake is used when it is new enough;
# otherwise a prebuilt one is unpacked under the build root, as the AmgX and
# OPM installers do.
cmake_version() { "$1" --version 2>/dev/null | head -1 | sed -E 's/[^0-9]*([0-9]+)\.([0-9]+).*/\1 \2/'; }

pick_cmake() {
  local c maj min
  if [[ -n "${CMAKE:-}" ]]; then printf '%s\n' "$CMAKE"; return 0; fi
  for c in cmake3 cmake "$BUILD_ROOT/cmake/bin/cmake"; do
    command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]] || continue
    read -r maj min <<<"$(cmake_version "$c")"
    [[ -n "${maj:-}" ]] || continue
    if (( maj > 3 || (maj == 3 && min >= 18) )); then
      command -v "$c" 2>/dev/null || printf '%s\n' "$c"
      return 0
    fi
  done
  return 1
}

fetch_cmake() {
  local arch tgz url
  case "$(uname -m)" in
    aarch64|arm64) arch=linux-aarch64 ;;
    x86_64)        arch=linux-x86_64 ;;
    *) die "no prebuilt CMake for $(uname -m); set CMAKE=/path/to/cmake" ;;
  esac
  tgz="cmake-${CMAKE_FETCH_VERSION}-${arch}.tar.gz"
  url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_FETCH_VERSION}/${tgz}"
  say "fetching CMake ${CMAKE_FETCH_VERSION}"
  mkdir -p "$BUILD_ROOT/cmake"
  curl -fsSL "$url" | tar xz -C "$BUILD_ROOT/cmake" --strip-components=1 \
    || die "could not download $url"
}

CMAKE_BIN="$(pick_cmake || true)"
if [[ -z "$CMAKE_BIN" ]]; then
  fetch_cmake
  CMAKE_BIN="$(pick_cmake)" || die "still no usable CMake"
fi
echo "cmake      : $CMAKE_BIN ($("$CMAKE_BIN" --version | head -1))"

# --- which GPU to build for -------------------------------------------------
# The same resolution install_amgx.sh uses (see the long note there): the
# detected architecture when this CUDA can target it, the newest one not newer
# than the device as SASS plus PTX when the GPU is newer than the toolkit
# (Vera Rubin's sm_107), and sm_100 when no GPU is visible.
CUDA_ARCH_FALLBACK="${CUDA_ARCH_FALLBACK:-100}"
resolve_cuda_arch() {
  local want="${CUDA_ARCH:-}" nvcc_bin list seen le
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
    echo "resolve_cuda_arch: the GPU reports ${seen}, older than every architecture" >&2
    echo "  this CUDA can target.  Set CUDA_ARCH=<arch> to override." >&2
    return 1
  fi
  echo "resolve_cuda_arch: no GPU visible; falling back to sm_${CUDA_ARCH_FALLBACK}." >&2
  printf '%s\n' "$CUDA_ARCH_FALLBACK"
}
CUDA_ARCH_RESOLVED="$(resolve_cuda_arch)" || exit 1
CUDA_ARCH_NUM="${CUDA_ARCH_RESOLVED%-virtual}"
case "$CUDA_ARCH_RESOLVED" in
  *-virtual) CUDA_ARCH_CMAKE="${CUDA_ARCH_NUM};${CUDA_ARCH_NUM}-virtual" ;;
  *)         CUDA_ARCH_CMAKE="$CUDA_ARCH_RESOLVED" ;;
esac
# Ginkgo's own selector wants "major.minor" (10.0), not CMake's "100".
GINKGO_ARCH="${GINKGO_CUDA_ARCH:-${CUDA_ARCH_NUM:0:${#CUDA_ARCH_NUM}-1}.${CUDA_ARCH_NUM: -1}}"
echo "==> building for sm_${CUDA_ARCH_NUM} (CMake: ${CUDA_ARCH_CMAKE}; Ginkgo: ${GINKGO_ARCH})"
echo "source     : $GINKGO_SRC"
echo "prefix     : $GINKGO_PREFIX"
echo "version    : $GINKGO_VERSION"
echo "jobs       : $JOBS"

command -v nvcc >/dev/null 2>&1 || die "nvcc not on PATH"
echo "nvcc       : $(nvcc --version | grep -o 'release [0-9.]*')"

# --- source -----------------------------------------------------------------
if [[ ! -d "$GINKGO_SRC/.git" ]]; then
  say "cloning Ginkgo ($GINKGO_VERSION)"
  git clone --depth 1 -b "$GINKGO_VERSION" \
      https://github.com/ginkgo-project/ginkgo.git "$GINKGO_SRC" \
    || die "clone failed; set GINKGO_SRC to a copy you already have"
else
  echo "source     : reusing existing clone ($(git -C "$GINKGO_SRC" describe --tags 2>/dev/null || echo unknown))"
fi

# --- configure and build ----------------------------------------------------
BUILD="$GINKGO_SRC/build-nvrs"
mkdir -p "$BUILD"

# --- Thrust / CUB / libcudacxx CMake packages ---------------------------------
# Ginkgo's cmake/cuda.cmake calls find_package(Thrust).  CUDA 13 ships Thrust,
# CUB and libcudacxx as the CCCL bundle, and on an sbsa (Grace / Vera) install
# their CMake configs live under targets/<arch>/lib/cmake, which is not on
# CMake's default search path:
#
#   Could not find a package configuration file provided by "Thrust"
#
# Find the configs wherever this toolkit put them and hand the directories to
# CMake explicitly.  A clean (pre-existing) build directory is removed first,
# because a failed configure caches Thrust_DIR-NOTFOUND.
NVCC_BIN="$(command -v nvcc)"
CUDA_ROOT="${CUDA_HOME:-$(cd "$(dirname "$NVCC_BIN")/.." && pwd)}"
find_cfg() {              # find_cfg <package-lowercase> <PackageCamel>
  find -L "$CUDA_ROOT" /usr/local/cuda /usr/lib /usr/share /opt/nvidia \
       \( -name "$1-config.cmake" -o -name "$2Config.cmake" \) 2>/dev/null \
    | head -1
}
PKG_ARGS=()
PREFIXES=""
for pair in "thrust Thrust" "cub CUB" "libcudacxx libcudacxx" "cccl CCCL"; do
  set -- $pair
  cfg="$(find_cfg "$1" "$2")"
  if [[ -n "$cfg" ]]; then
    dir="$(dirname "$cfg")"
    PKG_ARGS+=("-D${2}_DIR=$dir")
    PREFIXES="${PREFIXES:+$PREFIXES;}$(dirname "$dir")"
    echo "  $2 config  : $dir"
  fi
done
[[ -n "$PREFIXES" ]] && PKG_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIXES")
if ! printf '%s\n' "${PKG_ARGS[@]}" | grep -q "Thrust_DIR"; then
  echo "  [warn] no thrust-config.cmake found under $CUDA_ROOT; configure may fail." >&2
  echo "         Set Thrust_DIR=/path/to/lib/cmake/thrust and re-run." >&2
fi
[[ -n "${Thrust_DIR:-}" ]] && PKG_ARGS+=("-DThrust_DIR=$Thrust_DIR")
if [[ -f "$BUILD/CMakeCache.txt" ]] && grep -q "Thrust_DIR:PATH=Thrust_DIR-NOTFOUND" "$BUILD/CMakeCache.txt"; then
  echo "  clearing the failed configure in $BUILD"
  rm -rf "$BUILD"; mkdir -p "$BUILD"
fi

say "configuring"
"$CMAKE_BIN" -S "$GINKGO_SRC" -B "$BUILD" \
  "${PKG_ARGS[@]}" \
  -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$GINKGO_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_CMAKE" \
  -DGINKGO_CUDA_ARCHITECTURES="$GINKGO_ARCH" \
  -DGINKGO_BUILD_CUDA=ON \
  -DGINKGO_BUILD_REFERENCE=ON \
  -DGINKGO_BUILD_OMP="${GINKGO_BUILD_OMP:-ON}" \
  -DGINKGO_BUILD_MPI=OFF \
  -DGINKGO_BUILD_HIP=OFF \
  -DGINKGO_BUILD_SYCL=OFF \
  -DGINKGO_BUILD_HWLOC=OFF \
  -DGINKGO_ENABLE_HALF=ON \
  -DGINKGO_ENABLE_BFLOAT16=ON \
  -DGINKGO_BUILD_TESTS=OFF \
  -DGINKGO_BUILD_EXAMPLES=OFF \
  -DGINKGO_BUILD_BENCHMARKS=OFF \
  -DGINKGO_BUILD_DOC=OFF \
  -DBUILD_SHARED_LIBS=ON \
  || die "cmake configure failed"

say "building with $JOBS jobs (the CUDA part takes a long while)"
"$CMAKE_BIN" --build "$BUILD" -j "$JOBS" || die "build failed (try a smaller JOBS if the compiler was killed)"

say "installing to $GINKGO_PREFIX"
"$CMAKE_BIN" --install "$BUILD" || die "install failed"

# --- verify -----------------------------------------------------------------
[[ -f "$GINKGO_PREFIX/include/ginkgo/ginkgo.hpp" ]] || die "no ginkgo.hpp under $GINKGO_PREFIX/include"
LIBDIR=""
for d in "$GINKGO_PREFIX/lib" "$GINKGO_PREFIX/lib64"; do
  [[ -e "$d/libginkgo.so" ]] && { LIBDIR="$d"; break; }
done
[[ -n "$LIBDIR" ]] || die "no libginkgo.so under $GINKGO_PREFIX/lib or lib64"
[[ -e "$LIBDIR/libginkgo_cuda.so" ]] || die "no libginkgo_cuda.so: Ginkgo was built without CUDA"
CFG="$GINKGO_PREFIX/include/ginkgo/config.hpp"
if grep -qE '^#define[[:space:]]+GINKGO_ENABLE_HALF[[:space:]]+1' "$CFG"; then
  echo "  half precision : enabled"
else
  die "this Ginkgo was configured without half precision ($CFG); LINSOLVER GINKGO needs it"
fi
echo "  libraries      : $(cd "$LIBDIR" && ls libginkgo*.so | tr '\n' ' ')"

# --- env file ---------------------------------------------------------------
cat > "$GINKGO_PREFIX/ginkgo_env.sh" <<EOF
# source this before building the simulator with Ginkgo support
export GINKGO_DIR="$GINKGO_PREFIX"
export LD_LIBRARY_PATH="$LIBDIR:\${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$GINKGO_PREFIX:\${CMAKE_PREFIX_PATH:-}"
EOF

say "done"
cat <<EOF
Ginkgo is installed at:
  $GINKGO_PREFIX

Build the simulator against it:

  source $GINKGO_PREFIX/ginkgo_env.sh
  cd $CUDA_DIR
  make clean && make -j

'make' prints the Ginkgo path it found.  A deck with

  LINSOLVER  GINKGO
  GINKGOPREC 16          -- 16 (default) or 32

then reports:
  Linear   : stage-1 multigrid from Ginkgo, FP16
  Ginkgo   : 1.11.0 (half precision enabled), FP16 PGM aggregation AMG, V-cycle, ...
EOF
