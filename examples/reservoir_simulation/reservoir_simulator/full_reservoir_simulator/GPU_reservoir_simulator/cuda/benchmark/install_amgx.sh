#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# install_amgx.sh — fetch, build and install AmgX, then print how to link it.
#
#   ./install_amgx.sh                 # clone, build, install beside the source
#   AMGX_PREFIX=/opt/amgx ./install_amgx.sh
#   AMGX_SRC=/path/to/AMGX ./install_amgx.sh    # reuse a clone you already have
#
# On success it writes amgx_env.sh next to the install, so a later shell only
# needs:  source <prefix>/amgx_env.sh
#
# Why a script rather than three commands: AmgX needs a CMake older than 4 (the
# same wall OPM hit here), the GB200 needs an SM the defaults do not include,
# and its install target has historically missed the public headers.  Each of
# those is a silent half-failure that shows up much later as "cannot open
# source file amgx_c.h" or a link against a library built for the wrong GPU.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$(cd "$HERE/.." && pwd)"
# Default to a sibling of the simulator tree, not /usr/local: this needs no
# root and keeps the build with the project it belongs to.
BUILD_ROOT="${BUILD_ROOT:-$(cd "$CUDA_DIR/../.." && pwd)}"
AMGX_SRC="${AMGX_SRC:-$BUILD_ROOT/AMGX}"
AMGX_PREFIX="${AMGX_PREFIX:-$BUILD_ROOT/amgx-install}"
AMGX_BRANCH="${AMGX_BRANCH:-main}"
CMAKE_FETCH_VERSION="${CMAKE_FETCH_VERSION:-3.31.6}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"

die() { echo "ERROR: $*" >&2; exit 1; }
say() { printf '\n== %s ==\n' "$*"; }
trap 'rc=$?; [[ $rc -ne 0 ]] && echo "install_amgx.sh failed at line $LINENO (rc=$rc)" >&2' ERR

# --- CMake ------------------------------------------------------------------
# AmgX does not build under CMake 4: it still uses policies and variables that
# 4.x removed.  Same problem, same fix, as install_opm_gpu.sh.
cmake_major() { "$1" --version 2>/dev/null | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/'; }

pick_cmake() {
  local c v
  if [[ -n "${CMAKE:-}" ]]; then printf '%s\n' "$CMAKE"; return 0; fi
  for c in cmake3 cmake; do
    command -v "$c" >/dev/null 2>&1 || continue
    v="$(cmake_major "$c" || true)"
    if [[ -n "$v" && "$v" -lt 4 ]]; then command -v "$c"; return 0; fi
  done
  [[ -x "$BUILD_ROOT/cmake/bin/cmake" ]] && { printf '%s\n' "$BUILD_ROOT/cmake/bin/cmake"; return 0; }
  return 1
}

fetch_cmake() {
  local arch tgz url
  case "$(uname -m)" in
    aarch64|arm64) arch=linux-aarch64 ;;
    x86_64)        arch=linux-x86_64 ;;
    *) die "no prebuilt CMake for $(uname -m); set CMAKE=/path/to/cmake3" ;;
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
# Getting this wrong does not fail the build; it fails at run time, with a
# kernel-launch error a long way from here.
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
CUDA_ARCH_FALLBACK="${CUDA_ARCH_FALLBACK:-}"
resolve_cuda_arch() {
  local want="${CUDA_ARCH:-}" nvcc_bin list seen le
  # An explicit CUDA_ARCH is the caller's decision and is honoured as given.
  if [ -n "$want" ] && [ "$want" != "auto" ]; then printf '%s\n' "$want"; return; fi
  nvcc_bin="$(command -v nvcc 2>/dev/null || printf '%s' "${CUDA_HOME:-}/bin/nvcc")"
  list="$("$nvcc_bin" --list-gpu-arch 2>/dev/null | sed 's/compute_//' | sort -n)"
  seen="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
  if [ -z "$list" ] && [ -n "${seen:-$CUDA_ARCH_FALLBACK}" ]; then printf '%s\n' "${seen:-$CUDA_ARCH_FALLBACK}"; return; fi
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
  # No GPU visible: nothing to detect, so stop rather than build for the wrong one.
  if [ -z "$CUDA_ARCH_FALLBACK" ]; then
    echo "resolve_cuda_arch: no GPU is visible, so there is nothing to build for." >&2
    echo "  Run this inside ./build_container.sh, which passes the GPU through," >&2
    echo "  and check that nvidia-smi works there." >&2
    return 1
  fi
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
echo "source     : $AMGX_SRC"
echo "prefix     : $AMGX_PREFIX"

command -v nvcc >/dev/null 2>&1 || die "nvcc not on PATH"
command -v mpicxx >/dev/null 2>&1 || echo "  [warn] mpicxx not found; AmgX will build without MPI and multi-GPU will not work"

# --- source -----------------------------------------------------------------
if [[ ! -d "$AMGX_SRC/.git" ]]; then
  say "cloning AmgX ($AMGX_BRANCH)"
  git clone --depth 1 -b "$AMGX_BRANCH" https://github.com/NVIDIA/AMGX.git "$AMGX_SRC" \
    || die "clone failed; set AMGX_SRC to a copy you already have"
else
  echo "source     : reusing existing clone"
fi

# --- build ------------------------------------------------------------------
BUILD="$AMGX_SRC/build"
mkdir -p "$BUILD"
say "configuring"
# CUDA_ARCH is what older AmgX reads; CMAKE_CUDA_ARCHITECTURES is what newer
# CMake reads.  Passing both is harmless -- an unused cache entry is a warning,
# a missing one is a library that will not run on this GPU.
"$CMAKE_BIN" -S "$AMGX_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$AMGX_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_CMAKE" \
  -DCUDA_ARCH="$CUDA_ARCH_NUM" \
  -DCMAKE_NO_MPI=0 \
  -DAMGX_NO_RPATH=0 \
  || die "cmake configure failed"

say "building with $JOBS jobs (this takes a while)"
"$CMAKE_BIN" --build "$BUILD" -j "$JOBS" || die "build failed"

# --- install ----------------------------------------------------------------
# AmgX's install target has not always shipped the public headers, so verify
# and stage by hand rather than trusting it.
say "installing to $AMGX_PREFIX"
"$CMAKE_BIN" --install "$BUILD" >/dev/null 2>&1 || true
mkdir -p "$AMGX_PREFIX/include" "$AMGX_PREFIX/lib"

stage_headers() {
  local h found=0
  for h in amgx_c.h amgx_config.h amgx_capi.h; do
    [[ -f "$AMGX_PREFIX/include/$h" ]] && { found=$((found+1)); continue; }
    local src
    src="$(find "$AMGX_SRC" -name "$h" -not -path '*/build/*' 2>/dev/null | head -1)"
    if [[ -n "$src" ]]; then cp -f "$src" "$AMGX_PREFIX/include/"; found=$((found+1)); fi
  done
  echo "  headers  : $found staged"
}
stage_libs() {
  local n=0 f
  while IFS= read -r f; do
    cp -Pf "$f" "$AMGX_PREFIX/lib/" && n=$((n+1))
  done < <(find "$BUILD" \( -name 'libamgxsh.so*' -o -name 'libamgx.a' \) 2>/dev/null)
  echo "  libraries: $n staged"
}
stage_headers
stage_libs

[[ -f "$AMGX_PREFIX/include/amgx_c.h" ]] || die "no amgx_c.h under $AMGX_PREFIX/include"
[[ -e "$AMGX_PREFIX/lib/libamgxsh.so" ]] || \
  die "no libamgxsh.so under $AMGX_PREFIX/lib (the simulator links the shared build)"

# --- env file ---------------------------------------------------------------
cat > "$AMGX_PREFIX/amgx_env.sh" <<EOF
# source this before building the simulator with AmgX support
export AMGX_DIR="$AMGX_PREFIX"
export LD_LIBRARY_PATH="\$AMGX_DIR/lib:\${LD_LIBRARY_PATH:-}"
EOF

say "done"
cat <<EOF
AmgX is installed at:
  $AMGX_PREFIX

Build the simulator against it:

  source $AMGX_PREFIX/amgx_env.sh
  cd $CUDA_DIR
  make clean && make -j

'make' prints the AmgX path it found.  A run then reports:
  Linear   : stage-1 multigrid from AmgX
  AmgX     : dFFI (FP32), aggregation AMG, 1 V-cycle per apply
EOF
