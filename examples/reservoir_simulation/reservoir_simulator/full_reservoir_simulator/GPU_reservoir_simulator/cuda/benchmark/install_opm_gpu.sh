#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
#
# install_opm_gpu.sh — build OPM Flow 2024.10 with CUDA, for the NVRS benchmark.
#
# This is the GPU-only path.  The interactive CPU/CUDA prompt is gone: the
# benchmark compares a GPU simulator against a GPU-enabled Flow, so a CPU build
# would silently answer a different question, and a script that can quietly
# fall back to one is a script that will.  No GPU means no build, and the
# script says so and stops.
#
#   ./install_opm_gpu.sh                      # sources beside the checkout,
#                                             # install into /usr/local if it
#                                             # is writable, else ~/opm
#   PREFIX=$HOME/opm ./install_opm_gpu.sh     # choose the install prefix
#   BUILD_ROOT=/scratch/opm ./install_opm_gpu.sh   # choose where to clone
#
set -euo pipefail

# `set -e` kills the script without printing anything, which turns a one-line
# mistake into "it just returns to the prompt".  This build previously died in
# exactly that way on a machine where one CUDA library sat in a different
# directory.  Say where it died and what the status was.
trap 'rc=$?; if [[ $rc -ne 0 ]]; then
        echo >&2
        echo "ERROR: aborted at line $LINENO with status $rc" >&2
        echo "       (command: ${BASH_COMMAND})" >&2
      fi' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Nothing here is hard-coded to one machine.  The sources go beside the
# repository, not at an absolute path that only exists in one container, and
# the install goes somewhere writable rather than assuming root owns /usr.
#
#   benchmark/ -> cuda/ -> GPU_reservoir_simulator/ -> full_reservoir_simulator/
#
# so four levels up is the checkout root, and the OPM tree becomes its sibling.
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
DEFAULT_BUILD_ROOT="$(dirname "$REPO_ROOT")/opm_2024_10_src"
BUILD_ROOT="${BUILD_ROOT:-$DEFAULT_BUILD_ROOT}"

# /usr needs root.  Use it when we have it, otherwise a per-user prefix, and
# say which so the PATH advice at the end is right.
if [[ -n "${PREFIX:-}" ]]; then
  INSTALL_DIR="$PREFIX"
elif [[ -w /usr/local ]]; then
  INSTALL_DIR=/usr/local
else
  INSTALL_DIR="$HOME/opm"
fi

TAG="${OPM_TAG:-release/2024.10/final}"
MODULES=(opm-common opm-grid opm-simulators opm-upscaling)

# `cmd | head -1` is a trap under `set -o pipefail`: head closes the pipe after
# the first line, the producer takes SIGPIPE, and the pipeline reports 141.
# Combined with `set -e` that ends the script with no message at all.  Read the
# first line without a pipe instead.
first_line() {
  local out
  out="$("$@" 2>/dev/null)" || return 1
  printf '%s\n' "${out%%$'\n'*}"
}

# ---------------------------------------------------------------------------
# CMake 3.x is required, and a too-NEW CMake is the failure people hit.
#
# OPM 2024.10 does two things CMake 4.0 removed outright:
#
#   cmake_policy(SET CMP0026 OLD)      OLD behaviour for policies introduced
#                                      before 3.5 no longer exists
#   get_target_property(... LOCATION)  reading LOCATION is now a hard error
#
# On CMake 4.x the configure dies with a wall of "Policy CMP0026 may not be set
# to OLD behavior" and "The LOCATION property may not be read from target ...".
# No flag restores them; CMAKE_POLICY_VERSION_MINIMUM lets an old project
# configure but does not bring LOCATION back.  So find or fetch a 3.x.
# ---------------------------------------------------------------------------
CMAKE_MIN=3.16
CMAKE_FETCH_VERSION="${CMAKE_FETCH_VERSION:-3.31.6}"

cmake_major() { first_line "$1" --version | sed -n 's/^cmake version \([0-9]*\).*/\1/p'; }

pick_cmake() {
  local c v
  # An explicit choice wins.
  if [[ -n "${CMAKE:-}" ]]; then printf '%s\n' "$CMAKE"; return 0; fi
  for c in cmake3 cmake; do
    command -v "$c" >/dev/null 2>&1 || continue
    v="$(cmake_major "$c" || true)"
    if [[ -n "$v" && "$v" -lt 4 ]]; then command -v "$c"; return 0; fi
  done
  # Any previously fetched copy.
  if [[ -x "$BUILD_ROOT/cmake/bin/cmake" ]]; then
    printf '%s\n' "$BUILD_ROOT/cmake/bin/cmake"; return 0
  fi
  return 1
}

fetch_cmake() {
  local arch tgz url dest
  case "$(uname -m)" in
    aarch64|arm64) arch=linux-aarch64 ;;
    x86_64)        arch=linux-x86_64 ;;
    *) echo "ERROR: no prebuilt CMake for $(uname -m)." >&2; return 1 ;;
  esac
  tgz="cmake-${CMAKE_FETCH_VERSION}-${arch}.tar.gz"
  url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_FETCH_VERSION}/${tgz}"
  dest="$BUILD_ROOT/cmake"
  echo "Fetching CMake ${CMAKE_FETCH_VERSION} for ${arch}" >&2
  mkdir -p "$BUILD_ROOT"
  # No sudo, no system package manager: a self-contained copy under BUILD_ROOT
  # that only this build uses.
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$BUILD_ROOT/$tgz" || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$BUILD_ROOT/$tgz" || return 1
  else
    echo "ERROR: neither curl nor wget is available to fetch CMake." >&2
    return 1
  fi
  rm -rf "$dest" && mkdir -p "$dest"
  tar -xzf "$BUILD_ROOT/$tgz" -C "$dest" --strip-components=1 || return 1
  rm -f "$BUILD_ROOT/$tgz"
  [[ -x "$dest/bin/cmake" ]] || return 1
  printf '%s\n' "$dest/bin/cmake"
}

echo "=== OPM Flow ${TAG}, CUDA build ==="

# --- GPU is mandatory ------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found.  This script builds the CUDA version of" >&2
  echo "       Flow only; there is nothing to fall back to." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found.  Load the CUDA toolkit first." >&2
  exit 1
fi

# The architecture is resolved the same way as in the HYPRE and AmgX scripts:
# the newest target this nvcc knows that is not newer than the GPU, as PTX when
# there is no exact match.  Taking nvidia-smi at its word instead is what put
# -DCMAKE_CUDA_ARCHITECTURES=107 in front of a CUDA that has no sm_107.
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
GPU_ARCH="$CUDA_ARCH_NUM"
GPU_ARCH_CMAKE="$CUDA_ARCH_CMAKE"
CUDA_VERSION="$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | awk '{print $2}' || true)"
# opm-simulators gates its cusparse solver on CMake finding the CUDA toolkit,
# not on -DUSE_CUDA (which is not one of its options).  USE_BDA_BRIDGE is ON by
# default, so a build with no GPU solver almost always means CUDA was simply
# not found at configure time -- point CMake at it explicitly.
#
# Pointing CMake at a *root* is not enough.  opm-simulators still goes through
# the deprecated FindCUDA module, and the variable that has to resolve is
# CUDA_TOOLKIT_INCLUDE.  Given a root with no include/cuda_runtime.h under it,
# the configure gets far enough to add CUDA include directories to every target
# and then fails on all of them at once with
#
#     Found relative path while evaluating include directories of "flow":
#       "CUDA_TOOLKIT_INCLUDE-NOTFOUND"
#
# which is worse than not configuring CUDA at all.  So find a directory that
# actually holds the headers, and if there is none, build without CUDA rather
# than half with it.
# The header is not always at <root>/include.  The NVIDIA HPC SDK ships CUDA
# as
#
#   /opt/nvidia/hpc_sdk/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/include
#
# so a search that only looks at <root>/include reports "no CUDA" on a machine
# that plainly has it.  Look for the header itself, and report the include and
# library directories separately from the root, because in that layout they are
# not under it.
CUDA_ROOT="" CUDA_INC="" CUDA_LIB=""
find_cuda_paths() {
  local roots=() r inc lib nvcc_real
  roots+=("${CUDA_HOME:-}" "${CUDA_PATH:-}")
  if command -v nvcc >/dev/null 2>&1; then
    # Resolve symlinks: nvcc is often a wrapper several directories away from
    # the toolkit it belongs to.
    nvcc_real="$(readlink -f "$(command -v nvcc)")"
    roots+=("$(dirname "$(dirname "$nvcc_real")")")
  fi
  roots+=(/usr/local/cuda /usr/lib/cuda)
  for r in /usr/local/cuda-* /opt/nvidia/hpc_sdk/*/*/cuda/[0-9]*; do
    [[ -d "$r" ]] && roots+=("$r")
  done

  for r in "${roots[@]}"; do
    [[ -z "$r" || ! -d "$r" ]] && continue
    for inc in "$r/include" "$r"/targets/*/include; do
      [[ -f "$inc/cuda_runtime.h" ]] || continue
      # Libraries sit beside the headers in the targets layout, and in lib64 or
      # lib under a classic root.
      for lib in "${inc%/include}/lib" "${inc%/include}/lib64" \
                 "$r/lib64" "$r/lib"; do
        [[ -f "$lib/libcudart.so" || -f "$lib/libcudart_static.a" ]] && break
        lib=""
      done
      CUDA_ROOT="$r" CUDA_INC="$inc" CUDA_LIB="$lib"
      return 0
    done
  done
  return 1
}

find_cuda_paths || true
# Seeded rather than empty: under `set -u`, expanding an empty array as
# "${arr[@]}" is an unbound-variable error on bash before 4.4.  USE_BDA_BRIDGE
# is ON by default anyway, so it is a no-op when there is no CUDA to bridge to.
CUDA_CMAKE_ARGS=(-DUSE_BDA_BRIDGE=ON)
# Filled in only when Boost has to be unpacked into a prefix we own;
# empty on any machine where Boost is installed normally.
BOOST_CMAKE_ARGS=()
# Every prefix CMake should search, collected here and emitted once.
PREFIX_PATHS=()
DEP_CMAKE_ARGS=()
# Where anything fetched without root is unpacked.
DEPS_PREFIX="${DEPS_PREFIX:-$BUILD_ROOT/deps-userland}"

# --- fetching a library without root ----------------------------------------
# dpkg needs to be root; dpkg-deb does not.  apt-get download needs no
# privilege either, so a dependency that is only missing -- rather than
# unbuildable -- can be unpacked into a directory we own and pointed at with
# CMAKE_PREFIX_PATH.
#
# Packages are fetched one at a time on purpose: apt-get download is
# all-or-nothing, so one name a given release does not carry would discard
# every other package in the same call.  What matters is what lands.
# Every package in the dependency closure of the named packages, one per line.
# apt-cache needs no privilege.  Top-level names sit at column 0 in its output;
# dependencies are indented, and virtual packages appear as <name>.
apt_closure() {
  apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts \
      --no-breaks --no-replaces --no-enhances "$@" 2>/dev/null \
    | grep -E '^[a-zA-Z0-9]' | sed 's/:.*$//' | sort -u
}

# ...minus whatever dpkg says is already installed.  This is not an
# optimisation.  The closure of any -dev package reaches libc6, libstdc++6 and
# gcc-base; unpacking those into a prefix that then goes on CMAKE_PREFIX_PATH
# is how a build ends up linking against a stray C library.
apt_missing() {
  local pkg out=()
  for pkg in $(apt_closure "$@"); do
    # NEVER unpack the MPI stack.  This container's Open MPI is built from
    # source under /opt/openmpi, so dpkg does not know it exists and reports
    # libopenmpi3, libpmix2, libhwloc15 and libevent as missing.  Unpacking
    # Ubuntu's versions and putting that prefix on LD_LIBRARY_PATH makes
    # orterun load a runtime that is not its own, and it then rejects every
    # argument it has, including --version:
    #
    #   /opt/openmpi/bin/mpirun: Error: unknown option "-np"
    #
    # which looks like a launcher with strange syntax and is a poisoned
    # library path.  The same goes for the fabric and process-manager
    # libraries underneath it.
    case "$pkg" in
      libopenmpi*|openmpi-*|mpi-default-*|libmpich*|mpich*) continue ;;
      libpmix*|libhwloc*|libevent*|libucx*|libfabric*|libnl-*) continue ;;
      libcaf-*|libcoarrays*) continue ;;
    esac
    dpkg -s "$pkg" >/dev/null 2>&1 || out+=("$pkg")
  done
  [[ ${#out[@]} -gt 0 ]] && printf '%s\n' "${out[@]}"
  return 0
}

userland_unpack() {
  local prefix="$1"; shift
  local pkg got=0
  mkdir -p "$prefix/debs"
  for pkg in "$@"; do
    if ( cd "$prefix/debs" && apt-get download "$pkg" >/dev/null 2>&1 ); then
      got=$((got + 1))
    else
      echo "  (skipped $pkg -- not in this release)" >&2
    fi
  done
  shopt -s nullglob
  local debs=("$prefix"/debs/*.deb)
  shopt -u nullglob
  for pkg in "${debs[@]}"; do
    dpkg-deb -x "$pkg" "$prefix" || true
  done
  [[ ${#debs[@]} -gt 0 ]]
}


if [[ -n "$CUDA_INC" ]]; then
  echo "CUDA root  : ${CUDA_ROOT}"
  echo "CUDA incl  : ${CUDA_INC}  (cuda_runtime.h found)"
  echo "CUDA libs  : ${CUDA_LIB:-<not located; letting CMake search>}"
  CUDA_CMAKE_ARGS+=(
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
    -DCUDAToolkit_ROOT="$CUDA_ROOT"
    -DCUDA_TOOLKIT_INCLUDE="$CUDA_INC"
    -DCUDAToolkit_INCLUDE_DIR="$CUDA_INC"
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
    -DCMAKE_CUDA_STANDARD=17
    -DCMAKE_CUDA_FLAGS=-std=c++17
    -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH_CMAKE"
  )
  if [[ -n "$CUDA_LIB" ]]; then
    # FindCUDA looks under <root>/lib64 and <root>/lib, neither of which exists
    # in the targets layout, so name the libraries it needs directly.
    #
    # cuBLAS and cuSPARSE are NOT beside libcudart in the NVIDIA HPC SDK.  The
    # SDK splits the toolkit from the math libraries:
    #
    #   /opt/nvidia/hpc_sdk/<arch>/<ver>/cuda/12.8/lib64        libcudart
    #   /opt/nvidia/hpc_sdk/<arch>/<ver>/math_libs/12.8/lib64   libcublas,
    #                                                           libcusparse
    #
    # so look for each library across both trees rather than assuming one
    # directory holds them all.
    LIB_DIRS=("$CUDA_LIB")
    for d in "${CUDA_ROOT%/cuda/*}"/math_libs/*/lib64 \
             "${CUDA_ROOT%/cuda/*}"/math_libs/lib64 \
             "${CUDA_ROOT}/../../math_libs"/*/lib64; do
      # `if`, not `[[ ]] && ...`: see the note below about set -e.  A glob that
      # matches nothing expands to itself, so most iterations here are false.
      if [[ -d "$d" ]]; then LIB_DIRS+=("$d"); fi
    done

    # find_lib <soname> -> echoes the full path, or nothing.  Accepts a
    # versioned .so.N when the unversioned development symlink is absent, which
    # is normal on a runtime-only install.
    find_lib() {
      local name="$1" d f
      for d in "${LIB_DIRS[@]}"; do
        [[ -f "$d/$name.so" ]] && { printf '%s\n' "$d/$name.so"; return 0; }
      done
      for d in "${LIB_DIRS[@]}"; do
        f="$(ls -1 "$d/$name.so."* 2>/dev/null | sort -V | tail -1)"
        [[ -n "$f" ]] && { printf '%s\n' "$f"; return 0; }
      done
      return 1
    }

    # Every one of these used to be written as `[[ -f X ]] && ARGS+=(...)`.
    # Under `set -e` a false test makes the whole && list return non-zero, and
    # bash then exits the script *silently* -- no message, no non-zero status
    # visible to the caller, just a prompt.  That is exactly what happened on
    # the HPC SDK, where libcusparse is not in the cuda lib64.  Use `if`.
    CUDA_CMAKE_ARGS+=(-DCMAKE_LIBRARY_PATH="$(IFS=';'; echo "${LIB_DIRS[*]}")")
    for spec in "libcudart:CUDA_CUDART_LIBRARY" \
                "libcusparse:CUDA_cusparse_LIBRARY" \
                "libcublas:CUDA_CUBLAS_LIBRARIES"; do
      soname="${spec%%:*}"
      cmakevar="${spec##*:}"
      if found="$(find_lib "$soname")"; then
        CUDA_CMAKE_ARGS+=("-D${cmakevar}=${found}")
        echo "  ${soname}: ${found}"
      else
        echo "  ${soname}: not found in ${LIB_DIRS[*]}" >&2
      fi
    done
  fi
else
  echo "WARNING: nvcc is on PATH but no directory containing cuda_runtime.h" >&2
  echo "         was found.  Locate it with" >&2
  echo "             find / -name cuda_runtime.h 2>/dev/null | head" >&2
  echo "         and set CUDA_HOME to the directory ABOVE its include/ (or" >&2
  echo "         above targets/<arch>/include/).  Building WITHOUT the GPU" >&2
  echo "         linear solver so the build still completes." >&2
fi
GPU_NAME="$(first_line nvidia-smi --query-gpu=name --format=csv,noheader || echo unknown)"

# --- CMake: must be 3.x ----------------------------------------------------
if ! CMAKE="$(pick_cmake)"; then
  SYS_CMAKE_VER="$(first_line cmake --version 2>/dev/null || echo 'not found')"
  echo
  echo "CMake       : ${SYS_CMAKE_VER}" >&2
  echo "OPM ${TAG} cannot be configured by CMake 4.x: it sets policy CMP0026" >&2
  echo "to OLD and reads the LOCATION target property, both of which CMake 4" >&2
  echo "removed.  Fetching a private CMake 3.x for this build only." >&2
  if ! CMAKE="$(fetch_cmake)"; then
    echo >&2
    echo "ERROR: could not obtain a CMake 3.x automatically.  Install one and" >&2
    echo "       re-run with, for example:" >&2
    echo "           pip install --user 'cmake<4'" >&2
    echo "           CMAKE=\$HOME/.local/bin/cmake ./install_opm_gpu.sh" >&2
    exit 1
  fi
fi
CMAKE_VER="$(first_line "$CMAKE" --version | awk '{print $3}')"

# --- Boost ------------------------------------------------------------------
# opm-common wants Boost, and on Ubuntu the headers and the per-component CMake
# configs come from different packages.  With only libboost-dev installed,
# BoostConfig.cmake is present and every boost_<component>-config.cmake is
# missing, so configure dies 350 lines in with
#
#   Could not find a package configuration file provided by "boost_system"
#
# which reads like a CMake problem and is a missing package.  Check it here,
# where the fix can be named, instead of letting CMake report it.
# Globbed with nullglob rather than `ls`: this script runs under pipefail
# with an ERR trap, and `ls -d a* b*` exits 2 when either pattern matches
# nothing -- the normal case here, since only one of those two directories
# ever exists.  2>/dev/null hid the message and not the status, so the
# check aborted the script it was added to protect.
shopt -s nullglob
_boost_cfgs=(/usr/lib/*/cmake/Boost-* /usr/lib/cmake/Boost-*)
boost_cfg_dir="${_boost_cfgs[0]:-}"
if [[ -n "$boost_cfg_dir" ]]; then
  missing=()
  # opm-common needs more than system and unit_test_framework: configure
  # also looks for boost_filesystem and boost_regex, which are warnings at
  # this stage and errors further down the module chain.
  for comp in system unit_test_framework filesystem regex date_time iostreams; do
    _comp_cfgs=("$(dirname "$boost_cfg_dir")"/boost_${comp}-*)
    [[ ${#_comp_cfgs[@]} -gt 0 ]] || missing+=("$comp")
  done
  shopt -u nullglob
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Boost components not installed: ${missing[*]}" >&2
    echo "  ${boost_cfg_dir} exists, the component configs next to it do not," >&2
    echo "  so opm-common's find_package(Boost ...) cannot succeed." >&2

    # Install it rather than printing a command for someone to paste back.
    # This script already installs OPM itself into /usr/local and clones
    # four repositories to do it, so fetching its own prerequisite is not a
    # larger liberty than the ones it already takes.  It needs root, so:
    # use apt directly when root, sudo when sudo works without a password,
    # and otherwise print the command, because prompting for a password
    # from inside a build script run under srun is worse than stopping.
    apt_do=""
    if [[ -z "${NO_APT:-}" ]] && command -v apt-get >/dev/null 2>&1; then
      if [[ "$(id -u)" -eq 0 ]]; then
        apt_do="apt-get"
      elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        apt_do="sudo apt-get"
      fi
    fi

    if [[ -n "$apt_do" ]]; then
      echo "==> installing libboost-all-dev ($apt_do)" >&2
      if $apt_do update >&2 && $apt_do install -y libboost-all-dev >&2; then
        # Trust the re-check, not the exit code: a package can install and
        # still not put the component configs where CMake looks.
        shopt -s nullglob
        still=()
        for comp in "${missing[@]}"; do
          _c=("$(dirname "$boost_cfg_dir")"/boost_${comp}-*)
          [[ ${#_c[@]} -gt 0 ]] || still+=("$comp")
        done
        shopt -u nullglob
        if [[ ${#still[@]} -eq 0 ]]; then
          echo "==> Boost components now present; continuing" >&2
          missing=()
        else
          echo "ERROR: still missing after install: ${still[*]}" >&2
        fi
      else
        echo "ERROR: the install failed -- no network, or no apt sources." >&2
      fi
    fi

    # No root, and none to be had: dpkg refuses with "requested operation
    # requires superuser privilege", and sudo inside a Slurm container is
    # typically not setuid, so it refuses too.  apt will still DOWNLOAD,
    # though, and dpkg-deb -x unpacks into any directory we own.  A Boost
    # that CMake can find does not have to be a Boost that is installed.
    if [[ ${#missing[@]} -gt 0 && -z "${NO_APT:-}" ]] && command -v apt-get >/dev/null 2>&1 && command -v dpkg-deb >/dev/null 2>&1; then
      BOOST_PREFIX="${BOOST_PREFIX:-$BUILD_ROOT/boost-userland}"
      echo "==> unpacking Boost into $BOOST_PREFIX (no root needed)" >&2
      bver="$(basename "$boost_cfg_dir")"; bver="${bver#Boost-}"; bver="${bver%.*}"
      want=("libboost${bver}-dev")
      for comp in "${missing[@]}"; do
        # The component is not always the package name: Boost's
        # unit_test_framework lives in libboost-test.
        # Boost component names and Debian package names do not match:
        # components use underscores, packages use hyphens, and
        # unit_test_framework is packaged as libboost-test.
        case "$comp" in
          unit_test_framework) pk="test" ;;
          *)                   pk="${comp//_/-}" ;;
        esac
        want+=("libboost-${pk}${bver}-dev" "libboost-${pk}${bver}.0")
      done
      mkdir -p "$BOOST_PREFIX/debs"
      # One package at a time: apt-get download is all-or-nothing, so a
      # single name this release does not carry would discard the ones it
      # does.  What matters is what lands, not what apt returns.
      for pkg in "${want[@]}"; do
        ( cd "$BOOST_PREFIX/debs" && apt-get download "$pkg" >&2 2>/dev/null ) \
          || echo "  (skipped $pkg -- not in this release)" >&2
      done
      shopt -s nullglob
      _debs=("$BOOST_PREFIX"/debs/*.deb)
      shopt -u nullglob
      if [[ ${#_debs[@]} -gt 0 ]]; then
        for deb in "${_debs[@]}"; do
          dpkg-deb -x "$deb" "$BOOST_PREFIX" || true
        done
        shopt -s nullglob
        _bd=("$BOOST_PREFIX"/usr/lib/*/cmake/Boost-* "$BOOST_PREFIX"/usr/lib/cmake/Boost-*)
        shopt -u nullglob
        if [[ ${#_bd[@]} -gt 0 ]]; then
          # Boost_DIR is named explicitly rather than left to
          # CMAKE_PREFIX_PATH: the system BoostConfig.cmake is still on
          # disk, and it looks for components next to ITSELF, so if CMake
          # picks that one the unpacked components are invisible.
          BOOST_CMAKE_ARGS+=(
            -DBoost_DIR="${_bd[0]}"
            -DBoost_NO_SYSTEM_PATHS=ON
          )
          # Appended to the shared list rather than passed as its own
          # -DCMAKE_PREFIX_PATH: DUNE adds one too, and two of the same
          # flag means the last one silently wins.
          PREFIX_PATHS+=("$BOOST_PREFIX/usr")
          echo "==> using unpacked Boost: ${_bd[0]}" >&2
          missing=()
        else
          echo "ERROR: unpacked the packages but found no BoostConfig.cmake" >&2
        fi
      else
        echo "ERROR: could not download the Boost packages." >&2
      fi
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
      echo >&2
      echo "  apt-get update && apt-get install -y libboost-all-dev" >&2
      echo >&2
      echo "       (as root, or with sudo; the narrower set is" >&2
      echo "       libboost-system-dev libboost-test-dev" >&2
      echo "       libboost-date-time-dev libboost-filesystem-dev)" >&2
      echo >&2
      echo "       NO_APT=1 stops this script installing anything." >&2
      echo "       SKIP_BOOST_CHECK=1 bypasses the check and lets CMake try." >&2
      if [[ -z "${SKIP_BOOST_CHECK:-}" ]]; then exit 1; fi
    fi
  fi
else
  shopt -u nullglob
fi

echo "GPU        : ${GPU_NAME}"
echo "Arch       : sm_${GPU_ARCH}$([[ "$GPU_ARCH_CMAKE" == *-virtual* ]] && echo " (SASS + PTX; the cubin loads directly on a newer minor revision)")"
echo "CUDA       : ${CUDA_VERSION}"
echo "CMake      : ${CMAKE_VER}  (${CMAKE})"
echo "Install to : ${INSTALL_DIR}"
echo "Sources in : ${BUILD_ROOT}"
echo

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

# --- Clone -----------------------------------------------------------------
for repo in "${MODULES[@]}"; do
  if [[ -d "$repo/.git" ]]; then
    echo "== $repo already cloned, fetching =="
    git -C "$repo" fetch --tags --quiet
  else
    echo "== cloning $repo =="
    rm -rf "$repo"
    git clone --quiet "https://github.com/OPM/$repo.git"
  fi
  git -C "$repo" checkout --quiet "$TAG" || {
    echo "WARNING: $repo has no tag $TAG; staying on the default branch" >&2
  }
done

# --- DUNE -------------------------------------------------------------------
# OPM is built on DUNE and this script never provided it, so configure ran the
# whole way through Boost and then stopped at
#
#   Could not find a package configuration file provided by "dune-common"
#
# Ubuntu 22.04 ships DUNE 2.8, older than OPM 2024.10 wants, and installing it
# would need root in any case.  Built from source into the prefix OPM installs
# to, so one CMAKE_PREFIX_PATH covers both.
#
#   SKIP_DUNE=1        assume DUNE is already available
#   DUNE_BRANCH=...    a different release (default releases/2.9)
DUNE_BRANCH="${DUNE_BRANCH:-releases/2.9}"
DUNE_MODULES=(dune-common dune-geometry dune-grid dune-istl)

# Tested with -d, one candidate at a time.  nullglob only removes patterns
# that CONTAIN a wildcard: a literal path stays in the array whether or not
# it exists, so two of these four were always present and the array was
# never empty -- which silently skipped this entire block.
_dune_found=""
shopt -s nullglob
for _d in /usr/lib/*/cmake/dune-common /usr/lib/cmake/dune-common "$INSTALL_DIR"/lib/cmake/dune-common "$INSTALL_DIR"/lib/*/cmake/dune-common; do
  if [[ -d "$_d" ]]; then _dune_found="$_d"; break; fi
done
shopt -u nullglob

if [[ -z "$_dune_found" && -z "${SKIP_DUNE:-}" ]]; then
  echo
  echo "== DUNE not found; building ${DUNE_BRANCH} =="
  for dm in "${DUNE_MODULES[@]}"; do
    if [[ ! -d "$BUILD_ROOT/$dm/.git" ]]; then
      echo "-- cloning $dm"
      git -C "$BUILD_ROOT" clone --quiet --depth 1 -b "$DUNE_BRANCH" \
        "https://gitlab.dune-project.org/core/$dm.git" || {
          echo "ERROR: could not clone $dm from gitlab.dune-project.org." >&2
          echo "       If this machine cannot reach it, install DUNE another" >&2
          echo "       way and re-run with SKIP_DUNE=1." >&2
          exit 1; }
    fi
    echo "-- building $dm"
    mkdir -p "$BUILD_ROOT/$dm/build"
    cd "$BUILD_ROOT/$dm/build"
    rm -f CMakeCache.txt; rm -rf CMakeFiles
    # The Python bindings pull in a venv and pip, neither of which is wanted
    # here and both of which fail without root in some containers.
    "$CMAKE" .. \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DDUNE_ENABLE_PYTHONBINDINGS=OFF \
      -DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON \
      -DMPI_C_COMPILER=mpicc \
      -DMPI_CXX_COMPILER=mpicxx \
      >/dev/null || { echo "ERROR: cmake failed for $dm" >&2; exit 1; }
    make -j"$(nproc)" >/dev/null || { echo "ERROR: build failed for $dm" >&2; exit 1; }
    make install >/dev/null || {
      echo "ERROR: could not install $dm into $INSTALL_DIR." >&2
      echo "       Set INSTALL_DIR to somewhere writable and re-run." >&2
      exit 1; }
  done
  cd "$BUILD_ROOT"
  echo "== DUNE installed into $INSTALL_DIR =="
fi
PREFIX_PATHS+=("$INSTALL_DIR")

# --- external dependencies --------------------------------------------------
# Everything OPM can use that this image does not have, fetched in one pass.
#
# Two of these are hard requirements and the rest are not, which is worth being
# precise about.  SuiteSparse is REQUIRED by dune-istl, and opm-grid refuses to
# configure with MPI enabled and no ZOLTAN:
#
#   opm-grid with MPI support requires the package ZOLTAN
#
# METIS, Scotch, PTScotch, SuperLU, GMP, cjson and fmt are all optional -- the
# configure log reports each as "Could NOT find" and carries on -- but they are
# fetched anyway, because each one absent means a slower or less capable Flow,
# and fetching them costs a few megabytes.
#
#   SKIP_DEPS=1   do not fetch anything; assume the system has what it needs
DEP_PACKAGES=(
  libsuitesparse-dev        # UMFPACK etc: REQUIRED by dune-istl
  libtrilinos-zoltan-dev    # graph partitioning: REQUIRED by opm-grid with MPI
  # The graph partitioners are deliberately absent. They divide a grid
  # between ranks, which a single-rank run never does, and Scotch ships
  # compatibility headers called metis.h and parmetis.h that make opm-grid
  # believe METIS is available when it is not. NVRS_OPM_PARTITIONERS=1 adds
  # them back.
  libsuperlu-dev            # direct solver, optional
  libgmp-dev                # exact arithmetic in dune-grid, optional
  libcjson-dev              # otherwise opm-common downloads its own
  libfmt-dev                # otherwise opm-common downloads its own
)

if [[ -z "${SKIP_DEPS:-}" ]] && command -v apt-get >/dev/null 2>&1 \
   && command -v dpkg-deb >/dev/null 2>&1; then
  echo
  echo "== resolving external dependencies =="
  mapfile -t _need < <(apt_missing "${DEP_PACKAGES[@]}")
  if [[ ${#_need[@]} -eq 0 ]]; then
    echo "   nothing missing"
  else
    echo "   ${#_need[@]} package(s) to fetch"
    userland_unpack "$DEPS_PREFIX" "${_need[@]}" || true
  fi
fi
PREFIX_PATHS+=("$DEPS_PREFIX/usr")

# CMake is told exactly where these landed rather than left to infer it.  The
# Debian layout puts Zoltan's headers under include/trilinos and names the
# library libtrilinos_zoltan, which is not where or what OPM's FindZOLTAN looks
# for first, and an unpacked prefix is not a place it would search at all.
shopt -s nullglob
_zh=("$DEPS_PREFIX"/usr/include/trilinos/zoltan.h "$DEPS_PREFIX"/usr/include/zoltan.h)
_zl=("$DEPS_PREFIX"/usr/lib/*/libtrilinos_zoltan.so "$DEPS_PREFIX"/usr/lib/*/libzoltan.so)
_ss=("$DEPS_PREFIX"/usr/include/suitesparse)
shopt -u nullglob
if [[ ${#_zh[@]} -gt 0 && ${#_zl[@]} -gt 0 ]]; then
  DEP_CMAKE_ARGS+=(
    -DZOLTAN_INCLUDE_DIR="$(dirname "${_zh[0]}")"
    -DZOLTAN_LIBRARY="${_zl[0]}"
  )
  echo "   zoltan     : ${_zl[0]}"
fi
if [[ ${#_ss[@]} -gt 0 ]]; then
  DEP_CMAKE_ARGS+=(-DSuiteSparse_INCLUDE_DIRS="${_ss[0]}")
fi

# --- graph partitioners -----------------------------------------------------
# Disabled rather than searched for. opm-grid guards its METIS partitioner on a
# macro and then includes <metis.h> unconditionally, so any detection that
# succeeds partially fails later during compilation, with a message naming a
# file rather than a package. Scotch makes that outcome likely by shipping its
# own metis.h and parmetis.h as compatibility headers.
#
# None of them is needed: they choose how a grid is split across ranks, and
# every run here uses one. ZOLTAN, which opm-grid does require under MPI, is a
# different package and is left alone.
if [[ -z "${NVRS_OPM_PARTITIONERS:-}" ]]; then
  for _name in METIS ParMETIS Scotch PTScotch; do
    DEP_CMAKE_ARGS+=("-DCMAKE_DISABLE_FIND_PACKAGE_${_name}=ON")
  done
  echo "   partitioners: METIS, ParMETIS, Scotch and PTScotch disabled"
  echo "                 (single-rank runs do not partition; "
  echo "                  NVRS_OPM_PARTITIONERS=1 to look for them)"
fi

# Scotch's compatibility headers are removed from the unpacked prefix if an
# earlier run put them there, because opm-grid finds them by searching the
# include path rather than only through find_package.
if [[ -z "${NVRS_OPM_PARTITIONERS:-}" && -d "$DEPS_PREFIX/usr/include/scotch" ]]; then
  echo "   removing $DEPS_PREFIX/usr/include/scotch (its metis.h misleads opm-grid)"
  rm -rf "$DEPS_PREFIX/usr/include/scotch"
fi

# --- Build -----------------------------------------------------------------
for repo in "${MODULES[@]}"; do
  echo
  echo "== building $repo (CUDA, sm_${GPU_ARCH}) =="
  cd "$BUILD_ROOT/$repo"
  mkdir -p build && cd build
  # A cache that once recorded CUDA_TOOLKIT_INCLUDE-NOTFOUND keeps handing it
  # back no matter what is passed on the command line, so one failed configure
  # would poison every retry.  Start from a clean cache.
  rm -f CMakeCache.txt
  rm -rf CMakeFiles
  "$CMAKE" .. \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_CXX_STANDARD=17 \
    -DBUILD_SHARED_LIBS=ON \
    -DMPI_C_COMPILER=mpicc \
    -DMPI_CXX_COMPILER=mpicxx \
    -DENABLE_MPI=ON -DUSE_MPI=ON \
    -DUSE_OPENMP=ON -DENABLE_OPENMP=ON \
    -DBUILD_TESTING=OFF \
    "${CUDA_CMAKE_ARGS[@]}" ${BOOST_CMAKE_ARGS[@]+"${BOOST_CMAKE_ARGS[@]}"} \
    ${DEP_CMAKE_ARGS[@]+"${DEP_CMAKE_ARGS[@]}"} \
    -DCMAKE_PREFIX_PATH="$(IFS=';'; echo "${PREFIX_PATHS[*]}")"
  make -j"$(nproc)"
  make install
done

# --- Verify ----------------------------------------------------------------
echo
FLOW="$(command -v flow || true)"
if [[ -z "$FLOW" && -x "$INSTALL_DIR/bin/flow" ]]; then
  FLOW="$INSTALL_DIR/bin/flow"
fi
if [[ -z "$FLOW" ]]; then
  echo "ERROR: flow was not found after install.  Looked on PATH and in" >&2
  echo "       $INSTALL_DIR/bin." >&2
  exit 1
fi
echo "flow installed at: $FLOW"
"$FLOW" --version || true

# Which GPU linear solver this build exposes decides what run_comparison.sh
# can ask for, and the option name has moved between OPM releases, so report
# what this binary actually understands rather than assuming.
echo
# --accelerator-mode is an ADVANCED parameter: it is listed by --help-all, not
# by --help.  Checking only the short help reports "no GPU solver" on a build
# that has one.
echo "GPU linear-solver support:"
if "$FLOW" --help-all 2>&1 | grep -iE -- '--accelerator-mode' \
  || "$FLOW" --help 2>&1 | grep -iE -- '--accelerator-mode'; then
  echo "  --accelerator-mode is registered."
  # Registration is unconditional; the backend is not.  A build where CMake did
  # not find CUDA still advertises the option and then refuses at run time with
  # "Error cusparseSolver was chosen, but CUDA was not found by CMake", so ask
  # it now rather than in the middle of the benchmark.
  if "$FLOW" --accelerator-mode=cusparse --version >/dev/null 2>&1; then
    echo "  cusparse backend accepted -- Flow will use the GPU."
  else
    echo "  WARNING: cusparse was rejected.  CMake did not find CUDA when"
    echo "           opm-simulators was configured, so this build has no GPU"
    echo "           linear solver.  Check the configure output above for"
    echo "           'cusparse' and re-run with CUDA_HOME set." >&2
  fi
else
  echo "  --accelerator-mode is NOT registered: no GPU linear solver in this"
  echo "  build.  The benchmark still runs and the report says GPU vs CPU." >&2
fi

echo
echo "Done.  Next:"
# A per-user prefix is not on PATH by default, so say so rather than letting
# run_comparison.sh fail to find flow.
if [[ "$INSTALL_DIR" != "/usr" && "$INSTALL_DIR" != "/usr/local" ]]; then
  echo "  export PATH=\"$INSTALL_DIR/bin:\$PATH\""
  echo "  export LD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\${LD_LIBRARY_PATH:-}\""
fi
# Anything unpacked without root has to be on LD_LIBRARY_PATH, not just in an
# RPATH.  OPM builds with -Wl,--enable-new-dtags, so the linker writes
# DT_RUNPATH, which resolves an object's OWN dependencies only: flow finds
# libumfpack.so through its runpath and then dies on libamd.so.2, which is
# libumfpack's dependency, looked up against libumfpack's runpath instead.
# run_norne.sh and run_comparison.sh set this themselves; it is printed here
# for anyone running flow by hand.
if [[ -d /usr/lib ]]; then
  for _d in /usr/lib /usr/lib/*/; do
    [[ -d "$_d" ]] && echo "  export LD_LIBRARY_PATH=\"${_d%/}:\${LD_LIBRARY_PATH:-}\""
  done
fi
echo "  cd $HERE && ./run_comparison.sh"
