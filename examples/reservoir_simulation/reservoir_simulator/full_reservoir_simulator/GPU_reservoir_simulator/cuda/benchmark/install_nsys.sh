#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# install_nsys.sh -- get Nsight Systems onto a machine that has no root.
#
# Several images ship Nsight Compute and not Nsight Systems, which leaves the
# profiling half-working: ncu answers "why is this kernel slow" while nsys, the
# one that answers "where does the time go", is missing entirely.
#
#   [profile] Nsight Systems not found; skipping the timeline
#
# Installing it normally needs root, and inside a Slurm container there is none
# to be had -- dpkg refuses outright.  But apt-get download needs no privilege
# and dpkg-deb -x unpacks into any directory we own, which is enough: nsys is a
# self-contained tree that runs from wherever it is put.
#
#   ./install_nsys.sh                 # into ../tools, and print the export line
#   PREFIX=/somewhere ./install_nsys.sh
#
# Then either put it on PATH, or point the profiling scripts straight at it:
#
#   export NSYS=<prefix>/opt/nvidia/nsight-systems/<version>/target-linux-.../nsys
#
# profile_run.sh also finds it by itself if it lands under /opt/nvidia.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HERE/../tools/nsight}"

command -v apt-get  >/dev/null 2>&1 || { echo "ERROR: no apt-get here." >&2; exit 1; }
command -v dpkg-deb >/dev/null 2>&1 || { echo "ERROR: no dpkg-deb here." >&2; exit 1; }

echo "==> looking for an Nsight Systems package"
# The CLI package is the one worth having: the full package drags in the Qt GUI,
# which is a large download for a machine nobody opens a window on.  Names carry
# the version, so the newest match is taken rather than a name being guessed.
PKG="${NSYS_PKG:-}"
if [[ -z "$PKG" ]]; then
  PKG="$(apt-cache search --names-only 'nsight-systems' 2>/dev/null \
          | awk '{print $1}' | grep -E 'nsight-systems-cli' | sort -V | tail -1)"
fi
if [[ -z "$PKG" ]]; then
  PKG="$(apt-cache search --names-only 'nsight-systems' 2>/dev/null \
          | awk '{print $1}' | sort -V | tail -1)"
fi
if [[ -z "$PKG" ]]; then
  echo "ERROR: apt knows no nsight-systems package." >&2
  echo "       The CUDA repository provides it; check that this is present:" >&2
  echo "         grep -r developer.download.nvidia.com /etc/apt/sources.list.d/" >&2
  echo "       Otherwise download it from" >&2
  echo "         https://developer.nvidia.com/nsight-systems" >&2
  echo "       and set NSYS=/path/to/nsys when profiling." >&2
  exit 1
fi
echo "    package : $PKG"

mkdir -p "$PREFIX/debs" || { echo "ERROR: cannot write to $PREFIX" >&2; exit 1; }
echo "==> downloading (no root needed)"
( cd "$PREFIX/debs" && apt-get download "$PKG" ) || {
  echo "ERROR: could not download $PKG." >&2; exit 1; }

echo "==> unpacking into $PREFIX"
shopt -s nullglob
debs=("$PREFIX"/debs/*.deb)
shopt -u nullglob
for d in "${debs[@]}"; do
  dpkg-deb -x "$d" "$PREFIX" || echo "  (failed to unpack $(basename "$d"))" >&2
done

# The binary sits under a version and an architecture directory, and the
# architecture differs -- sbsa-armv8 on Grace and Vera, x64 elsewhere -- so it
# is found rather than assumed.
shopt -s nullglob
found=("$PREFIX"/opt/nvidia/nsight-systems*/*/target-linux-*/nsys)
shopt -u nullglob
if [[ ${#found[@]} -eq 0 ]]; then
  echo "ERROR: unpacked, but no nsys binary underneath $PREFIX." >&2
  echo "       Look with: find $PREFIX -name nsys -type f" >&2
  exit 1
fi

NSYS_BIN="${found[0]}"
chmod +x "$NSYS_BIN" 2>/dev/null || true
echo "==> nsys: $NSYS_BIN"
"$NSYS_BIN" --version 2>&1 | head -2 | sed 's/^/    /'

echo
echo "next:"
echo "  export NSYS=\"$NSYS_BIN\""
echo "  cd $HERE && NVRS_PROFILE=1 ./run_norne.sh"
