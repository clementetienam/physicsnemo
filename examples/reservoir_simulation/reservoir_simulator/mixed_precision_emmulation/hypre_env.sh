# Source this before building or running anything that links HYPRE.
#
#   source <repo>/mixed_precision_emmulation/hypre_env.sh
#
# The HYPRE prefixes used to be written out as absolute paths.  They pointed at
# /workspace/project/hypre-install, the tree moved to
# full_reservoir_simulator/Mixed_precision/hypre-install, and the mismatch
# surfaced much later as "cannot open source file HYPRE.h".  The prefixes are
# now located relative to this file, so moving or re-cloning the repository
# does not silently break the build.  Set HYPRE_FP32_DIR / HYPRE_FP64_DIR
# yourself beforehand and the search is skipped.

# Works whether sourced from bash or zsh.
_hy_self="${BASH_SOURCE[0]:-${(%):-%x}}"
_hy_root="$(cd "$(dirname "$_hy_self")/.." && pwd)"

# First match wins.  The second entry is where the tree lives today; the others
# are older layouts, kept so an existing checkout keeps working.
_hy_find() {
  local want="$1" c
  for c in \
    "$_hy_root/hypre-install/$want" \
    "$_hy_root/full_reservoir_simulator/Mixed_precision/hypre-install/$want" \
    "$_hy_root/Mixed_precision/hypre-install/$want" \
    "$_hy_root/mixed_precision_emmulation/hypre-install/$want" \
    "$(dirname "$_hy_root")/hypre-install/$want"
  do
    [ -f "$c/include/HYPRE.h" ] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}

[ -n "${HYPRE_FP32_DIR:-}" ] && [ -f "${HYPRE_FP32_DIR}/include/HYPRE.h" ] || \
  HYPRE_FP32_DIR="$(_hy_find fp32 || true)"
[ -n "${HYPRE_FP64_DIR:-}" ] && [ -f "${HYPRE_FP64_DIR}/include/HYPRE.h" ] || \
  HYPRE_FP64_DIR="$(_hy_find fp64 || true)"

export HYPRE_FP32_DIR HYPRE_FP64_DIR

# lib or lib64 depending on how HYPRE was configured.
_hy_lib() {
  [ -n "$1" ] || return 0
  [ -d "$1/lib" ] && { printf '%s\n' "$1/lib"; return 0; }
  [ -d "$1/lib64" ] && { printf '%s\n' "$1/lib64"; return 0; }
}
export HYPRE_FP32_LIB="$(_hy_lib "${HYPRE_FP32_DIR:-}")"
export HYPRE_FP64_LIB="$(_hy_lib "${HYPRE_FP64_DIR:-}")"

# Deliberately NOT added to LD_LIBRARY_PATH.  The two builds ship the same
# libHYPRE.so soname and differ only in precision, so putting either on the
# search path risks loading the wrong one at run time with nothing to show for
# it but a silently different answer.  The link uses -rpath instead.

# --- MPI ------------------------------------------------------------------
# Prefer an mpicc already on PATH.  The container built by build_container.sh
# has OpenMPI 4.1.5 at /opt/openmpi and no HPC SDK at all, so looking only for
# HPCX found nothing there.
#
# The HPCX fallback used to name one absolute path:
#   /opt/nvidia/hpc_sdk/Linux_aarch64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi
# which pinned the architecture, the SDK version, the CUDA version and the HPCX
# version all at once.  On x86_64 the directory simply does not exist, the `if`
# was false, and the script set no MPI at all -- silently.  Glob every one of
# those four instead.
_hy_mpi=""
if command -v mpicc >/dev/null 2>&1; then
  _hy_l="$(mpicc --showme:link 2>/dev/null | tr ' ' '
' | grep -E '^-L.*/lib6?4?$' | head -1 || true)"
  [ -n "$_hy_l" ] && _hy_mpi="$(dirname "${_hy_l#-L}")"
  unset _hy_l
fi
if [ -z "$_hy_mpi" ]; then
  case "$(uname -m)" in
    aarch64|arm64) _hy_arch=aarch64 ;;
    x86_64|amd64)  _hy_arch=x86_64  ;;
    *)             _hy_arch="$(uname -m)" ;;
  esac
  for _hy_c in /opt/openmpi                /opt/nvidia/hpc_sdk/Linux_${_hy_arch}/*/comm_libs/*/hpcx/hpcx-*/ompi                /opt/nvidia/hpc_sdk/Linux_${_hy_arch}/*/comm_libs/hpcx/hpcx-*/ompi
  do
    [ -d "$_hy_c/bin" ] && _hy_mpi="$_hy_c"
  done
  unset _hy_arch _hy_c
fi
if [ -n "$_hy_mpi" ] && [ -d "$_hy_mpi" ]; then
  export OPAL_PREFIX="$_hy_mpi"
  export PATH="$_hy_mpi/bin:${PATH:-}"
  [ -d "$_hy_mpi/lib" ]   && export LD_LIBRARY_PATH="$_hy_mpi/lib:${LD_LIBRARY_PATH:-}"
  [ -d "$_hy_mpi/lib64" ] && export LD_LIBRARY_PATH="$_hy_mpi/lib64:${LD_LIBRARY_PATH:-}"
fi
unset _hy_mpi

if [ -z "${HYPRE_FP32_DIR:-}" ]; then
  echo "hypre_env.sh: no FP32 HYPRE found under $_hy_root." >&2
  echo "  Build it, or set HYPRE_FP32_DIR to a prefix with include/HYPRE.h." >&2
else
  echo "HYPRE_FP32_DIR=$HYPRE_FP32_DIR"
  [ -n "${HYPRE_FP64_DIR:-}" ] && echo "HYPRE_FP64_DIR=$HYPRE_FP64_DIR"
fi

unset _hy_self _hy_root
unset -f _hy_find
