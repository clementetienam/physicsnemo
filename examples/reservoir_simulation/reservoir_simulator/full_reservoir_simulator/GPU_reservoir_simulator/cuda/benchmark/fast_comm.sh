# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# fast_comm.sh -- the fastest working GPU-to-GPU transport for every backend,
# chosen automatically.  Sourced by nvrs.sh; it only
# exports environment variables, so every run underneath inherits them.
#
#   simulator   NCCL for the Krylov tail's halos and reductions whenever the
#               binary was built with NCCL (found automatically, including the
#               HPC SDK's comm_libs/nccl).  Nothing to choose at run time.
#   AmgX        no NCCL support.  Its fast mode passes GPU memory straight to
#               MPI (communicator MPI_DIRECT) instead of copying through the
#               host.  Some MPI builds accept device pointers and then fault
#               inside AmgX, so it is not assumed: a 2-GPU AmgX run of
#               decks/SMOKE.DATA (a few seconds) is tried once per binary, and
#               MPI_DIRECT is used only if it completes.  The answer is cached
#               in runs/.fast_comm next to the binary's size and date.
#   HYPRE       no NCCL support.  GPU-aware MPI is compiled into libHYPRE;
#               install_all.sh rebuilds HYPRE with it when the MPI supports
#               device buffers and the installed library lacks it.
#   Ginkgo      one GPU per run in this simulator: no transport.
#
# NVRS_AMGX_COMM set by hand always wins, and NVRS_FAST_COMM=0 skips all of
# this (AmgX then uses its host-buffer communicator, as before).

nvrs_fast_comm() {
  local exe="$1" cache_dir="$2" cuda_dir="$3"
  # Open MPI's UCC collective component has segfaulted 8-rank NVRS runs;
  # disabling it removed the crashes.  Off unless set by hand.
  export OMPI_MCA_coll_ucc_enable="${OMPI_MCA_coll_ucc_enable:-0}"
  [[ "${NVRS_FAST_COMM:-1}" == 0 ]] && return 0
  [[ -n "${NVRS_FAST_COMM_DONE:-}" ]] && return 0   # a parent script chose
  export NVRS_FAST_COMM_DONE=1
  if [[ -n "${NVRS_AMGX_COMM:-}" ]]; then
    echo "  transport : AmgX communicator $NVRS_AMGX_COMM (set by hand)"
    return 0
  fi
  [[ -x "$exe" ]] || return 0
  local ngpu
  ngpu="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
  if [[ "${ngpu:-0}" -lt 2 ]]; then
    return 0                       # one GPU: AmgX never communicates
  fi
  ldd "$exe" 2>/dev/null | grep -qi amgx || return 0

  local key cache k v
  key="$(stat -c '%Y_%s' "$exe" 2>/dev/null)"
  cache="$cache_dir/.fast_comm"
  if [[ -f "$cache" ]] && read -r k v < "$cache" && [[ "$k" == "$key" && -n "$v" ]]; then
    export NVRS_AMGX_COMM="$v"
    echo "  transport : AmgX communicator $v (probed earlier for this binary)"
    return 0
  fi

  local mr="" c
  for c in "${MPIRUN:-}" "${OMPI_HOME:-/opt/openmpi}/bin/mpirun" /opt/openmpi/bin/mpirun \
           "$(command -v mpirun 2>/dev/null || true)"; do
    [[ -n "$c" && -x "$c" ]] && { mr="$c"; break; }
  done
  [[ -n "$mr" ]] || return 0
  local run=("$mr")
  "$mr" --version 2>&1 | grep -qiE 'open[ -]?mpi|openrte' && run+=(--oversubscribe --allow-run-as-root)

  local tmp rc
  tmp="$(mktemp -d)"
  cp "$cuda_dir/decks/SMOKE.DATA" "$tmp/probe.DATA"
  printf '\nOUTDIR %s/OUT\nLINSOLVER AMGX\nAUTOPLOT 0\n' "$tmp" >> "$tmp/probe.DATA"
  echo "  transport : probing AmgX MPI_DIRECT on 2 GPUs (once per binary, a few seconds)"
  ( cd "$cuda_dir" && NVRS_AMGX_COMM=MPI_DIRECT timeout --kill-after=10 180 \
      "${run[@]}" -np 2 "$exe" "$tmp/probe.DATA" ) > "$tmp/probe.log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]] && grep -q 'simulation finished' "$tmp/probe.log"; then
    v=MPI_DIRECT
    echo "  transport : AmgX communicator MPI_DIRECT (GPU memory straight to MPI)"
  else
    v=MPI
    echo "  transport : AmgX communicator MPI (host buffers; MPI_DIRECT failed, rc=$rc,"
    echo "              log kept at $tmp/probe.log)"
  fi
  mkdir -p "$cache_dir" && echo "$key $v" > "$cache"
  [[ "$v" == MPI_DIRECT ]] && rm -rf "$tmp"
  export NVRS_AMGX_COMM="$v"
}
