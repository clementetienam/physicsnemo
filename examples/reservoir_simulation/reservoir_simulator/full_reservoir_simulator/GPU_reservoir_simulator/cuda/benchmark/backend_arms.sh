# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# backend_arms.sh -- the solver arms of the backend ablation, defined once.
#
# Sourced, not run, by run_norne.sh, profile_case.sh, nvrs.sh and
# scripts/backend_study.sh, so the five arms mean the same deck lines
# everywhere:
#
#   amgx        LINSOLVER AMGX                                  FP32 AMG
#   hypre       LINSOLVER HYPRE                                 FP32 AMG
#   ginkgo      LINSOLVER GINKGO, GINKGOPREC 16                 FP16 AMG
#   ginkgo32    LINSOLVER GINKGO, GINKGOPREC 32                 FP32 AMG, same hierarchy
#   ginkgo_abj  LINSOLVER GINKGO, GINKGOPREC 16, STAGE2 GINKGO  FP16 AMG + adaptive-
#               precision block-Jacobi in stage 2
#   amgx_fp64   LINSOLVER AMGX, EMULATE 0                       FP32 AMG, native FP64
#               operator instead of the df64-emulated one (Norne and SPE10
#               only; the matched SPE10 deck already carries EMULATE 0, so
#               there it repeats the AmgX arm by construction).  Despite the
#               name the multigrid is FP32: "fp64" is the operator.
#   amgx_fp64amg LINSOLVER AMGX, MIXED 0, EMULATE 0             everything FP64: the
#               AmgX hierarchy and V-cycle in FP64 (dDDI) as well as the
#               operator, run on nvrs_cuda_fp64.  Against amgx it measures
#               what the FP32 preconditioner buys.  Only when nvrs_cuda_fp64
#               is built (./nvrs.sh install, or make FP64_PRECOND=1).
#   amgx_tensorcore LINSOLVER AMGX, JACOBIAN TENSORCORE         FP32 AMG, connection
#               blocks by the Tensor Core operator product, run on nvrs_cuda_tc.
#               Only when nvrs_cuda_tc is built (TENSORCORE=1 ./nvrs.sh install).
#
NVRS_ALL_ARMS="amgx hypre ginkgo ginkgo32 ginkgo_abj amgx_fp64 amgx_fp64amg amgx_tensorcore"

arm_known() {
  case "$1" in
    amgx|hypre|ginkgo|ginkgo32|ginkgo_abj|amgx_fp64|amgx_fp64amg|amgx_tensorcore) return 0 ;;
  esac
  return 1
}

arm_needs_ginkgo() { [[ "$1" == ginkgo* ]]; }

# deck_single_gpu <deck>: true when the deck, as the simulator will read it,
# asks for Ginkgo (LINSOLVER/AMGLIB GINKGO or STAGE2 GINKGO|ADAPTIVE|2).  The
# Ginkgo backend is single-GPU and main.cu aborts a multi-rank run of it, so
# every launcher that chooses a rank count clamps to one when this is true.
# The last occurrence wins, as in deck.cpp; `--` comments are ignored.
deck_single_gpu() {
  [[ -f "$1" ]] || return 1
  awk '
    { sub(/--.*/, ""); if (NF < 2) next; k = toupper($1); v = toupper($2) }
    k == "LINSOLVER" || k == "AMGLIB" { ls = v }
    k == "STAGE2"                     { s2 = v }
    END { exit !(ls == "GINKGO" || ls == "2" ||
                 s2 == "GINKGO" || s2 == "ADAPTIVE" || s2 == "2") }
  ' "$1"
}

# Ginkgo arms are single-GPU in this build.
arm_single_gpu() { arm_needs_ginkgo "$1"; }

arm_label() {
  case "$1" in
    amgx)       echo "AmgX FP32" ;;
    hypre)      echo "HYPRE FP32" ;;
    ginkgo)     echo "Ginkgo FP16" ;;
    ginkgo32)   echo "Ginkgo FP32" ;;
    ginkgo_abj) echo "Ginkgo FP16 + adaptive block-Jacobi" ;;
    amgx_fp64)  echo "AmgX FP32, native FP64 operator" ;;
    amgx_fp64amg) echo "AmgX FP64 (all FP64: multigrid and operator)" ;;
    amgx_tensorcore) echo "AmgX FP32, Tensor Core Jacobian" ;;
    *)          echo "$1" ;;
  esac
}

# HYPRE's coarsening bundle and hierarchy reuse, per case, as measured by
# scripts/hypre_screen.sh on one GB200 (16 Sep 2026):
#
#   case     window  AMGFAST 1 + REUSE 10   AMGFAST 0 + REUSE 10   kept
#   NORNE    336 d    4.0 s, Krylov 3656     4.6 s, Krylov 1870     AMGFAST 1 (deck)
#   MILLION  300 d    6.3 s, Krylov 2495     9.1 s, Krylov 2418     AMGFAST 1 (deck)
#   SPE10    100 d   56.1 s, Krylov 20944   23.4 s, Krylov 5656     AMGFAST 0
#
# (AmgX on the same windows: 2.1 s, 5.3 s, 16.2 s.)  AMGFAST 0 pays for a
# stronger, costlier V-cycle; only on SPE10's rock does the Krylov saving
# outweigh it.  The 2M-5M rungs use the 1M deck and keep its setting.  So the decks keep their values and
# SPE10's HYPRE arm overrides them (run_spe10_matched.sh, profile_case.sh).
#
# NVRS_HYPRE_AMGFAST / NVRS_HYPRE_AMGREUSE, when set, override every HYPRE run
# in every case: they are appended to the HYPRE arm's lines here.
hypre_keywords() {
  [[ -n "${NVRS_HYPRE_AMGFAST:-}" ]]  && printf 'AMGFAST %s\n' "$NVRS_HYPRE_AMGFAST"
  [[ -n "${NVRS_HYPRE_AMGREUSE:-}" ]] && printf 'AMGREUSE %s\n' "$NVRS_HYPRE_AMGREUSE"
  return 0
}

# The arm's deck lines, one per line.
arm_keywords() {
  case "$1" in
    amgx)       printf 'LINSOLVER AMGX\n' ;;
    hypre)      printf 'LINSOLVER HYPRE\n'; hypre_keywords ;;
    ginkgo)     printf 'LINSOLVER GINKGO\nGINKGOPREC 16\n' ;;
    ginkgo32)   printf 'LINSOLVER GINKGO\nGINKGOPREC 32\n' ;;
    ginkgo_abj) printf 'LINSOLVER GINKGO\nGINKGOPREC 16\nSTAGE2 GINKGO\n' ;;
    amgx_fp64)  printf 'LINSOLVER AMGX\nEMULATE 0\n' ;;
    amgx_fp64amg) printf 'LINSOLVER AMGX\nMIXED 0\nEMULATE 0\n' ;;
    amgx_tensorcore) printf 'LINSOLVER AMGX\nJACOBIAN TENSORCORE\nTCPRECISION %s\nTCVALIDATE 0\n' "${NVRS_TCPRECISION:-TF32}" ;;
    *)          return 1 ;;
  esac
}

# The same lines joined with "; ", the form run_scaling.sh's EXTRA takes.
arm_extra() { arm_keywords "$1" | paste -sd';' - | sed 's/;/; /g'; }

# arm_apply_deck <src> <dst> <arm>
# Copy a deck with the arm's keywords in force: any line the deck already has
# for them is dropped, then the arm's lines are appended (the last occurrence
# of a keyword wins, and dropping the old ones keeps the copy unambiguous to a
# reader).  STAGE2 is only replaced for the arm that sets it, so a deck's own
# STAGE2 ZLINE survives in the others.
arm_apply_deck() {
  local src="$1" dst="$2" arm="$3" re='LINSOLVER|AMGLIB|GINKGOPREC'
  [[ "$arm" == ginkgo_abj ]] && re="$re|STAGE2"
  [[ "$arm" == amgx_fp64 ]] && re="$re|EMULATE"
  [[ "$arm" == amgx_fp64amg ]] && re="$re|EMULATE|MIXED"
  [[ "$arm" == amgx_tensorcore ]] && re="$re|JACOBIAN|TCPRECISION|TCVALIDATE"
  if [[ "$arm" == hypre ]]; then
    [[ -n "${NVRS_HYPRE_AMGFAST:-}" ]]  && re="$re|AMGFAST"
    [[ -n "${NVRS_HYPRE_AMGREUSE:-}" ]] && re="$re|AMGREUSE"
  fi
  grep -vE "^[[:space:]]*($re)([[:space:]]|$)" "$src" > "$dst"
  arm_keywords "$arm" >> "$dst"
}

# Every arm the binary can run: all six with Ginkgo linked in, else the three
# that need only AmgX and HYPRE.  case_backends (nvrs.sh) keeps the Ginkgo arms
# to Norne and amgx_fp64 to Norne and SPE10.
arm_defaults() {
  local exe="$1" arms
  if command -v ldd >/dev/null 2>&1 && ldd "$exe" 2>/dev/null | grep -qi ginkgo; then
    arms="$NVRS_ALL_ARMS"
  else
    arms="amgx hypre amgx_fp64 amgx_fp64amg amgx_tensorcore"
  fi
  # The all-FP64 arm needs its own binary; without it the arm is left out.
  [[ -x "${exe%_fp64}_fp64" ]] || arms="${arms/ amgx_fp64amg/}"
  # So does the Tensor Core arm.
  [[ -x "${exe%_fp64}_tc" ]] || arms="${arms/ amgx_tensorcore/}"
  echo "$arms"
}

# nvrs_pick_exe EXE DECK: the binary that runs DECK as written.  The
# preconditioner precision is fixed at compile time: nvrs_cuda has the FP32
# preconditioner (AmgX dFFI), nvrs_cuda_fp64 (make FP64_PRECOND=1, built by
# ./nvrs.sh install) the FP64 one (AmgX dDDI).  So a deck that says MIXED 0
# runs on the _fp64 sibling of EXE and any other deck on EXE: the deck
# keyword is what chooses.  JACOBIAN TENSORCORE likewise needs the _tc sibling
# (make TENSORCORE=1).  The last line of each wins, as in deck.cpp.  Prints the
# path, or an error naming the build command and returns 1.
nvrs_pick_exe() {
  local exe="${1%_tc}" deck="$2" m j
  exe="${exe%_fp64}"
  m="$(awk '{ sub(/--.*/, "") } NF >= 2 && toupper($1) == "MIXED" { v = $2 }
            END { print v }' "$deck" 2>/dev/null)"
  j="$(awk '{ sub(/--.*/, "") } NF >= 2 && toupper($1) == "JACOBIAN" { v = toupper($2) }
            END { print v }' "$deck" 2>/dev/null)"
  if [[ "$m" == 0 ]]; then
    if [[ ! -x "${exe}_fp64" && "$j" != TENSORCORE && "$j" != TC && "$j" != 2 ]]; then
      echo "ERROR: $deck sets MIXED 0 (FP64 preconditioner), and ${exe}_fp64 is not built." >&2
      echo "       cd $(dirname "$exe") && make FP64_PRECOND=1     (or ./nvrs.sh install)" >&2
      return 1
    fi
    exe="${exe}_fp64"
  fi
  if [[ "$j" == TENSORCORE || "$j" == TC || "$j" == 2 ]]; then
    if [[ ! -x "${exe}_tc" ]]; then
      echo "ERROR: $deck sets JACOBIAN TENSORCORE, and ${exe}_tc is not built." >&2
      echo "       cd $(dirname "$exe")/benchmark && TENSORCORE=1 ./nvrs.sh install" >&2
      return 1
    fi
    exe="${exe}_tc"
  fi
  echo "$exe"
}
