#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# spe10_hypre_screen.sh -- the SPE10 run of hypre_screen.sh.
#
#   ./scripts/spe10_hypre_screen.sh                  # every candidate
#   ./scripts/spe10_hypre_screen.sh fast0_reuse10    # just these
#
# First 100 days of decks/SPE10_MATCHED.DATA, one GB200 (16 Sep 2026):
#   amgx           16.2 s  Krylov  4626  Newton 173
#   fast1_reuse0   49.5 s  Krylov 14532  Newton 257   (the deck, HYPRE)
#   fast1_reuse10  56.1 s  Krylov 20944  Newton 343
#   fast0_reuse0   28.6 s  Krylov  3603  Newton 193
#   fast0_reuse10  23.4 s  Krylov  5656  Newton 171   (the HYPRE arm now)
exec env CASE=SPE10 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hypre_screen.sh" "$@"
