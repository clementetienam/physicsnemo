// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
// wellcell.hpp — per-cell well parameters shared by the host driver and the
// assembly kernels (Peaceman well index, BHP target, injection rate).
#pragma once
#include "common.hpp"
namespace nvrs {
struct WellCell {
  int  is_inj = 0, is_prod = 0;
  Real wi = 0;       // Peaceman well index 2π k h /(ln(re/rw)+skin)·C_darcy
  Real pwf = 0;      // pressure at this completion [psia]
  Real bhp = 0;      // BHP at the WELSPECS reference depth [psia]
  // Injection, one rate per phase, so that a well injects what the deck says
  // it injects.
  //
  // This used to be a single water rate, with the phase discarded.  A field
  // deck injects gas as readily as water -- Norne's C-4H turns into a gas
  // injector three months in -- and putting a gas rate into the water equation
  // is not an approximation, it is a different well.  Converted through the
  // gas surface-volume factor as well, a gas rate of 1.4 million sm3/day
  // arrived as fifty thousand barrels a day of water into two cells; their
  // water saturation pinned at one immediately and no timestep could converge.
  Real qw_inj = 0;   // water surface rate [STB/D] (+ into the reservoir)
  Real qo_inj = 0;   // oil   surface rate [STB/D]
  Real qg_inj = 0;   // gas   surface rate [MSCF/D]
  // Frozen-mobility derivative of this completion's allocated surface rate
  // with respect to its drawdown.  Rate-controlled injectors share one BHP:
  // the local +A diagonal and the matrix-free -A_i*A_j/sum(A) correction
  // together differentiate redistribution at fixed total well rate.
  Real inj_dqdp = 0;
  Real inj_dqdu[3] = {0, 0, 0}; // fixed-BHP direct dq/du
  int inj_rate_coupled = 0;
  // Mode-9 injector crossflow: complete local rate derivative and the
  // eliminated [BHP, oil fraction, water fraction] factors.
  Real inj_direct[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  Real inj_dgdu[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  Real inj_drdg[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int inj_xflow_coupled = 0;

  // Crossflow composition: the WELLBORE's own reservoir-volume mixture,
  // as fractions summing to 1 (oil, water, gas).  See well_control.hpp for
  // why a reverse-flowing completion must carry the well's composition and
  // not the local formation's -- this is that composition, computed once per
  // well from its forward-flowing completions (well_report.cu,
  // apply_producer_rate_targets) and copied onto every completion of the
  // well, including the reverse-flowing ones that actually consume it.
  // `xflow_model` gates which reverse-flow formula the assembly kernel uses:
  // 0 (the default) is the untouched legacy formula -- each phase at its own
  // LOCAL formation mobility, whatever WELLXFLOW 1 or an uncontrolled well
  // already did.  1 means the no-crossflow constrained solve ran for this
  // well (well_control.hpp) and mix_o/mix_w/mix_g below are the wellbore's
  // own composition to use instead.  A well cannot be mid-way between the
  // two: it is one or the other for every completion, every call.
  int xflow_model = 0;
  // Surface-component fractions (OPM's cmix_s), not reservoir-phase
  // fractions. Reverse completions convert these with their local B/Rs/Rv.
  Real mix_o = 0, mix_w = 0, mix_g = 0;
};

// Matrix-free low-rank part of an eliminated well equation.  Mode zero is the
// shared BHP used by the current pressure-only correction.  Modes one and two
// are reserved for two independent wellbore-mixture fractions; together they
// can represent the nonlocal part of a complete black-oil producer elimination
// without inserting a dense completion-by-completion block into reservoir CSR.
struct WellSchurEntry {
  int cell = -1;
  int group = -1;
  // d(eliminated well modes)/d(cell reservoir primaries), row-major [mode][v].
  Real dg_du[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  // d(cell component residuals)/d(eliminated modes), row-major [eq][mode].
  Real dr_dg[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};
}  // namespace nvrs
