# Norne response-matching engineering log

Last updated: 2026-09-11

## Objective and experiment policy

Bring NVRS-CUDA's Norne well responses and reservoir states into satisfactory
agreement with OPM Flow. Use short checkpoints (currently 87, 268, and 700
days) for diagnosis and screening. Run the full 3312-day case only after a
change is clearly beneficial at the shorter horizons.

This file records both retained work and rejected experiments. Paths under
`/tmp` are local experiment artifacts and may disappear after a reboot.

## Reference cases and commands

- CUDA working directory:
  `full_reservoir_simulator/GPU_reservoir_simulator/cuda`
- Production deck: `decks/NORNE.DATA` (`TEND 3312`, `OPMCONV 1`)
- Original Norne dataset: `dataset/NORNE/NORNE_ATW2013.DATA`
- Stored Flow reference basename:
  `benchmark/runs/NORNE/opm/NORNE_ATW2013`
- Vendored matching OPM source: `opm_2024_10_src`
- Current scratch deck: `/tmp/NORNE_VAPPARS_268.DATA`. It is reset to
  `TEND 268` and `OUTDIR /tmp/nvrs_norne_next_268` after each experiment.

Build:

```sh
source ../../../mixed_precision_emmulation/hypre_env.sh >/dev/null
unset OPAL_PREFIX
make -j1
```

Single-GPU run:

```sh
HWLOC_COMPONENTS=-gl,-opencl /usr/bin/mpirun --oversubscribe -np 1 \
  ./nvrs_cuda /tmp/NORNE_VAPPARS_268.DATA
```

Targeted PVT regression:

```sh
g++ -O2 -std=c++17 -Iinclude tests/test_pvt.cpp src/pvt_tables.cpp \
  -o /tmp/nvrs_test_pvt
/tmp/nvrs_test_pvt dataset/NORNE/INCLUDE/PVT/PVT-WET-GAS.INC
```

Result comparison uses `scripts/ecl.py:EclipseCase`. Convert the metric Flow
output before comparing with NVRS field-unit output:

- liquid/reservoir rate or volume: multiply by 6.28981077
- gas rate or volume: multiply by 0.0353146667
- pressure: multiply bar by 14.5037738

## Retained implementation work

The working tree contains earlier changes as well as the work below. Preserve
unrelated dirty-tree changes.

- Carlson (`EHYSTR`, model 0) non-wetting scanning curves for gas/oil and
  oil/water, with reversal continuity.
- `IMBNUM` and imbibition endpoint handling, including partial I-endpoint edit
  fallback to the unscaled imbibition table endpoints.
- MPI halo synchronization and transactional rollback of hysteresis history.
- Three-point `SCALECRS`, SWATINIT capillary scaling, MULTPV/MINPV, exact
  corner-point connections, faults, MULTREGT/NNC, threshold pressure, and
  PERMZ-related corrections.
- OPM-style default three-phase oil relative permeability.
- Common-datum BHP well controls, no-crossflow/crossflow-composition handling,
  `WELOPEN`, and history-rate behavior.
- RSVD/DRSDT/VAPPARS history, including accepted-state cache invalidation and
  rollback.
- Full wet-gas `PVTG` pressure/Rv interpolation using OPM's right-extreme
  behavior. `Bg` and gas viscosity now use inverse-property interpolation like
  OPM.
- VAPPARS saturation derivatives in accumulation, flux, and well Jacobians:
  `dBg/dSo`, `d(mu_g)/dSo`, and product-rule component terms.
- Regression tests for Norne parsing/endpoints, Carlson continuity, PVT, and
  VAPPARS derivatives.

The current PVT/VAPPARS suite passes 88 checks with no failures. The Norne
schedule suite passes 111 checks with no failures.

### Retained 2026-09-07 corrections

- ROCK accumulation now evaluates both old and new pore-volume multipliers
  against the deck's ROCK reference pressure using OPM's
  `1 + x + 0.5*x*x` law. The host inventory uses the same multiplier.
- TUNING records are parsed; TSINIT and TSMAXZ are available to the timestep
  controller. The continuous iteration controller now uses OPM's exact growth
  and decay formulas. It remains opt-in; screening showed timestep choice was
  not the main Norne mismatch.
- PVTO now follows OPM LiveOilPvt's left-extreme guided interpolation and
  interpolates inverse B and inverse(B*mu).
- Most importantly, `host_endscale()` now copies `grid.scalecrs` into the
  host SatTables. Before this correction the CUDA reservoir used Norne's
  `SCALECRS YES` three-point relperm maps, but producer BHP control and well
  reporting silently evaluated two-point maps. This split physics suppressed
  D-1H productivity and caused excessive drawdown and water-rate error.
- `benchmark/checkpoint_compare.py` provides a reusable OPM/NVRS numeric
  comparison and safely handles zero reference BHP values.
- Carlson histories now start from the actual initial `Sg` and `Sw+Sg`, as
  OPM does, rather than sentinels that can discard a first-step reversal.
- The one-curve PVDO evaluator no longer indexes a nonexistent second
  saturated node; it uses inverse-B and inverse-(B*mu) interpolation. Focused
  tests also pin PVTO left-extreme guided interpolation.

### Mixed-precision accuracy audit (2026-09-08)

Three matched 87-day runs isolated solver precision from model semantics:

| Configuration | Runtime | Newton | Krylov |
|---|---:|---:|---:|
| FP32 CPR + df64 operator (production) | 4.64 s | 166 | 3405 |
| FP32 CPR + native-FP64 operator | 4.65 s | 166 | 3405 |
| FP64 CPR + native-FP64 operator | 4.77 s | 166 | 3406 |

All three have the same OPM errors to printed precision. Against full FP64,
the production mixed run differs by only 0.163 psi pressure RMS, `4.15e-8`
SWAT RMS and `1.72e-8` SGAS RMS; field-rate differences are at output
round-off. Mixed precision is therefore not limiting Norne accuracy and should
remain enabled.

The audit exposed an independent reporting error: FOPR/FWPR totals used
backward-Euler end-of-step rates, but FVPT/FVIT used trapezoidal report-rate
integration. OPM's report-step convention and this simulator's other totals
use backward Euler. Making FVPT/FVIT consistent improved day-87 FVPT error
from -14.770% to +0.122% with no solver or runtime change.

The retained mixed run reaches day 268 in 19.53 s. Endpoint errors are FOPR
-2.736%, FWPR -1.266%, FGPR -3.758%, FVPR +0.325% and FVPT +0.209%; pressure,
SWAT and SGAS RMS versus the nearest Flow restart are 59.71 psi, 0.00446 and
0.00872. Remaining differences are model/timestep semantics, not precision.

## Quantitative checkpoints

### Best retained 268-day result

Current best artifact: `/tmp/nvrs_norne_signed5_268/NORNE`.

Runtime: 49.08 s. Endpoint/median errors are FOPR -2.720/-1.487%, FWPR
-1.721/-1.153%, FGPR -3.807/-1.841%, FVPR +0.351/+0.167%, FOPT
-1.929/-0.725%, and FWPT -1.203/-0.700%. At day 268 the producer BHP errors
are +9 psi B-2H, +83 B-4H, +48 C-1H, -8 C-4H, +76 D-1H, +24 D-2H, and +68
D-4H. Against Flow day 252, pressure bias is -4.42 psi and RMS 57.19 psi; the
16-day restart-time mismatch materially contributes to that RMS. SWAT RMS is
0.00456 and SGAS RMS 0.00867.

The faster pre-guard result is `/tmp/nvrs_norne_hyst_init_268/NORNE`: 35.09 s,
FOPR/FWPR/FGPR -1.955/-2.908/-4.521%, pressure RMS 56.72 psi, and C-1H BHP
+173 psi. It has slightly better oil and state RMS but worse water, gas, BHP,
and twice the accumulated hydrocarbon component defect.

The preceding best is retained below for provenance, but is superseded.

Artifact: `/tmp/nvrs_norne_wetgas_satjac_268/NORNE`

Runtime: 28.11 s.

| Vector/state | Endpoint error vs Flow | Median error vs Flow |
|---|---:|---:|
| FOPR | -10.857% | -6.450% |
| FWPR | +20.081% | +20.676% |
| FGPR | +1.442% | +1.442% |
| FVPR | +1.119% | +0.548% |
| FOPT | -6.743% | -2.840% |
| FWPT | +21.290% | +23.760% |

At the nearest Flow restart (NVRS day 268, Flow day 252): pressure bias
-55.84 psi and RMS 93.36 psi; SWAT RMS 0.00561; SGAS RMS 0.01430.

The prior rollback-corrected 268-day baseline was slightly worse in every
reported field vector and state norm, so the wet-gas PVT plus saturation
Jacobian is retained.

### Best retained 700-day result

Artifact: `/tmp/nvrs_norne_signed5_700/NORNE`.

A relaxed signed-component balance guard now supplements the legacy L2 MB
surrogate: each accepted step must keep the largest signed component defect
below `5*MBTOL`. A device-side signed residual ledger makes this check and
reports cumulative discrete defects with negligible overhead. At 697 days,
endpoint/median errors are FOPR +11.561/-0.645%, FWPR -29.149/-5.076%, FGPR
-16.491/-8.305%, FVPR +0.014/+0.132%, FOPT -0.459/-1.895%, FWPT
+22.080/-1.228%, and FVPT -1.713/-2.960%. Pressure bias is +52.51 psi and RMS
88.20 psi; SWAT RMS is 0.01655 and SGAS RMS 0.02201. Runtime is 780.45 s.

This is materially more accurate than the preceding retained result in every
endpoint phase rate and every saturation-state RMS, though about 161% slower.
Pressure RMS also improves strongly (129.79 to 88.20 psi), but is not as good
as the faster 10x guard's 84.30 psi. The
signed ledger at 697 days still shows +20.92 MMstb oil, -9.82 MMstb water,
and -8.21 Bscf gas, so `5*MBTOL` reduces drift but does not eliminate it.

The faster `10*MBTOL` artifact is `/tmp/nvrs_norne_signed10_700/NORNE`:
460.86 s, endpoint FOPR/FWPR/FGPR +17.598/-34.592/-20.450%, pressure RMS
84.30 psi, SWAT RMS 0.02337, and SGAS RMS 0.02748. Use it when turnaround is
more important than the best current phase-response match.

The preceding retained result is preserved below for provenance.

Artifact: `/tmp/nvrs_norne_hyst_init_700/NORNE`.

Runtime: 298.86 s. Endpoint/median errors are FOPR +25.908/+0.304%, FWPR
-46.183/-15.804%, FGPR -25.519/-9.865%, FVPR -0.034/+0.027%, FOPT
+4.070/-1.036%, FWPT +8.789/-2.777%, and FVPT -1.793/-3.035%. Pressure bias
is +44.50 psi and RMS 129.79 psi; SWAT RMS is 0.03025 and SGAS RMS 0.03653.

The preceding host-SCALECRS artifact is
`/tmp/nvrs_norne_host_scalecrs_700/NORNE` (310.16 s). Correct initial Carlson
history improves both saturation RMS values and the late oil/gas endpoints
slightly, while water and pressure are slightly worse. It is retained as the
physically correct state definition, not claimed as a decisive response match.

### Signed residual ledger and 268-day guard result

The former in-run inventory comparison integrates only report-date rates and
substantially overstates early imbalance. A new device-side ledger sums each
component residual in the assembly and accumulates `sum(R)*dt` only for
accepted substeps. Internal connection fluxes cancel, so this is the exact
discrete component defect left by nonlinear acceptance without expensive
host-side well reevaluation.

At 87 days the ledger reports +0.160 MMstb oil, essentially zero water, and
-0.0825 Bscf gas; the report-rate diagnostic had claimed +0.723 MMstb oil.
At 268 days without a signed guard it reports +2.810 MMstb oil, -0.00165
MMstb water, and -2.589 Bscf gas.

With the retained `5*MBTOL` guard, the 268-day defects fall to +1.181 MMstb
oil and -1.116 Bscf gas. Endpoint/median errors are FOPR -2.720/-1.487%, FWPR
-1.721/-1.153%, FGPR -3.807/-1.841%, and FVPR +0.351/+0.167%. C-1H BHP
improves from +173 to +48 psi. Pressure RMS is 57.19 psi, SWAT RMS 0.00456,
and SGAS RMS 0.00867. Runtime is 49.08 s versus 35.87 s without the guard.
Artifact: `/tmp/nvrs_norne_signed5_268/NORNE`.

At 87 days the 5x guard remains essentially neutral: FOPR -0.620%, FWPR
-0.928%, FGPR -0.342%, pressure RMS 6.80 psi, and runtime 4.53 s. Artifact:
`/tmp/nvrs_norne_signed5_87/NORNE`.

## Rejected experiments

### Pressure-viscosity Jacobian terms

Adding `dmu/dp` to pressure-mobility derivatives increased the 268-day runtime
from about 28.1 to 42.3 s and slightly worsened agreement. The source change
was reverted. Artifact: `/tmp/nvrs_norne_pressure_visc_jac_268`.

### Signed field material-balance convergence

OPM computes MB from `abs(B_avg * sum(cell residual)) * dt / PV`; NVRS's
existing approximation uses per-equation L2 residuals. A trial changed the
reduction to signed global component sums but did not yet implement OPM's
cell-averaged formation-volume factors or its complete nonlinear controller.

At 268 days this regressed runtime from 28 to 116 s and worsened all main
metrics: FOPR endpoint -12.960%, FWPR +23.121%, FGPR +3.246%, pressure RMS
101.05 psi, SWAT RMS 0.00616, and SGAS RMS 0.01700. The source change was
reverted. Artifact: `/tmp/nvrs_norne_signedmb_268`.

Do not reintroduce only the signed sum. If revisited, implement the coherent
OPM formulation, including `B_avg`, strict/relaxed-iteration semantics, and
adaptive timestep behavior, then screen it at 87/268 days.

The coherent criterion was also screened: signed component sums, arithmetic
cell-average phase B, strict MB `1e-7`, and final-iteration relaxed MB `1e-6`,
matching the vendored OPM source. It advanced only five report states in 60 s
on the 87-day case (baseline 4.5 s), so it too was reverted. The criterion is
correct; the current Newton/well coupling cannot efficiently satisfy it.
Partial artifact: `/tmp/nvrs_norne_opm_mb_87`.

### VAPPARS Rs switching experiments

A direct within-Newton switch to saturated Rs whenever `Sg > 1e-7` is
discontinuous at phase appearance and failed before the first report, after
timestep halving to 0.125 d. Artifact:
`/tmp/nvrs_norne_vaprs_switch_268/NORNE`.

A lagged active set marked accepted free-gas cells uncapped for the next
solve. It was stable but regressed day 268 to FOPR -7.50%, pressure RMS 182.4
psi, and producer BHP errors of roughly 180--417 psi. Artifact:
`/tmp/nvrs_norne_vaprs_active_268/NORNE/NORNE`. Proper OPM behavior requires
the third primary variable to switch meaning between Sg and Rs; neither a raw
saturation threshold nor a post-step cap update is component-conservative.

### Tighter relaxed CNV

`CNVRELAXED 0.1` also became impractical just after day 268, reaching 27--30
substeps with CNV stalled around 0.1--0.3 before the 60 s screen ended.
Partial artifact: `/tmp/nvrs_norne_cnv01_424`. It was rejected.

### Rejecting non-descending line-search updates

Immediately rejecting a timestep when all six line-search trials increased
the residual preserved the 87-day answer and halved its simulator runtime
(about 2.2 versus 4.5 s), but selected a different nonlinear branch later.
At day 268 C-1H BHP was -548 psi relative to Flow, versus +173 psi for the
retained baseline. Artifact: `/tmp/nvrs_norne_lsreject_268/NORNE`.

A two-strike variant allowed one smallest non-descending update, then rejected
the timestep on a second consecutive failure. It also reproduced the retained
87-day result exactly and ran in 2.26 s. At day 268 it was less severe than
immediate rejection but still moved C-1H BHP to -248 psi and worsened pressure
RMS from 56.72 to 61.17 psi; field-rate differences were negligible (FOPR
-1.890%, FWPR -3.084%, FGPR -4.565%). Runtime was 32.42 s. Artifacts:
`/tmp/nvrs_norne_ls2_87/NORNE` and `/tmp/nvrs_norne_ls2_268/NORNE`.

Both rules were reverted. A non-descending step can be necessary to traverse
the nonsmooth rate-control/phase boundary, and the number of failed searches
is not a safe rejection criterion by itself.

### Smooth Rs/free-gas complementarity without full Jacobian

A continuous bridge kept the accepted Rs cap at `Sg=0` and approached
`Rs_sat(p)` as `Sg/(Sg+1e-5)` when free gas appeared. This addresses a real
defect in the compact formulation: under VAPPARS, NVRS can leave gas-present
oil undersaturated after pressure cycling, while OPM switches that cell onto
the saturated PVTO line. However, changing the residual without adding
`dRs/dSg`, `dBo/dSg`, oil-viscosity, density, flux, well, and accumulation
terms to the analytic Jacobian made the 87-day run impractical: it produced
only four report states in more than 60 s. The experiment was terminated and
reverted. Partial artifact: `/tmp/nvrs_norne_rssmooth_87`.

Do not retry a hard or smooth Rs switch as a residual-only patch. Implement
the third-primary-variable formulation and its derivatives coherently.

## Findings that rule out suspected causes

### Mixed precision is not the primary mismatch source

Two matched 268-day precision ablations were run on 8 September 2026 with
the retained 5x signed-balance guard and otherwise identical deck settings.

Changing only `EMULATE` from 1 (the two-float, roughly 48-bit matrix
operator) to 0 (native FP64 operator arithmetic) changed sampled producer
rates by at most 0.052%. Final-state RMS differences were 4.44 psi in
pressure, 1.0e-5 in water saturation, and 4.4e-5 in gas saturation. Accepted
Newton counts were 704 versus 714 and wall times were 54.18 versus 54.04 s.
Artifacts: `/tmp/nvrs_norne_next_268` and
`/tmp/nvrs_norne_emulate0_268`.

A separate binary was then built in `/tmp/nvrs-build-fp64` with
`NVRS_MIXED_PRECISION=OFF`, FP64 HYPRE, and AmgX dDDI. Relative to the current
mixed FP32-preconditioner build, its sampled producer rates again differed by
less than 0.05%. Final-state RMS differences were 4.08 psi in pressure,
8.6e-5 in water saturation, and 3.2e-5 in gas saturation. Accepted Newton
counts were 704 mixed versus 699 FP64; linear counts were 13,535 versus
14,337. Artifact: `/tmp/nvrs_norne_fp64_268`.

The localized maxima (about 400--463 psi and 0.0038--0.0067 saturation) show
that precision can select a slightly different path near nonsmooth phase or
well-control switches. It does not generate the systematic 10--30% phase-rate
bias. The main state, nonlinear residual/Jacobian, Newton update, and outer
FGMRES are FP64 in the mixed build; FP32 is confined to the approximate CPR
preconditioner. Continue prioritizing nonlinear conservation, coherent
Sg/Rs primary-variable switching, and well-control/Jacobian coupling.

### Saturation functions are matching Flow

A cellwise audit evaluated NVRS's endpoint-scaled curves using Flow's own
state and INIT endpoint arrays.

- At initial conditions: WATKR RMS 1.24e-9 and OILKR RMS 1.47e-8.
- The apparent initial GASKR offset was diagnostic rounding: Norne's table
  SOGCR is 0.0001, not zero.
- At day 87, with Carlson history reconstructed from Flow report states:
  WATKR RMS 3.18e-9, GASKR RMS 4.39e-8, and OILKR RMS 2.54e-6. Only 28 of
  44,431 oil cells differed by more than 1e-5; none differed by 1e-3.

Therefore endpoint scaling, the default three-phase oil law, and Carlson
scanning are not the current limiting error.

### The phase split is already wrong at startup

At day 8, D-1H oil and gas rates match Flow within about 0.4% because RESV is
the active history control, but water is +13.3% and NVRS BHP is 286 psi lower.
By day 25 water is +28.0% and BHP is 507 psi lower. This occurs before material
hysteresis history develops, so it is a completion/well or early reservoir
state issue, not Carlson.

At day 268 all active producers show low BHP relative to Flow: B-2H -360 psi,
D-1H -535 psi, D-2H -228 psi, B-4H -72 psi, and D-4H -365 psi. Their combined
FVPR remains within about 1%, confirming that the rate controls are broadly
meeting reservoir-volume targets while the completion allocation and pressure
needed to meet them differ.

### D-1H connection factors are converted correctly

The day-87 debug run (`NVRS_DEBUG_WELL=D-1H`) shows, for example, the first
deck connection factor 5.505 m3/day/bar converted to 2.38734 rb/day/psi, which
is exactly 5.505 * 6.28981077 / 14.5037738. All eight D-1H COMPDAT factors
follow the same conversion. Thus the gross BHP mismatch is not a missing
metric-to-field conversion.

The debug artifact is `/tmp/nvrs_norne_debug_d1h_87`. At the initial state the
eight D-1H RESV productivity coefficients sum to about 46.74 rb/day/psi and
the corrected completion datum pressures are all about 3909 psi. The solved
BHP is 3137 psi for the 36,060 rb/day target. This is internally consistent.

### Full-run regression and convergence-policy diagnosis (2026-09-08)

The first post-merge full run reached all 247 report steps (3312 days) in
1846.71 s on an RTX 6000 Ada. It generated
`benchmark/runs/compare_NORNE.html`. Endpoint errors were FOPR +123.4%, FWPR
-84.7%, FGPR +48.4%, and FVPR +1.29%; cumulative oil and water errors were
+36.6% and -90.6%. The nearly correct reservoir-volume total beside the bad
phase split rules out a common rate-unit error. Time-resolved diagnostics show
the water bank beginning to lag around days 424--700 and then diverging rapidly.

Inspection of OPM Flow 2024.10 corrected an earlier assumption: its defaults are
`RelaxedMaxPvFraction=0.03`, `ToleranceCnvRelaxed=1.0`, strict MB `1e-7`,
and relaxed MB `1e-6`--not an unlimited `CNV` bound. The real NVRS defect was
policy composition. `OPMCONV` was only an alternate early exit: the combined
relative-L2 test could still accept a timestep whose signed component MB or CNV
failed. Line search and stall detection also minimized that raw L2 norm. Gas
residuals are roughly three orders larger in surface units, so those decisions
were almost blind to a coherent water residual/defect.

Making MB+CNV authoritative alone still failed: the unchanged raw-L2 line
search chopped the 697-day case into thousands of substeps, took 1012.55 s, and
accumulated -20.78 MMSTB of accepted water residual. It improved day-424 FWPR
from -27.2% to -11.0%, but day-697 FWPR was still -44.5%. This screen was
rejected.

The retained correction uses one normalized component merit,
`max(signed_MB/MBTOL, CNV/effective_CNV_tolerance)`, consistently for
acceptance, line search, and stall detection. At day 424 it took 44.50 s and
gave FOPR -4.56%, FWPR +1.56%, FGPR -7.03%, and FVPR +0.21%. At day 697 it
took 136.44 s; errors were FOPR -6.06%, FWPR -9.25%, FGPR -5.35%, and FVPR
-0.14%. Accepted water drift was only -0.177 MMSTB, versus -20.78 MMSTB in the
rejected strict-exit/raw-L2 run. Individual water-rate improvements include
D-1H from -85.8% to +1.5%, D-4H from -44.8% to -12.8%, and E-3H from -46.6%
to -10.3% at day 697.

The halfway checkpoint reached day 1637 in 269.58 s. Field-rate errors were
FOPR -8.15%, FWPR +7.51%, FGPR -2.78%, and FVPR -1.51%; the rejected full-run
trajectory was already at +40.25%, -91.92%, -16.07%, and +1.23% there. This
confirms that the correction addresses the reported second-half divergence,
not merely the early report steps.

The final retained run completed all 247 report steps (3312 days) in 461.52 s
on the RTX 6000 Ada, using 13,817 Newton iterations and 222,440 linear
iterations. Endpoint errors are FOPR -3.68%, FWPR -1.84%, FGPR -2.61%, and
FVPR -0.65%; cumulative errors are FOPT -6.57%, FWPT +4.62%, and FVPT -0.97%.
The final accepted residual defect is +4.96 MMSTB oil, -2.37 MMSTB water, and
-6.99 MMSCF gas. The self-contained comparison report was regenerated at
`benchmark/runs/compare_NORNE.html` from this output and the existing OPM Flow
reference.

### Rejected 424-day screens (2026-09-08)

- Disabling Carlson hysteresis changed endpoint FWPR only from -27.19% to
  -27.71% and slightly worsened pressure RMS. Carlson is not causing the
  day-424 water regression.
- Disabling residual line search failed to complete within the 120 s screen
  and was already slower by step 32. Retain line search.
- Relaxing the former signed component guard from `5*MBTOL` to `50*MBTOL`
  reduced runtime from 74.86 to 56.06 s but worsened FWPR to about -42.3%,
  pressure RMS to 62.3 psi, and oil inventory error. That guard has since been
  superseded by authoritative MB+CNV acceptance and component-scaled merit.
- `WELLITER 0` reduced runtime to 64.70 s but damaged all cumulative responses:
  FOPT -18.57%, FWPT -36.24%, and FVPT -18.89%. Re-solving the common-BHP
  controls during Newton is necessary.
- Reusing an accepted line-search assembly on the next Newton iteration was
  rejected. The apparently duplicate well-control update is part of the
  fixed-point coupling; skipping it made the checkpoint slower rather than
  preserving the trajectory.
- Replacing the direct datum-to-completion density correction with cumulative
  segment drops using each segment's local reservoir density gave small
  day-424 improvements: FWPR -25.92%, pressure RMS 40.61 psi, and SWAT RMS
  0.00990 in 76.06 s. It worsened C-1H BHP around day 268 and was reverted.
  OPM `StandardWellConnections` accumulates segment drops but computes each
  segment density from bottom-up flowing wellbore component rates, not the
  terminal reservoir cell's local mobility mixture. A complete correction
  needs that rate/composition pass and a repeated BHP solve; retaining only
  the geometric half-model is not justified. Artifact:
  `/tmp/nvrs_norne_segdp_424/NORNE_SEGDP`.
- Disabling per-cell SWATINIT Pcow scaling took 65.90 s and improved endpoint
  FWPR to +0.18% and E-3H BHP error to +37 psi, but worsened cumulative FWPT
  to +19.59% and SWAT RMS to 0.02821. OPM 2024.10's
  `EclMaterialLawManager::applySwatinit()` confirms that persistent maximum-
  Pcow scaling by `Pc_equil/Pc(SWATINIT)` is intended. The ablation is a useful
  localization result, not a physical correction; it was reverted. Artifact:
  `/tmp/nvrs_norne_nopc_424/NORNE_NOPC`.

### Runtime interpretation

The rejected 1847 s result is not representative of GPU strength alone. On the
same RTX 6000 Ada, earlier complete Norne runs took 281--367 s, including a
296.93 s run with 46,035 linear systems; the slow run solved 46,548 systems.
At day 424 the slow trajectory used 74.8 s versus 17.9 s historically. The
dominant regression is nonlinear/control/assembly work and timestep collapse,
not linear algebra throughput or mixed precision. The GB200's 77 s full-run
record remains plausible because it used only 6150 Newton iterations and 616
AMG rebuilds, in addition to the faster hardware.

## Current diagnosis and next work

The host/device SCALECRS inconsistency was the dominant early-time error. At
day 87 it changed FWPR from +25.55% to -1.01%, D-1H BHP from -650 psi to +5
psi, and pressure RMS from 42.24 to 6.83 psi. Day-8 D-1H BHP is now within 7
psi of Flow. Artifact: `/tmp/nvrs_norne_host_scalecrs_87/NORNE`; runtime 4.45
s. The cumulative oil inventory diagnostic defect also fell from about 3.04
MMstb to 0.72 MMstb, though that diagnostic is not yet exact conservation.

At day 697 the retained trajectory is within -6.1% FOPR, -9.3% FWPR, -5.4%
FGPR, and -0.14% FVPR. At day 1637 it remains within -8.2%, +7.5%, -2.8%, and
-1.5%, respectively. The remaining mismatch is mainly phase allocation and
individual breakthrough timing; total produced reservoir volume remains close.
Mid-history cumulative water is still the largest field-level discrepancy
(about +21% at day 1637), and wells E-1H and E-3AH remain useful targets for a
future well-hydraulics/completion-density investigation.

The former 700-day conservation diagnosis was mostly an acceptance-policy
artifact. With authoritative signed MB+CNV and component-scaled nonlinear
merit, accepted water drift is -0.177 MMSTB at day 697 and -2.37 MMSTB at the
end of the full simulation. Mixed precision is not the cause: the 87-day
native-FP64 operator and full-FP64 screens changed rates by less than 0.05%
relative to the retained mixed-precision path.

Next work:

1. Improve nonlinear coupling/Jacobian behavior to reduce the remaining
   iteration count while preserving authoritative signed MB+CNV. Inspect the
   host fixed-point RESV BHP update and line-search iterations that retain the
   smallest trial when no trial descends.
2. Implement real per-cell Sg/Rs primary-variable switching, with the third
   unknown interpreted as Rs in oil-only cells and the corresponding analytic
   accumulation, flux, and well derivatives. At day 696, 462 oil-only cells
   changed RS by more than 1 metric unit from initial (max 27.6); by day 1637,
   3684 oil-only cells did.
3. Investigate mid-history cumulative water and the E-1H/E-3AH breakthrough
   timing using short 424-, 697-, and 1637-day screens. Do not tune well indices;
   early BHP agreement already validates their conversion.
4. Repeat the full 3312-day validation only after a short-screen change improves
   both field material balance and well responses.

The retained full validation is complete. Further experiments should use the
short/medium screens above before replacing its promoted output.


## Nonlinear runtime investigation (2026-09-08)

The final 3312-day NVRS run accepted 4,329 internal substeps, versus 355
timesteps in the OPM Flow reference. NVRS used 13,817 accepted Newton updates,
or 3.19 per accepted substep; Flow used 1,382, or 3.89 per timestep. Therefore
the roughly tenfold Newton-count gap is primarily a timestep-count gap, not
poor Newton efficiency inside a successful timestep. NVRS carried a median
timestep of 1.03 days (mean 1.32, 90th percentile 2.6), whereas Flow averaged
about 9.3 days. Rejected work was secondary: 1,520 Newton and 24,479 linear
iterations on failed attempts.

A matched 424-day profile localized 20.0 s of 43.41 s (46%) to host-side well
control refreshes. Assembly and linear solution were each about 10.8 s. Both
producer and injector routines copy the complete reservoir state and well-cell
array from device to host, rebuild saturation/PVT support, solve common-BHP
allocations, and copy the well-cell array back. This happened at the top of
every Newton iteration and again for every line-search trial. The reservoir
Jacobian itself treats those well controls as fixed, so recomputing all of them
inside line search both adds synchronization cost and changes the nonlinear
function used to judge the update. OPM instead includes coupled well
contributions through its well Schur complement. That missing coupled well
Jacobian is the main architectural reason NVRS cannot yet carry Flow-sized
timesteps.

Controller-only screens did not fix the cause. Disabling the persistent
failure ceiling repeatedly regrew to a failing step and timed out before day
424. Raising per-success growth to 2.0 took 50.75 s, and increasing the stall
limit to eight took 47.44 s, both slower than the 43.41 s baseline. Some
failure memory and early stall detection are useful; the limiting issue is the
well/reservoir nonlinear coupling.

Freezing both injector and producer controls during trial assemblies reduced
the matched 424-day runtime to 35.90 s, but was rejected. By day 1637 it had
changed C-1H from the pressure-limited branch to the rate-controlled branch,
raised field water injection from 277,646 to 296,962 STB/day, and required
3,947 accepted substeps instead of 1,786. The last 30-day report interval alone
needed 90 substeps. Injector BHP limits are active-set constraints and must be
reevaluated for a trial state.

A hybrid screen refreshed injector controls during line-search trials but kept
producer common-BHP allocations fixed until the next Newton iteration. It
looked promising on the short screens: 37.84 s at day 424, 59.56 s at day 697,
and 159.73 s at day 1637, with field-rate changes still small and the C-1H
pressure-limit branch restored. Full validation rejected it. Between report
steps 199 and 223 its trajectory collapsed, including 316 substeps on step 203
and 297 on step 221. It finished in 486.89 s with 9,574 accepted substeps; well-control refreshes
alone consumed 229.21 s (47.1% of wall time),
worse than the exact-control baseline at 461.52 s and 4,329 substeps. Trial
producer control updates have therefore been restored to preserve exact semantics; neither
injector nor producer control may be frozen safely on the evidence available.

Instrumentation now reports accepted substeps, line-search trial assemblies,
updates for which no trial descended, and total well-control refresh time.
Mixed precision is not driving the Newton count: prior full-FP64 screens
changed well rates by less than 0.05%, while the substep behavior follows the
well-control coupling and active-set treatment.

### Retained exact-semantics speedup

The hot path always evaluates injector controls immediately before producer
controls. Previously both routines independently downloaded the same complete
reservoir state, WellCell array, and accepted-history arrays. The injector now
hands its host state and updated WellCell buffers directly to the immediately
following producer solve. The handoff is explicit at the two Newton call sites
and guarded by reservoir address, field address, schedule step, accepted-state
identifier, and buffer sizes; calls used for schedule changes and reporting
cannot consume it. In the paired path the intermediate injector upload is also
unnecessary because no device work occurs before the producer performs the
final upload. Buffer capacity is retained across pairs.

DRSDT, VAPOIL and hysteresis histories are frozen throughout a nonlinear
substep and advance only after acceptance. Their host mirrors are now reused
within that substep, keyed by the same accepted-state identifier. Reporting
still performs its own refresh after the histories advance, and the next
Newton substep observes a new identifier and refreshes again.

This changes data movement only, not the control equations or update order.
At day 424, runtime fell from 43.51 to 35.28 s and measured well-control time
from 20.17 to 11.96 s. The run retained exactly 693 substeps, 1,703 Newton
attempts, and 28,513 linear iterations. Its well-rate CSV and every state
snapshot were byte-identical to the exact baseline. At day 1637, all 130 state
snapshots and the well-rate CSV were again byte-identical, with exactly 1,786
substeps, 6,531 Newton attempts, and 113,471 linear iterations; runtime was
144.09 s.

The final 3312-day validation completed in 304.20 s, down from 461.52 s
(34.1%), with the same 4,329 accepted substeps, 13,817 Newton attempts, and
222,440 linear iterations. All 248 state snapshots and the complete well-rate
CSV are byte-for-byte identical to the promoted accuracy baseline. Therefore
the OPM comparison metrics are unchanged: endpoint FOPR -3.68%, FWPR -1.84%,
FGPR -2.61%, and FVPR -0.65%; cumulative FOPT -6.57%, FWPT +4.62%, and FVPT
-0.97%. Against the 211.46 s OPM Flow reference on this machine, NVRS is now
1.44x slower rather than 2.18x slower. The next architectural speed target is
a coupled well Jacobian/Schur complement so NVRS can carry larger Flow-sized
timesteps; the safe host-transfer redundancy has now been removed.

### CUDA profile and rejected micro-optimizations

The 211.46 s Flow reference is local, not a B200 artifact.  Its PRT identifies
the host as `TONISHI-DT`, accelerator mode `none`, and 1 MPI process with 2
OpenMP threads.  It used 355 timesteps, 1,382 Newton iterations, and 2,491
linear iterations.  The old 77 s NVRS record was on a GB200 and used the older
inaccurate trajectory with only 6,150 Newton iterations and 616 AMG rebuilds.
It is therefore a combined hardware and reduced-work result, not an
apples-to-apples target for the corrected RTX 6000 Ada run.

Nsight Systems on the exact 424-day trajectory measured 4,576 launches and
10.65 s of GPU time in `k_assemble_bo3_analytic_conn`, 58.0% of all measured
GPU kernel time.  Synchronous `cudaMemcpy` calls occupied 18.01 s of CUDA API
time.  Of that, 10.86 s was charged to the 112-byte residual-statistics copy:
this is principally the synchronization that waits for the preceding assembly
kernel, not bandwidth for 112 bytes.  The well-control path made about 4,700
copies in each direction of the complete 3.91 MB `WellCell` array.  The
device engines spent about 2.95 s on those two streams of copies.

Registering the persistent injector-to-producer handoff vectors as page-locked
host memory looked positive on the short screen: 35.28 fell to 34.30 s while
retaining exactly 693 accepted substeps, 1,703 Newton attempts, and 28,513
linear iterations.  All 34 states and all 42 common well-rate columns were
byte-identical.  Full validation rejected it.  All 248 states and the complete
nonlinear work again remained identical, but runtime increased from 304.20 to
325.19 s and measured well-control time from 92.64 to 112.23 s.  Repeated CPU
access to page-locked arrays costs more over the late, well-heavy history than
the transfer staging it saves.  The registration change has been removed.

`cuobjdump` reports 231 registers per thread and 256 bytes of local stack for
the analytic connection kernel on sm_89.  A 128-thread-block experiment was
rejected: solver counts remained exact but runtime increased to 38.10 s.
Both block sizes have approximately the same register-limited active-thread
ceiling, so the extra blocks did not improve effective occupancy.  The
256-thread launch has been restored.  Further kernel work should reduce live
state/register pressure (for example by splitting cell accumulation from
connection flux/Jacobian evaluation), not merely retune block size.

Two small host-side cleanups were retained after a second full validation.
The immutable SWOF/SGOF host representation is now cached instead of building
six vectors in both well-control routines on every call.  Hydrostatic-drop
scratch is indexed by the schedule's completions (at most 504 on Norne)
instead of allocating and zeroing two 44,431-cell arrays per control pair.
The clean day-424 screen was 35.19 s versus 35.28 s.  The full run completed
in 303.853 s with well-control time 92.29 s, compared with 304.20 and 92.64 s.
All 248 state snapshots and the complete well-rate CSV are byte-identical,
with the same 4,329 substeps, 13,817 Newton attempts, and 222,440 linear
iterations.  This is only a 0.11% incremental gain, but removes avoidable
allocation/memory traffic without changing simulator semantics.

### Eliminated-well pressure coupling (2026-09-09)

The reservoir CSR differentiated every producer at fixed completion BHP even
though the host control solve recomputes one common BHP per well after each
state change.  Eliminating a rate equation gives a rank-one term per well:
`dR/dpwf * (sum(A_k dp_k) / sum(A_k))`.  `WELLSCHUR 1` applies its pressure
part matrix-free with a compact completion list; it does not densify the CSR or
change the CPR preconditioner.  BHP-limited and group-allocated wells remain on
the established path.

The first version also applied the frozen-composition formula to crossflowing
wells.  It was fast in short screens but failed full validation.

The ungated 424/1,637-day runs took 31.08/125.95 s, but full runtime was
316.42 s and accepted substeps rose from 4,329 to 5,321.  Median errors were
also slightly worse.  The missing saturation and wellbore-mixture derivatives
matter once late-life completion crossflow becomes common.  Full rejected log:
`/tmp/NORNE_WELLSCHUR_FULL.log`.

The retained safety rule disables the term for a complete well as soon as any
live completion flows backwards.  The 424/1,637-day screens then took
32.03/133.83 s.  Full Norne took 295.45 s versus the 303.853 s baseline;
accepted substeps fell 4,329 -> 4,269 and Krylov applies 222,440 -> 211,220.
Full accepted-candidate log: `/tmp/NORNE_WELLSCHUR_XFGATE_FULL.log`.

Against the same local OPM Flow summary, median NRMSE changed: WOPR
8.934 -> 8.681%, WWPR 8.695 -> 8.674%, WGPR 10.669 -> 10.655%, WWCT
4.334 -> 4.380%, WGOR 26.759 -> 26.287%, and WBHP 41.043 -> 41.574%.
Median cumulative errors were oil -6.34 -> -6.51%, water +0.45 -> +1.58%,
and gas -7.37 -> -7.45%.  Phase-rate shapes therefore remain as accurate or
improve slightly; pressure and cumulative-water bias trade a small amount for
the 8.4 s runtime reduction.  The switch stays opt-in until the missing
saturation/composition derivatives or an explicit coupled well unknown are
implemented.

Two follow-up optimizations were screened and rejected on 2026-09-09.  Adding
the analytic saturation-mobility part of the eliminated BHP derivative took
32.68 s at day 424, with 638 accepted substeps and 2,040 line-search trials,
versus 32.03 s, 634, and 1,933 for the retained pressure-only form.  The
partial extension still omits pressure-dependent mobility and hydrostatic
density derivatives, so it is not a consistent Jacobian and has been removed.

A single-rank fused kernel was also tried to replace the correction's device
memset and two kernels.  Assigning one thread per well changed the floating
summation order, took 32.94 s, used 635 accepted substeps, and increased
operator time from 2.37 to 2.84 s.  The completion-parallel two-kernel
implementation is restored.  Experimental output is under
`/tmp/nvrs_norne_wellschur_satjac_424`; that path was reused by the fused
kernel run, so its final contents belong to the latter.

### Cached assembly properties and compact well traffic (2026-09-09)

The next profile showed that the remaining exact path was dominated by two
forms of redundant work. The analytic connection kernel evaluated each cell's
pressure-dependent PVT and capillary properties once for almost every
neighbour, and the well path still transferred the complete 3.91 MB
`WellCell` array even though a schedule touches only completion cells.

The retained assembly path now prepares one dense per-cell cache of PVT,
capillary, and AIM-aware donor mobility values. The connection kernel reads
both endpoint properties from it. When water, oil, and gas choose the same
upwind cell, the cached mobility is exact and replaces two duplicate
`relperm3` evaluations; counter-current faces keep the original per-phase
evaluation. On sm_89 the main kernel fell from 231 registers plus a 256-byte
local stack to 215 registers with no local stack. In the 424-day profile,
preparation plus assembly fell from about 2.337 ms to about 1.003 ms per
assembly.

`WellCell` now has a canonical host mirror. Schedule/control changes gather
the unique local completion indices, upload only their values, and scatter
them with a small CUDA kernel. A full 424-day verifier compared the mirror
against a device download after every update and found zero byte differences.
The index list and staging allocations persist across calls. Similarly, well
control now gathers only primary variables belonging to completion cells; the
full reservoir state is downloaded only when an accepted state changes and
the regional RESV average genuinely needs every owned cell. Producer scratch
storage persists across calls, and the mobility/PVT work from its coefficient
pass is reused by its wellbore-mixture pass.

With these changes, a deterministic `WELLSCHUR 0` full run completed in
195.06 s and a repeat in 195.42 s. After the final clean rebuild and
correctness cleanup, a third run completed in 195.03 s. All three had exactly
4,429 accepted substeps, 28,173 line-search trial assemblies, 1,540 discarded
Newton iterations, 24,597 discarded linear iterations, 1,389 AMG rebuilds,
and 227,256 AMG applications. The final run spent 55.90 s in well-control
refresh, 39.92 s in AMG V-cycles, and 18.28 s in operator/halo application.
Using the slowest repeat, NVRS is 16.04 s or 7.6% faster than the local
211.46 s OPM Flow reference.

The optimized trajectory stays in the promoted accuracy band. Against OPM,
median active-period NRMSE is 9.067% WOPR, 8.698% WWPR, 10.657% WGPR, 4.556%
WWCT, 27.382% WGOR, and 41.844% WBHP. The promoted exact baseline values were
8.934%, 8.695%, 10.669%, 4.334%, 26.759%, and 41.043%, respectively. Median
cumulative errors are -6.39% oil, +2.30% water, and -7.69% gas, versus -6.34%,
+0.45%, and -7.37% in the promoted baseline. Across every active cell and all
248 snapshots, the optimized-versus-promoted relative L2 differences are
2.694% pressure, 0.290% water saturation, 0.675% oil saturation, and 1.143%
gas saturation; the final mean shifts are +37.4 psi, +0.000151 Sw, -0.000112
So, and -0.000040 Sg.

The experimental crossflow-gated `WELLSCHUR 1` path became nondeterministic
enough to be unsuitable as the production benchmark: otherwise identical
full runs took 186.09 and 227.91 s and followed different timestep paths.
It remains available for development, but `decks/NORNE.DATA` explicitly uses
`WELLSCHUR 0`. The final CUDA build and the 442-check host regression suite,
including Norne-specific deck assertions, both pass.

### GPU regional controls and final speed lead (2026-09-09)

The remaining 55.90 s well-control cost was almost entirely an accepted-state
operation: copy all 44,431 primary-variable triples to the host and evaluate
the full PVT tables cell by cell to form the history-RESV regional average.
The connection assembler had already computed those same cell PVT values on
the GPU. The retained path now prepares that existing property cache from the
control routine, reduces the eight hydrocarbon/full-pore-volume sums with one
fixed-order partial per CUDA block, and lets the immediately following
assembly consume the prepared cache once. A Newton update explicitly
invalidates the handoff. Only the small partial array is copied to the host;
MPI still reduces the same eight regional quantities. Convergence-statistic
buffers and injector scratch storage now persist as well, avoiding repeated
allocator traffic without changing their arithmetic.

Two 424-day runs were identical byte for byte and took 16.44 and 16.47 s,
versus 24.69 s for the exact host-average path. The full production
configuration (`WELLSCHUR 0`, `LINTOL 1e-4`) then completed twice in 143.14 and
143.00 s with identical RSM output and identical work: 4,604 accepted
substeps, 28,358 line-search assemblies, 1,514 discarded Newton iterations,
24,156 discarded linear iterations, 1,427 AMG rebuilds, and 224,241 AMG
applications. Well-control refresh fell to 6.46--6.49 s. Conservatively using
the slower repeat, this is 52.28 s (26.8%) faster than the previous 195.42 s
NVRS repeat and 68.32 s (32.3%) faster than the local 211.46 s OPM Flow run.

Accuracy did not trade away the speedup. Against that OPM output, median
active-period NRMSE is 8.852% WOPR, 8.636% WWPR, 10.629% WGPR, 4.534% WWCT,
26.297% WGOR, and 41.754% WBHP. Median cumulative errors are -6.33% oil,
+1.93% water, and -7.60% gas. All six NRMSE values improve slightly from the
immediately preceding 195 s NVRS path.

The formerly nondeterministic Schur reduction was repaired separately. One
block now owns each well, finds its contiguous completion range, and performs
a fixed-tree reduction instead of atomically accumulating completion
pressures. Repeated 424-day `WELLSCHUR 1` runs were byte-identical at
15.55--15.56 s. The full run completed in 142.51 s and stayed in the accuracy
band, but needed 5,155 accepted substeps, 12% more than `WELLSCHUR 0`, for only
0.63 s wall-time benefit. It therefore remains a deterministic opt-in feature;
the production deck stays on the more conservative setting until the missing
saturation/composition well derivatives or explicit well unknowns make the
coupling complete.

The solver sweep did not find another production win. AmgX strength thresholds
0.10, 0.25 and 0.75, two smoother sweeps, aggregation coarsening, FGMRES
restart 40/120, and BiCGSTAB were neutral or slower on the 424-day screen.
`LINTOL 2e-4` initially saved 2.91 s through day 1637, but on the accepted GPU
control path the full run took 143.45 s and expanded to 5,600 accepted
substeps, so `1e-4` remains the default. Moving completion PVT/mobility
evaluation itself to CUDA was also rejected: it did not reduce wall time and
needlessly changed the nonlinear path. The final CUDA build, all 442 host/deck
checks, standalone unit conversions, and endpoint-scaling derivative tests
pass.

### Failure-cap recovery and Flow PID (2026-09-09, continuation)

The exact Flow `PIDAndIterationCount` relative-change limiter is now available
as `DTPID 1` (`DTPIDTOL 0.1`).  Its state metric is evaluated by one
deterministic CUDA-block reduction over the accepted black-oil primary state,
and the controller retains Flow's three-error history and 0.075/0.175/0.01
PID exponents.  On Norne through day 424 it was neutral: the iteration-count
candidate was always the smaller one, giving the same 693 substeps and 28,395
AMG applications as the non-PID run.  It therefore remains off in the
production deck; this result also confirms that the missing limiter was not
the source of NVRS's timestep gap.

The remembered failure ceiling was then swept.  Through day 424,
`DTCAPGROW` 1.0/1.1/1.2/1.5 took 15.71/15.62/16.44/17.80 s.  The respective
accepted-substep counts were 779/687/693/667 and discarded Newton counts were
132/190/242/370: faster recovery makes fewer accepted steps but pays for more
failed attempts.  On the complete 3,312-day history, 1.1 completed in
134.07 s versus 143.14 s at 1.2.  It used 5,272 accepted substeps, 13,643
accepted plus 938 discarded Newton iterations, 25,312 line-search assemblies,
and 210,908 AMG applications.  Thus it lowers total attempted Newton work from
15,745 to 14,581 and wall time by 6.3%, despite accepting more, cheaper
substeps.  Both canonical Norne decks now state `DTCAPGROW 1.1` explicitly.

Accuracy stayed in the established range.  Median active-period NRMSE is
9.058% WOPR, 8.676% WWPR, 10.652% WGPR, 4.526% WWCT, 26.823% WGOR, and
41.397% WBHP.  Median cumulative errors are -6.407% oil, +1.756% water, and
-7.581% gas.  This is a small rate-shape trade against the 143 s trajectory,
with slightly better water cumulative error and BHP, for the 9.1 s speedup.

Three more nonlinear experiments were rejected.  Raising `NEWTONMAX` from 20
to 40 reduced accepted substeps to 4,227 but increased runtime to 150.91 s.
Removing the failure ceiling reduced them further to 2,866 but caused 22,491
discarded Newton iterations and a 391.42 s run; resetting it every report gave
3,157 substeps but still took 178.78 s.  A full finite-difference producer BHP
derivative, lagged injector allocation, and the exact frozen-mobility injector
pressure Schur block all regressed their screens or full runs.  Injector-only
Schur plus the 1.1 cap took 146.81 s and 6,175 substeps; the implementation was
removed rather than retaining a partial well Jacobian that changes the path
adversely.

The remaining gap to Flow (355 timesteps and 1,382 Newton iterations) is
therefore structural, not controller arithmetic.  NVRS now averages only 2.59
accepted Newton iterations per accepted substep, better than Flow's 3.89, but
uses 5,272 substeps.  Closing that gap requires the complete coupled well
system and black-oil primary-variable switching (`Sg`/`Rs`/`Rv`) used by Flow;
partial eliminated-well derivatives consistently make late-life crossflow and
phase-appearance behavior worse.


### Accepted line-search assembly handoff (2026-09-10)

An accepted line-search trial already leaves the reservoir properties, residual,
and Jacobian assembled at exactly the state from which the next Newton
iteration starts.  The Newton loop now hands that assembly forward instead of
immediately repeating the well-control refresh and identical full assembly.
Fallback updates for which no descending trial was found are deliberately not
reused, because that path applies an unassembled clipped update.

This is a path-preserving work optimization.  On the 424-day screen it reduced
wall time from 15.48 to 13.93 s and reused 1,566 accepted-trial assemblies.
Accepted substeps (687), discarded Newton iterations (190), discarded linear
iterations (3,007), line-search trials (1,898), and AMG applications (27,163)
were unchanged, and all saved states plus RSM_FVM.csv were byte-identical.

On the complete 3,312-day Norne history, two runs took 120.10 and 120.91 s,
down from 134.07 s (9.8--10.4%).  Both retained exactly 5,272 accepted
substeps, 13,643 accepted plus 938 discarded Newton iterations, 25,312
line-search trials, and 210,908 AMG applications, while reusing 13,085
assemblies.  All 248 reservoir snapshots and the well CSV are byte-identical
to the validated 134.07 s trajectory; the CSV SHA-256 is
b59d0a35ca077c9923b43e0370e354bf5b8dfdbd5ab54c42290c498f62782173.
Consequently the previously reported OPM accuracy metrics are unchanged.
Against the local 211.46 s Flow run, the conservative 120.91 s repeat is
1.75x faster and uses 42.8% less wall time.

Two alternative line-search merit changes were screened and removed.  Keeping
the best evaluated trial as a no-descent fallback was neutral (15.51 versus
15.48 s and identical nonlinear work through day 424).  Replacing the maximum
scaled component norm with a scaled component 2-norm regressed to 726 accepted
substeps and 27,806 AMG applications (15.87 s).  Neither is retained.  All 442
host/deck regression checks pass after the handoff change.

### Flow-style convergence acceptance and retained well coupling (2026-09-10)

A failure audit showed that the 14,581 attempted Newton iterations were not
caused by slow convergence inside accepted steps: NVRS averaged 2.59 Newtons
per accepted substep.  The dominant cost was 5,272 very small substeps, with
material balance rejecting otherwise useful states in tiny phase-boundary
pore volumes.  A full-history sweep located the accuracy/conservation knee at
`MBTOL 1e-5` and `CNVRELAXED 5.0`; the latter applies only to the existing
lowest-three-percent pore-volume exemption.  `MBTOL 5e-5` was rejected after
oil defect tripled and water/gas well errors worsened.

With those limits, enabling the deterministic matrix-free `WELLSCHUR 1` is
now beneficial.  It analytically eliminates the common BHP of forward-flowing
rate-controlled producers and applies the resulting rank-one completion
coupling without enlarging the reservoir CSR.  Two exact retained-deck runs
followed the identical trajectory and reported 33.83 and 34.10 s, 979 accepted
substeps, 2,483 accepted plus 322 discarded Newton iterations, 4,962 line-
search trial assemblies, and 58,365 AMG applications.  The conservative repeat
is 6.20x faster than the local 211.46 s Flow run.  Total attempted Newton work
is now 2.03x Flow (2,805 versus 1,382), down from 10.55x (14,581).

The full-history progression is:

| Configuration | Time (s) | Accepted substeps | Accepted Newton | Discarded Newton | Total Newton |
|---|---:|---:|---:|---:|---:|
| OPM Flow | 211.46 | 355 | 1,382 | included | 1,382 |
| strict NVRS + assembly handoff | 120.91 | 5,272 | 13,643 | 938 | 14,581 |
| MB/CNV policy, local well Jacobian | 40.38 | 1,722 | 3,477 | 307 | 3,784 |
| retained MB/CNV + common-BHP Schur | 34.10 | 979 | 2,483 | 322 | 2,805 |

Against OPM, retained median active-period NRMSE is 8.952% WOPR, 8.522%
WWPR, 10.719% WGPR, 4.450% WWCT, 30.846% WGOR, and 40.860% WBHP.
Median cumulative errors are -6.138% oil, +1.425% water, and -7.661% gas.
The primary rate quantities therefore stay in the established accuracy band;
WGOR is the one visible tradeoff versus the strict path (26.823%).  Final
accepted residual defects are +5.212 million STB oil, -2.230 million STB
water, and -11.277 million MSCF gas.

Several tempting extensions were rejected rather than left half-coupled.
Residual-aware failure restart factors and failure ceilings outside 0.75
regressed work.  `DTCAPGROW` 1.05 improved the 424-day screen but regressed
the full run to 2,895 Newtons and 34.55 s; 1.1 remains retained.  Extending
the common-BHP Schur correction through reverse-flowing completions while
freezing wellbore composition regressed the 424-day total from 351 to 357:
the missing mixture derivative is numerically material.  A saturation floor
is not a substitute for primary switching; once correctly propagated it
changed initially oil-free cells and made the first step irrecoverable.

The remaining expensive retries cluster at completion-pressure reversals in
late-life producers, with the control active set and wellbore mixture changing
together.  Completing this last factor-of-two Newton gap requires an explicit
well/control residual (or an equivalent multi-rank Schur representation) with
BHP, saturation, PVT, mixture, and active-control derivatives, plus per-cell
black-oil primary-variable switching among `Sg`, `Rs`, and `Rv` when gas or
oil phases appear and disappear.  The current compact `[p, Sw, Sg]` state
cannot implement that safely by a local clamp or phase floor.

The retained repeats are byte-identical at both the well-history CSV
(`d5d94374f7e824bea9235f4a22795918dafea5903e8837bad0f2ed357657814d`)
and final state snapshot
(`d2fc8f58b03e1ebcb51d84f05fa0a19a372a367b9d745b9e44b262f0cb55f3f5`).
The final CUDA build, all 442 host/deck checks, the standalone unit-conversion
test, and all endpoint-scaling/Jacobian derivative checks pass.

### 2026-09-10: coupling derivatives cannot be added piecemeal

Two full-history experiments tested the next coupling layer and were rejected.
Extending the matrix-free producer Schur term from the established pressure-only
weight to pressure/Sw/Sg control gradients increased accepted substeps from
979 to 4,585 and runtime from 34.10 s to 57.01 s (310 discarded Newtons).
The added mobility derivative was inconsistent with the still-frozen completion
hydrostatic-density, wellbore-mixture, crossflow, and active-control terms.

Adding pressure-viscosity derivatives to reservoir face fluxes, producer wells,
crossflow, and pressure-limited injectors in isolation also regressed the full
run: 1,238 accepted substeps, 369 discarded Newtons, and 37.80 s.  Both source
experiments were fully reverted and the retained binary rebuilt.

This pins the architectural boundary: the next implementation must introduce
the well control equation and phase-state variable together, then form the full
Jacobian (or eliminate that exact block).  Incrementally enriching only one
side changes the Newton direction without differentiating the function that the
outer well refresh and phase clamp actually evaluate.

### Black-oil primary switching and adaptive-step recovery (2026-09-10)

The reservoir state now supports per-cell `Sg`, `Rs`, and `Rv` meanings for the
third primary variable, including rollback shadows and halo exchange.  The PVT,
accumulation, connection-flux, well-flux, reporting, and hysteresis paths use the
physical state selected by that meaning.  In particular, the retained Jacobian
includes the direct gas-mobility derivative with respect to `Rv`; omitting that
term made the full NORNE run stall before report 162.  Gas/oil phase-potential
derivatives, capillary conversion at `Sg -> Rv`, common Appleyard saturation
scaling, and the water-only fallback were made consistent with the switched
state.  Host well control and reporting also use oil pressure rather than gas
pressure in `Rv` cells.

The timestep failure ceiling is no longer pinned for the rest of a report
interval.  It relaxes after every successful substep by `DTCAPGROW 1.1`, which
lets the solver recover promptly after a difficult phase/control transition.
The retained `DTMAX 10` trajectory is reproducible:

| Configuration | Time (s) | Accepted substeps | Accepted Newton | Discarded Newton | Total Newton |
|---|---:|---:|---:|---:|---:|
| OPM Flow reference | 211.46 | 355 | 1,382 | included | 1,382 |
| previous NVRS Schur baseline | 34.10 | 979 | 2,483 | 322 | 2,805 |
| switched NVRS, retained | 41.66 | 595 | 2,477 | 513 | 2,990 |

The switched run is 5.08x faster than Flow.  It reduces NVRS accepted substeps
by 39.2% relative to the previous Schur baseline while keeping total attempted
Newton work within 6.6%.  Total Newton work remains 2.16x Flow, so this is not a
claim that the nonlinear algorithms are equivalent yet.

Against the same OPM summary, retained median active-period NRMSE is 9.195%
WOPR, 11.153% WWPR, 11.647% WGPR, 5.782% WWCT, 23.419% WGOR, and 36.346%
WBHP.  Relative to the previous Schur baseline, switching improves WGOR from
30.846% and WBHP from 40.860%; the primary rate metrics remain near the
established accuracy band.

Three full-history ablations were rejected.  `DTCAPGROW 1.05` became trapped
in small steps (87.46 s, 1,349 accepted substeps, 6,412 accepted plus 2,299
discarded Newton iterations).  `DTMAX 15` invited expensive failed large-step
attempts (65.88 s, 794 substeps, 4,525 accepted plus 1,637 discarded Newton
iterations); `DTMAX 31` had already shown the same failure mode.  Extending the
producer Schur correction to pressure/Sw/third-primary weights without the
matching hydrostatic-density, wellbore-mixture, crossflow, and control-active-set
derivatives also regressed to 617 substeps and 3,258 total Newton iterations.
These results reinforce the OPM-derived design boundary: the remaining gap
requires an exact well/control residual block and its complete derivatives,
not a more aggressive timestep knob or a partial Schur row.

### OPM-style cumulative connection pressure (2026-09-10)

The remaining BHP bias was traced against OPM's `StandardWellConnections`.
Flow computes the density of each wellbore segment from the component stream
flowing upward through that segment, then accumulates the hydrostatic pressure
drop from the WELSPECS datum through the ordered perforations.  NVRS instead
used each completion's local mobility mixture over the complete datum-to-cell
distance.  That approximation is especially poor when deeper and shallower
completions produce different water cuts or GORs.

The retained producer control now gathers the current perforation oil, water,
and gas component rates, accumulates them from the deepest connection upward,
inverts the coupled Rs/Rv component split, forms the flowing reservoir-volume
mixture density, and accumulates pressure drop segment by segment.  The column
is refreshed at the Newton base state and held fixed only across that Newton's
line-search trials.  Reservoir pressure, mobility, PVT and the common BHP are
still recomputed for every trial.  This removes trial-order dependence without
freezing the column for a complete substep.

| Configuration | Time (s) | Substeps | Accepted Newton | Discarded Newton | Total Newton | Median WBHP NRMSE |
|---|---:|---:|---:|---:|---:|---:|
| OPM Flow | 211.46 | 355 | 1,382 | included | 1,382 | reference |
| fastest switched NVRS | 41.66 | 595 | 2,477 | 513 | 2,990 | 36.346% |
| retained cumulative-column NVRS | 44.91 | 642 | 2,786 | 578 | 3,364 | 32.438% |

The accuracy-oriented run remains 4.70x faster than Flow.  Its median active-
period errors are 8.887% WOPR, 11.433% WWPR, 11.559% WGPR, 5.410% WWCT,
24.366% WGOR and 32.438% WBHP.  Relative to the fastest switched run, median
WBHP error falls 10.8% and WOPR improves slightly; the additional well physics
costs 7.8% wall time and 12.5% attempted Newton work.

Four variants were rejected.  Refreshing the cumulative density on every
line-search trial improved WBHP slightly further to 31.957%, but regressed to
47.44 s and 3,689 total Newtons because the lagged well state made successive
trials path-dependent.  Freezing it for an entire accepted reservoir state
regressed WBHP to 37.514% and runtime to 50.36 s.  Three internal BHP/density
fixed-point sweeps gave 36.734% WBHP and 44.99 s.  Extending the Schur input to
all three reservoir primaries, even with numerical differentiation of the exact
host control function including hydrostatic density, regressed to 90.82 s and
8,356 total Newtons because the reservoir completion residual still freezes
wellbore-mixture and active-set derivatives.  `DTCAPGROW 1.2` also regressed
the retained physical model to 61.72 s and 5,628 total Newtons.

The remaining BHP error is therefore not explained by metric conversion,
COMPDAT connection factors, shared completion cells, field-wide history RESV
conversion, or THP/VFP control of the worst wells; each was checked directly
against the NORNE deck and current OPM source.  It reflects the remaining
reservoir-pressure and fully coupled well-equation difference.  Closing it
without giving back nonlinear robustness still requires adding explicit BHP
and well component-rate unknowns to the Newton system, rather than eliminating
an incompletely differentiated host control equation.

### BHP/report separation and nonlinear follow-up (2026-09-10)

The cumulative connection column is now evaluated without forcing the
reservoir solve onto its less robust nonlinear trajectory.  NVRS retains the
validated local-density control and completion pressures, while reported
producer BHP receives the difference between that column and OPM's cumulative
flowing-mixture column.  Reservoir states, rates, substeps, and Newton work
therefore remain exactly on the fast switched-primary path.  The final run
took 41.72 s with 595 accepted substeps, 2,477 accepted and 513 discarded
Newton iterations (2,990 total).  Median WBHP NRMSE changes only from 36.346%
to 36.297%; this small movement confirms that datum conversion is not the
dominant error.  Initial mean pressure differs from Flow by 0.53 psi, but the
bias grows after roughly day 1,000 and reaches about +398 psi at the final
comparable restart.

Additional full-history experiments were rejected and removed.  Guarded
12--15 day easy-region caps increased Newton work.  A 0.5 cumulative-column
control blend produced 805 substeps and 4,787 total Newton iterations.  Adding
numerical pressure/Sw/Sg-Rs-Rv `B` columns to `-C D^-1 B` only for all-forward
RESV wells reduced final pressure bias modestly but regressed to 661 substeps
and 3,624 total Newton iterations.  `DSMAX 0.20`, `DPMAX 2000`, and
`DTFAILCAP 0.60` also increased total Newton work; the last reached 7,273.

These results agree with the OPM source structure: Flow eliminates a complete
well residual block with matching `B`, `C`, and `D` derivatives.  Isolated
reservoir-primary columns are inconsistent while wellbore mixture, cumulative
density, crossflow active sets, and control switching remain outside that
block.  The full host/deck regression suite passes after restoration,
including all 41 well-control checks.
### Coupled-well audit, timestep PID, and shut-well BHP state (2026-09-10)

The local OPM Flow source was used as the reference, principally
`StandardWell_impl.hpp`, `StandardWellAssemble.cpp`,
`StandardWellEquations.cpp`, `StandardWellPrimaryVariables.*`,
`StandardWellConnections.cpp`, and `RateConverter.cpp`.  Flow carries four
black-oil well primary variables (`WQTotal`, water fraction, gas fraction,
and BHP), assembles three component conservation equations plus one control
equation, and eliminates the dense local block as `A - C^T D^-1 B`.  NVRS's
current pressure-only well Schur term is therefore the exact frozen-mobility
reduction only for the restricted forward-flow RESV case; it is not a
replacement for Flow's complete block when phase fractions, crossflow, or
control switching move.

Two source-aligned nonlinear experiments were rejected after full-history
tests.  Adding all PVT viscosity-pressure derivatives that Flow gets through
automatic differentiation increased runtime from 41.67 to 47.13 s, accepted
substeps from 595 to 608, and discarded Newton work from 513 to 657; the
exact-time pressure bias also worsened.  The change was reverted.  Enabling
NVRS's existing relative-change PID at `DTMAX 10` reduced retries (21 to 17)
and substeps (595 to 586), but raised accepted Newton work from 2,477 to
2,598 and runtime to 42.56 s.  Letting that PID use `DTMAX 20` was worse:
600 substeps, 3,123 accepted plus 879 discarded Newton iterations, and
51.04 s.  Both tuning changes were reverted.  This is consistent with the
earlier larger-step screens: controller tuning cannot compensate for an
incomplete nonlinear well/control block.

One reporting fix is retained.  Flow preserves a well's last solved BHP while
the well is shut, whereas NVRS recomputed a shut row from zero or a fallback
completion pressure.  `field_well_rates()` now retains the last finite active
BHP per `FieldModel`, only for output; it never feeds the well controls or the
reservoir residual.  The final F-3H endpoint discrepancy, for example, changes
from -6,526.70 psi (-100%) to -55.44 psi (-0.849%), and the vertical BHP
spikes at shut dates disappear.  Active-well BHP differences remain physical
model differences rather than being hidden by this reporting state.

The final validated trajectory is unchanged: 41.67 s, 595 accepted substeps,
2,477 accepted plus 513 discarded Newton iterations (2,990 total), and 6,402
line-search trials.  Endpoint field errors remain FOPR +9.759%, FWPR -10.281%,
FGPR +3.932%, and FVPR -1.487%; cumulative FOPT/FWPT/FVPT errors are
-0.165%/-5.494%/-1.030%.  All 41 well-control checks and the complete deck
suite pass.

### Shut-rate audit and OPM nonlinear ablations (2026-09-10)

The shut-in state audit was extended from BHP to all rate and cumulative
vectors.  Both the ECL summary writer and the accepted-substep material
ledgers already select producer/injector rates only when `WellRateRow::type`
is negative/positive; a shut row (`type == 0`) cannot enter FOPR, FWPR, FGPR,
FVPR, injection totals, or their cumulative integrals.  Thus the prior stale
completion-flag risk did not contaminate field metrics.  The retained source
now also requires a non-shut row before a completion is considered live, so
WOPR/WWPR/WGPR/WWIR/WGIR are zero at their origin while the last solved WBHP
is preserved, matching Flow's output semantics.

The local OPM 2024.10 material-law source confirms that a three-phase deck
without STONE1/STONE2 selects `EclMultiplexerApproach::Default`.  NVRS already
matches its oil-relative-permeability weighted interpolation, its 1e-5
small-denominator regularisation, and the NORNE Carlson non-wetting hysteresis
coordinate.  All 504 NORNE completions use the deck's COMPDAT connection
factor, so no Peaceman/NTG reconstruction difference can explain active BHP.

Several further full-history experiments were rejected and removed:

| Experiment | Time (s) | Substeps | Accepted Newton | Discarded Newton | Total |
|---|---:|---:|---:|---:|---:|
| retained Schur baseline + shut-rate invariant | 41.60 | 595 | 2,477 | 513 | 2,990 |
| well pressure-viscosity and BHP-cap derivatives | 47.58 | 655 | 2,894 | 625 | 3,519 |
| disable pressure-only well Schur | 43.62 | 605 | 2,851 | 546 | 3,397 |
| faster failed-step ceiling recovery (`DTCAPGROW 1.5`) | 44.50 | 579 | 2,708 | 651 | 3,359 |
| OPM residual-oscillation damping | 48.85 | 688 | 3,214 | 787 | 4,001 |

The Schur ablation proves the restricted rank-one coupling is useful; removing
it adds 374 accepted Newton iterations.  Conversely, adding isolated AD-like
viscosity/control derivatives or Flow's oscillation damping to the reduced
system makes it less consistent and increases work.  Faster timestep recovery
does reduce accepted substeps, but the larger failed attempts cost more than
it saves.

Flow's PRT contains 355 accepted timesteps with 354 detailed Newton records:
1,361 recorded iterations (3.84 per step), median timestep 7 days, and 122
steps longer than NVRS's 10-day cap (37 longer than 20 days).  NVRS averages
4.16 accepted Newton iterations per substep, close to Flow's 3.84; most of the
2.16x total-work gap is therefore the 595 versus 355 timestep count plus 513
discarded iterations, not a twofold per-step iteration penalty.  Taking
Flow-sized steps safely still requires its consistent four-variable well
block and black-oil primary-variable system; the ablations show that timestep
aggression or isolated derivatives cannot substitute for that structure.

### Three-mode eliminated-well and controller boundary tests (2026-09-10)

The matrix-free well correction was generalized from one eliminated BHP mode
to three modes (BHP plus two independent wellbore reservoir-volume fractions).
Production `WELLSCHUR 1` now executes only its original single mode; experimental
modes can exercise the wider operator without changing the validated path.
Compatibility was exact: 595 accepted substeps, 2,477 accepted plus 513
discarded Newton iterations, and 6,402 line-search assemblies. A final run
took 41.88 s, within timing noise of the 41.60--41.72 s baseline.

The experimental host elimination differentiates the same local PVT,
relative-permeability, primary-variable switch and hydrostatic functions used
to form each completion coefficient. The complete control-pressure variant
(`dA/du` and `dtheta/du`) regressed to 715 substeps, 3,457 accepted plus 959
discarded Newton iterations, and 57.69 s. Adding the two crossflow-mixture
modes regressed further to 737 substeps, 3,699 accepted plus 1,107 discarded
Newton iterations, and 60.88 s. Freezing target mobility while retaining
datum-pressure sensitivity was worst: 840 substeps, 4,165 accepted plus 1,319
discarded Newton iterations, and 65.92 s. These controlled tests locate the
problem in nonsmooth phase/crossflow active-set sensitivity, rather than in
the low-rank CUDA contraction. The robust frozen-mobility BHP mode remains
the production choice.

Controller boundary tests were likewise rejected:

| Experiment | Time (s) | Substeps | Accepted Newton | Discarded Newton | Total |
|---|---:|---:|---:|---:|---:|
| retained configuration | 41.72 | 595 | 2,477 | 513 | 2,990 |
| `DTMAX 20` | 55.29 | 700 | 3,560 | 1,074 | 4,634 |
| `DTMAX 8` | 64.81 | 938 | 4,730 | 1,612 | 6,342 |
| `NEWTONMAX 15` | 49.05 | 703 | 3,182 | 858 | 4,040 |
| `NEWTONMAX 25` | 59.84 | 741 | 4,084 | 1,219 | 5,303 |
| Flow-style equal report remainder | 44.41 | 659 | 2,853 | 649 | 3,502 |
| absorb report tails below 25% | 57.04 | 770 | 3,943 | 1,140 | 5,083 |

`DTTARGET 4` was terminated by the 110 s guard at day 3,009 after collapsing
to sub-day steps (one report interval required 92 substeps). All settings
were restored. The non-monotonic results show that the current trajectory is
a sharp local optimum: both larger and smaller nominal steps, and both shorter
and longer nonlinear attempts, move the simulation onto more expensive phase-
transition paths.

The shut-rate audit was checked in the final generated summary as well as in
source. Oil and gas had zero leakage at inactive samples. Three apparent water
exceptions were F-1H/F-3H/F-4H producer-to-water-injector transitions, where
Flow and NVRS place the schedule change on adjacent summary dates; they are not
shut-well rates. At the source, `row.type != 0` gates every completion before
WOPR, WWPR, WGPR, WWIR or WGIR is formed, and all field/cumulative consumers
independently filter the row type.

Finally, the active-well BHP discrepancy is dominated by reservoir state, not
the shut-in BHP retention or datum conversion: across both-active samples the
BHP bias is +284 psi (343 psi MAE, 472 psi RMSE), while the exact common
day-3260 cell-pressure field is already +433 psi on average. The earlier
+399 psi figure compared different report endpoints and is not retained. The ROCK
reference pressure, compressibility, absolute pore-volume multiplier and its
Jacobian derivative were audited and are present. Endpoint field errors on
the restored run remain FOPR +9.759%, FWPR -10.281%, FGPR +3.932%, FVPR
-1.487%; cumulative FOPT/FWPT/FVPT remain -0.165%/-5.494%/-1.030%. The full
deck suite passes: 81 grid, 111 schedule, 65 equilibrium, 110 partition, 34
field-unit and 41 well-control checks, all with zero failures.

The restart writer was also corrected to export Eclipse `PRESSURE` semantics:
an Rv-primary cell stores gas pressure internally, so output now subtracts
`Pcog` and writes oil pressure. This is standards-correct for decks with
nonzero gas-oil capillary pressure, but deliberately leaves NORNE unchanged
because every SGOF `Pcog` value in this deck is zero. The final post-fix run
therefore retains the exact 595/2,990 trajectory and 41.71 s runtime, and the
+433 psi exact-common-time cell-pressure bias remains a reservoir-model
difference.

### Flow-consistent PINCH, precision audit, and nonlinear boundary screens (2026-09-10)

The corner-point connection graph was compared pair by pair with Flow's EGRID
and INIT transmissibilities. Before PINCH support, all 124,297 NVRS pairs were
present in Flow but 7,854 Flow pairs were missing, dominated by collapsed-layer
vertical bridges. CpGrid now parses the PINCH thickness and GAP/NOGAP policy
and crosses only geometrically thin MINPV/ACTNUM cells. The retained NORNE grid
has 125,447 connections (37,611 I, 42,641 J, 36,892 K, 8,303 NNC). Every NVRS
pair is present in Flow and the retained total transmissibility differs by only
-0.003225%; the remaining Flow-only tail has negligible strength. A synthetic
ACTNUM-hole/PINCH regression was added.

With the correct topology, CNVRELAXED 10 is the measured convergence knee:

| Solver | Time (s) | Accepted substeps | Accepted Newton | Discarded Newton | Total Newton |
|---|---:|---:|---:|---:|---:|
| OPM Flow reference | 211.46 | 355 | 1,382 | included | 1,382 |
| retained NVRS, PINCH + CNV 10 | 37.18--37.27 | 604 | 2,235 | 423 | 2,658 |
| PINCH + CNV 8 | 39.54 | 643 | 2,460 | 543 | 3,003 |
| PINCH + CNV 5 | 46.62 | 643 | 2,934 | 722 | 3,656 |
| PINCH + CNV 15 | 48.99 | 753 | 3,335 | 976 | 4,311 |

A final local screen confirmed the sharp minimum: CNV 11 took 36.34 s but
slightly increased total Newton work to 2,683 (2,294 accepted plus 389
discarded), while CNV 10.5 jumped to 3,975 total and 46.23 s. CNV 10 is
retained because Newton work and accuracy, not a sub-second timing change,
are the selection criteria.

NVRS is 5.67x faster than Flow, but still uses 1.70x as many accepted
substeps and 1.92x as much attempted Newton work. Endpoint errors are FOPR
+6.457%, FWPR -15.359%, FGPR +2.202%, and FVPR -5.630%; cumulative
FOPT/FWPT/FVPT are +0.030%/-4.464%/-1.055%. At exact common day 3260,
cell-pressure bias/MAE/RMSE/p95 are +479.3/560.2/579.9/698.4 psi. Active-well
BHP bias/MAE/RMSE/p95 are +248.7/374.8/486.2/851.1 psi. Thus PINCH corrects
the topology and slightly improves mean BHP bias, but does not explain the
remaining pressure or nonlinear gap.

A genuine all-FP64 binary was built against the FP64 HYPRE installation, not
merely requested through the deck. It took 48.81 s, 700 substeps, and
2,992 accepted plus 845 discarded Newton iterations (3,837 total), versus
37.27 s and 2,658 total for the mixed FP32-preconditioner/FP64-Krylov path.
The mixed-precision path is therefore not causing the convergence gap; it is
both faster and nonlinearly more robust on this problem.

Flow's INIT PORV sum is 673,148,288 m3, which converts to NVRS's
4.23398e9 reservoir barrels. Pore-volume magnitude and metric/field conversion
are therefore also excluded as the source of the pressure drift.

Several Flow-inspired or derivative experiments were screened and removed:

| Experiment | Time (s) | Substeps | Accepted Newton | Discarded Newton | Total |
|---|---:|---:|---:|---:|---:|
| Flow CNV two-cycle update stabilization | 40.23 | 639 | 2,493 | 655 | 3,148 |
| primary-switch hysteresis 1e-3 | 43.15 | 617 | 2,680 | 607 | 3,287 |
| immediate primary switching | 75.71 | 1,093 | 6,064 | 2,264 | 8,328 |
| legacy formulas on all-Sg faces | 88.01 | 1,106 | 6,233 | 2,723 | 8,956 |
| Flow-style average-B MB only | 40.75 | 617 | 2,451 | 573 | 3,024 |
| Flow-style average-B MB and CNV | 60.34 | 804 | 3,957 | 1,453 | 5,410 |

An environment-only directional finite-difference audit was temporarily added
because the regular --jac-check numerical kernel cannot represent PINCH/NNC
graphs. At the difficult step-6 control transition, smooth-interior agreement
was excellent for water-Sw and Rv columns; relative L2 discrepancies were
0.4%/3.5% for oil/gas under Sw, 1.0%/8.5% under Sg, 5.3%/1.5% under Rs, and
about 3% under pressure. Including phase endpoints made the Sg comparison
appear nearly 100% inconsistent because a mixed-sign perturbation samples
opposite one-sided derivatives at Sg=0 and So=0. The large signal is therefore
localized to nonsmooth phase boundaries, exactly where the line search reports
277 updates with no descending trial, rather than to ordinary floating-point
error or a missing smooth-state reservoir column. The temporary diagnostic and
all regressive branch experiments were removed.

NORNE's WELSPECS defaults permit crossflow. WELLXFLOW 2 now keeps the
common-wellbore mixture on reverse completion flow for every control type,
instead of mode 1's known-wrong local-cell composition. On NORNE it is
trajectory-neutral--37.18 s and exactly the retained 604/2,235+423 counts--
because non-RESV controls do not reverse-flow on this solution, but it makes
the configured semantics explicit without an accuracy or speed cost.

The final restored repeat took 37.17 s with the same 604 substeps,
2,235 accepted and 423 discarded Newton iterations.

The shut-in source audit remains clean: oil and gas have zero inactive-sample
rate leakage, and the three apparent water cases are adjacent-date
producer-to-injector schedule transitions. They do not contaminate field oil,
water, gas, voidage, injection, or cumulative metrics.

Final validation reports 83 grid, 111 schedule, 65 equilibrium, 110 partition,
34 field-unit, and 41 well-control checks, all with zero failures. The
remaining reduction toward Flow's 1,382 Newton iterations requires the exact
coupled well/control block and a semismooth treatment of phase/control active
sets; precision, topology, timestep tuning, and partial eliminated derivatives
have now been directly excluded.

### Final well-allocation and switch-policy screens (2026-09-10)

The history-RESV converter was checked once more against OPM's current
`RateConverter`. Both implementations use one field region for this Flow run,
hydrocarbon-pore-volume averages of pressure/Rs/Rv, and the same coupled
oil/gas component inversion. OPM writes the coefficients in inverse-FVF
notation while NVRS holds formation-volume factors locally; after substituting
`B=1/b`, the expressions are identical. This path is not the pressure-bias
source.

A frozen-mobility injector common-BHP elimination was implemented as a
controlled experiment. It differentiated completion redistribution
`q_k=A_k(pwf-theta_k)` with the matching local `+A_k dtheta_k` and rank-one
`-A_k dpwf` terms. The run did not finish within 110 s and was already trapped
in severe late-life stalls near report step 95. As with the producer modes
above, a partial low-rank well derivative is counterproductive without the
complete mobility, capillary, mixture and active-control block and a
preconditioner for it. The experiment was removed.

Flow's nominal nonlinear update limits were also isolated. Raising only DPMAX
from 1000 to 1500 psi failed the 110 s guard near report step 92; raising DSMAX
from 0.15 to 0.20 at the same time failed near step 94. The retained tighter
limits therefore protect, rather than cause, the difficult gas-transition
trajectory.

Finally, Flow's exact one-iteration primary-switch oscillation memory was
reproduced per cell and tested over the full history. It removed the tiny
`So`-negative output warnings left inside the 1e-5 switching band, but
regressed to 43.22 s, 687 accepted substeps, 2,890 accepted and 776 discarded
Newton iterations (3,666 total). The always-on 1e-5 guard was restored. Its
final clean repeat took 37.12 s and exactly reproduced the retained 604
substeps, 2,235 accepted plus 423 discarded Newton iterations, 5,414 line
trials and 277 no-descent updates.

### Well-aware CPR and physical connection-column isolation (2026-09-10)

The three-mode eliminated-well operator was absent from both CPR stages even
when it was present in the outer Krylov operator. A block-Jacobi Woodbury
correction now forms each well's small `I + V^T D^-1 U` system, solves its at
most 3-by-3 dense block on device, and applies the corresponding low-rank
correction after CPR stage 2. It is deliberately enabled only for experimental
multi-mode systems: applying it to the retained rank-one system changed the
inexact-Newton trajectory catastrophically despite reducing average Krylov
work (96.45 s, 1,265 substeps, 7,505 accepted plus 2,922 discarded Newtons).
The production `WELLSCHUR 1` path remains unchanged.

The preconditioner does make the otherwise poor complete-derivative
experiments materially less expensive:

| Well block | Time (s) | Substeps | Accepted Newton | Discarded Newton | Total |
|---|---:|---:|---:|---:|---:|
| retained rank-one, no new correction | 37.12 | 604 | 2,235 | 423 | 2,658 |
| complete pressure/control derivative, Woodbury | 47.42 | 674 | 2,708 | 614 | 3,322 |
| pressure plus two wellbore-mixture modes, Woodbury | 53.28 | 759 | 3,187 | 1,020 | 4,207 |

For comparison, the same pressure/control and three-mode blocks without the
well-aware preconditioner previously needed 4,416 and 4,806 total Newton
iterations. A half-strength Woodbury correction was rejected: it regressed to
57.14 s, 786 substeps and 1,117 discarded Newtons. This is an inexact-Newton
trajectory effect, not a linear-iteration monotonicity problem.

The installed OPM source was then checked directly. `StandardWellConnections`
forms each segment density from cumulative component outflow from all deeper
perforations, accumulates the pressure drop in topological completion order,
and stores those drops as explicit scalar quantities rather than AD
evaluations. NVRS already computed this exact style of column, but used it only
to translate the reported datum BHP while retaining a local-mobility column in
the control equation and reservoir source. The OPM-style column is now
available consistently to experimental complete well modes, with its pressure
drop frozen within each linearization and line-search trial.

That change improved the complete pressure/control run from 47.42 to 46.66 s,
674 to 633 substeps, and 614 to 582 discarded Newtons. Its pooled active BHP
bias/MAE/RMSE were +233.7/356.2/479.4 psi, versus
+274.0/364.8/496.3 psi for the retained run. However, median water-rate NRMSE
worsened from 10.82% to 11.84%, and the complete block remained slower and
more nonlinear work than production. Isolating the cumulative column with the
rank-one block produced 37.40 s, the same 604 substeps, 2,284 accepted plus 431
discarded Newtons, and slightly worse median WGOR and WBHP NRMSE. It was
therefore not promoted to the production trajectory.

The pressure/BHP discrepancy is predominantly a reservoir-state discrepancy,
not shut-in reporting, PINCH topology, or floating-point roundoff. Flow and
NVRS initial restart pressures already agree after unit conversion, while the
late exact-time cell-pressure comparison develops a roughly +479 psi NVRS
bias. Active producer BHP errors are correspondingly mostly positive and range
from roughly +300 to +1,000 psi for the largest contributors. Full FP64 was
already slower and used more Newton work than the retained FP32-preconditioner
/ FP64-Krylov solve, so mixed precision is excluded as the cause. Correct
Flow-style PINCH transmissibilities remain enabled and covered by the grid
tests.

### Pressure-compartment and effective-storage audit (2026-09-10)

The late pressure bias was decomposed over every exact matching OPM/NVRS
restart time and over EQLNUM, FIPNUM, FLUXNUM, saturation, and pore-volume
weights. Initial pressure agrees to 0.53 psi in the mean (2.05 psi RMS), so
equilibration is not the source. The discrepancy changes sign by disconnected
pressure component at day 3260: EQL regions 1--3 are about 238--354 psi below
Flow, while connected regions 4--5 are about 564--621 psi above Flow. Within a
region the error is nearly a uniform pressure offset. Region 4 has no active
well and negligible saturation discrepancy, yet follows region 5 through 220
connections with total transmissibility 209.93 and no threshold pressure.

The grid topology explains the partition cleanly. The only cross-EQL pairs are
1--2 (39 connections), 1--3 (28), and 4--5 (220); NORNE's explicit THPRES
records concern 1--2 and 1--3, while 4--5 is open. No well spans the two
components. The small component's three producers cumulatively withdraw more
oil/water in NVRS while its F-4H water injection is 0.60% lower, consistent
with its extra depletion. The main component carries the compensating field
rate error. Thus the BHP bias is primarily the integrated phase/control
allocation and reservoir state, not a missing cross-region connection or a
perforation pressure datum error.

ROCK and storage were checked again. Accumulation uses the deck's final PORV,
including NTG, MULTPV, and MINPV; ROCK is converted from 4.84e-5/bar to
3.337e-6/psi and uses OPM's exact quadratic `1+x+x^2/2` law about 277 bar.
Rock and fluid terms are not double counted. Gas-control units are also
consistent end-to-end: schedule rates are SCF/day, Rs is SCF/STB, Bg is
SCF/RB, and only summary output converts to MSCF/day.

One remaining PVT convention was isolated. OPM's LiveOilPvt interpolates
inverse Bo inside each undersaturated pressure branch, whereas NVRS presently
interpolates deck Bo there before the already-correct LeftExtreme Rs blend.
The exact OPM version passed 88 focused PVT checks but strongly regressed the
full field: 75.99 s, 1,050 substeps, 5,249 accepted plus 2,264 discarded
Newton iterations, and 1,219 no-descent updates. It did not improve pressure:
late mean bias changed only from +479.3 to +470.1 psi, RMS worsened from 579.9
to 594.1 psi, and the small-component negative offsets grew substantially.
It was therefore reverted; this small table interpolation difference is not
the effective-storage or BHP root cause and destabilizes the current reduced
well/phase system when changed in isolation.

### Zero-rate well isolation and Newton-cap rescreen (2026-09-10)

F-4H's apparent 1.66 MMSTB cumulative water production was not merely a
shut-in summary defect.  Its all-zero WCONHIST record selected ORAT while all
three completions were water-filled.  The producer control loop tested the
controlled oil productivity before testing the zero target; because oil
mobility was zero it returned early and left the default 500-psi producer
flags live.  The reservoir equations and summary therefore both saw
38--53 MSTB/day of water production before F-4H converted to injection.
OPM's stoppedOrZeroRateTarget() instead solves such a well stopped and
isolated, independent of the controlled phase mobility.

Zero targets are now handled before the productivity test.  Every completion
is removed from the reservoir source, all oil/water/gas production rates are
zero, and an average hydrostatically translated connection pressure is
retained for BHP reporting.  Direct comparison confirms Flow reports zero for
all three F-4H producer rates throughout this interval.  The correction also
improves the small pressure component: at day 3260 EQL region 2 changes from
-354 psi relative to Flow to -138 psi in the retained trajectory.

Removing the artificial sink exposed harder genuine well/phase transitions,
so NEWTONMAX was rescreened.  The corrected results were:

| Newton cap | Time (s) | Substeps | Accepted | Discarded | Total | Chops |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 41.29 | 674 | 2,648 | 758 | 3,406 | 52 |
| 30 | 40.25 | 639 | 2,483 | 586 | 3,069 | 35 |
| 40 | 41.61 | 639 | 2,585 | 688 | 3,273 | 35 |

The 30-iteration cap is retained.  Relative to cap 20 it also improves the
day-3260 pressure mean/RMS error from +503/661 psi to +460/543 psi.  Median
active NRMSE remains comparable at 9.58% WOPR, 11.64% WWPR, 11.31% WGPR,
24.53% WGOR and 34.22% WBHP.  The remaining large late-life Newton cluster is
centered on D-4AH's RESV transition and gas-primary switching, not F-4H
reporting or floating-point precision.

Mixed precision was separately excluded as the cause: the earlier full
operator/preconditioner-precision control reproduced the retained mixed run's
604 substeps and 2,235+423 Newton counts exactly (37.28 versus 37.26 s).
Forcing the alternative all-FP64 path instead worsened to 700 substeps and
2,526+845 Newtons.  The nonlinear gap is controlled by model/Jacobian and
active-set behavior, not FP32 roundoff in the CPR preconditioner.

The late D-4AH audit then found that rank-one common-BHP coupling was disabled
for an entire well whenever any completion crossflowed.  Although the
wellbore-mixture derivative is not yet in mode 1, the common-pressure
derivative remains the exact frozen-mobility derivative of the rate equation.
Retaining that pressure column through crossflow reduced the corrected run
from 40.18 to 34.59 s, 639 to 549 substeps, and 2,483+586 to 1,968+271 Newton
iterations.  Timestep chops fell from 35 to 13 and no-descent updates from 308
to 179.  Rechecking a 40-iteration cap gave the same substeps/chops but
2,028+331 Newtons and 35.80 s, so the cap remains 30.
Rechecking 20 likewise regressed to 571 substeps, 2,142+344 Newtons, 21 chops
and 36.21 s.

Accuracy was retained or improved relative to the pre-correction production
trajectory: median active NRMSE is 9.50% WOPR, 11.60% WWPR, 11.08% WGPR,
23.94% WGOR and 37.06% WBHP; pooled BHP bias/MAE/RMS/p95 are
+301.5/344.7/472.1/831.5 psi.  Day-3260 pressure mean/RMS error is
+470.5/545.8 psi, and the previously worst small-component offsets improve to
-146, -96 and -142 psi in EQL regions 1--3.  F-4H remains zero in every
producer phase-rate column before its injector conversion.

### Complete wellbore-mixture block and convergence screens (2026-09-11)

The reverse-flow branch now follows StandardWell::computePerfRate's physical
variables: wellbore fractions are surface-component fractions, gas is put on
a common STB-equivalent surface-volume basis while forming the mixture, and a
receiving completion converts the mixture to reservoir volume with its local
B, Rs, and Rv.  A damped local component balance includes material removed by
reverse completions.  The cumulative wellbore-density calculation now consumes
those same reverse-flow component rates instead of the receiving cell's local
phase mixture.

WELLSCHUR 6 eliminates BHP, oil fraction, and water fraction with a full 3 by
3 well block.  Its D matrix contains the active control and two component
balances.  Each completion contributes the local Cw terms for PVT, mobility,
capillary pressure, and the reverse-flow volume ratio.  The reservoir-side B
columns contain BHP and both mixture derivatives; the matrix-free operator and
Woodbury correction carry all three modes.  ORAT, LRAT, GRAT, and WRAT now
solve BHP against the net surface stream and use that same control row in the
Jacobian.  This last correction is trajectory-neutral on NORNE because its
nonzero history producers are overwhelmingly RESV, but removes a real
inconsistency for other schedules.

| Configuration | Time (s) | Substeps | Accepted Newton | Discarded Newton | Total | WGOR NRMSE | WBHP NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPM Flow | 211.46 | 355 | 1,382 | included | 1,382 | reference | reference |
| exact-on-any-crossflow hybrid (7), retained | 42.4 | 581 | 2,268 | 371 | 2,639 | 23.93% | 34.19% |
| exact block plus corrected density feed | 43.34 | 587 | 2,335 | 351 | 2,686 | 26.24% | 34.82% |
| frozen pressure plus mixture modes (5) | 38.85 | 562 | 2,054 | 332 | 2,386 | 23.27% | 37.65% |
| corrected residual plus rank one (1) | 41.88 | 596 | 2,516 | 596 | 3,112 | 23.31% | 31.86% |

Mode 7 is the balanced retained choice.  It keeps the exact 3 by 3 block when
reverse flow makes the wellbore fractions active, but uses the more robust
frozen-mobility pressure column on an all-forward well, where the reservoir
has no mixture-side B column.  Relative to always-exact mode 6 it reduces
substeps, accepted Newtons, runtime, WGOR error, WBHP error, and the day-3260
pressure mean/RMSE error; the latter changes from +453/+555 to +443/+532 psi.
Median active NRMSE is 9.49% WOPR, 12.70% WWPR, 11.89% WGPR, 23.93% WGOR,
and 34.19% WBHP.  The water and gas rate metrics are respectively 1.45 and
0.68 percentage points worse than mode 6, which is the retained tradeoff for
the better pressure, BHP, GOR, and nonlinear work.

The hybrid averages 3.90 accepted Newton iterations per accepted substep,
essentially Flow's 3.89.  Its remaining 1.91x total-Newton gap is therefore
not an inefficient Newton solve within successful steps: it is 1.64x as many
accepted timesteps plus 371 iterations from retries.  It is nevertheless
4.99x faster in wall time than Flow on this GPU.

The following apparently reasonable changes were rejected after complete
full-history runs: OPM's segment Rs/Rv mixture limiting (61.94 s and 5,032
total Newtons), one-sided Appleyard damping of newly appeared gas (more than
4,000 total Newtons), checking the state after the final Newton update (59.99
s and 4,666 total), LINTOL 1e-5 (61.15 s and 3,977 total), a frozen-mobility
semismooth control row (47.56 s and 3,530 total), and Flow-style
final-iteration-only MB relaxation (56.64 s and 4,086 total).  Global MBTOL
screens at 1.5e-5 and 2e-5 reduced work, but their trajectories worsened WGOR
or WBHP enough that strict 1e-5 remains retained.  These tests reinforce that
precision, PINCH topology, tolerance shortcuts, and isolated active-set
clipping are not the remaining factor-of-two Newton cause.

Preserving a large controller proposal across a shortened report-boundary
fragment was also tested after the hybrid.  Although it appeared capable of
removing administrative small steps, the full history proved that the small
boundary solve was useful stability feedback: the change regressed to 87.38
s, 858 accepted substeps, 5,571 accepted plus 2,346 discarded Newtons, and
1,120 no-descent updates.  It was fully reverted.

Trace-crossflow thresholds were screened as a possible way to suppress
active-set chatter.  A 0.1% threshold initially looked promising at 40.57 s,
558 substeps, 2,146+413 Newtons and 33.44% WBHP, but an identical clean repeat
followed a different active-set trajectory: 43.72 s, 567 substeps,
2,398+449 Newtons, and a different late pressure state.  It is therefore too
fragile to retain.  At 0.01%, the run used 47.79 s and 2,676+631 Newtons with
38.43% WBHP; at 0.3%, it used 43.10 s and 2,317+577 Newtons with 38.60% WBHP.
All threshold variants were rejected.  Exact activation on any reverse flow
was reproducible across clean builds; the non-monotone threshold response
confirms that it separates active-set trajectories rather than acting as an
ordinary smooth tuning parameter.

### Coupled injector block (experimental mode 8)

The injector path had the same structural omission the producer originally
had: every nonlinear residual refresh solved one common rate-control BHP and
redistributed the requested rate among completions, but the Jacobian treated
each allocated completion source as constant.  Mode 8 now supplies the exact
eliminated pressure block.  For each active completion it differentiates
PVT, total mobility, capillary pressure, and the black-oil primary variable
numerically with the same host physics used by the residual.  The sparse
diagonal receives the fixed-BHP direct derivative and the matrix-free Schur
operator applies the common-BHP redistribution term.

Adding injector entries exposed and fixed a general ordering invariant:
the GPU reducer locates contiguous well groups with binary search, whereas
appending injectors before producers made the combined list non-monotone.
The upload now stable-sorts entries by schedule well index.  The unsorted
104.78-second result was invalid and is not used below.

| Injector coupling | Time (s) | Substeps | Accepted | Discarded | Total | WOPR | WWPR | WGPR | WGOR | WBHP | Day-3260 pressure mean/RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none, retained mode 7 | 42.4 | 581 | 2,268 | 371 | 2,639 | 9.49% | 12.70% | 11.89% | 23.93% | 34.19% | +443 / 532 psi |
| frozen pressure only | 54.04 | 645 | 3,123 | 797 | 3,920 | 9.06% | 10.05% | 10.61% | 23.29% | 31.63% | +385 / 467 psi |
| complete, water and gas | 59.57 | 662 | 3,376 | 849 | 4,225 | 9.07% | 11.22% | 11.51% | 23.93% | 26.46% | +392 / 482 psi |
| complete, water only | 62.95 | 764 | 3,582 | 1,105 | 4,687 | 9.06% | 10.23% | 10.77% | 24.77% | 33.79% | +377 / 468 psi |
| complete, gas only | 43.52 | 581 | 2,330 | 363 | 2,693 | 9.58% | 11.12% | 10.95% | 25.16% | 39.55% | +475 / 566 psi |

This is the strongest evidence so far that the remaining BHP bias is tied to
well physics rather than reporting or precision: the residual equations are
unchanged, yet a consistent injector allocation Jacobian lets the adaptive
solve follow a trajectory substantially closer to Flow.  Water-injector
coupling provides most of the pressure correction and most of the nonlinear
difficulty.  Mode 8 remains available for accuracy studies, but mode 7 stays
the balanced default.  The next complete-model step is to allow Flow-style
reverse production through an injector completion using reservoir composition.
The present one-sided max(0, BHP-theta) allocation changes its active set at
completion pressure crossings; adding ever more exact derivatives to that
nonsmooth, no-crossflow model cannot reproduce Flow's crossflow trajectory.

### Flow-style injector crossflow and retained mode 9

Mode 9 implements the missing injector physics from Flow's
`StandardWell::computePerfRate`.  At a completion producing into an injector,
the surface-component stream is computed from that cell's local phase
mobilities and black-oil conversion.  The produced stream joins the scheduled
wellhead injection stream to define one wellbore surface mixture.  At a
completion injecting into the reservoir, total mobility controls reservoir
volume and the wellbore mixture is converted with that completion's local B,
Rs, and Rv.  A scalar host solve chooses the common BHP so net wellhead rate
matches the active control while permitting internal crossflow.

The eliminated injector Jacobian uses well unknowns [BHP, oil fraction, water
fraction].  Its numerical local derivatives call the same pressure, mobility,
capillary, PVT, and black-oil primary-variable physics as the nonlinear
residual.  The complete 3 by 3 D block, reservoir-to-well C columns,
well-to-reservoir B rows, and fixed-well-variable direct derivatives are then
assembled as one Schur contribution.  Entries are stable-sorted by schedule
well index before GPU upload so each reducer group remains contiguous.

| Configuration | Time (s) | Substeps | Accepted | Discarded | Total Newton | WOPR | WWPR | WGPR | WGOR | WBHP | Day-3260 pressure mean/RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OPM Flow | 211.46 | 355 | 1,382 | included | 1,382 | reference | reference | reference | reference | reference | reference |
| mode 7, MB 1e-5 | 42.4 | 581 | 2,268 | 371 | 2,639 | 9.49% | 12.70% | 11.89% | 23.93% | 34.19% | +443 / 532 psi |
| mode 9, MB 1e-5 | 82.45 | 694 | 4,155 | 980 | 5,135 | 9.01% | 11.14% | 10.86% | 19.02% | 27.40% | +257 / 346 psi |
| mode 9, MB 2e-5 | 58.20 | 635 | 2,880 | 749 | 3,629 | 8.96% | 11.16% | 11.04% | 21.20% | 27.22% | +241 / 342 psi |
| mode 9, MB 3e-5 | 47.99 | 610 | 2,318 | 433 | 2,751 | 9.18% | 11.69% | 11.34% | 20.85% | 31.63% | +231 / 346 psi |
| **mode 9, MB 3.5e-5, retained** | **37.7** | **541** | **1,698** | **169** | **1,867** | **10.40%** | **11.01%** | **11.18%** | **19.43%** | **29.01%** | **+225 / 360 psi** |
| mode 9, MB 4e-5, rejected | 45.88 | 597 | 2,247 | 489 | 2,736 | 10.32% | 12.28% | 12.27% | 28.29% | 28.45% | +203 / 346 psi |

The retained trajectory reproduced exactly in two independent complete runs:
541 accepted substeps, 1,698 accepted Newton iterations, and 169 discarded
iterations.  It averages 3.14 accepted Newton iterations per substep versus
Flow's 3.89.  Total Newton work is now 1.35 times Flow rather than mode 7's
1.91 times, while wall time is about 5.6 times faster than Flow.  The remaining
Newton gap is almost entirely the 1.52 times larger number of accepted
substeps, not inefficient Newton convergence inside a successful step.

Relative to mode 7, pressure mean bias is nearly halved and pressure RMSE falls
32%; WGOR and WBHP NRMSE improve by 4.50 and 5.18 percentage points.  WOPR is
0.91 point worse, while WWPR and WGPR improve by 1.69 and 0.71 points.  The
3.5e-5 MB screen is deliberately retained instead of 4e-5: the latter saves
essentially no nonlinear work and causes a large WGOR and phase-rate
regression.  Final field-injection reporting was also corrected not to apply
the legacy one-sided capacity clip a second time after mode 9's host solve;
that reporting-only fix leaves the nonlinear trajectory unchanged.

The final WBHP score was also decomposed with the report's existing
Flow-activity mask, so shut-in samples cannot explain the remaining 29.01%
median.  The largest normalized outliers are short or intermittent histories:
K-3H has only 6 active comparison points and B-1BH 28.  Sustained producers
E-3H, D-1H, E-4AH, D-4H, E-1H, E-2H, B-1H, and B-3H lie between 8.22% and
22.11%.  D-2H and B-2H remain near 30%, while several switching wells remain
above 50%.  Thus the reporting correction is complete, but exact replication
of Flow's remaining per-well BHP trajectories would require further matching
of completion/control-state semantics around intermittent openings and
switches; it is no longer evidence of a missing common-well Jacobian column or
mixed-precision convergence failure.

### OPM TRACK ordering and final retained result

The default completion ordering was then checked directly against OPM
2024.10. `Well::updateRefDepth()` takes the first ordered connection when the
WELSPECS datum is defaulted, and the default COMPORD is TRACK.
`WellConnections::orderTRACK()` starts at the completion closest to the well
head in I/J and walks to the closest remaining I/J completion, with depth only
breaking ties. This is topological order, not minimum geometric depth and not
raw COMPDAT order. NVRS previously used minimum depth as the datum and raw
COMPDAT order for the cumulative column, creating a well-specific pressure
offset on deviated trajectories.

The retained implementation now reproduces OPM TRACK for both the default BHP
datum and segment hydrostatics. The immutable ordering is cached until the
schedule's completion topology changes, avoiding repeated nearest-neighbour
sorts on every Newton and line-search refresh. The cache is trajectory-neutral:
the cached and uncached runs produce byte-identical `RSM_FVM.csv` files and the
same nonlinear and linear counts.

| Configuration | Time (s) | Substeps | Accepted | Discarded | Total Newton | WOPR | WWPR | WGPR | WGOR | WBHP | Day-3260 pressure mean/RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| previous retained mode 9 | 37.60 | 541 | 1,698 | 169 | 1,867 | 10.40% | 11.01% | 11.18% | 19.43% | 29.01% | +225 / 360 psi |
| TRACK, uncached | 36.56 | 549 | 1,659 | 135 | 1,794 | 9.99% | 11.67% | 11.15% | 19.25% | 30.45% | +221 / 318 psi |
| **TRACK, cached, retained** | **36.31** | **549** | **1,659** | **135** | **1,794** | **9.99%** | **11.67%** | **11.15%** | **19.25%** | **30.45%** | **+221 / 318 psi** |
| Flow-style segment PVT experiment, rejected | 35.98 | 550 | 1,648 | 153 | 1,801 | 9.61% | 11.13% | 11.49% | 20.86% | 31.67% | -- |

TRACK reduces total Newton work by 73 and the day-3260 pressure RMSE by about
42 psi. Median normalized WBHP rises by 1.44 percentage points because that
statistic is dominated by several short, narrow-range histories, while pooled
active-well BHP MAE improves from 311 to 297 psi and RMS improves from 408 to
405 psi. The retained run is 5.82 times faster than Flow and uses 1.30 times
Flow's total Newton work. Accepted Newton efficiency is already better than
Flow (3.02 versus 3.89 iterations per accepted substep); the remaining work
gap is the 549 versus 355 accepted timesteps.

An additional source-faithful density experiment evaluated connection PVT at
lagged mean segment pressure and inferred Rs/Rv from the whole-well stream, as
`StandardWellConnections::computePropertiesForPressures()` does. In isolation
from Flow's full well-state update it marginally increased total Newton work
and worsened both WGOR and WBHP, so it was reverted. Likewise, OPM's retained
history-well BHP limit was verified to default to only 1.01325 bar; it is an
emergency atmospheric floor and does not explain the remaining pressure bias.

The final canonical host/deck suite passes all 444 checks. The benchmark
artifacts and `compare_NORNE.html` were regenerated from the cached TRACK run.

### Newton-state primary meaning in the coupled well block

The remaining retry clusters exposed a state-cache error at the exact boundary
the coupled black-oil formulation is meant to handle.  The device changes a
cell's third-primary meaning (`Sg`, `Rs`, or `Rv`) after every Newton update,
but the host well-control block refreshed its copy only once per accepted
substep.  A completion could therefore be assembled as `Rs` in the reservoir
while its well coefficient, PVT evaluation, BHP solve, and Schur derivative
still interpreted the same value as `Sg`.  The meaning array is Newton state,
not timestep history; it is now refreshed once per injector-to-producer well
refresh, while DRSDT, VAPPARS, and hysteresis remain cached at the accepted
state where they belong.

| Configuration | Time (s) | Substeps | Accepted | Discarded | Total Newton | WOPR | WWPR | WGPR | WGOR | WBHP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prior TRACK baseline | 36.31 | 549 | 1,659 | 135 | 1,794 | 9.99% | 11.67% | 11.15% | 19.25% | 30.45% |
| **current primary-meaning fix** | **30.65** | **529** | **1,313** | **87** | **1,400** | **8.71%** | **10.72%** | **10.21%** | **18.07%** | **27.40%** |
| OPM Flow 2025.10 | 211.46 | 354 detailed | 1,361 | 21 | 1,382 | reference | reference | reference | reference | reference |

The corrected run is reproducible: an independent second-GPU repeat has the
same substeps and Newton counts and a byte-identical `RSM_FVM.csv`.  It is
6.90x faster than the CPU Flow reference and total Newton work is now only 18
iterations (1.3%) higher.  All five tracked well metrics improve, including
the two previous concerns: WGOR improves by 1.18 percentage points and WBHP by
3.05 points.  Non-descent updates fall from 169 to 94.

The dormant relative-change PID controller was also compared with OPM source.
Its history indices were one accepted step stale; the implementation now uses
the same old-e1, old-e2, current triplet as Flow before shifting the history.
`DTPID` remains disabled in the retained Norne deck, so this correctness fix
does not alter the promoted trajectory.

Several isolated controller screens were rejected. `DTCAPGROW 1.075` reached
1,517 total Newtons but worsened WGOR and WBHP; `DTFAILCAP 0.70`, `DTMAX 20`,
`NEWTONSTALLMAX 8`, and `DTRESTART 0.40` all increased total Newton work.  A
line-search handoff experiment that forced the producer hydrostatic column to
be rebuilt after every accepted trial also regressed to 573 substeps and 2,171
total Newtons, so the path-preserving assembly handoff remains enabled.  After
the primary-meaning fix, the controller neighborhood was screened again:
Flow-style `DTPID 1` required 2,400 total Newtons, `DTCAPGROW 1.075` and 1.125
both required 1,600, and 1.105 required 2,770.  The 1.075 variant recovered
day-3260 pressure mean/RMSE from +276/+369 psi to +211/+325 psi and improved
WBHP to 26.24%, but its extra 200 Newton iterations miss the stated work
objective.  The retained 1.10 setting is therefore the sharp nonlinear-work
knee, not an untested round-number default.

The final canonical host/deck suite again passes all 444 checks.  The promoted
benchmark artifacts and `compare_NORNE.html` are generated from this corrected
primary-meaning run.


## SPE10 model 2 handoff (2026-09-11)

The public 60 x 220 x 85 SPE10 model 2 property set is now available under
`dataset/SPE10`, and the generated NVRS deck is
`benchmark/runs/SPE10/SPE10.DATA`. The original OPM dataset used for the
reference is `/home/tonishi/Workdir/OPM/opm-data/spe10model2`.

A fresh full OPM Flow 2025.10 CPU CPR reference was run with one MPI process
and two OpenMP threads using the untouched original deck:

```sh
HWLOC_COMPONENTS=-gl /usr/bin/flow \
  /home/tonishi/Workdir/OPM/opm-data/spe10model2/SPE10_MODEL2.DATA \
  --output-dir=benchmark/opm_spe10_flow_2025_10 \
  --check-satfunc-consistency=false
```

The consistency switch is needed only because Flow 2025.10 computes the legacy
SGOF oil critical saturation as -5.55e-17 and otherwise rejects the deck. It
does not modify the tables. The run completed 2000 days in 2780.55 seconds of
simulation time plus 7.70 seconds setup. There were 109 attempted timesteps:
91 accepted and 18 failed. Accepted steps used 569 Newton iterations; failed
attempts used 38, for 607 total. Flow used 4399 linear iterations, including
377 wasted. Through day 400 it used 45 accepted steps and 349 accepted Newton
iterations, plus one failed 20-iteration attempt. Outputs are in
`benchmark/opm_spe10_flow_2025_10`.

The supported NVRS full run completed with:

```sh
GPUS=1 ./scripts/run_spe10.sh
```

The solver reported 11.84 seconds, 20 accepted 100-day steps, 20 Newton
updates, 2778 AmgX applications, and no timestep retries on one RTX 6000 Ada.
The end-to-end executable wall time was 13.73 seconds; report generation then
produced `benchmark/spe10/spe10.html`. The one-GPU harness now launches the
MPI-initializing executable directly because the redundant outer
`mpirun -np 1` hangs on this workstation with stale singleton orted helpers.
Multi-GPU runs retain mpirun.

The 11.84-second NVRS result is provisional and must not be quoted as a Flow
speedup. From day 700 onward several FGMRES solves reached LINMAXIT 200 with
relative residuals as poor as 0.33, while OPM-style relaxed MB/CNV accepted the
state after one Newton update even when the printed nonlinear L2 residual grew.
This is also a different physical problem: the generated NVRS stress deck is
two-phase dead oil with constant PVT, zero rock/fluid compressibility, a 5000
STB/day surface injector, a porosity floor of 1e-4, and uniform 100-day report
steps. The original Flow deck is three-phase compressible black oil, has PVDO,
PVTW, PVDG and ROCK data, uses a 5000 RB/day RESV injector, and begins with
1, 2, 3, 4, 5, 5, 10 and other nonuniform report increments. Field and well
differences currently combine model, control, schedule, and numerical error.

Before publishing a SPE10 comparison, construct a Flow-readable twin of the
NVRS two-phase deck, including the same floored porosity, constant properties,
surface-rate control, initialization, and 20 x 100-day schedule. Then compare
states and well curves at every common report time. Independently, require a
validation screen for an accepted NVRS step following an unconverged linear
solve; the current relaxed MB/CNV exit may be legal but is not sufficient
evidence of an accurate trajectory in this case.

### Matched 100-day SPE10 result

The provisional comparison above is superseded by a strict two-phase twin:
both executables read the same 1,122,000-cell properties, FIELD-unit PVT,
relative-permeability tables, initialization, five completions, controls, and
single 100-day report interval. OPM Flow 2025.10 ran its default CPU CPRW
true-IMPES configuration with one MPI rank and two OpenMP threads. NVRS ran
one RTX 6000 Ada GPU with native FP64 storage and AmgX CPR.

The decisive convergence improvement was replacing the field two-phase CPR
quasi-IMPES weights with the analytic true-IMPES storage null vector
[1, Bo/Bw], normalized by its largest entry. On the initial 0.33-day
problem this reduced NVRS from 14 Newton / 423 Krylov iterations to 11 / 92,
exactly matching Flow's 11 Newton iterations. A conditional line-search guard
also requires raw-CNV descent only while CNV exceeds the relaxed Flow
threshold; keeping that guard active below the threshold caused needless
chopping.

| Matched 100-day result | OPM Flow | NVRS |
|---|---:|---:|
| Simulation time | 303.59 s | 56.60 s |
| Accepted substeps | 16 | 47 |
| Accepted Newton | 122 | 163 |
| Discarded Newton | 21 | 20 |
| Total Newton | 143 | 183 |
| Linear iterations | 519 | 12,725 |

NVRS is 5.36x faster and uses 1.28x Flow's total Newton work. It still takes
2.94x as many accepted substeps, so its remaining nonlinear gap is primarily
time-step robustness, not floating-point precision. Native FP64 was marginally
faster than emulated precision with the same Newton trajectory.

At day 100, NVRS differs from Flow by -0.867% FOPR, +0.559% FOPT, +13.544%
FWPR, and -4.827% FWPT. The water-rate percentage is dominated by producer P3;
the cumulative is the more stable diagnostic. Pressure bias/RMS is
-36.26/58.76 psi and SWAT bias/RMS is -2.51e-5/0.007327. All four
fixed-BHP producers agree exactly. Injector BHP differs by -49.05 psi
(-0.865%). NVRS cumulative oil and water are now integrated over every
accepted internal substep rather than approximated by a report-end rectangle.

Rejected isolated experiments included solve-level well Woodbury coupling,
extra block-Jacobi or colored Gauss-Seidel sweeps, MGR, BiCGSTAB, looser
linear tolerances, reduced AmgX strength thresholds, and more aggressive
time-step growth. Each increased nonlinear work, wall time, or both. The
validated self-contained report is benchmark/runs/compare_SPE10.html.

### SWOF, fixed-BHP hydrostatics, and matched 1000-day SPE10 result (2026-09-12)

This result supersedes the 100-day numbers above. OPM Flow and NVRS used the
same 1,122,000-cell rock, PVT, SWOF, wells, controls, and initialization, with
the same 25-day report schedule through day 1000. Flow 2025.10 used CPU CPRW
with one MPI rank and two OpenMP threads; NVRS used one RTX 6000 Ada GPU.

Two independent causes explained most of the apparently large discrepancy:

1. NVRS evaluated the two-phase relative permeability with quadratic Corey
   curves even when the deck supplied a 203-row SWOF table. The residual,
   Jacobian, wells, reporting, AIM, and fallback paths now all use one
   piecewise-linear SWOF implementation when `FIELDSATFUNC 1` is active.
2. Fixed-BHP producers returned before connection hydrostatics were applied.
   Every completion therefore received the datum BHP, rather than datum BHP
   plus the well-segment hydrostatic pressure. The fixed-BHP path now goes
   through the same completion-pressure construction as coupled controls.
3. The comparison HTML treated NVRS's instantaneous day-zero Peaceman
   diagnostic as a produced interval. Well plots and diagnostics now exclude
   that synthetic point, and recovery uses the simulator-reported FOPT rather
   than trapezoidal integration of sparse report samples.

| Matched 1000-day result | OPM Flow | NVRS |
|---|---:|---:|
| Simulation time | 800.99 s | 824.70 s |
| Accepted substeps | 53 | 372 |
| Accepted Newton | 308 | 607 |
| Discarded Newton | 0 | 500 |
| Total Newton | 308 | 1107 |
| Linear iterations | unavailable | 192,241 |
| Final FOPR difference | reference | -3.215% |
| Final FWPR difference | reference | +0.703% |
| Final FOPT difference | reference | -0.381% |
| Final FWPT difference | reference | +0.350% |
| Pressure bias / RMS | reference | -7.86 / 17.90 psi |
| SWAT bias / RMS | reference | -0.001743 / 0.015364 |
| P1-P4 fixed WBHP | reference | exact |

NVRS is 1.03x slower than Flow on this long run, so wall time is now close;
it is not yet faster. Its accepted Newton count is 1.97x Flow's and total
Newton work is 3.59x, with 7.02x as many accepted substeps. The remaining gap
is nonlinear timestep robustness rather than mixed-precision roundoff or
PINCH transmissibility. The close cumulative production and pressure results
also show that the formerly dramatic curve mismatch was not one single solver
error: SWOF and well hydrostatics were physical errors, while the extreme
day-zero peak and cumulative shown by the report were presentation errors.

Safe performance tuning selected `LINTOL 4e-2` and `AMGREUSE 0`. Rebuilding a
fresh PMIS hierarchy was faster here than reusing it, and removing numerical
well-Schur derivatives for fixed-BHP producers eliminated dead work without
changing the 100-day state. Looser 5e-2 linear tolerance, AMG reuse, disabling
the raw-CNV line-search guard, a second line-search trial, and lower Schur
levels all increased wall time or nonlinear work and were rejected.

The final self-contained report is `benchmark/runs/compare_SPE10.html`. Its
recovery summary uses FOPT and excludes the synthetic day-zero well sample;
the oil/water curves use only physical positive-time samples. Temporary output
directories under `/tmp` held the large ECL files for this experiment and are
not repository artifacts.

### Completion-only well refresh and final cross-case timing (2026-09-13)

Profiling the matched 1,122,000-cell SPE10 run isolated a non-physical CPU
cost in `apply_injector_rate_targets()`: every Newton and line-search refresh
cleared injector coupling fields across every owned reservoir cell, although
only scheduled completion cells can ever contain those fields. SPE10 has at
most 425 scheduled completion entries. Clearing `state_index.loc` instead
of all 1,122,000 cells is algebraically identical and reduced well-control
refresh time from 115.80 s to 24.24 s.

The 100-day A/B test retained exactly 68 accepted substeps, 208 accepted and
45 discarded Newton iterations, and 9,877 linear applications. Wall time
fell from 56.95 s to 45.70 s (19.8%). The full 1000-day trajectory likewise
retained exactly 372 accepted substeps, 607 accepted and 500 discarded Newton
iterations, and 192,241 linear applications. Wall time fell from 824.70 s
to 733.115 s, making NVRS 1.09x faster than Flow's 800.99 s. All checkpoint
and summary differences remained unchanged: final FOPR -3.215%, FWPR +0.703%,
FOPT -0.381%, FWPT +0.350%, pressure bias/RMS -7.86/17.90 psi, SWAT bias/RMS
-0.001743/0.015364, and exact fixed BHP for P1--P4.

Three nonlinear alternatives were rejected. Raising NEWTONMAX from 20 to 30
made the 100-day run take 114.57 s because failed large steps consumed their
larger budgets. Tightening LINTOL from 4e-2 to 1e-2 improved the initial
retry but increased the full 100-day time to 67.61 s. Applying that tighter
tolerance only during startup still took 66.22 s and increased discarded work.
Increasing DPMAXREL from 0.3 to 1.0 did not change the trajectory. Thus the
remaining SPE10 gap is not a Newton iteration cap, startup linear accuracy,
or pressure-update clipping.

The promoted NORNE configuration was restored after those SPE10 screens:
WELLXFLOW 2, WELLSCHUR 9, PVSWITCH 1, NEWTONMAX 30, MBTOL 3.5e-5, and
CNVRELAXED 10.0. The current full-history run is:

| NORNE full history | OPM Flow | NVRS |
|---|---:|---:|
| Simulation time | 211.46 s | 33.338 s |
| Accepted timesteps/substeps | 355 | 552 |
| Accepted Newton | 1361 | 1377 |
| Discarded Newton | 21 | 197 |
| Total Newton | 1382 | 1574 |
| Speedup | reference | 6.34x |

NVRS now uses only 1.14x Flow's total Newton work on NORNE while remaining
6.34x faster. Its current endpoint field differences are FOPR +8.753%, FWPR
-15.452%, FGPR +9.975%, FOPT +0.567%, and FWPT -8.470%; median errors are
generally smaller. Pressure bias/RMS at the nearest final restart are
+206.8/323.6 psi. BHP remains the dominant accuracy gap for NORNE, especially
for inactive/shut wells, and is a well/group-control physics problem rather
than a reason to weaken the validated speed result.

### Persistent failure cap, CGS1, and final 1000-day SPE10 result (2026-09-13)

The 733.115-second SPE10 trajectory spent 500 Newton iterations and 141,164
of its 192,241 linear applications on rejected attempts. Nearly every report
interval repeated the same cycle: grow to roughly 4.5--5.4 days, exhaust up
to 15 Newton iterations, then chop to roughly 1.5 days. The learned failure
ceiling was both polluted by the unavoidable startup homotopy and relaxed
again after each accepted step.

Two changes close most of that gap:

1. A two-phase startup failure is no longer learned as a permanent timestep
   ceiling before the first state has been accepted. Subsequent failures are
   still learned. With `DTCAPGROW 1.0`, SPE10 does not deliberately grow back
   into the failed length. The three-phase/NORNE path is unchanged.
2. FGMRES now accepts `ORTHOPASSES 1|2`. CGS2 remains the default, while the
   matched SPE10 deck opts into CGS1. On the 100-day screen, CGS1 preserved
   38 accepted substeps and essentially the same nonlinear/linear trajectory,
   but reduced Arnoldi orthogonalisation from 13.40 to 7.23 seconds. Endpoint
   rates differed by at most 0.203%, pressure by 0.046 psi RMS, saturation by
   4.54e-5 RMS, and fixed BHP by 0.05 psi from CGS2.

The explicit deck-keyword regression, without the experimental environment
override, completed 100 days in 35.95 seconds with 38 accepted substeps, 171
accepted plus 36 discarded Newton iterations, and 10,084 linear applications.

| Matched 1000-day result | OPM Flow | Previous NVRS | Final NVRS |
|---|---:|---:|---:|
| Simulation time | 800.99 s | 733.115 s | 164.041 s |
| Speedup vs Flow | reference | 1.09x | 4.88x |
| Accepted substeps | 53 | 372 | 290 |
| Accepted Newton | 308 | 607 | 554 |
| Discarded Newton | 0 | 500 | 36 |
| Total Newton | 308 | 1107 | 590 |
| Linear applications | unavailable | 192,241 | 39,723 |

Only one of 40 report intervals now rejects a timestep. The final run spends
8.23 seconds in assembly and 114.07 seconds in the linear solver: 36.28 in
AmgX V-cycles, 33.01 in operator applications, 19.01 in stage 2, and 28.53 in
CGS1 orthogonalisation (overlapping timed components are not additive).

Accuracy is retained. Against Flow at day 1000, final FOPR is -3.220%, FWPR
+0.705%, FOPT -0.356%, and FWPT +0.326%; pressure bias/RMS is
-7.854/17.899 psi and SWAT bias/RMS is -0.001706/0.015329. P1--P4 fixed BHP
remains exact. Relative to the previous NVRS trajectory, endpoint FOPR/FWPR/
FOPT/FWPT move only -0.005/+0.002/+0.025/-0.023 percentage points, pressure
changes by 0.023 psi RMS, and SWAT by 2.26e-4 RMS.

Pressure-preconditioner screens did not improve time. `STAGE2 ZLINE` reproduced
the Jacobi trajectory and 37.4-second cost. AmgX strength thresholds 0.25 and
0.70 took 44.3 and 41.8 seconds. Two AMG sweeps reduced linear applications
to 8,084 but raised time to 38.9 seconds because each V-cycle cost 1.45 ms.
FGMRES restart 20 took 47.1 seconds and 14,465 applications. Aggregation AMG,
MGR, BiCGSTAB, hierarchy reuse, and stronger stage-2 variants had already been
rejected. The default PMIS strength 0.5, one V-cycle, restart 30 configuration
therefore remains the measured optimum.

The regenerated self-contained report is `benchmark/runs/compare_SPE10.html`.

### Large-step controls and restart-60 SPE10 result (2026-09-13)

Five follow-up avenues were implemented or audited independently:

1. `DTCAPPROBE`, `DTCAPPROBEITERS`, and `DTCAPPROBEFACTOR` cautiously reopen
   a learned failure ceiling after a configurable run of cheap solves. A failed
   probe restores the last proven ceiling instead of ratcheting it downward.
2. Flow's existing relative-change PID controller was tested with `DTPID 1`.
3. `LINEBEST 1` retains the best finite line-search trial when none strictly
   descends, rather than blindly taking the smallest trial.
4. `LINFORCINGMAX` enables a bounded Eisenstat--Walker-style inexact-Newton
   tolerance, varying between `LINTOL` and the requested maximum.
5. A verbose audit was run at the reproducible second 3.996-day step near day
   200, where a cautious timestep probe failed.

The audit showed that the large-step blocker was linear, not primarily a
missing Jacobian derivative. At that state every solve reached `LINMAXIT 300`
with relative linear residual 0.36--0.50. The line search could then accept
only alpha 1/64 and Newton descended too slowly to finish. CGS2 recovered the
step but its second global projection made the 250-day run slower. A longer
CGS1 FGMRES basis instead fixed the pressure-spectrum stagnation efficiently.

| 250-day restart screen, CGS1 | Wall time | Total Newton | Linear applies |
|---|---:|---:|---:|
| 30 | 63.63 s | baseline | 17,396 |
| 45 | 50.48 s | 268 | 11,334 |
| 55 | 48.95 s | 297 | 9,751 |
| **60** | **42.89 s** | **260** | **8,432** |
| 65 | 52.75 s | 285 | 10,886 |
| 75 | 46.70 s | 292 | 8,961 |
| 90 | 44.03 s | 264 | 8,583 |

Adaptive forcing was rejected: `LINFORCINGMAX 0.08` created 631 accepted
substeps and took 91.86 seconds. Best-trial globalization did not recover the
hard pressure solve and took 75.47 seconds with probing. DTPID reproduced the
fixed-ceiling trajectory at 63.96 seconds. A 12-step/1.25x probe took 76.03
seconds. Later 1.10x probes successfully reached 4--6.4 days, proving that the
controller works, but larger pressure systems cost enough extra Krylov work to
lose wall time.

Restart 60 without probing is therefore promoted in `SPE10_MATCHED.DATA`:

| Matched 1000-day result | OPM Flow | Previous NVRS | Restart-60 NVRS |
|---|---:|---:|---:|
| Simulation time | 800.99 s | 164.041 s | **101.814 s** |
| Speedup vs Flow | reference | 4.88x | **7.87x** |
| Accepted substeps | 53 | 290 | 314 |
| Accepted Newton | 308 | 554 | 484 |
| Discarded Newton | 0 | 36 | 36 |
| Total Newton | 308 | 590 | **520** |
| Linear applications | unavailable | 39,723 | **17,747** |

The optional 40- and 80-cheap-step probes reduced total Newton to 492 and 496,
but increased runtime to 120.10 and 111.55 seconds. This is direct evidence
that minimizing Newton/substep count is not the same as minimizing GPU time:
the larger-timestep matrices are substantially harder.

Accuracy remains stable. Against Flow, restart 60 gives final FOPR -3.191%,
FWPR +0.697%, FOPT -0.374%, and FWPT +0.338%; pressure bias/RMS is
-7.750/17.856 psi, SWAT bias/RMS is -0.001740/0.015336, and all four fixed
producer BHPs are exact. Relative to restart 30, the final pressure changes by
0.109 psi RMS and SWAT by 8.19e-5 RMS. The self-contained comparison HTML was
regenerated from the restart-60 run.

### Report-boundary timestep state and zero-guess Krylov path (2026-09-13)

The restart-60 run exposed a controller artefact unrelated to the reservoir
physics. The last internal step in each 25-day report interval was truncated
to the remaining time, and that shortened physical step was then used as the
growth baseline for the next interval. This produced a repeating sawtooth
instead of retaining the controller's nominal proposal. `on_success` now gets
both lengths and grows from the untruncated controller length. Over 1000 days,
accepted substeps fell from 314 to 289 and accepted linear applications from
17,747 to 17,180. Accepted and discarded Newton work remained 484 and 36.

Every FIM and sequential solve clears its correction before calling the
linear solver. An explicit zero-initial-guess flag now skips the redundant
initial `A*0` operator application in FGMRES and BiCGSTAB. The flag is not used
for nonzero Woodbury cleanup guesses or Krylov restarts. The Arnoldi host
projection buffers are also allocated once per solve instead of once per
iteration. The complete well-rate CSV is byte-identical before and after this
change, as are all Newton and Krylov counts.

| Matched 1000-day result | OPM Flow | Earlier restart-60 NVRS | Final NVRS |
|---|---:|---:|---:|
| Simulation time | 800.99 s | 101.814 s | **97.7206 s** |
| Speedup vs Flow | reference | 7.87x | **8.20x** |
| Accepted substeps | 53 | 314 | **289** |
| Accepted Newton | 308 | 484 | 484 |
| Discarded Newton | 0 | 36 | 36 |
| Total Newton | 308 | 520 | 520 |
| Accepted linear applications | unavailable | 17,747 | **17,180** |

Large-step experiments confirm why Newton count alone must not be optimized.
Allowing 30 Newton iterations removed most startup chops and reduced the
250-day run to 227 total Newton iterations, but the large-step systems needed
17,240 linear applications and 69.13 seconds. Capping those steps at four days
took 40.60 seconds with 253 Newton iterations, but settles into nine internal
steps per report interval and does not beat the retained controller over the
full trajectory. BiCGSTAB, CGS2/three line-search trials, Z-line stage 2, and
restart 90 were also slower. The remaining Newton gap to Flow is therefore a
preconditioner/nonlinear-model issue: Flow-size timesteps can already be
requested, but their pressure systems are too costly with the present CPR
stage to improve wall time.

Accuracy versus Flow is unchanged to reporting precision: day-1000 FOPR is
-3.181%, FWPR +0.694%, FOPT -0.385%, and FWPT +0.345%; pressure bias/RMS is
-7.716/17.843 psi and SWAT bias/RMS is -0.001742/0.015326. Fixed producer BHP
is exact. The final comparison report is `benchmark/runs/compare_SPE10.html`.
