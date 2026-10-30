#!/usr/bin/env python3
"""Compact numeric comparison of one NVRS checkpoint against OPM Flow.

Usage:
  PYTHONPATH=scripts python benchmark/checkpoint_compare.py OPM_BASE NVRS_BASE

The Flow series is interpolated to NVRS report times.  ECLIPSE summary units
are converted from METRIC to the FIELD units written by NVRS.  Restart-state
norms use the closest Flow report to the last NVRS restart.
"""

import argparse
import numpy as np

from ecl import EclipseCase


BAR_TO_PSI = 14.5037738
M3_TO_BBL = 6.28981077
SM3_TO_MSCF = 0.0353146667


def scale(keyword):
    if keyword in {"FOPR", "FWPR", "FVPR", "FOPT", "FWPT", "FVPT"}:
        return M3_TO_BBL
    if keyword in {"FGPR", "FGIR", "FGPT", "FGIT"}:
        return SM3_TO_MSCF
    if keyword == "FPR" or keyword.startswith("WBHP:"):
        return BAR_TO_PSI
    return 1.0


def finite_relative(got, ref):
    mask = np.isfinite(got) & np.isfinite(ref) & (np.abs(ref) > 1e-12)
    return 100.0 * (got[mask] - ref[mask]) / np.abs(ref[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("opm_base")
    ap.add_argument("nvrs_base")
    ap.add_argument("--until", type=float,
                    help="compare only NVRS reports/restarts through this day")
    args = ap.parse_args()

    opm, nv = EclipseCase(args.opm_base), EclipseCase(args.nvrs_base)
    opm_metric = (opm.unit_system or "").upper() == "METRIC"
    to, so = opm.summary()
    tn, sn = nv.summary()
    if args.until is not None:
        time_eps = max(1e-6, abs(args.until) * 1e-6)
        keep = tn <= args.until + time_eps
        tn = tn[keep]
        sn = {k: {**v, "values": np.asarray(v["values"])[keep]}
              for k, v in sn.items()}
    print("last NVRS summary time: %.6g d" % tn[-1])
    print("vector       endpoint       median     samples")
    for kw in ("FOPR", "FWPR", "FGPR", "FVPR", "FOPT", "FWPT", "FVPT"):
        if kw not in so or kw not in sn:
            continue
        use = (tn >= to[0]) & (tn <= to[-1])
        ref_scale = scale(kw) if opm_metric else 1.0
        ref = np.interp(tn[use], to, so[kw]["values"]) * ref_scale
        got = np.asarray(sn[kw]["values"])[use]
        err = finite_relative(got, ref)
        endpoint = finite_relative(got[-1:], ref[-1:])
        print("%-8s %12.3f%% %12.3f%% %9d" %
              (kw, endpoint[0] if endpoint.size else np.nan,
               np.median(err) if err.size else np.nan, err.size))

    # Selected well BHPs at the endpoint are often more diagnostic than a
    # field pressure average under RESV history control.
    print("\nendpoint well BHP error")
    for label in sorted(k for k in sn if k.startswith("WBHP:")):
        if label not in so:
            continue
        ref_scale = BAR_TO_PSI if opm_metric else 1.0
        ref = np.interp(tn[-1], to, so[label]["values"]) * ref_scale
        got = sn[label]["values"][-1]
        pct = 100.0 * (got - ref) / abs(ref) if abs(ref) > 1e-12 else np.nan
        print("%-16s %+9.2f psi (%+7.3f%%)" %
              (label, got - ref, pct))

    tr_o, tr_n = opm.restart_times(), nv.restart_times()
    if tr_o.size and tr_n.size:
        if args.until is not None:
            time_eps = max(1e-6, abs(args.until) * 1e-6)
            eligible = np.flatnonzero(tr_n <= args.until + time_eps)
            if eligible.size == 0:
                return
            inn = int(eligible[-1])
        else:
            inn = len(tr_n) - 1
        io = int(np.argmin(np.abs(tr_o - tr_n[inn])))
        print("\nrestart state: NVRS %.6g d, nearest Flow %.6g d" %
              (tr_n[inn], tr_o[io]))
        for kw, fac in (("PRESSURE", BAR_TO_PSI), ("SWAT", 1.0),
                        ("SGAS", 1.0)):
            xo, xn = opm.restart_field(kw), nv.restart_field(kw)
            if xo is None or xn is None:
                continue
            ref_scale = fac if opm_metric else 1.0
            d = np.asarray(xn[inn]) - np.asarray(xo[io]) * ref_scale
            d = d[np.isfinite(d)]
            print("%-8s bias %+12.6g  RMS %12.6g" %
                  (kw, np.mean(d), np.sqrt(np.mean(d * d))))


if __name__ == "__main__":
    main()
