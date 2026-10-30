#!/usr/bin/env python3
"""units_probe.py -- what unit system is each case actually written in?

Norne is a METRIC deck, so OPM Flow writes metric output: pressures in bar,
volumes in m3, rates in sm3/day.  NVRS solves and reports in field units.
Comparing the two without converting puts 14.5x on every pressure and 6.29x
on every rate, which looks like a modelling difference and is not.

This prints what each file DECLARES -- the INTEHEAD unit flag and the UNITS
string of each summary vector -- alongside the magnitudes, so the question is
settled by the files rather than by inference.

    python units_probe.py runs/NORNE/opm/NORNE runs/NORNE/amgx/NORNE
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
import ecl  # noqa: E402


def discover(roots=("runs/NORNE", "runs", ".")):
    """Every ECLIPSE output base under `roots`, found rather than guessed.

    The base is the path without its extension, and the name is the deck's,
    not the case directory's -- OPM writes NORNE_ATW2013.UNRST where the
    directory is called opm.  Guessing it wrong gives "cannot determine grid\ndimensions", which sounds like a corrupt file and is a wrong path.
    """
    bases = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for ext in (".UNRST", ".SMSPEC", ".EGRID"):
            for f in glob.glob(os.path.join(root, "**", "*" + ext),
                               recursive=True):
                b = f[: -len(ext)]
                if b not in bases:
                    bases.append(b)
        if bases:
            break
    return sorted(bases)


def probe(base):
    print("=" * 74)
    print(base)
    print("=" * 74)
    try:
        c = ecl.EclipseCase(base)
    except Exception as exc:
        print("  cannot open: %s" % exc)
        return
    for ext in (".INIT", ".UNRST", ".SMSPEC", ".EGRID"):
        print("  %-8s %s" % (ext, "yes" if os.path.exists(base + ext) else "MISSING"))
    print("  INTEHEAD unit system : %s" % (c.unit_system or "unknown"))

    try:
        porv = c.init_field("PORV")
    except Exception as exc:
        print("  PORV unavailable: %s" % exc)
        porv = None
    if porv is not None:
        v = np.nan_to_num(np.asarray(porv, float), nan=0.0)
        print("  PORV  sum %.4g   max %.4g   (rm3 if METRIC, rb if FIELD)"
              % (v.sum(), v.max()))

    try:
        p = c.restart_field("PRESSURE")
        sw = c.restart_field("SWAT")
    except Exception as exc:
        print("  no restart fields: %s" % exc)
        p = sw = None
    if p is not None and len(p):
        a = np.nan_to_num(np.asarray(p[0], float), nan=0.0)
        a = a[a > 0]
        if a.size:
            print("  PRESSURE at t=0      : %.1f .. %.1f   (bar if METRIC, "
                  "psi if FIELD)" % (a.min(), a.max()))
    if sw is not None and len(sw):
        a = np.nan_to_num(np.asarray(sw[0], float), nan=0.0)
        print("  SWAT at t=0          : %.4f .. %.4f   (dimensionless)"
              % (a.min(), a.max()))

    try:
        t, vec = c.summary()
    except Exception as exc:
        print("  no summary: %s" % exc)
        return
    if not vec:
        print("  no summary")
        return
    print("  summary vectors, as the SMSPEC declares them:")
    want = ("WBHP", "WOPR", "WWPR", "WGPR", "FOPT", "FWPT", "WWCT", "WGOR")
    seen = set()
    for name, e in sorted(vec.items()):
        kw = e.get("keyword", "")
        if kw not in want or kw in seen:
            continue
        seen.add(kw)
        val = np.asarray(e["values"], float)
        nz = val[np.isfinite(val) & (val != 0)]
        print("    %-10s unit %-10s peak %12.4g   (%s)"
              % (kw, e.get("unit", "?") or "-",
                 nz.max() if nz.size else 0.0, name))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = discover()
        if not args:
            print("no ECLIPSE output found under runs/ or here")
            sys.exit(1)
        print("found %d case(s): %s\n" % (len(args), ", ".join(args)))
    for b in args:
        try:
            probe(b)
        except Exception as exc:
            print("  probe failed: %s: %s" % (type(exc).__name__, exc))
