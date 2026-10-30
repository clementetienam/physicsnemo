#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""opm_timesteps.py -- the substeps OPM Flow accepted, from its log.

    python3 opm_timesteps.py flow.log > flow_dt_sequence.txt

Flow prints one block per attempted substep,

    Starting time step 1, stepsize 5.6 days, at day 3306.4/3312, date = ...
     Newton its= 3, linearizations= 4 (0.1sec), linear its= 55 (0.2sec)

and, when an attempt fails, a convergence-failure line and "Timestep chopped
to ..." before the retry.  An attempt is kept when no failure is reported
before the next attempt starts.  Each accepted substep ends where the next
accepted one starts (the printed start days are exact boundaries; the printed
stepsize is rounded), and the last one ends at the final report day.

Output: "t_start_days dt_days" per line, the format NVRS's DTSEQFILE reads.
Also prints a one-line summary on stderr: accepted and failed attempts.
"""
import re
import sys

START = re.compile(r"Starting time step\s+\d+,\s*stepsize\s+([0-9.eE+-]+)\s*days?,"
                   r"\s*at day\s+([0-9.eE+-]+)\s*/\s*([0-9.eE+-]+)")
FAIL = re.compile(r"(Timestep chopped|convergence failure|Solver convergence "
                  r"failure|Problem:|failed to converge)", re.I)


def accepted_steps(lines):
    attempts = []                 # [start_day, stepsize, end_day_total, ok]
    for ln in lines:
        m = START.search(ln)
        if m:
            attempts.append([float(m.group(2)), float(m.group(1)),
                             float(m.group(3)), True])
            continue
        if attempts and FAIL.search(ln):
            attempts[-1][3] = False
    ok = [a for a in attempts if a[3]]
    steps = []
    for i, a in enumerate(ok):
        t0 = a[0]
        if i + 1 < len(ok):
            t1 = ok[i + 1][0]
        else:
            t1 = min(a[0] + a[1], a[2])
        if t1 > t0:
            steps.append((t0, t1 - t0))
    return steps, len(attempts) - len(ok)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    with open(argv[1], errors="replace") as f:
        steps, failed = accepted_steps(f)
    print("# accepted substeps from %s: t_start_days dt_days" % argv[1])
    for t0, dt in steps:
        print("%.10g %.10g" % (t0, dt))
    sys.stderr.write("[opm_timesteps] %d accepted substeps, %d failed attempts, "
                     "%.1f days\n" % (len(steps), failed,
                                      sum(d for _, d in steps)))
    return 0 if steps else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
