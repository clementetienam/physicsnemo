#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
"""How many launches to skip so the trace fits a row budget.

The per-launch trace of a long run is millions of rows and the timeline needs
far fewer. The obvious way to bound it is to stop reading after N, and that is
the wrong operation: a run's kernels are spread across its whole duration, so
truncating keeps the opening minutes and discards the rest. On the 247-step
Norne deck that is a few percent of the history, drawn as though it were the
history.

Taking every Nth launch instead covers the whole run at lower resolution, which
is what a timeline binned into a thousand columns wants anyway.

N is derived, not chosen. The kernel summary is written before the trace and
its Instances column counts every launch, so the number needed is already
known by the time it is needed:

    N = ceil(total launches / row budget)

Prints the stride on stdout. Prints 1, and succeeds, whenever the total cannot
be established -- an unusable stride would cost the whole timeline, and a
stride of 1 is the old behaviour.
"""
import csv
import math
import sys


def main():
    if len(sys.argv) < 3:
        print(1)
        return 0
    path, budget = sys.argv[1], sys.argv[2]

    try:
        budget = int(budget)
    except ValueError:
        budget = 400000
    if budget <= 0:
        print(1)
        return 0

    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        print(1)
        return 0

    # The header is not reliably line 1: nsys prints a report title above it.
    start = 0
    for i, ln in enumerate(lines[:80]):
        if "," in ln and "instances" in ln.lower():
            start = i
            break
    else:
        print(1)
        return 0

    total = 0
    for row in csv.DictReader(lines[start:]):
        for k, v in row.items():
            if k and "instances" in k.lower():
                try:
                    total += int(float(str(v).replace(",", "").strip()))
                except (ValueError, TypeError):
                    pass
                break

    if total <= 0:
        print(1)
        return 0
    print(max(1, math.ceil(total / float(budget))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
