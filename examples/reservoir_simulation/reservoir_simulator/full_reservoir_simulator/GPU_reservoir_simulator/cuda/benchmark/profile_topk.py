"""Print the kernels that account for most of the GPU time.

nsys cannot be asked to collect only the interesting kernels -- it traces
everything or nothing -- so this reads the summary it already wrote.  The point
is to answer "what dominates?" from the terminal, without opening the report in
a GUI or loading the CSV into anything.

Kernel names contain commas, inside template arguments and parameter lists, so
this uses a real CSV reader rather than cutting on the separator.  Column names
have moved between nsys versions; they are matched by substring, and fall back
to position, because a rename should not cost the whole summary.
"""
import csv
import os
import sys

path = sys.argv[1]
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 3
# What the profile covered, so a share can be read for what it is.  Passed in
# rather than inferred, because nothing in the CSV records it.
basis = sys.argv[3] if len(sys.argv) > 3 else ""
if not os.path.exists(path):
    sys.exit(0)

with open(path, newline="") as fh:
    rows = [r for r in csv.reader(fh) if r]
if len(rows) < 2:
    sys.exit(0)

head = [h.strip().lower() for h in rows[0]]


def col(*wants, default=None):
    for w in wants:
        for i, h in enumerate(head):
            if w in h:
                return i
    return default


i_pct = col("time (%)", "time(%)", "percent", default=0)
i_tot = col("total time", default=1)
i_ins = col("instances", "count", default=2)
i_nam = col("name", default=len(head) - 1)


def num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return 0.0


body = [r for r in rows[1:] if len(r) > max(i_pct, i_nam)]
# Sorted by total time, which is one quantity in one unit.
#
# This was `num(pct) or num(total)`, meaning a row whose percentage parsed as
# zero fell back to nanoseconds and was then compared against rows holding
# percentages. Twenty-two million nanoseconds beats thirty percent, so the
# three kernels reported as costliest were the three the CSV recorded as 0.0%,
# and the line beneath said the other 122 accounted for 99.5%.
body.sort(key=lambda r: num(r[i_tot]), reverse=True)

shown = body[:topk]
if not shown:
    sys.exit(0)

rest = sum(num(r[i_pct]) for r in body[topk:])
print("  [profile] top %d kernels by GPU time (of %d distinct)%s:"
      % (len(shown), len(body), (", " + basis) if basis else ""))
for n, r in enumerate(shown, 1):
    name = r[i_nam].strip()
    # Templated CUDA names run to hundreds of characters and the useful part is
    # at the front, so keep the front.
    if len(name) > 62:
        name = name[:59] + "..."
    ms = num(r[i_tot]) / 1e6
    print("  [profile]   %d. %5.1f%%  %9.1f ms  %8d x  %s"
          % (n, num(r[i_pct]), ms, int(num(r[i_ins])), name))
if rest > 0:
    print("  [profile]      %5.1f%%  the remaining %d" % (rest, len(body) - topk))

# A short profile names the right kernels and misstates their shares.  Context
# creation, first-touch allocation, library handle setup and the first AMG
# hierarchy build all happen once, so over a couple of steps they are divided by
# a couple rather than by a history, and setup work is over-weighted.  Worth one
# line, because the identities and the percentages here are trustworthy to
# different degrees and nothing else says so.
if basis.startswith("over ") and basis.endswith(" days"):
    try:
        days = float(basis[5:-5])
    except ValueError:
        days = 1e9
    if days <= 42:
        print("  [profile] short profile: the kernels named are the right ones,")
        print("  [profile] but one-time setup is spread over a few steps only,")
        print("  [profile] so its share is overstated.  PROF_STEPS raises it.")
