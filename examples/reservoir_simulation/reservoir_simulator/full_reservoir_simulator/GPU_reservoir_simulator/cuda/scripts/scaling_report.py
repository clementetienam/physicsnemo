#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Attribute a strong-scaling result to the phase that actually lost the time.

Given the output tree from run_scaling.sh, this reads each rank count's
meta.json and run log and prints one table.  The point is not the speed-up
column, which only says that something went wrong; it is the per-phase columns,
which say what:

  * iteration counts flat        the preconditioner is rank-invariant and every
                                 second lost is overhead, not algorithm
  * iteration counts climbing    BoomerAMG coarsened differently; the loss is
                                 algorithmic and more ranks cannot fix it
  * AMG setup growing            parallel coarsening, which is latency-bound in
                                 the coarse levels and barely divides
  * halo seconds growing         the exchange itself, which host staging makes
                                 several times more expensive than it needs
  * non-linear part shrinking    assembly and wells are parallelising correctly,
                                 so the problem is confined to the linear solve
"""
import glob
import json
import os
import re
import sys


def read_meta(rundir):
    for p in sorted(glob.glob(os.path.join(rundir, "**", "meta.json"),
                              recursive=True)):
        try:
            with open(p) as fh:
                return json.load(fh) or {}
        except (OSError, ValueError):
            return {}
    return {}


def read_log(rundir):
    """Pull the lines the simulator prints that are not in meta.json."""
    out = {}
    p = os.path.join(rundir, "run.log")
    try:
        s = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return out
    m = re.search(r"Halo\s*:\s*(\d+) exchanges,\s*([\d.]+) s.*?([\d.]+) GB", s)
    if m:
        out["halo_calls"] = int(m.group(1))
        out["halo_seconds"] = float(m.group(2))
        out["halo_gb"] = float(m.group(3))
    m = re.search(r"of which ([\d.]+) s was cudaMemcpy staging", s)
    if m:
        out["halo_stage_seconds"] = float(m.group(1))
    out["host_staged"] = "host-staged" in s
    m = re.search(r"coarsening stops at (\d+) rows", s)
    if m:
        out["amg_max_coarse"] = int(m.group(1))
    m = re.search(r"Halo\s*:\s*(device pointers|staged through)", s)
    if m:
        out["halo_mode"] = ("device" if "device" in m.group(1) else "host")
    return out


def num(m, k, d=0.0):
    v = m.get(k)
    return float(v) if isinstance(v, (int, float)) else d


def isum(m, k):
    v = m.get(k)
    return int(sum(v)) if isinstance(v, list) and v else 0


root = sys.argv[1] if len(sys.argv) > 1 else "."
rows = []
for d in sorted(glob.glob(os.path.join(root, "*gpu")),
                key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
    g = int(re.sub(r"\D", "", os.path.basename(d)) or 0)
    m = read_meta(d)
    if not m:
        continue
    log = read_log(d)
    wall = num(m, "wall_seconds")
    lin = num(m, "linear_seconds")
    setup = num(m, "amg_setup_seconds")
    rows.append(dict(
        gpus=g, wall=wall, linear=lin, setup=setup,
        rest_of_linear=lin - setup,
        nonlinear=wall - lin,
        newton=isum(m, "newton_iters"), krylov=isum(m, "linear_iters"),
        rebuilds=int(num(m, "amg_setups")), reuses=int(num(m, "amg_reuses")),
        cells=int(num(m, "nx") * num(m, "ny") * num(m, "nz")),
        **log))

if not rows:
    print("no completed runs found under %s" % root)
    sys.exit(1)

b = rows[0]
W = ("%-5s %11s %8s %8s %9s %9s %9s %8s %8s %8s" %
     ("GPUs", "cells/GPU", "wall", "speedup", "AMGsetup", "restLinr",
      "nonlinr", "halo s", "Newton", "Krylov"))
print()
print(W)
print("-" * len(W))
for r in rows:
    print("%-5d %11s %8.1f %8.2fx %9.1f %9.1f %9.1f %8s %8d %8d" % (
        r["gpus"],
        "{:,}".format(r["cells"] // r["gpus"]) if r["cells"] else "-",
        r["wall"], (b["wall"] / r["wall"]) if r["wall"] else 0,
        r["setup"], r["rest_of_linear"], r["nonlinear"],
        ("%.1f" % r["halo_seconds"]) if "halo_seconds" in r else "-",
        r["newton"], r["krylov"]))

print()
print("Scaling of each phase against 1 GPU (lower is better; 1.00 = no change):")
print("%-5s %10s %10s %10s %10s %10s" %
      ("GPUs", "AMGsetup", "restLinr", "nonlinr", "halo", "Krylov its"))
for r in rows[1:]:
    def rat(k):
        return (r[k] / b[k]) if b.get(k) else 0.0
    print("%-5d %10.2f %10.2f %10.2f %10s %10.3f" % (
        r["gpus"], rat("setup"), rat("rest_of_linear"), rat("nonlinear"),
        ("%.2f" % (r["halo_seconds"] / b["halo_seconds"]))
        if b.get("halo_seconds") else "-",
        (r["krylov"] / b["krylov"]) if b["krylov"] else 0))

# --- say what it means -----------------------------------------------------
print()
last = rows[-1]
notes = []
if b["krylov"] and last["krylov"]:
    drift = abs(last["krylov"] / b["krylov"] - 1.0)
    if drift <= 0.15:
        notes.append(
            "Krylov counts held to within %.0f%%, so the preconditioner is "
            "rank-invariant: every second lost is overhead, not algorithm."
            % (100 * drift))
    else:
        notes.append(
            "Krylov counts moved %.0f%%, so BoomerAMG is coarsening differently "
            "on %d ranks. That part is algorithmic." % (100 * drift, last["gpus"]))
if b["nonlinear"] and last["nonlinear"] / b["nonlinear"] < 1.0:
    notes.append(
        "The non-linear part (assembly, wells, PVT) got FASTER, %.1f s to "
        "%.1f s, so the local work is parallelising correctly and the loss is "
        "confined to the linear solve."
        % (b["nonlinear"], last["nonlinear"]))
if b["setup"] and last["setup"] / b["setup"] > 1.3:
    notes.append(
        "AMG setup grew %.1fx for %s rebuilds. Parallel coarsening is a chain "
        "of small per-level exchanges and global reductions whose cost barely "
        "divides; raising AMGMAXCOARSE stops the coarsening earlier and removes "
        "the most latency-bound levels."
        % (last["setup"] / b["setup"],
           "the same number of" if abs(last["rebuilds"] - b["rebuilds"])
           <= 0.05 * max(b["rebuilds"], 1) else "a different number of"))
if last.get("halo_mode") == "host":
    notes.append(
        "The halo exchange is staged through host memory: MPI is not reporting "
        "CUDA support. That is four blocking copies per exchange, twice per "
        "Krylov iteration. Check the MPI build, or force it with "
        "NVRS_CUDA_AWARE_MPI=1 if you know it is capable.")
for n in notes:
    print("  * " + n)

out = os.path.join(root, "scaling.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=2)
print()
print("wrote %s" % out)
