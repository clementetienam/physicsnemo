#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# backend_report.py -- HYPRE BoomerAMG, AmgX and Ginkgo (FP16), as one
# self-contained page.
#
# Every figure is inline SVG rather than a PNG.  That is deliberate: the page
# stays a single file with nothing beside it to go stale or be deleted by
# mistake, it is a few hundred kilobytes instead of several megabytes, and it
# stays sharp at any zoom.  The earlier comparison reports embedded hundreds of
# rasterised panels and ran to ten megabytes each.
import json
import os
import sys
import datetime

# --- palette ---------------------------------------------------------------
# Two series, one accent.  Distinguishable in greyscale and to the most common
# forms of colour blindness, which rules out the obvious red/green pairing.
C_HYPRE = "#5B8FF9"
C_AMGX = "#61D6A8"
C_OPM = "#B9A6FF"
C_GINKGO = "#F2A33A"
C_GINKGO32 = "#A77BD8"
C_GINKGO_ABJ = "#2BB5A0"
# Chart ink follows the page rather than being fixed, so the same SVG is
# legible in light and dark.  currentColor inherits body colour; the two
# greys are neutral enough to sit on either background.
C_INK = "currentColor"
C_MUTE = "rgba(127,127,127,.95)"
C_GRID = "rgba(127,127,127,.22)"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def load(root):
    """{study: {backend: [row, ...]}} from the tree backend_study.sh writes."""
    out = {}
    if not os.path.isdir(root):
        return out
    for study in sorted(os.listdir(root)):
        sdir = os.path.join(root, study)
        if not os.path.isdir(sdir):
            continue
        for backend in sorted(os.listdir(sdir)):
            j = os.path.join(sdir, backend, "scaling.json")
            if not os.path.isfile(j):
                continue
            try:
                rows = json.load(open(j, encoding="utf-8"))
            except Exception as e:                    # noqa: BLE001
                print("  skipping %s: %s" % (j, e))
                continue
            out.setdefault(study, {})[backend] = sorted(
                rows, key=lambda r: r.get("gpus", 0))
    return out


# --- charts ----------------------------------------------------------------
def bar_chart(groups, series, title, ylabel, fmt="%.1f", height=260):
    """Grouped bars.  groups = x labels; series = [(name, colour, [v, ...])]."""
    if not groups or not series:
        return ""
    W, H = 720, height
    pad_l, pad_b, pad_t, pad_r = 58, 42, 26, 12
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    vals = [v for _, _, vs in series for v in vs if v is not None]
    if not vals:
        return ""
    vmax = max(vals) * 1.18 or 1.0

    def y(v):
        return pad_t + ph - (v / vmax) * ph

    o = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" '
         'aria-label="%s" xmlns="http://www.w3.org/2000/svg" fill="currentColor">' % (W, H, esc(title))]
    # horizontal rules with labels
    for i in range(5):
        v = vmax * i / 4.0
        yy = y(v)
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                 'stroke-width="1"/>' % (pad_l, yy, W - pad_r, yy, C_GRID))
        o.append('<text x="%d" y="%.1f" font-size="11" fill="%s" '
                 'text-anchor="end">%s</text>'
                 % (pad_l - 8, yy + 4, C_MUTE, fmt % v))
    o.append('<text x="14" y="%d" font-size="11" fill="%s" transform="rotate(-90 14 %d)" '
             'text-anchor="middle">%s</text>'
             % (pad_t + ph / 2, C_MUTE, pad_t + ph / 2, esc(ylabel)))

    gw = pw / float(len(groups))
    n = len(series)
    bw = min(46.0, (gw * 0.68) / n)
    for gi, g in enumerate(groups):
        cx = pad_l + gw * (gi + 0.5)
        x0 = cx - (n * bw) / 2.0
        for si, (name, colour, vs) in enumerate(series):
            v = vs[gi] if gi < len(vs) else None
            if v is None:
                continue
            bx = x0 + si * bw
            o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'rx="3" fill="%s"><title>%s, %s: %s</title></rect>'
                     % (bx, y(v), bw - 4, max(0.0, pad_t + ph - y(v)),
                        colour, esc(name), esc(g), fmt % v))
            o.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="%s" '
                     'text-anchor="middle">%s</text>'
                     % (bx + (bw - 4) / 2.0, y(v) - 5, C_INK, fmt % v))
        o.append('<text x="%.1f" y="%d" font-size="12" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (cx, H - 14, C_INK, esc(g)))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
             % (pad_l, pad_t + ph, W - pad_r, pad_t + ph, C_MUTE))
    o.append("</svg>")
    return "".join(o)


def legend(series):
    bits = []
    for name, colour, _ in series:
        bits.append('<span class="key"><i style="background:%s"></i>%s</span>'
                    % (colour, esc(name)))
    return '<div class="legend">%s</div>' % "".join(bits)


def speedup_chart(groups, ranks, series, title):
    """Speed-up against that arm's own 1-GPU time, with the ideal line.

    groups are the axis labels ("2 GPUs"); ranks are the rank counts the
    ideal line is drawn from.  They are separate because the label is prose.
    """
    W, H = 720, 250
    pad_l, pad_b, pad_t, pad_r = 58, 42, 26, 12
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    ideal = [float(g) for g in ranks]
    vmax = max(ideal + [v for _, _, vs in series for v in vs if v]) * 1.15

    def y(v):
        return pad_t + ph - (v / vmax) * ph

    def x(i):
        return pad_l + pw * (i + 0.5) / float(len(groups))

    o = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="%s" '
         'xmlns="http://www.w3.org/2000/svg" fill="currentColor">' % (W, H, esc(title))]
    for i in range(5):
        v = vmax * i / 4.0
        yy = y(v)
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
                 % (pad_l, yy, W - pad_r, yy, C_GRID))
        o.append('<text x="%d" y="%.1f" font-size="11" fill="%s" '
                 'text-anchor="end">%.1fx</text>' % (pad_l - 8, yy + 4, C_MUTE, v))
    pts = " ".join("%.1f,%.1f" % (x(i), y(v)) for i, v in enumerate(ideal))
    o.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.5" '
             'stroke-dasharray="5 4"/>' % (pts, C_MUTE))
    o.append('<text x="%.1f" y="%.1f" font-size="11" fill="%s">ideal</text>'
             % (x(len(groups) - 1) - 34, y(ideal[-1]) - 8, C_MUTE))
    for name, colour, vs in series:
        pts = " ".join("%.1f,%.1f" % (x(i), y(v))
                       for i, v in enumerate(vs) if v)
        o.append('<polyline points="%s" fill="none" stroke="%s" '
                 'stroke-width="2.5"/>' % (pts, colour))
        for i, v in enumerate(vs):
            if not v:
                continue
            o.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"><title>%s at '
                     '%s GPU: %.2fx</title></circle>'
                     % (x(i), y(v), colour, esc(name), groups[i], v))
    for i, g in enumerate(groups):
        o.append('<text x="%.1f" y="%d" font-size="12" fill="%s" '
                 'text-anchor="middle">%s</text>' % (x(i), H - 14, C_INK, esc(g)))
    o.append("</svg>")
    return "".join(o)


# --- page ------------------------------------------------------------------
CSS = """
:root{--ink:#1a1d23;--mute:#6b7280;--line:#e5e7eb;--bg:#ffffff;--card:#fbfbfd;
      --accent:#5B8FF9;--good:#0f9960;--bad:#c23934}
@media (prefers-color-scheme:dark){
 :root{--ink:#e8eaed;--mute:#9aa0a6;--line:#2c3038;--bg:#14161a;--card:#1b1e24}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
.wrap{max-width:940px;margin:0 auto;padding:44px 24px 96px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:21px;margin:52px 0 4px;letter-spacing:-.01em}
h3{font-size:15px;margin:26px 0 8px;color:var(--mute);font-weight:600;
 text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--mute);margin:0 0 30px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
 padding:20px 22px;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--mute);font-weight:600;font-size:12px;text-transform:uppercase;
 letter-spacing:.05em}
tbody tr:last-child td{border-bottom:none}
.legend{display:flex;gap:18px;margin:2px 0 8px;font-size:13px;color:var(--mute)}
.key{display:flex;align-items:center;gap:7px}
.key i{width:11px;height:11px;border-radius:3px;display:inline-block}
.win{color:var(--good);font-weight:600}
.lose{color:var(--bad);font-weight:600}
.note{font-size:13.5px;color:var(--mute);margin:10px 0 0}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;
 background:rgba(127,127,127,.13);padding:1px 5px;border-radius:4px}
.miss{color:var(--mute);font-style:italic}
"""

STUDY_TITLE = {
    "bench3p": "46 x 112 x 22 &mdash; 113,344 cells, full schedule",
    "million": "200 x 200 x 25 &mdash; 1,000,000 cells, full schedule",
    "2m": "400 x 200 x 25 &mdash; 2,000,000 cells, full schedule",
    "4m": "400 x 400 x 25 &mdash; 4,000,000 cells, full schedule",
    "5m": "500 x 400 x 25 &mdash; 5,000,000 cells, full schedule",
    "spe10": "SPE10 model 2 &mdash; 60 x 220 x 85 = 1,122,000 cells, "
             "1000 days (decks/SPE10_MATCHED.DATA)",
}
STUDY_NOTE = {
    "bench3p": "The case with an OPM Flow twin, so all three can be compared "
               "on the same grid and schedule.",
    "million": "All ten report steps. A single step is the one with the "
               "opening transient and all the chopping in it, and none of the "
               "settled behaviour that follows.",
    "2m": "Halfway up the ladder, and the first size where the two backends diverge on how they scale rather than only on how fast they are.",
    "4m": "A million rows per GPU on four ranks &mdash; the point at which distributing this problem starts to be worth doing at all.",
    "5m": "The top of the ladder: 15 million unknowns, and 1.25M rows per GPU on four ranks. This is where the trend either holds or levels off.",
    "spe10": "The Tenth SPE Comparative Solution Project, model 2: the "
             "geostatistical field with a 10<sup>7</sup> permeability "
             "contrast that is the standard AMG stress test. Real rock "
             "rather than a synthetic refinement. Run with the Flow-matched "
             "deck and method (SWOF, CPRW, LINTOL 3e-2, "
             "RESTART 60, EMULATE 0), AmgX and HYPRE on one GPU. The HYPRE "
             "arm uses AMGFAST 0 and AMGREUSE 10: the deck's AMGFAST 1 and "
             "AMGREUSE 0 were chosen with AmgX, and for BoomerAMG they "
             "gave aggressive coarsening and a weak V-cycle (first 100 "
             "days: 49.5 s and 14,532 Krylov against 23.4 s and 5,656).",
}


def fmt(v, spec="%.1f"):
    return spec % v if isinstance(v, (int, float)) else "&mdash;"


# Set by main() so section() can test for the physics pages beside the JSONs.
ROOT_DIR = ""

# The arms, in display order: (key, legend name, table name, colour).  The
# keys are the directory names backend_study.sh writes, defined once in
# benchmark/backend_arms.sh.
ARMS = [
    ("hypre", "HYPRE BoomerAMG (FP32)", "HYPRE FP32", C_HYPRE),
    ("amgx", "AmgX (FP32)", "AmgX FP32", C_AMGX),
    ("ginkgo", "Ginkgo FP16 (1 GPU)", "Ginkgo FP16", C_GINKGO),
    ("ginkgo32", "Ginkgo FP32 (1 GPU)", "Ginkgo FP32", C_GINKGO32),
    ("ginkgo_abj", "Ginkgo FP16 + adaptive bJ (1 GPU)",
     "Ginkgo FP16 + adaptive bJ", C_GINKGO_ABJ),
]


def section(study, arms, opm_seconds):
    present = [(k, leg, tab, c, arms.get(k, [])) for k, leg, tab, c in ARMS
               if arms.get(k)]
    every = [r for *_, rows in present for r in rows]
    gpus = sorted({r["gpus"] for r in every})
    if not gpus:
        return ""
    labels = ["%d GPU" % g if g == 1 else "%d GPUs" % g for g in gpus]
    by_key = {k: rows for k, _, _, _, rows in present}
    hy, ax = by_key.get("hypre", []), by_key.get("amgx", [])

    def col(rows, key):
        by = {r["gpus"]: r.get(key) for r in rows}
        return [by.get(g) for g in gpus]

    title = STUDY_TITLE.get(study, esc(study))
    if study == "bench3p":
        # BENCH_GRID sets this grid, so name it from the run, not a constant.
        cells = max((r.get("cells", 0) for r in every), default=0)
        if cells:
            title = "BENCH3P &mdash; {:,} cells, full schedule".format(int(cells))
    o = ['<h2>%s</h2>' % title]
    o.append('<p class="sub">%s</p>' % STUDY_NOTE.get(study, ""))

    # --- wall clock ---------------------------------------------------------
    series = [(leg, c, col(rows, "wall")) for _, leg, _, c, rows in present]
    if study == "bench3p" and opm_seconds:
        series.append(("OPM Flow (GPU solver, 1 rank)", C_OPM,
                       [opm_seconds] + [None] * (len(gpus) - 1)))
    o.append('<div class="card"><h3>Wall clock</h3>')
    o.append(legend(series))
    o.append(bar_chart(labels, series, "wall clock", "seconds"))
    o.append('</div>')

    # --- the phase that never scaled ---------------------------------------
    ser2 = [(leg, c, col(rows, "setup")) for _, leg, _, c, rows in present]
    o.append('<div class="card"><h3>AMG setup &mdash; the phase that did not '
             'scale on HYPRE</h3>')
    o.append(legend(ser2))
    o.append(bar_chart(labels, ser2, "AMG setup", "seconds", "%.2f"))
    o.append('<p class="note">Building the hierarchy, summed over the run. '
             'On HYPRE this grew with rank count while the local problem '
             'shrank, which is why AmgX was tried at all.</p></div>')

    # --- speed-up -----------------------------------------------------------
    def spd(rows):
        by = {r["gpus"]: r.get("wall") for r in rows}
        base = by.get(1)
        if not base:
            return []
        return [(base / by[g]) if by.get(g) else None for g in gpus]

    ser3 = [(leg, c, spd(rows)) for k, leg, _, c, rows in present
            if k in ("hypre", "amgx")]
    if any(s[2] for s in ser3):
        o.append('<div class="card"><h3>Strong scaling, each against its own '
                 '1-GPU time</h3>')
        o.append(legend(ser3))
        o.append(speedup_chart(labels, gpus, ser3, "speed-up"))
        o.append('</div>')

    # --- the numbers --------------------------------------------------------
    o.append('<h3>Every number behind those charts</h3><table><thead><tr>'
             '<th>Backend</th><th>GPUs</th><th>cells/GPU</th><th>wall s</th>'
             '<th>AMG setup s</th><th>rest of linear s</th><th>non-linear s</th>'
             '<th>Newton</th><th>Krylov</th><th>rebuilds</th></tr></thead><tbody>')
    for _, _, name, _, rows in present:
        for r in rows:
            cells = r.get("cells", 0)
            o.append('<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td>'
                     '<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
                     % (name, r.get("gpus", 0),
                        "{:,}".format(int(cells / max(1, r.get("gpus", 1)))),
                        fmt(r.get("wall")), fmt(r.get("setup"), "%.2f"),
                        fmt(r.get("rest_of_linear"), "%.2f"),
                        fmt(r.get("nonlinear"), "%.2f"),
                        r.get("newton", "&mdash;"), r.get("krylov", "&mdash;"),
                        r.get("rebuilds", "&mdash;")))
    o.append('</tbody></table>')

    # --- the physics reports, if build_physics rendered them ----------------
    # A speed-up is only worth quoting if the answer did not move, and that is
    # a different report: well curves, material balance and the per-step
    # pressure and saturation panels, each backend against OPM Flow.
    rel = "physics.html"
    if os.path.isfile(os.path.join(ROOT_DIR, study, rel)):
        o.append('<div class="card"><h3>Does it give the same answer?</h3>'
                 '<p style="margin:0 0 10px"><a href="%s/%s"><strong>'
                 'Well curves, material balance and the panel series</strong></a> '
                 '&mdash; every backend and OPM Flow on one page: WOPR, '
                 'WWPR, WWCT, WGOR, WBHP, water in place, and every report '
                 'step of pressure and saturation.</p>'
                 '<p class="note">Rendered from the 1-GPU run of each '
                 'backend, so the fields carry no domain decomposition to '
                 'argue about, and all are set against the same OPM Flow '
                 'run.</p></div>' % (study, rel))

    # --- verdict ------------------------------------------------------------
    def one(rows):
        return next((r for r in rows if r["gpus"] == 1 and r.get("wall")), None)

    def versus(a_name, a, b_name, b, what):
        ratio = b["wall"] / a["wall"]
        cls = "win" if ratio > 1.02 else ("lose" if ratio < 0.98 else "")
        return ('<li>%s: on 1 GPU %s is <span class="%s">%.2fx %s</span> than '
                '%s (%.1f s against %.1f s; Krylov %s against %s).</li>'
                % (what, a_name, cls, ratio if ratio >= 1 else 1 / ratio,
                   "faster" if ratio >= 1 else "slower", b_name,
                   a["wall"], b["wall"], a.get("krylov", "?"),
                   b.get("krylov", "?")))

    lines = []
    for g in gpus:
        h = next((r for r in hy if r["gpus"] == g), None)
        a = next((r for r in ax if r["gpus"] == g), None)
        if not (h and a and h.get("wall") and a.get("wall")):
            continue
        ratio = h["wall"] / a["wall"]
        cls = "win" if ratio > 1.02 else ("lose" if ratio < 0.98 else "")
        verb = "faster" if ratio >= 1 else "slower"
        lines.append('<li>At %d GPU%s AmgX is <span class="%s">%.2fx %s</span> '
                     'than HYPRE (%.1f s against %.1f s).</li>'
                     % (g, "" if g == 1 else "s", cls,
                        ratio if ratio >= 1 else 1 / ratio, verb,
                        a["wall"], h["wall"]))
    g16 = one(by_key.get("ginkgo", []))
    g32 = one(by_key.get("ginkgo32", []))
    gab = one(by_key.get("ginkgo_abj", []))
    # The two controlled comparisons first: each changes exactly one thing.
    if g16 and g32:
        lines.append(versus("Ginkgo FP16", g16, "Ginkgo FP32", g32,
                            "AMG precision alone"))
    if gab and g16:
        lines.append(versus("Ginkgo FP16 + adaptive block-Jacobi", gab,
                            "Ginkgo FP16 with the built-in FP32 block-Jacobi",
                            g16, "Stage-2 smoother alone"))
    # Then the library comparisons, which change the hierarchy as well.
    best = min((x for x in (g16, g32, gab) if x), key=lambda r: r["wall"],
               default=None)
    if best:
        nm = ("Ginkgo FP16" if best is g16 else
              "Ginkgo FP32" if best is g32 else
              "Ginkgo FP16 + adaptive block-Jacobi")
        for other, rows in (("AmgX FP32", ax), ("HYPRE FP32", hy)):
            r1 = one(rows)
            if r1:
                lines.append(versus(nm, best, other, r1,
                                    "Best Ginkgo arm against %s" % other))
    if lines:
        o.append('<div class="card"><h3>Verdict</h3><ul>%s</ul></div>'
                 % "".join(lines))
    return "".join(o)


def main(argv):
    root = argv[1] if len(argv) > 1 else "benchmark/backend_study"
    opm = None
    if "--opm" in argv:
        opm = float(argv[argv.index("--opm") + 1])
    global ROOT_DIR
    ROOT_DIR = root
    data = load(root)
    if not data:
        print("no scaling.json found under %s -- run backend_study.sh first"
              % root)
        return 1

    body = []
    order = ("bench3p", "million", "2m", "4m", "5m", "spe10")
    for study in order:
        if study in data:
            body.append(section(study, data[study], opm))
    for study in data:
        if study not in order:
            body.append(section(study, data[study], opm))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # The GPU this ran on, not a fixed name: the same report is built on GB200
    # and on VR72.
    try:
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.splitlines()[0].strip()
    except Exception:                                  # noqa: BLE001
        gpu = ""
    gpu = gpu or "an NVIDIA GPU"
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage-1 multigrid ablation &mdash; NVRS-CUDA</title>
<style>%s</style></head><body><div class="wrap">
<h1>Preconditioner ablation: HYPRE and AmgX</h1>
<p class="sub">NVRS-CUDA on %s &middot; generated %s</p>
<div class="card">
<p style="margin:0">Only the pressure multigrid differs between the two arms.
The outer Krylov is FP64 FGMRES over a df64-emulated FP64 operator, the
quasi-IMPES weights that form the pressure system are FP64, and the second CPR
stage is the same block-Jacobi correction in both arms. HYPRE and AmgX run
their multigrid in FP32. The three Ginkgo arms (FP16, FP32 and FP16 with
adaptive-precision block-Jacobi) are compared on Norne only, in
compare_NORNE.html: on these cases they were far slower.</p>
<p class="note">Select with <code>LINSOLVER AMGX</code> or
<code>LINSOLVER HYPRE</code>. AmgX is the default.</p>
</div>
%s
</div></body></html>""" % (CSS, gpu, stamp, "".join(body))

    out = os.path.join(root, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote %s  (%.0f kB)" % (out, len(html) / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
