#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
"""Reference implementation of the corner-point grid, in NumPy.

This mirrors ``src/cpgrid.cpp`` step for step -- the same corner numbering, the
same tetrahedral decomposition, the same two-point transmissibility, the same
juxtaposition rule, the same deck operators.  It exists for two reasons.

The first is that a geometry kernel is easy to get subtly wrong and hard to
debug inside a GPU simulator.  Having an independent implementation of the same
mathematics, written in a different language against the same specification,
turns "the answer looks plausible" into "two implementations agree or one of
them is wrong".

The second is that it produces the reference numbers -- active cell count, pore
volume, connection counts, transmissibility extrema -- that the C++ tests then
assert.  Those numbers have to come from somewhere, and deriving them here,
from the deck itself, is better than copying them out of another simulator's
output and hoping the conventions matched.

Usage
-----
    python cpgrid_ref.py NORNE_ATW2013.DATA [--json out.json] [--npz out.npz]

A directory may be given instead of a deck, in which case the loose Norne
includes in it are read directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

# Darcy constant: METRIC (mD, m, cP, bar, m3/day) and FIELD (mD, ft, cP, psi).
CDARCY = {"METRIC": 0.0085267014, "FIELD": 0.00112712}

# Face corner loops, identical to kFaceCorner[6][4] in cpgrid.cpp.
# Corner c = (a,b,cc): a along I, b along J, cc along K; 0-3 top, 4-7 bottom.
FACE_CORNERS = np.array(
    [
        [0, 2, 6, 4],  # 0  I-
        [1, 5, 7, 3],  # 1  I+
        [0, 4, 5, 1],  # 2  J-
        [2, 3, 7, 6],  # 3  J+
        [0, 1, 3, 2],  # 4  K-  (top, shallower)
        [4, 6, 7, 5],  # 5  K+  (bottom)
    ],
    dtype=np.int64,
)

ARRAY_KEYWORDS = {
    "PORO", "PERMX", "PERMY", "PERMZ", "NTG", "ACTNUM",
    "MULTX", "MULTY", "MULTZ", "MULTPV", "MULTX-", "MULTY-", "MULTZ-",
    "TOPS", "DX", "DY", "DZ",
    "SATNUM", "EQLNUM", "FIPNUM", "PVTNUM", "FLUXNUM", "MULTNUM",
    "ROCKNUM", "IMBNUM", "ENDNUM",
    "SWL", "SWU", "SWCR", "SGL", "SGU", "SGCR",
    "SOWCR", "SOGCR", "ISGCR", "ISWCR", "ISGU", "ISWL",
    "KRW", "KRO", "KRG", "PCW", "PCG", "SWATINIT",
}

ONE_DEFAULT = {"NTG", "ACTNUM", "SATNUM", "EQLNUM", "FIPNUM", "PVTNUM",
               "FLUXNUM", "MULTNUM", "ROCKNUM", "IMBNUM", "ENDNUM"}

TERMINATING = {"PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE"}

NO_DATA = {"ECHO", "NOECHO", "GRID", "EDIT", "RUNSPEC", "OIL", "WATER", "GAS",
           "DISGAS", "VAPOIL", "NEWTRAN", "GRIDFILE", "INIT", "NOGGF",
           "METRIC", "FIELD"}


def default_value(name):
    if name in ONE_DEFAULT or name.startswith("MULT"):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Tokenizer -- the same grammar the C++ reader implements
# ---------------------------------------------------------------------------
class Tok:
    _COMMENT = re.compile(r"--[^\n]*")

    def __init__(self, path):
        with open(path, "r", errors="replace") as f:
            self.s = self._COMMENT.sub("", f.read())
        self.p = 0
        self.n = len(self.s)

    def skip(self):
        while self.p < self.n and self.s[self.p].isspace():
            self.p += 1

    def eof(self):
        self.skip()
        return self.p >= self.n

    def at_slash(self):
        self.skip()
        return self.p < self.n and self.s[self.p] == "/"

    def _consume_slash(self):
        """Step over a record terminator and the free text that may follow it.

        A deck is allowed to annotate a record after its closing slash --
        ``'PERMZ' 0.64  1 46 1 112 13 13 /  Tofte 2.1.3`` is one of Norne's --
        and everything from the slash to the end of that line is a comment.
        Treating it as data is how a reader ends up trying to parse a formation
        name as a number.
        """
        self.p += 1
        while self.p < self.n and self.s[self.p] != "\n":
            self.p += 1

    def eat_slash(self):
        if self.at_slash():
            self._consume_slash()

    def next(self):
        self.skip()
        if self.p >= self.n:
            return ""
        if self.s[self.p] == "'":
            e = self.s.find("'", self.p + 1)
            if e < 0:
                e = self.n
            t = self.s[self.p + 1 : e]
            self.p = e + 1
            return t
        if self.s[self.p] == "/":
            self._consume_slash()
            return "/"
        st = self.p
        while self.p < self.n and not self.s[self.p].isspace() and self.s[self.p] != "/":
            self.p += 1
        return self.s[st : self.p]

    def read_values(self):
        """Values up to the terminating slash, with ``N*value`` expanded."""
        out = []
        while True:
            self.skip()
            if self.p >= self.n:
                break
            if self.s[self.p] == "/":
                self._consume_slash()
                break
            st = self.p
            while self.p < self.n and not self.s[self.p].isspace() and self.s[self.p] != "/":
                self.p += 1
            t = self.s[st : self.p]
            if not t:
                continue
            star = t.find("*")
            if star >= 0:
                try:
                    rep = int(t[:star])
                except ValueError:
                    continue
                v = float(t[star + 1 :]) if star + 1 < len(t) else 0.0
                out.append(np.full(rep, v))
            else:
                if not (t[0].isdigit() or t[0] in "+-."):
                    continue          # a trailing flag such as SPECGRID's 'F'
                out.append(np.array([float(t)]))
        return np.concatenate(out) if out else np.zeros(0)

    def skip_to_slash(self):
        while True:
            self.skip()
            if self.p >= self.n:
                return
            if self.s[self.p] == "/":
                self._consume_slash()
                return
            self.next()


# ---------------------------------------------------------------------------
# Deck reader
# ---------------------------------------------------------------------------
class Deck:
    def __init__(self):
        self.nx = self.ny = self.nz = 0
        self.coord = None
        self.zcorn = None
        self.arrays = {}
        self.faults = {}
        self.multflt = {}
        self.regmult = []
        self.equil = []
        self.minpv = 0.0
        self.units = "METRIC"
        self.ignored = set()
        self.box = None

    # -- helpers ---------------------------------------------------------
    @property
    def ng(self):
        return self.nx * self.ny * self.nz

    def arr(self, name):
        if name not in self.arrays:
            self.arrays[name] = np.full(self.ng, default_value(name))
        return self.arrays[name]

    def reset_box(self):
        self.box = (0, self.nx - 1, 0, self.ny - 1, 0, self.nz - 1)

    def box_mask_idx(self, b):
        i1, i2, j1, j2, k1, k2 = b
        i1, j1, k1 = max(i1, 0), max(j1, 0), max(k1, 0)
        i2 = min(i2, self.nx - 1)
        j2 = min(j2, self.ny - 1)
        k2 = min(k2, self.nz - 1)
        if i1 > i2 or j1 > j2 or k1 > k2:
            return None
        kk, jj, ii = np.meshgrid(
            np.arange(k1, k2 + 1), np.arange(j1, j2 + 1), np.arange(i1, i2 + 1),
            indexing="ij")
        return ((kk * self.ny + jj) * self.nx + ii).ravel()

    # -- parsing ---------------------------------------------------------
    def read(self, path):
        tk = Tok(path)
        d = os.path.dirname(os.path.abspath(path))

        while not tk.eof():
            kw = tk.next()
            if not kw:
                break
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]{0,7}", kw):
                continue
            if kw in TERMINATING:
                break

            if kw in ("SPECGRID", "DIMENS"):
                v = tk.read_values()
                self.nx, self.ny, self.nz = (int(x) for x in v[:3])
                self.reset_box()

            elif kw == "COORD":
                self.coord = tk.read_values()

            elif kw == "ZCORN":
                self.zcorn = tk.read_values()

            elif kw in ARRAY_KEYWORDS:
                v = tk.read_values()
                if v.size != self.ng:
                    raise ValueError(
                        f"{kw}: expected {self.ng} values, read {v.size}")
                self.arrays[kw] = v

            elif kw == "BOX":
                v = tk.read_values()
                if v.size >= 6:
                    self.box = (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1,
                                int(v[3]) - 1, int(v[4]) - 1, int(v[5]) - 1)

            elif kw == "ENDBOX":
                self.reset_box()

            elif kw == "COPY":
                while not tk.at_slash() and not tk.eof():
                    src = tk.next()
                    if not src or src == "/":
                        break
                    dst = tk.next()
                    v = tk.read_values()
                    b = (tuple(int(x) - 1 for x in v[:6]) if v.size >= 6
                         else self.box)
                    if src not in self.arrays:
                        self.ignored.add(f"COPY from undefined {src}")
                        continue
                    idx = self.box_mask_idx(
                        (b[0], b[1], b[2], b[3], b[4], b[5]))
                    if idx is None:
                        continue
                    self.arr(dst)[idx] = self.arrays[src][idx]
                tk.eat_slash()

            elif kw in ("EQUALS", "MULTIPLY", "ADD"):
                while not tk.at_slash() and not tk.eof():
                    name = tk.next()
                    if not name or name == "/":
                        break
                    v = tk.read_values()
                    if v.size == 0:
                        continue
                    val = v[0]
                    b = (tuple(int(x) - 1 for x in v[1:7]) if v.size >= 7
                         else self.box)
                    idx = self.box_mask_idx(b)
                    if idx is None:
                        continue
                    a = self.arr(name)
                    if kw == "EQUALS":
                        a[idx] = val
                    elif kw == "MULTIPLY":
                        a[idx] *= val
                    else:
                        a[idx] += val
                tk.eat_slash()

            elif kw == "EQUIL":
                while True:
                    if tk.at_records_end():
                        tk.eat_slash()
                        break
                    r = tk.read_record()
                    if not r:
                        break
                    def num(i, d=0.0):
                        if i >= len(r) or "*" in r[i]:
                            return d
                        try:
                            return float(r[i])
                        except ValueError:
                            return d
                    self.equil.append([num(i) for i in range(6)])

            elif kw in ("MINPV", "MINPVV"):
                v = tk.read_values()
                if v.size:
                    self.minpv = float(v[0])

            elif kw == "FAULTS":
                while True:
                    if tk.at_slash():
                        tk.eat_slash()
                        break
                    if tk.eof():
                        break
                    name = tk.next()
                    if not name or name == "/":
                        break
                    nums = [int(tk.next()) for _ in range(6)]
                    face = tk.next().strip().upper()
                    tk.skip_to_slash()
                    f = {"X": 1, "I": 1, "Y": 3, "J": 3, "Z": 5, "K": 5}.get(
                        face[0] if face else "")
                    if f is None:
                        continue
                    if len(face) > 1 and face[1] == "-":
                        f -= 1
                    self.faults.setdefault(name, []).append(tuple(nums) + (f,))

            elif kw == "MULTFLT":
                while True:
                    if tk.at_slash():
                        tk.eat_slash()
                        break
                    if tk.eof():
                        break
                    name = tk.next()
                    if not name or name == "/":
                        break
                    m = float(tk.next())
                    tk.skip_to_slash()
                    self.multflt[name] = m

            elif kw == "MULTREGT":
                while True:
                    if tk.at_slash():
                        tk.eat_slash()
                        break
                    if tk.eof():
                        break
                    t1 = tk.next()
                    if not t1 or t1 == "/":
                        break
                    r1 = -1 if t1 == "*" else int(float(t1))
                    t2 = tk.next()
                    r2 = -1 if t2 == "*" else int(float(t2))
                    mult = float(tk.next())
                    dirs, nnc, reg = "", "", ""
                    while True:
                        if tk.at_slash():
                            tk.eat_slash()
                            break
                        if tk.eof():
                            break
                        t = tk.next()
                        if not t or t == "/":
                            break
                        if "*" in t:
                            continue          # an `N*` run of defaulted items
                        if not dirs and t in ("X", "Y", "Z", "XY", "XZ", "YZ",
                                              "XYZ"):
                            dirs = t
                        elif not nnc and t in ("ALL", "NNC", "NONNC"):
                            nnc = t
                        else:
                            reg = t
                    bits = 7
                    if dirs:
                        bits = (("X" in dirs) * 1 + ("Y" in dirs) * 2
                                + ("Z" in dirs) * 4)
                    # The last item is a single-letter code for which region
                    # array to read -- 'F' is FLUXNUM, 'M' is MULTNUM -- not an
                    # array name.  Taking it literally fails silently: no array
                    # is called "F", every record is skipped, and a deck that
                    # seals twenty flux regions runs as if fully open.
                    reg = {"F": "FLUXNUM", "M": "MULTNUM"}.get(reg, reg)
                    self.regmult.append(
                        (r1, r2, mult, bits, reg or "MULTNUM"))

            elif kw == "INCLUDE":
                inc = tk.next()
                tk.skip_to_slash()
                if inc.startswith("./"):
                    inc = inc[2:]
                self.read(os.path.join(d, inc))

            elif kw in NO_DATA:
                if kw in ("METRIC", "FIELD"):
                    self.units = kw

            else:
                self.ignored.add(kw)
                tk.skip_to_slash()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class RefGrid:
    def __init__(self, deck):
        self.d = deck
        nx, ny, nz = deck.nx, deck.ny, deck.nz
        self.nx, self.ny, self.nz = nx, ny, nz
        self.coord = deck.coord.reshape((ny + 1, nx + 1, 6))
        self.zcorn = deck.zcorn.reshape((2 * nz, 2 * ny, 2 * nx))
        self.C = CDARCY[deck.units]

    def corners(self):
        """All eight corners of every cell: shape (nz, ny, nx, 8, 3)."""
        nx, ny, nz = self.nx, self.ny, self.nz
        xyz = np.empty((nz, ny, nx, 8, 3))
        ii, jj = np.arange(nx), np.arange(ny)
        for c in range(8):
            a, b, cc = c & 1, (c >> 1) & 1, (c >> 2) & 1
            z = self.zcorn[2 * np.arange(nz) + cc][:, 2 * jj + b][:, :, 2 * ii + a]
            pil = self.coord[np.ix_(jj + b, ii + a)]
            xt, yt, zt = pil[..., 0], pil[..., 1], pil[..., 2]
            xb, yb, zb = pil[..., 3], pil[..., 4], pil[..., 5]
            dzp = zb - zt
            safe = np.where(np.abs(dzp) > 1e-12, dzp, 1.0)
            t = np.where(np.abs(dzp) > 1e-12, (z - zt) / safe, 0.0)
            xyz[..., c, 0] = xt + t * (xb - xt)
            xyz[..., c, 1] = yt + t * (yb - yt)
            xyz[..., c, 2] = z
        return xyz

    def geometry(self, v):
        """Volume, centroid and thickness by tetrahedral decomposition."""
        cc = v.mean(axis=3)
        vol = np.zeros(v.shape[:3])
        cen = np.zeros(v.shape[:3] + (3,))
        for f in range(6):
            fq = v[:, :, :, FACE_CORNERS[f], :]
            fc = fq.mean(axis=3)
            for e in range(4):
                p0 = fq[:, :, :, e, :]
                p1 = fq[:, :, :, (e + 1) & 3, :]
                tv = np.abs(np.einsum("...i,...i->...", p0 - cc,
                                      np.cross(p1 - cc, fc - cc))) / 6.0
                vol += tv
                cen += (cc + p0 + p1 + fc) * 0.25 * tv[..., None]
        safe = np.where(vol > 0, vol, 1.0)[..., None]
        cen = np.where(vol[..., None] > 0, cen / safe, cc)
        zt = v[:, :, :, FACE_CORNERS[4], 2].mean(axis=3)
        zb = v[:, :, :, FACE_CORNERS[5], 2].mean(axis=3)
        return vol, cen, np.abs(zb - zt)

    def half_trans(self, v, cen, face, kx, ky, kz):
        fq = v[:, :, :, FACE_CORNERS[face], :]
        fc = fq.mean(axis=3)
        area = np.zeros(v.shape[:3] + (3,))
        for e in range(4):
            p0 = fq[:, :, :, e, :]
            p1 = fq[:, :, :, (e + 1) & 3, :]
            area += 0.5 * np.cross(p0 - fc, p1 - fc)
        d = fc - cen
        s = np.sign(np.einsum("...i,...i->...", area, d))
        area = area * np.where(s == 0, 1.0, s)[..., None]
        d2 = np.einsum("...i,...i->...", d, d)
        akd = (kx * area[..., 0] * d[..., 0]
               + ky * area[..., 1] * d[..., 1]
               + kz * area[..., 2] * d[..., 2])
        return np.where((d2 > 0) & (akd > 0),
                        self.C * akd / np.where(d2 > 0, d2, 1.0), 0.0)

    def face_z_range(self, v, face):
        idx = FACE_CORNERS[face]
        zt = v[:, :, :, idx[idx < 4], 2].mean(axis=3)
        zb = v[:, :, :, idx[idx >= 4], 2].mean(axis=3)
        return np.minimum(zt, zb), np.maximum(zt, zb)


def _series(T1, T2):
    """Harmonic (series) combination, zero where either side is zero."""
    ok = (T1 > 0) & (T2 > 0)
    a = np.where(ok, T1, np.inf)
    b = np.where(ok, T2, np.inf)
    return np.where(ok, 1.0 / (1.0 / a + 1.0 / b), 0.0)


# ---------------------------------------------------------------------------
def build(path, verbose=True):
    t0 = time.time()
    deck = Deck()
    if os.path.isdir(path):
        # The loose-file layout: read the grid, then each property include.
        deck.read(os.path.join(path, "IRAP_1005.GRDECL"))
        for f in ("ACTNUM_0704.prop", "PORO_0704.prop", "PERM_0704.prop",
                  "NTG_0704.prop", "FAULT_JUN_05.INC",
                  "FAULTMULT_AUG-2006.INC"):
            p = os.path.join(path, f)
            if os.path.exists(p):
                deck.read(p)
    else:
        deck.read(path)

    nx, ny, nz = deck.nx, deck.ny, deck.nz
    shp = (nz, ny, nx)
    if verbose:
        print(f"  parsed in {time.time()-t0:.1f}s: {nx}x{ny}x{nz}, "
              f"{len(deck.faults)} faults, "
              f"{sum(len(v) for v in deck.faults.values())} fault-face records, "
              f"MINPV {deck.minpv:g}")
        if deck.ignored:
            ig = sorted(deck.ignored)
            print(f"  not implemented: {', '.join(ig[:14])}"
                  + (f" (+{len(ig)-14} more)" if len(ig) > 14 else ""))

    g = RefGrid(deck)
    v = g.corners()
    vol, cen, dzc = g.geometry(v)

    def A(name, dflt=None):
        if name in deck.arrays:
            return deck.arrays[name].reshape(shp)
        if dflt is None:
            return None
        return np.full(shp, dflt)

    poro = A("PORO", 0.0)
    ntg = A("NTG", 1.0)
    kx = A("PERMX", 0.0)
    ky = A("PERMY")
    kz = A("PERMZ")
    if ky is None:
        ky = kx
    if kz is None:
        kz = kx
    mpv = A("MULTPV", 1.0)
    act = A("ACTNUM", 1.0) != 0

    porv = vol * poro * ntg * mpv
    n_inactive = int((~act).sum())

    # MINPV: drop cells with too little pore volume to be worth a row.
    # minpv_cut records WHICH inactive cells are a genuine pinch-out, as
    # opposed to a cell ACTNUM already excluded -- the vertical pass below
    # bridges the first and must stop at the second.  See the C++
    # build_connections() for why: on Norne there are 138 times as many plain
    # ACTNUM=0 cells as MINPV cuts, and treating all of them as bridgeable
    # turns however much non-reservoir rock sits between two active cells
    # into a connection as strong as if they touched.
    n_minpv = 0
    minpv_cut = np.zeros(shp, dtype=bool)
    if deck.minpv > 0:
        drop = act & (porv < deck.minpv)
        n_minpv = int(drop.sum())
        minpv_cut = drop
        act = act & ~drop

    n_active = int(act.sum())
    aidx = np.full(shp, -1, dtype=np.int64)
    aidx[act] = np.arange(n_active)

    if verbose:
        print(f"  geometry in {time.time()-t0:.1f}s: {n_active} active "
              f"({n_inactive} inactive, {n_minpv} below MINPV)")

    # ---- half-transmissibilities on every face ---------------------------
    ht = {}
    for f in (0, 1, 2, 3):
        ht[f] = g.half_trans(v, cen, f, kx * ntg, ky * ntg, kz)
    for f in (4, 5):
        ht[f] = g.half_trans(v, cen, f, kx, ky, kz)

    mx = A("MULTX", 1.0)
    my = A("MULTY", 1.0)
    mz = A("MULTZ", 1.0)
    mxm = A("MULTX-", 1.0)
    mym = A("MULTY-", 1.0)
    mzm = A("MULTZ-", 1.0)

    conns = []

    # ---- lateral connections with juxtaposition --------------------------
    for d, (di, dj, fpos, fneg, mpos, mneg) in enumerate(
            ((1, 0, 1, 0, mx, mxm), (0, 1, 3, 2, my, mym))):
        t1, b1 = g.face_z_range(v, fpos)
        t2, b2 = g.face_z_range(v, fneg)
        sl_a = (slice(None), slice(0, ny - dj), slice(0, nx - di))
        sl_b = (slice(None), slice(dj, None), slice(di, None))

        A_t, A_b = t1[sl_a], b1[sl_a]
        B_t, B_b = t2[sl_b], b2[sl_b]

        ov = (np.minimum(A_b[:, None], B_b[None, :])
              - np.maximum(A_t[:, None], B_t[None, :]))
        h1 = (A_b - A_t)[:, None]
        h2 = (B_b - B_t)[None, :]
        good = ((ov > 1e-9) & (h1 > 0) & (h2 > 0)
                & act[sl_a][:, None] & act[sl_b][None, :])

        T1 = np.where(good, ht[fpos][sl_a][:, None] * ov / np.where(h1 > 0, h1, 1), 0.0)
        T2 = np.where(good, ht[fneg][sl_b][None, :] * ov / np.where(h2 > 0, h2, 1), 0.0)
        T = _series(T1, T2) * mpos[sl_a][:, None] * mneg[sl_b][None, :]

        sel = np.nonzero(T > 0)
        k1, k2 = sel[0], sel[1]
        a = np.broadcast_to(aidx[sl_a][:, None], T.shape)[sel]
        b = np.broadcast_to(aidx[sl_b][None, :], T.shape)[sel]
        conns.append((a, b, T[sel], np.where(k1 == k2, d, 3)))
        if verbose:
            print(f"  {'IJ'[d]} faces: {a.size} connections "
                  f"({int((k1 != k2).sum())} across layers)")

    # ---- vertical connections, skipping inactive layers -------------------
    a_l, b_l, t_l = [], [], []
    for s in range(1, nz):
        pair = act[: nz - s] & act[s:]
        for m in range(1, s):
            # Bridge only across a run that is entirely a MINPV pinch-out; a
            # plain ACTNUM=0 layer anywhere in the run is a real barrier.
            pair = pair & minpv_cut[m : nz - s + m]
        if not pair.any():
            continue
        T = _series(np.where(pair, ht[5][: nz - s], 0.0),
                    np.where(pair, ht[4][s:], 0.0))
        T = T * mz[: nz - s] * mzm[s:]
        sel = np.nonzero(T > 0)
        a_l.append(aidx[: nz - s][sel])
        b_l.append(aidx[s:][sel])
        t_l.append(T[sel])
    if a_l:
        a = np.concatenate(a_l)
        b = np.concatenate(b_l)
        t = np.concatenate(t_l)
        conns.append((a, b, t, np.full(a.size, 2)))
        if verbose:
            print(f"  K faces: {a.size} connections")

    ca = np.concatenate([c[0] for c in conns])
    cb = np.concatenate([c[1] for c in conns])
    ct = np.concatenate([c[2] for c in conns])
    cd = np.concatenate([c[3] for c in conns])

    # ---- global indices of each connection's two cells --------------------
    flat_act = np.nonzero(act.reshape(-1))[0]
    GA, GB = flat_act[ca], flat_act[cb]
    ia, ja = GA % nx, (GA // nx) % ny
    ib, jb = GB % nx, (GB // nx) % ny

    # ---- fault multipliers -------------------------------------------------
    fm = np.ones((nz, ny, nx, 6))
    for name, faces in deck.faults.items():
        m = deck.multflt.get(name, 1.0)
        for (i1, i2, j1, j2, k1, k2, face) in faces:
            fm[k1 - 1 : k2, j1 - 1 : j2, i1 - 1 : i2, face] *= m
    fmf = fm.reshape(-1, 6)
    # Which face a connection crosses depends on which cell is on the low side,
    # and that must be tested rather than assumed: a connection offset in K by
    # a fault throw can have its cells in either order.  Same rule as
    # apply_fault_multipliers() in the C++.
    ka, kb = GA // (nx * ny), GB // (nx * ny)
    fa = np.where(ia != ib, np.where(ia < ib, 1, 0),
                  np.where(ja != jb, np.where(ja < jb, 3, 2),
                           np.where(ka < kb, 5, 4)))
    fb = np.where(ia != ib, 1 - fa, np.where(ja != jb, 5 - fa, 9 - fa))
    mult = fmf[GA, fa] * fmf[GB, fb]
    n_fault_scaled = int((mult != 1.0).sum())
    ct = ct * mult

    # ---- MULTREGT ----------------------------------------------------------
    bit = np.where(ib != ia, 1, np.where(jb != ja, 2, 4))
    n_regt = 0
    for (r1, r2, m, bits, regname) in deck.regmult:
        if regname not in deck.arrays:
            deck.ignored.add(f"MULTREGT on undefined {regname}")
            continue
        reg = deck.arrays[regname].astype(np.int64)
        R1, R2 = reg[GA], reg[GB]
        sel = (bits & bit).astype(bool) & (R1 != R2)
        fwd = ((r1 < 0) | (R1 == r1)) & ((r2 < 0) | (R2 == r2))
        rev = ((r1 < 0) | (R2 == r1)) & ((r2 < 0) | (R1 == r2))
        sel = sel & (fwd | rev)
        if sel.any():
            ct = np.where(sel, ct * m, ct)
            n_regt += int(sel.sum())

    # ---- merge duplicate pairs --------------------------------------------
    lo = np.minimum(ca, cb).astype(np.int64)
    hi = np.maximum(ca, cb).astype(np.int64)
    key = lo * np.int64(n_active) + hi
    order = np.argsort(key, kind="stable")
    key_s, t_s, d_s = key[order], ct[order], cd[order]
    uniq, start = np.unique(key_s, return_index=True)
    t_m = np.add.reduceat(t_s, start)
    d_m = d_s[start]

    # A multiplier of zero -- a sealing MULTFLT or MULTREGT -- means there is
    # no connection, not a connection carrying nothing.  Keeping it would put a
    # structurally zero entry in the matrix and hand the AMG hierarchy a graph
    # edge across a barrier it is meant to respect.  Dropped after the merge,
    # so partial faces that sum to something real survive.
    alive = t_m > 0
    n_sealed = int((~alive).sum())
    uniq, t_m, d_m = uniq[alive], t_m[alive], d_m[alive]

    res = {
        "deck": os.path.basename(path),
        "dims": [nx, ny, nz],
        "n_global": int(nx * ny * nz),
        "n_active": n_active,
        "n_inactive": n_inactive,
        "n_minpv_dropped": n_minpv,
        "minpv": deck.minpv,
        "units": deck.units,
        "pore_volume": float(porv[act].sum()),
        "bulk_volume": float(vol[act].sum()),
        "volume_min": float(vol[act].min()),
        "volume_max": float(vol[act].max()),
        "depth_min": float(cen[..., 2][act].min()),
        "depth_max": float(cen[..., 2][act].max()),
        "dz_min": float(dzc[act].min()),
        "dz_max": float(dzc[act].max()),
        "permx_min": float(kx[act].min()),
        "permx_max": float(kx[act].max()),
        "permz_min": float(kz[act].min()),
        "permz_max": float(kz[act].max()),
        "n_faults": len(deck.faults),
        "n_fault_face_records": sum(len(x) for x in deck.faults.values()),
        "n_conn": int(uniq.size),
        "n_conn_I": int((d_m == 0).sum()),
        "n_conn_J": int((d_m == 1).sum()),
        "n_conn_K": int((d_m == 2).sum()),
        "n_conn_NNC": int((d_m == 3).sum()),
        "n_conn_fault_scaled": n_fault_scaled,
        "n_conn_multregt_scaled": n_regt,
        "n_conn_sealed_dropped": n_sealed,
        "n_multregt_records": len(deck.regmult),
        "trans_min": float(t_m.min()),
        "trans_max": float(t_m.max()),
        "trans_total": float(t_m.sum()),
        "ignored_keywords": sorted(deck.ignored),
        "seconds": round(time.time() - t0, 1),
    }
    arrays = {
        "conn_a": (uniq // n_active).astype(np.int32),
        "conn_b": (uniq % n_active).astype(np.int32),
        "conn_t": t_m,
        "conn_d": d_m.astype(np.int8),
        "porv": porv[act],
        "depth": cen[..., 2][act],
        "volume": vol[act],
    }
    return res, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", help="a .DATA deck, or a directory of includes")
    ap.add_argument("--json", default=None)
    ap.add_argument("--npz", default=None)
    args = ap.parse_args()

    print("NVRS corner-point reference build")
    print("=================================")
    res, arrays = build(args.deck)

    print()
    skip = {"ignored_keywords"}
    w = max(len(k) for k in res if k not in skip)
    for k, val in res.items():
        if k in skip:
            continue
        if isinstance(val, float):
            print(f"  {k:<{w}} : {val:.6g}")
        else:
            print(f"  {k:<{w}} : {val}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")
    if args.npz:
        np.savez_compressed(args.npz, **arrays)
        print(f"wrote {args.npz}")


if __name__ == "__main__":
    sys.exit(main())
