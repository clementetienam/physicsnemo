#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# ecl.py — a dependency-free reader for ECLIPSE / OPM Flow output files.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# Handles the "unformatted" ECLIPSE convention: big-endian Fortran sequential
# records, each bracketed by a 4-byte byte-count marker.  A named array is a
# 24-byte header record
#
#     [int32 16]["KEYWORD "][int32 count]["TYPE"][int32 16]
#
# followed by the data split into blocks of at most 1000 elements (105 for
# CHAR), each block bracketed by its own markers.
#
# This reads what the simulator writes (see src/ecl_io.cpp) and, equally, what
# OPM Flow or ECLIPSE writes — so the same plotting path serves both, which is
# how you compare runs.
#
#   from ecl import read_unformatted, EclipseCase
#   case = EclipseCase("OUTPUT/NVRS")
#   nx, ny, nz = case.dims
#   P = case.restart_field("PRESSURE")        # (nsteps, nz, ny, nx)
#   t, vectors = case.summary()               # time series
#
# Run directly for a quick dump:
#   python3 ecl.py OUTPUT/NVRS.UNRST
import os
import struct
import sys

import numpy as np

# Elements per Fortran record, per ECLIPSE convention.
_NUM_BLOCK = 1000
_CHAR_BLOCK = 105
_CHAR_LEN = 8

_DTYPE = {
    b"INTE": (">i4", 4),
    b"REAL": (">f4", 4),
    b"DOUB": (">f8", 8),
    b"LOGI": (">i4", 4),
    b"CHAR": (None, _CHAR_LEN),
    b"MESS": (None, 0),
}


class EclFormatError(RuntimeError):
    pass


def _read_marker(f):
    b = f.read(4)
    if len(b) < 4:
        return None
    return struct.unpack(">i", b)[0]


def read_unformatted(path):
    """Read an ECLIPSE unformatted file.

    Returns a list of (keyword, type, numpy array or list-of-str) in file order.
    Duplicate keywords are preserved — a unified restart repeats PRESSURE once
    per report step, and the order is what separates them.
    """
    out = []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        while f.tell() < size:
            m = _read_marker(f)
            if m is None:
                break
            if m != 16:
                raise EclFormatError(
                    f"{path}: expected a 16-byte keyword header at offset "
                    f"{f.tell()-4}, got marker {m}")
            name = f.read(8).decode("ascii", "replace").strip()
            count = struct.unpack(">i", f.read(4))[0]
            typ = f.read(4)
            tail = _read_marker(f)
            if tail != 16:
                raise EclFormatError(f"{path}: unterminated header for {name}")

            if typ not in _DTYPE:
                raise EclFormatError(f"{path}: unknown type {typ!r} for {name}")
            dt, esize = _DTYPE[typ]

            if typ == b"MESS" or count == 0:
                out.append((name, typ.decode(), np.array([])))
                continue

            block = _CHAR_BLOCK if typ == b"CHAR" else _NUM_BLOCK
            remaining = count
            chunks = []
            while remaining > 0:
                nb = _read_marker(f)
                m_elems = min(block, remaining)
                expect = m_elems * esize
                if nb != expect:
                    raise EclFormatError(
                        f"{path}: {name} block marker {nb} != expected {expect}")
                raw = f.read(nb)
                if _read_marker(f) != expect:
                    raise EclFormatError(f"{path}: {name} block not terminated")
                if typ == b"CHAR":
                    chunks.extend(
                        raw[i:i + _CHAR_LEN].decode("ascii", "replace").strip()
                        for i in range(0, len(raw), _CHAR_LEN))
                else:
                    chunks.append(np.frombuffer(raw, dtype=dt))
                remaining -= m_elems

            if typ == b"CHAR":
                out.append((name, "CHAR", chunks))
            else:
                arr = np.concatenate(chunks) if chunks else np.array([], dtype=dt)
                if typ == b"LOGI":
                    arr = arr.astype(bool)
                out.append((name, typ.decode(), arr))
    return out


class EclipseCase:
    """Convenience view over a CASE.{EGRID,INIT,UNRST,SMSPEC,UNSMRY} set."""

    def __init__(self, base):
        # Accept either "DIR/CASE" or any "DIR/CASE.EXT".
        root, ext = os.path.splitext(base)
        self.base = root if ext.upper() in (
            ".EGRID", ".INIT", ".UNRST", ".SMSPEC", ".UNSMRY", ".RSM") else base
        self._egrid = None
        self._init = None
        self._unrst = None

    # -- lazy file access --------------------------------------------------
    def _load(self, attr, ext):
        cur = getattr(self, attr)
        if cur is None:
            path = self.base + ext
            cur = read_unformatted(path) if os.path.exists(path) else []
            setattr(self, attr, cur)
        return cur

    @staticmethod
    def _first(recs, kw):
        for name, _t, val in recs:
            if name == kw:
                return val
        return None

    @property
    def unit_system(self):
        """"METRIC", "FIELD", "LAB" or None, from INTEHEAD item 3.

        ECLIPSE stamps the unit system of the run into every output file, so
        this is what the file itself says rather than what the deck was
        believed to be.  It matters because a metric run writes pressures in
        bar and volumes in m3, and comparing those against a field-unit run
        without converting puts a factor of 14.5 on every pressure and 6.29 on
        every volume -- large enough to look like a modelling difference and
        smooth enough not to look like a bug.
        """
        for attr, ext in (("_init", ".INIT"), ("_unrst", ".UNRST")):
            try:
                ih = self._first(self._load(attr, ext), "INTEHEAD")
            except Exception:
                ih = None
            if ih is not None and len(ih) > 2:
                return {1: "METRIC", 2: "FIELD", 3: "LAB"}.get(int(ih[2]))
        return None

    @property
    def dims(self):
        """(nx, ny, nz) from EGRID, falling back to INIT."""
        gh = self._first(self._load("_egrid", ".EGRID"), "GRIDHEAD")
        if gh is not None and len(gh) > 3:
            return int(gh[1]), int(gh[2]), int(gh[3])
        ih = self._first(self._load("_init", ".INIT"), "INTEHEAD")
        if ih is not None and len(ih) > 10:
            return int(ih[8]), int(ih[9]), int(ih[10])
        raise EclFormatError("cannot determine grid dimensions")

    def actnum(self):
        """ACTNUM from EGRID as a flat int array, or None if the file has none."""
        a = self._first(self._load("_egrid", ".EGRID"), "ACTNUM")
        return None if a is None else np.asarray(a, dtype=np.int64)

    def _expand(self, v, nx, ny, nz):
        """Reshape an ECLIPSE array to (nz, ny, nx), scattering active cells.

        ECLIPSE — and therefore OPM Flow — stores dynamic arrays for ACTIVE
        cells only.  On a grid with no inactive cells the two lengths coincide
        and this is a plain reshape; where they do not, the values are
        scattered back into the global grid through ACTNUM and inactive cells
        come back as NaN so callers can exclude them rather than silently
        averaging a hole into their statistics.
        """
        n = nx * ny * nz
        v = np.asarray(v, dtype=np.float64)
        if v.size == n:
            return v.reshape((nz, ny, nx))
        act = self.actnum()
        if act is not None and act.size == n and int((act > 0).sum()) == v.size:
            full = np.full(n, np.nan)
            full[act > 0] = v
            return full.reshape((nz, ny, nx))
        raise EclFormatError(
            "array has %d values; grid is %dx%dx%d = %d cells and ACTNUM "
            "accounts for %s" % (v.size, nx, ny, nz, n,
                                 "no active cells" if act is None
                                 else int((act > 0).sum())))

    def init_field(self, kw):
        """A static INIT array reshaped to (nz, ny, nx)."""
        nx, ny, nz = self.dims
        v = self._first(self._load("_init", ".INIT"), kw)
        if v is None:
            return None
        return self._expand(v, nx, ny, nz)

    def restart_times(self):
        """Simulation time (days) of each report step in the unified restart."""
        recs = self._load("_unrst", ".UNRST")
        return np.array([v[0] for n, _t, v in recs
                         if n == "DOUBHEAD" and len(v) > 0], dtype=float)

    def restart_field(self, kw):
        """A dynamic UNRST array over all report steps → (nsteps, nz, ny, nx)."""
        nx, ny, nz = self.dims
        recs = self._load("_unrst", ".UNRST")
        frames = [self._expand(v, nx, ny, nz) for n, _t, v in recs if n == kw]
        if not frames:
            return None
        return np.stack(frames, axis=0)

    def summary(self):
        """Return (times, {label: values}) from SMSPEC + UNSMRY.

        Labels are "KEYWORD:WELL" for well vectors and "KEYWORD" for field ones.
        """
        spec_path, data_path = self.base + ".SMSPEC", self.base + ".UNSMRY"
        if not (os.path.exists(spec_path) and os.path.exists(data_path)):
            return None, {}
        spec = read_unformatted(spec_path)
        kws = self._first(spec, "KEYWORDS") or []
        wgn = self._first(spec, "WGNAMES") or []
        units = self._first(spec, "UNITS") or []

        data = read_unformatted(data_path)
        params = [np.asarray(v, dtype=float)
                  for n, _t, v in data if n == "PARAMS"]
        if not params:
            return None, {}
        mat = np.stack(params, axis=0)          # (nsteps, nlist)

        times, out = None, {}
        for j, kw in enumerate(kws):
            if j >= mat.shape[1]:
                break
            wg = wgn[j] if j < len(wgn) else ""
            unit = units[j] if j < len(units) else ""
            if kw == "TIME":
                times = mat[:, j]
                continue
            label = f"{kw}:{wg}" if wg and wg != ":+:+:+:+" else kw
            out[label] = {"values": mat[:, j], "unit": unit,
                          "keyword": kw, "well": wg}
        return times, out


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for path in argv[1:]:
        print(f"=== {path} ===")
        for name, typ, val in read_unformatted(path):
            n = len(val)
            if typ == "CHAR":
                preview = ", ".join(repr(x) for x in val[:6])
            elif n:
                preview = ", ".join(f"{x:.6g}" for x in np.asarray(val).ravel()[:6])
            else:
                preview = ""
            print(f"  {name:<10s} {typ:<5s} n={n:<8d} {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
