#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# loaders.py — turn anything that holds a reservoir field into one canonical
# in-memory form.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# Canonical form
# --------------
# Every field is a 4-D array indexed (time, k, j, i).  A static field is simply
# a single time step.  Every reader below returns that shape, so the renderers
# never need to know where the data came from.
#
# Supported sources
# -----------------
#   NVRS native   OUTPUT/meta.json + state_%04d.bin
#   ECLIPSE/OPM   CASE.EGRID + CASE.UNRST  (also reads OPM Flow output)
#   MATLAB        .mat        3-D (k,j,i) or 4-D (t,k,j,i) arrays
#   NumPy         .npy/.npz   same shapes
#   ASCII         .txt/.dat/.out/.grdecl  flat whitespace- or comma-separated
#
# Shape inference for flat/ambiguous input
# ----------------------------------------
# A flat ASCII list carries no shape, so `--dims NX NY NZ` is required for those.
# For arrays that already carry a shape we infer:
#   3-D  (nz, ny, nx)             → one time step
#   4-D  (nt, nz, ny, nx)         → time first, which is the convention here
# A 4-D array whose *last* axis matches nz while the first matches nx is
# recognised as (i,j,k,t) and transposed, since that ordering also occurs in
# the wild.
import os
import re

import numpy as np


class LoadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Shape normalisation
# ---------------------------------------------------------------------------
def to_tkji(arr, dims=None):
    """Coerce an array to (nt, nz, ny, nx).

    `dims` is an optional (nx, ny, nz) hint, used to disambiguate axis order and
    to reshape flat data.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        if not dims:
            raise LoadError(
                "flat data needs an explicit shape: pass --dims NX NY NZ")
        nx, ny, nz = dims
        cells = nx * ny * nz
        if a.size % cells != 0:
            raise LoadError(
                f"{a.size} values is not a whole number of {nx}x{ny}x{nz} "
                f"({cells}-cell) grids")
        nt = a.size // cells
        return a.reshape(nt, nz, ny, nx)

    if a.ndim == 2:
        # (nt, ncell) — a common export layout.
        if not dims:
            raise LoadError("2-D data needs --dims NX NY NZ")
        nx, ny, nz = dims
        if a.shape[1] == nx * ny * nz:
            return a.reshape(a.shape[0], nz, ny, nx)
        if a.shape[0] == nx * ny * nz:
            return a.T.reshape(a.shape[1], nz, ny, nx)
        raise LoadError(f"2-D array {a.shape} does not match {nx}x{ny}x{nz}")

    if a.ndim == 3:
        if dims:
            nx, ny, nz = dims
            if a.shape == (nx, ny, nz) and a.shape != (nz, ny, nx):
                a = np.transpose(a, (2, 1, 0))       # (i,j,k) → (k,j,i)
        return a[None, ...]

    if a.ndim == 4:
        if dims:
            nx, ny, nz = dims
            # (t,i,j,k) → (t,k,j,i)
            if a.shape[1:] == (nx, ny, nz) and a.shape[1:] != (nz, ny, nx):
                return np.transpose(a, (0, 3, 2, 1))
            # (i,j,k,t) → (t,k,j,i): time trailing rather than leading.
            if a.shape[:3] == (nx, ny, nz) and a.shape[0] != nz:
                return np.transpose(a, (3, 2, 1, 0))
        return a

    raise LoadError(f"cannot interpret an array of shape {a.shape}")


# ---------------------------------------------------------------------------
# Individual readers
# ---------------------------------------------------------------------------
def load_ascii(path, dims=None):
    """Read a flat ASCII field.

    Tolerates comma or whitespace separation, comment lines (`--`, `#`, `!`),
    and the ECLIPSE `123*0.25` repeat-count notation used in GRDECL files.
    """
    text = open(path, "r", errors="replace").read()
    # Strip ECLIPSE/GRDECL comments and the trailing record slash.
    text = re.sub(r"--[^\n]*", " ", text)
    text = re.sub(r"[#!][^\n]*", " ", text)
    text = text.replace(",", " ").replace("/", " ")
    vals = []
    for tok in text.split():
        if "*" in tok:                       # ECLIPSE repeat count: n*value
            try:
                n, v = tok.split("*", 1)
                vals.extend([float(v)] * int(n))
                continue
            except ValueError:
                pass
        try:
            vals.append(float(tok))
        except ValueError:
            continue                         # skip keywords and stray text
    if not vals:
        raise LoadError(f"{path}: no numeric values found")
    return to_tkji(np.asarray(vals, dtype=float), dims)


def load_mat(path, key=None, dims=None):
    """Read a MATLAB .mat file.

    Returns {name: (nt,nz,ny,nx)} for every array that can be interpreted, or
    just the requested `key`.
    """
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise LoadError(f"reading .mat needs scipy: {exc}")
    try:
        md = loadmat(path)
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5.
        try:
            import h5py
        except Exception as exc:
            raise LoadError(
                f"{path} is a MATLAB v7.3 (HDF5) file and needs h5py: {exc}")
        out = {}
        with h5py.File(path, "r") as h:
            for k in h.keys():
                v = np.array(h[k])
                if v.ndim >= 3:
                    # h5py returns MATLAB arrays transposed.
                    try:
                        out[k] = to_tkji(v.T if v.ndim == 3 else v, dims)
                    except LoadError:
                        continue
        if key:
            if key not in out:
                raise LoadError(f"{path}: no array named {key!r}")
            return {key: out[key]}
        return out

    out = {}
    for k, v in md.items():
        if k.startswith("__"):
            continue
        v = np.asarray(v)
        if v.ndim < 1 or not np.issubdtype(v.dtype, np.number):
            continue
        try:
            out[k] = to_tkji(v, dims)
        except LoadError:
            continue
    if key:
        if key not in out:
            raise LoadError(f"{path}: no usable array named {key!r} "
                            f"(found: {', '.join(sorted(out)) or 'none'})")
        return {key: out[key]}
    if not out:
        raise LoadError(f"{path}: no array could be interpreted as a field")
    return out


def load_npy(path, dims=None):
    if path.endswith(".npz"):
        z = np.load(path)
        return {k: to_tkji(z[k], dims) for k in z.files}
    return {os.path.splitext(os.path.basename(path))[0]:
            to_tkji(np.load(path), dims)}


def load_any(path, dims=None, key=None):
    """Dispatch on file extension. Always returns {name: (nt,nz,ny,nx)}."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        return load_mat(path, key=key, dims=dims)
    if ext in (".npy", ".npz"):
        return load_npy(path, dims)
    if ext in (".txt", ".dat", ".out", ".asc", ".grdecl", ".inc", ""):
        name = os.path.splitext(os.path.basename(path))[0] or "field"
        return {name: load_ascii(path, dims)}
    raise LoadError(f"{path}: unrecognised extension {ext!r}")


# ---------------------------------------------------------------------------
# Field-name recognition
# ---------------------------------------------------------------------------
# Map common naming conventions onto the canonical variable names so that a
# user's own .mat file plots with the right labels and colour maps.
_ALIASES = {
    "pressure": ("pressure", "press", "p", "pres", "bhp3d", "prs"),
    "water":    ("swat", "sw", "water", "watsat", "s_w", "saturation_water"),
    "oil":      ("soil", "so", "oil", "oilsat", "s_o", "saturation_oil"),
    "gas":      ("sgas", "sg", "gas", "gassat", "s_g", "saturation_gas"),
    "perm":     ("permx", "perm", "k", "permeability", "kx"),
    "poro":     ("poro", "porosity", "phi", "por"),
}

LABELS = {
    "pressure": ("Pressure (psia)", "jet"),
    "water":    ("Water saturation", "Blues"),
    "oil":      ("Oil saturation", "copper"),
    "gas":      ("Gas saturation", "inferno"),
    "perm":     ("Permeability (mD)", "viridis"),
    "poro":     ("Porosity", "cividis"),
}


def canonical_name(raw):
    """Best-effort mapping of an arbitrary array name to a canonical variable.

    Exact matches win outright.  The substring fallback ignores aliases shorter
    than three characters, because one- and two-letter aliases ("p", "sw", "k")
    match almost anything as substrings — "perm_flat" would otherwise be
    classified as pressure on the strength of its leading "p".
    """
    s = re.sub(r"[^a-z0-9]", "", str(raw).lower())
    for canon, alts in _ALIASES.items():
        for a in alts:
            if s == re.sub(r"[^a-z0-9]", "", a):
                return canon
    for canon, alts in _ALIASES.items():
        for a in alts:
            key = re.sub(r"[^a-z0-9]", "", a)
            if len(key) >= 3 and key in s:
                return canon
    return None
