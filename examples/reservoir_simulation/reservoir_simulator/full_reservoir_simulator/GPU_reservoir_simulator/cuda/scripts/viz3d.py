#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# viz3d.py — 3-D voxel rendering, 2-D slice panels and movie assembly.
#
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# The 3-D style follows the original CuPy reference plotter
# (GPU_reservoir_simulator/NVRS/plotting.py :: plot3d2static): a jet-coloured
# voxel block in orthographic projection, aspect [nx, ny, 2*nz], viewed from
# elev=30/azim=45, with each well drawn as a vertical line running the full
# height of the model and continuing *above* it, labelled with the well name —
# injectors black, producers red.
#
# Two things are fixed relative to the reference:
#   * `ax.voxels` cost grows with the cell count and becomes unusable past a few
#     tens of thousands of cells, so large grids are decimated to a target
#     budget before rendering (the colour field is block-averaged, not
#     point-sampled, so the picture stays faithful).
#   * The reference passed the normalised field as the `filled` mask, which
#     silently made every cell whose value happened to be zero invisible.  The
#     fill mask is now explicit.
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import cm

# Rendering more voxels than this is slower than it is informative.  Measured
# on a 200x200x25 (1e6 cell) field: ax.voxels is ~5 s at 15k voxels and scales
# with the count, while tight_layout plus bbox_inches="tight" on a 3-D axes add
# a further ~5 s of fixed cost by forcing extra full renders.  Both are avoided
# below, and the budget is kept where the picture is still faithful.
MAX_VOXELS = 16000


def _decimate(field3d, budget=MAX_VOXELS):
    """Block-average a (nz, ny, nx) field down to at most `budget` voxels.

    Returns (decimated, (sx, sy, sz)) where the strides let callers map well
    locations into the reduced index space.
    """
    nz, ny, nx = field3d.shape
    n = nx * ny * nz
    if n <= budget:
        return field3d, (1, 1, 1)
    # Uniform stride in each direction, rounded up so the budget is respected.
    f = (n / budget) ** (1.0 / 3.0)
    sx, sy, sz = max(1, int(math.ceil(f))), max(1, int(math.ceil(f))), max(1, int(math.ceil(f)))
    # Trim to a whole number of blocks, then mean-reduce.
    tz, ty, tx = (nz // sz) * sz, (ny // sy) * sy, (nx // sx) * sx
    tz, ty, tx = max(tz, sz), max(ty, sy), max(tx, sx)
    cut = field3d[:tz, :ty, :tx]
    out = cut.reshape(tz // sz, sz, ty // sy, sy, tx // sx, sx).mean(axis=(1, 3, 5))
    return out, (sx, sy, sz)


# Cells per drawn face before it is block-averaged.  The SPE10 faces (60 x 220
# top, 220 x 85 side) are 37,000 cells in all and draw in about a second.
MAX_FACE_CELLS = 60000


def _face_reduce(a, log):
    """Block-average a 2-D face to at most MAX_FACE_CELLS cells (geometric
    mean for a log-scaled field).  Returns (face, stride)."""
    n0, n1 = a.shape
    s = max(1, int(math.ceil(math.sqrt(n0 * n1 / float(MAX_FACE_CELLS)))))
    if s == 1:
        return a, 1
    t0, t1 = max(s, (n0 // s) * s), max(s, (n1 // s) * s)
    cut = a[:t0, :t1]
    if log:
        cut = np.log10(cut)
    out = cut.reshape(t0 // s, s, t1 // s, s).mean(axis=(1, 3))
    return (10.0 ** out if log else out), s


def draw_faces(ax, field3d, wells=None, vmin=None, vmax=None, cmap="jet",
               elev=24, azim=-58, label_wells=True, well_fontsize=11,
               axis_labels=True, log=False, cut=None):
    """Draw a (nz, ny, nx) Cartesian field as its three visible faces, cell by
    cell, the way the paper draws SPE10 (paper/spe10_rock_fig.py): the top
    layer, the front section and the side section at native resolution, each
    cell a flat quad of its own colour, on a log scale for a field spanning
    decades.  Unlike ax.voxels it needs no decimation up to large grids and no
    transparency, so channels and fronts stay sharp.  Which side faces are
    visible follows the azimuth.  Returns the Normalize instance.

    cut=(i, j), 0-based cell indices, removes the quarter of the block between
    that column and the two faces turned to the viewer, through its full
    depth.  The two cut planes then show the field down the column itself --
    for a moving front (pressure, saturation) the column of the injector,
    where the faces of the closed block would show almost nothing happening.
    """
    wells = wells or []
    arr = np.asarray(field3d, dtype=float)
    nz, ny, nx = arr.shape
    lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
    hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-30:
        hi = lo + 1.0
    if log and lo > 0:
        norm = matplotlib.colors.LogNorm(vmin=lo, vmax=hi)
        arr = np.clip(arr, lo, hi)
    else:
        log = False
        norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
    cmo = plt.get_cmap(cmap)

    def colours(face):
        f = np.clip(face, lo, hi)
        c = np.asarray(norm(f.ravel())).reshape(f.shape)
        return cmo(np.pad(c, ((0, 1), (0, 1)), mode="edge"))

    # Index units, k counted downward from the top.  The height is the voxel
    # view's 2 nz, capped at 0.35 of the longer side: SPE10's 85 layers then
    # read as the flat slab of the paper's figure rather than a tower.
    top_z = float(min(2.0 * nz, 0.35 * max(nx, ny)))
    top_z = max(top_z, 1.0)
    xmax_side = math.cos(math.radians(azim)) > 0
    ymax_side = math.sin(math.radians(azim)) > 0
    kw = dict(rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)

    def top(x0, x1, y0, y1):
        if x1 <= x0 or y1 <= y0:
            return
        v, _ = _face_reduce(arr[0, y0:y1, x0:x1].T, log)
        X, Y = np.meshgrid(np.linspace(x0, x1, v.shape[0] + 1),
                           np.linspace(y0, y1, v.shape[1] + 1), indexing="ij")
        ax.plot_surface(X, Y, np.full_like(X, top_z), facecolors=colours(v), **kw)

    def yplane(yv, jcell, x0, x1):
        if x1 <= x0:
            return
        v, _ = _face_reduce(arr[:, jcell, x0:x1].T, log)
        X, Z = np.meshgrid(np.linspace(x0, x1, v.shape[0] + 1),
                           np.linspace(top_z, 0, v.shape[1] + 1), indexing="ij")
        ax.plot_surface(X, np.full_like(X, float(yv)), Z, facecolors=colours(v), **kw)

    def xplane(xv, icell, y0, y1):
        if y1 <= y0:
            return
        v, _ = _face_reduce(arr[:, y0:y1, icell].T, log)
        Y, Z = np.meshgrid(np.linspace(y0, y1, v.shape[0] + 1),
                           np.linspace(top_z, 0, v.shape[1] + 1), indexing="ij")
        ax.plot_surface(np.full_like(Y, float(xv)), Y, Z, facecolors=colours(v), **kw)

    xo, xc = (nx, nx - 1) if xmax_side else (0, 0)     # outer x face, its cells
    yo, yc = (ny, ny - 1) if ymax_side else (0, 0)
    ic = jc = None
    if cut is not None:
        ci, cj = int(cut[0]), int(cut[1])
        ic = ci + 1 if xmax_side else ci
        jc = cj + 1 if ymax_side else cj
        if not (1 <= ic <= nx - 1 and 1 <= jc <= ny - 1):
            ic = jc = None
    if ic is None:
        top(0, nx, 0, ny)
        yplane(yo, yc, 0, nx)
        xplane(xo, xc, 0, ny)
    else:
        kx0, kx1 = (0, ic) if xmax_side else (ic, nx)   # kept x range
        cx0, cx1 = (ic, nx) if xmax_side else (0, ic)   # cut x range
        ky0, ky1 = (0, jc) if ymax_side else (jc, ny)
        cy0, cy1 = (jc, ny) if ymax_side else (0, jc)
        top(kx0, kx1, 0, ny)
        top(cx0, cx1, ky0, ky1)
        yplane(yo, yc, kx0, kx1)
        xplane(xo, xc, ky0, ky1)
        yplane(jc, jc - 1 if ymax_side else jc, cx0, cx1)
        xplane(ic, ic - 1 if xmax_side else ic, cy0, cy1)

    if axis_labels:
        ax.set_xlabel("X (i)")
        ax.set_ylabel("Y (j)")
        ax.set_zlabel("Z (k)")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, top_z)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.set_box_aspect([nx, ny, top_z])
    ax.set_proj_type("ortho")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticklabels([])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)

    stick = top_z + max(2.0, 0.25 * top_z)
    for w in wells:
        try:
            xi = int(w["i"]) - 0.5
            yj = int(w["j"]) - 0.5
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= xi <= nx and 0 <= yj <= ny):
            continue
        col = well_colour(w)
        # Above the top face only, as in the paper: a line through the solid
        # block would be drawn over the faces in front of it.
        ax.plot([xi, xi], [yj, yj], [top_z, stick], color=col, linewidth=2.5,
                zorder=10)
        if label_wells:
            ax.text(xi, yj, stick, str(w.get("name", "")), color=col,
                    weight="bold", fontsize=well_fontsize, zorder=11)
    return norm


def draw_voxels(ax, field3d, wells=None, vmin=None, vmax=None, cmap="jet",
                elev=30, azim=45, alpha=0.52, label_wells=True,
                well_fontsize=11, axis_labels=True, log=False, cut=None):
    """Draw a (nz, ny, nx) field as a voxel block into an existing 3-D axes.

    Split out of plot3d_voxels so callers that compose a grid of 3-D panels
    (benchmark/compare.py lays out simulator x variable) share exactly the same
    rendering, view angle and well overlay as the single-figure path.  Returns
    the Normalize instance so the caller can attach its own colourbar.

    `wells` is a list of dicts with keys i, j (1-based, as in the deck), name
    and type ("inj" or "prod").
    """
    wells = wells or []
    arr = np.asarray(field3d, dtype=float)
    # A Cartesian block with every cell active is drawn face by face at native
    # resolution, as the paper draws SPE10 (draw_faces).  Voxels remain for
    # corner-point and other grids with inactive (NaN) cells, whose interior
    # holes the three faces cannot show.  NVRS_VOXELS=1 forces voxels.
    if (os.environ.get("NVRS_VOXELS", "0") != "1" and arr.ndim == 3
            and arr.size and np.isfinite(arr).all()):
        if (elev, azim) == (30, 45):
            elev, azim = 24, -58          # the paper's view
        return draw_faces(ax, arr, wells=wells, vmin=vmin, vmax=vmax,
                          cmap=cmap, elev=elev, azim=azim,
                          label_wells=label_wells,
                          well_fontsize=well_fontsize,
                          axis_labels=axis_labels, log=log, cut=cut)
    if log:
        # A field spanning decades (SPE10 permeability: 1e-3 to 2e4 mD) is
        # block-averaged geometrically and coloured on a log scale; on a
        # linear one every cell but the channels is the bottom colour.
        floor = np.nanmin(arr[arr > 0]) if np.any(arr > 0) else 1e-30
        dec, (sx, sy, sz) = _decimate(np.log10(np.maximum(arr, floor)))
        dec = 10.0 ** dec
    else:
        dec, (sx, sy, sz) = _decimate(arr)
    # matplotlib's voxel axes index as (x, y, z); the field is (z, y, x).
    #
    # The k index counts DOWNWARD from the top of the model, while the plot's
    # z axis increases upward, so the layer order has to be reversed or the
    # reservoir is drawn upside down: the shallowest layer at the bottom of
    # the figure, the deepest at the top, and the well sticks -- which are
    # drawn upward, to emerge from the top face -- emerging from the base.
    # plot_slices already renders its vertical sections with origin="upper"
    # for the same reason, so this makes the two views agree.
    vol = np.transpose(dec, (2, 1, 0))[:, :, ::-1]
    vnx, vny, vnz = vol.shape

    lo = float(np.nanmin(vol)) if vmin is None else float(vmin)
    hi = float(np.nanmax(vol)) if vmax is None else float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-30:
        hi = lo + 1.0

    if log and lo > 0:
        norm = matplotlib.colors.LogNorm(vmin=lo, vmax=hi)
    else:
        norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
    # Normalised as a flat array and reshaped: LogNorm transforms through a
    # matplotlib Transform, which accepts only (N,) or (N, 1) input and
    # raised "Input values must have shape (N, 1) or (1,)" on the 3-D block,
    # so SPE10's log-scale permeability fell back to a linear scale.
    colors = plt.get_cmap(cmap)(
        np.asarray(norm(np.clip(vol, lo, hi).ravel())).reshape(vol.shape))
    # Draw only the cells that HAVE a value.  On a corner-point grid the
    # inactive 61% of the box is NaN, and rendering it as solid produced a
    # rectangular slab wrapped around the reservoir -- the field invisible
    # inside it.  np.isfinite is the active-cell mask, for free.
    filled = np.isfinite(vol)
    if not filled.any():
        filled = np.ones(vol.shape, dtype=bool)
    ax.voxels(filled, facecolors=colors, alpha=alpha, edgecolor="none", shade=True)

    if axis_labels:
        # In a dense grid of panels these collide with the neighbouring
        # colourbar and carry no information, since the axes have no ticks.
        ax.set_xlabel("X (i)")
        ax.set_ylabel("Y (j)")
        ax.set_zlabel("Z (k)")
    ax.set_xlim(0, vnx)
    ax.set_ylim(0, vny)
    ax.set_zlim(0, vnz)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.set_box_aspect([vnx, vny, vnz * 2])
    ax.set_proj_type("ortho")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticklabels([])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)

    # Wells: a vertical line through the full model height that continues above
    # the top face, so the trajectory reads clearly against the voxel block.
    stick = vnz + max(2.0, 0.25 * vnz)
    for w in wells:
        try:
            xi = (int(w["i"]) - 1) / sx
            yj = (int(w["j"]) - 1) / sy
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= xi <= vnx and 0 <= yj <= vny):
            continue
        col = well_colour(w)
        ax.plot([xi, xi], [yj, yj], [0, stick], color=col, linewidth=2.2,
                zorder=10)
        if label_wells:
            ax.text(xi, yj, stick, str(w.get("name", "")), color=col,
                    weight="bold", fontsize=well_fontsize, zorder=11)
    return norm


# Well colour by what the well DOES.  Black for injectors and red for
# producers said only "injector or not", which on a field that alternates a
# well between water and gas injection -- Norne's C-4H does it all run -- is
# the distinction that matters and the one that was missing.  Producers are
# green because they take fluid out, water injectors blue and gas injectors
# red because they put different things in.
WELL_COLOUR = {"winj": "blue", "ginj": "red", "prod": "green"}


def well_colour(w):
    t = str(w.get("type", "prod")).lower()
    if t in WELL_COLOUR:
        return WELL_COLOUR[t]
    # Older sidecars only wrote "inj"/"prod"; treat a bare injector as water,
    # which is what the great majority of them are.
    return "blue" if t.startswith("inj") else "green"


def plot3d_voxels(field3d, title, outpath, wells=None, vmin=None, vmax=None,
                  cbar_label="", cmap="jet", elev=30, azim=45, log=False):
    """Render a (nz, ny, nx) field as a 3-D voxel block with wells sticking out."""
    fig = plt.figure(figsize=(11, 7.5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    # Reclaim the margins that bbox_inches="tight" used to crop.  Setting the
    # positions directly costs nothing, whereas tight_layout re-renders.
    fig.subplots_adjust(left=0.01, right=0.90, top=0.97, bottom=0.02)
    norm = draw_voxels(ax, field3d, wells=wells, vmin=vmin, vmax=vmax,
                       cmap=cmap, elev=elev, azim=azim, log=log)

    m = cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, fraction=0.022, pad=0.10, label=cbar_label or title)

    fig.suptitle(title, fontsize=15)
    # No tight_layout and no bbox_inches="tight": each forces an additional
    # full render of the 3-D scene, which together cost more than drawing the
    # voxels.  The axes box is already set explicitly above.
    fig.savefig(outpath, dpi=100)
    plt.close(fig)
    return outpath


def plot_slices(field3d, title, outpath, wells=None, vmin=None, vmax=None,
                cbar_label="", cmap="jet", islice=None, jslice=None, kslice=None):
    """Three orthogonal 2-D slices (K-plan, J-section, I-section) in one figure."""
    wells = wells or []
    a = np.asarray(field3d, dtype=float)
    nz, ny, nx = a.shape
    i0 = nx // 2 if islice is None else int(islice)
    j0 = ny // 2 if jslice is None else int(jslice)
    k0 = nz // 2 if kslice is None else int(kslice)
    i0 = min(max(i0, 0), nx - 1); j0 = min(max(j0, 0), ny - 1); k0 = min(max(k0, 0), nz - 1)

    lo = float(np.nanmin(a)) if vmin is None else float(vmin)
    hi = float(np.nanmax(a)) if vmax is None else float(vmax)
    if hi - lo < 1e-30:
        hi = lo + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # K slice — areal plan view (i horizontal, j vertical).
    im = axes[0].imshow(a[k0], origin="lower", cmap=cmap, aspect="auto",
                        vmin=lo, vmax=hi)
    axes[0].set_title(f"Areal plan, k = {k0}")
    axes[0].set_xlabel("i"); axes[0].set_ylabel("j")
    for w in wells:
        try:
            xi, yj = int(w["i"]) - 1, int(w["j"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        axes[0].plot(xi, yj,
                     marker="^" if well_colour(w) == "green" else "v",
                     color=well_colour(w), markersize=8,
                     markeredgecolor="black", markeredgewidth=0.6)
        axes[0].annotate(str(w.get("name", "")), (xi, yj),
                         textcoords="offset points", xytext=(5, 4),
                         fontsize=7, color="black")

    # J slice — cross-section along i, depth increasing downwards.
    axes[1].imshow(a[:, j0, :], origin="upper", cmap=cmap, aspect="auto",
                   vmin=lo, vmax=hi)
    axes[1].set_title(f"Cross-section, j = {j0}")
    axes[1].set_xlabel("i"); axes[1].set_ylabel("k (depth)")

    # I slice — cross-section along j.
    axes[2].imshow(a[:, :, i0], origin="upper", cmap=cmap, aspect="auto",
                   vmin=lo, vmax=hi)
    axes[2].set_title(f"Cross-section, i = {i0}")
    axes[2].set_xlabel("j"); axes[2].set_ylabel("k (depth)")

    fig.colorbar(im, ax=axes.ravel().tolist(), label=cbar_label or title,
                 fraction=0.02, pad=0.02)
    fig.suptitle(title, fontsize=13)
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    return outpath


def frame_index(path):
    """Trailing integer of a frame filename, for numeric ordering.

    Lexicographic order puts frame 10 between 1 and 2, so an animation built
    with sorted(glob(...)) plays 0, 1, 10, 11, ... 2, 20, 3 ...  Zero padding
    hides this until something writes an unpadded name, so order numerically
    and never rely on the padding.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else -1


def sort_frames(paths):
    """Frame paths in true time order."""
    return sorted(paths, key=frame_index)


def make_movie(frame_paths, out_base, fps=3, scale=1.0):
    """Assemble frames into a GIF and, when ffmpeg is available, an MP4.

    Returns the list of files actually written.  Missing optional dependencies
    are not an error — the caller just gets fewer outputs.

    `scale` shrinks each frame before assembly.  A GIF of full-resolution 3-D
    panels is by far the largest thing in a self-contained report, and an
    animation is watched for the shape of the front moving, not for reading
    axis labels off it — so halving the linear size, which quarters the pixel
    count, costs the viewer almost nothing and is what lets the animation stay
    in the page instead of being dropped for size.
    """
    written = []
    frames = sort_frames([p for p in frame_paths if os.path.exists(p)])
    if len(frames) < 2:
        return written

    # --- GIF via Pillow ---
    try:
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in frames]
        if scale and scale < 0.999:
            w, h = imgs[0].size
            size = (max(int(w * scale), 1), max(int(h * scale), 1))
            imgs = [im.resize(size, Image.LANCZOS) for im in imgs]
        # A GIF palette is 256 colours; quantise once so every frame agrees.
        pal = imgs[0].convert("P", palette=Image.ADAPTIVE, colors=255)
        qimgs = [im.quantize(palette=pal) for im in imgs]
        gif = out_base + ".gif"
        qimgs[0].save(gif, format="GIF", append_images=qimgs[1:], save_all=True,
                      duration=int(1000 / max(fps, 1)), loop=0)
        written.append(gif)
    except Exception as exc:
        print(f"[viz] GIF skipped: {type(exc).__name__}: {exc}")

    # --- MP4 via ffmpeg ---
    try:
        import matplotlib.animation as manim
        if manim.writers.is_available("ffmpeg"):
            import matplotlib.image as mpimg
            first = mpimg.imread(frames[0])
            h, w = first.shape[:2]
            fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            im = ax.imshow(first)
            writer = manim.FFMpegWriter(fps=fps, bitrate=2400)
            mp4 = out_base + ".mp4"
            with writer.saving(fig, mp4, dpi=100):
                for p in frames:
                    im.set_data(mpimg.imread(p))
                    writer.grab_frame()
            plt.close(fig)
            written.append(mp4)
    except Exception as exc:
        print(f"[viz] MP4 skipped: {type(exc).__name__}: {exc}")

    return written
