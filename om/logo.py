"""Official logo handling.

The logo is only ever rasterised from the official vector (never redrawn or
recoloured), then moved / scaled / faded / masked.  For the trace-on we derive,
from the logo's own strokes, a centre-line path (for the light line, which is
our own drawing) and a per-pixel "arc-length" map used purely as a reveal mask.
"""
import re
from functools import lru_cache

import cairosvg
import numpy as np
from PIL import Image

from .config import LOGO_AZUL, LOGO_BLANCO, ROOT

_SVG = LOGO_AZUL.read_text()
_D = re.search(r' d="([^"]*)"', _SVG).group(1)
_VB = [float(x) for x in re.search(r'viewBox="([^"]*)"', _SVG).group(1).split()]
ASPECT = _VB[2] / _VB[3]          # width / height

# subpaths of the official artwork, in drawing order for the trace:
# left crown + side (8), right crown (7), natural root (6), then the six implant threads top->bottom
_SUBS = [x for x in re.split(r"(?=M )", _D) if x.strip()]
ORDER = [8, 7, 6, 5, 4, 0, 1, 3, 2]
DIRECTION = {8: "inner_first", 7: "inner_first", 6: "top_first", 5: "left_first", 4: "left_first",
             0: "left_first", 1: "left_first", 3: "left_first", 2: "left_first"}


def raster(height, white=False):
    """RGBA uint8 array of the official logo at an exact pixel height."""
    src = (LOGO_BLANCO if white else LOGO_AZUL).read_bytes()
    w = int(round(height * ASPECT))
    png = cairosvg.svg2png(bytestring=src, output_width=w, output_height=int(height))
    import io
    return np.array(Image.open(io.BytesIO(png)).convert("RGBA"))


@lru_cache(None)
def raster_cached(height, white=False):
    return raster(height, white)


def _sub_mask(i, height):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{_VB[0]} {_VB[1]} {_VB[2]} {_VB[3]}">'
           f'<path fill-rule="evenodd" fill="#000" d="{_SUBS[i]}"/></svg>')
    w = int(round(height * ASPECT))
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=w, output_height=height)
    import io
    return np.array(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3] > 100


def _order_skeleton(mask):
    """Longest path through the skeleton of a stroke, as (N,2) xy points."""
    from skimage.morphology import skeletonize
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import coo_matrix

    sk = skeletonize(mask)
    ys, xs = np.nonzero(sk)
    n = len(xs)
    if n < 3:
        return np.stack([xs, ys], 1).astype(float)
    index = -np.ones(sk.shape, int)
    index[ys, xs] = np.arange(n)
    rows, cols, wts = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < sk.shape[0]) & (xx >= 0) & (xx < sk.shape[1])
            j = np.full(n, -1)
            j[ok] = index[yy[ok], xx[ok]]
            ok = j >= 0
            rows.append(np.arange(n)[ok]); cols.append(j[ok]); wts.append(np.full(ok.sum(), np.hypot(dx, dy)))
    G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    d0 = shortest_path(G, indices=[0], directed=False)[0]
    d0[np.isinf(d0)] = -1
    a = int(np.argmax(d0))
    da, pred = shortest_path(G, indices=[a], directed=False, return_predecessors=True)
    da = da[0]
    da[np.isinf(da)] = -1
    b = int(np.argmax(da))
    path = [b]
    while path[-1] != a:
        path.append(pred[0, path[-1]])
    P = np.stack([xs[path], ys[path]], 1).astype(float)
    # smooth
    k = 7
    if len(P) > 2 * k:
        ker = np.ones(k) / k
        Ps = np.stack([np.convolve(np.pad(P[:, j], (k // 2, k // 2), mode="edge"), ker, mode="valid") for j in range(2)], 1)
        P = Ps
    return P


def _orient(P, mode, width):
    if mode == "inner_first":
        # the inner tip of each crown stroke is the end closest to the logo's vertical centre line
        cx = 0.5 * width
        return P if abs(P[0, 0] - cx) < abs(P[-1, 0] - cx) else P[::-1]
    if mode == "top_first":
        return P if P[0, 1] < P[-1, 1] else P[::-1]
    return P if P[0, 0] < P[-1, 0] else P[::-1]


@lru_cache(None)
def trace(height):
    """Centre-line trace of the logo at a given pixel height.

    Returns (pts (N,2) in logo-pixel coords, u (N,) global 0..1 param,
             on (N,) 1 on a stroke / 0 on a bridge between strokes,
             umap (H,W) per-pixel reveal param (nan outside the logo))."""
    ref = 900
    s = height / ref
    allp, onv = [], []
    masks = {}
    for i in ORDER:
        m = _sub_mask(i, ref)
        masks[i] = m
        P = _orient(_order_skeleton(m), DIRECTION[i], m.shape[1])
        if i not in (8, 7, 6):
            # threads: draw across their width at mid-height
            ys, xs = np.nonzero(m)
            y = ys.mean()
            P = np.array([[xs.min() + 2, y], [xs.max() - 2, y]], float)
        if allp:
            a, b = allp[-1][-1], P[0]
            bridge = np.linspace(a, b, 8)[1:-1]
            # bow the bridge slightly so the light visibly travels
            nrm = np.array([-(b - a)[1], (b - a)[0]])
            nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
            bridge = bridge + nrm * np.sin(np.linspace(0, np.pi, len(bridge)))[:, None] * 12
            allp.append(bridge)
            onv.append(np.zeros(len(bridge)))
        allp.append(P)
        onv.append(np.ones(len(P)))
    pts = np.vstack(allp)
    on = np.concatenate(onv)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # bridges count less towards time (the light jumps quickly between strokes)
    w = np.where((on[1:] + on[:-1]) > 1, 1.0, 0.35)
    cum = np.concatenate([[0], np.cumsum(seg * w)])
    u = cum / cum[-1]
    # per-pixel reveal map: nearest centre-line point of the same stroke
    from scipy.spatial import cKDTree
    H, Wd = masks[ORDER[0]].shape
    umap = np.full((H, Wd), np.nan, np.float32)
    start = 0
    for k, i in enumerate(ORDER):
        idx0 = sum(len(a) for a in allp[:2 * k])
        Pi = allp[2 * k]
        ui = u[idx0: idx0 + len(Pi)]
        ys, xs = np.nonzero(masks[i])
        _, nn = cKDTree(Pi).query(np.stack([xs, ys], 1))
        umap[ys, xs] = ui[nn]
    # fill every other pixel with its nearest stroke value (covers anti-aliased edges)
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(np.isnan(umap), return_indices=True)
    umap = umap[iy, ix]
    from PIL import Image as _I
    um = _I.fromarray(umap).resize((int(round(height * ASPECT)), int(height)), _I.NEAREST)
    umap_s = np.array(um)
    return pts * s, u, on, umap_s


def revealed(height, head, white=False, soft=0.02):
    """Logo raster with only the part traced up to `head` (0..1) visible.
    head >= 1 returns the untouched official raster (pixel exact)."""
    img = raster_cached(height, white)
    if head >= 1.0:
        return img
    _, _, _, umap = trace(height)
    a = np.clip((head - umap) / soft + 0.5, 0, 1)
    out = img.copy()
    out[..., 3] = (out[..., 3].astype(np.float32) * a).astype(np.uint8)
    return out
