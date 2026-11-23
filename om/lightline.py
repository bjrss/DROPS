"""THE light line.  One function draws every appearance of the line in the film:
the logo trace, the canal file, the incision and the stitches, the implant
thread, the scan, the contact points, the smile curve and the return to the
logo.  Everything else only decides *where* the line is (a polyline in pixels)
and how much of it is lit (tail..head).

Look: a white-blue core #CFEAFF with a halo in the brand blue #729FBB, a bright
spark at the head and a soft fade towards the tail.
"""
import numpy as np
import skia

from .config import C


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def resample(pts, n=None, spacing=3.0):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts, np.zeros(len(pts))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    if L <= 1e-6:
        return pts[:1], np.zeros(1)
    if n is None:
        n = max(2, int(L / spacing) + 1)
    u = np.linspace(0, L, n)
    out = np.stack([np.interp(u, cum, pts[:, j]) for j in range(pts.shape[1])], 1)
    return out, u / L


def _paint(color, alpha, blur=0.0, width=1.0, stroke=True):
    r, g, b = _rgb(color) if isinstance(color, str) else color
    p = skia.Paint(AntiAlias=True, Color=skia.Color(int(r), int(g), int(b), int(np.clip(alpha, 0, 1) * 255)))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(width)
        p.setStrokeCap(skia.Paint.kRound_Cap)
        p.setStrokeJoin(skia.Paint.kRound_Join)
    if blur > 0.3:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    p.setBlendMode(skia.BlendMode.kSrcOver)
    return p


def draw_lightline(canvas, pts, head=1.0, tail=0.0, width=5.0, intensity=1.0, core=None, halo=None,
                   spark=True, fade_tail=0.35, vis=None, dash=None, closed=False, scale=1.0):
    """Draw the lit span [tail, head] (fractions of arc length) of polyline `pts` (pixels).

    vis     optional per-point visibility 0..1 (occlusion / depth fade)
    dash    (on, off) in pixels for dashed "ghost" outlines
    scale   output-size factor (1.0 = 1080 px short side)
    """
    if intensity <= 0.004 or head <= tail:
        return
    core = core or C["line_core"]
    halo = halo or C["line_halo"]
    pts = np.asarray(pts, float)
    if closed and len(pts) > 2:
        pts = np.vstack([pts, pts[:1]])
    w = width * scale
    if len(pts) == 1:
        _dot(canvas, pts[0], w, intensity, core, halo)
        return
    P, u = resample(pts, spacing=max(1.5, w * 0.6))
    if vis is not None:
        V = np.interp(u, np.linspace(0, 1, len(vis)), np.asarray(vis, float))
    else:
        V = np.ones(len(P))
    m = (u >= tail) & (u <= head)
    if m.sum() < 2:
        if spark and 0 <= head <= 1:
            k = int(np.argmin(np.abs(u - head)))
            _dot(canvas, P[k], w, intensity * V[k], core, halo)
        return
    idx = np.where(m)[0]
    span = max(head - tail, 1e-6)
    # alpha along the span: fade near the tail, full at the head
    rel = (u[idx] - tail) / span
    A = np.clip(rel / max(fade_tail, 1e-3), 0, 1) if fade_tail > 0 else np.ones(len(idx))
    A = A * V[idx]
    # draw in chunks so alpha can vary along the path
    chunk = 6
    layers = [  # (color, alpha mul, width mul, blur mul)
        (halo, 0.26, 7.0, 3.2),
        (halo, 0.45, 3.2, 1.3),
        (core, 0.80, 1.6, 0.45),
        (core, 1.00, 0.62, 0.0),
    ]
    for color, am, wm, bm in layers:
        for s in range(0, len(idx) - 1, chunk):
            e = min(s + chunk + 1, len(idx))
            a = float(A[s:e].mean()) * intensity * am
            if a <= 0.004:
                continue
            path = skia.Path()
            path.moveTo(*P[idx[s]])
            for k in idx[s + 1:e]:
                path.lineTo(*P[k])
            paint = _paint(color, a, blur=w * bm, width=max(w * wm, 0.6))
            if dash:
                paint.setPathEffect(skia.DashPathEffect.Make([dash[0] * scale, dash[1] * scale], 0.0))
            canvas.drawPath(path, paint)
    if spark and head < 1.0 + 1e-6:
        k = idx[-1]
        _dot(canvas, P[k], w * 1.25, intensity * V[k], core, halo)


def _dot(canvas, p, w, intensity, core, halo):
    if intensity <= 0.004:
        return
    x, y = float(p[0]), float(p[1])
    canvas.drawCircle(x, y, w * 4.2, _paint(halo, 0.30 * intensity, blur=w * 3.0, stroke=False))
    canvas.drawCircle(x, y, w * 1.6, _paint(core, 0.75 * intensity, blur=w * 0.9, stroke=False))
    canvas.drawCircle(x, y, w * 0.62, _paint("#FFFFFF", 0.95 * intensity, stroke=False))


def draw_dots(canvas, pts, size=2.2, intensity=1.0, core=None, halo=None, scale=1.0, vis=None):
    """Point clouds (scan mesh, contacts, sparks) use the same look as the line head."""
    core = core or C["line_core"]
    halo = halo or C["line_halo"]
    pts = np.asarray(pts, float)
    if len(pts) == 0 or intensity <= 0.004:
        return
    vis = np.ones(len(pts)) if vis is None else np.asarray(vis, float)
    w = size * scale
    # batch: halo + core as point sets grouped by alpha bucket
    for lo in np.linspace(0, 1, 6)[:-1]:
        sel = (vis > lo) & (vis <= lo + 0.2 + 1e-9)
        if not sel.any():
            continue
        a = (lo + 0.1) * intensity
        P = [skia.Point(float(x), float(y)) for x, y in pts[sel]]
        hp = _paint(halo, 0.35 * a, blur=w * 1.6, width=w * 3.0)
        canvas.drawPoints(skia.Canvas.kPoints_PointMode, P, hp)
        cp = _paint(core, 0.95 * a, blur=0.0, width=w)
        canvas.drawPoints(skia.Canvas.kPoints_PointMode, P, cp)
