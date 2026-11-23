"""Tiny numpy SDF toolkit used to model the anatomy procedurally.

All functions take P with shape (..., 3) and return distances with shape (...).
Units: 1 = 1 cm.
"""
import numpy as np


def length(v):
    return np.sqrt(np.sum(v * v, axis=-1))


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sphere(P, c, r):
    return length(P - np.asarray(c, float)) - r


def ellipsoid(P, c, r):
    r = np.asarray(r, float)
    q = (P - np.asarray(c, float))
    k0 = length(q / r)
    k1 = length(q / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def round_box(P, c, half, rad):
    q = np.abs(P - np.asarray(c, float)) - (np.asarray(half, float) - rad)
    return length(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0) - rad


def round_cone(P, a, b, ra, rb):
    """iq's round cone between points a (radius ra) and b (radius rb)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ba = b - a
    l2 = float(np.dot(ba, ba))
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = np.sum(pa * ba, axis=-1)
    z = y - l2
    xv = pa * l2 - y[..., None] * ba
    x2 = np.sum(xv * xv, axis=-1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    out = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - ra
    c1 = np.sign(z) * a2 * z2 > k
    c2 = np.sign(y) * a2 * y2 < k
    out = np.where(c1, np.sqrt(x2 + z2) * il2 - rb, out)
    out = np.where(c2 & ~c1, np.sqrt(x2 + y2) * il2 - ra, out)
    return out


def chain(P, pts, radii, k=0.02):
    """Smooth union of round cones through a polyline (tapered tube)."""
    d = None
    for i in range(len(pts) - 1):
        di = round_cone(P, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = di if d is None else smin(d, di, k)
    return d


def cylinder_z(P, c, r, h):
    """Capped cylinder along z centred at c, radius r, half height h."""
    q = P - np.asarray(c, float)
    d = np.stack([np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2) - r, np.abs(q[..., 2]) - h], -1)
    return np.minimum(np.max(d, -1), 0.0) + length(np.maximum(d, 0.0))


def plane(P, p0, n):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return np.sum((P - np.asarray(p0, float)) * n, axis=-1)


def scale_eval(fn, P, s):
    """Evaluate fn on non-uniformly scaled space (approximate distance)."""
    s = np.asarray(s, float)
    return fn(P / s) * float(np.min(s))


def poly_sample(pts, n):
    """Resample a polyline to n points, uniform arc length."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, cum[-1], n)
    out = np.empty((n, pts.shape[1]))
    for j in range(pts.shape[1]):
        out[:, j] = np.interp(u, cum, pts[:, j])
    return out


def catmull(pts, n_per=12, closed=False):
    """Catmull-Rom spline through control points."""
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-2])
    return np.array(out)
