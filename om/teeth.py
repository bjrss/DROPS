"""Procedural tooth shapes (SDF), in a local frame:

    x = mesio-distal (+x = distal), y = bucco-lingual (-y = buccal / outward),
    z = up for a LOWER tooth (crown up, roots down), origin at the CEJ centre.

Upper teeth reuse the same shapes mirrored in z by the placement transform.
"""
import numpy as np

from .sdf import ellipsoid, round_box, round_cone, chain, sphere, smin, smax


def tube(P, pts, radii, n=7, k=0.0):
    """Smooth tapered tube through control points (Catmull-Rom, radius interpolated)."""
    from .sdf import catmull
    C = catmull(pts, n_per=max(2, n // (len(pts) - 1)))
    u = np.linspace(0, 1, len(C))
    uc = np.linspace(0, 1, len(radii))
    R = np.interp(u, uc, radii)
    return chain(P, C, R, k)


def _rooted(P, pts, radii, flat=1.9, k=0.07):
    """Root, flattened mesio-distally (wide bucco-lingually), smoothly tapered."""
    Q = P.copy()
    Q[..., 1] = Q[..., 1] / flat
    return tube(Q, [(p[0], p[1] / flat, p[2]) for p in pts], radii, n=8)


# ----------------------------------------------------------------- molar
MOLAR = dict(w=1.10, b=1.00, hc=0.75, lr=1.42)


def molar_outer(P, hc=0.75, lr=1.42, crown_only=False):
    body = round_box(P, (0, 0, 0.34), (0.52, 0.46, 0.27), 0.21)
    trunk = round_box(P, (0, 0, -0.10), (0.43, 0.38, 0.30), 0.17)
    cz = hc - 0.21
    cusps = None
    for cx, cy, r in [(-0.31, -0.23, 0.205), (0.04, -0.26, 0.205), (0.37, -0.12, 0.18),
                      (-0.28, 0.22, 0.205), (0.16, 0.23, 0.20)]:
        s = sphere(P, (cx, cy, cz), r)
        cusps = s if cusps is None else smin(cusps, s, 0.06)
    crown = smin(body, cusps, 0.13)
    crown = smin(crown, trunk, 0.16)
    # occlusal anatomy: central fissure + buccal / lingual grooves
    crown = smax(crown, -ellipsoid(P, (0.03, 0.0, hc + 0.035), (0.44, 0.055, 0.13)), 0.035)
    crown = smax(crown, -ellipsoid(P, (-0.13, -0.18, hc - 0.02), (0.045, 0.22, 0.10)), 0.03)
    crown = smax(crown, -ellipsoid(P, (0.22, -0.18, hc - 0.03), (0.04, 0.2, 0.09)), 0.03)
    crown = smax(crown, -ellipsoid(P, (-0.05, 0.2, hc - 0.02), (0.045, 0.2, 0.10)), 0.03)
    if crown_only:
        return crown
    mes = _rooted(P, [(-0.29, 0.0, -0.25), (-0.33, 0.0, -0.80), (-0.27, 0.0, -lr + 0.05)], [0.20, 0.15, 0.085, 0.055])
    dis = _rooted(P, [(0.29, 0.0, -0.25), (0.34, 0.0, -0.78), (0.40, 0.0, -lr + 0.12)], [0.20, 0.14, 0.08, 0.055])
    roots = smin(mes, dis, 0.05)
    return smin(crown, roots, 0.14)


MOLAR_CANALS = [
    # mesio-buccal, mesio-lingual, distal  (orifice -> apex)
    [(-0.26, -0.11, 0.06), (-0.30, -0.12, -0.55), (-0.30, -0.10, -1.05), (-0.27, -0.07, -1.38)],
    [(-0.26, 0.11, 0.06), (-0.30, 0.12, -0.55), (-0.30, 0.10, -1.05), (-0.27, 0.07, -1.38)],
    [(0.27, 0.0, 0.06), (0.33, 0.0, -0.55), (0.36, 0.0, -1.0), (0.39, 0.0, -1.30)],
]
MOLAR_CANAL_R = [[0.062, 0.045, 0.03, 0.016], [0.062, 0.045, 0.03, 0.016], [0.075, 0.055, 0.035, 0.018]]


def molar_pulp(P):
    ch = round_box(P, (0.0, 0.0, 0.16), (0.30, 0.25, 0.11), 0.09)
    for cx, cy in [(-0.2, -0.15), (0.03, -0.17), (-0.18, 0.14), (0.12, 0.15)]:
        ch = smin(ch, round_cone(P, (cx, cy, 0.18), (cx * 1.15, cy * 1.15, 0.36), 0.08, 0.035), 0.05)
    for pts, rr in zip(MOLAR_CANALS, MOLAR_CANAL_R):
        ch = smin(ch, tube(P, pts, rr, n=9), 0.07)
    return ch


def molar_canal_tube(P, i, grow=0.0):
    return tube(P, MOLAR_CANALS[i], [r + grow for r in MOLAR_CANAL_R[i]], n=9)


# ----------------------------------------------------------------- premolar
def premolar_outer(P, hc=0.80, lr=1.45):
    body = round_box(P, (0, 0, 0.36), (0.33, 0.37, 0.30), 0.21)
    trunk = round_box(P, (0, 0, -0.08), (0.26, 0.31, 0.28), 0.15)
    cb = sphere(P, (0.0, -0.13, hc - 0.23), 0.23)
    cl = sphere(P, (0.0, 0.18, hc - 0.36), 0.16)
    crown = smin(smin(body, cb, 0.12), cl, 0.1)
    crown = smin(crown, trunk, 0.15)
    crown = smax(crown, -ellipsoid(P, (0.0, 0.05, hc - 0.05), (0.2, 0.05, 0.08)), 0.03)
    root = _rooted(P, [(0.0, 0.0, -0.2), (0.02, 0.0, -0.8), (0.05, 0.0, -lr + 0.05)], [0.21, 0.15, 0.085, 0.05], flat=1.45)
    return smin(crown, root, 0.14)


# ----------------------------------------------------------------- canine
def canine_outer(P, hc=1.0, lr=1.6):
    body = round_box(P, (0, 0, 0.40), (0.33, 0.35, 0.34), 0.22)
    tip = round_cone(P, (0.0, -0.04, 0.55), (0.0, -0.08, hc - 0.07), 0.29, 0.07)
    crown = smin(body, tip, 0.18)
    trunk = round_box(P, (0, 0, -0.06), (0.24, 0.30, 0.26), 0.15)
    crown = smin(crown, trunk, 0.14)
    root = _rooted(P, [(0.0, 0.0, -0.2), (0.0, 0.02, -0.9), (0.03, 0.04, -lr + 0.05)], [0.21, 0.16, 0.09, 0.05], flat=1.35)
    return smin(crown, root, 0.14)


# ----------------------------------------------------------------- incisor
def incisor_outer(P, hc=0.90, lr=1.25):
    Q = P.copy()
    z = np.clip(P[..., 2] / hc, -0.2, 1.0)
    thin = np.clip(1.0 - 0.62 * np.maximum(z, 0.0), 0.36, 1.0)
    Q[..., 1] = (P[..., 1] + 0.03 * np.maximum(z, 0) ** 2) / thin
    crown = round_box(Q, (0, 0, 0.42), (0.27, 0.27, 0.42), 0.2)
    crown = crown * 0.75
    trunk = round_box(P, (0, 0, -0.05), (0.2, 0.26, 0.22), 0.13)
    crown = smin(crown, trunk, 0.12)
    root = _rooted(P, [(0.0, 0.0, -0.15), (0.0, 0.02, -0.7), (0.0, 0.04, -lr + 0.04)], [0.19, 0.13, 0.075, 0.045], flat=1.5)
    return smin(crown, root, 0.12)


KINDS = {
    "incisor": dict(fn=incisor_outer, hc=0.90, lr=1.25, ext=(0.36, 0.40, 1.0)),
    "canine": dict(fn=canine_outer, hc=1.00, lr=1.60, ext=(0.42, 0.45, 1.1)),
    "premolar": dict(fn=premolar_outer, hc=0.80, lr=1.45, ext=(0.42, 0.47, 0.9)),
    "molar": dict(fn=molar_outer, hc=0.75, lr=1.42, ext=(0.62, 0.56, 0.85)),
}

# FDI tooth -> (kind, mesiodistal width cm, scale factors (x, y, z))
LOWER = {
    1: ("incisor", 0.54, (0.92, 0.95, 0.95)),
    2: ("incisor", 0.59, (1.0, 1.0, 1.0)),
    3: ("canine", 0.70, (1.0, 1.0, 1.0)),
    4: ("premolar", 0.70, (1.0, 1.0, 1.0)),
    5: ("premolar", 0.71, (1.02, 1.02, 0.98)),
    6: ("molar", 1.12, (1.0, 1.0, 1.0)),
    7: ("molar", 1.05, (0.95, 0.98, 0.97)),
    8: ("molar", 1.00, (0.91, 0.95, 0.95)),
}
UPPER = {
    1: ("incisor", 0.86, (1.48, 1.12, 1.18)),
    2: ("incisor", 0.66, (1.15, 1.0, 1.02)),
    3: ("canine", 0.79, (1.1, 1.08, 1.02)),
    4: ("premolar", 0.70, (1.0, 1.15, 0.98)),
    5: ("premolar", 0.68, (0.97, 1.13, 0.96)),
    6: ("molar", 1.04, (0.95, 1.1, 1.0)),
    7: ("molar", 0.95, (0.87, 1.05, 0.97)),
}
