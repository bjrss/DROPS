"""The 3D world in plain numpy: dental arches, tooth placements, jaw SDFs and
key landmarks.  Imported by the Blender build (to make meshes) and by the 2D
compositor (to project light-line paths), so both always agree.

World frame: x = patient's left (+) / right (-), y = posterior (+), z = up.
Occlusal plane at z = 0.  Units: cm.
"""
from functools import lru_cache

import numpy as np

from . import teeth as T
from .sdf import catmull, smin, smax, length, round_box

# ------------------------------------------------------------------ arch curve
# right half of the LOWER arch centre-line through the tooth centres (x >= 0)
_LOWER_CTRL = np.array([
    (0.00, 0.00), (0.52, 0.10), (1.00, 0.40), (1.40, 0.86), (1.72, 1.45),
    (1.98, 2.15), (2.18, 2.95), (2.32, 3.80), (2.42, 4.70), (2.50, 5.60), (2.56, 6.50),
])


@lru_cache(None)
def arch(upper=False):
    """Dense polyline (x>=0 half) and its cumulative arc length."""
    C = catmull(_LOWER_CTRL, n_per=40)
    if upper:
        C = C * np.array([1.07, 1.0]) + np.array([0.0, -0.13])
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    return C, s


def arch_point(s, upper=False):
    """Point + unit tangent (pointing distally) at signed arc length s
    (s>0: patient's left / +x, s<0: right / -x)."""
    C, S = arch(upper)
    a = abs(s)
    x = np.interp(a, S, C[:, 0])
    y = np.interp(a, S, C[:, 1])
    i = min(np.searchsorted(S, a), len(S) - 1)
    i0 = max(i - 1, 0)
    t = C[min(i0 + 1, len(C) - 1)] - C[i0]
    t = t / (np.linalg.norm(t) + 1e-12)
    if s < 0:
        x = -x
        t = np.array([-t[0], t[1]])
    return np.array([x, y]), t


def tooth_frame(s, upper=False):
    """4x4 matrix: local tooth frame -> world.  local x = distal, y = lingual, z = occlusal->up(lower)."""
    p, t = arch_point(s, upper)
    tx = np.array([t[0], t[1], 0.0])
    out = np.array([t[1], -t[0], 0.0]) if s >= 0 else np.array([-t[1], t[0], 0.0])
    ty = -out
    tz = np.array([0, 0, 1.0])
    M = np.eye(4)
    M[:3, 0] = tx
    M[:3, 1] = ty
    M[:3, 2] = tz
    M[:3, 3] = (p[0], p[1], 0.0)
    if upper:  # mirror z so the crown points down
        M[:3, 2] = -tz
    return M


# ------------------------------------------------------------------ teeth
LOWER_CEJ = {"incisor": -0.86, "canine": -0.98, "premolar": -0.80, "molar": -0.74}
UPPER_CEJ = {"incisor": 1.02, "canine": 1.06, "premolar": 0.84, "molar": 0.76}

# special sites
SITE_ENDO = 36      # lower left first molar
SITE_WISDOM = 38    # impacted lower left third molar
SITE_IMPLANT = 46   # missing lower right first molar


@lru_cache(None)
def placements():
    """List of dicts: fdi, kind, upper, M (4x4 local->world), scale (3,), s (arc)."""
    out = []
    for upper, table, ceji in ((False, T.LOWER, LOWER_CEJ), (True, T.UPPER, UPPER_CEJ)):
        quads = (2, 1) if upper else (3, 4)
        for qi, quad in enumerate(quads):
            sign = 1 if qi == 0 else -1
            acc = 0.0
            for n in range(1, 9):
                if n not in table:
                    continue
                kind, w, sc = table[n]
                sc = np.array(sc, float)
                s_c = acc + w / 2
                acc += w
                fdi = quad * 10 + n
                M = tooth_frame(sign * s_c, upper)
                if sign < 0:  # mirror mesio-distal so roots curve distally on both sides
                    M[:3, 0] *= -1
                z = ceji[kind] * (sc[2] if not upper else 1.0)
                M[:3, 3] += np.array([0, 0, z])
                d = dict(fdi=fdi, kind=kind, upper=upper, s=sign * s_c, scale=sc, width=w, M=M,
                         present=True, impacted=False)
                if fdi == SITE_IMPLANT:
                    d["present"] = False
                if fdi == SITE_WISDOM:
                    d["impacted"] = True
                    R = _rot_local_y(np.radians(-44))  # tip the crown mesially
                    M2 = M.copy()
                    M2[:3, :3] = M[:3, :3] @ R
                    M2[:3, 3] += M[:3, 2] * -0.78 - M[:3, 0] * 0.12  # sunk in bone, slid mesially
                    d["M"] = M2
                out.append(d)
    return out


def tooth(fdi):
    for d in placements():
        if d["fdi"] == fdi:
            return d
    raise KeyError(fdi)


def _rot_local_y(a):
    """Rotation about the local bucco-lingual axis (y): positive tips crown distally."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def to_local(P, M, scale=None):
    """World points (...,3) -> tooth-local (unscaled) coords."""
    R = M[:3, :3]
    t = M[:3, 3]
    Q = (P - t) @ R  # R orthonormal (possibly mirrored) -> inverse = R^T
    if scale is not None:
        Q = Q / np.asarray(scale)
    return Q


def to_world(Q, M, scale=None):
    Q = np.asarray(Q, float)
    if scale is not None:
        Q = Q * np.asarray(scale)
    return Q @ M[:3, :3].T + M[:3, 3]


def tooth_sdf_world(P, d, dilate=0.0):
    k = T.KINDS[d["kind"]]
    Q = to_local(P, d["M"], d["scale"])
    return k["fn"](Q) * float(np.min(d["scale"])) - dilate


def teeth_union(P, filt, dilate=0.0, pad=0.25):
    """Union SDF of selected teeth, evaluated only inside each tooth's bounds."""
    D = np.full(P.shape[:-1], 10.0)
    for d in placements():
        if not filt(d):
            continue
        k = T.KINDS[d["kind"]]
        ex = np.array(k["ext"]) * d["scale"] + pad
        Q = to_local(P, d["M"])
        lo = np.array([-ex[0], -ex[1], -k["lr"] * d["scale"][2] - pad])
        hi = np.array([ex[0], ex[1], k["hc"] * d["scale"][2] + pad])
        m = np.all((Q >= lo) & (Q <= hi), axis=-1)
        if not m.any():
            continue
        D[m] = np.minimum(D[m], tooth_sdf_world(P[m], d, dilate))
    return D


# ------------------------------------------------------------------ jaw
def _arch_coords(P, upper=False):
    """For world points: signed arc length s, lateral offset d (+ = outward/buccal)."""
    from scipy.spatial import cKDTree

    C, S = arch(upper)
    full = np.vstack([C[::-1] * [-1, 1], C[1:]])
    Sf = np.concatenate([-S[::-1], S[1:]])
    tree = _tree(upper)
    q = P[..., :2].reshape(-1, 2)
    dist, idx = tree.query(q)
    idx = np.clip(idx, 1, len(full) - 2)
    tang = full[idx + 1] - full[idx - 1]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)  # points outward for the +x half when moving distally
    sgn_side = np.where(Sf[idx] >= 0, 1.0, -1.0)
    rel = q - full[idx]
    d = np.sum(rel * nrm, axis=1) * sgn_side
    s = Sf[idx] + np.sum(rel * tang, axis=1)
    return s.reshape(P.shape[:-1]), d.reshape(P.shape[:-1])


_TREES = {}


def _tree(upper):
    from scipy.spatial import cKDTree

    if upper not in _TREES:
        C, S = arch(upper)
        full = np.vstack([C[::-1] * [-1, 1], C[1:]])
        _TREES[upper] = cKDTree(full)
    return _TREES[upper]


S_END = 6.9  # arch length at the distal end (behind the third molars)


def _interp_s(s, table):
    xs = np.array([a for a, _ in table])
    ys = np.array([b for _, b in table])
    return np.interp(np.abs(s), xs, ys)


def bone_sdf(P, sockets=True):
    """Lower alveolar bone + mandible body."""
    s, d = _arch_coords(P)
    z = P[..., 2]
    crest = _interp_s(s, [(0, -1.02), (1.3, -1.12), (2.5, -0.98), (6.0, -0.94), (S_END, -1.0)])
    # healed ridge at the implant site is a little lower
    crest = crest - 0.1 * np.exp(-((s - tooth(SITE_IMPLANT)["s"]) / 0.45) ** 2)
    half_top = _interp_s(s, [(0, 0.36), (1.5, 0.42), (3.0, 0.52), (6.0, 0.56), (S_END, 0.56)])
    bottom = -3.05
    # width grows towards the base (mandible body)
    k = np.clip((crest - z) / 1.6, 0, 1)
    half = half_top + 0.28 * k
    # 2D rounded box in (d, z)
    cz = (crest + bottom) / 2
    hz = (crest - bottom) / 2
    q0 = np.abs(d) - (half - 0.3)
    q1 = np.abs(z - cz) - (hz - 0.3)
    q = np.stack([q0, q1], -1)
    D = length(np.maximum(q, 0)) + np.minimum(np.maximum(q0, q1), 0) - 0.3
    D = smax(D, np.abs(s) - S_END, 0.4)
    if sockets:
        T_ = teeth_union(P, lambda t: (not t["upper"]) and t["present"], dilate=0.012)
        D = smax(D, -T_, 0.01)
    return D


def gum_outer_sdf(P):
    """Outer soft-tissue envelope of the lower jaw (before subtracting teeth)."""
    s, d = _arch_coords(P)
    z = P[..., 2]
    base = _interp_s(s, [(0, -0.80), (1.3, -0.92), (2.5, -0.74), (6.0, -0.70), (S_END, -0.78)])
    # papillae between erupted teeth
    pap = np.zeros_like(s)
    lows = [t for t in placements() if not t["upper"]]
    edges = []
    for t in lows:
        edges += [t["s"] - t["width"] / 2, t["s"] + t["width"] / 2]
    for e in sorted(set(np.round(edges, 3))):
        pap += 0.26 * np.exp(-((s - e) / 0.11) ** 2)
    imp = tooth(SITE_IMPLANT)["s"]
    wis = tooth(SITE_WISDOM)
    # no papillae over the missing tooth / impacted wisdom tooth: smooth ridge
    flat = np.exp(-((s - imp) / 0.5) ** 2) + np.exp(-((s - wis["s"]) / 0.45) ** 2)
    pap = pap * np.clip(1 - flat, 0, 1)
    top = base + np.minimum(pap, 0.3) - 0.10 * np.exp(-((s - imp) / 0.45) ** 2)
    top = top + 0.02 * np.exp(-((s - wis["s"]) / 0.5) ** 2)
    # gum = alveolar band: as thick as needed near the crest, thinning out ~1.2 cm below
    bone_crest = _interp_s(s, [(0, -1.02), (1.3, -1.12), (2.5, -0.98), (6.0, -0.94), (S_END, -1.0)])
    bone_half_top = _interp_s(s, [(0, 0.36), (1.5, 0.42), (3.0, 0.52), (6.0, 0.56), (S_END, 0.56)])
    kb = np.clip((bone_crest - z) / 1.6, 0, 1)
    thick = np.interp(top - z, [0.0, 0.75, 1.35, 3.0], [0.13, 0.12, -0.06, -0.06])
    half_top = bone_half_top
    bottom = -3.25
    half = bone_half_top + 0.28 * kb + thick
    cz = (top + bottom) / 2
    hz = (top - bottom) / 2
    r = 0.34
    q0 = np.abs(d) - (half - r)
    q1 = np.abs(z - cz) - (hz - r)
    q = np.stack([q0, q1], -1)
    D = length(np.maximum(q, 0)) + np.minimum(np.maximum(q0, q1), 0) - r
    D = smax(D, np.abs(s) - (S_END + 0.15), 0.45)
    # cuffs hugging the erupted teeth necks
    cuff = teeth_union(P, lambda t: (not t["upper"]) and t["present"] and not t["impacted"], dilate=0.07)
    neck = np.abs(z - (base + 0.05)) - 0.2
    D = smin(D, smax(cuff, neck, 0.05), 0.08)
    return D


def gum_sdf(P):
    """Gum layer = envelope minus bone minus teeth."""
    G = gum_outer_sdf(P)
    B = bone_sdf(P, sockets=False)
    Tt = teeth_union(P, lambda t: (not t["upper"]) and t["present"], dilate=0.004)
    D = smax(G, -B, 0.0)
    D = smax(D, -Tt, 0.0)
    return D


def upper_gum_sdf(P):
    s, d = _arch_coords(P, upper=True)
    z = P[..., 2]
    base = _interp_s(s, [(0, 1.10), (1.3, 1.16), (2.5, 0.96), (6.0, 0.88), (S_END, 0.95)])
    pap = np.zeros_like(s)
    ups = [t for t in placements() if t["upper"]]
    edges = []
    for t in ups:
        edges += [t["s"] - t["width"] / 2, t["s"] + t["width"] / 2]
    for e in sorted(set(np.round(edges, 3))):
        pap += 0.26 * np.exp(-((s - e) / 0.11) ** 2)
    bottom_edge = base - np.minimum(pap, 0.3)
    half = _interp_s(s, [(0, 0.55), (1.5, 0.62), (3.0, 0.74), (6.2, 0.78), (S_END, 0.78)])
    top = 2.6
    cz = (top + bottom_edge) / 2
    hz = (top - bottom_edge) / 2
    r = 0.34
    q0 = np.abs(d) - (half - r)
    q1 = np.abs(z - cz) - (hz - r)
    q = np.stack([q0, q1], -1)
    D = length(np.maximum(q, 0)) + np.minimum(np.maximum(q0, q1), 0) - r
    D = smax(D, np.abs(s) - 6.2, 0.45)
    cuff = teeth_union(P, lambda t: t["upper"], dilate=0.07)
    neck = np.abs(z - (base - 0.05)) - 0.2
    D = smin(D, smax(cuff, neck, 0.05), 0.08)
    Tt = teeth_union(P, lambda t: t["upper"], dilate=0.004)
    return smax(D, -Tt, 0.0)


# ------------------------------------------------------------------ sites
def site_frame(fdi):
    """Frame at a site (on the arch centre-line, at the bone crest height):
    x = distal tangent, y = lingual, z = up.  Used for cut-aways and cameras."""
    t = tooth(fdi)
    M = tooth_frame(t["s"])
    if t["s"] < 0:
        M[:3, 0] *= -1  # keep it right-handed for cutters (x along +s direction)
    M[:3, 3] = np.array([M[0, 3], M[1, 3], -0.9])
    return M


CUTS = {
    # name: (fdi, half length along arch, offset along arch)
    "cutA": (SITE_ENDO, 0.95, 0.0),
    "cutB": (SITE_WISDOM, 0.95, -0.05),
    "cutC": (SITE_IMPLANT, 0.95, 0.0),
}


def cut_matrix(name):
    fdi, hl, off = CUTS[name]
    M = site_frame(fdi)
    M[:3, 3] = M[:3, 3] + M[:3, 0] * off
    S = np.diag([hl, 1.0, 1.0, 1.0])
    return M @ S
