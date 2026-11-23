"""Everything that moves, as pure functions of time t (seconds).

Blender bakes these into keyframes (one per frame, linear) and the 2D compositor
evaluates the same functions to project the light line, so 3D and 2D never
drift apart.
"""
import math
from functools import lru_cache

import numpy as np

from . import world as W
from .easing import (lin, smooth, smoother, in_out_sine, in_out_cubic, out_cubic, out_quint,
                     in_out_quint, noise1, clamp, keyed)
from .timeline import B

# =================================================================== camera
# key: time -> dict(s, z, dist, el, az, fov, inset)
#   s     arc length of the target on the lower arch (+ = patient's left)
#   z     target height, dist camera distance (cm), el elevation (deg),
#   az    azimuth offset around the outward normal (deg), fov short-side FOV (deg)
_S36 = W.tooth(36)["s"]
_S46 = W.tooth(46)["s"]
_S38 = 5.72

CAM_KEYS = [
    (0.00, dict(s=_S36, z=-1.08, dist=14.6, el=1, az=0, fov=16.8, inset=0.0)),
    (2.55, dict(s=_S36, z=-1.08, dist=14.2, el=1, az=0, fov=16.8, inset=0.0)),
    (3.30, dict(s=_S36, z=-1.06, dist=12.6, el=3, az=-3, fov=16.0, inset=0.0)),
    (4.60, dict(s=_S36, z=-1.02, dist=10.4, el=7, az=-9, fov=15.5, inset=0.0)),
    (7.40, dict(s=_S36, z=-1.00, dist=9.8, el=9, az=-2, fov=15.5, inset=0.0)),
    (9.60, dict(s=_S36, z=-1.02, dist=9.2, el=12, az=5, fov=15.5, inset=0.0)),
    (12.4, dict(s=_S36, z=-0.95, dist=8.9, el=15, az=-3, fov=15.5, inset=0.0)),
    (14.6, dict(s=_S36, z=-0.98, dist=9.8, el=9, az=4, fov=15.5, inset=0.0)),
    (16.7, dict(s=_S38, z=-1.18, dist=10.4, el=9, az=-7, fov=16.5, inset=0.0)),
    (19.5, dict(s=_S38, z=-1.15, dist=9.6, el=11, az=2, fov=16.5, inset=0.0)),
    (22.4, dict(s=_S38, z=-1.05, dist=8.8, el=17, az=-2, fov=16.5, inset=0.0)),
    (24.6, dict(s=_S38, z=-1.10, dist=9.2, el=12, az=4, fov=16.5, inset=0.0)),
    (26.8, dict(s=_S38, z=-1.12, dist=10.0, el=8, az=-3, fov=16.5, inset=0.0)),
    (27.9, dict(s=0.6, z=-0.55, dist=15.5, el=13, az=0, fov=24.0, inset=0.6)),
    (29.0, dict(s=_S46, z=-1.10, dist=10.2, el=7, az=5, fov=16.0, inset=0.0)),
    (32.0, dict(s=_S46, z=-1.05, dist=10.8, el=8, az=-5, fov=16.0, inset=0.0)),
    (33.6, dict(s=_S46, z=-1.15, dist=9.4, el=10, az=1, fov=16.0, inset=0.0)),
    (35.7, dict(s=_S46, z=-1.25, dist=8.9, el=6, az=-2, fov=16.0, inset=0.0)),
    (38.2, dict(s=_S46, z=-1.15, dist=9.6, el=8, az=4, fov=16.0, inset=0.0)),
    (41.0, dict(s=_S46, z=-1.00, dist=9.4, el=10, az=-3, fov=16.0, inset=0.0)),
    (43.2, dict(s=_S46, z=-0.85, dist=8.9, el=19, az=1, fov=16.0, inset=0.0)),
    (45.0, dict(s=_S46, z=-0.70, dist=9.0, el=13, az=-2, fov=16.0, inset=0.0)),
    (47.3, dict(s=_S46, z=-0.30, dist=10.6, el=5, az=7, fov=17.5, inset=0.0)),
    (48.7, dict(s=-3.0, z=-0.10, dist=11.6, el=4, az=5, fov=19.0, inset=0.0)),
    (50.5, dict(s=0.0, z=0.02, dist=15.0, el=3, az=0, fov=23.0, inset=0.9)),
    (51.6, dict(s=0.0, z=0.05, dist=15.8, el=3, az=0, fov=23.0, inset=0.9)),
    (53.2, dict(s=0.0, z=0.3, dist=27.0, el=6, az=0, fov=25.0, inset=0.9)),
    (60.0, dict(s=0.0, z=0.3, dist=29.0, el=6, az=0, fov=25.0, inset=0.9)),
]

FMT_CAM = {
    "9x16": dict(fov_mul=1.0, shift_x=0.0, shift_y=0.0),
    "16x9": dict(fov_mul=1.28, shift_x=0.0, shift_y=0.0),
}


def _cam_params(t):
    keys = CAM_KEYS
    if t <= keys[0][0]:
        return dict(keys[0][1])
    for (ta, a), (tb, b) in zip(keys[:-1], keys[1:]):
        if t <= tb:
            x = in_out_sine(lin(t, ta, tb))
            # longer moves get a stronger ease
            if tb - ta < 2.2:
                x = in_out_cubic(lin(t, ta, tb))
            return {k: a[k] + (b[k] - a[k]) * x for k in a}
    return dict(keys[-1][1])


def _site_shift_16x9(t):
    """In 16:9 the subject sits left of centre while chapter text lives on the right."""
    on = 0.0
    for a, b in ((4.3, 15.4), (16.3, 27.4), (28.4, 39.8), (40.2, 48.4)):
        on = max(on, smooth(lin(t, a, a + 0.9)) * (1 - smooth(lin(t, b, b + 0.9))))
    # opening: the 3D tooth materialises exactly where the logo sits (left of centre in 16:9)
    opening = 1 - smooth(lin(t, 3.6, 4.6))
    return 0.11 * on + 0.132 * opening


def camera(t, fmt="9x16"):
    p = _cam_params(t)
    # constant gentle drift ("la cámara siempre deriva")
    p["az"] += 1.6 * noise1(t * 0.9, 1.0)
    p["el"] += 1.0 * noise1(t * 0.8, 4.0)
    p["dist"] *= 1.0 + 0.012 * noise1(t * 0.7, 9.0)
    pt, tan = W.arch_point(p["s"])
    tan3 = np.array([tan[0], tan[1], 0.0])
    if p["s"] >= 0:
        out = np.array([tan[1], -tan[0], 0.0])
    else:
        out = np.array([-tan[1], tan[0], 0.0])
    if abs(p["s"]) < 1e-6:
        out = np.array([0.0, -1.0, 0.0])
    target = np.array([pt[0], pt[1], p["z"]]) - out * p["inset"]
    az = math.radians(p["az"])
    el = math.radians(p["el"])
    side = np.cross(np.array([0, 0, 1.0]), out)
    d = math.cos(az) * out + math.sin(az) * side
    d = math.cos(el) * d + math.sin(el) * np.array([0, 0, 1.0])
    pos = target + d * p["dist"]
    fc = FMT_CAM[fmt]
    fov = p["fov"] * fc["fov_mul"]
    shift_x = fc["shift_x"] + (_site_shift_16x9(t) if fmt == "16x9" else 0.0)
    # depth of field: focus on the target, aperture opens on close-ups
    fstop = 3.2 if p["dist"] < 12 else 5.6
    return dict(pos=pos, target=target, fov=fov, focus=float(np.linalg.norm(pos - target)) * 0.985,
                fstop=fstop, shift_x=shift_x, shift_y=fc["shift_y"])


def cam_matrix(c):
    f = c["target"] - c["pos"]
    f = f / np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, u, -f, c["pos"]
    return M


def project(P, c, w, h, fmt="9x16"):
    """World points -> pixel coords in a (w, h) image of the given plate format.
    Matches Blender: sensor_fit HORIZONTAL for 9:16 (short side = width),
    VERTICAL for 16:9 (short side = height); shift in units of the long side."""
    P = np.atleast_2d(np.asarray(P, float))
    M = cam_matrix(c)
    R = M[:3, :3]
    q = (P - M[:3, 3]) @ R           # camera space: x right, y up, -z forward
    zf = -q[:, 2]
    short = w if fmt == "9x16" else h
    longs = max(w, h)
    f_px = (short / 2) / math.tan(math.radians(c["fov"]) / 2)
    x = q[:, 0] / np.maximum(zf, 1e-6) * f_px
    y = q[:, 1] / np.maximum(zf, 1e-6) * f_px
    x = x + w / 2 - c["shift_x"] * longs
    y = h / 2 - (y + c["shift_y"] * longs)
    return np.stack([x, y], 1), zf


# =================================================================== object states
def _tm(M, R=None, T=None):
    """Compose: world = T * M * R (R rotates in object space)."""
    out = M.copy()
    if R is not None:
        out = out @ R
    if T is not None:
        out = T @ out
    return out


def rot_axis(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    R = np.eye(4)
    R[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                 [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                 [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
    return R


def translate(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def about(point, R):
    return translate(point) @ R @ translate(-np.asarray(point))


def cut_open(name, t):
    """0 = closed (nothing hidden) .. 1 = fully open."""
    spans = {"cutA": [(B["xray0"] + 0.15, 0.6, B["c1.out"], 0.55)],
             "cutB": [(B["c2.cut"], 0.65, B["c2.out"], 0.5)],
             "cutC": [(B["c3.cut"], 0.6, B["c4.s3"] - 0.25, 0.6)]}
    v = 0.0
    for a, da, b, db in spans[name]:
        v = max(v, in_out_cubic(lin(t, a, a + da)) * (1 - in_out_cubic(lin(t, b, b + db))))
    return v


def cutter_matrix(name, t):
    M = W.cut_matrix(name)
    o = cut_open(name, t)
    a = 2.6 * (1 - o) + (40.0 if o <= 0.0 else 0.0)
    y = M[:3, 1] / np.linalg.norm(M[:3, 1])
    M = M.copy()
    M[:3, 3] = M[:3, 3] - y * a
    return M


@lru_cache(None)
def _site(fdi):
    return W.site_frame(fdi)


def wisdom_matrix(t):
    """Extraction: the wisdom tooth uprights gently and rises out, turning."""
    d = W.tooth(W.SITE_WISDOM)
    M = d["M"] @ np.diag([*d["scale"], 1.0])
    t0 = B["c2.s2"] + 0.95
    x = in_out_sine(lin(t, t0, t0 + 1.25))
    if x <= 0:
        return M
    Mf = _site(W.SITE_WISDOM)
    c = W.to_world(np.array([0, 0, 0.3]), d["M"], d["scale"])
    R = about(c, rot_axis(Mf[:3, 1], math.radians(30) * x) @ rot_axis(Mf[:3, 2], math.radians(55) * x))
    T = translate(np.array([0, 0, 2.3 * x ** 1.4]) + Mf[:3, 0] * 0.25 * x - Mf[:3, 1] * 0.6 * x)
    return T @ R @ M


def flap_matrix(t):
    Mf = _site(W.SITE_WISDOM)
    hinge = Mf[:3, 3] + Mf[:3, 1] * 0.62 + np.array([0, 0, 0.32])
    t_open = B["c2.s2"] + 0.35
    t_close = B["c2.s3"]
    a = in_out_cubic(lin(t, t_open, t_open + 0.5)) * (1 - in_out_cubic(lin(t, t_close, t_close + 0.5)))
    return about(hinge, rot_axis(Mf[:3, 0], math.radians(-105) * a)), a


def implant_matrix(t):
    Mc = _site(W.SITE_IMPLANT)
    Mi = Mc.copy()
    Mi[:3, 3] = Mi[:3, 3] + Mi[:3, 2] * 0.06
    t0, t1 = B["c3.s2"] + 0.35, B["c3.s2"] + 1.75
    x = out_cubic(lin(t, t0, t1))
    depth = 0.55 * (1 - x)
    ang = -2.0 * math.pi * 2.2 * (1 - x)
    R = rot_axis((0, 0, 1), ang)
    return translate(Mc[:3, 2] * depth) @ Mi @ R


def healing_matrix(t):
    Mi = implant_matrix(1e9)
    t_in = B["c3.s3"] + 1.05
    t_out = B["c4.s1"]
    x_in = out_cubic(lin(t, t_in, t_in + 0.6))
    x_out = in_out_cubic(lin(t, t_out, t_out + 0.6))
    up = 0.45 * (1 - x_in) + 0.9 * x_out
    ang = -2 * math.pi * (1 - x_in) - 2 * math.pi * x_out
    return translate(Mi[:3, 2] * up) @ Mi @ rot_axis((0, 0, 1), ang)


def abutment_matrix(t):
    Mi = implant_matrix(1e9)
    t0 = B["c4.s2"]
    x = in_out_quint(lin(t, t0, B["c4.click"]))
    return translate(Mi[:3, 2] * 0.9 * (1 - x)) @ Mi


def crown_matrix(t):
    d = W.tooth(W.SITE_IMPLANT)
    M = d["M"] @ np.diag([*d["scale"], 1.0])
    t0 = B["c4.click"] + 0.55
    x = in_out_cubic(lin(t, t0, B["c4.seat"]))
    return translate(np.array([0, 0, 1.0 * (1 - x)])) @ M


def upper_matrix(t):
    """Opposing arch closes onto the lower one (hinge behind the molars)."""
    t0 = B["c4.s3"] + 0.05
    x = in_out_cubic(lin(t, t0, t0 + 1.05))
    hinge = np.array([0.0, 7.5, 1.2])
    R = rot_axis((1, 0, 0), math.radians(-16) * (1 - x))
    return about(hinge, R) @ translate(np.array([0, 0, 0.55 * (1 - x)]))


def upper_vis(t):
    return 1.0 if t >= B["c4.s3"] - 0.02 else 0.0


def reveal_radius(t):
    """Opening: the jaw and neighbours grow out from the hero tooth."""
    return 0.0 + 14.0 * out_quint(lin(t, B["jaw.reveal0"], B["jaw.reveal1"] + 0.6))


def states(t):
    """name -> dict(props) and optional 'M' (4x4 world matrix)."""
    S = {}
    # ---- hero tooth 36 + x-ray
    appear = in_out_sine(lin(t, B["tooth.in0"], B["tooth.in1"]))
    xr_on = in_out_sine(lin(t, B["xray0"], B["xray1"]))
    xr_off = in_out_sine(lin(t, B["c1.out"] - 0.4, B["c1.out"] + 0.45))
    # wipe: visible where z_local > wipe.  materialise top->down, x-ray removes bottom->up
    wipe = 1.3 - 3.6 * appear
    wipe = wipe + 3.6 * xr_on * (1 - xr_off)
    S["t36"] = dict(vis=1.0 if t >= B["tooth.in0"] - 0.05 else 0.0, wipe=wipe,
                    caries_amt=smooth(lin(t, B["c1.problem"], B["c1.caries"])) * (1 - smooth(lin(t, B["c1.s2"], B["c1.s2"] + 0.4))))
    S["t36_xray"] = dict(xray=xr_on * (1 - xr_off), vis=1.0)
    inflame = smooth(lin(t, B["c1.problem"] + 0.4, B["c1.nervio"] - 0.2))
    beat = 0.5 + 0.5 * math.sin(2 * math.pi * 0.72 * (t - B["c1.problem"]))
    S["pulp36"] = dict(vis=1.0 if (B["xray0"] < t < B["c1.out"] + 1) else 0.0, inflame=inflame,
                       pulse=1.4 * beat ** 2 * inflame,
                       dissolve=smooth(lin(t, B["c1.s2"] + 0.25, B["c1.s2"] + 2.3)))
    S["caries36"] = dict(fill=1.02 * smooth(lin(t, B["c1.problem"] + 0.1, B["c1.caries"] + 0.6)) *
                         (1 - smooth(lin(t, B["c1.s2"] + 0.05, B["c1.s2"] + 0.45))),
                         vis=1.0 if t > B["c1.problem"] else 0.0)
    for i in range(3):
        a = B["c1.s3"] + 0.12 + 0.28 * i
        S[f"gutta36_{i}"] = dict(fill=-0.02 + 1.05 * in_out_sine(lin(t, a, a + 0.55)),
                                 hot=1.0 - smooth(lin(t, a + 0.25, a + 1.0)),
                                 vis=1.0 if t < B["c1.out"] + 1 else 0.0)
    r0 = B["c1.s3"] + 1.05
    S["resto36"] = dict(fill=-0.02 + 1.05 * in_out_sine(lin(t, r0, r0 + 0.45)), hot=0.0)
    # ---- cutters / cut segments
    for name in W.CUTS:
        o = cut_open(name, t)
        S[name] = dict(M=cutter_matrix(name, t))
        v = 1.0 if o > 0.0 else 0.0
        S[f"bone_{name}"] = dict(vis=v)
        S[f"gum_{name}"] = dict(vis=v)
    # ---- surgery
    vB = 1.0 if cut_open("cutB", t) > 0 else 0.0
    Mflap, fa = flap_matrix(t)
    S["flapB"] = dict(M=Mflap, vis=vB if fa < 0.08 else 0.0)
    S["flapBx"] = dict(M=Mflap, vis=1.0, xray=smooth(lin(fa, 0.0, 0.15)) * vB)
    os0 = B["c2.s2"] + 0.6
    S["osteoB"] = dict(fill=1.02 - 1.06 * smooth(lin(t, os0, os0 + 0.45)), vis=vB, hot=0.0)
    gone = B["c2.s2"] + 0.95 + 1.25
    S["t38"] = dict(M=wisdom_matrix(t), vis=1.0, wipe=-10.0 + 13.0 * smooth(lin(t, gone - 0.55, gone - 0.05)))
    f0 = B["c2.s3"] + 0.75
    S["fillB"] = dict(fill=-0.02 + 1.05 * in_out_sine(lin(t, f0, f0 + 1.3)), vis=vB,
                      hot=0.35 * (1 - smooth(lin(t, f0 + 0.8, f0 + 1.8))))
    # ---- implant
    Mi = implant_matrix(t)
    i0 = B["c3.s2"] + 0.2
    S["implant"] = dict(M=Mi, fill=-0.02 + 1.05 * in_out_sine(lin(t, i0, i0 + 0.75)),
                        hot=1.0 - smooth(lin(t, B["c3.s2"] + 1.2, B["c3.s3"] + 0.1)))
    g0 = B["c3.s3"] + 0.1
    S["osseoC"] = dict(fill=-0.02 + 1.05 * in_out_sine(lin(t, g0, g0 + 1.4)),
                       hot=0.6 * smooth(lin(t, g0, g0 + 0.3)) * (1 - smooth(lin(t, g0 + 1.3, g0 + 2.2))),
                       vis=1.0 if cut_open("cutC", t) > 0 or t < B["c3.cut"] else 0.0)
    th = B["c3.s3"] + 1.05
    S["healing"] = dict(M=healing_matrix(t), vis=1.0 if th - 0.02 <= t <= B["c4.s1"] + 0.62 else 0.0)
    S["abutment"] = dict(M=abutment_matrix(t), vis=1.0 if t >= B["c4.s2"] - 0.02 else 0.0)
    cv = B["c4.click"] + 0.5
    S["crown46"] = dict(M=crown_matrix(t), vis=1.0 if t >= cv else 0.0)
    # ---- upper arch
    S["upper_rig"] = dict(M=upper_matrix(t))
    S["__upper_vis"] = dict(vis=upper_vis(t))
    S["__scene"] = dict(rr=reveal_radius(t))
    return S
