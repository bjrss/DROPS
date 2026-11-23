"""The light line's journey through the film.

`line_items(t, fmt, out_map)` returns what to draw at time t: a list of dicts
consumed by compose.py, which passes every one of them to the single
lightline.draw_lightline() (or draw_dots for point clouds in the same look).
3D paths are projected with the same camera functions Blender is keyed with.
"""
import math
from functools import lru_cache

import numpy as np

from . import anim as A
from . import teeth as TH
from . import world as W
from .config import C, ROOT
from .easing import lin, smooth, in_out_sine, in_out_cubic, out_cubic, out_quint, clamp, in_cubic
from .sdf import catmull
from .timeline import B

GEO = ROOT / "build" / "geo"


# ------------------------------------------------------------------ static 3D paths (world)
@lru_cache(None)
def canals36():
    d = W.tooth(W.SITE_ENDO)
    out = []
    for pts in TH.MOLAR_CANALS:
        P = catmull(pts, n_per=14)
        out.append(W.to_world(P, d["M"], d["scale"]))
    return out


@lru_cache(None)
def endo_path():
    """Enter through the caries, down and up each canal (MB, ML, D)."""
    d = W.tooth(W.SITE_ENDO)
    entry = W.to_world(np.array([[-0.12, -0.05, 0.85], [-0.12, -0.05, 0.62], [-0.05, -0.02, 0.25]]), d["M"], d["scale"])
    segs = [entry]
    for c in canals36():
        segs.append(c)
        segs.append(c[::-1])
    P = np.vstack(segs)
    return P


def tooth_local(fdi, P):
    d = W.tooth(fdi)
    return W.to_world(np.asarray(P, float), d["M"], d["scale"])


def ring_world(center, axis, r, n=72):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    a = np.cross(axis, [1.0, 0, 0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(axis, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    th = np.linspace(0, 2 * np.pi, n)
    return center + r * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)


@lru_cache(None)
def silhouette(kind="molar", crown_only=False):
    """Outline of a tooth in its local x-z plane (buccal view), for ghost outlines."""
    from skimage.measure import find_contours
    xs = np.linspace(-0.75, 0.75, 151)
    zs = np.linspace(-1.6, 0.95, 256)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    best = None
    for y in np.linspace(-0.4, 0.4, 9):
        P = np.stack([X, np.full_like(X, y), Z], -1)
        D = TH.molar_outer(P, crown_only=crown_only)
        best = D if best is None else np.minimum(best, D)
    cs = find_contours(best, 0.0)
    c = max(cs, key=len)
    pts = np.stack([np.interp(c[:, 1], np.arange(len(xs)), xs), np.zeros(len(c)), np.interp(c[:, 0], np.arange(len(zs)), zs)], 1)
    return pts[::2]


@lru_cache(None)
def gum_crest_line(fdi, x0=-0.5, x1=0.5, y=0.08, n=40):
    """Points on the gum surface along the site's mesio-distal axis (for the incision)."""
    Mf = W.site_frame(fdi)
    xs = np.linspace(x0, x1, n)
    out = []
    for x in xs:
        base = Mf[:3, 3] + Mf[:3, 0] * x + Mf[:3, 1] * y
        zz = np.linspace(0.6, -0.6, 241)
        P = np.array([base + np.array([0, 0, z]) for z in zz])
        D = W.gum_outer_sdf(P)
        k = np.argmax(D < 0)
        out.append(P[max(k - 1, 0)] + np.array([0, 0, 0.012]))
    return np.array(out)


@lru_cache(None)
def mesh_points(name, n=900, seed=1):
    d = np.load(GEO / f"{name}.npz")
    v = d["v"].astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), size=min(n, len(v)), replace=False)
    return v[idx]


@lru_cache(None)
def scan_cloud():
    """Point cloud of the implant site (gum + neighbouring teeth), as an intra-oral scan would see it."""
    Mc = W.site_frame(W.SITE_IMPLANT)
    pts = []
    g = np.load(GEO / "gum_full.npz")["v"].astype(float)
    Q = (g - Mc[:3, 3]) @ Mc[:3, :3]
    sel = (np.abs(Q[:, 0]) < 1.25) & (Q[:, 2] > -0.25) & (Q[:, 1] < 0.55)
    pts.append(g[sel])
    rng = np.random.default_rng(4)
    pts[0] = pts[0][rng.choice(len(pts[0]), size=min(500, len(pts[0])), replace=False)]
    for fdi in (45, 47):
        d = W.tooth(fdi)
        v = np.load(GEO / f"tooth_{d['kind']}.npz")["v"].astype(float)
        v = v[(v[:, 2] > -0.05) & (v[:, 1] < 0.12)]       # crown, camera-facing half
        v = v[rng.choice(len(v), size=min(160, len(v)), replace=False)]
        pts.append(W.to_world(v, d["M"], d["scale"]))
    P = np.vstack(pts)
    return P


@lru_cache(None)
def smile_curve():
    pts = []
    for fdi in (13, 12, 11, 21, 22, 23):
        d = W.tooth(fdi)
        k = TH.KINDS[d["kind"]]
        pts.append(W.to_world(np.array([0.0, -0.12, k["hc"] - 0.04]), d["M"], d["scale"]))
    pts = np.array(pts)
    return catmull(pts, n_per=16)


@lru_cache(None)
def contact_points():
    out = []
    for fdi in (45, 46, 47):
        d = W.tooth(fdi)
        if d["kind"] == "molar":
            loc = [(-0.31, -0.23), (0.04, -0.26), (0.37, -0.12), (-0.28, 0.22), (0.16, 0.23)]
        else:
            loc = [(0.0, -0.13), (0.0, 0.18)]
        hc = TH.KINDS[d["kind"]]["hc"]
        for x, y in loc:
            out.append(W.to_world(np.array([x, y, hc - 0.03]), d["M"], d["scale"]))
    return np.array(out)


# ------------------------------------------------------------------ helpers
class Proj:
    """Projects world points into OUTPUT pixels for a format at time t."""

    def __init__(self, t, fmt):
        from .config import FORMATS, PLATES
        self.fmt = fmt
        F = FORMATS[fmt]
        self.plate = F["plate"]
        self.PW, self.PH = PLATES[self.plate]["w"], PLATES[self.plate]["h"]
        self.cam = A.camera(t, self.plate)
        self.oy = (self.PH - F["h"]) / 2 if fmt == "4x5" else 0.0
        self.scale = min(F["w"], F["h"]) / 1080.0

    def __call__(self, P):
        xy, z = A.project(np.asarray(P, float), self.cam, self.PW, self.PH, self.plate)
        xy[:, 1] -= self.oy
        return xy, z


def item(pts, head=1.0, tail=0.0, width=5.0, intensity=1.0, **kw):
    return dict(kind="line", pts=pts, head=head, tail=tail, width=width, intensity=intensity, **kw)


def comet(P, prog, length=0.22, **kw):
    """A comet travelling along P: head at prog, tail behind."""
    return item(P, head=prog, tail=max(0.0, prog - length), **kw)


def mix_color(a, b, x):
    def h2(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)
    c = h2(a) * (1 - x) + h2(b) * x
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in c)


# ------------------------------------------------------------------ choreography
def line_items(t, fmt, layout):
    pr = Proj(t, fmt)
    S = pr.scale
    out = []
    LW = 5.0

    # ======================= S1: logo trace (2D) =======================
    L = layout["logo_open"]
    from . import logo as LG
    tr_pts, tr_u, tr_on, _ = LG.trace(L["h"])
    lp = tr_pts + np.array([L["x"], L["y"]])
    if t < B["logo.trace1"] + 0.2:
        start = lp[0]
        if t < B["logo.trace0"]:
            # frame 0 already moves: the light glides in towards the first stroke
            k = in_out_sine(lin(t, -0.45, B["logo.trace0"]))
            entry = np.array([[layout["W"] * 1.05, L["y"] + L["h"] * 1.15], [L["x"] + L["w"] * 1.0, L["y"] - L["h"] * 0.05],
                              start + [6, -18], start])
            E = catmull(entry, n_per=12)
            out.append(comet(E, k, 0.5, width=LW, intensity=1.0))
        else:
            k = in_out_sine(lin(t, B["logo.trace0"], B["logo.trace1"]))
            fade = 1 - smooth(lin(t, B["logo.trace1"], B["logo.trace1"] + 0.18))
            vis = 0.35 + 0.65 * tr_on
            out.append(item(lp, head=k, tail=max(0, k - 0.16), width=LW, intensity=fade, vis=vis, fade_tail=0.6))
    # detach (2.40): the light leaves from the tip of the natural root and scans the 3D tooth
    if B["logo.detach"] - 0.05 < t < B["tooth.in1"] + 0.1:
        k = in_out_cubic(lin(t, B["logo.detach"], B["tooth.in0"] + 0.15))
        root_tip = lp[np.argmax(np.where(tr_on > 0, lp[:, 1] * (lp[:, 0] > L["x"] + L["w"] * 0.5), -1))]
        st = A.states(t)["t36"]
        z = st["wipe"]
        if t >= B["tooth.in0"] + 0.15:
            P = tooth_local(36, np.stack([np.linspace(-0.7, 0.7, 30), np.full(30, -0.62), np.full(30, np.clip(z, -1.45, 0.8))], 1))
            xy, _ = pr(P)
            inten = 1 - smooth(lin(t, B["tooth.in1"] - 0.15, B["tooth.in1"] + 0.1))
            out.append(item(xy, head=1, tail=0, width=LW * 0.9, intensity=inten, spark=False))
        else:
            P0 = tooth_local(36, np.array([[0.0, -0.62, 0.8]]))
            tgt, _ = pr(P0)
            path = catmull(np.array([root_tip, root_tip + [40, 60], tgt[0] + [60, -40], tgt[0]]), n_per=10)
            out.append(comet(path, k, 0.35, width=LW))
    # x-ray scan (bottom -> top)
    if B["xray0"] - 0.05 < t < B["xray1"] + 0.15:
        st = A.states(t)["t36"]
        z = float(np.clip(st["wipe"], -1.45, 0.85))
        inten = smooth(lin(t, B["xray0"], B["xray0"] + 0.1)) * (1 - smooth(lin(t, B["xray1"] - 0.1, B["xray1"] + 0.15)))
        P = tooth_local(36, np.stack([np.linspace(-0.75, 0.75, 30), np.full(30, -0.62), np.full(30, z)], 1))
        out.append(item(pr(P)[0], width=LW * 0.9, intensity=inten, spark=False))

    # ======================= C1: endodoncia =======================
    if B["c1.problem"] < t < B["c1.s1"] + 0.1:
        # light circles the caries, then dives to the inflamed pulp ("llega al nervio")
        cc = tooth_local(36, [[-0.12, -0.05, 0.6]])[0]
        d = W.tooth(36)
        ring = ring_world(cc, d["M"][:3, 2], 0.28, 60)
        k = in_out_sine(lin(t, B["c1.problem"] + 0.3, B["c1.caries"] + 0.5))
        inten = smooth(lin(t, B["c1.problem"] + 0.2, B["c1.problem"] + 0.5)) * (1 - smooth(lin(t, B["c1.s1"] - 0.3, B["c1.s1"])))
        out.append(item(pr(ring)[0], head=k, tail=max(0, k - 0.6), width=LW * 0.8, intensity=inten * 0.9))
        dive = tooth_local(36, [[-0.12, -0.05, 0.6], [-0.1, -0.05, 0.42], [0.0, 0.0, 0.2]])
        kd = in_out_sine(lin(t, B["c1.nervio"] - 0.35, B["c1.nervio"] + 0.2))
        if kd > 0:
            out.append(comet(pr(catmull(dive, 10))[0], kd, 0.6, width=LW * 0.8, intensity=inten))
    # step 1: anaesthesia — soft blue waves around the tooth (no needle)
    if B["c1.s1"] < t < B["c1.s2"] + 0.4:
        out += waves(pr, 36, t, B["c1.s1"], S)
    # step 2: the line files down each canal
    if B["c1.s2"] < t < B["c1.s3"]:
        P = endo_path()
        k = in_out_sine(lin(t, B["c1.s2"] + 0.05, B["c1.irrig"] - 0.15))
        inten = 1 - smooth(lin(t, B["c1.irrig"] - 0.25, B["c1.irrig"]))
        out.append(comet(pr(P)[0], k, 0.12, width=LW * 0.85, intensity=inten))
        # inflamed pulp leaves as particles
        out += pulp_particles(pr, t, S)
        # irrigation glint along every canal
        g = lin(t, B["c1.irrig"], B["c1.irrig"] + 0.8)
        if 0 < g < 1:
            for c in canals36():
                out.append(comet(pr(c)[0], in_out_sine(g), 0.5, width=LW * 0.55, intensity=0.85 * math.sin(math.pi * g),
                                 core="#FFFFFF", halo="#9CC7E0"))
    # step 3: the line cools into gutta-percha, then closes the crown
    if B["c1.s3"] < t < B["c1.s3"] + 1.6:
        for i, c in enumerate(canals36()):
            a = B["c1.s3"] + 0.12 + 0.28 * i
            k = in_out_sine(lin(t, a, a + 0.55))
            if 0 < k:
                cool = smooth(lin(t, a + 0.1, a + 0.7))
                col = mix_color(C["line_core"], C["gutta"], cool)
                inten = 1 - smooth(lin(t, a + 0.55, a + 0.95))
                out.append(comet(pr(c)[0], k, 0.4, width=LW * 0.8, intensity=inten, core=col,
                                 halo=mix_color(C["line_halo"], "#E8A07B", cool)))
        r0 = B["c1.s3"] + 1.05
        lid = tooth_local(36, [[-0.38, -0.3, 0.68], [-0.15, -0.36, 0.74], [0.12, -0.33, 0.72], [0.3, -0.25, 0.66]])
        k = in_out_sine(lin(t, r0, r0 + 0.45))
        if k > 0:
            out.append(comet(pr(catmull(lid, 10))[0], k, 0.8, width=LW * 0.8,
                             intensity=1 - smooth(lin(t, r0 + 0.45, r0 + 0.6))))
    # result + travel to the wisdom tooth
    if B["c1.result"] - 0.1 < t < B["c2"] + 0.8:
        out += result_halo(pr, tooth_local(36, [[0, 0, -0.2]])[0], W.tooth(36)["M"][:3, 2], t, B["c1.result"], 0.85)
        k = in_out_sine(lin(t, B["c1.out"] - 0.1, B["c2"] + 0.6))
        if 0 < k < 1:
            path = np.vstack([tooth_local(36, [[0.0, -0.1, 0.85]]), tooth_local(37, [[0.0, -0.1, 0.8]]),
                              tooth_local(38, [[0.0, -0.1, 1.05]]), tooth_local(38, [[0.0, -0.1, 0.65]])])
            out.append(comet(pr(catmull(path, 16))[0], k, 0.3, width=LW))

    # ======================= C2: cirugía oral =======================
    w38 = W.tooth(38)
    if B["c2.problem"] < t < B["c2.s1"] + 0.2:
        # outline of the retained wisdom tooth + soft pulse where it presses on the neighbour
        sil = silhouette()
        P = W.to_world(sil * [1.04, 1, 1.03] + [0, -0.18, 0], w38["M"], w38["scale"])
        k = in_out_sine(lin(t, B["c2.problem"] + 0.2, B["c2.problem"] + 1.4))
        inten = 1 - smooth(lin(t, B["c2.s1"] - 0.3, B["c2.s1"]))
        out.append(item(pr(P)[0], head=k, tail=0, width=LW * 0.7, intensity=0.85 * inten, dash=(16, 12), spark=k < 1, closed=False))
        cp = W.to_world(np.array([-0.52, -0.4, 0.55]), w38["M"], w38["scale"])
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.1 * (t - B["c2.juicio"]))
        if t > B["c2.juicio"] - 0.4:
            out.append(dict(kind="dots", pts=pr(cp[None])[0], size=9 + 5 * pulse, intensity=inten * (0.5 + 0.5 * pulse)))
    if B["c2.s1"] < t < B["c2.s2"] + 0.4:
        out += waves(pr, 38, t, B["c2.s1"], S)
    if B["c2.s2"] < t < B["c2.result"] + 0.6:
        inc = gum_crest_line(38)
        k = in_out_sine(lin(t, B["c2.s2"], B["c2.s2"] + 0.45))
        heal = smooth(lin(t, B["c2.s3"] + 0.9, B["c2.s3"] + 1.8))
        inten = 1 - heal * 0.95
        if t < B["c2.s2"] + 0.5:
            out.append(comet(pr(inc)[0], k, 1.0, width=LW * 0.75, intensity=1.0))
        elif t < B["c2.s3"] + 0.4:
            out.append(item(pr(inc)[0], width=LW * 0.55, intensity=0.55, spark=False))
        else:
            out.append(item(pr(inc)[0], width=LW * 0.55, intensity=0.55 * inten, spark=False))
        # stitches of light, drawn one by one, then they fade as the tissue heals
        for j, xs in enumerate((-0.3, 0.0, 0.3)):
            a = B["c2.s3"] + 0.45 + 0.22 * j
            ks = in_out_sine(lin(t, a, a + 0.3))
            if ks <= 0:
                continue
            Mf = W.site_frame(38)
            c = inc[np.argmin(np.abs(np.linspace(-0.5, 0.5, len(inc)) - xs))]
            st = np.array([c - Mf[:3, 1] * 0.0 - Mf[:3, 0] * 0.06 + [0, 0, 0.01],
                           c + Mf[:3, 1] * 0.16 + [0, 0, 0.05],
                           c + Mf[:3, 1] * 0.28 + Mf[:3, 0] * 0.06 + [0, 0, -0.005]])
            fadeS = 1 - smooth(lin(t, B["c2.s3"] + 1.3 + 0.15 * j, B["c2.s3"] + 2.0 + 0.15 * j))
            out.append(comet(pr(catmull(st, 8))[0], ks, 1.0, width=LW * 0.7, intensity=fadeS))
    if B["c2.result"] - 0.1 < t < B["c3"] + 1.0:
        Mf = W.site_frame(38)
        out += result_halo(pr, Mf[:3, 3] + [0, 0, 0.35], Mf[:3, 2], t, B["c2.result"], 0.75)
        k = in_out_sine(lin(t, B["c2.out"], B["c3"] + 0.85))
        if 0 < k < 1:
            ss = np.linspace(W.tooth(37)["s"], W.tooth(46)["s"], 40)
            path = []
            for s_ in ss:
                p, _ = W.arch_point(s_)
                path.append([p[0], p[1], 0.12 + 0.25 * math.sin(math.pi * (s_ - ss[0]) / (ss[-1] - ss[0]))])
            path.append(list(W.site_frame(46)[:3, 3] + [0, 0, 0.5]))
            out.append(comet(pr(np.array(path))[0], k, 0.22, width=LW))

    # ======================= C3: implantología =======================
    t46 = W.tooth(46)
    Mc = W.site_frame(46)
    if B["c3.problem"] < t < B["c3.s2"]:
        sil = silhouette()
        P = W.to_world(sil + [0, -0.18, 0], t46["M"], t46["scale"])
        k = in_out_sine(lin(t, B["c3.problem"] + 0.3, B["c3.problem"] + 1.5))
        inten = 0.75 * (1 - smooth(lin(t, B["c3.s2"] - 0.5, B["c3.s2"] - 0.1)))
        root_glow = 1 + 0.6 * smooth(lin(t, B["c3.raiz"] - 0.2, B["c3.raiz"] + 0.3)) * (1 - smooth(lin(t, B["c3.raiz"] + 0.6, B["c3.raiz"] + 1.1)))
        out.append(item(pr(P)[0], head=k, tail=0, width=LW * 0.65, intensity=inten * root_glow, dash=(14, 12), spark=k < 1))
    if B["c3.s1"] < t < B["c3.s2"] + 0.2:
        # guide from the planning marker on the glass radiograph to the bone
        mk = layout.get("_xray_marker")
        if mk is not None:
            tgt = pr((Mc[:3, 3] + [0, 0, 0.05])[None])[0][0]
            k = in_out_sine(lin(t, B["c3.s1"] + 0.6, B["c3.s1"] + 1.1))
            path = np.array([mk, (np.array(mk) + tgt) / 2 + [0, -40 * S], tgt])
            inten = 1 - smooth(lin(t, B["c3.s2"] - 0.2, B["c3.s2"] + 0.2))
            out.append(item(pr_identity(catmull(path, 12)), head=k, tail=0, width=LW * 0.6, intensity=inten, dash=(12, 10)))
    if B["c3.s2"] - 0.05 < t < B["c3.s3"]:
        # the line winds into a thread and becomes the implant
        Mi = A.implant_matrix(B["c3.s2"] + 1.75)
        n = 420
        th = np.linspace(0, 2 * np.pi * 9.5, n)
        z = np.linspace(0.35, -1.0, n)
        r = 0.27 * (1 - 0.12 * np.clip(-z, 0, 1) ** 2)
        Pl = np.stack([r * np.cos(th), r * np.sin(th), z], 1)
        P = Pl @ Mi[:3, :3].T + Mi[:3, 3]
        xy, zf = pr(P)
        # dim the half of the helix behind the axis
        cam = pr.cam["pos"]
        axis_pt = Mi[:3, 3]
        front = np.sum((P - axis_pt) * (cam - axis_pt), axis=1)
        vis = np.where(front > 0, 1.0, 0.32)
        k = in_out_sine(lin(t, B["c3.s2"], B["c3.s2"] + 1.0))
        inten = 1 - smooth(lin(t, B["c3.s2"] + 1.0, B["c3.s2"] + 1.6))
        out.append(item(xy, head=k, tail=max(0, k - 0.55), width=LW * 0.75, intensity=inten, vis=vis))
    if B["c3.s3"] < t < B["c3.result"] + 0.3:
        # osseointegration: brand-blue sparks where new bone meets the threads
        P = osseo_points()
        g0 = B["c3.s3"] + 0.1
        prog = in_out_sine(lin(t, g0, g0 + 1.4))
        Q = (P - Mc[:3, 3]) @ Mc[:3, :3]
        u = np.clip(-Q[:, 2] / 1.1, 0, 1)
        on = np.clip(1 - np.abs(u - prog) / 0.12, 0, 1)
        twinkle = 0.6 + 0.4 * np.sin(t * 9 + np.arange(len(P)) * 1.7)
        fade = 1 - smooth(lin(t, g0 + 1.5, g0 + 2.1))
        out.append(dict(kind="dots", pts=pr(P)[0], size=2.6, intensity=fade, vis=on * twinkle,
                        core="#E6F3FF", halo=C["brand_light"]))
    if B["c3.result"] - 0.1 < t < B["c4"] + 0.3:
        out += result_halo(pr, Mc[:3, 3] + [0, 0, 0.12], Mc[:3, 2], t, B["c3.result"], 0.55)

    # ======================= C4: rehabilitación =======================
    if B["c4.problem"] < t < B["c4.s1"] + 0.3:
        sil = silhouette(crown_only=True)
        P = W.to_world(sil + [0, -0.18, 0], t46["M"], t46["scale"])
        k = in_out_sine(lin(t, B["c4.problem"] + 0.2, B["c4.problem"] + 1.2))
        pulse = 0.75 + 0.25 * math.sin(2 * math.pi * 0.9 * t)
        inten = 0.8 * pulse * (1 - smooth(lin(t, B["c4.s1"] - 0.2, B["c4.s1"] + 0.2)))
        out.append(item(pr(P)[0], head=k, tail=0, width=LW * 0.65, intensity=inten, dash=(14, 12), spark=k < 1))
    if B["c4.s1"] + 0.55 < t < B["c4.s2"] + 1.4:
        # scan: a light bar sweeps the site, leaving a blue point mesh
        a0 = B["c4.s1"] + 0.6
        k = in_out_sine(lin(t, a0, a0 + 0.9))
        Pc = scan_cloud()
        Q = (Pc - Mc[:3, 3]) @ Mc[:3, :3]
        xq = Q[:, 0]
        sweep_x = -1.3 + 2.6 * k
        got = np.clip((sweep_x - xq) / 0.15 + 1, 0, 1)
        fade = 1 - smooth(lin(t, B["c4.s2"] + 0.6, B["c4.s2"] + 1.3))
        out.append(dict(kind="dots", pts=pr(Pc)[0], size=1.8, intensity=0.8 * fade, vis=got * 0.85,
                        core="#CFEAFF", halo=C["brand_deep"]))
        if 0 < k < 1:
            bar = np.array([Mc[:3, 3] + Mc[:3, 0] * sweep_x + Mc[:3, 1] * -0.5 + [0, 0, z] for z in np.linspace(-0.6, 0.55, 20)])
            out.append(item(pr(bar)[0], width=LW * 0.9, intensity=math.sin(math.pi * k), spark=False))
    if B["c4.s2"] + 0.2 < t < B["c4.seat"] + 0.6:
        # the crown is first a mesh of points made to measure, then solid ceramic
        Mcr = A.crown_matrix(t)
        v = mesh_points("crown46", 320, 3)
        v = v[v[:, 1] < 0.15]
        P = v @ Mcr[:3, :3].T + Mcr[:3, 3]
        a = smooth(lin(t, B["c4.s2"] + 0.2, B["c4.s2"] + 0.5)) * (1 - smooth(lin(t, B["c4.click"] + 0.5, B["c4.click"] + 0.75)))
        if a > 0:
            out.append(dict(kind="dots", pts=pr(P)[0], size=1.9, intensity=0.85 * a, core="#CFEAFF", halo=C["brand_deep"]))
    for tc, rad in ((B["c4.click"], 0.32), (B["c4.seat"], 0.62)):
        if tc - 0.02 < t < tc + 0.6:
            k = out_cubic(lin(t, tc, tc + 0.55))
            cen = Mc[:3, 3] + Mc[:3, 2] * (0.15 if rad < 0.5 else 0.6)
            ring = ring_world(cen, Mc[:3, 2], rad * (0.7 + 0.6 * k))
            out.append(item(pr(ring)[0], width=LW * 0.7, intensity=1 - k, spark=False))
    if B["c4.s3"] + 0.95 < t < B["c4.result"] + 0.6:
        P = contact_points()
        a0 = B["c4.s3"] + 1.0
        k = lin(t, a0, a0 + 0.6)
        order = np.linspace(0, 1, len(P))
        vis = np.clip((k - order * 0.7) / 0.3, 0, 1)
        fade = 1 - smooth(lin(t, B["c4.result"], B["c4.result"] + 0.5))
        out.append(dict(kind="dots", pts=pr(P)[0], size=7.5, intensity=fade, vis=vis))
    if B["c4.smile0"] < t < B["s6.snap"] + 0.4:
        P = smile_curve()
        xy = pr(P)[0]
        k = in_out_sine(lin(t, B["c4.smile0"], B["c4.smile1"]))
        if t < B["s6"] - 0.2:
            out.append(item(xy, head=k, tail=0, width=LW, intensity=1.0, fade_tail=0.0))
        else:
            # the smile lifts and flies back into the logo contour
            out += return_to_logo(xy, t, layout, S)
    return out


def pr_identity(xy):
    return np.asarray(xy, float)


@lru_cache(None)
def osseo_points():
    return mesh_points("osseoC", 600, 7)


def waves(pr, fdi, t, t0, S):
    """Three soft blue waves expanding around the tooth's neck (local anaesthesia, no needle)."""
    d = W.tooth(fdi)
    M = d["M"] if fdi != 38 else W.site_frame(38)
    axis = np.array([0, 0, 1.0])
    c = (M[:3, 3] + np.array([0, 0, -0.05])) if fdi != 38 else M[:3, 3] + [0, 0, 0.25]
    out = []
    for j in range(3):
        a = t0 + 0.12 + 0.28 * j
        k = lin(t, a, a + 1.0)
        if 0 < k < 1:
            ring = ring_world(c, axis, 0.35 + 0.65 * out_cubic(k))
            out.append(item(pr(ring)[0], width=4.0, intensity=0.8 * math.sin(math.pi * k) ** 1.2, spark=False,
                            core="#BFE0F5", halo=C["brand_light"]))
    return out


def pulp_particles(pr, t, S):
    d = W.tooth(36)
    t0 = B["c1.s2"] + 0.25
    t1 = B["c1.s2"] + 2.3
    if not (t0 < t < t1 + 1.0):
        return []
    rng = np.random.default_rng(11)
    n = 140
    src_local = np.vstack([np.array(c)[rng.integers(0, len(c), n // 4)] for c in TH.MOLAR_CANALS] +
                          [rng.uniform([-0.25, -0.2, 0.08], [0.25, 0.2, 0.26], (n - 3 * (n // 4), 3))])
    src_local = src_local[:n]
    birth = t0 + rng.uniform(0, t1 - t0, len(src_local))
    life = rng.uniform(0.7, 1.2, len(src_local))
    age = (t - birth) / life
    alive = (age > 0) & (age < 1)
    if not alive.any():
        return []
    a = age[alive]
    P = src_local[alive].copy()
    drift = rng.normal(0, 0.12, (len(src_local), 3))[alive]
    P = P + drift * a[:, None] + np.stack([np.zeros_like(a), -0.15 * a, 1.1 * a ** 1.3], 1)
    Pw = W.to_world(P, d["M"], d["scale"])
    vis = np.sin(np.pi * a) ** 0.8
    return [dict(kind="dots", pts=pr(Pw)[0], size=3.4, intensity=0.95, vis=vis, core="#FFE3DE", halo=C["pulp_inflamed"])]


def result_halo(pr, center, axis, t, t0, r):
    k = lin(t, t0, t0 + 1.1)
    if not (0 < k < 1):
        return []
    ring = ring_world(center, axis, r)
    head = in_out_sine(k)
    return [item(pr(ring)[0], head=head, tail=max(0, head - 0.45), width=4.0, intensity=0.9 * math.sin(math.pi * k))]


def return_to_logo(smile_xy, t, layout, S):
    L = layout["logo_close"]
    from . import logo as LG
    tr_pts, tr_u, tr_on, _ = LG.trace(L["h"])
    lp = tr_pts + np.array([L["x"], L["y"]])
    k = in_out_cubic(lin(t, B["s6"] - 0.2, B["s6.snap"]))
    # morph the smile polyline into the logo trace
    from .lightline import resample
    a, _ = resample(smile_xy, n=400)
    b, _ = resample(lp, n=400)
    lift = np.array([0, -120 * S]) * math.sin(math.pi * k)
    P = a * (1 - k) + b * k + lift
    inten = 1 - smooth(lin(t, B["s6.snap"], B["s6.snap"] + 0.35))
    return [item(P, head=1.0, tail=0.0, width=5.0 * (1 + 0.3 * math.sin(math.pi * k)), intensity=inten,
                 fade_tail=0.0, spark=k < 0.98)]
