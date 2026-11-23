"""Generate every mesh of the world from the SDFs in om/ (numpy + scikit-image).

    python3 tools/make_geometry.py [--coarse]

Writes build/geo/*.npz (vertices, faces, per-vertex attributes).  Blender only
loads these; nothing is modelled by hand.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from om import teeth as T  # noqa: E402
from om import world as W  # noqa: E402
from om.mesh import mesh_sdf, laplacian  # noqa: E402
from om.sdf import smax, smin, sphere, ellipsoid, round_box, cylinder_z, length, chain  # noqa: E402

COARSE = "--coarse" in sys.argv
OUT = Path(__file__).resolve().parent.parent / "build" / "geo"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def save(name, v, f, **attrs):
    np.savez_compressed(OUT / f"{name}.npz", v=v.astype(np.float32), f=f.astype(np.int32),
                        **{k: np.asarray(a, np.float32) for k, a in attrs.items()})
    log(name, len(v), "verts", len(f), "tris")


def st(x):
    return x * (2.2 if COARSE else 1.0)


# ------------------------------------------------------------------ teeth (local, lower orientation)
def make_teeth():
    for k, d in T.KINDS.items():
        ex = d["ext"]
        v, f = mesh_sdf(d["fn"], (-ex[0], -ex[1], -d["lr"] - 0.1), (ex[0], ex[1], d["hc"] + 0.1), st(0.011))
        v = laplacian(v, f, 2)
        save(f"tooth_{k}", v, f, zl=v[:, 2])
    # endo molar internals
    v, f = mesh_sdf(T.molar_pulp, (-0.55, -0.45, -1.5), (0.55, 0.45, 0.5), st(0.008))
    v = laplacian(v, f, 2)
    save("pulp36", v, f)
    for i in range(3):
        pts = np.array(T.MOLAR_CANALS[i])
        v, f = mesh_sdf(lambda P, i=i: T.molar_canal_tube(P, i, grow=0.004), pts.min(0) - 0.12, pts.max(0) + 0.12, st(0.006))
        v = laplacian(v, f, 1)
        # u = normalised position along the canal (0 = orifice, 1 = apex)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        dense = np.concatenate([np.linspace(pts[j], pts[j + 1], 40) for j in range(len(pts) - 1)])
        dcum = np.concatenate([np.linspace(cum[j], cum[j + 1], 40) for j in range(len(pts) - 1)])
        from scipy.spatial import cKDTree
        idx = cKDTree(dense).query(v)[1]
        save(f"gutta36_{i}", v, f, u=dcum[idx] / cum[-1])
    # caries: irregular blob from the occlusal fissure down to the pulp horn
    def caries(P):
        d = ellipsoid(P, (-0.12, -0.05, 0.62), (0.2, 0.16, 0.16))
        d = smin(d, round_cone_safe(P, (-0.12, -0.05, 0.6), (-0.16, -0.1, 0.32), 0.12, 0.06), 0.06)
        n = np.sin(P[..., 0] * 41) * np.sin(P[..., 1] * 37) * np.sin(P[..., 2] * 43) * 0.018
        d = d + n
        return smax(d, T.molar_outer(P) + 0.006, 0.01)
    v, f = mesh_sdf(caries, (-0.45, -0.4, 0.15), (0.25, 0.25, 0.85), st(0.007))
    save("caries36", laplacian(v, f, 1), f)
    # restoration (access cavity filling), sits in the occlusal surface
    def resto(P):
        d = ellipsoid(P, (-0.06, -0.02, 0.62), (0.3, 0.22, 0.2))
        d = smax(d, T.molar_outer(P) + 0.002, 0.005)
        d = smax(d, -(P[..., 2] - 0.42), 0.04)
        return d
    v, f = mesh_sdf(resto, (-0.5, -0.4, 0.3), (0.4, 0.4, 0.85), st(0.007))
    v = laplacian(v, f, 1)
    save("resto36", v, f, u=1.0 - (v[:, 2] - v[:, 2].min()) / np.ptp(v[:, 2]))
    # ceramic crown for the implant (molar crown, hollow for the abutment)
    def crown(P):
        d = T.molar_outer(P, crown_only=True)
        d = smax(d, -(P[..., 2] + 0.02), 0.03)
        hole = round_box(P, (0, 0, 0.1), (0.2, 0.2, 0.3), 0.08)
        return smax(d, -hole, 0.02)
    v, f = mesh_sdf(crown, (-0.62, -0.56, -0.1), (0.62, 0.56, 0.86), st(0.009))
    v = laplacian(v, f, 2)
    save("crown46", v, f, zl=v[:, 2])


def round_cone_safe(P, a, b, ra, rb):
    from om.sdf import round_cone
    return round_cone(P, a, b, ra, rb)


# ------------------------------------------------------------------ implant parts (local: z up, platform at z=0)
IMPL_LEN = 1.0
IMPL_R = 0.22


def implant_sdf(P):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    pitch = 0.085
    # body tapers to the apex
    zz = np.clip(-z / IMPL_LEN, 0, 1)
    core = IMPL_R * 0.78 * (1 - 0.18 * zz ** 2)
    phase = ((z / pitch) - th / (2 * np.pi)) % 1.0
    tri = np.abs(phase - 0.5) * 2  # 0 at crest, 1 at root
    thread_h = 0.055 * np.clip((IMPL_LEN - (-z)) / 0.08, 0, 1) * np.clip(-z / 0.12, 0, 1)
    outer = core + thread_h * (1 - tri)
    d = (r - outer) * 0.6
    d = smax(d, -z - IMPL_LEN, 0.05)
    d = smax(d, z, 0.01)
    # polished collar
    collar = cylinder_z(P, (0, 0, -0.06), IMPL_R * 0.98, 0.06)
    return smin(d, collar, 0.01)


def make_implant():
    v, f = mesh_sdf(implant_sdf, (-0.35, -0.35, -IMPL_LEN - 0.08), (0.35, 0.35, 0.06), st(0.006))
    v = laplacian(v, f, 1)
    # u: 0 at apex -> 1 at platform (for the light-line "winding" reveal)
    save("implant", v, f, u=(v[:, 2] + IMPL_LEN) / IMPL_LEN)
    # healing screw (tornillo de cicatrización): smooth dome cylinder
    def heal(P):
        d = cylinder_z(P, (0, 0, 0.2), 0.235, 0.2)
        d = smin(d, ellipsoid(P, (0, 0, 0.4), (0.235, 0.235, 0.07)), 0.03)
        slot = round_box(P, (0, 0, 0.47), (0.03, 0.12, 0.05), 0.01)
        return smax(d, -slot, 0.01)
    v, f = mesh_sdf(heal, (-0.3, -0.3, -0.02), (0.3, 0.3, 0.52), st(0.006))
    save("healing", laplacian(v, f, 1), f)
    # abutment (pilar): collar + tapered post
    def abut(P):
        d = cylinder_z(P, (0, 0, 0.06), 0.215, 0.06)
        z = P[..., 2]
        r = np.sqrt(P[..., 0] ** 2 + P[..., 1] ** 2)
        post_r = 0.17 - 0.06 * np.clip((z - 0.12) / 0.4, 0, 1)
        post = smax(r - post_r, smax(z - 0.52, -(z - 0.1), 0.0), 0.03)
        d = smin(d, post, 0.03)
        flat = P[..., 0] - 0.13
        return smax(d, smax(flat, -(z - 0.15), 0.0), 0.01)
    v, f = mesh_sdf(abut, (-0.3, -0.3, -0.02), (0.3, 0.3, 0.6), st(0.006))
    save("abutment", laplacian(v, f, 1), f)


# ------------------------------------------------------------------ jaw
def make_jaw():
    step = st(0.024)
    lo, hi = (-3.35, -1.0, -3.4), (3.35, 7.4, -0.25)
    log("bone (full)")
    v, f = mesh_sdf(W.bone_sdf, lo, hi, step)
    v = laplacian(v, f, 2)
    save("bone_full", v, f, cap=np.zeros(len(v)), depth=np.zeros(len(v)))
    log("gum (full)")
    v, f = mesh_sdf(W.gum_sdf, lo, hi, step)
    v = laplacian(v, f, 2)
    save("gum_full", v, f, cap=np.zeros(len(v)), depth=np.zeros(len(v)))
    log("upper gum")
    v, f = mesh_sdf(W.upper_gum_sdf, (-3.5, -1.2, 0.3), (3.5, 6.8, 2.75), step)
    v = laplacian(v, f, 2)
    save("upper_gum", v, f, cap=np.zeros(len(v)), depth=np.zeros(len(v)))


def site_bounds(name, pad=0.15):
    M = W.cut_matrix(name)
    # corners of the cut box in world space
    hl = 1.0 + 0.2
    pts = []
    for sx in (-hl, hl):
        for sy in (-1.6, 1.6):
            for sz in (-3.0, 0.9):
                pts.append(M[:3, :3] @ np.array([sx, sy, sz]) + M[:3, 3])
    pts = np.array(pts)
    return pts.min(0) - pad, pts.max(0) + pad


def cut_fields(P, name):
    """Local cutter coords and SDF of the removed buccal box."""
    M = W.cut_matrix(name)
    R = M[:3, :3]
    hl = np.linalg.norm(R[:, 0])
    Rn = R / np.linalg.norm(R, axis=0)
    Q = (P - M[:3, 3]) @ Rn
    x, y = Q[..., 0], Q[..., 1]
    removed = np.maximum(np.abs(x) - hl, y)        # <0 inside removed box
    slab = np.abs(x) - (hl + 0.07)                   # keep a sliver beyond the box ends
    return removed, slab


def make_cuts():
    step = st(0.016)
    for name in ("cutA", "cutB", "cutC"):
        lo, hi = site_bounds(name)
        log(name)

        def bone_cut(P, name=name):
            rem, slab = cut_fields(P, name)
            B = W.bone_sdf(P) + 0.004
            return np.maximum(np.maximum(B, slab), -rem)

        def gum_cut(P, name=name):
            rem, slab = cut_fields(P, name)
            G = W.gum_sdf(P) + 0.004
            return np.maximum(np.maximum(G, slab), -rem)

        for kind, fn, ref in (("bone", bone_cut, lambda P: W.bone_sdf(P)), ("gum", gum_cut, None)):
            v, f = mesh_sdf(fn, lo, hi, step)
            v = laplacian(v, f, 1)
            rem, _ = cut_fields(v, name)
            cap = (np.abs(rem) < step * 0.75).astype(np.float32)
            depth = -W.bone_sdf(v, sockets=False) if kind == "bone" else np.zeros(len(v))
            # gum layer depth is not used; bone depth drives the cortical band
            save(f"{kind}_{name}", v, f, cap=cap, depth=depth)


# ------------------------------------------------------------------ surgery (site B) pieces
def make_surgery():
    step = st(0.014)
    wis = W.tooth(W.SITE_WISDOM)
    M = W.cut_matrix("cutB")
    lo, hi = site_bounds("cutB")
    crown_c = W.to_world(np.array([0, 0, 0.45]), wis["M"], wis["scale"])

    def flap_box(P):
        # gum over the wisdom tooth crown, lingual half (visible after the cut)
        Mf = W.site_frame(W.SITE_WISDOM)
        Q = (P - Mf[:3, 3]) @ Mf[:3, :3]
        return round_box(Q, (-0.02, 0.35, 0.25), (0.5, 0.45, 0.42), 0.08)

    def flap(P):
        rem, slab = cut_fields(P, "cutB")
        G = W.gum_sdf(P) + 0.002
        d = np.maximum(np.maximum(G, slab), -rem)
        return smax(d, flap_box(P), 0.0)

    v, f = mesh_sdf(flap, lo, hi, step)
    rem, _ = cut_fields(v, "cutB")
    save("flapB", laplacian(v, f, 1), f, cap=(np.abs(rem) < step * 0.75).astype(np.float32), depth=np.zeros(len(v)))
    # bone covering the crown (osteotomy window): dissolves before the extraction
    def osteo(P):
        rem, slab = cut_fields(P, "cutB")
        B = W.bone_sdf(P) + 0.002
        d = np.maximum(np.maximum(B, slab), -rem)
        win = sphere(P, crown_c, 0.55)
        return smax(d, win, 0.0)
    v, f = mesh_sdf(osteo, lo, hi, step)
    rem, _ = cut_fields(v, "cutB")
    save("osteoB", laplacian(v, f, 1), f, cap=(np.abs(rem) < step * 0.75).astype(np.float32),
         depth=-W.bone_sdf(v, sockets=False))
    # socket fill: volume of the wisdom tooth inside the bone envelope (lingual half)
    def fill(P):
        rem, slab = cut_fields(P, "cutB")
        Tw = W.tooth_sdf_world(P, wis, dilate=0.01)
        Bn = W.bone_sdf(P, sockets=False) + 0.003
        d = smax(Tw, Bn, 0.0)
        return np.maximum(d, -rem)
    v, f = mesh_sdf(fill, lo, hi, step)
    v = laplacian(v, f, 1)
    rem, _ = cut_fields(v, "cutB")
    zz = v[:, 2]
    save("fillB", v, f, cap=(np.abs(rem) < step * 0.75).astype(np.float32),
         depth=-W.bone_sdf(v, sockets=False), u=(zz - zz.min()) / np.ptp(zz))


# ------------------------------------------------------------------ osseointegration sleeve (site C)
def make_osseo():
    step = st(0.008)
    Mc = W.site_frame(W.SITE_IMPLANT)

    def sleeve(P):
        Q = (P - Mc[:3, 3]) @ Mc[:3, :3]
        Q = Q - np.array([0, 0, -0.06])
        d_impl = implant_sdf(Q)
        shell = smax(d_impl - 0.06, -(d_impl + 0.004), 0.0)  # bone hugging the implant threads
        rem, _ = cut_fields(P, "cutC")
        return np.maximum(shell, -rem)

    c = Mc[:3, 3]
    v, f = mesh_sdf(sleeve, c - np.array([0.45, 0.45, 1.2]), c + np.array([0.45, 0.45, 0.1]), step)
    v = laplacian(v, f, 1)
    Q = (v - Mc[:3, 3]) @ Mc[:3, :3]
    rem, _ = cut_fields(v, "cutC")
    save("osseoC", v, f, u=np.clip((-Q[:, 2]) / 1.1, 0, 1), cap=(np.abs(rem) < step * 0.75).astype(np.float32),
         depth=np.full(len(v), 0.5))


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["teeth", "implant", "jaw", "cuts", "surgery", "osseo"]
    for w in which:
        globals()[f"make_{w}"]()
    log("done")
