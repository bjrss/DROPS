"""Bake om.anim (pure functions of time) into per-frame linear keyframes."""
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from om import anim as A  # noqa: E402
from om import config as CFG  # noqa: E402


def _fc(id_data, path, index, frames, values, scalar=False):
    """Create/replace an F-curve with one linear key per frame.
    keyframe_insert() first, so Blender 4.4+ binds the action slot correctly."""
    id_data.keyframe_insert(data_path=path, index=-1 if scalar else index, frame=float(frames[0]))
    ad = id_data.animation_data
    fc = None
    try:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        fc = cb.fcurves.find(path, index=0 if scalar else index)
    except Exception:
        pass
    if fc is None:
        fc = ad.action.fcurves.find(path, index=0 if scalar else index)
    fc.keyframe_points.clear()
    n = len(frames)
    fc.keyframe_points.add(n)
    co = np.empty(2 * n, np.float32)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", np.full(n, 1, np.int32))  # LINEAR
    fc.update()


def bake_all(scn, f0=0, f1=None, step=1):
    f1 = CFG.NFRAMES - 1 if f1 is None else f1
    frames = np.arange(f0, f1 + 1, step)
    rec = {}          # (obj, key) -> list
    mats = {}         # obj -> list of 4x4
    for f in frames:
        S = A.states(f / CFG.FPS)
        for name, st in S.items():
            for k, v in st.items():
                if k == "M":
                    mats.setdefault(name, []).append(np.asarray(v, float))
                else:
                    rec.setdefault((name, k), []).append(float(v))
    # props
    for (name, k), vals in rec.items():
        vals = np.asarray(vals)
        targets = []
        if name == "__upper_vis":
            targets = [bpy.data.objects[n] for n in scn["upper_objects"].split(",")]
        elif name == "__scene":
            targets = [scn]
        elif name in bpy.data.objects:
            targets = [bpy.data.objects[name]]
        for ob in targets:
            ob[k] = float(vals[0])
            if np.ptp(vals) == 0:
                continue
            _fc(ob, f'["{k}"]', 0, frames, vals, scalar=True)
    # hide_render whenever an object cannot be seen (alpha 0 objects still cost shading time)
    hidden = {}
    for (name, k), vals in rec.items():
        vals = np.asarray(vals)
        if name == "__upper_vis":
            names = scn["upper_objects"].split(",")
        elif name in bpy.data.objects:
            names = [name]
        else:
            continue
        if k == "vis":
            h = vals <= 0.0
        elif k == "fill":
            h = vals <= 0.0
        elif k == "xray" and name in ("t36_xray", "flapBx"):
            h = vals <= 0.001
        elif k == "wipe" and name == "t38":
            h = vals >= 2.5
        elif k == "dissolve":
            h = vals >= 0.999
        else:
            continue
        for nm in names:
            hidden[nm] = h if nm not in hidden else (hidden[nm] | h)
    for nm, h in hidden.items():
        ob = bpy.data.objects[nm]
        ob.hide_render = bool(h[0])
        if h.all() or not h.any():
            continue
        ob.keyframe_insert("hide_render", frame=float(frames[0]))
        ad = ob.animation_data
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        fc = cb.fcurves.find("hide_render")
        fc.keyframe_points.clear()
        # only key the changes (constant interpolation)
        idx = [0] + [i for i in range(1, len(h)) if h[i] != h[i - 1]]
        fc.keyframe_points.add(len(idx))
        for j, i in enumerate(idx):
            kp = fc.keyframe_points[j]
            kp.co = (float(frames[i]), float(h[i]))
            kp.interpolation = "CONSTANT"
        fc.update()
    print("hide_render keyed on", len(hidden), "objects", flush=True)

    # transforms
    for name, Ms in mats.items():
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        Ms = np.array(Ms)
        if np.ptp(Ms, axis=0).max() == 0:
            ob.matrix_world = mathutils.Matrix([list(r) for r in Ms[0]])
            continue
        ob.rotation_mode = "QUATERNION"
        loc, rot, sca = [], [], []
        prev = None
        for M in Ms:
            m = mathutils.Matrix([list(r) for r in M])
            l, q, s = m.decompose()
            if prev is not None and prev.dot(q) < 0:
                q = -q
            prev = q
            loc.append(l[:]); rot.append(q[:]); sca.append(s[:])
        loc, rot, sca = np.array(loc), np.array(rot), np.array(sca)
        for i in range(3):
            _fc(ob, "location", i, frames, loc[:, i])
            _fc(ob, "scale", i, frames, sca[:, i])
        for i in range(4):
            _fc(ob, "rotation_quaternion", i, frames, rot[:, i])
    # cameras
    for fmt in CFG.PLATES:
        cam = bpy.data.objects[f"cam_{fmt}"]
        cam.rotation_mode = "QUATERNION"
        L, R, lens, foc, fst, sx, sy = [], [], [], [], [], [], []
        prev = None
        for f in frames:
            c = A.camera(f / CFG.FPS, fmt)
            m = mathutils.Matrix([list(r) for r in A.cam_matrix(c)])
            l, q, _ = m.decompose()
            if prev is not None and prev.dot(q) < 0:
                q = -q
            prev = q
            L.append(l[:]); R.append(q[:])
            lens.append(18.0 / np.tan(np.radians(c["fov"]) / 2))
            foc.append(c["focus"]); fst.append(c["fstop"]); sx.append(c["shift_x"]); sy.append(c["shift_y"])
        L, R = np.array(L), np.array(R)
        for i in range(3):
            _fc(cam, "location", i, frames, L[:, i])
        for i in range(4):
            _fc(cam, "rotation_quaternion", i, frames, R[:, i])
        _fc(cam.data, "lens", 0, frames, scalar=True, values= np.array(lens))
        _fc(cam.data, "dof.focus_distance", 0, frames, scalar=True, values= np.array(foc))
        _fc(cam.data, "dof.aperture_fstop", 0, frames, scalar=True, values= np.array(fst))
        _fc(cam.data, "shift_x", 0, frames, scalar=True, values= np.array(sx))
        _fc(cam.data, "shift_y", 0, frames, scalar=True, values= np.array(sy))
    print("baked", len(frames), "frames", flush=True)
