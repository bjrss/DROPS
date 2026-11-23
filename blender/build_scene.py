"""Build the whole world in Blender from build/geo/*.npz and bake the animation
from om/anim.py (one keyframe per frame for every animated channel).

    python blender/build_scene.py [--no-anim] [--out build/scene.blend]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "blender"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from om import world as W  # noqa: E402
from om import config as CFG  # noqa: E402
from bl_util import reset, make_mesh, link_dup, set_matrix, look_at_matrix, eevee_settings  # noqa: E402

GEO = ROOT / "build" / "geo"
DEFAULT_PROPS = dict(vis=1.0, wipe=-10.0, xray=0.0, inflame=0.0, pulse=0.0, dissolve=0.0,
                     fill=2.0, hot=0.0, heal=0.0, caries_amt=0.0)


def load(name):
    d = np.load(GEO / f"{name}.npz")
    attrs = {k: d[k] for k in d.files if k not in ("v", "f")}
    return d["v"], d["f"], attrs


def obj(name, geo, mat, coll, M=None, scale=None, **props):
    v, f, attrs = load(geo)
    ob = make_mesh(name, v, f, attrs=attrs, collection=coll)
    ob.data.materials.append(mat)
    for k, val in {**DEFAULT_PROPS, **props}.items():
        ob[k] = float(val)
    if M is not None:
        set_matrix(ob, M if scale is None else M @ np.diag([*scale, 1.0]))
    return ob


def build():
    reset()
    import materials as MT

    scn = bpy.context.scene
    scn.frame_start, scn.frame_end = 0, CFG.NFRAMES - 1
    coll = {}
    for c in ("teeth", "jaw", "cuts", "endo", "surgery", "implant", "upper", "fx"):
        coll[c] = bpy.data.collections.new(c)
        scn.collection.children.link(coll[c])

    # cutters first (materials reference them)
    for name in W.CUTS:
        e = bpy.data.objects.new(name, None)
        e.empty_display_type = "CUBE"
        scn.collection.objects.link(e)
        M = W.cut_matrix(name)
        # start "closed": pushed far out along -y (buccal) so nothing is hidden
        M2 = M.copy()
        M2[:3, 3] = M2[:3, 3] - M2[:3, 1] * 20.0
        set_matrix(e, M2)

    mats = dict(
        tooth=MT.tooth_material("tooth"),
        tooth36=MT.tooth_material("tooth36", stain=True),
        xray=MT.xray_material("xray"),
        pulp=MT.pulp_material("pulp"),
        gutta=MT.fill_material("gutta", "gutta", rough=0.35, field="u", emission_hot="line_core"),
        caries=MT.simple_material("caries", "caries", rough=0.8),
        resto=MT.fill_material("resto", (0.93, 0.90, 0.83), rough=0.3, field="u", coat=0.4, sss=0.15),
        gum=MT.tissue_material("gum", "gum"),
        bone=MT.tissue_material("bone", "bone"),
        titanium=MT.fill_material("titanium", "titanium", rough=0.28, field="u", metal=1.0, aniso=0.6,
                                  emission_hot="line_core"),
        ti_simple=MT.simple_material("ti_simple", "titanium", rough=0.3, metal=1.0, aniso=0.5),
        ceramic=MT.simple_material("ceramic", "ceramic", rough=0.22, coat=0.5, sss=0.25),
        osseo=MT.fill_material("osseo", "bone", rough=0.7, field="u", emission_hot="brand_light"),
    )
    # glow of the osseo sleeve uses brand blue; bone fill uses bone colours
    mats["fill_bone"] = MT.fill_material("fill_bone", "bone", rough=0.7, field="u", emission_hot="brand_light")

    # ---------------- teeth
    kinds = {}
    for k in ("incisor", "canine", "premolar", "molar"):
        v, f, attrs = load(f"tooth_{k}")
        kinds[k] = (v, f, attrs)
    meshes = {}
    for d in W.placements():
        if not d["present"]:
            continue
        k = d["kind"]
        name = f"t{d['fdi']}"
        target = coll["upper"] if d["upper"] else coll["teeth"]
        if d["fdi"] == W.SITE_ENDO:
            ob = make_mesh(name, kinds[k][0], kinds[k][1], attrs=kinds[k][2], collection=target)
            v = kinds[k][0]
            # caries stain mask near the occlusal fissure (mesial-central)
            cm = np.exp(-(((v[:, 0] + 0.12) / 0.2) ** 2 + ((v[:, 1] + 0.05) / 0.16) ** 2)) * (v[:, 2] > 0.45)
            a = ob.data.attributes.new("caries_mask", "FLOAT", "POINT")
            a.data.foreach_set("value", cm.astype(np.float32))
            ob.data.materials.append(mats["tooth36"])
        else:
            if k not in meshes:
                meshes[k] = make_mesh(f"_{k}", kinds[k][0], kinds[k][1], attrs=kinds[k][2], collection=target)
                meshes[k].data.materials.append(mats["tooth"])
                ob = meshes[k]
                ob.name = name
            else:
                ob = link_dup(meshes[k], name, target)
        for kk, val in DEFAULT_PROPS.items():
            ob[kk] = float(val)
        set_matrix(ob, d["M"] @ np.diag([*d["scale"], 1.0]))

    # ---------------- endo internals (36)
    t36 = W.tooth(W.SITE_ENDO)
    M36 = t36["M"]
    S36 = t36["scale"]
    x36 = obj("t36_xray", "tooth_molar", mats["xray"], coll["endo"], M36, S36, xray=0.0)
    obj("pulp36", "pulp36", mats["pulp"], coll["endo"], M36, S36, vis=0.0)
    for i in range(3):
        obj(f"gutta36_{i}", f"gutta36_{i}", mats["gutta"], coll["endo"], M36, S36, fill=-0.01)
    obj("caries36", "caries36", mats["caries"], coll["endo"], M36, S36, vis=0.0)
    obj("resto36", "resto36", mats["resto"], coll["endo"], M36, S36, fill=-0.01)

    # ---------------- jaw
    obj("bone_full", "bone_full", mats["bone"], coll["jaw"])
    obj("gum_full", "gum_full", mats["gum"], coll["jaw"])
    for name in W.CUTS:
        obj(f"bone_{name}", f"bone_{name}", mats["bone"], coll["cuts"], vis=0.0)
        obj(f"gum_{name}", f"gum_{name}", mats["gum"], coll["cuts"], vis=0.0)
    # tissue cut segments must not be clipped by their own cutter: give them a material copy without cutters
    for o in coll["cuts"].objects:
        base = o.data.materials[0]
        o.data.materials[0] = _no_cut_copy(base)

    # ---------------- surgery pieces (site B)
    for nm, geo, mat in (("flapB", "flapB", "gum"), ("osteoB", "osteoB", "bone")):
        o = obj(nm, geo, _no_cut_copy(mats[mat]), coll["surgery"], vis=0.0)
    obj("fillB", "fillB", mats["fill_bone"], coll["surgery"], fill=-0.01)

    # ---------------- implant site (C)
    Mc = W.site_frame(W.SITE_IMPLANT)
    Mi = Mc.copy()
    Mi[:3, 3] = Mi[:3, 3] + Mi[:3, 2] * 0.06
    obj("implant", "implant", mats["titanium"], coll["implant"], Mi, fill=-0.01)
    obj("healing", "healing", mats["ti_simple"], coll["implant"], Mi, vis=0.0)
    obj("abutment", "abutment", mats["ti_simple"], coll["implant"], Mi, vis=0.0)
    obj("osseoC", "osseoC", mats["osseo"], coll["implant"], fill=-0.01)
    t46 = W.tooth(W.SITE_IMPLANT)
    obj("crown46", "crown46", mats["ceramic"], coll["implant"], t46["M"], t46["scale"], vis=0.0)

    # ---------------- upper arch soft tissue
    obj("upper_gum", "upper_gum", _no_cut_copy(mats["gum"]), coll["upper"])

    # ---------------- camera, lights, world
    for fmt in CFG.PLATES:
        cam = bpy.data.objects.new(f"cam_{fmt}", bpy.data.cameras.new(f"cam_{fmt}"))
        scn.collection.objects.link(cam)
        cam.data.sensor_fit = "HORIZONTAL" if fmt == "9x16" else "VERTICAL"
        cam.data.sensor_width = 36
        cam.data.sensor_height = 36
        cam.data.clip_start = 0.05
        cam.data.clip_end = 200
        cam.data.dof.use_dof = True
    scn.camera = bpy.data.objects["cam_9x16"]

    rig = bpy.data.objects.new("light_rig", None)
    scn.collection.objects.link(rig)

    def area(name, loc, energy, size, color):
        Ld = bpy.data.lights.new(name, "AREA")
        Ld.energy, Ld.size, Ld.color = energy, size, color
        Ld.use_shadow = False
        o = bpy.data.objects.new(name, Ld)
        scn.collection.objects.link(o)
        set_matrix(o, look_at_matrix(loc, (0, 0, 0)))
        o.parent = rig
        return o

    area("key", (-5, -7, 6), 2600, 7, (1.0, 0.98, 0.95))
    area("fill", (7, -5, 1), 1100, 7, (0.86, 0.92, 1.0))
    area("rim", (0, 8, 5), 1800, 6, (0.75, 0.86, 1.0))
    area("under", (0, -4, -6), 500, 6, (0.9, 0.94, 1.0))
    w = bpy.data.worlds.new("stage")
    scn.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.80, 0.87, 0.93, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.75
    eevee_settings(scn)
    return scn


_NOCUT = {}


def _no_cut_copy(mat):
    """Same material but without the cut-away clip (for the section pieces)."""
    if mat.name in _NOCUT:
        return _NOCUT[mat.name]
    import materials as MT
    kind = "gum" if "gum" in mat.name else "bone"
    saved = {n: bpy.data.objects.get(n) for n in W.CUTS}
    for n, o in saved.items():
        if o:
            o.name = n + "_tmp"
    m = MT.tissue_material(mat.name + "_nocut", kind)
    for n, o in saved.items():
        if o:
            o.name = n
    _NOCUT[mat.name] = m
    return m


if __name__ == "__main__":
    out = ROOT / "build" / "scene.blend"
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    scn = build()
    if "--no-anim" not in sys.argv:
        import bake
        bake.bake_all(scn)
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=False)
    print("SAVED", out, flush=True)
    import os
    os._exit(0)
