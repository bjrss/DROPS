"""Material library — colours come from om.config so brand / clinical palette is exact.

Every material that needs to appear / disappear uses a *binary* alpha mask
(threshold of a progress value against a per-vertex or procedural field), so the
dithered transparency is noise-free even at low EEVEE sample counts.  The only
truly translucent material is the x-ray glass, which uses the BLENDED method.
"""
import bpy

from om.config import RGB
from bl_util import NT


def lin(c):
    """sRGB tuple -> linear (Blender colour sockets are linear)."""
    def f(x):
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    return tuple(f(x) for x in c)


L = {k: lin(v) for k, v in RGB.items()}


def _mat(name, method="DITHERED"):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.surface_render_method = method
    m.use_backface_culling = False
    return m


def principled(nt, **kw):
    b = nt.n("ShaderNodeBsdfPrincipled")
    for k, v in kw.items():
        nt.set(b, k, v)
    return b


def reveal_mask(nt, field, prop, edge=0.0):
    """alpha = field < prop  (binary).  `prop` is an object attribute name."""
    p = nt.attr(prop)
    return nt.math("LESS_THAN", field, p.outputs["Fac"])


def with_alpha(nt, shader, alpha):
    """Mix shader with transparent by alpha (binary or not)."""
    tr = nt.n("ShaderNodeBsdfTransparent")
    mx = nt.n("ShaderNodeMixShader")
    nt.link(alpha, mx.inputs[0])
    nt.link(tr.outputs[0], mx.inputs[1])
    nt.link(shader.outputs[0] if hasattr(shader, "outputs") else shader, mx.inputs[2])
    return mx


REVEAL_CENTER = None


def reveal_alpha(nt):
    """1 where the world position lies within scene['rr'] (+ noise) of the hero tooth."""
    geo = nt.n("ShaderNodeNewGeometry")
    rel = nt.vmath("SUBTRACT", geo.outputs["Position"], tuple(REVEAL_CENTER))
    dist = nt.vmath("LENGTH", rel)
    noi = nt.n("ShaderNodeTexNoise", Scale=2.5, Detail=0.0)
    nt.link(geo.outputs["Position"], noi.inputs["Vector"])
    d2 = nt.math("ADD", dist, nt.math("MULTIPLY", noi.outputs["Fac"], 0.9))
    rr = nt.attr("rr", "VIEW_LAYER").outputs["Fac"]
    return nt.math("LESS_THAN", d2, rr)


def clip_fields(nt):
    """Common clip: object attribute 'vis' (0/1) and cut-away boxes evaluated in world space.
    Returns alpha socket (1 = visible)."""
    vis = nt.attr("vis").outputs["Fac"]
    return vis


# ------------------------------------------------------------------ tooth
def tooth_material(name="tooth", stain=False):
    m = _mat(name)
    nt = NT(m)
    tc = nt.n("ShaderNodeTexCoord")
    sep = nt.n("ShaderNodeSeparateXYZ")
    nt.link(tc.outputs["Object"], sep.inputs[0])
    z = sep.outputs["Z"]
    k = nt.n("ShaderNodeMapRange", **{"From Min": -0.02, "From Max": 0.07})
    nt.link(z, k.inputs["Value"])
    kz = k.outputs["Result"]
    # subtle enamel variation
    noi = nt.n("ShaderNodeTexNoise", Scale=18.0, Detail=1.0)
    nt.link(tc.outputs["Object"], noi.inputs["Vector"])
    enamel = nt.mixc(nt.math("MULTIPLY", noi.outputs["Fac"], 0.25), L["enamel"], (0.93, 0.89, 0.80))
    col = nt.mixc(kz, L["dentin"], enamel)
    if stain:
        # caries stain on the occlusal fissure: attribute 'caries' (0..1) * proximity
        sa = nt.attr("caries_amt").outputs["Fac"]
        cm = nt.attr("caries_mask", "GEOMETRY").outputs["Fac"]
        f = nt.math("MULTIPLY", sa, cm, clamp=True)
        col = nt.mixc(f, col, L["caries"])
    rough = nt.math("ADD", nt.math("MULTIPLY", kz, -0.25), 0.55)
    b = principled(nt, **{"Base Color": col, "Roughness": rough,
                          "Subsurface Weight": 0.22, "Subsurface Scale": 0.035,
                          "Subsurface Radius": (1.0, 0.55, 0.35),
                          "Coat Weight": nt.math("MULTIPLY", kz, 0.45), "Coat Roughness": 0.08,
                          "Specular IOR Level": 0.55})
    alpha = nt.attr("vis").outputs["Fac"]
    # dissolve field: object z + noise (for the x-ray scan wipe): visible where z < wipe
    wipe = nt.attr("wipe").outputs["Fac"]
    n2 = nt.n("ShaderNodeTexNoise", Scale=9.0, Detail=0.0)
    nt.link(tc.outputs["Object"], n2.inputs["Vector"])
    field = nt.math("ADD", z, nt.math("MULTIPLY", n2.outputs["Fac"], 0.12))
    keep = nt.math("GREATER_THAN", field, wipe)
    alpha = nt.math("MULTIPLY", alpha, keep)
    if not stain and REVEAL_CENTER is not None:
        alpha = nt.math("MULTIPLY", alpha, reveal_alpha(nt))
    nt.out(with_alpha(nt, b, alpha))
    return m


def xray_material(name="xray", color="brand_light"):
    """Glass with fresnel rim in brand blue — 'modo rayos X'."""
    m = _mat(name, "BLENDED")
    m.use_transparency_overlap = False
    nt = NT(m)
    lw = nt.n("ShaderNodeLayerWeight", Blend=0.35)
    fr = lw.outputs["Facing"]
    rim = nt.math("POWER", fr, 2.2)
    amt = nt.attr("xray").outputs["Fac"]
    em = nt.n("ShaderNodeEmission", Color=L[color])
    nt.link(nt.math("ADD", nt.math("MULTIPLY", rim, 4.0), 0.25), em.inputs["Strength"])
    gl = principled(nt, **{"Base Color": (0.85, 0.93, 1.0), "Roughness": 0.08, "Alpha": 1.0,
                           "Specular IOR Level": 0.8})
    add = nt.n("ShaderNodeAddShader")
    nt.link(em.outputs[0], add.inputs[0])
    nt.link(gl.outputs[0], add.inputs[1])
    alpha = nt.math("MULTIPLY", amt, nt.math("ADD", nt.math("MULTIPLY", rim, 0.9), 0.12), clamp=True)
    nt.out(with_alpha(nt, add, alpha))
    return m


def pulp_material(name="pulp"):
    m = _mat(name)
    nt = NT(m)
    infl = nt.attr("inflame").outputs["Fac"]
    pulse = nt.attr("pulse").outputs["Fac"]
    col = nt.mixc(infl, L["pulp"], L["pulp_inflamed"])
    em = nt.math("MULTIPLY", infl, pulse)
    b = principled(nt, **{"Base Color": col, "Roughness": 0.45, "Subsurface Weight": 0.3,
                          "Subsurface Scale": 0.02, "Emission Color": L["pulp_inflamed"],
                          "Emission Strength": em})
    # dissolve: per-vertex random 'rnd' vs attribute 'dissolve'
    tc = nt.n("ShaderNodeTexCoord")
    n2 = nt.n("ShaderNodeTexNoise", Scale=22.0, Detail=0.0)
    nt.link(tc.outputs["Object"], n2.inputs["Vector"])
    d = nt.attr("dissolve").outputs["Fac"]
    keep = nt.math("GREATER_THAN", n2.outputs["Fac"], nt.math("ADD", nt.math("MULTIPLY", d, 1.2), -0.1))
    alpha = nt.math("MULTIPLY", nt.attr("vis").outputs["Fac"], keep)
    nt.out(with_alpha(nt, b, alpha))
    return m


def fill_material(name, color, rough=0.4, field="u", emission_hot=None, metal=0.0, aniso=0.0, coat=0.0, sss=0.0,
                  hot_gain=6.0):
    """Generic material revealed progressively by a per-vertex float `field`
    compared to the object attribute 'fill'.  'hot' (0..1) adds an emissive glow
    (used for the light line 'cooling' into gutta-percha / titanium)."""
    m = _mat(name)
    nt = NT(m)
    hot = nt.attr("hot").outputs["Fac"]
    b = principled(nt, **{"Base Color": L[color] if isinstance(color, str) else color,
                          "Roughness": rough, "Metallic": metal, "Anisotropic": aniso,
                          "Coat Weight": coat, "Coat Roughness": 0.1,
                          "Subsurface Weight": sss, "Subsurface Scale": 0.02,
                          "Emission Color": L[emission_hot or "line_core"],
                          "Emission Strength": nt.math("MULTIPLY", hot, hot_gain)})
    fld = nt.attr(field, "GEOMETRY").outputs["Fac"]
    keep = nt.math("LESS_THAN", fld, nt.attr("fill").outputs["Fac"])
    alpha = nt.math("MULTIPLY", nt.attr("vis").outputs["Fac"], keep)
    nt.out(with_alpha(nt, b, alpha))
    return m


def tissue_material(name, kind, section_kind=None):
    """Gum or bone outer surface + flat section faces (per-vertex attribute 'cap' = 1)
    + world-space cut-away peel driven by cutter empties.  `kind` in {'gum','bone'}."""
    m = _mat(name)
    nt = NT(m)
    tc = nt.n("ShaderNodeTexCoord")
    cap = nt.attr("cap", "GEOMETRY").outputs["Fac"]
    depth = nt.attr("depth", "GEOMETRY").outputs["Fac"]
    if kind == "gum":
        noi = nt.n("ShaderNodeTexNoise", Scale=40.0, Detail=1.0)
        nt.link(tc.outputs["Object"], noi.inputs["Vector"])
        gum_deep = tuple(c * 0.86 for c in L["gum"])
        base = nt.mixc(nt.math("MULTIPLY", noi.outputs["Fac"], 0.35), gum_deep, (0.70, 0.24, 0.27))
        sect = nt.mixc(nt.math("MULTIPLY", noi.outputs["Fac"], 0.5), (0.86, 0.42, 0.42), (0.78, 0.30, 0.32))
        col = nt.mixc(cap, base, sect)
        heal = nt.attr("heal").outputs["Fac"]
        b = principled(nt, **{"Base Color": col, "Roughness": nt.math("ADD", nt.math("MULTIPLY", cap, 0.3), 0.32),
                              "Subsurface Weight": 0.35, "Subsurface Scale": 0.04,
                              "Subsurface Radius": (1.0, 0.3, 0.25), "Coat Weight": 0.25,
                              "Coat Roughness": 0.18, "Emission Color": L["brand_light"],
                              "Emission Strength": nt.math("MULTIPLY", heal, 0.0)})
    else:
        vor = nt.n("ShaderNodeTexVoronoi", Scale=14.0, Randomness=0.9)
        vor.feature = "DISTANCE_TO_EDGE"
        nt.link(tc.outputs["Object"], vor.inputs["Vector"])
        # trabecular struts: bright thin network on darker marrow spaces
        strut = nt.math("LESS_THAN", vor.outputs["Distance"], 0.085)
        cortical = nt.math("LESS_THAN", depth, 0.10)
        tra = nt.mixc(strut, (0.74, 0.62, 0.48), L["bone"])
        trab_col = nt.mixc(cortical, tra, (0.96, 0.93, 0.86))
        noi = nt.n("ShaderNodeTexNoise", Scale=30.0, Detail=1.0)
        nt.link(tc.outputs["Object"], noi.inputs["Vector"])
        outer = nt.mixc(nt.math("MULTIPLY", noi.outputs["Fac"], 0.3), L["bone"], (0.86, 0.80, 0.68))
        col = nt.mixc(cap, outer, trab_col)
        b = principled(nt, **{"Base Color": col, "Roughness": 0.7, "Subsurface Weight": 0.1,
                              "Subsurface Scale": 0.02})
    # visibility + cut-away peel (objects 'cutA','cutB','cutC' give local frames)
    alpha = nt.attr("vis").outputs["Fac"]
    if REVEAL_CENTER is not None:
        alpha = nt.math("MULTIPLY", alpha, reveal_alpha(nt))
    for cname in ("cutA", "cutB", "cutC"):
        cut = bpy.data.objects.get(cname)
        if cut is None:
            continue
        tcc = nt.n("ShaderNodeTexCoord")
        tcc.object = cut
        sep = nt.n("ShaderNodeSeparateXYZ")
        nt.link(tcc.outputs["Object"], sep.inputs[0])
        # cutter local: x along arch (|x|<1 inside), y buccal (-) ... peel where y < 0 and |x| < 1
        inside_x = nt.math("LESS_THAN", nt.math("ABSOLUTE", sep.outputs["X"]), 1.0)
        buccal = nt.math("LESS_THAN", sep.outputs["Y"], 0.0)
        hide = nt.math("MULTIPLY", inside_x, buccal)
        alpha = nt.math("MULTIPLY", alpha, nt.math("SUBTRACT", 1.0, hide))
    nt.out(with_alpha(nt, b, alpha))
    return m


def simple_material(name, color, rough=0.4, metal=0.0, coat=0.0, aniso=0.0, sss=0.0, emis=0.0, ecol=None):
    m = _mat(name)
    nt = NT(m)
    hot = nt.attr("hot").outputs["Fac"]
    b = principled(nt, **{"Base Color": L[color] if isinstance(color, str) else color, "Roughness": rough,
                          "Metallic": metal, "Coat Weight": coat, "Coat Roughness": 0.08,
                          "Anisotropic": aniso, "Subsurface Weight": sss, "Subsurface Scale": 0.03,
                          "Emission Color": L[ecol or "line_core"],
                          "Emission Strength": nt.math("ADD", nt.math("MULTIPLY", hot, 6.0), emis)})
    alpha = nt.attr("vis").outputs["Fac"]
    nt.out(with_alpha(nt, b, alpha))
    return m
