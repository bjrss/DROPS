"""Blender helpers (bpy 4.5, EEVEE Next)."""
import math

import bmesh
import bpy
import numpy as np


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_mesh(name, v, f, smooth=True, attrs=None, collection=None, recalc=True):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", np.asarray(v, np.float32).ravel())
    me.loops.add(len(f) * 3)
    me.loops.foreach_set("vertex_index", np.asarray(f, np.int32).ravel())
    me.polygons.add(len(f))
    me.polygons.foreach_set("loop_start", np.arange(0, len(f) * 3, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(f), 3, np.int32))
    me.update(calc_edges=True)
    me.validate()
    if recalc:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
    me.polygons.foreach_set("use_smooth", np.full(len(me.polygons), smooth, bool))
    if attrs:
        for an, av in attrs.items():
            av = np.asarray(av, np.float32)
            if av.ndim == 1:
                a = me.attributes.new(an, "FLOAT", "POINT")
                a.data.foreach_set("value", av)
            else:
                a = me.attributes.new(an, "FLOAT_VECTOR", "POINT")
                a.data.foreach_set("vector", av.ravel())
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def link_dup(src, name, collection=None):
    ob = bpy.data.objects.new(name, src.data)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def set_matrix(ob, M):
    import mathutils
    ob.matrix_world = mathutils.Matrix([list(r) for r in np.asarray(M, float)])


# ------------------------------------------------------------------ nodes
class NT:
    """Minimal node-tree builder."""

    def __init__(self, mat):
        self.mat = mat
        mat.use_nodes = True
        self.t = mat.node_tree
        self.t.nodes.clear()
        self.x = 0

    def n(self, kind, **inputs):
        node = self.t.nodes.new(kind)
        node.location = (self.x, 0)
        self.x += 200
        for k, v in inputs.items():
            if k.startswith("_"):
                setattr(node, k[1:], v)
            else:
                self.set(node, k, v)
        return node

    def set(self, node, key, v):
        sock = node.inputs[key]
        if hasattr(v, "bl_idname") or isinstance(v, bpy.types.NodeSocket):
            out = v if isinstance(v, bpy.types.NodeSocket) else v.outputs[0]
            self.t.links.new(out, sock)
        else:
            if isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == "RGBA":
                v = (*v, 1.0)
            sock.default_value = v

    def link(self, a, b):
        self.t.links.new(a, b)

    def math(self, op, a, b=None, clamp=False):
        m = self.t.nodes.new("ShaderNodeMath")
        m.operation = op
        m.use_clamp = clamp
        m.location = (self.x, -300)
        self.x += 120
        for i, v in enumerate([a, b]):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.t.links.new(v if isinstance(v, bpy.types.NodeSocket) else v.outputs[0], m.inputs[i])
        return m.outputs[0]

    def vmath(self, op, a, b=None):
        m = self.t.nodes.new("ShaderNodeVectorMath")
        m.operation = op
        for i, v in enumerate([a, b]):
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                m.inputs[i].default_value = v
            else:
                self.t.links.new(v if isinstance(v, bpy.types.NodeSocket) else v.outputs[0], m.inputs[i])
        out = m.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else m.outputs["Vector"]
        return out

    def mixc(self, fac, a, b):
        m = self.t.nodes.new("ShaderNodeMix")
        m.data_type = "RGBA"
        for sock, v in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
            if isinstance(v, (int, float)):
                sock.default_value = v
            elif isinstance(v, (tuple, list)):
                sock.default_value = (*v, 1.0) if len(v) == 3 else v
            else:
                self.t.links.new(v if isinstance(v, bpy.types.NodeSocket) else v.outputs[0], sock)
        return m.outputs[2]

    def attr(self, name, kind="OBJECT"):
        a = self.t.nodes.new("ShaderNodeAttribute")
        a.attribute_type = kind
        a.attribute_name = name
        return a

    def out(self, shader):
        o = self.t.nodes.new("ShaderNodeOutputMaterial")
        self.link(shader if isinstance(shader, bpy.types.NodeSocket) else shader.outputs[0], o.inputs["Surface"])
        return o


def eevee_settings(scene, samples=6, res=(810, 1440), mb=True):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    e = scene.eevee
    e.taa_render_samples = samples
    e.use_raytracing = False
    e.use_shadows = False
    e.use_gtao = True
    e.gtao_distance = 0.25
    e.use_volumetric_shadows = False
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.use_motion_blur = mb
    scene.render.motion_blur_shutter = 0.5
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
    scene.view_settings.exposure = 0.0
    scene.render.fps = 30


def look_at_matrix(pos, target, roll=0.0):
    pos = np.asarray(pos, float)
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        r, u = c * r + s * u, -s * r + c * u
    M = np.eye(4)
    M[:3, 0] = r
    M[:3, 1] = u
    M[:3, 2] = -f
    M[:3, 3] = pos
    return M
