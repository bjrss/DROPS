import sys, os
sys.path.insert(0, '/home/user/DROPS'); sys.path.insert(0, '/home/user/DROPS/blender')
import bpy, numpy as np
from bl_util import reset, make_mesh, eevee_settings, set_matrix, look_at_matrix
reset()
import materials as M
scn = bpy.context.scene
d = np.load('/home/user/DROPS/build/test/teeth.npz')
mt = M.tooth_material()
xr = M.xray_material()
pm = M.pulp_material()
xs = {'incisor': -1.9, 'canine': -1.0, 'premolar': -0.0, 'molar': 1.2}
for k, x in xs.items():
    ob = make_mesh(k, d[k + '_v'], d[k + '_f'])
    ob.location = (x, 0, 0)
    ob.data.materials.append(mt)
    ob['vis'] = 1.0; ob['wipe'] = -10.0
# x-ray molar copy
ob = make_mesh('molar_x', d['molar_v'], d['molar_f']); ob.location = (2.6, 0, 0)
ob.data.materials.append(xr); ob['xray'] = 1.0; ob['vis'] = 1.0
p = make_mesh('pulp', d['pulp_v'], d['pulp_f']); p.location = (2.6, 0, 0)
p.data.materials.append(pm); p['vis'] = 1.0; p['inflame'] = 1.0; p['pulse'] = 0.6; p['dissolve'] = 0.0
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scn.collection.objects.link(cam); scn.camera = cam
cam.data.lens = 50; cam.data.sensor_fit = 'HORIZONTAL'
set_matrix(cam, look_at_matrix((0.3, -9, 0.6), (0.3, 0, -0.35)))
cam.data.dof.use_dof = True; cam.data.dof.focus_distance = 9; cam.data.dof.aperture_fstop = 4
def area(name, loc, energy, size, color=(1,1,1)):
    L = bpy.data.lights.new(name, 'AREA'); L.energy = energy; L.size = size; L.color = color; L.use_shadow = False
    o = bpy.data.objects.new(name, L); scn.collection.objects.link(o); set_matrix(o, look_at_matrix(loc, (0,0,-0.3))); return o
area('key', (-4, -6, 5), 1500, 6)
area('fill', (6, -4, 1), 600, 6, (0.85, 0.92, 1.0))
area('rim', (0, 6, 4), 900, 5, (0.8, 0.9, 1.0))
w = bpy.data.worlds.new('w'); scn.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.78, 0.85, 0.92, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.8
eevee_settings(scn, samples=16, res=(1200, 600), mb=False)
scn.render.filepath = '/home/user/DROPS/build/test/teeth_preview.png'
bpy.ops.render.render(write_still=True)
os._exit(0)
