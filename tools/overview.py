"""Static overview renders of the world (no animation) for layout checks."""
import sys, os
sys.path.insert(0, '/home/user/DROPS'); sys.path.insert(0, '/home/user/DROPS/blender')
import bpy, numpy as np
import build_scene as B
from bl_util import set_matrix, look_at_matrix
from om import world as W
scn = B.build()
scn.render.use_motion_blur = False
scn.eevee.taa_render_samples = 8
cam = bpy.data.objects['cam_9x16']
views = {
 'front': ((0, -14, 1.0), (0, 2.0, -0.8), 30, (900, 600)),
 'top': ((0, 2.6, 16), (0, 2.6, -0.8), 30, (700, 900)),
 'site36': None, 'site38': None, 'site46': None,
}
mode = sys.argv[-1]
for name in [n for n in views if mode in ('all', n)]:
    if name.startswith('site'):
        fdi = int(name[4:])
        Mf = W.site_frame(fdi)
        c = Mf[:3, 3] + np.array([0, 0, 0.0])
        out = Mf[:3, 1] * -1
        pos = c + out * 7.5 + np.array([0, 0, 1.2])
        tgt, fov_res = c, (540, 960)
        # open the cut
        cut = {36: 'cutA', 38: 'cutB', 46: 'cutC'}[fdi]
        e = bpy.data.objects[cut]
        set_matrix(e, W.cut_matrix(cut))
        bpy.data.objects[f'bone_{cut}']['vis'] = 1.0
        bpy.data.objects[f'gum_{cut}']['vis'] = 1.0
        if fdi == 46:
            bpy.data.objects['implant']['fill'] = 2.0
        lens = 60
        res = fov_res
    else:
        pos, tgt, lens, res = views[name]
    set_matrix(cam, look_at_matrix(pos, tgt))
    cam.data.lens = lens
    cam.data.dof.use_dof = False
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.filepath = f'/home/user/DROPS/build/test/ov_{name}.png'
    bpy.ops.render.render(write_still=True)
    if name.startswith('site'):
        set_matrix(e, W.cut_matrix(cut) @ np.diag([1,1,1,1.0]) if False else np.eye(4) @ __import__('numpy').array([[1,0,0,0],[0,1,0,-50],[0,0,1,0],[0,0,0,1.0]]) @ np.eye(4))
os._exit(0)
