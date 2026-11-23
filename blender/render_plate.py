"""Render the 3D plate for one format and a frame range.

    python blender/render_plate.py --fmt 9x16 --start 0 --end 1799 --scale 0.75 --samples 12 --out build/plates/9x16
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "blender"))
import bpy  # noqa: E402

from om import config as CFG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fmt", default="9x16")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=CFG.NFRAMES - 1)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--scale", type=float, default=0.75)
ap.add_argument("--samples", type=int, default=12)
ap.add_argument("--blend", default=str(ROOT / "build" / "scene.blend"))
ap.add_argument("--out", default=None)
ap.add_argument("--nomb", action="store_true")
ap.add_argument("--frames", default=None, help="comma separated frame list")
a = ap.parse_args([x for x in sys.argv[1:] if x != "--"])

bpy.ops.wm.open_mainfile(filepath=a.blend)
scn = bpy.context.scene
P = CFG.PLATES[a.fmt]
scn.camera = bpy.data.objects[f"cam_{a.fmt}"]
scn.render.resolution_x = int(round(P["w"] * a.scale / 2) * 2)
scn.render.resolution_y = int(round(P["h"] * a.scale / 2) * 2)
scn.eevee.taa_render_samples = a.samples
scn.render.use_motion_blur = not a.nomb
scn.unit_settings.scale_length = 0.01
out = Path(a.out or (ROOT / "build" / "plates" / a.fmt))
out.mkdir(parents=True, exist_ok=True)
frames = [int(x) for x in a.frames.split(",")] if a.frames else range(a.start, a.end + 1, a.step)
for f in frames:
    p = out / f"{f:04d}.png"
    if p.exists():
        continue
    scn.frame_set(f)
    scn.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    print("FRAME", f, flush=True)
print("DONE", a.fmt, a.start, a.end, flush=True)
os._exit(0)
