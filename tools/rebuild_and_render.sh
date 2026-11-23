#!/bin/bash
cd "$(dirname "$0")/.."
python3 tools/make_geometry.py jaw cuts surgery osseo > build/geo_fine.log 2>&1 || exit 1
/opt/om/venv311/bin/python blender/build_scene.py >> build/geo_fine.log 2>&1 || exit 1
echo SCENE_READY >> build/geo_fine.log
(env LP_NUM_THREADS=1 nice -n 10 /opt/om/venv311/bin/python blender/render_plate.py --fmt 9x16 --start 75 --end 1600 --step 2 \
   --scale 0.25 --samples 2 --nomb --out build/animatic_plates/9x16 > build/animatic_plates.log 2>&1 &)
tools/render_plates.sh
