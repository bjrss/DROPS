#!/bin/bash
# Final 3D plates: both formats in parallel, 2 llvmpipe threads each.
cd "$(dirname "$0")/.."
S=${SCALE:-0.75}; N=${SAMPLES:-4}; A=${START:-75}; Z=${END:-1600}
for fmt in 9x16 16x9; do
  LP_NUM_THREADS=2 /opt/om/venv311/bin/python blender/render_plate.py --fmt $fmt --start $A --end $Z \
     --scale $S --samples $N --out build/plates/$fmt > build/plates_$fmt.log 2>&1 &
done
wait
echo ALL_PLATES_DONE
