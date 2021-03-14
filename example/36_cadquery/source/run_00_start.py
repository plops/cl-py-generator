# %% imports
import cadquery as cq
from Helpers import show
_code_git_version="225585ac54655a966adc9b86ac83631108f6033e"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="19:59:24 of Sunday, 2021-03-14 (GMT+1)"
length=(80.    )
height=(60.    )
thickness=(10.    )
center_hole_dia=(22.    )
cbore_hole_dia=(2.40    )
cbore_dia=(4.40    )
cbore_depth=(2.10    )
r=cq.Workplane("XY").box(length, height, thickness).faces(">Z").workplane().hole(center_hole_dia)
show(r, (204,204,204,(0.    ),))