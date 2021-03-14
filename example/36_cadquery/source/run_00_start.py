# %% imports
import cadquery as cq
from Helpers import show
_code_git_version="13180afe144a504ac9efb955e967a9f18423b566"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="19:56:18 of Sunday, 2021-03-14 (GMT+1)"
length=(80.    )
height=(60.    )
thickness=(10.    )
center_hole_dia=(22.    )
cbore_hole_dia=(2.40    )
cbore_dia=(4.40    )
cbore_depth=(2.10    )
r=cq.Workplane("XY").box(length, height, thickness)
show(r, (204,204,204,(0.    ),))