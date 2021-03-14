# %% imports
import cadquery as cq
from Helpers import show
_code_git_version="95173b39df78d3a608b1a96c436cc5d1ab34f036"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="20:02:19 of Sunday, 2021-03-14 (GMT+1)"
length=(80.    )
height=(60.    )
thickness=(10.    )
center_hole_dia=(22.    )
cbore_hole_dia=(2.40    )
cbore_dia=(4.40    )
cbore_depth=(2.10    )
r=cq.Workplane("XY").box(length, height, thickness).faces(">Z").workplane().hole(center_hole_dia).faces(">Z").rect(((length)-(8)), ((height)-(8)), forConstruction=True).vertices().cboreHole(cbore_hole_dia, cbore_dia, cbore_depth)
show(r, (204,204,204,(0.    ),))