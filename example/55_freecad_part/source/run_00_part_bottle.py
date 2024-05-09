import numpy as np
import Part
import math
from FreeCAD import Base
_code_git_version="c4e9cfdde50d6135bed970993b5a0acac807556a"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="23:28:30 of Thursday, 2024-05-09 (GMT+1)"
w=(50.    )
h=(70.    )
thick=(30.    )
p1=Base.Vector(((w)/(-2)), 0, 0)
p2=Base.Vector(((w)/(-2)), ((thick)/(-4)), 0)
p3=Base.Vector(0, ((thick)/(-2)), 0)
p4=Base.Vector(((w)/(2)), ((thick)/(-4)), 0)
p5=Base.Vector(((w)/(2)), 0, 0)
arc=Part.Arc(p2, p3, p4)
l1=Part.LineSegment(p1, p2)
l2=Part.LineSegment(p4, p5)
e1=l1.toShape()
e2=arc.toShape()
e3=l2.toShape()
wire=Part.Wire([e1, e2, e3])
M=Base.Matrix()
M.rotateZ(math.pi)
# mirror wire
wire_=wire.copy()
wire_.transformShape(M)
wire_profile=Part.Wire([wire, wire_])
face_profile=Part.Face(wire_profile)
prism=Base.Vector(0, 0, h)
body=face_profile.extrude(prism)
neck_location=Base.Vector(0, 0, h)
neck_normal=Base.Vector(0, 0, 1)
neck_r=((thick)/(4))
neck_h=((h)/(10))
neck=Part.makeCylinder(neck_r, neck_h, neck_location, neck_normal)
body=body.fuse(neck)
body=body.makeFillet(((thick)/((12.    ))), body.Edges)
Part.show(body)