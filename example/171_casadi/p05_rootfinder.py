from __future__ import annotations
from casadi import *
z=SX.sym("z", nz)
x=sym1(x)
g0=sin(x+z)
g1=cos(x-z)
g=Function("g", [z, x], [g0, g1])