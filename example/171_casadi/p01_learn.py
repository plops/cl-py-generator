from __future__ import annotations
from casadi import *
x=SX.sym("x")
y=SX.sym("y", 5)
z=SX.sym("y", 4, 2)
f=((((x)**(2)))+(10))
f=sqrt(f)
B1=SX.zeros(4, 5)
B2=SX(4, 5)
B3=SX.eye(4)
v=SX([1, 2, 3])
M=SX([[1, 2], [2, 3], [4, 5]])