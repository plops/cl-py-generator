from __future__ import annotations
from casadi import *
x=SX.sym("x")
y=SX.sym("y")
z=SX.sym("z")
nlp=dict(x=vertcat(x, y, z), f=(x**2)+100*(z**2), g=z+((1-x)**2)+-y)
S=nlpsol("S", "ipopt", nlp)
r=S(x0=[2.5    , 3, 0.75    ], lbg=0, ubg=0)
x_opt=r["x"]
print(f"x_opt={x_opt}")