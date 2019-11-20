# https://youtu.be/CQDsT81GyS8?t=2394 Valentin Haenel: Create CUDA kernels from Python using Numba and CuPy | PyData Amsterdam 2019
import numpy as np
import math
from numba import jit
@jit(nopython=True)
def hypot(x, y):
    x=abs(x)
    y=abs(y)
    tt=min(x, y)
    x=max(x, y)
    tt=((tt)/(x))
    return ((x)*(math.sqrt(((1)+(((tt)*(tt)))))))
hypot((3.e+0), (4.e+0))
hypot.py_func((3.e+0), (4.e+0))
@jit(nopython=True)
def ex1(x, y, out):
    for i in range(x.shape[0]):
        out[i]=hypot(x[i], y[i])
in1=np.arange(10, dtype=np.float32)
in2=((1)+(((2)*(in1))))
out=np.empty_like(in1)
ex1(in1, in2, out)