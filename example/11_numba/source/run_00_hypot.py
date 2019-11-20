# https://youtu.be/CQDsT81GyS8?t=2394 Valentin Haenel: Create CUDA kernels from Python using Numba and CuPy | PyData Amsterdam 2019
import numpy as np
from numba import jit
@jit
def hypot(x, y):
    x=np.abs(x)
    y=np.abs(y)
    tt=np.min(x, y)
    x=np.max(x, y)
    tt=((tt)/(x))
    return ((x)*(np.sqrt(((1)+(((tt)*(tt)))))))