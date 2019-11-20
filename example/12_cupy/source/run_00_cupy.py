# https://youtu.be/CQDsT81GyS8?t=3238 Valentin Haenel: Create CUDA kernels from Python using Numba and CuPy | PyData Amsterdam 2019
# pip3 install --user cupy-cuda101
import numpy as np
import cupy as cp
import math
from numba import vectorize
ary=cp.arange(10).reshape((2,5,))
print(repr(ary))
print("dtype = {}".format(ary.dtype))
print("shape = {}".format(ary.shape))
print("strides = {}".format(ary.strides))
print("device = {}".format(ary.device))
# numba on gpu
@vectorize(["int64(int64,int64)"], target="cuda")
def add_ufunc(x, y):
    return ((x)+(y))
a=np.array([1, 2, 3, 4])
b=np.array([10, 20, 30, 40])
b_col=b[:,np.newaxis]
c=np.arange(((4)*(4))).reshape((4,4,))
print(add_ufunc(a, b))
print(add_ufunc(b_col, c))