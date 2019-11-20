# https://youtu.be/CQDsT81GyS8?t=3238 Valentin Haenel: Create CUDA kernels from Python using Numba and CuPy | PyData Amsterdam 2019
# pip3 install --user cupy-cuda101
import numpy as np
import cupy as cp
import math
ary=cp.arange(10).reshape((2,5,))
print(repr(ary))
print("dtype = {}".format(ary.dtype))
print("shape = {}".format(ary.shape))
print("strides = {}".format(ary.strides))
print("device = {}".format(ary.device))