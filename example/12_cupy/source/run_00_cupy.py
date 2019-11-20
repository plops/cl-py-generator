# https://youtu.be/CQDsT81GyS8?t=3238 Valentin Haenel: Create CUDA kernels from Python using Numba and CuPy | PyData Amsterdam 2019
# pip3 install --user cupy-cuda101
# pip3 install --user numba
#  sudo ln -s /opt/cuda/nvvm/lib64/libnvvm.so* /usr/lib
import numpy as np
import cupy as cp
import math
import numba
ary=cp.arange(10).reshape((2,5,))
print(repr(ary))
print("dtype = {}".format(ary.dtype))
print("shape = {}".format(ary.shape))
print("strides = {}".format(ary.strides))
print("device = {}".format(ary.device))
# numba on gpu
@numba.vectorize(["int64(int64,int64)"], target="cuda")
def add_ufunc(x, y):
    return ((x)+(y))
a=np.array([1, 2, 3, 4])
b=np.array([10, 20, 30, 40])
b_col=b[:,np.newaxis]
c=np.arange(((4)*(4))).reshape((4,4,))
print(add_ufunc(a, b))
print(add_ufunc(b_col, c))
ag=numba.cuda.to_device(a)
bg=numba.cuda.to_device(b)
bcg=numba.cuda.to_device(b_col)
print(add_ufunc(ag, bg))
out_device=numba.cuda.device_array(shape=(4,), dtype=np.float32)
add_ufunc(ag, bg, out=out_device)
print(out_device.copy_to_host())
out_device2=numba.cuda.device_array(shape=(4,), dtype=np.float32)
add_ufunc(ag, bcg, out=out_device2)
print(out_device2.copy_to_host())