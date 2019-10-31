# https://documen.tician.de/pycuda/tutorial.html
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
a=np.random.randn(4, 4).astype(np.float32)
a_gpu=cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
mod=SourceModule("""        __global__ void doublify (float* a){
                        int idx  = ((threadIdx.x)+(((4)*(threadIdx.y))));
        a[idx]*=(2);
}""")
func=mod.get_function("doublify")
func(a_gpu, block=(4,4,1,))
a_doubled=np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
print(a)