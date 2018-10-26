import sys
import os
import loopy as lp
import pyopencl as cl
import numpy as np
import pyopencl.array

X=np.random.random((7000, 3)).astype(np.float32)

ctx=cl.create_some_context(interactive=False)
q=cl.CommandQueue(ctx)
X_dev=cl.array.to_device(q, X)

knl=lp.make_kernel("{[i,j,k]:0<=i,j<M and 0<=k<N}", "D[i,j]=sqrt(sum(k, ((((X[i,k])-(X[j,k])))**(2))))", lang_version=(2018, 2))
knl=lp.set_options(knl, write_cl=True)
knl=lp.set_loop_priority(knl, "i,j")
result=knl(q, X=X_dev)

