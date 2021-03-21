import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import xarray as xr
import xarray.plot as xrp
import scipy.optimize
import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev
from jax.numpy import sqrt, newaxis, sinc, abs
_code_git_version="5311b1718a3e6ebc6d0c3f88caa4fd0db19e04ef"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="19:38:24 of Sunday, 2021-03-21 (GMT+1)"
def tanh(x):
    y=jnp.exp((((-2.0    ))*(x)))
    return (((((1.0    ))-(y)))/((((1.0    ))+(y))))
grad_tanh=grad(tanh)
print(grad_tanh((1.0    )))
nx=32
ny=27
x=jnp.linspace(-1, 1, nx)
y=jnp.linspace(-1, 1, ny)
q=jnp.sqrt(((((x[...,jnp.newaxis])**(2)))+(((y[jnp.newaxis,...])**(2)))))
xs=xr.DataArray(data=q, coords=[x, y], dims=["x", "y"])
def model(param, xs=None):
    x0, y0, radius, amp=param
    res=xs.copy()
    r=jnp.sqrt(((((((xs.x.values[...,jnp.newaxis])+(x0)))**(2)))+(((((xs.y.values[jnp.newaxis,...])+(y0)))**(2)))))
    s=abs(((amp)*(sinc(((r)/(radius))))))
    res.values=s
    return res
xs_mod=model(((0.10    ),(-0.20    ),(0.50    ),(1.0    ),), xs=xs)
xrp.imshow(xs_mod)
scipy.optimize.least_squares()