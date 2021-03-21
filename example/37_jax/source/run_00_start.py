import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import xarray as xr
import xarray.plot as xrp
import scipy.optimize
import jax.numpy as jnp
import jax
import jax.random
from jax import grad, jit, jacfwd, jacrev
from jax.numpy import sqrt, newaxis, sinc, abs
_code_git_version="e836d3a39d4e384a6a94cf3d3b7a08e084c013a8"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="20:02:54 of Sunday, 2021-03-21 (GMT+1)"
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
def model(param, xs=None, noise=False):
    x0, y0, radius, amp=param
    res=xs.copy()
    r=jnp.sqrt(((((((xs.x.values[...,jnp.newaxis])+(x0)))**(2)))+(((((xs.y.values[jnp.newaxis,...])+(y0)))**(2)))))
    s=abs(((amp)*(sinc(((r)/(radius))))))
    if ( noise ):
        key=jax.random.PRNGKey(0)
        s=((s)+(jax.random.uniform(key, s.shape)))
    res.values=s
    return res
def model_merit(param, xs=None):
    res=model(param, xs=xs, noise=False)
    return ((res.values.astype(jnp.float64))-(xs.values.astype(jnp.float64))).ravel()
xs_mod=model(((0.10    ),(-0.20    ),(0.50    ),(10.    ),), xs=xs, noise=True)
def jax_model(param, x, y, goal):
    x0, y0, radius, amp=param
    r=jnp.sqrt(((((((x[...,jnp.newaxis])+(x0)))**(2)))+(((((y[jnp.newaxis,...])+(y0)))**(2)))))
    s=abs(((amp)*(sinc(((r)/(radius))))))
    return ((goal)-(s)).ravel()
param_opt=scipy.optimize.least_squares(model_merit, ((0.120    ),(-0.270    ),(0.80    ),(13.    ),), jac=jit(jacfwd(model_merit)), kwargs={("xs"):(xs_mod)})
xs_fit=model(param_opt.x, xs=xs)
xrp.imshow(((xs_fit)-(xs_mod)))