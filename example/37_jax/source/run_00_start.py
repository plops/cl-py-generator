import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import xarray as xr
import xarray.plot as xrp
import scipy.optimize
import jax.numpy as jnp
import jax
import jax.random
import jax.config
from jax import grad, jit, jacfwd, jacrev
from jax.numpy import sqrt, newaxis, sinc, abs
jax.config.update("jax_enable_x64", True)
_code_git_version="5cfe69f6bfb54572d916832806badc3e5d9da62b"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="21:04:55 of Sunday, 2021-03-21 (GMT+1)"
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
    xx=xs.x.values[...,jnp.newaxis]
    yy=xs.y.values[jnp.newaxis,...]
    r=jnp.sqrt(((((((xx)+(x0)))**(2)))+(((((yy)+(y0)))**(2)))))
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
    return ((goal.astype(jnp.float64))-(s.astype(jnp.float64))).ravel()
j=jit(jacrev(jax_model, argnums=0))
def j_for_call(param, xs=None):
    x=xs.x.values.astype(jnp.float64)
    y=xs.y.values.astype(jnp.float64)
    goal=xs.values.astype(jnp.float64)
    return j(param, x, y, goal)
x0=((0.120    ),(-0.270    ),(0.30    ),(13.    ),)
param_opt=scipy.optimize.least_squares(model_merit, x0, jac=j_for_call, xtol=None, verbose=2, kwargs={("xs"):(xs_mod)})
xs_fit=model(param_opt.x, xs=xs)
pl=(1,3,)
ax=plt.subplot2grid(pl, (0,0,))
xrp.imshow(xs_mod)
ax=plt.subplot2grid(pl, (0,1,))
xrp.imshow(xs_fit)
ax=plt.subplot2grid(pl, (0,2,))
xrp.imshow(((xs_fit)-(xs_mod)))