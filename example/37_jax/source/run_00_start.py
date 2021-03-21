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
_code_git_version="385a839c43ce62d00606443125970d4d6fa86989"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="21:24:45 of Sunday, 2021-03-21 (GMT+1)"
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
    return ((res.values.astype(jnp.float32))-(xs.values.astype(jnp.float32))).ravel()
xs_mod=model(((0.10    ),(-0.20    ),(0.50    ),(30.    ),), xs=xs, noise=False)
def jax_model(param, x, y, goal):
    x0, y0, radius, amp=param
    r=jnp.sqrt(((((((x[...,jnp.newaxis])+(x0)))**(2)))+(((((y[jnp.newaxis,...])+(y0)))**(2)))))
    s=abs(((amp)*(sinc(((r)/(radius))))))
    return ((goal.astype(jnp.float32))-(s.astype(jnp.float32))).ravel()
j=jit(jacrev(jax_model, argnums=0))
def j_for_call(param, xs=None):
    x=xs.x.values.astype(jnp.float32)
    y=xs.y.values.astype(jnp.float32)
    goal=xs.values.astype(jnp.float32)
    return jnp.array(j(param, x, y, goal))
x0=((0.120    ),(-0.270    ),(0.450    ),(28.    ),)
param_opt=scipy.optimize.least_squares(model_merit, x0, jac=j_for_call, verbose=2, kwargs={("xs"):(xs_mod)})
print(param_opt)
xs_fit=model(param_opt.x, xs=xs)
plt.figure(figsize=(14,6,))
pl=(1,3,)
ax=plt.subplot2grid(pl, (0,0,))
xrp.imshow(xs_mod)
ax=plt.subplot2grid(pl, (0,1,))
xrp.imshow(xs_fit)
ax=plt.subplot2grid(pl, (0,2,))
xrp.imshow(((xs_fit)-(xs_mod)))