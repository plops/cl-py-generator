#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.scipy.optimize

# the model is a 2d gaussian centered at cx, cy with stddev sigma, amplitude A
# and offset o

def model(params, x, y):
    A, cx, cy, sigma, o = params
    return A * jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)) + o


# create an image with a 2d gaussian and additive normal noise

def create_image(params, x, y):
    A, cx, cy, sigma, o = params
    return model(params, x, y) + jax.random.normal(jax.random.PRNGKey(0), x.shape) * 0.1


# jax.scipy.optimize.minimize(fun, x0, args=(), *, method, tol=None, options=None)
# Minimization of scalar function of one or more variables.
# This API for this function matches SciPy with some minor deviations:
# Gradients of fun are calculated automatically using JAX’s autodiff support when required.
# The method argument is required. You must specify a solver.
# Various optional arguments in the SciPy interface have not yet been implemented.
# Optimization results may differ from SciPy due to differences in the line search implementation.
# minimize supports jit() compilation. It does not yet support differentiation or arguments in the form of multi-dimensional arrays, but support for both is planned.
# Parameters
# :
# fun (Callable) – the objective function to be minimized, fun(x, *args) -> float, where x is a 1-D array with shape (n,) and args is a tuple of the fixed parameters needed to completely specify the function. fun must support differentiation.
# x0 (Array) – initial guess. Array of real elements of size (n,), where n is the number of independent variables.
# args (tuple) – extra arguments passed to the objective function.
# method (str) – solver type. Currently only "BFGS" is supported.
# tol (float | None) – tolerance for termination. For detailed control, use solver-specific options.
# options (Mapping[str, Any] | None) –
# a dictionary of solver options. All methods accept the following generic options:
# maxiter (int): Maximum number of iterations to perform. Depending on the method each iteration may use several function evaluations.


# create the model function for the optimization, it shall return a scalar

def model_fun(params, x, y, data):
    return jnp.sum((model(params, x, y) - data) ** 2)

# this function undoes the ravel operation in model_fun

def model_fun_reshape(params, x, y, data):
    return model_fun(params, x, y, data).reshape(data.shape)

# create the data

x = jnp.linspace(-5, 5, 100)
y = jnp.linspace(-5, 5, 100)
x, y = jnp.meshgrid(x, y)
params_true = jnp.array([189.0, 1.2, 2.3, 1.0, 3.0])
data = create_image(params_true, x, y) 

# initial guess for the parameters as jax array

params0 = jnp.array([100.0, 0.0, 0.0, 1.0, 0.0])
# optimize the parameters

result = jax.scipy.optimize.minimize(
    model_fun, params0, args=(x, y, data), method="BFGS"
    )



# print the optimized parameters

print(result.x)


# 1. Jacobian Calculation (for amplitude A)
# `jax.jacfwd` calculates the forward-mode Jacobian 
# (more efficient for sensitivity to many inputs)

# argnums (int | Sequence[int]) – Optional, integer or sequence of integers. Specifies which positional argument(s) to differentiate with respect to (default 0).

get_amplitude_sensitivity = jax.jacfwd(model_fun, argnums=3)  # Sensitivity w.r.t. data



# 3. Calculate and Visualize the Sensitivity

sensitivity_map = get_amplitude_sensitivity(result.x, x, y, data)
sensitivity_map = sensitivity_map.reshape(data.shape) # Reshape to image dimensions

# --- (Visualization code using matplotlib - add your own) ---
import matplotlib.pyplot as plt
plt.imshow(data, cmap='gray') 
plt.imshow(sensitivity_map, cmap='hot', alpha=0.5) # Overlay sensitivity as a heatmap
plt.show()



# Explanation:
# Jaxopt GradientDescent: We use Jaxopt's GradientDescent solver for the optimization. This solver has built-in implicit differentiation capabilities.
# jax.jacfwd on gd.run: We now compute the Jacobian of the gd.run function, which represents the entire optimization process. This gives us the sensitivity of the fitted parameters to the input data.
# Visualization: We iterate through the sensitivity maps for each parameter and visualize them as heatmaps, similar to your previous examples.
# Key Points:
# Differentiable Minimizer: GradientDescent from Jaxopt is a differentiable minimizer, allowing us to differentiate through the optimization.
# argnums=3: We are computing the sensitivity with respect to the data argument in gd.run.
# Interpretation: The heatmaps show how each pixel influences the fitted values of each parameter.
# This example demonstrates how Jaxopt enables you to go beyond analyzing model sensitivity and directly analyze the sensitivity of the parameters obtained through optimization. This is a more powerful technique that gives you deeper insights into how your fitting process is affected by the input data.

import jax
import jax.numpy as jnp
from jaxopt import GradientDescent
from jaxopt import objective
import matplotlib.pyplot as plt

# Model and Image Creation (same as before)

def model(params, x, y):
    A, cx, cy, sigma, o = params
    return A * jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)) + o


def create_image(params, x, y):
    A, cx, cy, sigma, o = params
    return model(params, x, y) + jax.random.normal(jax.random.PRNGKey(0), x.shape) * 0.1


# --- Objective Function ---

def model_fun(params, x, y, data):
    return jnp.mean((model(params, x, y) - data) ** 2)  # Using jnp.mean for better scaling


# --- Optimization with Jaxopt ---

# 1. Create a GradientDescent solver
gd = GradientDescent(fun=model_fun, maxiter=1000)

# 2. Data and Initial Parameters (same as before)
x = jnp.linspace(-5, 5, 100)
y = jnp.linspace(-5, 5, 100)
x, y = jnp.meshgrid(x, y)
params_true = jnp.array([189.0, 1.2, 2.3, 1.0, 3.0])
data = create_image(params_true, x, y)
params0 = jnp.array([100.0, 0.0, 0.0, 1.0, 0.0])

# 3. Run the Optimization
fitted_params, state = gd.run(init_params=params0, x=x, y=y, data=data)

print("Fitted parameters:", fitted_params)

# --- Sensitivity Analysis of Fitted Parameters ---

# 1. Jacobian Calculation
get_param_sensitivity = jax.jacfwd(gd.run, argnums=3) # Sensitivity w.r.t. data

# 2. Calculate Sensitivities
sensitivity_map = get_param_sensitivity(params0, x, y, data)[0] # Get sensitivity of the output params

# 3. Visualize Sensitivity for Each Parameter
for i, param_name in enumerate(["A", "cx", "cy", "sigma", "o"]):
    plt.figure()
    plt.title(f"Sensitivity of {param_name}")
    plt.imshow(data, cmap='gray') 
    plt.imshow(sensitivity_map[i].reshape(data.shape), cmap='hot', alpha=0.5)
    plt.show()