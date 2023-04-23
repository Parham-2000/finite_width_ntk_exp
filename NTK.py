### **NTK**

import jax.numpy as jnp

from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

import functools

import neural_tangents as nt
from neural_tangents import stax

skey = random.PRNGKey(1337)

# **UNDERSTANDING THE ROLE OF DEPTH**
# Here, we will investigate the role on depth and non-linearities for NTK. Remarkably, performance improves as we make it deeper

from neural_tangents._src.stax.elementwise import Relu
# neural network with 2 layers
layer_sz= np.array([2])
"""
Activation functions in stax 
Relu: Rectified linear unit (ReLU) activation function.
Erf: very similar to Tanh (Hyperbolic tangent) and has a closed-form kernel expression
"""
activation_fn = np.array(["Relu"])
layer_fn = []

# To manually create your desired Neural Net architecture
# for i in range(layer_sz.shape[0]):
#  layer_fn.append(stax.Dense(layer_sz[i]))
#  layer_fn.append(getattr(stax, activation_fn[i])())

# To add quickly more layers and see the changes
# Note that the width doesn't matter since we're computing infinite width NTK
# only the number of layers and activation functions matter

DEPTH = 10

# please also try "ReLu"

for i in range(DEPTH):
  layer_fn.append(stax.Dense(1))
  layer_fn.append(stax.Erf())
layer_fn.append(stax.Dense(1))

init_fn, apply_fn, kernel_fn = stax.serial(*layer_fn)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnames='get',static_argnums=(2,))

predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, y_train)
ntk_mean = predict_fn(x_test= X_test, get="ntk")

# We now investigate not just the absolute performance of infinitely wide neural networks, but also their ability to extract genuinely non-linear features from the data. Our key benchmark is thus linear kitchen sink constructed above

linear_oos_preds['ntk'] = ntk_mean
linear_oos_preds['mkt'] = 1.

ntk_mean

sr_plot(predictions=linear_oos_preds, returns=y_test, dates = dates)
managed_rets = linear_oos_preds * y_test
tstats = regression_with_tstats(predicted_variable=managed_rets['ntk'], explanatory_variables=managed_rets[[0.00001, 'mkt']])
print(tstats)