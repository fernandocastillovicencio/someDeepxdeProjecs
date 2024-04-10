# https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.dirichlet.html
# set current directory
import os
os.chdir('06_poisson1d_dirichletBCs')

# 00 import libraries
import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt

# 01 geometry
geom = dde.geometry.Interval(-1, 1)

# 02 PDE residual of the Poisson eq
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

"""
03
First form
- simple python function returning a boolean, to desine the subdomain for the
Dirichlet BC(-1,1)
- True: inside the domain
- False: outside the domain
- dde.utils.isclose: to test whether two floating points are equivalent

def boundary(x, _)
    return dde.utils.isclose(x[0],-1) or dde.utils.isclose(x[0],1)

- To facilitate the implementation of boundary, a boolean on_boundary is used as
the second argument.
- True: the point x is on the entire boundary
- False: otherwise

In a simpler way:
"""
def boundary(x, on_boundary):
    return on_boundary

""" 
# 04
define a function to return u(x) for the points on the boundary. u(x)=0

def func(x):
    return 0

for non-constant values of the function:
"""
def func(x):
    return np.sin(np.pi*x)

# 05 Dirichlet BC:
bc = dde.icbc.DirichletBC(geom,func, boundary)

""" 
06
defining the PDE problem 
- 16: number of training points inside the domain
- 2: training points on the boundary
- solution=func: reference solution
- 100 residual points for testing
"""
data = dde.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)

# 07 Define the network: 4-depth, 50-width
layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# 08 Build the model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# 09 Train the model
losshistory, train_state = model.train(iterations=10000)
"""
Optional
using callbacks to save the model during training (optional):

checkpointer = dde.callbacks.ModelCheckpoint(
    "model/model.ckpt", verbose=1, save_better_only=True
)
# Image magick is required
movie = dde.callbacks.MovieDumper(
    "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
)
#Train the model for 10k iterations
losshistory, train_state = model.train(
    iterations=10000, callbacks=[checkpointer, movie]
)
"""

# 10 saveplot
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

""" 
11
Optional: Restore the saved model with the smallest training loss
model.restore(f"model/model-{train_state.best_step}.ckpt", verbose=1)
"""

# 12 Plot PDE residual
x = geom.uniform_points(1000, True)
y = model.predict(x, operator=pde)
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.show()