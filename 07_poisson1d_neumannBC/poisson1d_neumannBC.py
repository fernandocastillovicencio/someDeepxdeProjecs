# https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.neumanndirichlet.html
# set current directory
import os
os.chdir('07_poisson1d_neumannBC')

# 00 
import deepxde as dde
from deepxde.backend import tf

# 01 geometry
geom = dde.geometry.Interval(-1, 1)

# 02 pde
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx - 2

# 03 define right boundary
def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

# 04 similary, define the left boundary
def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], -1)

# 05 exact solution
def func(x):
    return (x + 1) ** 2

# 06 left, Dirichlet BC
bc_l = dde.icbc.DirichletBC(geom, func, boundary_l)

# 07 right, Neumann BC
bc_r = dde.icbc.NeumannBC(geom, lambda X: 2*(X+1), boundary_r)

# 08 define the PDE problem
#   16 training points inside
#   2 training points on the boundary
data = dde.data.PDE(geom, pde, [bc_l, bc_r], 16, 2, solution=func, num_test=100)

# 09 choose the network
layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# 10 build the model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# 11 train the model
losshistory, train_state = model.train(iterations=10000)

# 12 saveplot
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
