# https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/ode.2nd.html
# set current directory
import os
os.chdir('05_secondOrderODE')

# 00 import libraries
import deepxde as dde
import numpy as np

# 01 timeDomain
geom = dde.geometry.TimeDomain(0, 0.25)

# 02 Residual ODE
def ode(t,y):
    dy_dt = dde.grad.jacobian(y,t)
    d2y_dt2 = dde.grad.hessian(y,t)
    return d2y_dt2 - 10* dy_dt + 9*y - 5*t

# 03 define the IC: y(0)=-1
ic1 = dde.icbc.IC(geom, lambda x: -1, lambda _, on_initial: on_initial)

""" 
04
The other IC: y'(0)=2
The location of IC is defined by a simple function.
The function shuld be return TRuefor t=0 and False otherwise.
For rounding-off errors, it is often wise to use dd.utils.isclose.
"""
def boundary_1(t,on_boundary):
    return on_boundary and dde.utils.isclose(t[0], 0)

"""
05
Now we defie a function that returns the error of the IC,
difference between derivative and 2
 """
def error_2(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0,j=None) - 2

# 06 IC
ic2 = dde.icbc.OperatorBC(geom, error_2, boundary_1)

# 07 Define the function with the exact solution
def func(t):
    return 50/81 + 5/9*t - 31/81*np.exp(9*t) - 2*np.exp(t)

"""
08
Define the problem:
    16 training residual points sampled inside the domain
    2 training residual points sampled on the boundary
    func: reference solution to compute the error
    500: residual poins for testing the PDE residual
 """
data = dde.data.TimePDE(geom, ode, [ic1, ic2], 16, 2, solution=func, \
                        num_test=500)

"""
09
    Fully connected network of depth 4 (3 hidden layers)
    50-width
"""
layer_size = [1] + [50]*3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size,activation,initializer)

""" 
10
    Build the model
    Optimizer, learning rate, 15k iterations
    weights: ODE loss 0.01, ICs 1
 """
model = dde.Model(data, net)
model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[0.01, 1, 1]
)
losshistory, train_state = model.train(iterations=15000)

# 11 Saveplot
dde.saveplot(losshistory, train_state, issave=True, isplot=True)