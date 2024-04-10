# 00 import libraries
import deepxde as dde
import numpy as np

# 01 computational domain
geom = dde.geometry.TimeDomain(0,10)

# 02 express the ODE system
def ode_system(x,y):
    y1,y2 = y[:,0:1] , y[:,1:] #to mantain the verticality
    # y1,y2 = y[:,0] , y[:,1] #but it works even if it is horizontal
    dy1_dx = dde.grad.jacobian(y,x,i=0)
    dy2_dx = dde.grad.jacobian(y,x,i=1)
    return [ dy1_dx-y2 , dy2_dx-y1 ] 

""" 
03 
define a function that returns True for points inside the sudomain, and 
False for points outside
isclose: compare two floating point values are equivalent
 """
def boundary(x, on_initial):
    return dde.utils.isclose(x[0],0)


""" 
04 
set IC
if t=0 -> True
otherwise -> False
"""
def boundary(_,on_initial):
    return on_initial


# 05 set the IC
ic1= dde.icbc.IC( geom , lambda x:0, boundary , component=0 )
ic2= dde.icbc.IC( geom , lambda x:1, boundary , component=1 )

# 07 define the function func
def func(x):
    return np.hstack( ( np.sin(x), np.cos(x) ) )


""" 
06
define the problem
35: number of training residual points sample inside the domain
2: number of trained points trained on the boundary (left and right)
func: reference solution to compute the error of our solution
100: number of points for testing the ODE residuals
 """
data = dde.data.PDE(geom, ode_system, [ic1,ic2], 35, 2, solution=func,\
                     num_test=100)

# 08 choose the network, depth:4 , width:50
layer_size = [1] + [50]*3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# 09 build a Model and choose the optimizer and LR
model = dde.Model(data,net)
model.compile("adam",lr=0.001, metrics=["l2 relative error"])

# 10 Training with 20k iterations
losshistory, train_state = model.train(iterations=20000)

# 11 save and plot the best trained result and loss history
dde.saveplot(losshistory, train_state, issave=True, isplot=True)