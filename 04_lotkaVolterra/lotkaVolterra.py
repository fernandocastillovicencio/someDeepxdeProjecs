# 00 import libraries
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from deepxde.backend import tf

""" 
01
ub: upper bound
rb: right bound
late, we scale by these fadctors to obtain a gah between 0 and 1
(population vs time)
"""
ub = 200
rb = 200

# 02 define the time domain
geom = dde.geometry.TimeDomain(0.0,1.0)

# 03 express the ODE system
def ode_system(x,y):
    r = y[:,0:1]
    p = y[:,1:2]
    dr_t = dde.grad.jacobian(y,x,i=0)
    dp_t = dde.grad.jacobian(y,x,i=1)

    return [
        dr_t - 1/ub*rb * ( 2*ub*rb - 0.04*ub*ub*r*p ) , 
        dp_t - 1/ub*rb * ( 0.02*ub*ub*r*p - 1.0*ub*p ),
    ]

""" 
04
define the ODE problem
3000 training residual points
2 points on the boundary
3000 point for testing the ODE residual
 """
data = dde.data.PDE(geom, ode_system, [], 3000, 2, num_test=3000)

""" 
05
NN with deep 7, 6 hidden layers, width 64
 """
layer_size= [1] + [64] * 6 + [2]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size,activation,initializer)

""" 
06
Since we expect to have periodic behavion in the PDE. we add a feature layer
with sin(kt).
This forces the prediction to be periodic and therefore more accurate
 """
def input_transform(t):
    return tf.concat(
        (
        t,
        tf.sin(t),
        tf.sin(2*t),
        tf.sin(3*t),
        tf.sin(4*t),
        tf.sin(5*t),
        tf.sin(6*t),
        ),
        axis=1,
    )

""" 
07
Initial conditions:
  r(0)=100/U
  p(0)=15/U
to be hard constraints, so we transform the output
 """
def output_transform(t,y):
   y1 = y[ : , 0:1 ]
   y2 = y[ : , 1:2 ]

   return tf.concat(
       [
        y1 * tf.tanh(t) + 100/ub ,
        y2 * tf.tanh(t) + 15/ub ,
       ],
       axis = 1
   )

# 08 add these layers 
net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)

# 09 Build th model, optimizer, LR, 50k iterations
model = dde.Model(data,net)
model.compile("adam" , lr=0.001)
losshistory, train_state = model.train(iterations=50000)

"""
10
After training with Adam, we continue with L-BFGS to have a ever smaller loss
"""
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory,train_state,issave=True,isplot=True)

# 11 plot: https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/lotka.volterra.html
plt.xlabel("t")
plt.ylabel("population")

t = np.linspace(0, 1, 100)
x_true, y_true = gen_truedata()
plt.plot(t, x_true, color="black", label="x_true")
plt.plot(t, y_true, color="blue", label="y_true")

t = t.reshape(100, 1)
sol_pred = model.predict(t)
x_pred = sol_pred[:, 0:1]
y_pred = sol_pred[:, 1:2]

plt.plot(t, x_pred, color="red", linestyle="dashed", label="x_pred")
plt.plot(t, y_pred, color="orange", linestyle="dashed", label="y_pred")
plt.legend()
plt.show()