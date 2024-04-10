import deepxde as dde
import numpy as np


# 02 simple function
def func(x):
    return x*np.sin(5*x)

# 03 define the computational domain
geom = dde.geometry.Interval(-1,1)

# 04 Define the problem
num_train = 16
num_test = 100
data = dde.data.Function(geom,func, num_train, num_test)

# 05 choose the network, 4-depth (3 hidden layers) and 20-width w/ tanh as the activation funcion, and Glorot uniform initializer
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN( [1] + [20]*3 + [1] , activation , initializer )

# 06 choose the model w/ the Adam optimizer, and the learning rate of 0.001
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])


# 07 train the model (10 000 iterations)
losshistory, train_state = model.train(iterations=10000)

# 08 save the plot and the best trained result
dde.saveplot(losshistory, train_state,issave=True, isplot=True)