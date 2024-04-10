# 01 import libraries
import deepxde as dde

# 02 import dataset
fname_train = "/home/fernando/workspace/phd/deepxde/datasets/02/dataset.train"
fname_test = "/home/fernando/workspace/phd/deepxde/datasets/02/dataset.test"

# 03 define both datasets and standarize it in an appropriate form
data = dde.data.DataSet(
    fname_train=fname_train,
    fname_test=fname_test,
    col_x=(0,),
    col_y=(1,),
    standardize=True,
)

# 04 specific of the model
layer_size = [1]+ [50]*3 + [1]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

# 05 optimizer and LR
model = dde.Model(data,net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=50000)

# 06 plot
dde.saveplot(losshistory,train_state, issave=True, isplot=True)