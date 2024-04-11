# https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.Lshape.html
# set current directory
import os
os.chdir('09_poisson2d_LShaped')

# 00 libraries
import deepxde as dde

# 01 define the computational geometry
geom = dde.geometry.Polygon([[0,0], [1,0], [1,-1], [-1,-1], [-1,1], [0,1]])

# 02 express the residual PDE
def pde(x,y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx- dy_yy - 1

# 03 define a Dirichlet BC
def boundary(_, on_boundary):
    return on_boundary

# 04 define the BC
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

# 05 Define the problem
data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_test=1500)

# 06 define the network
net = dde.nn.FNN([2] + [50]*4 + [1], "tanh", "Glorot uniform")

# 07 choose the optimizer and the learning rate 
model = dde.Model(data, net)
model.compile("adam",lr=50000)

# 08 first training with Adam optimizer
model.train(iterations=50000)

# 09 train again using L-BFGS
model.compile("L-BFGS")
losshistory, train_state = model.train()

# 10 save plot
dde.saveplot(losshistory, train_state, issave=True, isplot=True)