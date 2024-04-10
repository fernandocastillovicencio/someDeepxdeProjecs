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

# ---------------------------------------------------------------------------- #
# -------------------------------- incomplete -------------------------------- #