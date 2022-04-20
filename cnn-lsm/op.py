########## 定义了GAC模型分割中会用到的一些函数 ##############


import numpy as np


def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)


#求x的数值梯度
def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


#定义求曲率函数
def curvature(f):
    fy, fx = grad(f)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)
    return div(Nx, Ny)


#定义求散度函数
def div(fx, fy):
    fyy, fyx = grad(fy)
    fxy, fxx = grad(fx)
    return fxx + fyy
