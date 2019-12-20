# Differentiation Project
# Continuous or discrete? Comparing models of cellular differentiation
# File: Framework.py
# Author: Valentin Jacot-Descombes and Peter Ashcroft
# E-mail: vjd.jako@gmail.com

# 1
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fenics import *
from mshr import *
from ufl import nabla_grad, nabla_div, grad
from scipy.interpolate import interp2d
import os
import sys

sys.path.append(os.path.realpath('..'))

# 2
nu = 0.1
cDiff = 0.0027
dbar = 1.6925
cS = 0.5
N = 40 * 3  # Dimension of the grid
M = 100  # Dimension of the matrices
xmax = 1
tol = 1e-6
parameters['reorder_dofs_serial'] = False
mesh = UnitSquareMesh(N, N)  # 8X8 rectangles, each divided in 2 triangle, hence 128 cells, and 81 (9^2) vertices
Vhat = FunctionSpace(mesh, 'P', 1)  # P returns Lagrangian polynomials, 1 is the degree of the FE


# 3
def fun(x, y, muX, muY, sigma):
    return gaussian2d(x, y, muX, muY, sigma, sigma)


def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian2d(x, y, muX, muY, sigmaX, sigmaY):
    return 1 / (sigmaX * sigmaY * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * (((x - muX) / sigmaX) ** 2 + ((y - muY) / sigmaY) ** 2))


# N: dimension to, M: dimension from
def interpMat2(N, M, mat):
    xx = np.arange(0, 1.0, 1. / (N + 1))
    yy = np.arange(0, 1.0, 1. / (N + 1))
    x = np.arange(0, 1.0, 1. / M)
    y = np.arange(0, 1.0, 1. / M)
    f = interp2d(x, y, mat)
    mat = f(xx, yy)
    mat = mat.T
    mat = mat.reshape(np.prod(mat.shape))
    return mat


# 4
x = np.arange(0, 1.0, 1. / M)
y = np.arange(0, 1.0, 1. / M)
X, Y = np.meshgrid(x, y)
zs = np.array(-fun(np.ravel(X), np.ravel(Y), 0.2, 0.5, 0.15)) * 2
zs += np.array(-fun(np.ravel(X), np.ravel(Y), 0.5, 0.3, 0.1))
zs += np.array(-fun(np.ravel(X), np.ravel(Y), 0.5, 0.7, 0.1))
zs += np.array(-fun(np.ravel(X), np.ravel(Y), 0.8, 0.3, 0.1)) * 1.3
zs += np.array(-fun(np.ravel(X), np.ravel(Y), 0.8, 0.7, 0.1)) * 1.3

print(zs)
print((zs - min(zs)) / -min(zs))
Z = (zs - max(zs)) / -min(zs)
Z = Z.reshape(X.shape) * 0.2
Z -= 0.05
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)
# plt.matshow(Z, aspect=1, cmap=plt.get_cmap('seismic'))
# plt.title('Z projection')
# plt.colorbar()
# plt.axis('equal')
# plt.show()

zlog = -np.log(np.abs(zs))
zlog = zlog.reshape(X.shape)
# plt.matshow(zlog, aspect=1, cmap=plt.get_cmap('seismic'))
# plt.title('Zlog projection')
# plt.colorbar()
# plt.axis('equal')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, zlog)
zexp = np.exp(-((zs - min(zs)) / -min(zs)))
zexp = zexp.reshape(X.shape)
# plt.matshow(zexp, aspect=1, cmap=plt.get_cmap('seismic'))
# plt.title('zexp projection')
# plt.colorbar()
# plt.axis('equal')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, zexp)

# 5
parameters['reorder_dofs_serial'] = False
mesh = UnitSquareMesh(N, N)  # 8X8 rectangles, each divided in 2 triangle, hence 128 cells, and 81 (9^2) vertices
Vhat = FunctionSpace(mesh, 'P', 1)  # P returns Lagrangian polynomials, 1 is the degree of the FE

# 6
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
x = np.arange(0, 1.0, 1. / M)
y = np.arange(0, 1.0, 1. / M)
X, Y = np.meshgrid(x, y)
ui = np.array(fun(np.ravel(X), np.ravel(Y), 0.2, 0.5, 0.1))
ui = (ui - min(ui)) / max(ui)
Ui = ui.reshape(X.shape)
# Z2 = z2.reshape(X.shape)
# ax.plot_surface(X, Y, Ui)
# ax.plot_surface(X, Y, Z2, linewidth=1)
# ax.plot_wireframe(X, Y, Ui)
# plt.show()

# 6
from Dependencies.functionFactory import *

# from makeFunction import *
factory = functionFactory(Vhat, N, M)
uhmst = zexp.reshape(X.shape)
uhmst[uhmst < 0.4] = 1e-15
cDth = 1. / uhmst.T
cDthFct = factory(cDth)
# factory.plot(cDth, title='Death rate')

zlog = -np.log(np.abs(zs))
zlogFct = factory(zlog)
# factory.plot(zlog, title='Homeostasis landscape')

# Very basic initial conditon
UiFct = factory(0.1 * Ui.T)


# factory.plot(Ui.T, title='initial conditions')


def rct(x_, y_):
    return 1 + 2 ** x_


cRctMat = np.array(rct(np.ravel(X), np.ravel(Y)))
cRctMat = cRctMat.reshape(X.shape)
cRctFct = factory(cRctMat.T)


# factory.plot(cRctMat.T,title='Reaction rate')

def R1(u_):
    uv = u_.vector().get_local()
    cDth = cDthFct.vector().get_local()
    cRct = cRctFct.vector().get_local()
    temp = (cDth * uv)
    temp[temp > dbar] = dbar
    temp = (1 - temp) * cRct
    return temp


def v1():
    return zlogFct


def boundary(x, on_boundary):
    return on_boundary


# Differtiation rate
# Needs to be 1d --> np-ravel
def a(x_, y_):
    return 0.5 * (1 - x_ / max(x_))


# PT/ direction of the differentiation
def c_k_x(x_, y_):
    return x_ ** 2


def c_k_y(x_, y_):
    return 0 * y_


cAdvMat = np.array(a(np.ravel(X), np.ravel(Y)))
cAdvMat = cAdvMat.reshape(X.shape)
cAdvFct = factory(cAdvMat.T)
# factory.plot(cAdvMat.T, title='Advection rate')

cAMat_x = np.array(c_k_x(np.ravel(X), np.ravel(Y)))
cAMat_x = cAMat_x.reshape(X.shape)
cAFct_x = factory(cAMat_x.T)
# factory.plot(cAMat_x.T, title='Differentiation rate x')


cAMat_y = np.array(c_k_y(np.ravel(X), np.ravel(Y)))
cAMat_y = cAMat_y.reshape(Y.shape)
cAFct_y = factory(cAMat_y.T)


# factory.plot(cAMat_y.T, title='Differentiation rate y')


def v2_x(u_):
    #     cAdv = cDthFct.vector().get_local()
    cAdv = cAdvFct.vector().get_local()
    cA = cAFct_x.vector().get_local()
    cRct = cRctFct.vector().get_local()
    return cA * (1 - cAdv) * cRct


def v2_y(u_):
    #     cAdv = cDthFct.vector().get_local()
    cAdv = cAdvFct.vector().get_local()
    cA = cAFct_y.vector().get_local()
    cRct = cRctFct.vector().get_local()
    return cA * (1 - cAdv) * cRct


# 7
zero = Constant(0)
bc = DirichletBC(Vhat, zero, boundary)

u = TrialFunction(Vhat)  # here it is just defined as an unknown to define a.
v = TestFunction(Vhat)
u_n = Function(Vhat)
R = Function(Vhat)
V2Fct_x = Function(Vhat)
V2Fct_y = Function(Vhat)

u_n = interpolate(UiFct, Vhat)  # initial value

# plt.figure()
# p = plot(u_n)
# plt.colorbar(p)
# plt.show()

T = 80.0  # final time
num_step = 200
dt = T / num_step
k = 1. / dt
tol = 1e-6

R.vector()[:] = R1(u_n)
V2Fct_x.vector()[:] = v2_x(u_n)
V2Fct_y.vector()[:] = v2_y(u_n)

F = dot((u - u_n) * k, v) * dx + nu * dot(grad(u), grad(v)) * dx + dot(nabla_grad(v1()) * u, nabla_grad(v)) * dx - \
    dot(R * u, v) * dx - dot(V2Fct_x * u, v.dx(0)) * dx - dot(V2Fct_y * u, v.dx(1)) * dx

u = Function(Vhat)
a = lhs(F)
L = rhs(F)
t = 0

## uncomment to save the solution onto pvd files. Can be used to plot dynamically in paraview
# vtkfile = File('framework2/solution.pvd')
# vtkfile << (u_n, t)

for n in range(num_step):
    R.vector()[:] = R1(u_n)
    V2Fct_x.vector()[:] = v2_x(u_n)
    V2Fct_y.vector()[:] = v2_y(u_n)
    t += dt
    solve(a == L, u, bc)
    u_e = interpolate(u_n, Vhat)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t = %.2f: difference = %.3g' % (t, error))  # relative errror
    if (error < tol):
        break
    #     vtkfile << (u, t)
    u_n.assign(u)

plt.figure()
p = plot(u)
plt.colorbar(p)
plt.show()
