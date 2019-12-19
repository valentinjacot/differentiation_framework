from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from ufl import nabla_grad, nabla_div
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import axes3d

L = 1
W = 0.2
mu = 1
rho = 1000
delta = W / L
gamma = 0.4 * delta ** 2
beta = 1.25
lambda_ = beta
g = gamma

mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
V = VectorFunctionSpace(mesh, 'P', 1)

tol = 1E-14


def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol


bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)


def epsilon(u_):
    return 0.5 * (nabla_grad(u_) + nabla_grad(u_).T)


def sigma(u_):
    return lambda_ * nabla_div(u_) * Identity(d) + 2 * mu * epsilon(u_)


u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0, -rho * g))
T = Constant((0, 0, 0))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds

u = Function(V)
solve(a == L, u, bc)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_surface(u)
#
#

s = sigma(u) - (1/3.)* tr(sigma(u)) * Identity(d)
von_Misses = sqrt(3./2*inner(s,s))
V = FunctionSpace(mesh, 'P', 1)
von_Misses = project(von_Misses, V)

u_magnitude = sqrt(dot(u,u))
u_magnitude = project(u_magnitude, V)
print('min/max u:',
      u_magnitude.vector().get_local().min(),
      u_magnitude.vector().get_local().max())
# Save solution to file in VTK format
File('elasticity/displacement.pvd') << u
File('elasticity/von_mises.pvd') << von_Misses
File('elasticity/magnitude.pvd') << u_magnitude

plot(u_magnitude)
