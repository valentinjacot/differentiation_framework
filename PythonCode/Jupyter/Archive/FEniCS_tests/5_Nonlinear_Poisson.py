# Warning: from fenics import * will import both ‘sym‘ and
# ‘q‘ from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.

from fenics import *
import numpy as np
import matplotlib.pyplot as plt


def q(u):
    "Return nonlinear coefficients"
    return 1 + u ** 2


import sympy as sym

x, y = sym.symbols('x[0], x[1]')  # usually x,y but here we will use these for the
u = 1 + x + 2 * y
f = -sym.diff(q(u) * sym.diff(u, x), x) - sym.diff(q(u) * sym.diff(u, y), y)
f = sym.simplify(f)
u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)
print('u =', u_code)
print('f =', f_code)

nx = ny = 30
mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

a = 5
T = 2.0  # final time
num_step = 10
dt = T / num_step

u_D = Expression(u_code, degree=1)
f = Expression(f_code, degree=1)


def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, u_D, boundary)

# u_n = project(u_D, V)  # or u_n = interpolate(u_D, V)

u = Function(V)
v = TestFunction(V)

F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx

solve(F == 0, u, bc)

plot(u)
# update previous solution
u_e = interpolate(u_D, V)
error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
print(' error max = %.3g' % error)

plt.show()
