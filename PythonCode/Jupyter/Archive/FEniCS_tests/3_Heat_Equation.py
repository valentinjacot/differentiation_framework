from fenics import *
import numpy as np
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(20,20)
V = FunctionSpace(mesh, 'P', 1)

alpha = 1.3
beta = 1.2
T = 2.0  # final time
num_step = 10
dt = T / num_step
u_D = Expression('1 + x[0]*x[0] + alpha * x[1]*x[1] + beta * t', degree=2, alpha=alpha, beta=beta,
                 t=0)  # t is passed as a parameter


def boundary(x, on_boundary):
    return on_boundary

zero = Constant(0)
bc = DirichletBC(V, zero, boundary)

# u is u(n+1) and u_n is u(n)
u_n = project(u_D, V)  # or u_n = interpolate(u_D, V)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2 * alpha)

F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * f) * v * dx
a, L = lhs(F), rhs(F)

u = Function(V)
t = 0
for n in range(num_step):
    # Update current time
    t += dt
    u_D.t = t
    # solve variational problem
    solve(a == L, u, bc)
    p = plot(u)
    # update previous solution
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t= %.2f: error = %.3g' % (t, error))

    u_n.assign(u)

plt.colorbar(p)
plt.show()
