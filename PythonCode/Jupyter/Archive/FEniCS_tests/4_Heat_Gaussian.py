from fenics import *
import numpy as np
import matplotlib.pyplot as plt

nx = ny = 30
mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

a = 5
T = 2.0  # final time
num_step = 10
dt = T / num_step

u_0 = Expression('exp(-a*x[0]*x[0] - a* x[1]*x[1]) ', degree=2, a=a)


def boundary(x, on_boundary):
    return on_boundary


zero = Constant(0)
bc = DirichletBC(V, zero, boundary)

u_n = project(u_0, V)  # or u_n = interpolate(u_D, V)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * f) * v * dx
a, L = lhs(F), rhs(F)
vtkfile = File('heat_gaussian/solution.pvd')

u = Function(V)
t = 0
for n in range(num_step):
    # Update current time
    t += dt
    # solve variational problem
    solve(a == L, u, bc)

    vtkfile << (u, t)
    plot(u)
    # update previous solution
    u_e = interpolate(u_0, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t= %.2f: error = %.3g' % (t, error))

    u_n.assign(u)
# plot(mesh)
plt.show()
#The plot can be viewed in Paraview as a video