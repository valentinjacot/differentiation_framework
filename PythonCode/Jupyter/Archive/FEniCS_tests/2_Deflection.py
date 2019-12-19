from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from mshr import *

# R
# R0
# Dc = A*R*R/(8*pi*sigma*T)
# D
# sigma
# #Dimension scaling
# xb = x / R
# yb = y / R
# w = D / Dc
# R0b = R0 / R
# # alpha = R*R*A/(2*pi*T*Dc*sigma)
# # beta = R/(sqrt(2) * sigma)
alpha = 4
beta = 8
R0 = 0.6
domain = Circle(Point(0, 0), 1)  # center and radius
mesh = generate_mesh(domain, 64)  # resolution
V = FunctionSpace(mesh, 'P', 2)

p = Expression('4*exp(-pow(beta,2)*(x[0]*x[0] + (pow(x[1] - R0,2) ) ) )', degree=1, beta=beta, R0=R0)
w_D = Constant(0)


# p.beta = 12 can be changed p.R0 = 0

def boundary(x, on_boudary):
    return on_boudary


w = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(w), grad(v)) * dx
L = p * v * dx

bc = DirichletBC(V, w_D, boundary)

w = Function(V)
solve(a == L, w, bc)

p2 = interpolate(p, V)

plot(w, title='Deflection')
plot(p2, title='Load')

plt.show()
#
# vtkfile_w = File('poisson_membrane/deflection.pvd')
# vtkfile_w << w
# vtkfile_p = File('poisson_membrane/load.pvd')
# vtkfile_p << p2
#
# tol = 0.001
# y = np.linspace(-1 + tol, 1 - tol, 101)
# points = [(0, y_) for y_ in y]
# w_line = np.array([w(point) for point in points])
# p_line = np.array([p2(point) for point in points])
# plt.plot(y,50*w_line,'k', linewidth=2)
# plt.plot(y,p_line,'b--', linewidth=2)
# plt.grid(True)
# plt.xlabel('$y$')
# plt.legend(['Deflection ($\\times 50$)', 'Load'], loc='upper left')
# plt.savefig('poisson_membrane/curves.pdf')
# plt.savefig('poisson_membrane/curves.png')
#
# plt.show()