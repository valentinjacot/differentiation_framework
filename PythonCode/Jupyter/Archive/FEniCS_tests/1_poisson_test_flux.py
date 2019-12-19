from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from mshr import *

r=0.1

# Create mesh and define function space
#mesh = UnitSquareMesh(20, 20)  # 8X8 rectangles, each divided in 2 triangle, hence 128 cells, and 81 (9^2) vertices
domain = Circle(Point(0.5,0.5), r)  # center and radius
mesh = generate_mesh(domain,100)

V = FunctionSpace(mesh, 'P', 1)  # P returns Lagrangian polynomials, 1 is the degree of the FE

# Define boundary condition (value)


def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, Constant(0), boundary)

# Define variational problem
uh = TrialFunction(V)  # here it is just defined as an unknown to define a.
vh = TestFunction(V)
# f = Constant(10)
# f = Expression('abs(x[0] - 0.5)<1E-3 && abs(x[1] - 0.5)<1E-3  ? 60000 : 0', degree=2)  # or Expression(’-6’, degree=0)
f = Expression('std::exp(-(pow(x[0] - 0.5, 2) + pow(x[1]-0.5,2))/2 ) * (pow(x[0] - 0.5, 2) + pow(x[1]-0.5,2) - 2)', degree=2)

a = dot(grad(uh), grad(vh)) * dx  # bilinear form
L = -f * vh * dx  # linear form
u_D = Expression('6 * 1/(2*M_PI)*std::log(std::max(pow(pow(x[0]-0.5,2) + pow(x[1]-0.5,2),0.5),0.1))', degree=2)
u_D_grad = Expression('-6 * 1/(2*M_PI)*1/(std::max(pow(pow(x[0]-0.5,2) + pow(x[1]-0.5,2),0.5),0.1))', degree=2)

# Compute solution
uh = Function(V) # here we redefine it for the solver
solve(a == L, uh, bc)

# Plot Solution and Mesh
# p= plot(uh,title= 'FE solution')
# plot(mesh,title= 'FE mesh')
# save solution --> omit
# vtkfile = File('poisson/solution.pvd')
# vtkfile << uh


# Compute error in L2 norm
errorL2_var = errornorm(u_D, uh, 'L2')
# errorL2_var_grad = errornorm(u_D_grad, grad(uh), 'L2')

# Compute maximum error at vertices
vertex_values_u_D_var = u_D.compute_vertex_values(mesh)
vertex_values_u_var = uh.compute_vertex_values(mesh)
errorMax_var = np.max(np.abs(vertex_values_u_D_var - vertex_values_u_var))
print(uh.vector().get_local().max())
# print error

r= r
domain = Circle(Point(0.5,0.5), r)  # center and radius
new_mesh = generate_mesh(domain,100)
Vout = FunctionSpace(new_mesh, 'P',1)
uOut = project(uh, Vout)
n = FacetNormal(new_mesh)
p = plot(uOut)

# line_segment = AutoSubDomain(lambda x: near(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2), pow(r,2)) )
# markers = FacetFunction("size_t", new_mesh)
# markers.set_all(0)
# line_segment.mark(markers, 1)
# dS = dS[markers]
# flux = assemble(<expression for flux>*dS(1))
Vvect = VectorFunctionSpace(new_mesh, 'P',1)

# flux = -grad(uOut)
# val = dot(grad(uOut),n)
# print(assemble(flux))
# plot(flux)

# p = plot(dot(flux,flux))
flux2 = -dot(grad(uOut),n)*ds
print(assemble(flux2))

f_exact = exp(-r**2/2 )*np.pi*2* r**2
print(f_exact)
# print(assemble(uOut*dS))
# print(assemble(uOut*ds))
# print( -r*np.log(r)*10)
# print( -r*np.log(r)*10/assemble(uOut*ds))
      # /(2*pi))
# p= plot(uOut,title= 'FE solution')
# plot(new_mesh)
# plt.colorbar(p)
# plot(flux)
# p = plot(uh)
plt.colorbar(p)
print()
plt.show()