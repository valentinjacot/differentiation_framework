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

tol = 1E-14

#
# class K(Expression):
#     def set_k(self, k_0, k_1):
#         self.k_0, self.k_1 = k_0, k_1
#
#     # Useful function, but is called at every node, thus slow
#     def eval(self, value, x):
#         if x[1] <= 0.5 + tol:
#             value[0] = self.k_0
#         else:
#             value[0] = self.k_1
# kappa = K(degree=0)
# kappa.set_k(1, 0.01)
kappa2 = Expression('x[1] <= 0.5 + tol ? k_0 : k_1 ', degree=0, tol=tol, k_0=1, k_1=0.01)

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


#
# def boundary(x, on_boundary):
#     return on_boundary and near(x[0],0, tol)
bc = DirichletBC(V, u_D, boundary)


# class Boundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and near(x[0], 0, tol)
# #
# #
# boundary = Boundary()
# bc = DirichletBC(V, Constant(0), boundary)
#
#
# # u_n = project(u_D, V)  # or u_n = interpolate(u_D, V)
#
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.5 + tol


class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.5 - tol


# materials = MeshFunction('size_t', mesh, mesh.topology().dim(),0)
# materials.set_all(999)
# bc_0 = Omega_0()
# bc_1 = Omega_1()
# bc_0.mark(materials, 0)
# bc_1.mark(materials, 1)
# # bcs = [bc_0, bc_1]
# ds = Measure('ds', domain=mesh, subdomain_data=materials)
# bcs = []
# bc1 = DirichletBC(V, Constant(0), materials,0)
# bc2 = DirichletBC(V, Constant(0), materials,0)
#
# bcs.append(bc1)
# bcs.append(bc2)

# class K(Expression):
#     def __init__(self, mat, k_0, k_1, **kwargs):
#         self.materials = mat
#         self.k_0 = k_0
#         self.k_1 = k_1
#
#     def eval_cell(self, val, x, cell):
#         if self.materials[cell.index] == 0:
#             val[0] = self.k_0
#         else:
#             val[0] = self.k_1


# kappa = K(materials, 1, 0.01, degree=0)
# plot(materials)

# subdomain_0 = CompiledSubDomain('x[0] <=  0.5 + tol', tol=tol)
# subdomain_1 = CompiledSubDomain('x[0] >=  0.5 - tol', tol=tol)

# boundary_R = CompiledSubDomain('on_boundary && near(x[0], 1, tol)', tol= 1E-14)
# bcs = [subdomain_0, subdomain_1]


# cppcode = """
# class K : public Expression
# {
# public:
#   void eval(Array<double>& values,
#             const Array<double>& x,
#             const ufc::cell& cell) const
#   {
#     if ((*materials)[cell.index] == 0)
#       values[0] = k_0;
#     else
#       values[0] = k_1;
#   }
#   std::shared_ptr<MeshFunction<std::size_t>> materials;
#   double k_0;
#   double k_1;
# }; """

# kappa = Expression(cppcode=cppcode, degree=0)
# kappa.materials = materials
# kappa.k_0 = 1
# kappa.k_1 = 0.01

left = DirichletBC(V, Constant(1), "near(x[0], -5)")
top = DirichletBC(V, Constant(0), "near(x[1], 3)")
bcs = [left, top]


class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

bottom = Bottom()

boundaries =  MeshFunction('size_t', mesh, mesh.topology().dim(),0)
boundaries.set_all(0)
bottom.mark(boundaries, 1)
ds = Measure("ds")[boundaries]

u = Function(V)
v = TestFunction(V)

F = q(u) * dot(grad(u), grad(v)) * dx - f * v * dx

solve(F == 0, u, bcs)

plot(u)
# update previous solution
u_e = interpolate(u_D, V)
error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
print(' error max = %.3g' % error)

plt.show()
