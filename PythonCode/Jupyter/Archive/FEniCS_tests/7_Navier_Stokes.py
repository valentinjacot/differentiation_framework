from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from ufl import nabla_grad, nabla_div


T = 10.0
num_steps = 100
dt = T / num_steps
L = 1
H = 1
mu = 1
rho = 1

# mesh = BoxMesh(Point(0, 0, 0), Point(L, H, H), 10, 3, 3)
mesh = UnitSquareMesh(16,16)

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# boundary = 'near(x[0], 0)' # boundary expression example
inflows = 'near(x[0],0)'
outflows = 'near(x[0],1)'
walls = 'near(x[1], 0) || near(x[1],1)'

bcu_noslip = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow = DirichletBC(Q, Constant(8), inflows)
bcp_outflow = DirichletBC(Q, Constant(0), outflows)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]


def epsilon(u_):
    return sym(nabla_grad(u_))


def sigma(u_, p):
    return 2 * mu * epsilon(u_) - p * Identity(len(u))


u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_n = Function(V)
u_1 = Function(V)
p_n = Function(Q)
p_1 = Function(Q)

U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

F1 = rho*dot((u-u_n)/k,v)*dx + rho*dot(dot(u_n, nabla_grad(u_n)),v)*dx + \
     inner(sigma(U,p_n),epsilon(v))*dx + dot(p_n*n,v)*ds -dot(mu*nabla_grad(U)*n, v)*ds - dot(f,v)*dx
a1 = lhs(F1)
L1 =  rhs(F1)

a2 = dot(nabla_grad(p),nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))* dx - (1/k)*div(u_1)*q*dx

a3 = dot(u, v)*dx
L3 = dot(u_1, v)*dx - k*dot(nabla_grad(p_1 - p_n), v)*dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

t = 0
for n in range(num_steps):
    t += dt

    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_1.vector(), b1)

    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_1.vector(), b2)

    b3 = assemble(L3)
    solve(A3, u_1.vector(), b3)

    plot(u_1)

    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector().get_local() - u_1.vector().get_local()).max()
    print('t= %f: error = %.3g' % (t,error))
    print('max u: ', u_1.vector().get_local().max())

    u_n.assign(u_1)
    p_n.assign(p_1)


plt.show()
# line_segment = AutoSubDomain(lambda x: near(x[0], 0.9))
# markers = FacetFunction("size_t", mesh)
# markers.set_all(0)
# line_segment.mark(markers, 1)
# dS = dS[markers]
# flux2 = -dot(grad(u_1),n)*dS(1)
# print(assemble(flux2))
# plt.show()