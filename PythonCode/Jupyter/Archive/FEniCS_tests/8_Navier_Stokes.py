from fenics import *
# from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ufl import nabla_grad, nabla_div
from mshr import *

T = 5.0
num_steps = 5000
dt = T / num_steps

mu = 0.001
rho = 1

channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 64)

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# boundary = 'near(x[0], 0)' # boundary expression example
inflows = 'near(x[0], 0)'
outflows = 'near(x[0], 2.2)'
walls = 'near(x[1], 0) || near(x[1],0.41)'
cylinder = 'on_boundary && x[0] > 0.1 && x[0] < 0.3 && x[1] > 0.1 && x[1] < 0.3'

inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# bcu_noslip = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflows)
bcp_outflow = DirichletBC(Q, Constant(0), outflows)

bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcu = [bcp_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]


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

F1 = rho * dot((u - u_n) / k, v) * dx + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx + \
     inner(sigma(U, p_n), epsilon(v)) * dx + dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) \
     * ds - dot(f, v) * dx
a1 = lhs(F1)
L1 = rhs(F1)

a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / k) * div(u_1) * q * dx

a3 = dot(u, v) * dx
L3 = dot(u_1, v) * dx - k * dot(nabla_grad(p_1 - p_n), v) * dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

xdmfile_u = XDMFFile('NS_cylinder/velocity.xdmf')
xdmfile_p = XDMFFile('NS_cylinder/pressure.xdmf')

timeseries_u = TimeSeries('NS_cylinder/velocity_series')
timeseries_p = TimeSeries('NS_cylinder/pressure_series')

File('NS_cylinder/cylinder.xml.gz') << mesh

progress = dolfin.Progress('Time stepping', num_steps)
set_log_level(0)

t = 0.0
for n in range(num_steps):
    t += dt

    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_1.vector(), b1, 'bicgstab', 'hypre_amg')

    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_1.vector(), b2, 'bicgstab', 'hypre_amg')

    b3 = assemble(L3)
    solve(A3, u_1.vector(), b3, 'cg', 'sor')

    plot(u_1, title='Velocity')
    plot(p_1, title='Pressure')

    xdmfile_u.write(u_1, t)
    xdmfile_p.write(p_1, t)

    timeseries_u.store(u_1.vector(), t)
    timeseries_p.store(p_1.vector(), t)

    # u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    # u_e = interpolate(u_e, V)
    # error = np.abs(u_e.vector().get_local() - u_1.vector().get_local()).max()
    # print('t= %f: error = %.3g' % (t,error))
    print('max u: ', u_1.vector().get_local().max())

    u_n.assign(u_1)
    p_n.assign(p_1)

    progress += 0

plt.show()
