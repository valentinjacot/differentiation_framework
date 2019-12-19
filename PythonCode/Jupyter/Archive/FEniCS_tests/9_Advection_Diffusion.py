from fenics import *
# from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ufl import nabla_grad, nabla_div
from mshr import *

T = 5.0
num_steps = 500
dt = T/num_steps
eps = 0.01
K = 10.0


mesh = Mesh('NS_cylinder/cylinder.xml.gz')

W = VectorFunctionSpace(mesh, 'P', 2)

timeseries_w = TimeSeries('NS_cylinder/velocity_series')

P1 = FiniteElement('P', triangle, 1)
element = MixedElement([P1,P1,P1])
V = FunctionSpace(mesh, element)

# for mixed Function Space we need TestFunctions and not TestFunction
v_1, v_2, v_3 = TestFunctions(V)

w = Function(W)
u = Function(V)
u_n = Function(V)

u_1, u_2, u_3 = split(u)
u_n1, u_n2, u_n3 = split(u_n)

# Define the source Term
f_1 = Expression('pow(x[0] - 0.1,2) + pow(x[1]-0.1,2) < 0.05 * 0.05 ? 0.1 : 0', degree=1)
f_2 = Expression('pow(x[0] - 0.1,2) + pow(x[1]-0.3,2) < 0.05 * 0.05 ? 0.1 : 0', degree=1)
f_3 = Constant(0)

k = Constant(dt)
K = Constant(K)
eps = Constant(eps)

F = ((u_1- u_n1)/k)*v_1*dx + dot(w,grad(u_1))*v_1*dx + eps*dot(grad(u_1), grad(v_1))*dx + K * u_1*u_2*v_1*dx \
    + ((u_2 -u_n2)/K)*v_2*dx + dot(w, grad(u_2))*v_2*dx + eps*dot(grad(u_2), grad(v_2))*dx + K * u_1*u_2*v_2*dx \
    + ((u_3 -u_n3)/K)*v_3*dx + dot(w, grad(u_3))*v_3*dx + eps*dot(grad(u_3), grad(v_3))*dx - K * u_1*u_2*v_3*dx \
    + K*u_3*v_3*dx -f_1*v_1*dx -f_2*v_2*dx -f_3*v_3*dx

vtkfile_u1 = File('AD_system/u_1.pvd')
vtkfile_u2 = File('AD_system/u_2.pvd')
vtkfile_u3 = File('AD_system/u_3.pvd')

t=0
for n in range(num_steps):
    t+=dt
    timeseries_w.retrieve(w.vector(),t)
    solve(F==0, u)
    u_1_, u_2_, u_3_ = u.split()
    vtkfile_u1 << (u_1_,t)
    vtkfile_u2 << (u_2_,t)
    vtkfile_u3 << (u_3_,t)
    u_n.assign(u)

plt.show()
