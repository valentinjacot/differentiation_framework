# 9
from abc import abstractmethod
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt


class Bin:
    def __init__(self, x_, y_, rad, N_, state_):
        self.x = x_
        self.y = y_
        self.radius = rad
        self.N = N_
        self.setState(state_)

    def setState(self, state_):
        if state_ == 'S' or state_ == 's' or state_ == '':
            self.state = SquareState(self.x, self.y, self.radius, self.N)
        elif state_ == 'C' or state_ == 'c':
            self.state = CircleState(self.x, self.y, self.radius, self.N)
        self.getMesh()

    def getMesh(self):
        self.mesh = self.state.getMesh()
        return self.mesh

    def getFunctionSpace(self):
        self.getMesh()
        self.Vbin = FunctionSpace(self.mesh, "Lagrange", 2)
        return self.Vbin

    def integrate(self, u_):
        self.Vbin = self.getFunctionSpace()
        uInt = project(u_, self.Vbin)
        return assemble(uInt * dx)

    def normedIntegral(self, arg_, u_):
        return self.integrate(arg_ * u_) / self.integrate(u_)

    def outflow(self, u_):
        self.getMesh()
        self.Vbin = self.getFunctionSpace()
        n = FacetNormal(self.mesh)
        uOut = project(u_, self.Vbin)
        flux = -dot(grad(uOut), n) * ds
        return assemble(flux)

    def inflow(self, u_):
        self.getMesh()
        self.Vbin = self.getFunctionSpace()
        n = FacetNormal(self.mesh)
        uOut = project(u_, self.Vbin)
        flux = dot(grad(uOut), n) * ds
        return assemble(flux)

    def lowerHalfOutflow(self, u_):
        def boundary_down(x, on_boundary):
            return on_boundary and x[1] <= self.y

        self.getMesh()
        self.Vbin = self.getFunctionSpace()
        nUD = FacetNormal(self.mesh)
        uUD = project(u_, self.Vbin)
        boundaries = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
        dsUD = Measure('ds', domain=self.mesh, subdomain_data=boundaries)

        bcDown = AutoSubDomain(boundary_down)
        bcDown.mark(boundaries, 2)
        fluxL = -dot(grad(uUD), nUD) * dsUD(2)
        return assemble(fluxL)

    def upperHalfOutflow(self, u_):
        def boundary_up(x, on_boundary):
            return on_boundary and x[1] >= self.y

        self.getMesh()
        self.Vbin = self.getFunctionSpace()
        nUD = FacetNormal(self.mesh)
        uUD = project(u_, self.Vbin)
        boundaries = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
        dsUD = Measure('ds', domain=self.mesh, subdomain_data=boundaries)

        bcUp = AutoSubDomain(boundary_up)
        bcUp.mark(boundaries, 1)
        fluxU = -dot(grad(uUD), nUD) * dsUD(1)
        return assemble(fluxU)

    def minValue(self, u_):
        self.Vbin = self.getFunctionSpace()
        uInt = project(u_, self.Vbin)
        return uInt.vector().get_local().min()

    def maxValue(self, u_):
        self.Vbin = self.getFunctionSpace()
        uInt = project(u_, self.Vbin)
        return uInt.vector().get_local().max()


##### State abstract class
class BinState:
    def __init__(self, x_, y_, rad, N_):
        self.x = x_
        self.y = y_
        self.radius = rad
        self.N = N_

    @abstractmethod
    def getMesh(self):
        pass


class SquareState(BinState):
    def __init__(self, x_, y_, rad, N_):
        super().__init__(x_, y_, rad, N_)

    def getMesh(self):
        p1 = Point(self.x - self.radius, self.y - self.radius)
        p2 = Point(self.x + self.radius, self.y + self.radius)
        self.mesh = RectangleMesh(p1, p2, self.N, self.N)
        return self.mesh


class CircleState(BinState):
    def __init__(self, x_, y_, rad, N_):
        super().__init__(x_, y_, rad, N_)

    def getMesh(self):
        domain = Circle(Point(self.x, self.y), self.radius)  # center and radius
        self.mesh = generate_mesh(domain, self.N)
        return self.mesh
