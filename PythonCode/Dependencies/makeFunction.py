from fenics import *
import numpy as np
from scipy.interpolate import interp2d
from deprecated import deprecated


@deprecated(reason="You should use the functionFactory class")
class makeFunction:
    def __init__(self, vhat, n, m, mat):
        self.fct = Function(vhat)
        self.matrix = self.interpMat2(n, m, mat)
        self.fct.vector()[:] = self.matrix

    def get_function(self):
        return self.fct

    # N: dimension to, M: dimension from
    @staticmethod
    def interpMat2(n, m, mat):
        xx = np.arange(0, 1.0, 1 / (n + 1))
        yy = np.arange(0, 1.0, 1 / (n + 1))
        x = np.arange(0, 1.0, 1 / m)
        y = np.arange(0, 1.0, 1 / m)
        f = interp2d(x, y, mat)
        mat = f(xx, yy)
        mat = mat.T
        mat = mat.reshape(np.prod(mat.shape))
        return mat
