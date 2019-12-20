from fenics import *
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt


class functionFactory:
    def __init__(self, vhat_, n_, m_):
        self.N = n_
        self.M = m_
        self.Vhat = vhat_

    def __call__(self, mat_):
        fct = Function(self.Vhat)
        matrix = self.interpMat2(self.N, self.M, mat_)
        fct.vector()[:] = matrix
        return fct

    def plot(self, mat_, title=None):
        fct = self(mat_)
        plt.figure()
        p = plot(fct)
        if title is not None:
            plt.title(title)
        plt.colorbar(p)
        plt.show()

    # N: dimension to, M: dimension from
    @staticmethod
    def interpMat2(n, m, mat):
        xx = np.arange(0, 1.0, 1. / (n + 1))
        yy = np.arange(0, 1.0, 1. / (n + 1))
        x = np.arange(0, 1.0, 1. / m)
        y = np.arange(0, 1.0, 1. / m)
        f = interp2d(x, y, mat)
        mat = f(xx, yy)
        mat = mat.T
        mat = mat.reshape(np.prod(mat.shape))
        return mat
