import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

kernels = [
    1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
    1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
    1.0 * ExpSineSquared(length_scale=1.0,
                         periodicity=3.0,
                         length_scale_bounds=(0.1, 10.0),
                         periodicity_bounds=(1.0, 10.0)),
    ConstantKernel(0.1, (0.01, 10.0)) *
    (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0))**2),
    1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
]


class RECI(object):
    def __init__(self):
        self.gpr0 = GPR(kernel=kernels[0])
        self.gpr1 = GPR(kernel=kernels[0])

    def predict(self, x, y):
        """ Prediction method for pairwise causal inference using the RECI model.

        :param x: Variable 1
        :param y: Variable 2
        :return: (Value : >0 if x->y and <0 if y->x)
        :rtype: float
        """

        self.gpr0.fit(x, y)
        self.gpr1.fit(y, x)

        u = y - self.gpr0.predict(x)
        v = x - self.gpr1.predict(y)
        cxy = np.mean(np.power(u, 2))
        cyx = np.mean(np.power(v, 2))
        score = cyx - cxy
        return score
        # return cxy, cyx, score