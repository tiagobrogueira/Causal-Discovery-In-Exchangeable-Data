# -*- coding: utf-8 -*-
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp_extras.kernels import HeteroscedasticKernel
from sklearn.cluster import KMeans
import numpy as np


class FOM(object):
    def __init__(self):
        self.n_clusters = 5

    def fit(self, x, y):
        prototypes0 = KMeans(n_clusters=self.n_clusters).fit(x).cluster_centers_
        kernel_hete0 = C(1.0, (1e-8, 1000)) * RBF(1, (0.01, 100.0)) \
            + HeteroscedasticKernel.construct(prototypes0, 1e-3, (1e-8, 50.0),
                                            gamma=5.0, gamma_bounds="fixed")

        prototypes1 = KMeans(n_clusters=self.n_clusters).fit(y).cluster_centers_
        kernel_hete1 = C(1.0, (1e-8, 1000)) * RBF(1, (0.01, 100.0)) \
            + HeteroscedasticKernel.construct(prototypes1, 1e-3, (1e-8, 50.0),
                                            gamma=5.0, gamma_bounds="fixed")

        self.gpr0 = GPR(kernel=kernel_hete0, alpha=0)
        self.gpr1 = GPR(kernel=kernel_hete1, alpha=0)

        self.gpr0.fit(x, y)
        self.gpr1.fit(y, x)

    def predict(self, x, y):
        """ Prediction method for pairwise causal inference using the FOM model.

        :param x: Variable 1
        :param y: Variable 2
        :return: (Value : >0 if x->y and <0 if y->x)
        :rtype: float
        """

        self.fit(x, y)

        y_mean, _ = self.gpr0.predict(x, return_std=True)
        x_mean, _ = self.gpr1.predict(y, return_std=True)
        u = y - y_mean
        v = x - x_mean
        cxy = np.mean(u**4)
        cyx = np.mean(v**4)
        score = cyx - cxy
        return score
        # return cxy, cyx, score