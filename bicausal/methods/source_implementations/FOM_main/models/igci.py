#coding=utf-8
""" Information Geometric Causal Inference (IGCI) model from
P. Daniušis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel,
K. Zhang, B. Schölkopf:  Inferring deterministic causal relations.
Proceedings of the 26th Annual Conference on Uncertainty in Artificial  Intelligence (UAI-2010).
http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf

Adapted by Diviyan Kalainathan
"""

from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from scipy.special import psi
import numpy as np

min_max_scale = MinMaxScaler()
standard_scale = StandardScaler()


def eval_entropy(x):
    """ Evaluate the entropy of the input variable

    :param x: input variable 1D
    :return: entropy of x
    """
    hx = 0.
    sx = sorted(x)
    for i, j in zip(sx[:-1], sx[1:]):
        delta = j-i
        if bool(delta):
            hx += np.log(np.abs(delta))
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1)

    return hx


def diff_entropy_estimator(x, y):
    """ Entropy estimator for causal inference

    :param x: input variable x 1D
    :param y: input variable y 1D
    :return: Return value of the IGCI model >0 if x->y otherwise if return <0
    """
    return eval_entropy(y) - eval_entropy(x)


def integral_approx_estimator(x, y):
    """ Integral approximation estimator for causal inference

    :param x: input variable x 1D
    :param y: input variable y 1D
    :return: Return value of the IGCI model >0 if x->y otherwise if return <0
    """
#     a, b = (0., 0.)
#     x = np.array(x)
#     y = np.array(y)
#     idx, idy = (np.argsort(x), np.argsort(y))

#     for x1, x2, y1, y2 in zip(x[[idx]][:-1], x[[idx]][1:], y[[idx]][:-1], y[[idx]][1:]):
#         if x1 != x2 and y1 != y2:
#             a = a + np.log(np.abs((y2 - y1) / (x2 - x1)))

#     for x1, x2, y1, y2 in zip(x[[idy]][:-1], x[[idy]][1:], y[[idy]][:-1], y[[idy]][1:]):
#         if x1 != x2 and y1 != y2:
#             b = b + np.log(np.abs((x2 - x1) / (y2 - y1)))

#     return (a - b)/len(x)
    a = 0.0
    b = 0.0
    n = len(x)
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])

    indx = np.argsort(x)
    indy = np.argsort(y)

    for i in range(n-1):
        x1 = x[indx[i]]
        x2 = x[indx[i+1]]
        y1 = y[indx[i]]
        y2 = y[indx[i+1]]
        if (x2!=x1) and (y2!=y1):
            a += np.log(np.abs((y2-y1)/(x2-x1)))

        x1 = x[indy[i]]
        x2 = x[indy[i+1]]
        y1 = y[indy[i]]
        y2 = y[indy[i+1]]
        if (x2!=x1) and (y2!=y1):
            b += np.log(np.abs((x2-x1)/(y2-y1)))
    f = (a - b)/(n-1)
    
    return f


def gaussian_scale(x):
    """ Standard scale

    :param x: Input variable
    :return: scaled input variable
    """
    return standard_scale.fit_transform(x)


def uniform_scale(x):
    """ Min-Max scale

    :param x: Input variable
    :return: scaled input variable
    """
    return min_max_scale.fit_transform(x)


class IGCI(object):
    """ Information Geometric Causal Inference (IGCI) model from
    P. Daniušis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel,
    K. Zhang, B. Schölkopf:  Inferring deterministic causal relations.
    Proceedings of the 26th Annual Conference on Uncertainty in Artificial  Intelligence (UAI-2010).
    http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf

    """
    def __init__(self):
        """ Initialize the IGCI model """
        super(IGCI, self).__init__()

    def predict_proba(self, a, b, **kwargs):
        """ Evaluate a pair using the IGCI model

        :param a: Input variable 1D
        :param b: Input variable 1D
        :param kwargs: {refMeasure: Scaling method (gaussian, integral or None),
                        estimator: method used to evaluate the pairs (entropy or integral)}
        :return: Return value of the IGCI model >0 if a->b otherwise if return <0
        """
        estimators = {'entropy': diff_entropy_estimator, 'integral': integral_approx_estimator}
        ref_measures = {'gaussian': gaussian_scale, 'uniform': uniform_scale, 'None': lambda x: x}

        ref_measure = ref_measures[kwargs.get('refMeasure', 'gaussian')]
        estimator = estimators[kwargs.get('estimator', 'entropy')]

        a = ref_measure(a)
        b = ref_measure(b)

        return - estimator(a, b)

    def predict(self, x, y):
        return self.predict_proba(x, y, estimator = 'integral', refMeasure = 'uniform')