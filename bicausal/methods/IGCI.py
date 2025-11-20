# %%
import numpy as np
import random
seedR = random.Random(42)
seedN = np.random.default_rng()
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import psi
np.seterr(divide='raise')  # Convert divide-by-zero warnings into exception
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# %%
#From cdt: entropy calculation

def eval_entropy(x):
    """Evaluate the entropy of the input variable.

    :param x: input variable 1D
    :return: entropy of x
    """
    if x.ndim>1:
        x=x.flatten()
    hx = 0
    sx = sorted(x)
    for i, j in zip(sx[:-1], sx[1:]):
        delta = j-i
        if bool(delta):
            hx += np.log(np.abs(delta))
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1)
    return hx

def integral_diff(x, y):
    
    
    x=x.flatten()
    y=y.flatten()
    # Reorder x and y in increasing order of x

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Compute x1, x2, y1, and y2
    # x1: all entries except the last one, x2: all entries except the first one
    x1 = x_sorted[:-1]
    x2 = x_sorted[1:]
    y1 = y_sorted[:-1]
    y2 = y_sorted[1:]
    
    # Compute result by looping over the zipped vectors and calling func
    soma=0
    for a, b, c, d in zip(x1, x2, y1, y2):
        if np.abs(a-b)>1e-3 and np.abs(c-d)>1e-3:
            soma=soma+np.log(np.abs((d - c) / (b - a)))
    
    return soma

def integral_approx_estimator(x, y):
    """Integral approximation estimator for causal inference.

    :param x: input variable x 1D
    :param y: input variable y 1D
    :return: Return value of the IGCI model >0 if x->y otherwise if return <0
    """
    return ((integral_diff(x,y) - integral_diff(y,x))/len(x))


# %%
def igci(d,mode="entropy",norm="uniform"):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    if norm=="uniform":
        scaler=MinMaxScaler()
    else:
        scaler=StandardScaler()
    x=scaler.fit_transform(x)
    y=scaler.fit_transform(y)

    if mode=="entropy":
        result= eval_entropy(x)-eval_entropy(y)
    else:
        result= integral_approx_estimator(x,y)
    return result

