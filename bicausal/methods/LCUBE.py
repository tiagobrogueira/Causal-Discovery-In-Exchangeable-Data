from bicausal.methods.source_implementations.LCube_main.LCube import infer_causal_direction
import numpy as np

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("LCube")
    except ModuleNotFoundError:
        return None  # or set to None

#adapted directly from paper (since it is absent from implementation)
def log_likelihood(m, uj, n, rss):
    """
    Implements Equation (24) from 'Identifying Causal Direction via Dense Functional Classes' (LCUBE paper).

    Parameters
    ----------
    m : int
        Number of spline knots.
    uj : array-like
        Sequence of u_j values (index differences between knots).
    n : int
        Sample size, i.e., len(x).
    rss : float
        Residual sum of squares (RSS).

    Returns
    -------
    L : float
        The MDL log-likelihood (negative description length) for the spline model.
    """
    uj = np.asarray(uj)
    uj=uj[uj>0]
    
    term1 = np.log(m)
    term2 = np.sum(np.log(uj))
    term3 = (m + 4) / 2 * np.log(n)
    term4 = (n / 2) * np.log(rss / n)
    L = term1 + term2 + term3 + term4
    return L

def lcube(d):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    if max_points() is not None:
        n = min(max_points(), len(x))
        idx = np.random.choice(len(x), n, replace=False)
        x = x[idx]
        y = y[idx]
        
    x=x.flatten()
    y=y.flatten()
    direction, strength = infer_causal_direction(x, y)
    return strength