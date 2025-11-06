import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
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

def compute_knots(n):
    return max(1, n // 40)

def optimal_spline_fit(x, y):
    """
      Fits cubic splines with varying number of knots and selects the best model
      by minimizing the negative log-likelihood.
  
      Returns:
          min_L (float): Minimum negative log-likelihood
          best_m (int): Optimal number of intervals (knots + 1)
    """
    x, y = x[np.argsort(x)], y[np.argsort(x)]
    best_rss, best_m, L_list = float("inf"), 0, []
    num_knots = compute_knots(len(x))

    for m in range(1, num_knots + 1):
        knots = np.linspace(x.min(), x.max(), m + 1)[1:-1]
        try:
            spline = LSQUnivariateSpline(x, y, t=knots, k=3)
            rss = np.sum((y - spline(x))**2)
            if rss < best_rss:
                best_rss, best_m = rss, m

            uj = np.histogram(x, bins=np.r_[[x.min()], knots, [x.max()]])[0]
            L_list.append(log_likelihood(m, uj, len(x), rss))
        except:
            return np.nan, 0
            #break adapted from paper to tackle unfitted problems.
    return min(L_list), best_m

def compute_delta_x_to_y(x, y):
    #L_x, L_y = compute_L(x, y)
    L_opt, best_m = optimal_spline_fit(x, y)
    return L_opt, best_m

def infer_causal_direction(x,y):
    """
      Infers the causal direction between two variables using a spline-based
      minimum description length principle.
  
      Returns:
          direction (str): "->" if X causes Y, "<-" if Y causes X, "undecided" otherwise
          strength (float): Difference in model scores (delta_YX - delta_XY)
    """
    delta_xy, m_x = compute_delta_x_to_y(x, y)
    delta_yx, m_y = compute_delta_x_to_y(y, x)
    
    #extra condition for np.nan scenario
    if delta_xy==np.nan or delta_yx==np.nan:
        return "undecided",np.nan

    if delta_xy < delta_yx:
        direction = "->"  # X causes Y
    elif delta_yx < delta_xy:
        direction = "<-"  # Y causes X
    else:
        direction = "undecided"

    strength = delta_yx - delta_xy  
    return direction, strength


