import numpy as np
from bicausal.methods.source_implementations.loci_main.causa.loci import loci as loci_original

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("LCube")
    except ModuleNotFoundError:
        return None  # or set to None
        
def loci(d):
    x,y=d
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    if max_points() is not None:
        n = min(max_points(), len(x))
        idx = np.random.choice(len(x), n, replace=False)
        x = x[idx]
        y = y[idx]
    x = x.flatten()
    y = y.flatten()
    return loci_original(x,y)