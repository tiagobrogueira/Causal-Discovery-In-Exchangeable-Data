import traceback
from bicausal.methods.source_implementations.ROCHE_main.causa import roche as original_roche
import numpy as np

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("ROCHE")
    except ModuleNotFoundError:
        return 500 #easy fix - because it never takes less time than this.
        return None  # or set to None
    
def roche(d):
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

    try:
        score= original_roche.roche(x, y)
    except Exception as e:
        print("Error in roche.roche:")
        traceback.print_exc()
        return np.nan
    return score