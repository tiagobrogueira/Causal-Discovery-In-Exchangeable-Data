from bicausal.methods.source_implementations.ROCHE-main.causa import ROCHE
import numpy as np

def roche(d):
    x,y=d

    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan

    x = x.flatten()
    y = y.flatten()

    try: 
        return ROCHE(x,y)
    except Exception:
        return np.nan
        