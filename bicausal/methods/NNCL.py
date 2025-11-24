from bicausal.methods.source_implementations.HECI_supplementary_upload import NNCL
import numpy as np

def nncl(d):
    x,y=d

    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan

    x = x.flatten()
    y = y.flatten()

    try: 
        direction, _, eta = NNCL.non_invertible_causal(x,y)
    except Exception:
        return np.nan
        
    if np.isinf(eta[0]):
        return np.nan

    if direction:
        return eta[0]
    else:
        return -eta[0]