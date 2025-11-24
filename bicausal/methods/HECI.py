from bicausal.methods.source_implementations.HECI_supplementary_upload import HECI
import numpy as np

def heci(d):
    x,y=d

    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan

    x = x.flatten()
    y = y.flatten()

    try: 
        _, scoreXtoY, scoreYtoX = HECI.HECI(x,y)
    except Exception:
        return np.nan
        
    return scoreYtoX - scoreXtoY