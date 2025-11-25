import numpy as np
from cdt.causality.pairwise import RCC

# Instantiate the RCC model
rcc_model = RCC()

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("RCC")
    except ModuleNotFoundError:
        return None

def rcc(d):
    x, y = d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    # Use the RCC model to predict
    return rcc_model.predict_proba((x, y))
