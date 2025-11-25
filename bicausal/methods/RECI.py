import numpy as np
from cdt.causality.pairwise import RECI

# Instantiate the RECI model
reci_model = RECI()

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("RECI")
    except ModuleNotFoundError:
        return None

def reci(d):
    x, y = d

    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan

    # Use the RECI model to predict
    return reci_model.predict_proba((x, y))
