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

    # Possibly subsample if too many points
    m = max_points()
    if m is not None and len(x) > m:
        idx = np.random.choice(len(x), m, replace=False)
        x = x[idx]
        y = y[idx]

    # Use the RECI model to predict
    return reci_model.predict_proba((x, y))
