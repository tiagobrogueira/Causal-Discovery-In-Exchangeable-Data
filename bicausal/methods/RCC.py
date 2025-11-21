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

    # Possibly subsample if too many points
    m = max_points()
    if m is not None and len(x) > m:
        idx = np.random.choice(len(x), m, replace=False)
        x = x[idx]
        y = y[idx]

    # Use the RCC model to predict
    return rcc_model.predict_proba((x, y))
