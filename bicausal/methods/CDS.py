import numpy as np
from cdt.causality.pairwise import CDS

# Instantiate the CDS model
cds_model = CDS()

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("CDS")
    except ModuleNotFoundError:
        return None

def cds(d):
    x, y = d

    # Possibly subsample if too many points
    m = max_points()
    if m is not None and len(x) > m:
        idx = np.random.choice(len(x), m, replace=False)
        x = x[idx]
        y = y[idx]

    # Use the CDS model to predict
    return cds_model.predict_proba((x, y))
