from cdt.causality.pairwise import ANM
import numpy as np

model = ANM()

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("ANM")
    except ModuleNotFoundError:
        return None  # or set to None

def anm(d):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    if max_points() is not None:
        n = min(max_points(), len(x))
        idx = np.random.choice(len(x), n, replace=False)
        x = x[idx]
        y = y[idx]

    return model.predict_proba((x,y))