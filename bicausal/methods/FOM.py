from bicausal.methods.source_implementations.FOM_main.models import FOM
import numpy as np


def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("FOM")
    except ModuleNotFoundError:
        return None  # or set to None


def fom(d):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    if max_points() is not None:
        n = min(max_points(), len(x))
        print("N:",n)
        idx = np.random.choice(len(x), n, replace=False)
        x = x[idx]
        y = y[idx]

    model=FOM()

    return model.predict(x, y)