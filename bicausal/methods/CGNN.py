import numpy as np
from cdt.causality.pairwise import GNN
from cdt.utils.Settings import SETTINGS

# 1. Explicitly set the default device for CDT to 'cpu'
#SETTINGS.default_device = 'cpu'
#SETTINGS.GPU = 0

# Instantiate the pairwise GNN model
model = GNN()

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("CGNN")
    except ModuleNotFoundError:
        return None

#preciso de diminiuir a complexidade computacional
def cgnn(d):
    """
    Compute causal direction score with CDT's GNN (pairwise).

    Args:
        d: tuple (x, y), each is a numpy array with shape (n_samples, 1)
        device: optional, pytorch device string like "cuda" or "cpu"

    Returns:
        float: causation score. 1 means x → y, -1 means y → x.
    """
    x, y = d

    # Possibly subsample if too many points
    m = max_points()
    if m is not None and len(x) > m:
        idx = np.random.choice(len(x), m, replace=False)
        x = x[idx]
        y = y[idx]

    return model.predict_proba((x, y))