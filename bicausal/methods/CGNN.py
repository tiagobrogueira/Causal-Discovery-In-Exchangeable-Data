import numpy as np
from cdt.causality.pairwise import GNN
from cdt.utils.Settings import SETTINGS

#SETTINGS.default_device = 'cpu'
#SETTINGS.GPU = 0

def max_points():
    try:
        from bicausal.helpers.timers import get_max_points
        return get_max_points("CGNN")
    except ModuleNotFoundError:
        return None

#preciso de diminiuir a complexidade computacional
def cgnn(d, train_epochs=None, nruns=None, test_epochs=None, nh=None):
    """
    Compute causal direction score with CDT's GNN (pairwise).

    Args:
        d: tuple (x, y), each is a numpy array with shape (n_samples, 1)
        device: optional, pytorch device string like "cuda" or "cpu"

    Returns:
        float: causation score. 1 means x → y, -1 means y → x.
    """
    x, y = d

    if x.shape[1] > 1 or y.shape[1] > 1:
        return np.nan

    m = max_points()
    if m is not None and len(x) > m:
        idx = np.random.choice(len(x), m, replace=False)
        x = x[idx]
        y = y[idx]

    gnn_kwargs = {}
    if train_epochs is not None:
        gnn_kwargs["train_epochs"] = train_epochs
    if nruns is not None:
        gnn_kwargs["nruns"] = nruns
    if test_epochs is not None:
        gnn_kwargs["test_epochs"] = test_epochs
    if nh is not None:
        gnn_kwargs["nh"] = nh

    model = GNN(**gnn_kwargs)

    return model.predict_proba((x, y))