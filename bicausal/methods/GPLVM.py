import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from bicausal.methods.source_implementations.gplvm_causal_discovery.train_methods import min_causal_score_gplvm_generalised


class ArgsObject:
    """Minimal configuration object mimicking the original 'args'."""
    def __init__(self, n):
        self.num_inducing = min(50, n)
        self.minibatch_size = min(50, n)
        self.data_start = 0
        self.data_end = n
        self.random_restarts = 50
        self.work_dir = "/tmp"
        self.plot_fit = False
        self.data = "single"
        
#tenho de analisar estes hyperparameters. dar cross check

def gplvm(d):
    """
    Thin wrapper so GPLVM can be called per-sample just like FOM().
    Accepts a tuple (x,y) where x,y are column vectors of shape (N,1).
    Returns: causal score = loss(Y|X) - loss(X|Y)
    """
    x, y = d

    # Reject multidimensional samples
    if x.shape[1] > 1 or y.shape[1] > 1:
        return np.nan



    # Normalize: required.
    x = StandardScaler().fit_transform(x).astype(np.float64)
    y = StandardScaler().fit_transform(y).astype(np.float64)

    # Build args
    args = ArgsObject(len(x))

    # Required: run numbers (single sample)
    run_number = 0
    restart_number = 0

    # Causal direction X→Y
    loss_x, loss_y_x = causal_score_gplvm_generalised(
        args=args,
        x=x,
        y=y,
        run_number=run_number,
        restart_number=restart_number,
        causal=True,
        save_name="single_gplvm_run"
    )

    # Anticausal direction Y→X
    loss_y, loss_x_y = causal_score_gplvm_generalised(
        args=args,
        x=y,
        y=x,
        run_number=run_number,
        restart_number=restart_number,
        causal=False,
        save_name="single_gplvm_run"
    )

    # Return final causal preference:
    # Smaller loss = preferred direction
    # Return positive if X→Y preferred, negative if Y→X preferred
    score = (loss_y_x.numpy() - loss_x_y.numpy())
    return float(score)


def gplvm_generalised(d):
    """
    Thin wrapper so GPLVM can be called per-sample just like FOM().
    Accepts a tuple (x,y) where x,y are column vectors of shape (N,1).
    Returns: causal score = loss(Y|X) - loss(X|Y)
    """
    x, y = d

    # Reject multidimensional samples
    if x.shape[1] > 1 or y.shape[1] > 1:
        return np.nan

    # Downsample using max_points()
    mp = max_points()
    if mp is not None:
        n = min(mp, len(x))
        idx = np.random.choice(len(x), n, replace=False)
        x = x[idx]
        y = y[idx]

    # Normalize
    x = StandardScaler().fit_transform(x).astype(np.float64)
    y = StandardScaler().fit_transform(y).astype(np.float64)

    # Build args
    args = ArgsObject(len(x))

    # Required: run numbers (single sample)
    run_number = 0
    restart_number = 0

    # Causal direction X→Y
    loss_x, loss_y_x = causal_score_gplvm_generalised(
        args=args,
        x=x,
        y=y,
        run_number=run_number,
        restart_number=restart_number,
        causal=True,
        save_name="single_gplvm_run"
    )

    # Anticausal direction Y→X
    loss_y, loss_x_y = causal_score_gplvm_generalised(
        args=args,
        x=y,
        y=x,
        run_number=run_number,
        restart_number=restart_number,
        causal=False,
        save_name="single_gplvm_run"
    )

    # Return final causal preference:
    # Smaller loss = preferred direction
    # Return positive if X→Y preferred, negative if Y→X preferred
    score = (loss_y_x.numpy() - loss_x_y.numpy())
    return float(score)

