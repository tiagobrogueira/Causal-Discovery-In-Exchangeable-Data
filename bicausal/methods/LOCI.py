import numpy as np
from bicausal.methods.source_implementations.loci_main.causa.loci import loci as loci_original

def loci(d):
    x,y=d
    n = min(1230, len(x))
    idx = np.random.choice(len(x), n, replace=False)
    x = x[idx]
    y = y[idx]
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    x = x.flatten()
    y = y.flatten()
    return loci_original(x,y)