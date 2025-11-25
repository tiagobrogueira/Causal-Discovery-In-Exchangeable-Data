from bicausal.methods.source_implementations.CDCI_main import CDCI
import numpy as np


def cdci(d, variant="CTV"):
    x,y=d
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan

    return CDCI.causal_score(variant, x.flatten(), y.flatten())