import numpy as np
from cdt.causality.pairwise import CDS

# Instantiate the CDS model
cds_model = CDS()

def cds(d):
    x, y = d

    # Possibly subsample if too many points
    if x.shape[1]>1 or y.shape[1]>1:
        return np.nan
    
    # Use the CDS model to predict
    return cds_model.predict_proba((x, y))
