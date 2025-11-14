import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auroc(scores,weights):
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)] 

    #AUROC
    y_true = np.random.choice([0, 1], size=scores.shape)
    y_predicted = scores.copy()
    mask = np.random.choice([0, 1], size=scores.shape)
    y_predicted[mask == 0] *= -1
    auroc = roc_auc_score(y_true,y_predicted,sample_weight=weights)
    
    return auroc