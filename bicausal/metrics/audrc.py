import numpy as np

def compute_audrc(scores, weights):
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)]  
    guesses=np.where(scores > 0, 1, np.where(scores < 0, 0, 0.5))

    #AUDRC
    sorted_indices = np.argsort(-np.abs(scores))
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_guesses = guesses[sorted_indices]

    acc=np.cumsum(sorted_guesses*sorted_weights)/np.cumsum(sorted_weights) #inner sum
    audrc=np.cumsum(acc)*np.cumsum(sorted_weights)/sum(sorted_weights) #outer sum
    return audrc