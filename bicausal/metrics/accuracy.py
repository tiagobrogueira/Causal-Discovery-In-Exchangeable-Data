import numpy as np

def compute_accuracy(scores, weights):
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)]  
    guesses=np.where(scores > 0, 1, np.where(scores < 0, 0, 0.5))

    #Accuracy
    accuracy = sum(guesses*weights)/sum(weights)
    return accuracy
