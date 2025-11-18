import numpy as np

def accuracy(scores, weights):
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)]  
    scores = scores[weights > 0]
    weights = weights[weights > 0]

    guesses=np.where(scores > 0, 1, np.where(scores < 0, 0, 0.5))

    #Accuracy
    accuracy = sum(guesses*weights)/sum(weights)
    return accuracy
