import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

def compute_alameda(scores, weights):
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)] 
    guesses=np.where(scores > 0, 1, np.where(scores < 0, 0, 0.5))

    #Alameda
    sorted_indices = np.argsort(-np.abs(scores))
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_guesses = guesses[sorted_indices]

    cum_acc=np.cumsum(sorted_guesses*sorted_weights)/sum(sorted_weights)
    dr=np.cumsum(sorted_weights)/sum(sorted_weights)

    # Include the origin point (0, 0)
    cum_acc = np.concatenate(([0], cum_acc))
    dr = np.concatenate(([0], dr))
    
    alameda=2*np.trapezoid(cum_acc,dr)   
    return alameda

#Note: The trapezoid function approximates the function by drawing straight lines between points (linear interpolation).