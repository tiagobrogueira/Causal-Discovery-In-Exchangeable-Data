import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib.pyplot as plt
import matplotlib.cm as cm



#Note: The trapezoid function approximates the function by drawing straight lines between points (linear interpolation).
def compute_lxcim(scores, weights):
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)] 
    scores = scores[weights > 0]
    weights = weights[weights > 0]

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
    
    lxcim=2*np.trapezoid(cum_acc,dr)   
    return lxcim,cum_acc,dr

def lxcim(scores,weights):
    lxcim,_,_=compute_lxcim(scores,weights)
    return lxcim

#Supports plotting inside larger pipeline.
def plot_lxcim(method_results,ax=None,baselines=True):
    """
    method_results: list of tuples in format
        (method_name, scores_vector, weights_vector)

    save_imgs: external function to save figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    for method_name, scores, weights in method_results:
        lxcim, cum_acc, dr = compute_lxcim(scores, weights)
        ax.plot(dr, cum_acc, label=f"{method_name} ({lxcim*100:.1f})")

    if baselines:
        x = np.linspace(0, 1, 100)
        ax.plot(x, x, "--", color="gray", label="Ideal Method")
        ax.plot(x, x / 2, "--", color="lightgray", label="Random baseline")

    ax.set_xlabel("Decision Rate")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("LxCIM Curves")
    ax.legend()


def plot_lxcim_vs(method_results_A, method_results_B, cmap_A=None, cmap_B=None,
                    ax=None, baselines=True):

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # Define color gradients
    if cmap_A is None:
        base_cmap = cm.get_cmap("Greens")
        cmap_A = base_cmap(np.linspace(0.5, 1, len(method_results_A)))
    if cmap_B is None:
        base_cmap = cm.get_cmap("Reds")
        cmap_B = base_cmap(np.linspace(0.5, 1, len(method_results_B)))

    # --- Plot Group A ---
    for idx, (method_name, scores, weights) in enumerate(method_results_A):
        lxcim, cum_acc, dr = compute_lxcim(scores, weights)
        ax.plot(dr, cum_acc, color=cmap_A[idx],
                label=f"{method_name} ({lxcim*100:.1f})")

    # --- Plot Group B ---
    for idx, (method_name, scores, weights) in enumerate(method_results_B):
        lxcim, cum_acc, dr = compute_lxcim(scores, weights)
        ax.plot(dr, cum_acc, color=cmap_B[idx],
                label=f"{method_name} ({lxcim*100:.1f})")

    # Baselines
    if baselines:
        x = np.linspace(0, 1, 100)
        ax.plot(x, x, "--", color="gray", label="Ideal")
        ax.plot(x, x/2, "--", color="lightgray", label="Random")

    ax.set_xlabel("Decision Rate")
    ax.set_ylabel("Cumulative Accuracy")
    
    ax.legend()




