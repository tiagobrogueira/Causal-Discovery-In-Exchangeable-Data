import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def compute_audrc(scores, weights):
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)]  
    scores = scores[weights > 0]
    weights = weights[weights > 0]

    guesses=np.where(scores > 0, 1, np.where(scores < 0, 0, 0.5))

    #AUDRC
    sorted_indices = np.argsort(-np.abs(scores))
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_guesses = guesses[sorted_indices]
    
    acc=np.cumsum(sorted_guesses*sorted_weights)/np.cumsum(sorted_weights) #inner sum
    dr=np.cumsum(sorted_weights)/sum(sorted_weights)
    audrc=np.inner(acc, sorted_weights)/sum(sorted_weights) #outer sum
    return audrc, acc, dr

def audrc(scores,weights):
    audrc,_,_=compute_audrc(scores,weights)
    return audrc

def plot_audrc(method_results,ax=None,baselines=True):
    """
    method_results: list of tuples in format
        (method_name, scores_vector, weights_vector)

    save_imgs: external function to save figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    for method_name, scores, weights in method_results:
        audrc, acc, dr = compute_audrc(scores, weights)
        ax.plot(dr, acc, label=f"{method_name} ({audrc*100:.1f})")

    if baselines:
        x = np.linspace(0, 1, 100)
        ax.plot(x, np.ones_like(x), "--", color="gray", label="Ideal Method")      # Constant = 1
        ax.plot(x, 0.5 * np.ones_like(x), "--",color="lightgray", label="Random baseline")  # Constant = 0.5

    ax.set_xlabel("Decision Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("AUDRC Curves")
    ax.legend()


def plot_audrc_vs(method_results_A, method_results_B,
                    ax=None, baselines=True, cmap_A=None, cmap_B=None):

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
        audrc, acc, dr = compute_audrc(scores, weights)
        ax.plot(dr, acc, color=cmap_A[idx],
                label=f"{method_name} ({audrc*100:.1f})")

    # --- Plot Group B ---
    for idx, (method_name, scores, weights) in enumerate(method_results_B):
        audrc, acc, dr = compute_audrc(scores, weights)
        ax.plot(dr, acc, color=cmap_B[idx],
                label=f"{method_name} ({audrc*100:.1f})")

    # Baselines
    if baselines:
        x = np.linspace(0, 1, 100)
        ax.plot(x, np.ones_like(x), "--", color="gray", label="Ideal Method")      # Constant = 1
        ax.plot(x, 0.5 * np.ones_like(x), "--",color="lightgray", label="Random baseline")  # Constant = 0.5

    ax.set_xlabel("Decision Rate")
    ax.set_ylabel("Accuracy")
    
    ax.legend()