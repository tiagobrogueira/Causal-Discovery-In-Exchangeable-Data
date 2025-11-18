import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_auc_score, roc_curve

def compute_auroc(scores,weights):
    weights = weights[~np.isnan(scores)]  
    scores=scores[~np.isnan(scores)] 
    scores = scores[weights > 0]
    weights = weights[weights > 0]


    #AUROC
    y_true = np.random.choice([0, 1], size=scores.shape)
    y_predicted = scores.copy()
    y_predicted[y_true == 0] *= -1
    auroc = roc_auc_score(y_true,y_predicted,sample_weight=weights)
    fpr, tpr, _ = roc_curve(y_true, y_predicted, sample_weight=weights)
    return auroc,tpr,fpr

def auroc(scores,weights):
    auroc,_,_=compute_auroc(scores,weights)
    return auroc

def plot_auroc(method_results,ax=None,baselines=True):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    for method_name, scores, weights in method_results:
        auroc, tpr, fpr = compute_auroc(scores, weights)
        ax.plot(fpr, tpr, label=f"{method_name} ({auroc*100:.1f})")

    if baselines:
        x = np.linspace(0, 1, 100)
        ax.plot([0,0,1], [0,1,1], "--", color="gray", label="Ideal Method")
        ax.plot(x, x, "--", color="lightgray", label="Random baseline")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("AUROC Curves")
    ax.legend()
    
def plot_auroc_vs(method_results_A, method_results_B,
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
        auroc, tpr, fpr = compute_auroc(scores, weights)
        ax.plot(fpr, tpr, color=cmap_A[idx],
                label=f"{method_name} ({auroc*100:.1f})")

    # --- Plot Group B ---
    for idx, (method_name, scores, weights) in enumerate(method_results_B):
        auroc, tpr, fpr = compute_auroc(scores, weights)
        ax.plot(fpr, tpr, color=cmap_B[idx],
                label=f"{method_name} ({auroc*100:.1f})")

    # Baselines
    if baselines:
        x = np.linspace(0, 1, 100)
        ax.plot([0,0,1], [0,1,1], "--", color="gray", label="Ideal Method")
        ax.plot(x, x, "--", color="lightgray", label="Random baseline")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    
    ax.legend()




