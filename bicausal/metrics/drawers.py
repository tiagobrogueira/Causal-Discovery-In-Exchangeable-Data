import numpy as np
import matplotlib.pyplot as plt
import csv

from bicausal.metrics.lxcim import plot_lxcim, plot_lxcim_vs, lxcim
from bicausal.metrics.auroc import plot_auroc, plot_auroc_vs
from bicausal.metrics.audrc import plot_audrc, plot_audrc_vs, audrc
from bicausal.metrics.evaluators import metric_order
from bicausal.helpers.processers import process_tuebingen_scores, process_lisbon_scores, process_synthetic_scores
from bicausal.helpers.utils import save_imgs



def plot_dataset_curves(
    dataset,
    methods=[],
    metrics=["LxCIM"],
    include_variations=False,
    img_dir="plots",
    scores_path=None,
    show_params=True,
    figure_name=None
):
    #Obtain method results
    if dataset == "Tuebingen" or dataset == "Tübingen":
        if scores_path is None:
            scores_path="results/tuebingen_scores.csv"
        methods_params, scores_list, weights = process_tuebingen_scores(
            methods=methods,
            scores_path=scores_path
        )
        dataset="Tübingen"
    elif dataset.startswith("Lisbon"):
        if scores_path is None:
            scores_path="results/lisbon_scores.csv"
        # dataset assumed to be one of Lisbon datasets
        methods_params_list_list, scores_list_list, weights_list, dataset_names = process_lisbon_scores(
            methods=methods,
            scores_path=scores_path
        )
        # Select the dataset
        if dataset not in dataset_names:
            raise ValueError(f"Dataset '{dataset}' not found inside Lisbon processed datasets.")

        idx = dataset_names.index(dataset)
        methods_params = methods_params_list_list[idx]
        scores_list = scores_list_list[idx]
        weights = weights_list[idx]

    else:
        if dataset.startswith("CE") and original_scores_path is None:
            scores_path="results/ce_scores.csv"
        elif dataset.startswith("SIM") and original_scores_path is None:
            scores_path="results/SIM_scores.csv"
        elif original_scores_path is None:
            scores_path="results/ANLSMN_scores.csv"


        methods_params_list_list, scores_list_list, weights_list, dataset_names = process_synthetic_scores(
                methods=[method],
                scores_path=scores_path
            )
        if dataset not in dataset_names:
            raise ValueError(f"Dataset '{dataset}' not found")

        idx = dataset_names.index(dataset)
        methods_params = methods_params_list_list[idx]
        scores_list = scores_list_list[idx]
        weights = weights_list[idx]
        dataset_name = dataset

    method_results = []
    for (method, params), scores in zip(methods_params, scores_list):
        # Apply variation filter
        if (not include_variations) and (params != ""):
            continue
        if method=="RDMDL" and params != "":
            label = "RDMDL*"
        elif show_params:
            label = f"{method} ({params})" if params != "" else f"{method}"
        else:
            label = f"{method}"
        method_results.append((label, scores, weights))

    # --- Determine plotting layout 
    selected_metrics = [m for m in metric_order if m.lower() in [x.lower() for x in metrics]]
    num_plots = len(selected_metrics)
    fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axs = [axs] 
    for ax, metric in zip(axs, selected_metrics):
        if metric == "LxCIM":
            plot_lxcim(method_results, ax=ax, baselines=True)
        elif metric == "AUROC":
            plot_auroc(method_results, ax=ax)
        elif metric == "AUDRC":
            plot_audrc(method_results, ax=ax)
        ax.set_title(f"{dataset} - {metric}")

    # --- Save final figure 
    if figure_name is None:
        figure_name = f"{dataset} curves"
    save_imgs(figure_name, img_dir)
    
    return fig, axs

def plot_dataset_curves_vs(
    dataset,
    methods_A,                         # required
    methods_B=[],                      # optional, auto-detected if empty
    metrics=["LxCIM"],
    include_variations=False,
    show_params=True,
    img_dir="plots",
    scores_path=None,
    figure_name=None
):
    # === Process dataset scores (IDENTICAL to baseline implementation) ===
    if dataset == "Tuebingen" or dataset == "Tübingen":
        if scores_path is None:
            scores_path = "results/tuebingen_scores.csv"
        methods_params, scores_list, weights = process_tuebingen_scores(
            methods=None,                # load all, filter later
            scores_path=scores_path
        )
        dataset = "Tübingen"

    elif dataset.startswith("Lisbon"):
        if scores_path is None:
            scores_path = "results/lisbon_scores.csv"
        methods_params_list_list, scores_list_list, weights_list, dataset_names = process_lisbon_scores(
            methods=None,
            scores_path=scores_path
        )
        if dataset not in dataset_names:
            raise ValueError(f"Dataset '{dataset}' not found.")
        idx = dataset_names.index(dataset)
        methods_params = methods_params_list_list[idx]
        scores_list = scores_list_list[idx]
        weights = weights_list[idx]

    else:
        if dataset.startswith("CE") and original_scores_path is None:
            scores_path="results/ce_scores.csv"
        elif dataset.startswith("SIM") and original_scores_path is None:
            scores_path="results/SIM_scores.csv"
        elif original_scores_path is None:
            scores_path="results/ANLSMN_scores.csv"


        methods_params_list_list, scores_list_list, weights_list, dataset_names = process_synthetic_scores(
                methods=[method],
                scores_path=scores_path
            )
        if dataset not in dataset_names:
            raise ValueError(f"Dataset '{dataset}' not found")

        idx = dataset_names.index(dataset)
        methods_params = methods_params_list_list[idx]
        scores_list = scores_list_list[idx]
        weights = weights_list[idx]
        dataset_name = dataset

    # === Expand method name selectors across parameter variations ===
    # Helper: returns True if method matches selector string
    def method_in_selector(method_name, selector_list):
        base = method_name.split("(")[0].strip()
        return base in selector_list

    # Build full method_results list (all methods first)
    all_method_results = []
    for (method, params), scores in zip(methods_params, scores_list):
        if (not include_variations) and (params != ""):
            continue
        elif show_params:
            label = f"{method} ({params})" if params != "" else f"{method}"
        else:
            label = f"{method}"
        all_method_results.append((label, scores, weights))

    # === Split into Groups A and B ===
    method_results_A = [mr for mr in all_method_results if method_in_selector(mr[0], methods_A)]

    if len(methods_B) == 0:  # auto-assign B = remaining methods
        method_results_B = [mr for mr in all_method_results if mr[0] not in [m[0] for m in method_results_A]]
    else:
        method_results_B = [mr for mr in all_method_results if method_in_selector(mr[0], methods_B)]

    # Sanity check
    if len(method_results_A) == 0:
        raise ValueError("No method matches provided methods_A selectors.")
    if len(method_results_B) == 0:
        raise ValueError("No method matches B group (auto or provided).")

    method_results_A = ["RDMDL*" if mr[0].startswith("RDMDL (") else mr for mr in method_results_A]
    # === Determine plot layout identical to baseline ===
    selected_metrics = [m for m in metric_order if m.lower() in [x.lower() for x in metrics]]
    num_plots = len(selected_metrics)
    fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axs = [axs]

    # === Plot using *VS* methods ===
    for ax, metric in zip(axs, selected_metrics):
        if metric == "LxCIM":
            plot_lxcim_vs(method_results_A, method_results_B, ax=ax, baselines=True)
        elif metric == "AUROC":
            plot_auroc_vs(method_results_A, method_results_B, ax=ax)
        elif metric == "AUDRC":
            plot_audrc_vs(method_results_A, method_results_B, ax=ax)
        ax.set_title(f"{dataset} - {metric}")

    # === Save figure matching original function ===
    if figure_name is None:
        figure_name = f"{dataset} curves VS"
    save_imgs(figure_name, img_dir)

    return fig, axs

def get_all_datasets(method, csv_path="results/results.csv"):
    datasets = set()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if row["method"] == method:
                dataset_name = row["dataset"].strip()
                if dataset_name != "":
                    datasets.add(dataset_name)

    return sorted(list(datasets))

def plot_method_curves(
    method,
    datasets=[],
    metrics=["LxCIM"],
    include_variations=False,
    img_dir="plots",
    original_scores_path=None,
    results_path="results/results.csv",
    figure_name=None
):

    method_results = []

    if datasets==[]:
        datasets = get_all_datasets(method, csv_path=results_path)

    # Loop over datasets (opposite of original)
    for dataset in datasets:

        # === Load scores for this dataset (original logic preserved) ===
        if dataset in ["Tuebingen", "Tübingen"]:
            if original_scores_path is None:
                scores_path_tuebingen = "results/tuebingen_scores.csv"
            else:
                scores_path_tuebingen = original_scores_path
            methods_params, scores_list, weights = process_tuebingen_scores(
                methods=[method],   # filter method directly
                scores_path=scores_path_tuebingen
            )
            dataset_name = "Tübingen"

        elif dataset.startswith("Lisbon"):
            if original_scores_path is None:
                scores_path_lisbon = "results/lisbon_scores.csv"
            else:
                scores_path_lisbon = original_scores_path

            methods_params_list_list, scores_list_list, weights_list, dataset_names = process_lisbon_scores(
                    methods=[method],
                    scores_path=scores_path_lisbon
                )

            if dataset not in dataset_names:
                raise ValueError(f"Dataset '{dataset}' not found in Lisbon group.")

            idx = dataset_names.index(dataset)
            methods_params = methods_params_list_list[idx]
            scores_list = scores_list_list[idx]
            weights = weights_list[idx]
            dataset_name = dataset

        else:
            if dataset.startswith("CE") and original_scores_path is None:
                scores_path="results/ce_scores.csv"
            elif dataset.startswith("SIM") and original_scores_path is None:
                scores_path="results/SIM_scores.csv"
            elif original_scores_path is None:
                scores_path="results/ANLSMN_scores.csv"


            methods_params_list_list, scores_list_list, weights_list, dataset_names = process_synthetic_scores(
                    methods=[method],
                    scores_path=scores_path
                )
            if dataset not in dataset_names:
                raise ValueError(f"Dataset '{dataset}' not found")

            idx = dataset_names.index(dataset)
            methods_params = methods_params_list_list[idx]
            scores_list = scores_list_list[idx]
            weights = weights_list[idx]
            dataset_name = dataset

        # === Collect method results (same logic, inverted labels) ===
        for (m, params), scores in zip(methods_params, scores_list):
            if m != method:
                continue
            if (not include_variations) and (params != ""):
                continue
            if method=="RDMDL" and params != "":
                label = f"{dataset_name}*"
            else:
                label = f"{dataset_name} ({params})" if params != "" else dataset_name
            method_results.append((label, scores, weights))

    # === Plotting (same as baseline) ===
    selected_metrics = [m for m in metric_order if m.lower() in [x.lower() for x in metrics]]
    num_plots = len(selected_metrics)

    fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axs = [axs]

    for ax, metric in zip(axs, selected_metrics):
        if metric == "LxCIM":
            plot_lxcim(method_results, ax=ax, baselines=True)
        elif metric == "AUROC":
            plot_auroc(method_results, ax=ax)
        elif metric == "AUDRC":
            plot_audrc(method_results, ax=ax)

        ax.set_title(f"{method} - {metric}")

    if figure_name is None:
        figure_name = f"{method} curves"
    save_imgs(figure_name, img_dir)
    return fig, axs


def plot_with_correctness(
    method,
    dataset,
    metric="LxCIM",
    scores_path=None,
    figsize=(6, 6),
    img_dir="plots",
    figure_name=None
):
    """
    Plot a single metric (LxCIM or AUDRC) for a dataset/method,
    with correctness in the background.
    """
    metric = metric.upper()
    if metric not in ("LXCIM", "AUDRC"):
        raise ValueError("metric must be 'LxCIM' or 'AUDRC'.")

    show_baselines = True  # baselines only make sense for single metric

    # --- Load scores ---
    if dataset in ("Tuebingen", "Tübingen"):
        if scores_path is None:
            scores_path = "results/tuebingen_scores.csv"
        methods_params, scores_list, weights = process_tuebingen_scores(
            methods=[method],
            scores_path=scores_path
        )
        dataset = "Tübingen"

    elif dataset.startswith("Lisbon"):
        if scores_path is None:
            scores_path = "results/lisbon_scores.csv"
        (methods_params_list_list,
         scores_list_list,
         weights_list,
         dataset_names) = process_lisbon_scores(
            methods=[method],
            scores_path=scores_path
        )
        if dataset not in dataset_names:
            raise ValueError(f"Dataset '{dataset}' not found.")
        idx = dataset_names.index(dataset)
        methods_params = methods_params_list_list[idx]
        scores_list = scores_list_list[idx]
        weights = weights_list[idx]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Extract single method
    (meth_name, params), scores = methods_params[0], scores_list[0]
    method_label = meth_name if params == "" else f"{meth_name} ({params})"

    # --- Sort, normalize, correctness ---

    weights=weights[~np.isnan(scores)]
    scores=scores[~np.isnan(scores)]
    idx = np.argsort(-np.abs(scores))
    scores = np.array(scores)[idx]
    weights = np.array(weights)[idx]
    weights = weights / np.sum(weights)
    correct = (scores > 0)
    xs = np.cumsum(weights)

    # --- Figure ---
    fig, ax1 = plt.subplots(figsize=figsize)
    step_out = ax1.step(xs, correct, where="pre", color="green",linewidth=0.5)
    correctness_handle = step_out[0] if isinstance(step_out, (list, tuple)) else step_out
    correctness_handle.set_label("Correctness")

    ax1.set_xlabel("Decision Rate")
    ax1.set_ylabel("Correct (1/0)")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    ax2 = ax1.twinx()

    # Plot metric
    if metric == "LXCIM":
        plot_lxcim([(method_label, scores, weights)], ax=ax2, baselines=show_baselines)
        val = lxcim(scores, weights) * 100
    else:
        plot_audrc([(method_label, scores, weights)], ax=ax2)
        val = audrc(scores, weights) * 100

    # Adjust legend
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend([correctness_handle] + handles,
               ["Correctness"] + labels,
               loc="lower right", fontsize=9, frameon=True, facecolor="white", edgecolor="white")

    ax2.set_title(f"{metric} - {dataset}")

    plt.tight_layout()
    if figure_name is None:
        figure_name = f"{metric}-{method}-{dataset}"
    save_imgs(figure_name, img_dir)
    return fig, (ax1, ax2)


def plot_both_with_correctness(
    method,
    dataset,
    scores_path=None,
    figsize=(6, 6),
    img_dir="plots",
    figure_name=None
):
    """
    Plot both LxCIM and AUDRC metrics together for a dataset/method,
    with correctness in the background.
    """
    metrics = ["LXCIM", "AUDRC"]
    show_baselines = False  # don't show baselines when multiple metrics

    # --- Load scores ---
    if dataset in ("Tuebingen", "Tübingen"):
        if scores_path is None:
            scores_path = "results/tuebingen_scores.csv"
        methods_params, scores_list, weights = process_tuebingen_scores(
            methods=[method],
            scores_path=scores_path
        )
        dataset = "Tübingen"

    elif dataset.startswith("Lisbon"):
        if scores_path is None:
            scores_path = "results/lisbon_scores.csv"
        (methods_params_list_list,
         scores_list_list,
         weights_list,
         dataset_names) = process_lisbon_scores(
            methods=[method],
            scores_path=scores_path
        )
        if dataset not in dataset_names:
            raise ValueError(f"Dataset '{dataset}' not found.")
        idx = dataset_names.index(dataset)
        methods_params = methods_params_list_list[idx]
        scores_list = scores_list_list[idx]
        weights = weights_list[idx]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Extract single method
    (meth_name, params), scores = methods_params[0], scores_list[0]
    method_label = meth_name if params == "" else f"{meth_name} ({params})"

    # --- Sort, normalize, correctness ---
    weights=weights[~np.isnan(scores)]
    scores=scores[~np.isnan(scores)]
    idx = np.argsort(-np.abs(scores))
    scores = np.array(scores)[idx]
    weights = np.array(weights)[idx]
    weights = weights / np.sum(weights)
    correct = (scores > 0).astype(float)
    xs = np.cumsum(weights)

    # --- Figure ---
    fig, ax1 = plt.subplots(figsize=figsize)
    step_out = ax1.step(xs, correct, where="pre", color="green",linewidth=0.5)
    correctness_handle = step_out[0] if isinstance(step_out, (list, tuple)) else step_out
    correctness_handle.set_label("Correctness")

    metric_values = []
    metric_handles = []

    for met in metrics:
        if met == "LXCIM":
            plot_lxcim([(method_label, scores, weights)], ax=ax1, baselines=False)
            val = lxcim(scores, weights) * 100
        else:
            plot_audrc([(method_label, scores, weights)], ax=ax1, baselines=False)
            val = audrc(scores, weights) * 100
        metric_values.append(val)
        metric_handles.append(ax1.get_lines()[-1])
    # Legend with metric values
    ax1.set_xlabel("Decision Rate")
    ax1.set_ylabel("Correct (1/0)")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    legend_handles = [correctness_handle] + metric_handles
    legend_labels = ["Correctness"] + [f"{m} ({v:.1f}%)" for m, v in zip(metrics, metric_values)]
    ax1.legend(legend_handles, legend_labels, loc="lower right",
               fontsize=9, frameon=True, facecolor="white", edgecolor="white")

    ax1.set_title(f"{dataset} — {method_label}")

    plt.tight_layout()

    if figure_name is None:
        figure_name = f"both-{method}-{dataset}"
        
    save_imgs(figure_name, img_dir)

    return fig, ax1