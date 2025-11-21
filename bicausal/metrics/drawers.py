import numpy as np
import matplotlib.pyplot as plt
import csv

from bicausal.metrics.lxcim import plot_lxcim, plot_lxcim_vs
from bicausal.metrics.auroc import plot_auroc, plot_auroc_vs
from bicausal.metrics.audrc import plot_audrc, plot_audrc_vs
from bicausal.metrics.evaluators import metric_order
from bicausal.helpers.processers import process_tuebingen_scores, process_lisbon_scores
from bicausal.helpers.utils import save_imgs



def plot_dataset_curves(
    dataset,
    methods=[],
    metrics=["LxCIM"],
    include_variations=False,
    img_dir="plots",
    scores_path=None
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

    method_results = []
    for (method, params), scores in zip(methods_params, scores_list):
        # Apply variation filter
        if (not include_variations) and (params != ""):
            continue
        label = f"{method} ({params})" if params != "" else f"{method}"
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
    save_imgs(f"{dataset} curves", img_dir)
    
    return fig, axs

def plot_dataset_curves_vs(
    dataset,
    methods_A,                         # required
    methods_B=[],                      # optional, auto-detected if empty
    metrics=["LxCIM"],
    include_variations=False,
    img_dir="plots",
    scores_path=None
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
        label = f"{method} ({params})" if params != "" else f"{method}"
        all_method_results.append((label, scores, weights))

    # === Split into Groups A and B ===
    method_results_A = [mr for mr in all_method_results if method_in_selector(mr[0], methods_A)]

    if len(methods_B) == 0:  # auto-assign B = remaining methods
        method_results_B = [mr for mr in all_method_results if mr not in method_results_A]
    else:
        method_results_B = [mr for mr in all_method_results if method_in_selector(mr[0], methods_B)]

    # Sanity check
    if len(method_results_A) == 0:
        raise ValueError("No method matches provided methods_A selectors.")
    if len(method_results_B) == 0:
        raise ValueError("No method matches B group (auto or provided).")

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
    save_imgs(f"{dataset} curves VS", img_dir)

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
    scores_path=None,
    results_path="results/results.csv"
):

    method_results = []

    if datasets==[]:
        datasets = get_all_datasets(method, csv_path=results_path)

    # Loop over datasets (opposite of original)
    for dataset in datasets:

        # === Load scores for this dataset (original logic preserved) ===
        if dataset in ["Tuebingen", "Tübingen"]:
            if scores_path is None:
                scores_path_tuebingen = "results/tuebingen_scores.csv"
            else:
                scores_path_tuebingen = scores_path
            methods_params, scores_list, weights = process_tuebingen_scores(
                methods=[method],   # filter method directly
                scores_path=scores_path_tuebingen
            )
            dataset_name = "Tübingen"

        elif dataset.startswith("Lisbon"):
            if scores_path is None:
                scores_path_lisbon = "results/lisbon_scores.csv"
            else:
                scores_path_lisbon = scores_path

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
            raise ValueError(f"Unknown dataset: {dataset}")

        # === Collect method results (same logic, inverted labels) ===
        for (m, params), scores in zip(methods_params, scores_list):
            if m != method:
                continue
            if (not include_variations) and (params != ""):
                continue

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

    save_imgs(f"{method} curves", img_dir)
    return fig, axs
