import pandas as pd
import numpy as np
import os
from datetime import datetime

from bicausal.helpers.processers import process_tuebingen_scores, process_lisbon_scores
from bicausal.metrics.accuracy import accuracy
from bicausal.metrics.auroc import auroc
from bicausal.metrics.alameda import alameda
from bicausal.metrics.audrc import audrc

computations_map = {
    "accuracy": accuracy,
    "AUROC": auroc,
    "Alameda": alameda,
    "AUDRC": audrc,
}

metric_order = ["Alameda", "accuracy", "AUROC", "AUDRC"]

def evaluate_and_save(
    dataset_name,
    methods_params,
    metrics,
    scores,
    weights,
    results_path="results/results.csv"
):  
    if not methods_params:
        print(f"No valid method/parameter combinations to evaluate for {dataset_name}.")
        return None

    for m in metrics:
        if m not in computations_map:
            raise ValueError(
                f"Unknown metric '{m}'. Valid options: {list(computations_map.keys())}"
            )

    # --- Load or initialize results file ---
    if os.path.exists(results_path):
        res = pd.read_csv(results_path, keep_default_na=False).replace("NA", np.nan)
        res["parameters"] = res["parameters"].fillna("").astype(str)
    else:
        res = pd.DataFrame(columns=["method", "parameters", "dataset"])

    # --- Ensure all metric columns exist ---
    existing_metrics = [m for m in metric_order if m in res.columns]
    all_metrics = sorted(
        set(existing_metrics + metrics),
        key=lambda m: metric_order.index(m)
    )

    for m in all_metrics:
        if m not in res.columns:
            res[m] = np.nan

    # --- Ensure timestamp column exists ---
    if "timestamp" not in res.columns:
        res["timestamp"] = ""

    # --- Compute metrics per method ---
    for (method, params), scores in zip(methods_params, scores):

        mask = (
            (res["method"] == method)
            & (res["parameters"] == params)
            & (res["dataset"] == dataset_name)
        )

        # Row does not exist → create and set timestamp now
        if not mask.any():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row = {"method": method, "parameters": params, "dataset": dataset_name}
            for m in all_metrics:
                new_row[m] = np.nan
            new_row["timestamp"] = timestamp

            res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)
            mask = (
                (res["method"] == method)
                & (res["parameters"] == params)
                & (res["dataset"] == dataset_name)
            )

        # Row exists → DO NOT update timestamp
        # Only update metrics
        for metric_name in metrics:
            try:
                val = computations_map[metric_name](scores, weights)
            except Exception as e:
                val = np.nan
                print(f"⚠️ Error computing {metric_name} for {method} ({params}): {e}")
            res.loc[mask, metric_name] = val

    # --- Save results ---
    final_cols = ["method", "parameters", "dataset"] + all_metrics + ["timestamp"]
    res = res[final_cols]
    res.to_csv(results_path, index=False, na_rep="NA")

    print(f"✅ Evaluation complete. Results written to {results_path}")
    return res


def evaluate_tuebingen(
    metrics=["Alameda", "accuracy"],
    methods=[],
    scores_path="results/tuebingen_scores.csv",
    results_path="results/results.csv"
):
    method_param_list, scores_list, weights = process_tuebingen_scores(
        methods=methods,
        scores_path=scores_path
    )

    return evaluate_and_save(
        dataset_name="Tübingen",
        methods_params=method_param_list,
        metrics=metrics,
        scores=scores_list,
        weights=weights,
        results_path=results_path
    )




def evaluate_lisbon(
    dataset_dir="benchmarks/Lisbon",
    metrics=["Alameda", "accuracy"],
    methods=[],
    scores_path="results/lisbon_scores.csv",
    results_path="results/results.csv",
    fields=True
):
    methods_params_list_list, scores_list_list, weights_list, dataset_names = process_lisbon_scores(
        methods=methods,
        scores_path=scores_path,
        dataset_dir=dataset_dir,
        fields=fields
    )
    overall_result=None
    for dataset_name, method_param_list, scores_list, weights in zip(dataset_names, methods_params_list_list, scores_list_list, weights_list):

        # --- Evaluate and save ---
        res = evaluate_and_save(
            dataset_name=dataset_name,
            methods_params=method_param_list,
            metrics=metrics,
            scores=scores_list,
            weights=weights,
            results_path=results_path
        )
        if dataset_name=="Lisbon":
            overall_result=res
    
    return overall_result



