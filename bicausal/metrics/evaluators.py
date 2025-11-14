from importlib_metadata import metadata
import pandas as pd
import numpy as np
import os
from datetime import datetime

from bicausal.metrics.accuracy import compute_accuracy
from bicausal.metrics.auroc import compute_auroc
from bicausal.metrics.alameda import compute_alameda
from bicausal.metrics.audrc import compute_audrc
from bicausal.benchmarks.Lisbon.lisbon_utils import load_lisbon_metadata

computations_map = {
    "Accuracy": compute_accuracy,
    "AUROC": compute_auroc,
    "Alameda": compute_alameda,
    "AUDRC": compute_audrc,
}

metric_order = ["Alameda", "Accuracy", "AUROC", "AUDRC"]

def evaluate_and_save(
    dataset_name,
    methods_params,
    metrics,
    scores,
    weights,
    results_path="results/results.csv"
):  
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
    metrics=["AUROC", "Alameda"],
    methods=[],
    scores_path="results/tuebingen_scores.csv",
    results_path="results/results.csv"
):
    df = pd.read_csv(scores_path, keep_default_na=False)
    df = df.replace("NA", np.nan)
    df["parameters"] = df["parameters"].fillna("").astype(str)

    # --- Filter methods if required ---
    if methods:
        df = df[df["method"].isin(methods)]

    all_pairs = sorted(df["Pair"].unique())
    pair_weights = (
        df[["Pair", "weight"]]
        .drop_duplicates(subset=["Pair"])
        .set_index("Pair")["weight"]
        .astype(float)
    )

    weights = pair_weights.loc[all_pairs].values

    # --- Group by method and parameters ---
    grouped = df.groupby(["method", "parameters"], dropna=False)

    method_param_list = []
    scores_list = []
    for (method, params), subdf in grouped:
        # Extract scores indexed by Pair
        scores_by_pair = (
            subdf[["Pair", "score"]]
            .set_index("Pair")["score"]
            .astype(float)
        )

        # Check whether this method/param covers *all* pairs
        missing_pairs = [p for p in all_pairs if p not in scores_by_pair.index]

        if missing_pairs:
            print(
                f"Skipping method={method}, params={params!r} "
                f"because missing pairs: {missing_pairs}"
            )
            continue

        # Create score vector matching the ordering of all_pairs
        score_vector = scores_by_pair.loc[all_pairs].values

        method_param_list.append((method, params))
        scores_list.append(score_vector)

    # If nothing is left after filtering
    if not method_param_list:
        print("No valid method/parameter combinations to evaluate.")
        return None

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
    metrics=["AUROC", "Alameda"],
    methods=[],
    scores_path="results/lisbon_scores.csv",
    results_path="results/results.csv",
    fields=True
):
    # --- Load scores ---
    df = pd.read_csv(scores_path, keep_default_na=False)
    df = df.replace("NA", np.nan)
    df["parameters"] = df["parameters"].fillna("").astype(str)

    # --- Filter methods if required ---
    if methods:
        df = df[df["method"].isin(methods)]

    # --- Load metadata ---
    metadata = load_lisbon_metadata(dataset_dir)
    all_fields = sorted(set(info["field"] for info in metadata.values()))

    # --- Define datasets to evaluate ---
    datasets_to_evaluate = ["Lisbon"]
    if fields:
        datasets_to_evaluate += [f"Lisbon - {f}" for f in all_fields]

    results_accumulated = []

    for dataset_name in datasets_to_evaluate:
        if dataset_name == "Lisbon":
            relevant_fields = all_fields
            relevant_files = list(metadata.keys())
        else:
            field = dataset_name.replace("Lisbon - ", "")
            relevant_fields = [field]
            relevant_files = [fname for fname, info in metadata.items() if info["field"] == field]


        # --- Determine weights per dataset ---
        weights = np.array([metadata[fname]["weight"] for fname in relevant_files])

        # --- Group by method/parameters ---
        grouped = df.groupby(["method", "parameters"], dropna=False)
        method_param_list = []
        scores_list = []

        for (method, params), subdf in grouped:
            # Filter for relevant fields
            subdf_fields = subdf[subdf["filename"].isin(relevant_files)]
            
            missing_files = [f for f in relevant_files if f not in subdf_fields["filename"].values]
            if missing_files:
                print(f"⚠️ Skipping method={method}, params={params!r} for {dataset_name} due to missing files: {missing_files}")
                continue

            # Create score vector in correct order
            scores_by_file = subdf_fields.set_index("filename")["score"].astype(float)
            score_vector = np.array([scores_by_file[fname] for fname in relevant_files])
            
            if np.isnan(score_vector).any():
                print(f"⚠️ Skipping method={method}, params={params!r} for {dataset_name} due to NaN values in scores")
                continue

            method_param_list.append((method, params))
            scores_list.append(score_vector)

        if not method_param_list:
            print(f"No valid method/parameter combinations to evaluate for {dataset_name}.")
            continue

        # --- Evaluate and save ---
        res = evaluate_and_save(
            dataset_name=dataset_name,
            methods_params=method_param_list,
            metrics=metrics,
            scores=scores_list,
            weights=weights,
            results_path=results_path
        )
        results_accumulated.append(res)
    
    return results_accumulated[0]