from bicausal.metrics.accuracy import compute_accuracy
from bicausal.metrics.auroc import compute_auroc
from bicausal.metrics.alameda import compute_alameda
from bicausal.metrics.audrc import compute_audrc
import pandas as pd
import numpy as np
import os

computations_map = {
    "Accuracy": compute_accuracy,
    "AUROC": compute_auroc,
    "Alameda": compute_alameda,
    "AUDRC": compute_audrc,
}

def evaluate_tuebingen(
    metrics=["AUROC", "Alameda"],
    methods=[],
    scores_path="results/tuebingen_scores.csv",
    results_path="results/results.csv"
):
    # --- Load scores ---
    df = pd.read_csv(scores_path)
    df = df.replace("NA", np.nan)
    df["parameters"] = df["parameters"].fillna("").astype(str)

    # Filter methods if specified
    if methods:
        df = df[df["method"].isin(methods)]

    if df.empty:
        raise ValueError("No methods to evaluate — check 'methods' argument or input CSV.")

    grouped = df.groupby(["method", "parameters"], dropna=False)

    # --- Validate metric names ---
    metric_order = ["Alameda", "Accuracy", "AUROC", "AUDRC"]
    for m in metrics:
        if m not in computations_map:
            raise ValueError(f"Unknown metric '{m}'. Valid options: {list(computations_map.keys())}")

    # --- Load or initialize results file ---
    if os.path.exists(results_path):
        res = pd.read_csv(results_path).replace("NA", np.nan)
        res["parameters"] = res["parameters"].fillna("").astype(str)
        if "dataset" not in res.columns:
            res.insert(2, "dataset", "Tübingen")
    else:
        res = pd.DataFrame(columns=["method", "parameters", "dataset"])

    # --- Ensure all necessary columns exist ---
    existing_metrics = [m for m in metric_order if m in res.columns]
    all_metrics = sorted(set(existing_metrics + metrics), key=lambda x: metric_order.index(x))
    for m in all_metrics:
        if m not in res.columns:
            res[m] = np.nan

    # --- Compute metrics ---
    for (method, params), subdf in grouped:
        scores = subdf["score"].astype(float).values
        weights = subdf["weight"].astype(float).values

        mask = (res["method"] == method) & (res["parameters"] == params)
        if not mask.any():
            # Add new row with dataset field and all metric columns
            new_row = {"method": method, "parameters": params, "dataset": "Tübingen"}
            for m in all_metrics:
                new_row[m] = np.nan
            res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)
            mask = (res["method"] == method) & (res["parameters"] == params)

        # Compute and overwrite only requested metrics
        for metric_name in metrics:
            try:
                val = computations_map[metric_name](scores, weights)
            except Exception as e:
                val = np.nan
                print(f"⚠️ Error computing {metric_name} for {method} ({params}): {e}")
            res.loc[mask, metric_name] = val

    # --- Save with proper column order ---
    final_cols = ["method", "parameters", "dataset"] + all_metrics
    res = res[final_cols]
    res.to_csv(results_path, index=False, na_rep="NA")

    print(f"✅ Evaluation complete. Results written to {results_path}")
    return res
