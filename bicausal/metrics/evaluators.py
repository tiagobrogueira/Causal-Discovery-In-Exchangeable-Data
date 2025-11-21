import pandas as pd
import numpy as np
import os
from datetime import datetime
from tabulate import tabulate

from bicausal.helpers.processers import process_tuebingen_scores, process_lisbon_scores, process_synthetic_scores
from bicausal.metrics.accuracy import accuracy
from bicausal.metrics.auroc import auroc
from bicausal.metrics.lxcim import lxcim
from bicausal.metrics.audrc import audrc

computations_map = {
    "accuracy": accuracy,
    "AUROC": auroc,
    "LxCIM": lxcim,
    "AUDRC": audrc,
}

metric_order = ["LxCIM", "accuracy", "AUROC", "AUDRC"]

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
    metrics=["LxCIM", "accuracy"],
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
    metrics=["LxCIM", "accuracy"],
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


def evaluate_synthetic(
    datasets,
    metrics=["LxCIM", "accuracy"],
    methods=[],
    scores_path=None,
    results_path="results/results.csv"):
    if datasets=="CE" and scores_path is None:
        scores_path="results/ce_scores.csv"
    elif datasets=="ANLSMN" and scores_path is None:
        scores_path="results/ANLSMN_scores.csv"
    elif datasets=="SIM" and scores_path is None:
        scores_path="results/SIM_scores.csv"
    else:
        raise ValueError(f"Unknown datasets: {datasets}")

    methods_params_list_list, scores_list_list, weights_list, dataset_names = process_synthetic_scores(
        methods=methods,
        scores_path=scores_path        
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



def construct_table(
    methods=[],
    datasets=[],
    readpath="results/results.csv",
    metrics=["LxCIM", "accuracy"],
    include_variations=False,
    writedir="table",
    outdated=None
):
    # =========================================================
    # Load results file
    # =========================================================
    if not os.path.exists(readpath):
        raise FileNotFoundError(f"Could not find results file at: {readpath}")

    res = pd.read_csv(readpath, keep_default_na=False).replace("NA", np.nan)
    res["parameters"] = res["parameters"].fillna("").astype(str)
    res["timestamp"] = pd.to_datetime(res["timestamp"], errors="coerce")

    # =========================================================
    # Filter by outdated timestamp
    # =========================================================
    if outdated is not None:
        res = res[res["timestamp"] > outdated]

    # =========================================================
    # Dataset filtering — expand abbreviations manually
    # =========================================================
    all_datasets_in_file = set(res["dataset"].unique())
    datasets_selected = set()

    if datasets:
        for ds in datasets:
            ds_lower = ds.lower()

            # Handle Tübingen variants
            if ds_lower in ["tuebingen", "tübingen", "tubingen", "t\\ubingen"]:
                for d in all_datasets_in_file:
                    dl = d.lower()
                    if "tüb" in dl or "tub" in dl:
                        datasets_selected.add(d)
                continue

            # SIM*
            if ds_lower == "sim":
                for d in all_datasets_in_file:
                    if d.lower().startswith("sim"):
                        datasets_selected.add(d)
                continue

            # CE*
            if ds_lower == "ce":
                for d in all_datasets_in_file:
                    if d.lower().startswith("ce"):
                        datasets_selected.add(d)
                continue

            if "Lisbon-fields" in [d.lower() for d in datasets]:
                for d in all_datasets_in_file:
                    if d.lower().startswith("lisbon"):
                        datasets_selected.add(d)
                continue

            # AN/LS/MN*
            if ds_lower == "anlsmn":
                for d in all_datasets_in_file:
                    dl = d.lower()
                    if dl.startswith("an") or dl.startswith("ls") or dl.startswith("mn"):
                        datasets_selected.add(d)
                continue

            # Literal dataset name
            if ds in all_datasets_in_file:
                datasets_selected.add(ds)

    else:
        datasets_selected = all_datasets_in_file

    res = res[res["dataset"].isin(datasets_selected)]

    # =========================================================
    # Method filtering
    # =========================================================
    if methods:
        res = res[res["method"].isin(methods)]

    # =========================================================
    # Match metrics (case-insensitive, keep original col names)
    # =========================================================
    matched_metrics = []
    for m in metrics:
        target = m.lower()
        for col in res.columns:
            if col.lower() == target:
                matched_metrics.append(col)
                break

    # =========================================================
    # Build table (convert metrics → percentages)
    # =========================================================
    rows = []
    for (method, params, dataset), sub in res.groupby(["method", "parameters", "dataset"]):
        
        if not include_variations and params != "":
            continue
        row = {
            "method": method,
            "parameters": params,
            "dataset": dataset
        }

        for m in matched_metrics:
            val = sub[m].values[0]
            if pd.isna(val) or val == "":
                row[m] = ""
            else:
                try:
                    val_float = float(val)
                    row[m] = f"{100*val_float:.1f}"
                except (ValueError, TypeError):
                    row[m] = ""



        rows.append(row)

    table_df = pd.DataFrame(rows)

    # =========================================================
    # Remove 'parameters' column if variations not included
    # =========================================================
    if not include_variations and "parameters" in table_df.columns:
        table_df = table_df.drop(columns=["parameters"])

    table_df = table_df.fillna("")


    # =========================================================
    # PRINT — console-friendly using `tabulate`
    # =========================================================
    print("\n" + "="*80)
    print("RESULT TABLE")
    print("="*80)
    print(tabulate(table_df, headers="keys", tablefmt="github", showindex=False))
    print("="*80 + "\n")

    # =========================================================
    # SAVE — LaTeX (with timestamp in filename)
    # =========================================================
    os.makedirs(writedir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    latex_path = os.path.join(writedir, f"table_{timestamp_str}.tex")

    latex_text = table_df.to_latex(index=False, escape=True)

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_text)

    print(f"Table saved to: {latex_path}")

    return table_df