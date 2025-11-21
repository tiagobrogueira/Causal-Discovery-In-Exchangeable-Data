import pandas as pd
import numpy as np
import os
from datetime import datetime
from bicausal.benchmarks.Lisbon.lisbon_utils import load_lisbon_metadata

def process_tuebingen_scores(methods=[], scores_path="results/tuebingen_scores.csv"):

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
    return method_param_list, scores_list, weights


def process_lisbon_scores(methods=[], scores_path="results/lisbon_scores.csv", dataset_dir="benchmarks/Lisbon", fields=True):
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

    methods_params_list_list=[]
    scores_list_list=[]
    weights_list=[]
    dataset_names = []

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

        methods_params_list_list.append(method_param_list)
        scores_list_list.append(scores_list)
        weights_list.append(weights)
        dataset_names.append(dataset_name)

    return methods_params_list_list, scores_list_list, weights_list, dataset_names


def process_synthetic_scores(methods=[],
                      scores_path=None):

    if scores_path is None:
        raise ValueError("scores_path must be provided for synthetic scores processing.")

    df = pd.read_csv(scores_path, keep_default_na=False)
    df = df.replace("NA", np.nan)
    df["parameters"] = df["parameters"].fillna("").astype(str)

    if methods:
        df = df[df["method"].isin(methods)]

    all_datasets = sorted(df["dataset"].unique())

    methods_params_list_list = []
    scores_list_list = []
    weights_list     = []
    dataset_names    = []
    
    for dataset_name in all_datasets:

        df_sub = df[df["dataset"] == dataset_name]

        # Identify all pairs belonging to this dataset
        all_pairs = sorted(df_sub["Pair"].unique())

        # Weights come directly from CE_scores
        # (weights per row are already normalized)
        weights = np.array([
            df_sub[df_sub["Pair"] == p]["weight"].values[0]
            for p in all_pairs
        ])

        # Group scores by method / parameters
        grouped = df.groupby(["method", "parameters"], dropna=False)

        method_param_list = []
        scores_list = []

        for (method, params), subdf in grouped:

            # Scores for this method restricted to this dataset
            sub = subdf[subdf["dataset"] == dataset_name]

            # Check missing pairs
            present_pairs = set(sub["Pair"].unique())
            missing_pairs = [p for p in all_pairs if p not in present_pairs]

            if missing_pairs:
                print(f"⚠️ Skipping method={method}, params={params!r} "
                      f"on {dataset_name} due to missing pairs: {missing_pairs}")
                continue

            # Build score vector in correct Pair order
            scores_by_pair = sub.set_index("Pair")["score"].astype(float)
            score_vector = np.array([scores_by_pair[p] for p in all_pairs])

            # Skip if NaNs inside
            if np.isnan(score_vector).any():
                print(f"⚠️ Skipping method={method}, params={params!r} "
                      f"on {dataset_name} due to NaN values")
                continue

            method_param_list.append((method, params))
            scores_list.append(score_vector)

        methods_params_list_list.append(method_param_list)
        scores_list_list.append(scores_list)
        weights_list.append(weights)
        dataset_names.append(dataset_name)

    return methods_params_list_list, scores_list_list, weights_list, dataset_names


