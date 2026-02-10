import pandas as pd
import numpy as np
import os
from datetime import datetime
from bicausal.benchmarks.Lisbon.lisbon_utils import load_lisbon_metadata
from bicausal.helpers.utils import getTuebingen


def process_tuebingen_scores(methods=[], scores_path="results/tuebingen_scores.csv", continuous=None):
    df = pd.read_csv(scores_path, keep_default_na=False)
    df = df.replace("NA", np.nan)
    df["parameters"] = df["parameters"].fillna("").astype(str)

    # data is likely a list/dict of pairs where each element is [x, y]
    data, initial_weights = getTuebingen()

    # --- Filter pairs based on "continuous" threshold ---
    valid_pair_indices = []
    valid_weights=[]
    
    if continuous is not None:
        for i, (x, y) in enumerate(data):
            pair_idx = i + 1  # Pairs start at 1
            
            # Count unique values for both variables
            unique_x = len(np.unique(x))
            unique_y = len(np.unique(y))
            
            # Keep only if BOTH variables meet the threshold
            if unique_x >= len(x)/continuous and unique_y >= len(y)/continuous:
                valid_pair_indices.append(pair_idx)
                valid_weights.append(initial_weights[i])
        
        # Filter the dataframe to only include these valid pairs
        df = df[df["Pair"].isin(valid_pair_indices)]
        
    print("Valid pairs:", valid_pair_indices)
    print("Percentage of valid weight:", sum(valid_weights)/sum(initial_weights))
    # --- Filter methods if required ---
    if methods:
        df = df[df["method"].isin(methods)]
        
    all_pairs = sorted(df["Pair"].unique())
    
    # If no pairs match the criteria, return empty lists
    if not all_pairs:
        print("No pairs matched the continuity threshold.")
        return [], [], []

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
        scores_by_pair = (
            subdf[["Pair", "score"]]
            .set_index("Pair")["score"]
            .astype(float)
        )

        missing_pairs = [p for p in all_pairs if p not in scores_by_pair.index]

        if missing_pairs:
            print(
                f"Skipping method={method}, params={params!r} "
                f"because missing pairs: {missing_pairs}"
            )
            continue

        score_vector = scores_by_pair.loc[all_pairs].values

        method_param_list.append((method, params))
        scores_list.append(score_vector)
        
    return method_param_list, scores_list, weights

def process_tuebingen_scores2(methods=[], scores_path="results/tuebingen_scores.csv", continuous=None):

    df = pd.read_csv(scores_path, keep_default_na=False)
    df = df.replace("NA", np.nan)
    df["parameters"] = df["parameters"].fillna("").astype(str)

    data,weights=getTuebingen()


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

        grouped = df_sub.groupby(["method", "parameters"], dropna=False)

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
            
            scores_by_pair = sub.set_index("Pair")["score"].astype(float)

            # Build score vector in correct Pair order
            
            score_vector = np.array([scores_by_pair[p] for p in all_pairs])

            method_param_list.append((method, params))
            scores_list.append(score_vector)

        methods_params_list_list.append(method_param_list)
        scores_list_list.append(scores_list)
        weights_list.append(weights)
        dataset_names.append(dataset_name)

    return methods_params_list_list, scores_list_list, weights_list, dataset_names


