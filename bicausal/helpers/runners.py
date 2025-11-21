import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from bicausal.helpers.utils import getTuebingen, serialize_params, normalize_str
from bicausal.helpers.namemap import get_method_name

def run_tuebingen(func, read_dir="benchmarks/Tuebingen", write_dir="results", overwrite=False, *args, **kwargs):
    """
    Runs func on the Tübingen dataset and saves results to a shared CSV file.
    Columns: ['method', 'parameters', 'Pair', 'score', 'weight', 'timestamp']
    """
    data, weights = getTuebingen(read_dir)
    os.makedirs(write_dir, exist_ok=True)
    path = os.path.join(write_dir, "tuebingen_scores.csv")

    method_name = get_method_name(func)
    parameters = serialize_params(args, kwargs)

    # Load or create dataframe
    if os.path.exists(path):
        df_existing = pd.read_csv(path, keep_default_na=False)
        df_existing["parameters"] = df_existing["parameters"].map(normalize_str)
    else:
        df_existing = pd.DataFrame(columns=["method", "parameters", "Pair", "score", "weight", "timestamp"])

    results = []
    for i, ((x, y), w) in enumerate(zip(data, weights)):
        exists = (
            (df_existing["method"] == method_name)
            & (df_existing["parameters"] == parameters)
            & (df_existing["Pair"] == i + 1)
        ).any()

        if exists and not overwrite:
            print(f"⏩ Skipping Pair {i + 1} for {method_name} (already computed with same parameters)")
            continue

        try:
            score = func([x, y], *args, **kwargs)
        except Exception as e:
            print(f"⚠️ Skipping Pair {i+1} due to error: {e}")
            continue

        if np.isnan(score):
            score = "NA"

        results.append({
            "method": method_name,
            "parameters": parameters,
            "Pair": i + 1,
            "score": score,
            "weight": w,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    if not results:
        print("❌ No results to save.")
        return

    df_new = pd.DataFrame(results)

    if overwrite:
        df_existing = df_existing[
            ~(
                (df_existing["method"] == method_name)
                & (df_existing["parameters"] == parameters)
                & (df_existing["Pair"].isin(df_new["Pair"]))
            )
        ]

    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_csv(path, index=False)
    print(f"✅ Saved Tuebingen results to {path}")
    return path


def run_lisbon(func, read_dir="benchmarks/Lisbon/data", write_dir="results", overwrite=False, *args, **kwargs):
    """
    Applies func(x, y, *args, **kwargs) to all .txt files under read_dir recursively.
    Saves results to a shared CSV file:
    Columns: ['method', 'parameters', 'filename', 'score', 'timestamp']
    """
    os.makedirs(write_dir, exist_ok=True)
    output_path = os.path.join(write_dir, "lisbon_scores.csv")
    parameters = serialize_params(args, kwargs)
    method_name = get_method_name(func)

    if os.path.exists(output_path):
        df_results = pd.read_csv(output_path)
        df_results["parameters"] = df_results["parameters"].map(normalize_str)
    else:
        df_results = pd.DataFrame(columns=["method", "parameters", "filename", "score", "timestamp"])

    for root, _, files in os.walk(read_dir):
        txt_files = [f for f in files if f.lower().endswith(".txt")]
        if not txt_files:
            continue

        print(f"🔍 Processing source: {os.path.basename(root)} ({len(txt_files)} files)")
        new_rows = []

        for fname in txt_files:
            exists = (
                (df_results["method"] == method_name)
                & (df_results["parameters"] == parameters)
                & (df_results["filename"] == fname)
            ).any()

            if exists and not overwrite:
                print(f"⏩ Skipping {fname} for {method_name} (already computed with same parameters)")
                continue

            path = os.path.join(root, fname)
            try:
                df = pd.read_csv(path, sep=None, engine="python", header=None)
                x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
                score = func([x.reshape(-1, 1), y.reshape(-1, 1)], *args, **kwargs)
                new_rows.append({
                    "method": method_name,
                    "parameters": parameters,
                    "filename": fname,
                    "score": score,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                print(f"⚠️ Skipping {fname} due to error: {e}")
                continue

        if new_rows:
            df_new = pd.DataFrame(new_rows)
            if overwrite:
                df_results = df_results[
                    ~(
                        (df_results["method"] == method_name)
                        & (df_results["parameters"] == parameters)
                        & (df_results["filename"].isin(df_new["filename"]))
                    )
                ]
            df_results = pd.concat([df_results, df_new], ignore_index=True)
            df_results.to_csv(output_path, index=False)
            print(f"💾 Saved {len(df_new)} results from {os.path.basename(root)}.")

    print(f"✅ All Lisbon results saved to {output_path}")


def run_ce(func, datasets=None, read_dir="benchmarks/synthetic/CE-Guyon",write_dir="results", overwrite=False, *args, **kwargs):
    # The CE datasets
    ALL_CE = ["CE-Cha", "CE-Gauss", "CE-Multi", "CE-Net"]

    # Dataset selection
    if datasets is None or len(datasets) == 0:
        datasets = ALL_CE
    else:
        for d in datasets:
            if d not in ALL_CE:
                raise ValueError(f"Invalid CE dataset: {d}. Must be one of {ALL_CE}")

    # Prepare output
    os.makedirs(write_dir, exist_ok=True)
    path = os.path.join(write_dir, "CE_scores.csv")

    method_name = get_method_name(func)
    parameters = serialize_params(args, kwargs)

    # Load or create previous results
    if os.path.exists(path):
        df_existing = pd.read_csv(path, keep_default_na=False)
        df_existing["parameters"] = df_existing["parameters"].map(normalize_str)
    else:
        df_existing = pd.DataFrame(columns=[
            "method", "parameters", "dataset", "Pair",
            "score", "weight", "timestamp"
        ])

    # ----------------------------------------------------
    # Main loop over datasets
    # ----------------------------------------------------
    for dataset in datasets:
        print(f"\n🚀 Running CE dataset: {dataset}")

        # -------------------------------
        # Inline CE data loading (old getOld)
        # -------------------------------
        pairs_file   = f"{read_dir}/{dataset}_pairs.csv"
        targets_file = f"{read_dir}/{dataset}_targets.csv"

        if not os.path.isfile(pairs_file):
            raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
        if not os.path.isfile(targets_file):
            raise FileNotFoundError(f"Targets file not found: {targets_file}")

        df_pairs   = pd.read_csv(pairs_file)
        df_targets = pd.read_csv(targets_file)

        if len(df_pairs) != len(df_targets):
            raise ValueError(
                f"Row count mismatch: {len(df_pairs)} in pairs vs {len(df_targets)} in targets"
            )

        weight= 1 / len(df_pairs)

        for idx, pair_row in df_pairs.iterrows():
            x_str = str(pair_row.iloc[1])
            y_str = str(pair_row.iloc[2])

            x = np.array([float(v) for v in x_str.split()]).reshape(-1, 1)
            y = np.array([float(v) for v in y_str.split()]).reshape(-1, 1)

            # Swap if target is -1
            if df_targets.iloc[idx, 1] == -1:
                x, y = y, x

            pair_idx = idx + 1

            # Check for existing row
            exists = (
                (df_existing["method"] == method_name)
                & (df_existing["parameters"] == parameters)
                & (df_existing["dataset"] == dataset)
                & (df_existing["Pair"] == pair_idx)
            ).any()

            if exists and not overwrite:
                print(f"⏩ Skipping {dataset} Pair {pair_idx} (already computed)")
                continue

            # Compute score
            try:
                score = func([x,y], *args, **kwargs)
            except Exception as e:
                print(f"⚠️ Error on {dataset} Pair {pair_idx}: {e}")
                continue

            if np.isnan(score):
                score = "NA"

            # ----------------------------------------------------
            # Inline row append (KeyboardInterrupt-safe)
            # ----------------------------------------------------
            new_row = {
                "method": method_name,
                "parameters": parameters,
                "dataset": dataset,
                "Pair": pair_idx,
                "score": score,
                "weight": weight,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            df_existing = pd.concat([df_existing, pd.DataFrame([new_row])],
                                    ignore_index=True)
            df_existing.to_csv(path, index=False)
            print(f"✔ Saved {dataset} Pair {pair_idx}")

    print(f"\n✅ Saved CE results to {path}")
    return path


def run_anlsmn(
        func,
        datasets=None,
        read_dir="benchmarks/synthetic/ANLSMN-Tagasovska",
        write_path="results/ANLSMN_scores.csv",
        overwrite=False,
        *args,
        **kwargs
    ):

    # -----------------------------
    # Dataset selection
    # -----------------------------
    if datasets is None:
        datasets = [
            d for d in os.listdir(read_dir)
            if os.path.isdir(os.path.join(read_dir, d))
        ]

    os.makedirs(os.path.dirname(write_path), exist_ok=True)

    method_name = get_method_name(func)
    parameters  = serialize_params(args, kwargs)

    # -----------------------------
    # Load previous results (if any)
    # -----------------------------
    if os.path.exists(write_path):
        df_existing = pd.read_csv(write_path, keep_default_na=False)
        df_existing["parameters"] = df_existing["parameters"].map(normalize_str)
    else:
        df_existing = pd.DataFrame(columns=[
            "method", "parameters", "dataset", "Pair",
            "score", "weight", "timestamp"
        ])

    # -----------------------------
    # Main dataset loop
    # -----------------------------
    for ext in datasets:
        print(f"\n🚀 Running ANLSMN dataset: {ext}")

        dir_ext = os.path.join(read_dir, ext)
        gt_file = os.path.join(dir_ext, "pairs_gt.txt")

        if not os.path.isfile(gt_file):
            raise FileNotFoundError(f"Missing ground truth file: {gt_file}")

        # Load ground truth:
        #   +1 means X->Y
        #   -1 means Y->X (must be swapped)
        pairs_gt = pd.read_csv(gt_file, header=None).iloc[:, 0].astype(int).values
        n_pairs  = len(pairs_gt)
        weight   = 1 / n_pairs

        # -----------------------------
        # Pair loop
        # -----------------------------
        for i in range(1, n_pairs + 1):
            pair_idx = i

            # Exists?
            exists = (
                (df_existing["method"] == method_name) &
                (df_existing["parameters"] == parameters) &
                (df_existing["dataset"] == ext) &
                (df_existing["Pair"] == pair_idx)
            ).any()

            if exists and not overwrite:
                print(f"⏩ Skipping {ext} Pair {i} (already computed)")
                continue

            # -----------------------------
            # Load pair file
            # -----------------------------
            pair_file = os.path.join(dir_ext, f"pair_{i}.txt")
            if not os.path.isfile(pair_file):
                print(f"⚠️ Missing pair file: {pair_file}, skipping.")
                continue

            df_pair = pd.read_csv(pair_file, sep=",", header=0, index_col=0)
            x = df_pair.iloc[:, 0].values.reshape(-1, 1)
            y = df_pair.iloc[:, 1].values.reshape(-1, 1)

            # -----------------------------
            # Correct direction using GT
            # -----------------------------
            if pairs_gt[i - 1] == -1:
                x, y = y, x     # swap

            # -----------------------------
            # Run method on correctly oriented (x,y)
            # -----------------------------
            try:
                score = func([x, y], *args, **kwargs)
            except Exception as e:
                print(f"⚠️ Error on {ext} Pair {i}: {e}")
                continue

            if isinstance(score, float) and np.isnan(score):
                score = "NA"

            # -----------------------------
            # Append row to CSV
            # -----------------------------
            new_row = {
                "method": method_name,
                "parameters": parameters,
                "dataset": ext,
                "Pair": pair_idx,
                "score": score,
                "weight": weight,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            df_existing = pd.concat([df_existing, pd.DataFrame([new_row])],
                                    ignore_index=True)
            df_existing.to_csv(write_path, index=False)
            print(f"✔ Saved {ext} Pair {i}")

    print(f"\n✅ Saved ANLSMN results to {write_path}")


def run_sim(
        func,
        datasets=None,
        read_dir="benchmarks/synthetic/SIM-Mooij",
        write_path="results/SIM_scores.csv",
        overwrite=False,
        *args,
        **kwargs
    ):

    # --------------------------------------------
    # Dataset list
    # --------------------------------------------
    if datasets is None:
        datasets = [
            d for d in os.listdir(read_dir)
            if os.path.isdir(os.path.join(read_dir, d))
        ]

    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    out_path = write_path

    method_name = get_method_name(func)
    parameters  = serialize_params(args, kwargs)

    # --------------------------------------------
    # Load existing CSV if available
    # --------------------------------------------
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path, keep_default_na=False)
        df_existing["parameters"] = df_existing["parameters"].map(normalize_str)
    else:
        df_existing = pd.DataFrame(columns=[
            "method", "parameters", "dataset", "Pair",
            "score", "weight", "timestamp"
        ])

    # --------------------------------------------
    # Loop over datasets
    # --------------------------------------------
    for dataset in datasets:
        print(f"\n🚀 Running SIM dataset: {dataset}")

        dataset_dir = os.path.join(read_dir, dataset)
        meta_file = os.path.join(dataset_dir, "pairmeta.txt")

        if not os.path.isfile(meta_file):
            raise FileNotFoundError(f"Missing pairmeta.txt in {dataset_dir}")

        # Load pairmeta.txt with no header
        meta = pd.read_csv(meta_file, sep=r"\s+", header=None,dtype={0: str})
        meta.columns = ["pair", "c_start", "c_end", "e_start", "e_end", "weight"]

        # Ensure integer indexing
        meta["pair"] = meta["pair"].astype(str)
        meta["c_start"] = meta["c_start"].astype(int)
        meta["c_end"]   = meta["c_end"].astype(int)
        meta["e_start"] = meta["e_start"].astype(int)
        meta["e_end"]   = meta["e_end"].astype(int)

        # --------------------------------------------
        # Loop through all pairs defined in pairmeta.txt
        # --------------------------------------------
        for _, row in meta.iterrows():
            pair_id  = row["pair"]            # e.g. "0001"
            pair_idx = int(pair_id)           # numeric for CSV

            # Check overwrite skip
            exists = (
                (df_existing["method"] == method_name) &
                (df_existing["parameters"] == parameters) &
                (df_existing["dataset"] == dataset) &
                (df_existing["Pair"] == pair_idx)
            ).any()

            if exists and not overwrite:
                print(f"⏩ Skipping {dataset} Pair {pair_id} (already computed)")
                continue

            # --------------------------------------------
            # Load corresponding pair file
            # --------------------------------------------
            print("Pair",pair_id)
            pair_file = os.path.join(dataset_dir, f"pair{pair_id}.txt")
            if not os.path.isfile(pair_file):
                print(f"⚠️ Missing pair file: {pair_file}, skipping.")
                continue

            # Load all variable columns
            df_pair = pd.read_csv(pair_file, sep=r"\s+", header=None)
            data = df_pair.values  # numpy array, shape (n, d)

            # Extract cause and effect variables (1-indexed in meta)
            c_start, c_end = row["c_start"], row["c_end"]
            e_start, e_end = row["e_start"], row["e_end"]

            X = data[:, c_start-1 : c_end].reshape(len(data), -1)
            Y = data[:, e_start-1 : e_end].reshape(len(data), -1)

            # --------------------------------------------
            # Run the method
            # --------------------------------------------
            try:
                score = func([X, Y], *args, **kwargs)
            except Exception as e:
                print(f"⚠️ Error on {dataset} Pair {pair_id}: {e}")
                continue

            if isinstance(score, float) and np.isnan(score):
                score = "NA"

            weight = float(row["weight"])

            # --------------------------------------------
            # Append row to CSV
            # --------------------------------------------
            new_row = {
                "method": method_name,
                "parameters": parameters,
                "dataset": dataset,
                "Pair": pair_idx,
                "score": score,
                "weight": weight,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            df_existing = pd.concat([df_existing, pd.DataFrame([new_row])],
                                    ignore_index=True)
            df_existing.to_csv(out_path, index=False)
            print(f"✔ Saved {dataset} Pair {pair_id}")

    print(f"\n✅ Saved SIM results to {out_path}")
    return out_path