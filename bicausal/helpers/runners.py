import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from bicausal.helpers.utils import getTuebingen, serialize_params, normalize_str





def run_tuebingen(func, read_dir="datasets/Tuebingen", write_dir="results", overwrite=False, *args, **kwargs):
    """
    Runs func on the Tübingen dataset and saves results to a shared CSV file.
    Columns: ['method', 'parameters', 'Pair', 'score', 'weight', 'timestamp']
    """
    data, weights = getTuebingen(read_dir)
    os.makedirs(write_dir, exist_ok=True)
    path = os.path.join(write_dir, "tuebingen_scores.csv")

    method_name = func.__name__
    parameters = serialize_params(args, kwargs)

    # Load or create dataframe
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
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


def run_lisbon(func, read_dir="datasets/Lisbon/data", write_dir="results", overwrite=False, *args, **kwargs):
    """
    Applies func(x, y, *args, **kwargs) to all .txt files under read_dir recursively.
    Saves results to a shared CSV file:
    Columns: ['method', 'parameters', 'filename', 'score', 'timestamp']
    """
    os.makedirs(write_dir, exist_ok=True)
    output_path = os.path.join(write_dir, "lisbon_scores.csv")
    parameters = serialize_params(args, kwargs)
    method_name = func.__name__

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


def benchmark_function(func, test_file, write_dir="results", overwrite=False, seed=42, *args, **kwargs):
    """
    Benchmarks execution time of func([x, y], *args, **kwargs) as a function of sample size.
    Saves to shared CSV: ['method', 'parameters', 'npoints', 'execution_time', 'timestamp']
    """
    np.random.seed(seed)
    os.makedirs(write_dir, exist_ok=True)
    output_path = os.path.join(write_dir, "times.csv")

    df = pd.read_csv(test_file, sep=None, engine="python", header=None)
    x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]
    n_total = len(x)

    sizes = []
    n = 10
    while n < n_total:
        sizes.append(n)
        n = int(n * 1.7 + 10)
    if sizes[-1] != n_total:
        sizes.append(n_total)

    method_name = func.__name__
    parameters = serialize_params(args, kwargs)

    if os.path.exists(output_path):
        times_df = pd.read_csv(output_path)
        times_df["parameters"] = times_df["parameters"].map(normalize_str)
    else:
        times_df = pd.DataFrame(columns=["method", "parameters", "npoints", "execution_time", "timestamp"])

    for n_points in sizes:
        exists = (
            (times_df["method"] == method_name)
            & (times_df["parameters"] == parameters)
            & (times_df["npoints"] == n_points)
        ).any()
        if exists and not overwrite:
            print(f"⏩ Skipping n={n_points} for {method_name} (already computed with same parameters)")
            continue

        subset = [x[:n_points].reshape(-1, 1), y[:n_points].reshape(-1, 1)]
        print(f"⏱ Running {method_name} with {n_points} points...")

        start = time.time()
        try:
            func(subset, *args, **kwargs)
        except Exception as e:
            print(f"⚠️ Error at n={n_points}: {e}")
            continue
        elapsed = time.time() - start

        new_row = {
            "method": method_name,
            "parameters": parameters,
            "npoints": n_points,
            "execution_time": elapsed,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if overwrite:
            times_df = times_df[
                ~(
                    (times_df["method"] == method_name)
                    & (times_df["parameters"] == parameters)
                    & (times_df["npoints"] == n_points)
                )
            ]

        times_df = pd.concat([times_df, pd.DataFrame([new_row])], ignore_index=True)
        times_df.to_csv(output_path, index=False)
        print(f"✅ Completed {n_points} points in {elapsed:.4f}s.")

    print(f"📊 Benchmark results saved to {output_path}")
