import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
from helpers.utils import getTuebingen


def run_tuebingen(func, read_dir="Tuebingen", write_dir="results/tuebingen", overwrite=True, *args, **kwargs):
    """
    Runs func on the Tübingen dataset and saves results to a CSV file.
    Includes 'Pair', 'score', 'weight', 'timestamp' columns.
    """
    data, weights = getTuebingen(read_dir)
    results = []

    for i, ((x, y), w) in enumerate(zip(data, weights)):
        try:
            score = func([x, y], *args, **kwargs)
        except Exception as e:
            print(f"⚠️ Skipping Pair {i+1} due to error: {e}")
            continue

        results.append({
            "Pair": i + 1,
            "score": score,
            "weight": w,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    if not results:
        print("❌ No results to save.")
        return

    os.makedirs(write_dir, exist_ok=True)
    method_name = func.__name__
    path = os.path.join(write_dir, f"{method_name}.csv")

    df_new = pd.DataFrame(results)
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        if overwrite:
            df_existing = df_existing[~df_existing["Pair"].isin(df_new["Pair"])]
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(path, index=False)
    print(f"✅ Saved Tuebingen results to {path}")
    return path

#pass way to data directory (or smaller if desired)
def run_lisbon(func, read_dir="datasets/Lisbon/data", write_dir="results/lisbon", overwrite=True, *args, **kwargs):
    """
    Applies func(x, y, *args, **kwargs) to all .txt files under read_dir recursively.
    Saves results in a CSV file: ['filename', 'score', 'timestamp'].
    """
    os.makedirs(write_dir, exist_ok=True)
    output_path = os.path.join(write_dir, f"{func.__name__}.csv")

    if os.path.exists(output_path):
        df_results = pd.read_csv(output_path)
    else:
        df_results = pd.DataFrame(columns=["filename", "score", "timestamp"])

    for root, _, files in os.walk(read_dir):
        txt_files = [f for f in files if f.lower().endswith(".txt")]
        if not txt_files:
            continue

        print(f"🔍 Processing source: {os.path.basename(root)} ({len(txt_files)} files)")
        new_rows = []

        for fname in txt_files:
            path = os.path.join(root, fname)
            try:
                df = pd.read_csv(path, sep=None, engine="python", header=None)
                x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
                score = func([x.reshape(-1, 1), y.reshape(-1, 1)], *args, **kwargs)
                new_rows.append({
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
                df_results = df_results[~df_results["filename"].isin(df_new["filename"])]
            df_results = pd.concat([df_results, df_new], ignore_index=True)
            df_results.to_csv(output_path, index=False)
            print(f"💾 Saved {len(df_new)} results from {os.path.basename(root)}.")

    print(f"✅ All Lisbon results saved to {output_path}")


#To be interrupted: for slow functions, it takes forever!
#Choose very long file for testing!!!
def benchmark_function(func, test_file, write_dir="results", overwrite=True, seed=42, *args, **kwargs):
    """
    Benchmarks execution time of func([x, y], *args, **kwargs) as a function of sample size.
    Saves results to CSV with ['function', 'npoints', 'execution_time', 'timestamp'].
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

    if os.path.exists(output_path):
        times_df = pd.read_csv(output_path)
    else:
        times_df = pd.DataFrame(columns=["function", "npoints", "execution_time", "timestamp"])

    for n_points in sizes:
        exists = ((times_df["function"] == func.__name__) & (times_df["npoints"] == n_points)).any()
        if exists and not overwrite:
            print(f"Skipping n={n_points} (already computed)")
            continue

        subset = [x[:n_points].reshape(-1, 1), y[:n_points].reshape(-1, 1)]
        print(f"⏱ Running {func.__name__} with {n_points} points...")

        start = time.time()
        try:
            func(subset, *args, **kwargs)
        except Exception as e:
            print(f"⚠️ Error at n={n_points}: {e}")
            continue
        elapsed = time.time() - start

        new_row = {
            "function": func.__name__,
            "npoints": n_points,
            "execution_time": elapsed,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if overwrite:
            times_df = times_df[~((times_df["function"] == func.__name__) & (times_df["npoints"] == n_points))]

        times_df = pd.concat([times_df, pd.DataFrame([new_row])], ignore_index=True)
        times_df.to_csv(output_path, index=False)
        print(f"✅ Completed {n_points} points in {elapsed:.4f}s.")

    print(f"📊 Benchmark results saved to {output_path}")
