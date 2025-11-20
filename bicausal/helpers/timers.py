import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from bicausal.helpers.namemap import name_map  # custom readable name map
from bicausal.helpers.utils import save_imgs, normalize_str

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

    method_name = get_method_name(func)
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


def plot_execution_times(
    data_dir="results",
    img_dir="plots",
    target_time=30,
):
    """
    Plots execution time vs number of points for each method.
    Stores the largest sample size for each method in storage/max_points_cache.json,
    only if the crossing point <= largest tested npoints.
    """
    import os, json

    STORAGE_DIR = "storage"
    os.makedirs(STORAGE_DIR, exist_ok=True)
    CACHE_FILE = os.path.join(STORAGE_DIR, "max_points_cache.json")

    # Load persistent cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            max_npoints_cache = json.load(f)
    else:
        max_npoints_cache = {}

    filename = "times.csv"
    csv_path = os.path.join(data_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found at {csv_path}")

    # === Load data ===
    df = pd.read_csv(csv_path)
    required_cols = {"method", "parameters", "npoints", "execution_time"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ Missing columns in CSV: expected {required_cols}")

    df["parameters"] = df["parameters"].apply(normalize_str)
    df["npoints"] = pd.to_numeric(df["npoints"], errors="coerce")
    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df = df.dropna(subset=["npoints", "execution_time", "method"])
    if df.empty:
        raise ValueError("❌ No valid data found in CSV.")

    plt.figure(figsize=(8, 6))
    plt.axhline(target_time, color="gray", linestyle="--", linewidth=1)
    cross_points = {}

    for method, group in df.groupby("method"):
        group = group.sort_values("npoints")
        x = group["npoints"].values
        y = group["execution_time"].values
        if len(x) < 2:
            continue

        try:
            f = interp1d(y, x, kind="linear", fill_value="extrapolate")
            point_at_target = float(f(target_time))
        except Exception:
            point_at_target = np.nan

        cross_points[method] = point_at_target
        readable_name = name_map.get(method, method)
        plt.plot(x, y, marker="o", label=readable_name)

        # Only save to cache if crossing point <= max tested npoints
        if not np.isnan(point_at_target) and point_at_target <= max(x):
            max_npoints_cache[method] = int(max(x))

    plt.xlabel("Number of Points")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Execution Time vs Sample Size (target={target_time}s)")
    plt.legend(title="Method", loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_imgs(img_dir, "execution_times_vs_points")
    plt.show()

    # Save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(max_npoints_cache, f)

    print("\n📊 Crossing Points (time ≈ target_time):")
    for method, value in cross_points.items():
        if np.isnan(value):
            continue
        print(f"{method}: ≈ {value:.2f} points")

    return cross_points


def get_max_points(method, storage_path=None):
    """
    Returns the cached max number of points for a given method.
    Returns None if no value is stored.
    """
    import os, json

    storage_path = storage_path or "storage/max_points_cache.json"
    if not os.path.exists(storage_path):
        return None

    with open(storage_path, "r") as f:
        cache = json.load(f)

    return cache.get(method)



