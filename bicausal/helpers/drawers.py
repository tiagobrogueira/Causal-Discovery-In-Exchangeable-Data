import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from bicausal.helpers.namemap import name_map  # custom readable name map
from bicausal.helpers.utils import save_imgs, normalize_str


def plot_execution_times(
    data_dir="results",
    img_dir="analysis/interesting_plots",
    target_time=30,
):
    """
    Plots execution time vs number of points for each (method, parameters) combo.
    - Each method/parameter combination is shown as a separate line.
    - Draws a dashed line at target_time seconds.
    - Saves figure and prints crossing points (where runtime ≈ target_time).
    """
    filename = "times.csv"
    csv_path = os.path.join(data_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found at {csv_path}")

    # === Load data ===
    df = pd.read_csv(csv_path)
    required_cols = {"method", "parameters", "npoints", "execution_time"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ Missing columns in CSV: expected {required_cols}")

    # Normalize and clean
    df["parameters"] = df["parameters"].apply(normalize_str)
    df["npoints"] = pd.to_numeric(df["npoints"], errors="coerce")
    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df = df.dropna(subset=["npoints", "execution_time", "method"])
    if df.empty:
        raise ValueError("❌ No valid data found in CSV.")

    # === Plot setup ===
    plt.figure(figsize=(8, 6))
    plt.axhline(target_time, color="gray", linestyle="--", linewidth=1)
    cross_points = {}

    # === Group by (method, parameters) ===
    for (method, params), group in df.groupby(["method", "parameters"], dropna=False):
        group = group.sort_values("npoints")
        x = group["npoints"].values
        y = group["execution_time"].values
        if len(x) < 2:
            continue

        # Interpolate to find the sample size that hits target_time
        try:
            f = interp1d(y, x, kind="linear", fill_value="extrapolate")
            point_at_target = float(f(target_time))
        except Exception:
            point_at_target = np.nan

        cross_points[f"{method} ({params})"] = point_at_target

        readable_name = name_map.get(method, method)
        # Simplify empty parameters in label
        label_text = f"{readable_name}" if params == "" else f"{readable_name} ({params})"

        plt.plot(x, y, marker="o", label=label_text)

    # === Final styling ===
    plt.xlabel("Number of Points")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Execution Time vs Sample Size (target={target_time}s)")
    plt.legend(title="Method / Parameters", loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_imgs(img_dir, "execution_times_vs_points")
    plt.show()

    # === Print numeric summary ===
    print("\n📊 Crossing Points (time ≈ target_time):")
    for combo, value in cross_points.items():
        if np.isnan(value):
            continue
        print(f"{combo}: ≈ {value:.2f} points")

    return cross_points

