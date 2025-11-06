import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from helpers.namemap import name_map  # custom readable name map
from helpers.utils import save_imgs

def plot_execution_times(
    data_dir="results",
    img_dir="analysis/interesting_plots",
    target_time=30
):
    """
    Plots execution time vs number of points for each method from a CSV file.
    - data_dir: directory containing the CSV (default='data')
    - img_dir: where to save the resulting plot (default='plots')
    - target_time: horizontal reference line in seconds (default=30)
    - filename: name of the CSV file (default='times.csv')
    """
    filename="times.csv"
    # === Load data ===
    csv_path = os.path.join(data_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure correct numeric types
    df["npoints"] = pd.to_numeric(df["npoints"], errors="coerce")
    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce")

    # Drop any NaNs
    df = df.dropna(subset=["npoints", "execution_time", "function"])
    if df.empty:
        raise ValueError("❌ No valid data found in CSV.")

    # === Plot setup ===
    plt.figure(figsize=(8, 6))
    plt.axhline(target_time, color="gray", linestyle="--", linewidth=1)

    cross_points = {}

    # === Loop through methods ===
    for method, group in df.groupby("function"):
        group = group.sort_values("npoints")
        x = group["npoints"].values
        y = group["execution_time"].values

        # Skip if insufficient data
        if len(x) < 2:
            continue

        # Interpolation: find number of points at target_time
        f = interp1d(y, x, kind="linear", fill_value="extrapolate")
        point_at_target = float(f(target_time))
        cross_points[method] = point_at_target

        within_range = np.min(x) <= point_at_target <= np.max(x)
        readable_name = name_map.get(method, method)

        if within_range:
            label_text = f"{readable_name} (≈{int(round(point_at_target))} pts)"
        else:
            label_text = readable_name

        plt.plot(x, y, marker="o", label=label_text)

    # === Final styling ===
    plt.xlabel("Points")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Execution Time vs Points (crossing at {target_time}s)")
    plt.legend(title="Method", loc="best")
    plt.grid(True)
    plt.tight_layout()

    # === Save image ===
    save_imgs("execution_time_vs_points", img_dir)
    plt.show()

    # === Print numeric output ===
    print("\n📊 Crossing Points:")
    for method, value in cross_points.items():
        readable_name = name_map.get(method, method)
        print(f"{readable_name}: reaches {target_time}s at ≈ {value:.2f} points")

    return cross_points
