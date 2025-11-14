import os
import pandas as pd
import numpy as np

def load_fields(dataset_dir="benchmarks/Lisbon"):
    df = pd.read_excel(os.path.join(dataset_dir, "source_specs.xlsx"))
    return {row["source"]: row["field"] for _, row in df.iterrows()}

def load_source_weights(dataset_dir="benchmarks/Lisbon"):
    df = pd.read_excel(os.path.join(dataset_dir, "source_specs.xlsx"))
    return {row["source"]: row["weight"] for _, row in df.iterrows()}

def load_pair_weights(dataset_dir="benchmarks/Lisbon"):
    meta_dir = os.path.join(dataset_dir, "meta")
    pair_weights = {}  # key: source (folder name), value: dict of filename -> weight

    for field in os.listdir(meta_dir):
        field_path = os.path.join(meta_dir, field)

        for source in os.listdir(field_path):
            source_path = os.path.join(field_path, source)

            pairs_file = os.path.join(source_path, "pairs.xlsx")
            if not os.path.exists(pairs_file):
                print(f"Warning: pairs.xlsx not found at {pairs_file}, skipping.")
                continue

            df = pd.read_excel(pairs_file)
            filename_col = "dataset_file"
            wcol = "weight"

            wsum = df[wcol].sum()
            if not np.isclose(wsum, 1.0, atol=1e-6):
                # Emit a clear error and set pair_weight = np.nan for all filenames in this source
                print(f"Error: weights in {pairs_file} (source '{source}', field '{field}') sum to {wsum:.6f} (expected ~1.0). Marking these files as invalid (np.nan).")
                pair_weights[source] = {row[filename_col]: np.nan for _, row in df.iterrows()}
            else:
                # Accept per-row weights
                pair_weights[source] = {row[filename_col]: float(row[wcol]) for _, row in df.iterrows()}

    return pair_weights

def load_lisbon_metadata(dataset_dir):
    fields = load_fields(dataset_dir)            # keys: source folder names
    source_weights = load_source_weights(dataset_dir)  # keys: source folder names
    pair_weights = load_pair_weights(dataset_dir)      # keys: source folder names

    metadata = {}
    for source in pair_weights.keys():
        if source not in source_weights:
            raise ValueError(f"Source '{source}' found in pairs.xlsx but missing in source_weights.xlsx")
        if source not in fields:
            raise ValueError(f"Source '{source}' missing in fields metadata")

        # Combine weights: multiply source weight by pair weights for all files under that source
        combined_weights = {fname: source_weights[source] * w for fname, w in pair_weights[source].items()}
        for fname, combined_weight in combined_weights.items():
            metadata[fname] = {
                "field": fields[source],  # assume all files in this source belong to the same field
                "weight": combined_weight
            }

    return metadata


