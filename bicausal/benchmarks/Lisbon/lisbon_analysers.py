import os
import pandas as pd
import numpy as np
import re

def source_weight(ne, nx, ny):
    """Compute dataset weight based on number of examples and unique X/Y groups."""
    return np.log(1 + ne) * np.sqrt(nx * ny) / np.log(2) #normalization


def parse_variable_group(var_str):
    """
    Parse a variable string and extract all variants if it contains parentheses
    with comma-separated entries.
    Example:
        "allelectrons(allelectrons_Average,allelectrons_Total)" ->
        ['allelectrons', 'allelectrons_Average', 'allelectrons_Total']
    """
    var_str = var_str.strip()
    m = re.match(r"^([^(]+)\(([^)]*)\)\s*$", var_str)
    if not m:
        return [var_str]
    base = m.group(1).strip()
    inner = m.group(2).strip()
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if len(parts) > 1:
        return [base] + parts
    return [var_str]


def build_canonical_map(pairs):
    parent = {}

    def make_set(x):
        if x not in parent:
            parent[x] = x

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    seen_tokens = set()
    for x_raw, y_raw in pairs:
        for token in parse_variable_group(x_raw) + parse_variable_group(y_raw):
            make_set(token)
            seen_tokens.add(token)

    for x_raw, y_raw in pairs:
        for group in (parse_variable_group(x_raw), parse_variable_group(y_raw)):
            if len(group) > 1:
                base = group[0]
                for token in group[1:]:
                    union(base, token)

    # build groups by root
    groups = {}
    for t in list(seen_tokens):
        root = find(t)
        groups.setdefault(root, []).append(t)

    canonical = {}
    for root, members in groups.items():
        rep = min(members, key=lambda s: (len(s), s))
        for m in members:
            canonical[m] = rep

    return canonical, seen_tokens


def obtain_source_specs(base_dir="benchmarks/Lisbon", excel=True, tex=True, table_dir="table", table_title="Source Specifications"):
    meta_root = os.path.join(base_dir, "meta")
    rows = []

    for root, _, files in os.walk(meta_root):
        if "README.md" not in files:
            continue

        readme_path = os.path.join(root, "README.md")
        meta_files = [f for f in files if f.endswith("_meta.txt")]
        pairs_path = os.path.join(root, "pairs.xlsx")

        # field/source
        rel_path = os.path.relpath(root, meta_root)
        parts = rel_path.split(os.sep)
        if len(parts) != 2:
            continue
        field, source = parts

        # --- n_examples from pairs.xlsx ---
        if os.path.exists(pairs_path):
            try:
                pairs_df = pd.read_excel(pairs_path)
                n_examples = len(pairs_df)
            except Exception as e:
                print(f"[ERROR] Reading {pairs_path}: {e}")
                n_examples = 0
        else:
            n_examples = 0

        # --- n_points from all _meta.txt files ---
        n_points_list = []
        for mf in meta_files:
            try:
                with open(os.path.join(root, mf), encoding="utf-8") as f:
                    text = f.read()
                m = re.search(r"Number of entries:\s*(\d+)", text)
                if m:
                    n_points_list.append(int(m.group(1)))
            except Exception as e:
                print(f"[ERROR] Reading {mf}: {e}")

        if n_points_list:
            n_points_mean = int(np.mean(n_points_list))
            n_points_median = int(np.median(n_points_list))
        else:
            n_points_mean = 0
            n_points_median = 0

        # --- parse README.md: extract citations + variable pairs ---
        pairs = []
        n_groupings = 0
        in_variables = False
        in_citations = False
        citation = None

        with open(readme_path, encoding="utf-8") as f:
            for line in f:
                l = line.strip()

                # --- Citation section ---
                if l.lower().startswith("## citation"):
                    in_citations = True
                    in_variables = False
                    continue

                if l.lower().startswith("## variables"):
                    in_variables = True
                    in_citations = False
                    continue

                if l.lower().startswith("## causal reasoning"):
                    in_variables = False
                    in_citations = False
                    continue

                if in_citations and l.startswith("-"):
                    if citation is None:  # only take first citation
                        citation = l[1:].strip()
                    continue

                # --- Variables section ---
                if in_variables and l.startswith("-"):
                    n_groupings += 1
                    if "→" in l:
                        parts_line = l[1:].split("→", 1)  # remove leading "-"
                        if len(parts_line) == 2:
                            pairs.append((parts_line[0].strip(), parts_line[1].strip()))

        if not pairs:
            print(f"[WARNING] No X→Y pairs found in {readme_path}")
            continue

        canonical, _ = build_canonical_map(pairs)

        # compute n_x and n_y
        x_reps = set()
        y_reps = set()
        for x_raw, y_raw in pairs:
            for token in parse_variable_group(x_raw):
                x_reps.add(canonical.get(token, token))
            for token in parse_variable_group(y_raw):
                y_reps.add(canonical.get(token, token))

        rows.append({
            "field": field,
            "source": source,
            "n_examples": n_examples,
            "n_points_mean": n_points_mean,
            "n_points_median": n_points_median,
            "n_x": len(x_reps),
            "n_y": len(y_reps),
            "Ngroupings": n_groupings,
            "citation": citation or ""  # ensure column always exists
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[INFO] No datasets found.")
        return df

    df["weight"] = df.apply(lambda r: source_weight(r["n_examples"], r["n_x"], r["n_y"]), axis=1)
    
    output_path = os.path.join(base_dir, "source_specs.xlsx")
    # Save
    if excel:
        df.to_excel(output_path, index=False)

    if tex:
        df.rename(columns=lambda x: x[0].upper() + x[1:] if x else x, inplace=True)
        latex_path = os.path.join(table_dir, f"{table_title}.tex")
        latex_text = df.to_latex(index=False, escape=True)
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_text)

    return df


def field_stats_excel(base_dir="benchmarks/Lisbon", excel=True, tex=True, table_dir="table", table_title="Field Statistics"):
    # Find the Excel file matching *_weights.xlsx
    input_path = os.path.join(base_dir, "source_specs.xlsx")
    output_path = os.path.join(base_dir, "field_stats.xlsx")

    # Read the Excel file
    df = pd.read_excel(input_path)

    # Ensure columns exist
    expected_cols = ["field", "source", "n_examples", "n_points_mean", "n_points_median",
                     "n_x", "n_y", "weight"]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input file: {missing}")

    # Group by field and compute stats
    grouped = df.groupby("field", as_index=False).agg({
        "source": "nunique",
        "n_examples": "mean",
        "n_points_mean": "mean",
        "n_x": "mean",
        "n_y": "mean",
        "weight": "sum"
    })
    grouped.rename(columns={"source": "n_sources"}, inplace=True)

    # Compute overall stats (the "all" row)
    overall = pd.DataFrame({
        "field": ["all"],
        "n_sources": [df["source"].nunique()],
        "n_examples": [df["n_examples"].mean()],
        "n_points_mean": [df["n_points_mean"].mean()],
        "n_x": [df["n_x"].mean()],
        "n_y": [df["n_y"].mean()],
        "weight": [df["weight"].sum()]
    })

    # Append overall row to grouped stats
    result = pd.concat([grouped, overall], ignore_index=True)

    # Save
    if excel:
        result.to_excel(output_path, index=False)

    if tex:
        result.rename(columns=lambda x: x[0].upper() + x[1:] if x else x, inplace=True)
        latex_path = os.path.join(table_dir, f"{table_title}.tex")
        latex_text = result.to_latex(index=False, escape=True)
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_text)

    print(f"Field statistics (including 'all' summary) saved to: {output_path}")

    return result


def points_sources_stats(base_dir="benchmarks/Lisbon"):
    """
    Returns:
        examples_per_source: list[int]
        pairs_per_example: list[int]
    """
    data_root = os.path.join(base_dir, "data")
    meta_root = os.path.join(base_dir, "meta")
    if not os.path.exists(data_root):
        raise ValueError(f"Data folder not found: {data_root}")

    examples_per_source = []
    pairs_per_example = []

    for root, dirs, files in os.walk(data_root):
        if root == data_root:
            continue

        txt_files = [f for f in files if f.endswith(".txt")]
        if not txt_files:
            continue

        # count examples for this source
        examples_per_source.append(len(txt_files))

        rel_path = os.path.relpath(root, data_root)
        meta_dir = os.path.join(meta_root, rel_path)

        # gather pairs per example
        for fname in txt_files:
            meta_file = os.path.join(meta_dir, fname.replace(".txt", "_meta.txt"))
            if not os.path.exists(meta_file):
                continue
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    for line in mf:
                        if line.strip().startswith("Number of entries:"):
                            n_points = int(line.split(":", 1)[1].strip())
                            pairs_per_example.append(n_points)
                            break
            except:
                continue

    return examples_per_source, pairs_per_example