# Causal Discovery in Exchangeable Data

This repository contains the code, benchmark data, evaluation pipeline, and reproduction notebooks associated with the work on **causal discovery in exchangeable data**.

At a high level, the repository serves four goals:

1. **Benchmarking bivariate causal discovery methods** on real-world and synthetic datasets.
2. **Providing the Lisbon benchmark**, a multi-domain real-world benchmark for bivariate causal discovery.
3. **Separating score generation from evaluation** to make experiments reproducible and easier to audit.
4. **Reproducing figures and tables** from the papers/notebooks included in the project.

---

## Table of contents

- [What this repository contains](#what-this-repository-contains)
- [Recommended way to use the repository](#recommended-way-to-use-the-repository)
  - [Option 1 — Clone the repository (recommended)](#option-1--clone-the-repository-recommended)
  - [Option 2 — Install with pip](#option-2--install-with-pip)
- [Core experimental philosophy](#core-experimental-philosophy)
- [Typical workflow](#typical-workflow)
- [Repository structure](#repository-structure)
- [Folder-by-folder explanation](#folder-by-folder-explanation)
- [Benchmarks included](#benchmarks-included)
- [Methods included](#methods-included)
- [Notebooks](#notebooks)
- [Important notes and caveats](#important-notes-and-caveats)
- [Outputs produced by the repository](#outputs-produced-by-the-repository)
- [Citation](#citation)
- [License](#license)

---

## What this repository contains

This repository contains:

- the **`bicausal` Python package**;
- the **Lisbon**, **Tübingen**, and **synthetic** benchmarks;
- method wrappers and original method sources;
- evaluation metrics such as **accuracy**, **AUROC**, **AUDRC**, and **LxCIM**;
- precomputed score files and aggregate result files;
- figures and LaTeX tables used in the associated papers;
- notebooks for reproducing the papers and for extending the benchmark/toolbox.

The package name is **`bicausal`**.

---

## Recommended way to use the repository

There are two main ways to use this project.

### Option 1 — Clone the repository (recommended)

This is the **recommended** option if you want the **full reproducibility package**, including:

- notebooks;
- benchmark data;
- R and MATLAB methods;
- precomputed results, figures, and tables;
- utilities for reproducing the papers end-to-end.

```bash
git clone https://github.com/tiagobrogueira/Causal-Discovery-In-Exchangeable-Data.git
cd Causal-Discovery-In-Exchangeable-Data
pip install -e .
```

Why this is the best option:

- it gives you the full repository exactly as used for the experiments;
- it is the safest option for reproducing the notebooks and papers;
- it includes non-Python assets that are **not naturally covered by a lightweight package install**.

After that, you can work with the notebooks inside `bicausal/`, for example:

- `bicausal/run.ipynb`
- `bicausal/evaluate.ipynb`
- `bicausal/time.ipynb`
- `bicausal/lisbon_paper.ipynb`
- `bicausal/lxcim_paper.ipynb`
- `bicausal/rdmdl_paper.ipynb`

---

### Option 2 — Install with pip

If you only want the Python package interface, you can install it with `pip`.

#### Install directly from GitHub

```bash
pip install git+https://github.com/tiagobrogueira/Causal-Discovery-In-Exchangeable-Data.git
```

#### Install from local build artifacts

This repository also includes prebuilt distribution files in `dist/`:

```bash
pip install dist/bicausal-0.1.1-py3-none-any.whl
# or
pip install dist/bicausal-0.1.1.tar.gz
```

#### Important distinction: `pip` install vs cloning

A `pip` installation is best understood as a **lightweight library install**.

Cloning the repository is still preferable if you want:

- the notebooks;
- the full benchmark assets in their original repository layout;
- the R/MATLAB entry points and method files;
- the precomputed CSV results, figures, and LaTeX tables;
- the exact folder structure used in the papers.

In short:

- **Clone the repo** if you want the **research repository**.
- **Use pip** if you want the **Python package**.

#### Downloading benchmark data when using the package

The package includes downloader helpers for at least the main real-world benchmarks:

```python
from bicausal.helpers.downloaders import download_lisbon, download_tuebingen

download_lisbon()
download_tuebingen()
```

`download_lisbon()` can also skip the `pictures/` subfolder by default, which is useful if you only need the benchmark itself and not the source images.

---

## Core experimental philosophy

A central design choice of this repository is the **strict separation between running methods and evaluating methods**.

### Step 1 — Run a method and save raw per-example scores

Whenever a method is run, the repository first stores its **score for each individual example** in CSV files such as:

- `*_scores.csv`
- for example: `lisbon_scores.csv`, `tuebingen_scores.csv`, `CE_scores.csv`, `ANLSMN_scores.csv`, `SIM_scores.csv`

These CSVs are the raw experimental outputs.

### Step 2 — Load those score files and apply metrics

Only after the raw scores are stored do we load them and compute aggregate metrics such as:

- accuracy
- AUROC
- AUDRC
- LxCIM

This separation is intentional and important.

### Why this separation matters

It serves **reproducibility**:

- the expensive or language-specific method run is done once;
- the exact raw scores are preserved;
- metrics can be recomputed later without rerunning the methods;
- evaluation logic can evolve independently from score generation;
- tables and figures can be regenerated from saved score files.

This is one of the key organizing principles of the repository.

---

## Typical workflow

A typical workflow looks like this:

### 1. Run a method on a benchmark

For Python methods, the repository provides runner utilities that expect a function with a signature like:

```python
func([x, y], *args, **kwargs)
```

Typical entry points include:

- `run_tuebingen(...)`
- `run_lisbon(...)`
- `run_ce(...)`
- `run_anlsmn(...)`
- `run_sim(...)`

These save raw scores into the corresponding CSV files inside `bicausal/results/`.

### 2. Evaluate the saved score files

After scores have been generated, use the evaluation helpers:

- `evaluate_tuebingen(...)`
- `evaluate_lisbon(...)`
- `evaluate_synthetic(...)`
- `construct_table(...)`

These load the saved score CSVs, apply the selected metrics, and write aggregate outputs such as `results.csv` and LaTeX tables.

### 3. Reproduce tables and figures

Use the paper notebooks and the plotting/table utilities to recreate the figures and tables saved in:

- `bicausal/plots/`
- `bicausal/table/`

### 4. Optionally benchmark run time

Use the timing utilities and notebook to profile execution-time scaling and write results into:

- `bicausal/results/times.csv`

---

## Repository structure

```text
Causal-Discovery-In-Exchangeable-Data/
├── bicausal/
│   ├── benchmarks/
│   │   ├── Lisbon/
│   │   │   ├── data/
│   │   │   ├── meta/
│   │   │   ├── pictures/
│   │   │   ├── field_stats.xlsx
│   │   │   ├── source_specs.xlsx
│   │   │   ├── lisbon_analysers.py
│   │   │   └── lisbon_utils.py
│   │   ├── Tuebingen/
│   │   │   ├── README
│   │   │   ├── README_polished_may18.tab
│   │   │   ├── TuebingenAnalysis.xlsx
│   │   │   ├── pairXXXX.txt
│   │   │   └── pairXXXX_des.txt
│   │   └── synthetic/
│   │       ├── ANLSMN-Tagasovska/
│   │       ├── CE-Guyon/
│   │       └── SIM-Mooij/
│   ├── helpers/
│   │   ├── extra/
│   │   ├── downloaders.py
│   │   ├── meanwhile.py
│   │   ├── namemap.py
│   │   ├── processers.py
│   │   ├── run_anlsmn.m
│   │   ├── run_ce.m
│   │   ├── run_sim.m
│   │   ├── run_tuebingen.m
│   │   ├── runners.R
│   │   ├── runners.py
│   │   ├── timers.py
│   │   └── utils.py
│   ├── methods/
│   │   ├── source_implementations/
│   │   ├── ANM.py
│   │   ├── BQCD.R
│   │   ├── CAM.R
│   │   ├── CDCI.py
│   │   ├── CDS.py
│   │   ├── CGNN.py
│   │   ├── FOM.py
│   │   ├── GPI.m
│   │   ├── GPI_lx.m
│   │   ├── GPIn.m
│   │   ├── HECI.py
│   │   ├── IGCI.py
│   │   ├── LCUBE.py
│   │   ├── LOCI.py
│   │   ├── NNCL.py
│   │   ├── RDMDL.py
│   │   ├── RECI.py
│   │   ├── ROCHE.py
│   │   ├── SLOPE.R
│   │   └── SLOPPY.R
│   ├── metrics/
│   │   ├── accuracy.py
│   │   ├── audrc.py
│   │   ├── auroc.py
│   │   ├── drawers.py
│   │   ├── evaluators.py
│   │   └── lxcim.py
│   ├── plots/
│   ├── results/
│   ├── storage/
│   ├── table/
│   ├── add_your_own_dataset.ipynb
│   ├── evaluate.ipynb
│   ├── lisbon_paper.ipynb
│   ├── lxcim_paper.ipynb
│   ├── rdmdl_paper.ipynb
│   ├── rdmdl_paper copy.ipynb
│   ├── run.ipynb
│   ├── run.R
│   ├── run.m
│   ├── test_your_own_method.ipynb
│   ├── time.ipynb
│   └── utilize.ipynb
├── dist/
│   ├── bicausal-0.1.1-py3-none-any.whl
│   └── bicausal-0.1.1.tar.gz
├── CITATION.cff
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Folder-by-folder explanation

### `bicausal/`
The main package directory. This is where the code, notebooks, datasets, outputs, and helper scripts live.

### `bicausal/benchmarks/`
Contains the datasets used for evaluation.

#### `bicausal/benchmarks/Lisbon/`
The Lisbon benchmark, organized into several subcomponents:

- `data/` — the actual benchmark pair files grouped by field/source.
- `meta/` — metadata for the benchmark, including per-source and per-pair information.
- `pictures/` — source images or related visual material for Lisbon benchmark sources.
- `source_specs.xlsx` — source-level specifications/weights.
- `field_stats.xlsx` — aggregate field-level statistics.
- `lisbon_utils.py` — utilities for loading Lisbon metadata and weights.
- `lisbon_analysers.py` — utilities for analysing and summarizing the Lisbon benchmark.

#### `bicausal/benchmarks/Tuebingen/`
The Tübingen cause-effect benchmark.

It contains:

- the pair files themselves (`pairXXXX.txt`);
- pair descriptions (`pairXXXX_des.txt`);
- benchmark documentation (`README`, `README_polished_may18.tab`);
- an analysis spreadsheet (`TuebingenAnalysis.xlsx`).

#### `bicausal/benchmarks/synthetic/`
Synthetic datasets used for benchmarking.

The repository includes at least three synthetic suites:

- `ANLSMN-Tagasovska/`
- `CE-Guyon/`
- `SIM-Mooij/`

These are handled by dedicated runners and later aggregated through the same evaluation pipeline.

---

### `bicausal/helpers/`
Utility code for running, downloading, timing, cleaning, and processing experiments.

Main files:

- `runners.py` — Python runners for Lisbon, Tübingen, and synthetic benchmarks.
- `runners.R` — R-side runners for methods implemented in R.
- `run_*.m` — MATLAB helpers for datasets/methods handled in MATLAB.
- `processers.py` — loads raw score CSVs and converts them into aligned score vectors + weights for evaluation.
- `downloaders.py` — helpers to download benchmark assets from the repository.
- `timers.py` — execution-time benchmarking and time plots.
- `utils.py` — dataset loading, cleanup helpers, naming normalization, deduplication, and other utilities.
- `namemap.py` — canonical method-name mapping.
- `extra/` — auxiliary helper code (for example statistical dependencies/utilities used by methods/helpers).
- `storage/` is used together with timing/cache-related logic.

---

### `bicausal/methods/`
Contains the method wrappers/implementations used by the benchmark.

This folder mixes:

- Python methods (`.py`)
- R methods (`.R`)
- MATLAB methods (`.m`)

This is intentional.

#### `bicausal/methods/source_implementations/`
A very important folder.

This folder contains the **original or source implementations** used/adapted by the repository.

In this project, methods are either:

1. obtained from the **CausalDiscoveryToolbox (CDT)**, or
2. directly imported/adapted from their original source implementations.

Because of that, `source_implementations/` is part of the repository and should be read as the place where the exact upstream/original implementations used by the project are stored or referenced.

This also explains why some methods remain in **R** or **MATLAB**: the project keeps them in their original language whenever appropriate, and only standardizes the **score processing and evaluation** in Python.

That design makes it easier to preserve fidelity to the original methods while still unifying benchmarking and metrics.

---

### `bicausal/metrics/`
Contains the evaluation logic.

Files include:

- `accuracy.py`
- `auroc.py`
- `audrc.py`
- `lxcim.py`
- `drawers.py`
- `evaluators.py`

This folder is responsible for the **second stage** of the pipeline: loading stored scores and converting them into benchmark metrics, tables, and summaries.

---

### `bicausal/results/`
Stores experiment outputs and aggregate result files.

Important files include:

- `lisbon_scores.csv`
- `tuebingen_scores.csv`
- `CE_scores.csv`
- `ANLSMN_scores.csv`
- `SIM_scores.csv`
- `results.csv`
- `times.csv`
- `unimplemented_results.csv`

This folder is the main place to inspect raw scores and final benchmark summaries.

---

### `bicausal/plots/`
Stores generated figures.

This includes paper figures for:

- the Lisbon paper;
- the LxCIM paper;
- the RDMDL paper.

If you want to inspect already-generated figures without rerunning notebooks, this is the first place to look.

---

### `bicausal/table/`
Stores generated LaTeX tables.

This includes:

- Lisbon paper tables and appendices;
- LxCIM paper tables;
- RDMDL paper tables;
- timestamped auto-generated tables.

This is useful if you want the final paper-ready table outputs directly.

---

### `bicausal/storage/`
Stores small cached artifacts.

Currently this includes cache-like data used by timing/execution utilities.

---

### `dist/`
Python distribution artifacts:

- wheel (`.whl`)
- source distribution (`.tar.gz`)

Useful for local `pip` installation without rebuilding the package.

---

### `CITATION.cff`
Citation metadata for the project.

---

### `pyproject.toml`
Package configuration for `bicausal`, including Python version and package metadata.

---

## Benchmarks included

### 1. Lisbon benchmark
A real-world, multi-domain benchmark included in `bicausal/benchmarks/Lisbon/`.

It is organized by field, and the repository structure shows five field folders under `data/`, `meta/`, and `pictures/`:

- `agriculture_environment`
- `biology_health`
- `economy`
- `human_predictions`
- `science_engineering`

The Lisbon benchmark also contains explicit source-level and pair-level weighting metadata.

### 2. Tübingen benchmark
The classic cause-effect pairs benchmark, stored in raw text form together with descriptions and analysis metadata.

### 3. Synthetic benchmarks
Three synthetic families are included:

- CE-Guyon
- ANLSMN-Tagasovska
- SIM-Mooij

These are useful both for benchmarking methods and for reproducing synthetic-dataset analyses in the project.

---

## Methods included

The repository contains implementations/wrappers for a broad set of bivariate causal discovery methods, including:

- ANM
- bQCD
- CAM
- CDCI
- CDS
- CGNN
- FOM
- GPI / GPI_lx / GPIn
- HECI
- IGCI
- LCUBE
- LOCI
- NNCL
- RDMDL
- RECI
- ROCHE
- SLOPE
- SLOPPY

Some are in Python, some in R, and some in MATLAB.

This mixed-language setup is not accidental: it reflects the repository’s goal of preserving the original implementation language whenever needed while keeping **evaluation and comparison unified in Python**.

---

## Notebooks

### Reproduction notebooks for papers

There is a notebook for each paper, and these notebooks are intended to reproduce the **tables and figures contained in those papers**:

- `lisbon_paper.ipynb`
- `lxcim_paper.ipynb`
- `rdmdl_paper.ipynb`

There is also:

- `rdmdl_paper copy.ipynb`

which appears to be a working copy / auxiliary variant of the RDMDL notebook.

### Workflow notebooks

Other notebooks support the benchmarking pipeline itself:

- `run.ipynb` — run methods and generate raw score CSVs.
- `evaluate.ipynb` — load saved score files and compute metrics/tables.
- `time.ipynb` — timing/efficiency experiments.
- `utilize.ipynb` — maintenance and result-cleaning utilities.
- `add_your_own_dataset.ipynb` — guide for extending the Lisbon benchmark with a new dataset.
- `test_your_own_method.ipynb` — guide/pipeline for integrating a new method.

---

## Important notes and caveats

This section consolidates the practical notes already present in the repository.

### 1. The repository assumes honest use

This repository, library, and its functions assume honesty on the part of the programmer.

For example, a dummy function that always returns the same answer could trivially game some metrics if used dishonestly. In the same spirit, methods that require training should be made robust to variable order and entry order when appropriate.

### 2. Training-heavy methods are not the main target

In general, the repository is **not built primarily to support methods that require training**, for two main reasons noted in the project itself:

1. there is no batching framework;
2. there is no implemented framework to deal with similarity between different pairs in the Lisbon and Tübingen datasets.

These limitations could be extended in future work, but they are important to keep in mind.

### 3. Some methods were removed from the main process

The project notes that **GPLVM** and **RCC** required training and were removed from the main process, as they were not being treated here as direct cause-effect methods in the final benchmarking flow.

### 4. Methods were kept self-contained

Methods were implemented so that they would work with the original implementations and by themselves, which means that **some code repetition is expected**.

### 5. Relative paths matter

The project notes that many functions use directory paths as if they are being called from a specific working-directory layout. If you move files around or run entry points from a different working directory, you may need to adjust paths.

When using the repository for the first time, it is best to preserve the original structure and run the provided notebooks/scripts before customizing paths.

### 6. `run_ce` flips negative labels

A specific implementation note in the repository is that `run_ce` inverts vectors for negative labels so that all labels become positive before scoring.

### 7. Timer recalculation may require manual cache cleanup

The repository notes that recalling/recomputing timers may require deleting entries manually in the storage/cache area.

### 8. CDCI variant used

The repository notes that **CDCI with CTV** was chosen because it performed best overall in the published paper associated with the project.

### 9. GPLVM implementation note

The repository notes that the **generalized GPLVM** variant was chosen because the authors argued it was better for Tübingen (and therefore for real-world data, which is a main focus here), and that a compatibility change was made in the `optimization_step` function to support a newer TensorFlow version.

---

## Outputs produced by the repository

The repository produces and/or stores four main kinds of outputs:

### Raw per-example scores
Saved in files ending with `_scores.csv`.

These are the most important outputs for reproducibility.

### Aggregate benchmark results
Saved in files such as:

- `results.csv`
- `unimplemented_results.csv`

### Timing results
Saved in:

- `times.csv`

### Publication artifacts
Saved in:

- `plots/` for figures
- `table/` for LaTeX tables

---

## Citation

If you use this repository, please check `CITATION.cff`.

At the time of writing, the citation metadata points to work by **Tiago Brogueira** and **Mário A. T. Figueiredo**, with the title:

> *The Lisbon benchmark: a new real-world multi-domain bivariate causal discovery dataset*

If you use the code, benchmark, or results in academic work, citing the repository and the associated paper(s) is strongly recommended.

---

## License

This repository is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Final practical advice

If your goal is **paper reproduction**, **clone the repository**.

If your goal is **library-style usage**, **install with pip**.

If your goal is **extending the benchmark**, start with:

- `add_your_own_dataset.ipynb`
- `test_your_own_method.ipynb`
- `evaluate.ipynb`

And if your goal is to understand the philosophy of the project, keep this in mind:

> **methods are run first, scores are saved first, and only then are metrics applied.**

That separation is the backbone of the repository’s reproducibility model.
