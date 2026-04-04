# Causal Discovery in Exchangeable Data

A reproducible toolbox and benchmark suite for **bivariate causal discovery on exchangeable data**.

This repository serves two roles at once:

1. a **research repository** containing datasets, methods, results, figures, tables, and notebooks;
2. a **Python package** (`bicausal`) that can be installed and used as a normal library.

Useful links:

- **Repository:** https://github.com/tiagobrogueira/Causal-Discovery-In-Exchangeable-Data
- **PyPI package:** https://pypi.org/project/bicausal/
- **Python requirement:** `>= 3.8`

It also contains three **original research contributions** beyond the repository infrastructure itself:

- **The Lisbon benchmark** — a new real-world multi-domain bivariate causal discovery benchmark.
- **RDMDL** — a new bivariate causal discovery method based on rate-distortion MDL and information dimension.
- **LxCIM** — a new rank-based binary-classifier performance metric invariant to local exchange of classes.

---

## Papers stemming from this repository

At the moment, no paper URL is public yet, so these are marked as **to come**.

### 1. *Bivariate Causal Discovery Using Rate-Distortion MDL: An Information Dimension Approach*
- **Status:** accepted at **CLeaR 2026**
- **Conference:** https://www.cclear.cc/2026
- **Paper URL:** to come
- **Notebook:** `bicausal/rdmdl_paper.ipynb`
- **Main new contribution:** **RDMDL**

### 2. *The Lisbon benchmark: a new real-world multi-domain bivariate causal discovery dataset*
- **Status:** not yet submitted
- **Paper URL:** to come
- **Notebook:** `bicausal/lisbon_paper.ipynb`
- **Main new contribution:** **the Lisbon benchmark**

### 3. *LxCIM: a new rank-based binary classifier performance metric invariant to local exchange of classes*
- **Status:** not yet submitted
- **Paper URL:** to come
- **Notebook:** `bicausal/lxcim_paper.ipynb`
- **Main new contribution:** **LxCIM**

> There is **one notebook per paper**, and each paper notebook is intended to reproduce the figures and tables contained in that paper.

---

## What is new in this repository?

A lot of the repository is benchmarking infrastructure or wrappers around existing causal discovery methods. The **new scientific contributions** are:

- **`bicausal/benchmarks/Lisbon/`** → the **Lisbon benchmark**.
- **`bicausal/methods/RDMDL.py`** → the **RDMDL method**.
- **`bicausal/metrics/lxcim.py`** → the **LxCIM metric**.

Everything else should mostly be read as one of the following:

- benchmark infrastructure;
- evaluation and plotting utilities;
- wrappers around existing methods;
- exact or adapted source implementations brought into the repository for reproducibility.

This distinction is important throughout the repository and is highlighted again in the file-structure section below.

---

## Table of contents

- [Installation and usage](#installation-and-usage)
  - [Option 1 — clone the repository](#option-1--clone-the-repository)
  - [Option 2 — install from PyPI](#option-2--install-from-pypi)
  - [Using `bicausal` as a normal library](#using-bicausal-as-a-normal-library)
- [Core experimental philosophy](#core-experimental-philosophy)
- [Benchmarking conventions](#benchmarking-conventions)
- [Repository structure](#repository-structure)
- [Folder-by-folder guide](#folder-by-folder-guide)
- [Important notes and caveats](#important-notes-and-caveats)
- [Outputs produced by the repository](#outputs-produced-by-the-repository)
- [Citation](#citation)
- [License](#license)

---

## Installation and usage

There are two main ways to use the project.

### Option 1 — clone the repository

This is the best option if you want the **full research artifact**, including:

- all notebooks;
- benchmark data in repository layout;
- precomputed result files;
- generated plots and LaTeX tables;
- R and MATLAB method files;
- the exact folder structure used in the experiments and papers.

```bash
git clone https://github.com/tiagobrogueira/Causal-Discovery-In-Exchangeable-Data.git
cd Causal-Discovery-In-Exchangeable-Data
pip install -e .
```

Use this mode if your goal is:

- full reproducibility;
- reproducing papers;
- running the notebooks as provided;
- inspecting all data, scores, tables, and figures together.

### Option 2 — install from PyPI

If your goal is to use the package as a normal Python library, installation is simply:

```bash
pip install bicausal
```

That installs the `bicausal` package.

Use this mode if your goal is:

- calling methods directly on your own pairs;
- plugging a method into the repository runners/evaluators from Python code;
- using the evaluation utilities without cloning the full repository.

For the **full benchmark assets, notebooks, precomputed results, and paper artifacts**, cloning the repository is still the safest and most complete route.

### Using `bicausal` as a normal library

The package is intentionally organized around **submodules**, so in normal usage you typically import from:

- `bicausal.methods.*`
- `bicausal.helpers.runners`
- `bicausal.metrics.evaluators`

rather than relying on top-level imports.

### Example 1 — score a single pair with a method

```python
import numpy as np
from bicausal.methods.RDMDL import rdmdl

x = np.asarray(x).reshape(-1, 1)
y = np.asarray(y).reshape(-1, 1)

score = rdmdl([x, y])
print(score)
```

The general convention is that a method receives:

```python
func([x, y], *args, **kwargs)
```

and returns a scalar score.

### Example 2 — run a method on a benchmark and save raw scores

```python
from bicausal.methods.RDMDL import rdmdl
from bicausal.helpers.runners import run_tuebingen

run_tuebingen(rdmdl)
```

This writes one score per example into the corresponding score file, here typically:

```text
results/tuebingen_scores.csv
```

### Example 3 — evaluate previously saved scores

```python
from bicausal.metrics.evaluators import evaluate_tuebingen

evaluate_tuebingen(metrics=["LxCIM", "accuracy", "AUROC", "AUDRC"])
```

This loads the saved score CSV, applies the selected metrics, and updates the aggregate results file.

### Example 4 — use your own method with the benchmark infrastructure

```python
import numpy as np
from bicausal.helpers.runners import run_lisbon


def my_method(d):
    x, y = d
    x = np.asarray(x)
    y = np.asarray(y)

    # return a scalar score for the ordered pair [X, Y]
    return float(np.mean(x) - np.mean(y))


run_lisbon(my_method)
```

This is one of the core design goals of the package: a method only needs to obey the repository’s simple callable interface, and then it can be benchmarked and evaluated through the same pipeline as the built-in methods.

If you are using the benchmark runners from a plain PyPI install, it is often best to pass explicit dataset/result paths or to work inside a cloned repository layout.

### A practical note on dependencies

`bicausal` can be installed with `pip install bicausal`, but some specific methods still depend on the ecosystems they originally came from or on external libraries used by their wrappers. In particular:

- some methods are wrapped from **CausalDiscoveryToolbox (CDT)**;
- some methods remain in **R**;
- some methods remain in **MATLAB**.

This is intentional: the repository preserves original implementations whenever possible, while standardizing **score storage**, **metric computation**, and **benchmark evaluation** in Python.

---

## Core experimental philosophy

The central idea of the repository is a strict separation between **running methods** and **evaluating methods**.

### Step 1 — run a method and save raw per-example scores

Whenever a method is run on a benchmark, the repository first stores the **score of each individual example** in a CSV file ending in `_scores.csv`, such as:

- `tuebingen_scores.csv`
- `lisbon_scores.csv`
- `CE_scores.csv`
- `ANLSMN_scores.csv`
- `SIM_scores.csv`

These files are the raw experimental outputs.

### Step 2 — load those CSV files and apply metrics

Only after the per-example scores are stored do we load them and apply metrics such as:

- `accuracy`
- `AUROC`
- `AUDRC`
- `LxCIM`

### Why this separation exists

This design is deliberate and serves **reproducibility**.

It means that:

- expensive method runs are separated from evaluation;
- methods implemented in different languages can still be compared uniformly;
- metrics can be recomputed later without rerunning the underlying methods;
- tables and figures can be regenerated directly from saved scores;
- the exact raw scores used in a paper remain inspectable and auditable.

This is one of the most important ideas in the repository.

---

## Benchmarking conventions

### Ordered-pair convention

Throughout the evaluation pipeline, examples are treated as ordered pairs **`[X, Y]`**, with the intended convention:

- **`X` = cause**
- **`Y` = effect**

### Synthetic datasets are normalized on load

For the synthetic benchmarks, the repository reorients the data **before** the method is evaluated so that the method always receives the example in the form **cause → effect**, i.e. **`X` as cause** and **`Y` as effect**.

Concretely:

- in **CE-Guyon**, pairs are swapped when the target label indicates the opposite direction;
- in **ANLSMN-Tagasovska**, the pair is reoriented using the ground-truth direction file;
- in **SIM-Mooij**, the cause/effect variable blocks are extracted directly from `pairmeta.txt`.

So the important practical rule is:

> **Synthetic examples are evaluated after orientation correction, always with `X` as cause and `Y` as effect.**

This is easy to miss, so it is worth stating explicitly.

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
│   │   │   ├── lisbon_analysers.py
│   │   │   ├── lisbon_utils.py
│   │   │   └── source_specs.xlsx
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
│   ├── run.ipynb
│   ├── run.R
│   ├── run.m
│   ├── test_your_own_method.ipynb
│   ├── time.ipynb
│   └── utilize.ipynb
├── dist/
│   ├── bicausal-*.whl
│   └── bicausal-*.tar.gz
├── CITATION.cff
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Folder-by-folder guide

### `bicausal/`

This is the main package directory.

It contains:

- the benchmark datasets;
- method wrappers and source-backed implementations;
- metric definitions and evaluation logic;
- saved results;
- generated plots and LaTeX tables;
- notebooks and entry points used throughout the project.

This folder is the real center of the repository.

---

### `bicausal/benchmarks/`

This folder contains the datasets used by the repository.

#### `bicausal/benchmarks/Lisbon/`  **← new contribution**

This folder contains the **Lisbon benchmark**, which is one of the main original contributions of the project.

Its purpose is to provide a **new real-world multi-domain benchmark** for bivariate causal discovery.

It is split into:

- `data/` — the actual benchmark pair files.
- `meta/` — metadata associated with the benchmark pairs and sources.
- `pictures/` — source images and visual material associated with the benchmark sources.
- `source_specs.xlsx` — source-level specifications and weighting-related information.
- `field_stats.xlsx` — field-level statistics and summaries.
- `lisbon_utils.py` — utilities for loading Lisbon metadata and weights.
- `lisbon_analysers.py` — analysis helpers specific to the Lisbon benchmark.

Inside `data/`, the benchmark is organized into five domains:

- `agriculture_environment`
- `biology_health`
- `economy`
- `human_predictions`
- `science_engineering`

So whenever you see `bicausal/benchmarks/Lisbon/`, read it as:

> **new dataset contribution + its metadata + its analysis utilities**

#### `bicausal/benchmarks/Tuebingen/`

This folder contains the **Tübingen cause-effect benchmark**, which is used here as a reference real-world benchmark.

It includes:

- the pair data files (`pairXXXX.txt`);
- textual descriptions for each pair (`pairXXXX_des.txt`);
- accompanying benchmark documentation (`README`, `README_polished_may18.tab`);
- an analysis spreadsheet (`TuebingenAnalysis.xlsx`).

#### `bicausal/benchmarks/synthetic/`

This folder contains the synthetic benchmark suites used in the experiments:

- `ANLSMN-Tagasovska/`
- `CE-Guyon/`
- `SIM-Mooij/`

These are not just stored here for convenience: they are wired into the benchmark runners and evaluation pipeline.

A very important convention for this folder is that the synthetic examples are reoriented on load so that methods are evaluated on **`X` as cause** and **`Y` as effect**.

---

### `bicausal/helpers/`

This folder contains the orchestration logic for running experiments, processing saved scores, timing methods, and handling utility functionality.

Important files here include:

- `runners.py` — the main Python benchmark runners:
  - `run_tuebingen(...)`
  - `run_lisbon(...)`
  - `run_ce(...)`
  - `run_anlsmn(...)`
  - `run_sim(...)`
- `runners.R` — R-side running utilities for methods that remain in R.
- `run_tuebingen.m`, `run_ce.m`, `run_anlsmn.m`, `run_sim.m` — MATLAB-side helpers for methods that remain in MATLAB.
- `processers.py` — the score-processing stage that reads `_scores.csv` files and aligns them into vectors + weights for metric computation.
- `timers.py` — utilities for time measurements and timing-related caching.
- `utils.py` — general dataset and helper utilities.
- `namemap.py` — canonical naming and method-name normalization.
- `meanwhile.py` and `extra/` — auxiliary helper code.

This folder is where the repository’s reproducibility model is operationalized:

1. run a method;
2. save per-example scores;
3. later process those scores for evaluation.

---

### `bicausal/methods/`

This folder contains the causal discovery methods used in the repository.

It intentionally mixes multiple languages:

- Python methods (`.py`)
- R methods (`.R`)
- MATLAB methods (`.m`)

That mixed-language setup is **by design**, not an accident.

The idea is to preserve methods in the language in which they were originally implemented whenever that is the cleanest or most faithful option, and then unify **benchmark execution**, **score storage**, and **metric evaluation** around a common pipeline.

Methods present here include:

- `ANM.py`
- `BQCD.R`
- `CAM.R`
- `CDCI.py`
- `CDS.py`
- `CGNN.py`
- `FOM.py`
- `GPI.m`
- `GPI_lx.m`
- `GPIn.m`
- `HECI.py`
- `IGCI.py`
- `LCUBE.py`
- `LOCI.py`
- `NNCL.py`
- `RDMDL.py`  **← new contribution**
- `RECI.py`
- `ROCHE.py`
- `SLOPE.R`
- `SLOPPY.R`

#### `bicausal/methods/RDMDL.py`  **← new contribution**

This file contains **RDMDL**, one of the repository’s original scientific contributions and the method associated with the paper:

> *Bivariate Causal Discovery Using Rate-Distortion MDL: An Information Dimension Approach*

#### `bicausal/methods/source_implementations/`

This is a very important folder.

Methods in the repository are either:

1. obtained through **CausalDiscoveryToolbox (CDT)** wrappers, or
2. brought in directly from their source/original implementations.

Because of that, `source_implementations/` contains the source-backed material used by the project, such as:

- `CAM`
- `CDCI_main`
- `FOM_main`
- `GPI`
- `HECI_supplementary_upload`
- `LCube_main`
- `ROCHE_main`
- `bqcd`
- `loci_main`
- `slope-20181208`
- `sloppy-v20190523/ Sloppy`

This folder should be read as the place where the repository keeps track of the exact imported or adapted source implementations it uses.

This also explains an important design choice:

> Methods originally implemented in **MATLAB** or **R** are intentionally kept in those languages whenever appropriate, and only the **processing of their scores** and the **metric computation** are unified in Python.

That decision improves reproducibility and preserves fidelity to the original implementations.

---

### `bicausal/metrics/`

This folder contains the evaluation metrics and the second stage of the benchmarking pipeline.

Files include:

- `accuracy.py`
- `auroc.py`
- `audrc.py`
- `lxcim.py`  **← new contribution**
- `evaluators.py`
- `drawers.py`

#### `bicausal/metrics/lxcim.py`  **← new contribution**

This file contains **LxCIM**, the new rank-based binary-classifier performance metric introduced in the repository and associated with the paper:

> *LxCIM: a new rank-based binary classifier performance metric invariant to local exchange of classes*

#### `bicausal/metrics/evaluators.py`

This file contains the main evaluation entry points, such as:

- `evaluate_tuebingen(...)`
- `evaluate_lisbon(...)`
- `evaluate_synthetic(...)`
- `construct_table(...)`

This is the stage that loads stored score CSVs and turns them into aggregate metrics, result tables, and publication-ready outputs.

---

### `bicausal/results/`

This folder stores the main experiment outputs.

Typical contents include:

- `tuebingen_scores.csv`
- `lisbon_scores.csv`
- `CE_scores.csv`
- `ANLSMN_scores.csv`
- `SIM_scores.csv`
- `results.csv`
- `times.csv`
- `unimplemented_results.csv`

This is one of the most important folders for reproducibility.

The files ending in `_scores.csv` are the raw per-example method outputs.

Everything downstream — metrics, tables, plots, summaries — is built from those saved scores.

---

### `bicausal/plots/`

This folder stores generated plots and figures.

In practice, this is where you look for already-generated publication figures, including figures produced by the paper notebooks.

---

### `bicausal/table/`

This folder stores generated LaTeX tables.

This is where the repository writes publication-oriented tabular outputs derived from the evaluation stage.

---

### `bicausal/storage/`

This folder stores small cached or timing-related artifacts.

It is mostly relevant for timing experiments and auxiliary cached state.

---

### Notebook and entry-point files inside `bicausal/`

These files are especially important for users.

#### Paper reproduction notebooks

- `lisbon_paper.ipynb` — reproduces the figures/tables for the Lisbon benchmark paper.
- `lxcim_paper.ipynb` — reproduces the figures/tables for the LxCIM paper.
- `rdmdl_paper.ipynb` — reproduces the figures/tables for the RDMDL paper.

#### Workflow notebooks

- `run.ipynb` — run methods and generate `_scores.csv` files.
- `evaluate.ipynb` — evaluate saved scores and aggregate results.
- `time.ipynb` — timing experiments.
- `utilize.ipynb` — utility and maintenance workflows.
- `add_your_own_dataset.ipynb` — how to add a new dataset.
- `test_your_own_method.ipynb` — how to plug in and benchmark a new method.

#### Cross-language entry points

- `run.R` — entry point for R-based workflows.
- `run.m` — entry point for MATLAB-based workflows.

---

### `dist/`

This folder contains local build artifacts for the Python package, such as:

- wheel files (`.whl`)
- source distributions (`.tar.gz`)

It is useful mostly for packaging and local distribution.

---

### `pyproject.toml`

This file contains the Python package metadata and build configuration for `bicausal`.

---

### `CITATION.cff`

This file contains citation metadata for the repository.

---

## Important notes and caveats

The points below consolidate the main practical notes underlying the project.

### 1. The repository assumes honest use

The repository assumes honest use by the experimenter.

As with any benchmarking framework, a deliberately adversarial or degenerate method could game certain metrics if used dishonestly. The toolkit is built for fair evaluation, not for defending against intentionally pathological usage.

### 2. Training-heavy methods are not the main focus of the current pipeline

The repository is not primarily designed around methods that require substantial training.

Two practical reasons are explicitly relevant here:

- there is no batching framework;
- there is no implemented framework for exploiting similarity between different pairs across Lisbon and Tübingen.

So the repository is especially natural for pairwise scoring methods, direct wrappers, and source-preserving benchmark evaluation.

### 3. Some methods were excluded from the main benchmark flow

The project notes that **GPLVM** and **RCC** were removed from the main process because of their training-related nature and how they fit into the final benchmark design.

### 4. Methods are kept self-contained

Methods were implemented or wrapped so they can work as independently as possible.

A consequence is that some code repetition is expected across method files.

### 5. Relative paths matter

Some functions assume the original repository layout and relative-path structure.

If you clone the repository and keep its structure intact, things are much easier. If you move files around or run code from different working directories, you may need to pass explicit paths or adjust defaults.

### 6. Timing reruns may require manual cache cleanup

If you are recomputing timing experiments, you may need to manually clear entries in the timing/cache storage area.

### 7. The CDCI variant used here is the CTV variant

The repository uses the **CDCI with CTV** variant.

### 8. A generalized GPLVM variant was considered in the broader project context

The notes associated with the project mention the generalized GPLVM variant as preferable for Tübingen/real-world data, together with a compatibility change in `optimization_step` for newer TensorFlow versions.

### 9. Mixed-language methods are intentional

If you see R or MATLAB files in the methods folder, that is not technical debt to be “cleaned up” away from the repository’s design.

It is a deliberate reproducibility choice:

- keep the method in its original ecosystem when appropriate;
- run it there if needed;
- save raw scores;
- evaluate the saved scores in Python together with everything else.

---

## Outputs produced by the repository

The repository produces four main kinds of outputs.

### 1. Raw per-example scores

These are the files ending in `_scores.csv`.

They are the primary reproducibility artifacts.

### 2. Aggregate benchmark results

These are typically stored in files such as:

- `results.csv`
- `unimplemented_results.csv`

### 3. Timing results

These are stored in:

- `times.csv`

### 4. Publication artifacts

These are stored in:

- `plots/` for figures
- `table/` for LaTeX tables

---

## Citation

If you use this repository, please cite the repository itself and, when available, the relevant paper(s).

In practice, the most natural citation targets will usually be:

- the repository;
- the Lisbon benchmark paper;
- the RDMDL paper;
- the LxCIM paper.

Please also check `CITATION.cff`.

---

## License

This repository is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Final summary

If you want the **full research repository**, clone it.

If you want the **Python package**, install:

```bash
pip install bicausal
```

If you want to understand the repository in one sentence, it is this:

> **Run methods first, save every per-example score first, and only then compute metrics from the saved score files.**

That separation is the backbone of the project’s reproducibility philosophy.
