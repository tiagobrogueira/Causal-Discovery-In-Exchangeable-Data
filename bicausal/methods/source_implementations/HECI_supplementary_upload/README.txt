This folder contains the implementation as well as the experiments for
"Heteroscedastic Noise Based Causal Inference"

We provide a function that predicts using the HECI algorithm for a given bivariate pair X and Y in HECI.py.
The dependencies to run HECI is just numpy.
To run the full experiments, use the script Benchmark.py in the conda environment provided in "environment.yml".
For the full experiments, include competitor_synthetic.R and competitor_synthetic.R as well.
The new heteroscedastic cause effect pairs are available in the Data folder.
The structure of this project is as follows:

HECI.py  		--- contains the code for HECI
NNCL.py 		--- reimplementation of NNCL (Wang, B. and Qing Z. Causal network learning with non-invertible functional relationships. Computational Statistics & Data Analysis 156 (2021): 107141.)
Benchmark.py	--- executes the experiments from the paper for HEC and NNCL on synthetic and Tuebingen data
SimulateData.py --- generation of our heteroscedastic synthetic data
competitor_synthetic.R		--- runs experiments on synthetic data for all other competitors
competitor_tuebingen.R		--- runs experiments on Tuebingen data for all other competitors
baselines.R		--- contains the baseline methods
bqcd.R          --- contains code for BQCD (Tagasovska, N., Chavez-Demoulin, V., & Vatter, T. Distinguishing cause from effect using quantiles: Bivariate quantile causal discovery.)
read_tueb.R     --- utility to read Tuebingen
environment.yml --- conda environment for HEC python implementation

Results			--- folder that contains the result files that are generated when executing all benchmarks
Data			--- contains the benchmark data sets
baselines       --- further R code for Sloppy, RESIT