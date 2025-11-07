# Copyright (c) 2018  Alexander Marx  [amarx@mpi-inf.mpg.de]
# All rights reserved.  See the file COPYING for license terms. 

# Corresponding Paper
This code is supplementary material for
@article{marx:18:slope,
	author="Marx, Alexander and Vreeken, Jilles",
	title="Telling cause from effect by local and global regression",
	journal="Knowledge and Information Systems",
	year="2018",
	issn="0219-3116",
	doi="10.1007/s10115-018-1286-7",
	url="https://doi.org/10.1007/s10115-018-1286-7"
}
and
@inproceedings{marx:17:slope,
	author="Marx, Alexander and Vreeken, Jilles",
	title="Telling cause from effect by MDL-based local and global regression",
	booktitle="Proceedings of the IEEE International Conference on Data Mining (ICDM)",
	pages="307--316",
	publisher="IEEE",
	year="2017"
}


**Installation:**
To run this code you need to have R installed.

**Run:**
To run the Slope algorithm, you need to enter this folder and load the Slope source file with

source("Slope.R")

which implicitly also loads the utilities file by

source("utilities.R")

Slope can than be executed by passing a data frame 't' with two columns (column 1 is X and column 2 is Y) by applying:

Slope(t, alpha=a)

Where a is the threshold for the significance test. To apply the extended version SlopeM it is necessary to set the boolean variable "mixedFunctions" to true.

To run Sloper instead of Slope, simply set the parameter mixedFunctions of Slope(..) to True.

**Data:**
The data folder contains the newly obtained octet binary pairs, the simulated data and the Tübingen pairs that we used in the paper.

**Results:**
The results folder contains all the (slow to compute) results of the competing methods as used in the paper. All other results can be obtained by running the corresponding script as described below.

**Experiments:**
The experiments containing Slope, ICGI and Resit can be performed with this package. 
* test_synthetic_data.R: implements several tests functions and stores the results
* apply_synthetic_Tests.R: applies the tests implemented in 'test_synthetic_data.R'
* test_octet_data.R: can be used to run tests on the octet data sets
* test_benchmark.R: can be used to apply Slope, IGCI and Resit to the Tuebingen benchmark pairs
* decision_rate_plot.R: implements a plotting scheme to plot the decision rates