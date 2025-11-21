## This is the README for the Thesis Causal Discovery in Exchangeable Data

under construction

run:
pip install -e .
after clonning the repo!!!

two options:
-clone repo: more flexible but longer.

-use pip install bicausal.

You can add more fields and more examples in the lisbon dataset!
In order to do so,...


run_ce inverts the vectors for negative labels, thus making sure all labels are positive.



NOTE: This repository, library and its functions assume honesty in the part of the programmers. The function my_method(d): return 1, would give an accuracy, auroc, alameda and audrc of 100%. similarly, and more sublty, all methods which require training must be ensured to be random to the order or the variables, by randomzing them at entry time. 

NOTE: In general, this repository is not built to support methods which require training due to two main reasons: 1) No batching. 2) No implemented framework to deal with similarity between different pairs in the Lisbon and Tuebingen datasets. However, these are changes which could be coded.

NOTE: All functions (apart from the R) have the dirs called as if being called inside "bicausal/".