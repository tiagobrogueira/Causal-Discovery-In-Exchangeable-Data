source("bicausal/helpers/runners.R")
source("bicausal/methods/BQCD.R")
source("bicausal/methods/SLOPE.R")
source("bicausal/methods/SLOPPY.R")
source("bicausal/methods/CAM.R")

#NOTES:
# 1) Always run from root (bicausal/) directory. (Note the imports)
# 2) Timed experiments can be loaded using Python (similar to other methods).
# 3) Comment out the functions you do not want to run.

test_file="bicausal/benchmarks/Lisbon/data/economy/taxi_fare_prediction/distance_traveled_fare.txt"



#benchmark_function(cam,test_file = test_file) #FALTA TRATAR DO BENCHMARK!!!!!
#run_tuebingen(cam)
run_lisbon(cam)
