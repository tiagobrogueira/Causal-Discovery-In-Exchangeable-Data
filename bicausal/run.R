source("bicausal/helpers/runners.R")

#NOTES:
# 1) Always run from root (bicausal/) directory. (Note the imports)
# 2) Timed experiments can be loaded using Python (similar to other methods).
# 3) Comment out the functions you do not want to run.

test_file="bicausal/benchmarks/Lisbon/data/economy/taxi_fare_prediction/distance_traveled_fare.txt"


source("bicausal/methods/CAM.R")
#benchmark_function(cam,test_file = test_file)
#run_tuebingen(cam)
#run_lisbon(cam,overwrite = TRUE)
#run_ce(cam)
#run_anlsmn(cam,overwrite=TRUE)
#run_sim(cam)

source("bicausal/methods/BQCD.R")
#run_tuebingen(bqcd)
#run_lisbon(bqcd)
#run_ce(bqcd)
#run_anlsmn(bqcd, overwrite=TRUE)
#run_sim(bqcd)

source("bicausal/methods/SLOPE.R")
#run_tuebingen(slope)
#run_lisbon(slope)
run_ce(slope)
run_anlsmn(slope,overwrite = TRUE)
run_sim(slope)

source("bicausal/methods/SLOPPY.R")
#run_tuebingen(sloppy)
#run_lisbon(sloppy)
run_ce(sloppy)
run_anlsmn(sloppy,overwrite = TRUE)
run_sim(sloppy)