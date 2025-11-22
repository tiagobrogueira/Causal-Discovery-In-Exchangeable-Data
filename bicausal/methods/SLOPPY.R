source("bicausal/methods/source_implementations/sloppy-v20190523/Sloppy/Sloppy.R")
source("bicausal/methods/source_implementations/sloppy-v20190523/Sloppy/utilities.R")

sloppy <- function(data, ...) {
  results <- Sloppy(data, ...)    # not Slope(data[,1], data[,2], ...)
  return(- results$eps)
}