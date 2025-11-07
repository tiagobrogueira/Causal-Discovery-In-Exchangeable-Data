source("bicausal/methods/source_implementations/slope-20181208/Slope.R")
source("bicausal/methods/source_implementations/slope-20181208/utilities.R")

slope <- function(data, ...) {
  results <- Slope(data, ...)    # not Slope(data[,1], data[,2], ...)
  return(- results$eps)
}