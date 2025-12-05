source("bicausal/methods/source_implementations/sloppy-v20190523/Sloppy/Sloppy.R")
source("bicausal/methods/source_implementations/sloppy-v20190523/Sloppy/utilities.R")

sloppy <- function(data, ...) {
  out <- tryCatch(
    {
      results <- Sloppy(data, ...)
      return(-results$epsilon)   # Sloppy stores epsilon as $epsilon, not $eps
    },
    error = function(e) {
      return(NaN)
    }
  )
  
}