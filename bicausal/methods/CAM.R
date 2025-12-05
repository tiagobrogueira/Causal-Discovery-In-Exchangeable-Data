source("bicausal/methods/source_implementations/CAM/R/computeScoreMat.R")
source("bicausal/methods/source_implementations/CAM/R/updateScoreMat.R")
source("bicausal/methods/source_implementations/CAM/R/computeScoreMatParallel.R")
source("bicausal/methods/source_implementations/CAM/R/CAM.R")
source("bicausal/methods/source_implementations/CAM/R/train_gam.R")

library(mgcv)

cam <- function(data, ...) {
  
  # ---- Run CAM on the bivariate data ----
  res <- tryCatch(
  {
      if (ncol(data) != 2) {
        return(NaN)
      }
  X <- as.matrix(data)
    cam_res <- CAM(X, variableSel = FALSE, pruning = FALSE, ...)
  
  Adj <- as.matrix(cam_res$Adj)
  
  score=NaN
  # Determine direction
  if (Adj[1,2] == 1 && Adj[2,1] == 0) {
    direction <- "X->Y"
    score <- cam_res$Score
  } else if (Adj[2,1] == 1 && Adj[1,2] == 0) {
    direction <- "Y->X"
    score <- -cam_res$Score
  } else {
    direction <- "undetermined"
  }
  },
  error = function(e) {
    print("error in CAM:")
    print(e)
    return(NaN)
  }
)

  if (is.nan(res)) {
    return(NaN)
  }

  
  # Return confidence score
  return(score)
}
