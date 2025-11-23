source("bicausal/methods/source_implementations/CAM/R/computeScoreMat.R")
source("bicausal/methods/source_implementations/CAM/R/updateScoreMat.R")
source("bicausal/methods/source_implementations/CAM/R/computeScoreMatParallel.R")
source("bicausal/methods/source_implementations/CAM/R/CAM.R")
source("bicausal/methods/source_implementations/CAM/R/train_gam.R")

library(mgcv)
library(jsonlite)

get_max_points <- function(storage_path = NULL) {
  if (is.null(storage_path)) {
    storage_path <- "storage/max_points_cache.json"
  }
  
  # If file does not exist → return NULL (same as Python)
  if (!file.exists(storage_path)) {
    return(NULL)
  }
  
  # Read JSON cache
  cache <- tryCatch(
    jsonlite::fromJSON(storage_path),
    error = function(e) return(NULL)
  )
  
  # Return method’s stored value, or NULL
  if (is.null(cache[["cam"]])) return(NULL)
  
  return(cache[["cam"]])
}

cam <- function(data, ...) {
  
  max_points <- get_max_points()
  if (is.null(max_points)) {
    n_points <- nrow(data)
  } else {
    n_points <- min(max_points, nrow(data))
  }
  

  
  # ---- Run CAM on the bivariate data ----
  res <- tryCatch(
  {
      sampled_data <- data[sample(seq_len(nrow(data)), n_points), ]
      if (ncol(sampled_data) != 2) {
        return(NaN)
      }
  X <- as.matrix(sampled_data)
    cam_res <- CAM(X, variableSel = FALSE, pruning = FALSE, ...)
      cam_res <- res$cam_res
  
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
    return(NaN)
  }
)

  if (is.nan(res)) {
    return(NaN)
  }

  print(score)
  
  # Return confidence score
  return(score)
}
