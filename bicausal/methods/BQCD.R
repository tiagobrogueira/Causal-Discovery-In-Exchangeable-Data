source("bicausal/methods/source_implementations/bqcd/bqcd.R")
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
  if (is.null(cache[["bqcd"]])) return(NULL)
  
  return(cache[["bqcd"]])
}


bqcd <- function(data, ...) {
  cat("=== Starting bqcd ===\n")
  
  # Determine number of points to sample, if needed
  max_points <- get_max_points()
  if (is.null(max_points)) {
    n_points <- nrow(data)
  } else {
    n_points <- min(max_points, nrow(data))
  }
  sampled_indices <- sample(seq_len(nrow(data)), n_points)
  sampled_data <- data[sampled_indices, ]

  X <- as.numeric(sampled_data[, 1])
  Y <- as.numeric(sampled_data[, 2])
  
  results <- QCD_wrap(X, Y)
  
  return(results$eps)
}
