source("bicausal/methods/source_implementations/bqcd/bqcd.R")

#Maximum number of points according to timed experiments
max_points <- 1000


bQCDR <- function(data, ...) {
  # Determine number of points to sample
  n_points <- min(max_points, nrow(data))

  # Randomly sample indices
  sampled_indices <- sample(seq_len(nrow(data)), n_points)
  
  # Subset the data
  sampled_data <- data[sampled_indices, ]
  
  # Run the QCD_wrap function
  results <- QCD_wrap(sampled_data[,1], sampled_data[,2], "QCD_qnn")
  
  return(results$eps)
}