library(jsonlite)
library(dplyr)
library(readr)
library(stringr)
library(purrr)
library(tibble)

# ------------------------
# Helper functions
# ------------------------

serialize_params <- function(args = list(), kwargs = list()) {
  # Convert args/kwargs into a clean, minimal JSON string for CSV storage
  if (length(args) == 0 && length(kwargs) == 0) return("")
  
  tryCatch({
    param_list <- list()
    if (length(args) > 0) param_list$args <- args
    if (length(kwargs) > 0) param_list$kwargs <- kwargs
    toJSON(param_list, auto_unbox = TRUE, pretty = FALSE)
  }, error = function(e) {
    parts <- c()
    if (length(args) > 0) parts <- c(parts, paste0("args=", toString(args)))
    if (length(kwargs) > 0) parts <- c(parts, paste0("kwargs=", toString(kwargs)))
    paste(parts, collapse = ", ")
  })
}

normalize_str <- function(value) {
  # Normalize values read from CSV so that NA or NULL become ""
  if (is.null(value) || is.na(value)) return("")
  as.character(value)
}

getTuebingen <- function(read_dir = "bicausal/datasets/Tuebingen") {
  # Reads the Tübingen dataset pairs and their weights from the given directory.
  # Returns:
  #   list(
  #     data = list(list(x, y), list(x, y), ...),
  #     weights = numeric_vector
  #   )
  
  pairmeta_file <- file.path(read_dir, "pairmeta.txt")
  pair_prefix <- "pair"
  
  if (!file.exists(pairmeta_file)) {
    stop("❌ pairmeta.txt not found in ", read_dir)
  }
  
  meta_lines <- readLines(pairmeta_file)
  data_list <- list()
  weights <- numeric(0)
  
  for (line in meta_lines) {
    if (trimws(line) == "") next
    entries <- strsplit(line, "\\s+")[[1]]
    
    pair_number <- sprintf("%04d", as.integer(entries[1]))
    x_start <- as.integer(entries[2]) - 1
    x_end   <- as.integer(entries[3])
    y_start <- as.integer(entries[4]) - 1
    y_end   <- as.integer(entries[5])
    weight  <- as.numeric(entries[6])
    
    pair_filename <- file.path(read_dir, paste0(pair_prefix, pair_number, ".txt"))
    
    if (!file.exists(pair_filename)) {
      message("⚠️ Missing ", pair_filename, ", skipping.")
      next
    }
    
    arr <- tryCatch({
      as.matrix(read.table(pair_filename))
    }, error = function(e) {
      message("⚠️ Error reading ", pair_filename, ": ", e$message)
      return(NULL)
    })
    
    if (is.null(arr)) next
    
    # Python indexing: x_start is 0-based, R is 1-based
    x <- arr[, (x_start + 1):x_end, drop = FALSE]
    y <- arr[, (y_start + 1):y_end, drop = FALSE]
    
    data_list[[length(data_list) + 1]] <- list(x, y)
    weights <- c(weights, weight)
  }
  
  list(data = data_list, weights = weights)
}

# ------------------------
# Core function: run_tuebingen
# ------------------------

run_tuebingen <- function(func,
                          read_dir = "bicausal/datasets/Tuebingen",
                          write_dir = "bicausal/results",
                          overwrite = FALSE,
                          ...) {
  # Runs func on the Tübingen dataset and saves results to a shared CSV file.
  # Columns: ['method', 'parameters', 'Pair', 'score', 'weight', 'timestamp']

  data_and_weights <- getTuebingen(read_dir)
  data <- data_and_weights$data
  weights <- data_and_weights$weights

  dir.create(write_dir, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(write_dir, "tuebingen_scores.csv")

  args <- list(...)
  method_name <- deparse(substitute(func))
  parameters <- serialize_params(list(), args)

  if (file.exists(path)) {
    df_existing <- read_csv(path, show_col_types = FALSE)
    df_existing$parameters <- sapply(df_existing$parameters, normalize_str)
  } else {
    df_existing <- tibble(
      method = character(),
      parameters = character(),
      Pair = integer(),
      score = numeric(),
      weight = numeric(),
      timestamp = character()
    )
  }

  results <- list()

  for (i in seq_along(data)) {
    x <- data[[i]][[1]]
    y <- data[[i]][[2]]
    w <- weights[[i]]

    exists <- any(
      df_existing$method == method_name &
      df_existing$parameters == parameters &
      df_existing$Pair == i
    )

    if (exists && !overwrite) {
      cat("⏩ Skipping Pair", i, "for", method_name, "(already computed)\n")
      next
    }

    score <- tryCatch({
    xy_data <- cbind(x, y)
    func(xy_data, ...)
    }, error = function(e) {
      cat("⚠️ Skipping Pair", i, "due to error:", e$message, "\n")
      return(NULL)
    })

    if (!is.null(score)) {
      results[[length(results) + 1]] <- tibble(
        method = method_name,
        parameters = parameters,
        Pair = i,
        score = score,
        weight = w,
        timestamp = as.POSIXct(Sys.time(), tz = "UTC")
      )
    }
  }

  if (length(results) == 0) {
    cat("❌ No results to save.\n")
    return()
  }

  df_new <- bind_rows(results)

  if (overwrite) {
    df_existing <- df_existing %>%
      filter(!(method == method_name &
               parameters == parameters &
               Pair %in% df_new$Pair))
  }

  df_final <- bind_rows(df_existing, df_new)
  write_csv(df_final, path)
  cat("✅ Saved Tuebingen results to", path, "\n")

  invisible(path)
}

# ------------------------
# Core function: run_lisbon
# ------------------------

run_lisbon <- function(func,
                       read_dir = "bicausal/datasets/Lisbon/data",
                       write_dir = "bicausal/results",
                       overwrite = FALSE,
                       ...) {
  # Applies func(x, y, ...) to all .txt files under read_dir recursively.
  # Saves results to a shared CSV file: ['method', 'parameters', 'filename', 'score', 'timestamp']

  dir.create(write_dir, showWarnings = FALSE, recursive = TRUE)
  output_path <- file.path(write_dir, "lisbon_scores.csv")

  args <- list(...)
  method_name <- deparse(substitute(func))
  parameters <- serialize_params(list(), args)

  if (file.exists(output_path)) {
    df_results <- read_csv(output_path, show_col_types = FALSE)
    df_results$parameters <- sapply(df_results$parameters, normalize_str)
  } else {
    df_results <- tibble(
      method = character(),
      parameters = character(),
      filename = character(),
      score = numeric(),
      timestamp = character()
    )
  }

  txt_files <- list.files(read_dir, pattern = "\\.txt$", recursive = TRUE, full.names = TRUE)

  if (length(txt_files) == 0) {
    cat("⚠️ No .txt files found in", read_dir, "\n")
    return()
  }

  new_rows <- list()

  for (path in txt_files) {
    fname <- basename(path)
    exists <- any(
      df_results$method == method_name &
      df_results$parameters == parameters &
      df_results$filename == fname
    )

    if (exists && !overwrite) {
      cat("⏩ Skipping", fname, "for", method_name, "(already computed)\n")
      next
    }

    df <- tryCatch({
      read_table(path, col_names = FALSE)
    }, error = function(e) {
      cat("⚠️ Skipping", fname, "due to error:", e$message, "\n")
      return(NULL)
    })

    if (is.null(df)) next

    x <- as.matrix(df[[1]])
    y <- as.matrix(df[[2]])

    score <- tryCatch({
    xy_data <- cbind(x, y)
    func(xy_data, ...)
    }, error = function(e) {
      cat("⚠️ Error in", fname, ":", e$message, "\n")
      return(NULL)
    })

    if (!is.null(score)) {
      new_rows[[length(new_rows) + 1]] <- tibble(
        method = method_name,
        parameters = parameters,
        filename = fname,
        score = score,
        timestamp = as.POSIXct(Sys.time(), tz = "UTC")
      )
    }
  }

  if (length(new_rows) > 0) {
    df_new <- bind_rows(new_rows)
    if (overwrite) {
      df_results <- df_results %>%
        filter(!(method == method_name &
                 parameters == parameters &
                 filename %in% df_new$filename))
    }
    df_results <- bind_rows(df_results, df_new)
    write_csv(df_results, output_path)
    cat("✅ All Lisbon results saved to", output_path, "\n")
  } else {
    cat("❌ No new results to save.\n")
  }
}

# ------------------------
# Core function: benchmark_function
# ------------------------

benchmark_function <- function(func,
                               test_file,
                               output_path = "bicausal/results/times.csv",
                               overwrite = FALSE,
                               seed = 42,
                               ...) {
  # Benchmarks execution time of func([x, y], ...) as a function of sample size.
  # Saves to shared CSV: ['method', 'parameters', 'npoints', 'execution_time', 'timestamp']

  set.seed(seed)
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)

  df <- read_table(test_file, col_names = FALSE)
  x <- df[[1]]
  y <- df[[2]]

  idx <- sample(seq_along(x))
  x <- x[idx]
  y <- y[idx]
  n_total <- length(x)

  sizes <- c()
  n <- 10
  while (n < n_total) {
    sizes <- c(sizes, n)
    n <- as.integer(n * 1.7 + 10)
  }
  if (tail(sizes, 1) != n_total) sizes <- c(sizes, n_total)

  args <- list(...)
  method_name <- deparse(substitute(func))
  parameters <- serialize_params(list(), args)

  if (file.exists(output_path)) {
    times_df <- read_csv(output_path, show_col_types = FALSE)
    times_df$parameters <- sapply(times_df$parameters, normalize_str)
  } else {
    times_df <- tibble(
      method = character(),
      parameters = character(),
      npoints = integer(),
      execution_time = numeric(),
      timestamp = character()
    )
  }

  for (n_points in sizes) {
    exists <- any(
      times_df$method == method_name &
      times_df$parameters == parameters &
      times_df$npoints == n_points
    )

    if (exists && !overwrite) {
      cat("⏩ Skipping n =", n_points, "for", method_name, "(already computed)\n")
      next
    }

    cat("⏱ Running", method_name, "with", n_points, "points...\n")

    subset <- cbind(x[1:n_points], y[1:n_points])


    start <- Sys.time()
    success <- tryCatch({
      func(subset, ...)
      TRUE
    }, error = function(e) {
      cat("⚠️ Error at n =", n_points, ":", e$message, "\n")
      FALSE
    })
    if (!success) next

    elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))

    new_row <- tibble(
      method = method_name,
      parameters = parameters,
      npoints = n_points,
      execution_time = elapsed,
      timestamp = as.POSIXct(Sys.time(), tz = "UTC")
    )

    if (overwrite) {
      times_df <- times_df %>%
        filter(!(method == method_name &
                 parameters == parameters &
                 npoints == n_points))
    }

    times_df <- bind_rows(times_df, new_row)
    write_csv(times_df, output_path)
    cat("✅ Completed", n_points, "points in", sprintf("%.4f", elapsed), "s.\n")
  }

  cat("📊 Benchmark results saved to", output_path, "\n")
}
