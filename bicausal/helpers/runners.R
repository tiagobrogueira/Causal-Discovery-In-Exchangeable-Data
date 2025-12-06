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

getTuebingen <- function(read_dir = "bicausal/benchmarks/Tuebingen") {
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
                          read_dir = "bicausal/benchmarks/Tuebingen",
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
  method_name_lower <- tolower(method_name)
  parameters <- serialize_params(list(), args)

  if (file.exists(path)) {
    df_existing <- read_csv(path, show_col_types = FALSE)
    df_existing$method <- tolower(df_existing$method)
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
      df_existing$method == method_name_lower &
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
      filter(!(method == method_name_lower &
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
                       read_dir = "bicausal/benchmarks/Lisbon/data",
                       write_dir = "bicausal/results",
                       overwrite = FALSE,
                       ...) {
  dir.create(write_dir, showWarnings = FALSE, recursive = TRUE)
  output_path <- file.path(write_dir, "lisbon_scores.csv")

  args <- list(...)
  method_name <- deparse(substitute(func))
  method_name_lower <- tolower(method_name)
  parameters <- ""

  # Load or initialize CSV
  if (file.exists(output_path)) {
    df_results <- read_csv(output_path, show_col_types = FALSE)
    df_results$method <- tolower(df_results$method)
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

  for (path in txt_files) {
    fname <- basename(path)

    # Check if already computed
    exists <- any(
      df_results$method == method_name_lower &
      df_results$parameters == parameters &
      df_results$filename == fname
    )

    if (exists && !overwrite) {
      cat("⏩ Skipping", fname, "for", method_name, "(already computed)\n")
      next
    }

    df <- tryCatch({
      read_table(path, col_names = FALSE, show_col_types = FALSE)
    }, error = function(e) {
      cat("⚠️ Skipping", fname, "due to read error:", e$message, "\n")
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

    if (is.null(score)) next

    # Construct one new row
    new_row <- tibble(
      method = method_name_lower,
      parameters = parameters,
      filename = fname,
      score = score,
      timestamp = as.POSIXct(Sys.time(), tz = "UTC")
    )

    # If overwriting, remove old rows for this file/method/params
    if (overwrite) {
      df_results <- df_results %>%
        filter(!(method == method_name_lower &
                 parameters == parameters &
                 filename == fname))
    }

    # Append new row to df_results
    df_results <- bind_rows(df_results, new_row)

    # Write CSV immediately
    write_csv(df_results, output_path)
    cat("💾 Saved result for", fname, "\n")
  }

  cat("✅ Incremental saving complete. Results saved to", output_path, "\n")
}

run_ce <- function(func,
                   datasets = NULL,
                   read_dir = "bicausal/benchmarks/synthetic/CE-Guyon",
                   write_dir = "bicausal/results",
                   overwrite = FALSE,
                   ...) {

  # -----------------------------
  # CE dataset list
  # -----------------------------
  ALL_CE <- c("CE-Cha", "CE-Gauss", "CE-Multi", "CE-Net")

  if (is.null(datasets) || length(datasets) == 0) {
    datasets <- ALL_CE
  } else {
    invalid <- setdiff(datasets, ALL_CE)
    if (length(invalid) > 0) {
      stop("Invalid CE dataset(s): ",
           paste(invalid, collapse = ", "),
           ". Must be one of: ",
           paste(ALL_CE, collapse = ", "))
    }
  }

  # -----------------------------
  # Prepare output file
  # -----------------------------
  dir.create(write_dir, showWarnings = FALSE, recursive = TRUE)
  output_path <- file.path(write_dir, "CE_scores.csv")

  args <- list(...)
  method_name <- deparse(substitute(func))
  method_name_lower <- tolower(method_name)
  parameters <- as.character(serialize_params(list(), args))

  # -----------------------------
  # Load or initialize CSV
  # -----------------------------
  if (file.exists(output_path)) {
    df_existing <- readr::read_csv(output_path, show_col_types = FALSE)
    df_existing$method <- tolower(df_existing$method)
    df_existing$parameters <- df_existing$parameters <- as.character(sapply(df_existing$parameters, normalize_str))
    df_existing$Pair <- as.integer(df_existing$Pair)
    df_existing$score <- as.numeric(df_existing$score)
    df_existing$weight <- as.numeric(df_existing$weight)
    df_existing$timestamp <- as.POSIXct(df_existing$timestamp, tz = "UTC")
  } else {
    df_existing <- tibble::tibble(
      method = character(),
      parameters = character(),
      dataset = character(),
      Pair = integer(),
      score = numeric(),
      weight = numeric(),
      timestamp = character()
    )
  }

  # =====================================================
  # Main loop over datasets
  # =====================================================
  for (dataset in datasets) {
    cat("\n🚀 Running CE dataset:", dataset, "\n")

    # -----------------------------
    # Build paths
    # -----------------------------
    pairs_file   <- file.path(read_dir, paste0(dataset, "_pairs.csv"))
    targets_file <- file.path(read_dir, paste0(dataset, "_targets.csv"))

    if (!file.exists(pairs_file))
      stop("Pairs file not found: ", pairs_file)
    if (!file.exists(targets_file))
      stop("Targets file not found: ", targets_file)

    # -----------------------------
    # Load data
    # -----------------------------
    df_pairs   <- readr::read_csv(pairs_file, show_col_types = FALSE)
    df_targets <- readr::read_csv(targets_file, show_col_types = FALSE)

    # Ensure consistent column names
    colnames(df_pairs) <- tolower(colnames(df_pairs))
    colnames(df_targets) <- tolower(colnames(df_targets))

    if (!"target" %in% colnames(df_targets)) {
      stop("targets CSV must contain column: target")
    }

    if (nrow(df_pairs) != nrow(df_targets)) {
      stop("Row count mismatch: ", 
           nrow(df_pairs), " pairs vs ", nrow(df_targets), " targets")
    }

    weight <- 1 / nrow(df_pairs)

    # -----------------------------
    # Loop through each pair
    # -----------------------------
    for (idx in seq_len(nrow(df_pairs))) {
  x_str <- as.character(df_pairs[[2]][idx])
  y_str <- as.character(df_pairs[[3]][idx])

  x <- matrix(as.numeric(strsplit(x_str, " ")[[1]]), ncol = 1)
  y <- matrix(as.numeric(strsplit(y_str, " ")[[1]]), ncol = 1)

  pair_idx <- idx

  if (df_targets[[2]][idx] == -1) {
    tmp <- x
    x <- y
    y <- tmp
  }

      # -----------------------------------------
      # Check existing
      # -----------------------------------------
      exists <- any(
        df_existing$method == method_name_lower &
        df_existing$parameters == parameters &
        df_existing$dataset == dataset &
        df_existing$Pair == pair_idx
      )

      if (exists && !overwrite) {
        cat("⏩ Skipping", dataset, "Pair", pair_idx, "(already computed)\n")
        next
      }

      # -----------------------------------------
      # Compute score (like run_tuebingen)
      # -----------------------------------------
      score <- tryCatch({
        xy_data <- cbind(x, y)  # match run_tuebingen
        func(xy_data, ...)       # pass all additional args
      }, error = function(e) {
        cat("⚠️ Error in", method_name, "on", dataset, "Pair", pair_idx, ":", e$message, "\n")
        return(NULL)
      })

      if (is.null(score)) next
      if (is.nan(score)) score <- NA

      # -----------------------------------------
      # Make new row
      # -----------------------------------------
      new_row <- tibble::tibble(
        method = method_name_lower,
        parameters = parameters,
        dataset = dataset,
        Pair = pair_idx,
        score = score,
        weight = weight,
        timestamp = as.POSIXct(Sys.time(), tz = "UTC")
      )

      # Overwrite old entry if needed
      if (overwrite) {
        df_existing <- df_existing %>%
          dplyr::filter(!(method == method_name_lower &
                          parameters == parameters &
                          dataset == dataset &
                          Pair == pair_idx))
      }

      # Append and save
      df_existing <- dplyr::bind_rows(df_existing, new_row)
      readr::write_csv(df_existing, output_path)

      cat("✔ Saved", dataset, "Pair", pair_idx, "\n")
    }
  }

  cat("\n✅ Saved CE results to", output_path, "\n")
  invisible(output_path)
}


run_anlsmn <- function(func,
                        datasets = NULL,
                        read_dir = "bicausal/benchmarks/synthetic/ANLSMN-Tagasovska",
                        write_dir = "bicausal/results",
                        overwrite = FALSE,
                        ...) {

  # -----------------------------
  # Dataset selection
  # -----------------------------
  all_entries <- list.files(read_dir, full.names = TRUE)
  datasets <- basename(all_entries[file.info(all_entries)$isdir])


  if (length(datasets) == 0) {
    stop("No datasets found in ", read_dir)
  }

  dir.create(write_dir, showWarnings = FALSE, recursive = TRUE)
  output_path <- file.path(write_dir, "ANLSMN_scores.csv")

  args <- list(...)
  method_name <- tolower(deparse(substitute(func)))
  parameters <- as.character(serialize_params(list(), args))

  # -----------------------------
  # Load or initialize CSV
  # -----------------------------
  if (file.exists(output_path)) {
    df_existing <- readr::read_csv(output_path, show_col_types = FALSE)
    df_existing$method <- tolower(df_existing$method)
    df_existing$parameters <- as.character(sapply(df_existing$parameters, normalize_str))
    df_existing$Pair <- as.integer(df_existing$Pair)
    df_existing$score <- as.numeric(df_existing$score)
    df_existing$weight <- as.numeric(df_existing$weight)
    df_existing$timestamp <- as.POSIXct(df_existing$timestamp, tz = "UTC")
  } else {
    df_existing <- tibble::tibble(
      method = character(),
      parameters = character(),
      dataset = character(),
      Pair = integer(),
      score = numeric(),
      weight = numeric(),
      timestamp = as.POSIXct(character(), tz = "UTC")
    )
  }

  # -----------------------------
  # Main loop over datasets
  # -----------------------------
  for (dataset in datasets) {
    cat("\n🚀 Running ANLSMN dataset:", dataset, "\n")
    
    dir_ext <- file.path(read_dir, dataset)
    gt_file <- file.path(dir_ext, "pairs_gt.txt")

    if (!file.exists(gt_file)) stop("Missing ground truth file: ", gt_file)

    # Load ground truth
    pairs_gt <- as.integer(readr::read_lines(gt_file))
    n_pairs <- length(pairs_gt)
    weight <- 1 / n_pairs

    # -----------------------------
    # Loop through each pair
    # -----------------------------
    for (pair_idx in seq_len(n_pairs)) {
      # Check existing
      exists <- any(
        df_existing$method == method_name &
        df_existing$parameters == parameters &
        df_existing$dataset == dataset &
        df_existing$Pair == pair_idx
      )

      if (exists && !overwrite) {
        cat("⏩ Skipping", dataset, "Pair", pair_idx, "(already computed)\n")
        next
      }

      # Load pair file
      pair_file <- file.path(dir_ext, paste0("pair_", pair_idx, ".txt"))
      if (!file.exists(pair_file)) {
        cat("⚠️ Missing pair file:", pair_file, "skipping.\n")
        next
      }

      df_pair <- readr::read_csv(pair_file, show_col_types = FALSE)
      x <- as.matrix(df_pair[[2]])
      y <- as.matrix(df_pair[[3]])

      # Correct direction using GT
      if (pairs_gt[pair_idx] == 0) {
        tmp <- x
        x <- y
        y <- tmp
      }

      # Run method
      score <- tryCatch({
        func(cbind(x, y), ...)
      }, error = function(e) {
        cat("⚠️ Error on", dataset, "Pair", pair_idx, ":", e$message, "\n")
        return(NA)
      })

      if (is.nan(score)) score <- NA

      # Append row
      new_row <- tibble::tibble(
        method = method_name,
        parameters = parameters,
        dataset = dataset,
        Pair = pair_idx,
        score = score,
        weight = weight,
        timestamp = as.POSIXct(Sys.time(), tz = "UTC")
      )

      current_dataset <- dataset
      current_parameters <- parameters

      if (overwrite) {
        df_existing <- df_existing %>%
          dplyr::filter(!(method == method_name &
                          parameters == current_parameters &
                          dataset == current_dataset &
                          Pair == pair_idx))
      }


      df_existing <- dplyr::bind_rows(df_existing, new_row)
      readr::write_csv(df_existing, output_path)

      cat("✔ Saved", dataset, "Pair", pair_idx, "\n")
    }
  }

  cat("\n✅ Saved ANLSMN results to", output_path, "\n")
  invisible(output_path)
}

run_sim <- function(func,
                    datasets = NULL,
                    read_dir = "bicausal/benchmarks/synthetic/SIM-Mooij",
                    write_dir = "bicausal/results",
                    overwrite = FALSE,
                    ...) {

  # -----------------------------
  # Dataset selection
  # -----------------------------
  all_entries <- list.files(read_dir, full.names = TRUE)
  datasets <- basename(all_entries[file.info(all_entries)$isdir])

  if (length(datasets) == 0) stop("No datasets found in ", read_dir)
  cat("Datasets detected:", paste(datasets, collapse = ", "), "\n")

  dir.create(write_dir, showWarnings = FALSE, recursive = TRUE)
  output_path <- file.path(write_dir, "SIM_scores.csv")

  args <- list(...)
  method_name <- tolower(deparse(substitute(func)))
  parameters <- as.character(serialize_params(list(), args))

  # -----------------------------
  # Load existing CSV if available
  # -----------------------------
  if (file.exists(output_path)) {
    df_existing <- readr::read_csv(output_path, show_col_types = FALSE)
    df_existing$method <- tolower(df_existing$method)
    df_existing$parameters <- as.character(sapply(df_existing$parameters, normalize_str))
    df_existing$Pair <- as.integer(df_existing$Pair)
    df_existing$score <- as.numeric(df_existing$score)
    df_existing$weight <- as.numeric(df_existing$weight)
    df_existing$timestamp <- as.POSIXct(df_existing$timestamp, tz = "UTC")
  } else {
    df_existing <- tibble::tibble(
      method = character(),
      parameters = character(),
      dataset = character(),
      Pair = integer(),
      score = numeric(),
      weight = numeric(),
      timestamp = as.POSIXct(character(), tz = "UTC")
    )
  }

  # -----------------------------
  # Loop over datasets
  # -----------------------------
  for (dataset in datasets) {
    cat("\n🚀 Running SIM dataset:", dataset, "\n")
    dataset_dir <- file.path(read_dir, dataset)
    meta_file <- file.path(dataset_dir, "pairmeta.txt")
    if (!file.exists(meta_file)) stop("Missing pairmeta.txt in ", dataset_dir)

    # Load pairmeta.txt (no header)
    meta <- readr::read_table(meta_file, col_names = FALSE, show_col_types = FALSE)
    colnames(meta) <- c("pair", "c_start", "c_end", "e_start", "e_end", "weight")

    # Convert types
    meta$c_start <- as.integer(meta$c_start)
    meta$c_end <- as.integer(meta$c_end)
    meta$e_start <- as.integer(meta$e_start)
    meta$e_end <- as.integer(meta$e_end)
    meta$weight <- as.numeric(meta$weight)
    meta$pair <- as.character(meta$pair)

    # -----------------------------
    # Loop through all pairs
    # -----------------------------
    for (i in seq_len(nrow(meta))) {
      row <- meta[i, ]
      pair_id <- row$pair
      pair_idx <- as.integer(pair_id)

      # Check if already computed
      exists <- any(
        df_existing$method == method_name &
        df_existing$parameters == parameters &
        df_existing$dataset == dataset &
        df_existing$Pair == pair_idx
      )
      if (exists && !overwrite) {
        cat("⏩ Skipping", dataset, "Pair", pair_id, "(already computed)\n")
        next
      }

      # Load pair file
      pair_file <- file.path(dataset_dir, paste0("pair", pair_id, ".txt"))
      if (!file.exists(pair_file)) {
        cat("⚠️ Missing pair file:", pair_file, "skipping.\n")
        next
      }

      df_pair <- readr::read_table(pair_file, col_names = FALSE, show_col_types = FALSE)
      data <- as.matrix(df_pair)

      # Extract cause and effect variables
      X <- data[, row$c_start:row$c_end, drop = FALSE]
      Y <- data[, row$e_start:row$e_end, drop = FALSE]

      # Run method
      score <- tryCatch({
        func(cbind(X, Y), ...)
      }, error = function(e) {
        cat("⚠️ Error on", dataset, "Pair", pair_id, ":", e$message, "\n")
        return(NA)
      })

      # Append row
      new_row <- tibble::tibble(
        method = method_name,
        parameters = parameters,
        dataset = dataset,
        Pair = pair_idx,
        score = score,
        weight = row$weight,
        timestamp = as.POSIXct(Sys.time(), tz = "UTC")
      )

      # Overwrite if needed
      if (overwrite) {
        df_existing <- df_existing %>%
          dplyr::filter(!(method == method_name &
                          parameters == parameters &
                          dataset == dataset &
                          Pair == pair_idx))
      }

      df_existing <- dplyr::bind_rows(df_existing, new_row)
      readr::write_csv(df_existing, output_path)
      cat("✔ Saved", dataset, "Pair", pair_id, "\n")
    }
  }

  cat("\n✅ Saved SIM results to", output_path, "\n")
  invisible(output_path)
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
  method_name_lower <- tolower(method_name)
  parameters <- serialize_params(list(), args)

  if (file.exists(output_path)) {
    times_df <- read_csv(output_path, show_col_types = FALSE)
    times_df$method <- tolower(times_df$method)
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
      times_df$method == method_name_lower &
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
        filter(!(method == method_name_lower &
                 parameters == parameters &
                 npoints == n_points))
    }

    times_df <- bind_rows(times_df, new_row)
    write_csv(times_df, output_path)
    cat("✅ Completed", n_points, "points in", sprintf("%.4f", elapsed), "s.\n")
  }

  cat("📊 Benchmark results saved to", output_path, "\n")
}
