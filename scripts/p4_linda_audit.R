#!/usr/bin/env Rscript
# Focused LinDA parameter audit for the Paper 4 sparse20_fc2 scenario.

project_rlib_candidates <- c(
  Sys.getenv("R_LIBS_USER"),
  file.path(getwd(), "Rlib")
)
project_rlib_candidates <- project_rlib_candidates[
  nzchar(project_rlib_candidates) &
    dir.exists(file.path(project_rlib_candidates, "MicrobiomeStat"))
]
if (!length(project_rlib_candidates)) {
  stop("Could not locate the project R library containing MicrobiomeStat")
}
project_rlib <- project_rlib_candidates[[1]]
.libPaths(unique(c(project_rlib, .libPaths())))

suppressPackageStartupMessages({
  library(MicrobiomeStat)
  library(lmerTest)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: p4_linda_audit.R <input_dir> <seed> <output_dir>")
}

input_dir <- args[[1]]
seed <- as.integer(args[[2]])
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

counts <- as.matrix(read.csv(
  file.path(input_dir, "counts.csv"),
  row.names = 1,
  check.names = FALSE
))
storage.mode(counts) <- "integer"
metadata <- read.csv(
  file.path(input_dir, "metadata.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
truth <- read.csv(
  file.path(input_dir, "truth.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
truth$is_target <- toupper(as.character(truth$is_target)) %in%
  c("TRUE", "T", "1")
rownames(metadata) <- metadata$sample
metadata <- metadata[colnames(counts), , drop = FALSE]
metadata$condition <- factor(metadata$condition, levels = c("control", "treatment"))
metadata$project <- factor(metadata$project)

settings <- list(
  list(
    id = "primary_fixed_pseudo0.5_winsor",
    adaptive = FALSE,
    zero_handling = "pseudo-count",
    pseudo_cnt = 0.5,
    is_winsor = TRUE
  ),
  list(
    id = "imputation_winsor",
    adaptive = FALSE,
    zero_handling = "imputation",
    pseudo_cnt = 0.5,
    is_winsor = TRUE
  ),
  list(
    id = "adaptive_default_choice_winsor",
    adaptive = TRUE,
    zero_handling = "pseudo-count",
    pseudo_cnt = 0.5,
    is_winsor = TRUE
  ),
  list(
    id = "fixed_pseudo0.5_no_winsor",
    adaptive = FALSE,
    zero_handling = "pseudo-count",
    pseudo_cnt = 0.5,
    is_winsor = FALSE
  )
)

metric_from_result <- function(result, setting) {
  result$feature <- rownames(result)
  merged <- merge(
    result,
    truth[, c(
      "feature",
      "is_target",
      "requested_log2_effect",
      "case_control_observational_log2"
    )],
    by = "feature",
    all.x = TRUE,
    sort = FALSE
  )
  merged$reject_bh_recomputed <- stats::p.adjust(merged$pvalue, method = "BH") < 0.05
  merged$reject_bh_recomputed[is.na(merged$pvalue)] <- FALSE
  merged$is_target <- as.logical(merged$is_target)
  merged$native_reject <- as.logical(merged$reject)
  merged$native_reject[is.na(merged$native_reject)] <- FALSE
  merged$false_positive <- !merged$is_target & merged$reject_bh_recomputed
  merged$true_positive <- merged$is_target & merged$reject_bh_recomputed
  merged$direction_correct <- ifelse(
    merged$true_positive,
    sign(merged$log2FoldChange) == sign(merged$requested_log2_effect),
    NA
  )

  tp <- sum(merged$true_positive)
  fp <- sum(merged$false_positive)
  fn <- sum(merged$is_target & !merged$reject_bh_recomputed)
  n_reject <- sum(merged$reject_bh_recomputed)
  precision <- if (tp + fp > 0) tp / (tp + fp) else NA_real_
  power <- if (tp + fn > 0) tp / (tp + fn) else NA_real_
  f1 <- if (
    is.finite(precision) &&
      is.finite(power) &&
      precision + power > 0
  ) {
    2 * precision * power / (precision + power)
  } else {
    NA_real_
  }
  fdp <- if (n_reject > 0) fp / n_reject else NA_real_
  fp_features <- merged$feature[merged$false_positive]
  fp_effects <- merged$log2FoldChange[merged$false_positive]
  fp_summary <- if (length(fp_features)) {
    paste0(fp_features, ":", sprintf("%.3f", fp_effects), collapse = ";")
  } else {
    ""
  }
  bh_diff <- abs(merged$padj - stats::p.adjust(merged$pvalue, method = "BH"))
  bh_max_abs_diff <- max(bh_diff, na.rm = TRUE)
  if (!is.finite(bh_max_abs_diff)) bh_max_abs_diff <- NA_real_
  native_reject_diff <- sum(
    merged$native_reject != merged$reject_bh_recomputed,
    na.rm = TRUE
  )

  data.frame(
    setting = setting$id,
    seed = seed,
    n_features = nrow(merged),
    n_target = sum(merged$is_target),
    n_reject = n_reject,
    tp = tp,
    fp = fp,
    fn = fn,
    fdp = fdp,
    power = power,
    precision = precision,
    f1 = f1,
    direction_accuracy = mean(merged$direction_correct, na.rm = TRUE),
    false_positive_features = fp_summary,
    native_bh_reject_mismatches = native_reject_diff,
    native_bh_max_abs_diff = bh_max_abs_diff,
    stringsAsFactors = FALSE
  )
}

summary_rows <- list()
feature_rows <- list()
log_lines <- character(0)

# Reproduce the package's adaptive zero-handling decision rule so the audit
# records which branch LinDA selected for each dataset.
temp <- t(t(counts) / colSums(counts))
keep_tax <- rowMeans(temp != 0) >= 0.10
counts_filtered <- counts[keep_tax, , drop = FALSE]
# LinDA performs its adaptive depth test after feature filtering and
# winsorization, so use the same processed matrix here.
counts_probe <- MicrobiomeStat:::winsor.fun(
  counts_filtered,
  quan = 0.97,
  feature.dat.type = "count"
)
depth <- colSums(counts_probe)
depth_fit <- tryCatch(
  lmerTest::lmer(
    log(depth) ~ condition + (1 | project),
    data = metadata
  ),
  error = function(error) error
)
if (inherits(depth_fit, "error")) {
  adaptive_choice <- "adaptive_depth_test_failed"
  adaptive_p_text <- paste(
    "depth range:",
    min(depth),
    "to",
    max(depth),
    ";",
    conditionMessage(depth_fit)
  )
} else {
  depth_coef <- coef(summary(depth_fit))
  depth_p <- depth_coef[-1, "Pr(>|t|)"]
  names(depth_p) <- rownames(depth_coef)[-1]
  adaptive_choice <- if (any(depth_p <= 0.10)) "imputation" else "pseudo-count"
  adaptive_p_text <- paste(
    paste0(names(depth_p), "=", format(depth_p, digits = 6, scientific = TRUE)),
    collapse = ";"
  )
}

for (setting in settings) {
  elapsed <- system.time({
    fit <- tryCatch(
      MicrobiomeStat::linda(
        feature.dat = counts,
        meta.dat = metadata,
        formula = "~ condition + (1 | project)",
        feature.dat.type = "count",
        prev.filter = 0.10,
        p.adj.method = "BH",
        alpha = 0.05,
        adaptive = setting$adaptive,
        zero.handling = setting$zero_handling,
        pseudo.cnt = setting$pseudo_cnt,
        is.winsor = setting$is_winsor,
        n.cores = 1,
        verbose = FALSE
      ),
      error = function(error) error
    )
  })[["elapsed"]]

  if (inherits(fit, "error")) {
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      setting = setting$id,
      seed = seed,
      status = "failed",
      adaptive_requested = setting$adaptive,
      zero_handling_requested = setting$zero_handling,
      is_winsor = setting$is_winsor,
      adaptive_choice = adaptive_choice,
      adaptive_depth_p = adaptive_p_text,
      elapsed_sec = elapsed,
      error = conditionMessage(fit),
      stringsAsFactors = FALSE
    )
    log_lines <- c(
      log_lines,
      paste(setting$id, "FAILED:", conditionMessage(fit))
    )
    next
  }

  coefficient_name <- names(fit$output)[grepl("condition", names(fit$output))][1]
  if (is.na(coefficient_name)) {
    stop("LinDA condition coefficient not found in ", setting$id)
  }
  result <- fit$output[[coefficient_name]]
  metrics <- metric_from_result(result, setting)
  metrics$status <- "ok"
  metrics$adaptive_requested <- setting$adaptive
  metrics$zero_handling_requested <- setting$zero_handling
  metrics$is_winsor <- setting$is_winsor
  metrics$adaptive_choice <- adaptive_choice
  metrics$adaptive_depth_p <- adaptive_p_text
  coefficient_index <- match(coefficient_name, fit$variables)
  metrics$estimated_bias <- fit$bias[[coefficient_index]]
  metrics$min_processed_value <- min(fit$feature.dat.use, na.rm = TRUE)
  metrics$processed_zero_count <- sum(fit$feature.dat.use == 0, na.rm = TRUE)
  metrics$elapsed_sec <- elapsed
  metrics$error <- ""
  summary_rows[[length(summary_rows) + 1]] <- metrics

  result$setting <- setting$id
  result$seed <- seed
  result$feature <- rownames(result)
  result$is_target <- truth$is_target[match(result$feature, truth$feature)]
  result$requested_log2_effect <-
    truth$requested_log2_effect[match(result$feature, truth$feature)]
  result$case_control_observational_log2 <-
    truth$case_control_observational_log2[match(result$feature, truth$feature)]
  result$reject_bh_recomputed <-
    stats::p.adjust(result$pvalue, method = "BH") < 0.05
  result$reject_bh_recomputed[is.na(result$pvalue)] <- FALSE
  result$false_positive <- !result$is_target & result$reject_bh_recomputed
  result$true_positive <- result$is_target & result$reject_bh_recomputed
  feature_rows[[length(feature_rows) + 1]] <- result
  log_lines <- c(
    log_lines,
    paste(
      setting$id,
      "ok; rejects =", metrics$n_reject,
      "TP =", metrics$tp,
      "FP =", metrics$fp,
      "FDP =", format(metrics$fdp, digits = 5)
    )
  )
}

summary_columns <- unique(unlist(lapply(summary_rows, names)))
summary_rows <- lapply(summary_rows, function(row) {
  missing_columns <- setdiff(summary_columns, names(row))
  for (column in missing_columns) row[[column]] <- NA
  row[, summary_columns, drop = FALSE]
})
summary_df <- do.call(rbind, summary_rows)
feature_df <- if (length(feature_rows)) do.call(rbind, feature_rows) else data.frame()
write.csv(
  summary_df,
  file.path(output_dir, paste0("linda_audit_summary_seed_", seed, ".csv")),
  row.names = FALSE,
  na = ""
)
write.csv(
  feature_df,
  file.path(output_dir, paste0("linda_audit_features_seed_", seed, ".csv")),
  row.names = FALSE,
  na = ""
)
writeLines(
  c(
    paste("Input:", input_dir),
    paste("Seed:", seed),
    paste("Samples:", ncol(counts)),
    paste("Features:", nrow(counts)),
    paste("Adaptive branch selected by package rule:", adaptive_choice),
    paste("Adaptive depth-test p-values:", adaptive_p_text),
    log_lines
  ),
  con = file.path(output_dir, paste0("linda_audit_log_seed_", seed, ".txt")),
  useBytes = TRUE
)
