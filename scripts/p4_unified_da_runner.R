#!/usr/bin/env Rscript
# Unified differential-abundance runner for the Paper 4 benchmark.

project_rlib <- Sys.getenv("R_LIBS_USER", file.path(getwd(), "Rlib"))
.libPaths(unique(c(project_rlib, .libPaths())))

suppressPackageStartupMessages({
  library(ALDEx2)
  library(ANCOMBC)
  library(MicrobiomeStat)
  library(Maaslin2)
  library(phyloseq)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: p4_unified_da_runner.R <method> <input_dir> <output_csv> <output_json> <seed> [scenario]")
}

method_id <- args[[1]]
input_dir <- args[[2]]
output_csv <- args[[3]]
output_json <- args[[4]]
seed <- as.integer(args[[5]])
scenario_name <- if (length(args) >= 6) args[[6]] else basename(input_dir)
set.seed(seed)

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
rownames(metadata) <- metadata$sample
metadata <- metadata[colnames(counts), , drop = FALSE]
if (any(is.na(metadata$sample))) {
  stop("Sample order mismatch between counts and metadata")
}
metadata$condition <- factor(
  metadata$condition,
  levels = c("control", "treatment")
)
metadata$project <- factor(metadata$project)
if (any(is.na(metadata$condition))) {
  stop("Unexpected condition value")
}

started <- Sys.time()
method_version <- ""
method_args <- list()

blank_failure <- function(error) {
  data.frame(
    feature = NA_character_,
    estimate = NA_real_,
    statistic = NA_real_,
    pvalue = NA_real_,
    padj = NA_real_,
    reject = FALSE,
    direction = NA_character_,
    method_flag = NA_character_,
    status = "failed",
    error = as.character(error),
    stringsAsFactors = FALSE
  )
}

run_aldex2 <- function(counts, metadata) {
  method_version <<- as.character(packageVersion("ALDEx2"))
  projects <- levels(metadata$project)
  aggregate_counts <- list()
  conditions <- character(0)
  project_ids <- character(0)
  for (project in projects) {
    project_meta <- metadata[metadata$project == project, , drop = FALSE]
    control_samples <- rownames(project_meta)[project_meta$condition == "control"]
    treatment_samples <- rownames(project_meta)[project_meta$condition == "treatment"]
    if (!length(control_samples) || !length(treatment_samples)) next
    aggregate_counts[[length(aggregate_counts) + 1]] <-
      rowSums(counts[, control_samples, drop = FALSE])
    conditions <- c(conditions, "control")
    project_ids <- c(project_ids, project)
    aggregate_counts[[length(aggregate_counts) + 1]] <-
      rowSums(counts[, treatment_samples, drop = FALSE])
    conditions <- c(conditions, "treatment")
    project_ids <- c(project_ids, project)
  }
  aggregate_matrix <- do.call(cbind, aggregate_counts)
  method_args <<- list(
    mc.samples = 128,
    test = "t",
    effect = TRUE,
    denom = "all",
    paired.test = TRUE,
    aggregation = "project-condition pseudobulk"
  )
  clr_object <- ALDEx2::aldex.clr(
    aggregate_matrix,
    conds = as.character(conditions),
    mc.samples = 128,
    denom = "all",
    verbose = FALSE
  )
  t_result <- ALDEx2::aldex.ttest(
    clr_object,
    paired.test = TRUE,
    verbose = FALSE
  )
  effect_result <- ALDEx2::aldex.effect(
    clr_object,
    paired.test = TRUE,
    verbose = FALSE
  )
  result <- cbind(effect_result, t_result)
  data.frame(
    feature = rownames(result),
    estimate = result$diff.btw,
    statistic = result$effect,
    pvalue = result$we.ep,
    padj = result$we.eBH,
    reject = !is.na(result$we.eBH) & result$we.eBH < 0.05,
    direction = ifelse(result$diff.btw > 0, "up", "down"),
    method_flag = "",
    status = "ok",
    error = "",
    stringsAsFactors = FALSE
  )
}

run_ancombc2 <- function(counts, metadata) {
  method_version <<- as.character(packageVersion("ANCOMBC"))
  otu <- phyloseq::otu_table(counts, taxa_are_rows = TRUE)
  sample_data <- phyloseq::sample_data(metadata)
  ps <- phyloseq::phyloseq(otu, sample_data)
  method_args <<- list(
    fix_formula = "condition",
    rand_formula = "(1 | project)",
    p_adj_method = "BH",
    prv_cut = 0.10,
    lib_cut = 0,
    pseudo_sens = TRUE,
    struc_zero = FALSE,
    global = FALSE,
    pairwise = FALSE,
    n_cl = 1
  )
  out <- ANCOMBC::ancombc2(
    data = ps,
    taxa_are_rows = TRUE,
    fix_formula = "condition",
    rand_formula = "(1 | project)",
    p_adj_method = "BH",
    prv_cut = 0.10,
    lib_cut = 0,
    pseudo_sens = TRUE,
    group = "condition",
    struc_zero = FALSE,
    global = FALSE,
    pairwise = FALSE,
    n_cl = 1,
    verbose = FALSE
  )
  result <- out$res
  pick <- function(prefix) {
    candidates <- grep(paste0("^", prefix, "_condition"), names(result), value = TRUE)
    if (!length(candidates)) NA_character_ else candidates[[1]]
  }
  lfc_col <- pick("lfc")
  se_col <- pick("se")
  stat_col <- pick("W")
  p_col <- pick("p")
  q_col <- pick("q")
  diff_col <- pick("diff")
  passed_col <- pick("passed_ss")
  estimate <- if (!is.na(lfc_col)) result[[lfc_col]] else rep(NA_real_, nrow(result))
  statistic <- if (!is.na(stat_col)) result[[stat_col]] else rep(NA_real_, nrow(result))
  pvalue <- if (!is.na(p_col)) result[[p_col]] else rep(NA_real_, nrow(result))
  padj <- if (!is.na(q_col)) result[[q_col]] else rep(NA_real_, nrow(result))
  reject <- if (!is.na(diff_col)) as.logical(result[[diff_col]]) else !is.na(padj) & padj < 0.05
  method_flag <- if (!is.na(passed_col)) as.character(result[[passed_col]]) else ""
  data.frame(
    feature = result$taxon,
    estimate = estimate,
    statistic = statistic,
    pvalue = pvalue,
    padj = padj,
    reject = !is.na(reject) & reject,
    direction = ifelse(estimate > 0, "up", "down"),
    method_flag = method_flag,
    status = "ok",
    error = "",
    stringsAsFactors = FALSE
  )
}

run_linda <- function(counts, metadata) {
  method_version <<- as.character(packageVersion("MicrobiomeStat"))
  method_args <<- list(
    formula = "~ condition + (1 | project)",
    feature.dat.type = "count",
    prev.filter = 0.10,
    p.adj.method = "BH",
    alpha = 0.05,
    adaptive = FALSE,
    zero.handling = "pseudo-count",
    pseudo.cnt = 0.5,
    n.cores = 1
  )
  fit <- MicrobiomeStat::linda(
    feature.dat = counts,
    meta.dat = metadata,
    formula = "~ condition + (1 | project)",
    feature.dat.type = "count",
    prev.filter = 0.10,
    p.adj.method = "BH",
    alpha = 0.05,
    adaptive = FALSE,
    zero.handling = "pseudo-count",
    pseudo.cnt = 0.5,
    n.cores = 1,
    verbose = FALSE
  )
  coefficient_name <- names(fit$output)[grepl("condition", names(fit$output))][1]
  if (is.na(coefficient_name)) {
    stop("LinDA condition coefficient not found")
  }
  result <- fit$output[[coefficient_name]]
  data.frame(
    feature = rownames(result),
    estimate = result$log2FoldChange,
    statistic = result$stat,
    pvalue = result$pvalue,
    padj = result$padj,
    reject = as.logical(result$reject),
    direction = ifelse(result$log2FoldChange > 0, "up", "down"),
    method_flag = coefficient_name,
    status = "ok",
    error = "",
    stringsAsFactors = FALSE
  )
}

run_maaslin2 <- function(counts, metadata) {
  method_version <<- as.character(packageVersion("Maaslin2"))
  output_dir <- file.path(
    dirname(output_csv),
    paste0("maaslin_tmp_", Sys.getpid(), "_", sample.int(1e8, 1))
  )
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  original_feature_names <- rownames(counts)
  safe_feature_names <- make.names(original_feature_names, unique = TRUE)
  feature_name_map <- setNames(original_feature_names, safe_feature_names)
  input_data <- as.data.frame(t(counts), check.names = FALSE)
  colnames(input_data) <- safe_feature_names
  method_args <<- list(
    fixed_effects = "condition",
    random_effects = "project",
    normalization = "TSS",
    transform = "LOG",
    analysis_method = "LM",
    min_prevalence = 0.10,
    max_significance = 0.05,
    correction = "BH",
    cores = 1
  )
  fit <- Maaslin2::Maaslin2(
    input_data = input_data,
    input_metadata = metadata,
    output = output_dir,
    min_prevalence = 0.10,
    normalization = "TSS",
    transform = "LOG",
    analysis_method = "LM",
    max_significance = 0.05,
    random_effects = "project",
    fixed_effects = "condition",
    correction = "BH",
    cores = 1,
    plot_heatmap = FALSE,
    plot_scatter = FALSE,
    max_pngs = 0,
    save_scatter = FALSE
  )
  results_path <- file.path(output_dir, "all_results.tsv")
  if (!file.exists(results_path)) {
    stop("MaAsLin2 all_results.tsv was not produced")
  }
  result <- read.delim(results_path, stringsAsFactors = FALSE, check.names = FALSE)
  result <- result[result$metadata == "condition", , drop = FALSE]
  result$feature <- unname(feature_name_map[result$feature])
  data.frame(
    feature = result$feature,
    estimate = result$coef,
    statistic = result$coef / result$stderr,
    pvalue = result$pval,
    padj = result$qval,
    reject = !is.na(result$qval) & result$qval < 0.05,
    direction = ifelse(result$coef > 0, "up", "down"),
    method_flag = "",
    status = "ok",
    error = "",
    stringsAsFactors = FALSE
  )
}

runner <- switch(
  method_id,
  aldex2 = run_aldex2,
  ancombc2 = run_ancombc2,
  linda = run_linda,
  maaslin2 = run_maaslin2,
  stop("Unknown method: ", method_id)
)

result <- tryCatch(
  runner(counts, metadata),
  error = function(error) blank_failure(conditionMessage(error))
)
result <- cbind(method = method_id, result)
write.csv(result, output_csv, row.names = FALSE, na = "")

elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
status <- if ("failed" %in% result$status) "failed" else "ok"
metadata_json <- list(
  method = method_id,
  method_version = method_version,
  scenario = scenario_name,
  seed = seed,
  status = status,
  elapsed_sec = elapsed,
  n_samples = ncol(counts),
  n_features_input = nrow(counts),
  n_features_output = sum(result$status == "ok"),
  n_reject = sum(result$reject, na.rm = TRUE),
  method_args = method_args,
  error = paste(result$error[result$status == "failed"], collapse = "; ")
)
write_json(metadata_json, output_json, auto_unbox = TRUE, pretty = TRUE)
quit(save = "no", status = 0)
