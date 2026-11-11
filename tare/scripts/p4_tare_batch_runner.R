#!/usr/bin/env Rscript
# Batch four DA methods in one R process for one TARE corpus dataset.

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
if (length(args) < 4) {
  stop("Usage: p4_tare_batch_runner.R <input_dir> <output_dir> <seed> <scenario>")
}

input_dir <- args[[1]]
output_dir <- args[[2]]
seed <- as.integer(args[[3]])
scenario_name <- args[[4]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
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
metadata$condition <- factor(
  metadata$condition,
  levels = c("control", "treatment")
)
metadata$project <- factor(metadata$project)

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
} else {
  getwd()
}
source_path <- file.path(
  script_dir,
  "p4_da_method_functions_source.R"
)
if (!file.exists(source_path)) {
  source_path <- file.path(script_dir, "p4_da_method_functions_source.R")
}
source_lines <- readLines(source_path, warn = FALSE, encoding = "UTF-8")
start <- grep(
  "^blank_failure <- function",
  source_lines,
  fixed = FALSE
)[1]
stop <- grep("^runner <- switch", source_lines, fixed = FALSE)[1]
if (is.na(start) || is.na(stop) || stop <= start) {
  stop("Could not isolate DA method functions")
}
eval(parse(text = source_lines[start:(stop - 1)]), envir = environment())

methods <- c("aldex2", "ancombc2", "linda", "maaslin2")
for (method_id in methods) {
  started <- Sys.time()
  method_version <- ""
  method_args <- list()
  output_csv <- file.path(output_dir, paste0(method_id, ".csv"))
  output_json <- file.path(output_dir, paste0(method_id, ".json"))
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
  write_json(
    metadata_json,
    output_json,
    auto_unbox = TRUE,
    pretty = TRUE
  )
}
