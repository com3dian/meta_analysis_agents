#!/usr/bin/env Rscript
# Fit the Yu et al. main LER ~ TND nested mixed model with nlme (REML).
#
# Usage:
#   Rscript fit_ler_tnd_lme.R --data analysis_data.csv --out lme_result.json
#
# Expected columns in --data:
#   Study, Expt, LER_final, TND_final

suppressPackageStartupMessages({
  if (!requireNamespace("nlme", quietly = TRUE)) {
    stop("Package 'nlme' is required. Install with install.packages('nlme').")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  library(nlme)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) {
    return(default)
  }
  args[[i + 1]]
}

data_path <- get_arg("--data")
out_path <- get_arg("--out", "lme_result.json")

if (is.null(data_path) || !file.exists(data_path)) {
  stop("Provide an existing --data CSV path.")
}

dat <- read.csv(data_path, stringsAsFactors = FALSE)
required <- c("Study", "Expt", "LER_final", "TND_final")
missing <- setdiff(required, names(dat))
if (length(missing) > 0) {
  stop(paste("Missing columns:", paste(missing, collapse = ", ")))
}

dat <- dat[is.finite(dat$LER_final) & is.finite(dat$TND_final), , drop = FALSE]
dat$Study <- as.factor(dat$Study)
dat$Expt <- as.factor(dat$Expt)

fit <- lme(
  LER_final ~ TND_final,
  random = ~ 1 | Study / Expt,
  data = dat,
  method = "REML",
  na.action = na.omit
)

tt <- summary(fit)$tTable
ci <- intervals(fit, which = "fixed")$fixed

payload <- list(
  intercept = unname(tt["(Intercept)", "Value"]),
  tnd_coef = unname(tt["TND_final", "Value"]),
  tnd_se = unname(tt["TND_final", "Std.Error"]),
  tnd_ci_low = unname(ci["TND_final", "lower"]),
  tnd_ci_high = unname(ci["TND_final", "upper"]),
  tnd_pvalue = unname(tt["TND_final", "p-value"]),
  n_records = nrow(dat),
  n_studies = nlevels(droplevels(dat$Study)),
  n_experiments = nlevels(interaction(dat$Study, dat$Expt, drop = TRUE)),
  engine = "nlme"
)

write_json(payload, out_path, auto_unbox = TRUE, pretty = TRUE)
message("Wrote ", out_path)
