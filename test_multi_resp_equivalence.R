#!/usr/bin/env Rscript
# =============================================================================
# Compare multiple homoscedastic normal responses joint fit to separate fits
# =============================================================================

source("MixedModel.R")
source("DataSimulation4.R")

set_threads_cpp(16)

## ============================================================
## Set up the data and known truth
## ============================================================

N <- nrow(dat2)

## ============================================================
## TEST 1: Joint fit with homoscedastic multiple responses sharing random effects
# var_pred indicates which response each observation belongs to for identification. 
# With two responses, this creates a multivariate model but still shares the global Z.
## ============================================================

cat("\n=== TEST 1: Multiple homogeneous normal responses (joint) ===\n")

fit_multi <- fit_mixed(
  responses = list(
    list(
      formula = sbp ~ bmi + age,
      random = ~ 1 + bmi + age | patient,
      response_type = "homo_normal"
    ),
    list(
      formula = hba1c ~ HDL + age,
      random = ~ 1 + HDL + age | patient,
      response_type = "homo_normal"
    )
  ),
  data = dat2,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

cat("Number of responses:", length(fit_multi$separate_fits), "\n")
if (!is.null(fit_multi$joint_fit)) {
  cat("Joint fit logLik:", fit_multi$joint_fit$logLik, "\n")
} else {
  cat("No joint fit available\n")
}

## ============================================================
## TEST 2: Separate fits 
# These won't share cross-response correlations but each response has its own
# single residual variance parameter.
## ============================================================

cat("\n=== TEST 2: Separate fits ===\n")

fit_sbp <- fit_mixed_simple(
  formula = sbp ~ bmi + age,
  random = ~ 1 + bmi | patient,
  var_pred = NULL,
  data = dat2,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

fit_hba1c <- fit_mixed_simple(
  formula = hba1c ~ HDL + age,
  random = ~ 1 + HDL | patient,
  var_pred = NULL,
  data = dat2,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

cat("Separate SBP logLik:", fit_sbp$logLik, "\n")
cat("Separate HbA1c logLik:", fit_hba1c$logLik, "\n")
cat("Sum of separate:", fit_sbp$logLik + fit_hba1c$logLik, "\n")

## ============================================================
## TEST 3: Long format with outcome indicators
# This tests equivalence between:
#   A) Joint multivariate fit (fit_multi$joint_fit) with shared global Z
#   B) Long-format indicator-based fit (fit_long) with SAME shared random effects
# 
# The long format combines both outcomes into one 'outcome' variable,
# uses response indicators for separate coefficients per outcome type via:
#   outcome ~ 0 + response_sbp*(bmi+age) + response_hba1c*(HDL+age)
# and shares the same random intercept ~ 1 | patient.
## ============================================================

cat("\n=== TEST 3: Long format with outcome indicators ===\n")

## Correctly reshape dat2 to long format
# Each row in original becomes TWO rows: one for sbp, one for hba1c
dat2_long <- data.frame(
  patient = rep(dat2$patient),
  response_sbp = c(rep(1L, N), rep(0L, N)),        # First N rows are SBP
  response_hba1c = c(rep(0L, N), rep(1L, N)),      # Last N rows are HBA1c
  outcome = c(dat2$sbp, dat2$hba1c),                # Stack the outcomes
  bmi = rep(dat2$bmi),                    # Repeat twice for stacking
  age = rep(dat2$age),
  HDL = rep(dat2$HDL)
)

cat("Long format observations:", nrow(dat2_long), "\n")
cat("Response variable 'outcome' range:", range(dat2_long$outcome), "\n")
cat("Indicator counts - response_sbp sum:", sum(dat2_long$response_sbp), 
    "response_hba1c sum:", sum(dat2_long$response_hba1c), "\n")

# Verify indicator structure is correct
cat("Rows with response_sbp = 1, response_hba1c = 0:", sum(dat2_long$response_sbp == 1 & dat2_long$response_hba1c == 0), "\n")
cat("Rows with response_hba1c = 1, response_sbp = 0:", sum(dat2_long$response_hba1c == 1 & dat2_long$response_sbp == 0), "\n")

## For TEST 3, fit a model where:
## - outcome is the response variable  
## - Separate intercepts for each outcome type via '0 +' (no baseline)
## - Separate slopes for predictors per outcome type via indicator interactions
## - SHARED random intercept ~ 1 | patient (same global Z for both outcomes)
## - heteroscedastic: var_pred = 0 + response_sbp + response_hba1c
## This should match fit_multi$joint_fit exactly.

fit_long <- fit_mixed_simple(
    formula = outcome ~ 0 + response_sbp+ response_sbp:bmi+ response_sbp:age + response_hba1c + response_hba1c:HDL + response_hba1c:age,
    random = ~ 0 + response_sbp+ response_sbp:bmi+ response_sbp:age + response_hba1c + response_hba1c:HDL + response_hba1c:age | patient,
    var_pred = ~ 0 + response_sbp + response_hba1c,
    data = dat2_long,
    engine = "C++",
    maxit = 500,
    verbose = FALSE
  )

cat("\nLong model logLik:", fit_long$logLik, "\n")
cat("Joint model logLik:  ", fit_multi$joint_fit$logLik, "\n")
cat("Difference in logLik:", (fit_multi$joint_fit$logLik - fit_long$logLik), "\n")
cat("\nNote: These two models should have identical or nearly identical logLik\n")


## Should be identical (up to numerical precision) since they are the same model in different parameterizations
fit_multi$joint_fit$opt_par
fit_long$opt_par

## Should theoretically be identical (up to numerical precision) since they are the same model in different parameterizations
fit_multi$joint_fit$reml_fun(fit_multi$joint_fit$opt_par)
fit_long$reml_fun(fit_multi$joint_fit$opt_par)


## test with a specific theta_init to verify that the two likelihood functions are equivalent
theta_init = c(-1,-1,-2,-1,-1,-1, # 6 random effects SDs (log scale)
  0.2, 0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.2, 0.2, # 15 correlations (Fisher z scale)
  0.5, 0.4) # 2 residual SDs (log scale)
fit_long$reml_fun(theta_init)
fit_multi$joint_fit$reml_fun(theta_init)

library(nlme)
## I believe this should be the same as the long format fit above, but using lme() from nlme package. I believe parameters are scaled a bit differently though due to varIdent
fit_lme = lme(
  fixed = outcome ~ 0 + response_sbp+ response_sbp:bmi+ response_sbp:age + response_hba1c + response_hba1c:HDL + response_hba1c:age,
  random = ~ 0 + response_sbp + response_sbp:bmi+ response_sbp:age + response_hba1c + response_hba1c:HDL + response_hba1c:age | patient,
  weights = varIdent(form = ~ 1 | response_sbp - 1),
  data = dat2_long,
  method = "REML",
  control = lmeControl(maxIter = 500, msMaxIter = 500, niterEM = 500, returnObject = TRUE)
)

summary(fit_lme)
