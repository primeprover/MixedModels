#!/usr/bin/env Rscript
# =============================================================================
# Demonstration: Multiple homoscedastic normal responses vs indicator-based approach
# =============================================================================
# 
# This script clarifies the relationship between different model formulations:
# 
# 1. Multiple homoscedastic normal responses (homo_normal with var_pred=NULL)
#    - Single residual variance shared across all observations
#    - Multivariate response y = (y1, y2, ..., yn) ~ MVN(mu(Y), Var(Y|Beta,B)) where:
#      Var(Y) = Sigma ⊗ I + L diag(sigma²) L^T with ONE sigma² for all responses
#
# 2. Indicator-based heterogeneous modeling (hetero_normal with var_pred)
#    - This uses model.matrix(var_pred, data) to split observations into groups
#    - Each fitted group has its own gamma parameter for log(variance)
#    - NOT the same as multiple normal responses in lme with varIdent!
#
# 3. What the user might actually want: fitting each response independently
#    in sequence (indicator-based). This would be separate fits, not one joint model.
#
# Key insight: There's NO single "varIdent-style" homoscedastic multiple response
# formulation that shares global Z - that only exists when varPred=NULL and the
# observations are truly multivariate responses.
# =============================================================================

source("MixedModel.R")
source("DataSimulation4.R")

set_threads_cpp(16)

## ============================================================
## SETUP
## ============================================================

N <- nrow(dat2)
shared_q <- 6  
shared_n_corr <- 15
tolerance <- 0.001

## ============================================================
## SECTION 1: Multiple homoscedastic normal responses (joint fit)
# This is the current implementation - it fits a multivariate response model
# with shared random effects and ONE residual variance parameter.
## ============================================================

cat("\n=============================================================================\n")
cat("SECTION 1: Multiple Homoscedastic Normal Responses\n")
cat("(homo_normal response_type, var_pred=NULL)\n")
cat("=============================================================================\n")

fit_multi_homo <- fit_mixed(
  responses = list( 
    list(
      formula = response_sbp ~ bmi + age,
      random = ~ 0 + response_sbp | patient,
      var_pred = NULL,  # homoscedastic: single residual variance
      response_type = "homo_normal"
    ),
    list(
      formula = response_hba1c ~ HDL + age,  
      random = ~ 0 + response_hba1c | patient,
      var_pred = NULL,
      response_type = "homo_normal"
    )
  ),
  data = dat2,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

if (!is.null(fit_multi_homo$joint_fit)) {
  cat("Joint fit logLik:", fit_multi_homo$joint_fit$logLik, "\n")
  
  opt_par <- fit_multi_homo$joint_fit$opt_par
  cat("\nEstimated parameters:\n")
  print(opt_par)
  cat("\nLast param (residual variance):\n")
  cat("log(var_residual):", opt_par[length(opt_par]), "\n")
  cat("sqrt(var_residual):", exp(0.5 * opt_par[length(opt_par)]), "\n")
}

## ============================================================
## SECTION 2: Separate fits (no shared Z between responses)
# This is NOT the same as joint fit when cross-response correlations exist.
## ============================================================

cat("\n=============================================================================\n")
cat("SECTION 2: Separate Fits\n")
cat("(Each response fitted independently, no shared random effects)\n")
cat("=============================================================================\n")

fit_sbp_sep <- fit_mixed_simple(
  formula = response_sbp ~ bmi + age,
  random = ~ 0 + response_sbp | patient,
  var_pred = NULL,
  data = dat2,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

fit_hba1c_sep <- fit_mixed_simple(
  formula = response_hba1c ~ HDL + age,
  random = ~ 0 + response_hba1c | patient,
  var_pred = NULL,
  data = dat2,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

cat("Separate SBP logLik:", fit_sbp_sep$logLik, "\n")
cat("Separate HbA1c logLik:", fit_hba1c_sep$logLik, "\n")
cat("Sum of separate:", fit_sbp_sep$logLik + fit_hba1c_sep$logLik, "\n")

## ============================================================
## SECTION 3: What does var_pred mean?
# In hetero_normal, var_pred ~ 0 + response_sbp + response_hba1c on reshaped
# data creates columns for each response indicator. Each column gets a gamma
# parameter - this models RESPONSE-SPECIFIC residual variance, not the same as
# multiple homoscedastic normal responses!
## ============================================================

cat("\n=============================================================================\n")
cat("SECTION 3: Understanding var_pred in hetero_normal\n")
cat("=============================================================================\n")

cat("In hetero_normal_response:\n")
cat("  - W = model.matrix(var_pred, data) creates indicator columns\n")
cat("  - gamma parameters estimate LOG-VARIANCE for each indicator\n")
cat("  - This is a HETEROSCEDASTIC variance model (like varIdent in nlme)\n\n")

cat("If you reshape data with:\n")
cat("  dat_reshaped <- bind_rows(dat2$sbp, dat2$hba1c)") 
cat("  # Each row has response indicators\n\n")

cat("Then fit_mixed with response_type = 'hetero_normal' and varPred ~ response_indicators\n")
cat("will estimate SEPARATE residual variance for each indicator.\n\n")

## ============================================================
## SECTION 4: The user's actual question
# "Does fitting multiple responses equal fitting with separate indicators per response?"
# Answer: NOT the same when there are cross-response correlations in random effects.
# With shared global Z and zero cross-correlation, they ARE equivalent. 
## ============================================================

cat("\n=============================================================================\n")
cat("SECTION 4: Equivalence Test\n")
cat("=============================================================================\n")

cat("Question: Is fitting multiple homoscedastic normal responses the same as\n")
cat("fitting with indicators for every response? Answer:\n\n")
cat("When cross-response correlations are ZERO → YES, equivalent logLik\n")  
cat("When cross-response correlations are NON-ZERO → NO, they differ\n\n")

## Demonstrate by simulating uncorrelated random effects
set.seed(123)
n_patients_demo <- 5000

# Truly independent random effects (no cross-corr)
re_sbp_demo <- rnorm(n_patients_demo * 3, sd = c(sd_sbp_intercept, sd_sbp_slope, sd_sbp_age))
re_hba1c_demo <- rnorm(n_patients_demo * 3, sd = c(sd_hba1c_intercept, sd_hba1c_slope, sd_hba1c_age))

eta_demo_sbp <- sbp_intercept + sbp_bmi * bmi + sbp_age * age + re_sbp_demo[patient_id, 1]
eta_demo_hba1c <- hba1c_intercept + hba1c_hdl * HDL + hba1c_age * age + re_hba1c_demo[patient_id, 2]

sbp_demo <- eta_demo_sbp + rnorm(N, sd = sigma_sbp)
hba1c_demo <- eta_demo_hba1c + rnorm(N, sd = sigma_hba1c)

dat_demo <- data.frame(
  sbp = sbp_demo, hba1c = hba1c_demo, bmi = bmi, age = age, HDL = HDL, patient = factor(patient_id)
)

## Joint multiple homoscedastic normal response fit (zero cross-corr in random effects)
fit_demo_joint <- fit_mixed(
  responses = list( 
    list(
      formula = response_sbp ~ bmi + age,
      random = ~ 0 + response_sbp | patient,
      var_pred = NULL,
      response_type = "homo_normal"
    ),
    list(
      formula = response_hba1c ~ HDL + age,  
      random = ~ 0 + response_hba1c | patient,
      var_pred = NULL,
      response_type = "homo_normal"
    )
  ),
  data = dat_demo,
  engine = "C++",
  maxit = 300,
  verbose = FALSE
)

## Separate fits (same as above for each)
fit_demo_sbp <- fit_mixed_simple(
  formula = response_sbp ~ bmi + age,
  random = ~ 0 + response_sbp | patient,
  var_pred = NULL,
  data = dat_demo,
  engine = "C++",
  maxit = 300,
  verbose = FALSE
)

fit_demo_hba1c <- fit_mixed_simple(
  formula = response_hba1c ~ HDL + age,
  random = ~ 0 + response_hba1c | patient,
  var_pred = NULL,
  data = dat_demo,
  engine = "C++",
  maxit = 300,
  verbose = FALSE
)

cat("Demo with independent (uncorrelated) random effects:\n")
cat("Joint multiple response logLik:", fit_demo_joint$joint_fit$logLik, "\n")
cat("Separate SBP logLik:", fit_demo_sbp$logLik, "\n")
cat("Separate HbA1c logLik:", fit_demo_hba1c$logLik, "\n")
cat("Sum of separate:", fit_demo_sbp$logLik + fit_demo_hba1c$logLik, "\n")
diff_zero <- abs((fit_demo_sbp$logLik + fit_demo_hba1c$logLik) - fit_demo_joint$joint_fit$logLik)
cat("\nDifference (should be near zero):", diff_zero, "\n")

if (diff_zero < tolerance) {
  cat("✓ PASS: With uncorrelated random effects, logLik are equivalent\n")
} else {
  cat("✗ FAIL: Difference too large (> tolerance)\n")
}

## ============================================================
## SECTION 5: What about correlated random effects?
# Now demonstrate that when there ARE cross-response correlations in Z,
# the joint fit differs from separate fits.
## ============================================================

cat("\n=============================================================================\n")
cat("SECTION 5: With Correlated Random Effects (Different Behavior)\n")
cat("=============================================================================\n")

# The original truth has non-zero correlations within each response block
set.seed(42)
dat_orig <- dat2

## Joint fit with shared Z and correlations
fit_with_corr <- fit_mixed(
  responses = list( 
    list(
      formula = response_sbp ~ bmi + age,
      random = ~ 0 + response_sbp + response_hba1c | patient,  # Shared Z
      var_pred = NULL,
      response_type = "homo_normal"
    ),
    list(
      formula = response_hba1c ~ HDL + age,  
      random = ~ 0 + response_sbp + response_hba1c | patient,
      var_pred = NULL,
      response_type = "homo_normal"
    )
  ),
  data = dat_orig,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

## Separate fits (no cross-response correlation)
fit_sbp_orig <- fit_mixed_simple(
  formula = response_sbp ~ bmi + age,
  random = ~ 0 + response_sbp | patient,
  var_pred = NULL,
  data = dat_orig,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

fit_hba1c_orig <- fit_mixed_simple(
  formula = response_hba1c ~ HDL + age,
  random = ~ 0 + response_hba1c | patient,
  var_pred = NULL,
  data = dat_orig,
  engine = "C++",
  maxit = 500,
  verbose = FALSE
)

cat("With correlated random effects in shared Z:\n")
cat("Joint multiple response logLik:", fit_with_corr$joint_fit$logLik, "\n")
cat("Separate SBP logLik:", fit_sbp_orig$logLik, "\n")
cat("Separate HbA1c logLik:", fit_hba1c_orig$logLik, "\n")
cat("Sum of separate:", fit_sbp_orig$logLik + fit_hba1c_orig$logLik, "\n")

diff_corr <- abs((fit_sbp_orig$logLik + fit_hba1c_orig$logLik) - fit_with_corr$joint_fit$logLik)
cat("\nDifference:", diff_corr, "\n")

if (diff_corr > 0.001) {
  cat("✓ EXPECTED: With correlated random effects, joint fit differs from separate fits\n")
  cat("  The joint fit captures cross-response correlation structure that separate ignores.\n")
} else {
  cat("Note: Log-likelihoods are essentially equal (correlations negligible)\n")
}

## ============================================================
## SUMMARY
## ============================================================

cat("\n=============================================================================\n")
cat("SUMMARY\n")
cat("=============================================================================\n")

cat("Multiple homoscedastic normal response formulation (homo_normal with var_pred=NULL)\n")
cat("fits a multivariate model with:\n")
cat("  - ONE residual variance parameter shared across all observations\n")  
cat("  - A single random effects covariance matrix for all responses\n")
cat("  - Multivariate likelihood\n\n")

cat("This is NOT equivalent to indicator-based modeling because that would be\n")
cat("heteroscedastic (varIdent-style) with separate variance per response.\n\n")

cat("If your goal is homoscedastic multiple normal responses, use:\n")
cat("  - homogeneous formulation (homo_normal)\n")
cat("  - var_pred = NULL or ~ 0 only (single variance indicator column not needed)\n\n")

cat("The shared global Z with separate residual variances per response is the\n")
cat("appropriate homoscedastic multiple normal response formulation.\n\n")

cat("=============================================================================\n")

cat("\nDone!\n")
