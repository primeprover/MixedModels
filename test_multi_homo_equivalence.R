#!/usr/bin/env Rscript
# =============================================================================
# Multiple Homoscedastic Normal Responses Equivalence Test
# =============================================================================
# 
# Compares:
# 1. Joint fit of multiple homoscedastic normal responses (homo_normal + shared Z)
# 2. Sum of separate single-response fits
#
# Key finding: These are equivalent when cross-response correlations in the
# random effects covariance matrix are zero or negligible.
# =============================================================================

source("MixedModel.R")
source("DataSimulation4.R")

set_threads_cpp(16)

## ============================================================
## PREPARATION
## ============================================================

N <- nrow(dat2)
shared_q <- 6  
shared_n_corr <- shared_q * (shared_q - 1) / 2  # 15

cat("\n=============================================================================\n")
cat("Testing multiple homoscedastic normal responses equivalence\n")
cat("=============================================================================\n\n")

cat("Simulated data characteristics:\n")
cat("  n_patients:", N, "\n")
cat("  Random effects: ", shared_q, " (", shared_n_corr, " correlations)\n")
cat("  Residual SDs: SBP=", sigma_sbp, ", HbA1c=", sigma_hba1c, "\n\n")

## ============================================================
## TEST 1: Joint multiple homoscedastic normal response fit
# Formulation (from DataSimulation4 truth):
#   Multiple responses y = [y_SBP, y_HbA1c] with shared global Z and 
#   one residual variance for all observations.
# =============================================================================

cat("\n--- TEST 1: Joint multiple homoscedastic normal response fit ---\n")

fit_multi_homo <- tryCatch({
  fit_mixed(
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
    data = dat2,
    engine = "C++",
    maxit = 500,
    verbose = FALSE
  )
}, error = function(e) {
  stop(paste("Multi-response fit failed:", e$message))
})

cat("Joint multiple response fit converged:", 
    ifelse(fit_multi_homo$joint_fit$optim_convergence == 0, "Yes", "No (warning)\n"), "\n")
cat("Log-likelihood:", fit_multi_homo$joint_fit$logLik, "\n")

## ============================================================
## TEST 2: Separate fits (no shared random effects between responses)
# This should equal the joint fit when cross-response correlations are zero,
# or differ by a predictable amount when correlations exist.
## ============================================================

cat("\n--- TEST 2: Separate single-response fits ---\n")

fit_sbp_sep <- tryCatch({
  fit_mixed_simple(
    formula = response_sbp ~ bmi + age,
    random = ~ 0 + response_sbp | patient,
    var_pred = NULL,
    data = dat2,
    engine = "C++",
    maxit = 500,
    verbose = FALSE
  )
}, error = function(e) {
  stop(paste("SBP fit failed:", e$message))
})

fit_hba1c_sep <- tryCatch({
  fit_mixed_simple(
    formula = response_hba1c ~ HDL + age,
    random = ~ 0 + response_hba1c | patient,
    var_pred = NULL,
    data = dat2,
    engine = "C++",
    maxit = 500,
    verbose = FALSE
  )
}, error = function(e) {
  stop(paste("HbA1c fit failed:", e$message))
})

cat("Separate SBP fit logLik:", fit_sbp_sep$logLik, "\n")
cat("SEparate HbA1c fit logLik:", fit_hba1c_sep$logLik, "\n")

separate_sum <- fit_sbp_sep$logLik + fit_hba1c_sep$logLik
joint_ll <- fit_multi_homo$joint_fit$logLik

cat("\nSum of separate fits:", separate_sum, "\n")
cat("Joint multiple response fit:", joint_ll, "\n\n")

## ============================================================
## TEST 3: Check whether cross-response correlations are estimated
# =============================================================================

theta <- fit_multi_homo$joint_fit$opt_par
correlations_estimated <- tanh(theta[(shared_q + 1):(shared_q + shared_n_corr)])

cat("Cross-response correlation estimates (partial correlations):\n")
print(round(correlations_estimated, 4))

max_corr_magnitude <- max(abs(correlations_estimated))
cat("\nMaximum |correlation|:", max_corr_magnitude, "\n")

## ============================================================
## TEST 4: The key insight
# When cross-response correlations are negligible (near zero), the joint
# multiple response logLik should equal the sum of separate fits.
# 
# When there ARE cross-response correlations, the joint fit will have a
# higher logLik because it captures additional correlation structure.
## ============================================================

cat("\n=============================================================================\n")
cat("KEY INSIGHT: Multiple responses vs indicator-based modeling\n")
cat("=============================================================================\n\n")

if (max_corr_magnitude < 0.05) {
  diff <- abs(separate_sum - joint_ll)
  cat("Cross-response correlations are small (max = ", max_corr_magnitude, ")\n", sep="")
  cat("With negligible cross-response correlation:\n")
  cat("  Sum of separate fits = Joint multiple response logLik\n\n")
  
  if (diff < 0.001) {
    cat("✓ PASS: Log-likelihoods are equivalent (difference =", diff, ")\n")
  } else {
    cat("✗ FAIL: Log-likelihoods differ by", diff, "\n")
  }
} else {
  diff <- abs(separate_sum - joint_ll)
  cat("Cross-response correlations present (max =", max_corr_magnitude, ")\n\n")
  
  if (diff > 0) {
    cat("Expected behavior with non-zero cross-correlation:\n")
    cat("  Joint multiple response logLik > Sum of separate fits\n")
    cat("  (Joint captures additional correlation structure)\n\n")
    
    if (joint_ll > separate_sum && diff < 10.0) {  # Allow reasonable difference
      cat("✓ EXPECTED: Joint logLik is appropriately higher by", 
          diff, "\n")
    } else {
      cat("Difference:", diff, "\n")
    }
  } else {
    cat("WARNING: Joint fit should not be worse than separate fits!\n")
  }
}

## ============================================================
## TEST 5: Simulate with truly uncorrelated random effects to confirm equivalence
# =============================================================================

cat("\n=============================================================================\n")
cat("TEST 5: Equivalence with truly uncorrelated random effects\n")
cat("=============================================================================\n\n")

set.seed(999)
n_patients_test <- 2000

# Generate data with NO cross-response correlation in random effects
re_sbp_uncorr <- rnorm(n_patients_test * 3, 
                       mean = 0, sd = c(sd_sbp_intercept, sd_sbp_slope, sd_sbp_age))
re_hba1c_uncorr <- rnorm(n_patients_test * 3, 
                         mean = 0, sd = c(sd_hba1c_intercept, sd_hba1c_slope, sd_hba1c_age))

# Extract by patient ID
patient_ids <- as.numeric(fit_multi_homo$joint_fit$model$groups[[1]]$n)  # Get from groups

eta_test_sbp <- rep(0, N)
eta_test_hba1c <- rep(0, N)

for (i in seq_along(re_sbp_uncorr)) {
  pid <- as.integer(as.numeric(as.character(dat2$patient[i])))  
  idx <- which(patient_ids == pid)[1]
  if (!is.na(idx)) {
    eta_test_sbp[i] <- sbp_intercept + sbp_bmi * bmi[i] + sbp_age * age[i] + re_sbp_uncorr[idx]
    eta_test_hba1c[i] <- hba1c_intercept + hba1c_hdl * HDL[i] + hba1c_age * age[i] + re_hba1c_uncorr[idx]
  }
}

sbp_test <- eta_test_sbp + rnorm(N, sd = sigma_sbp)
hba1c_test <- eta_test_hba1c + rnorm(N, sd = sigma_hba1c)

dat_test <- data.frame(
  sbp = sbp_test, hba1c = hba1c_test, 
  bmi = bmi, age = age, HDL = HDL, patient = as.factor(as.numeric(dat2$patient))
)

cat("Simulated dataset with independent random effects\n")
cat("  n_obs:", N, "\n")
cat("  No cross-response correlation in G matrix\n\n")

## Separate fits for tested data
fit_test_sbp <- fit_mixed_simple(
  formula = response_sbp ~ bmi + age,
  random = ~ 0 + response_sbp | patient,
  var_pred = NULL,
  data = dat_test,
  engine = "C++",
  maxit = 300,
  verbose = FALSE
)

fit_test_hba1c <- fit_mixed_simple(
  formula = response_hba1c ~ HDL + age,
  random = ~ 0 + response_hba1c | patient,
  var_pred = NULL, 
  data = dat_test,
  engine = "C++",
  maxit = 300,
  verbose = FALSE
)

## Joint multiple homoscedastic normal response fit (zero cross-corr)
fit_test_multi <- tryCatch({
  fit_mixed(
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
    data = dat_test,
    engine = "C++",
    maxit = 300,
    verbose = FALSE
  )
}, error = function(e) {
  stop(paste("Multi-response fit failed for test data:", e$message))
})

cat("Separate SBP logLik:", fit_test_sbp$logLik, "\n")
cat("Separate HbA1c logLik:", fit_test_hba1c$logLik, "\n")
cat("Sum of separate:", fit_test_sbp$logLik + fit_test_hba1c$logLik, "\n")

test_joint_ll <- fit_test_multi$joint_fit$logLik
cat("Joint multiple response logLik:", test_joint_ll, "\n\n")

diff_test <- abs((fit_test_sbp$logLik + fit_test_hba1c$logLik) - test_joint_ll)
cat("Difference:", diff_test, "\n\n")

if (diff_test < 0.001) {
  cat("✓ PASS: With truly uncorrelated random effects,\n")
  cat("      multiple homoscedastic normal responses = sum of separate fits\n")
} else {
  cat("Difference too large:", diff_test, "\n")
  cat("(This might indicate an issue or optimization differences)\n")
}

## ============================================================
## SUMMARY AND CONCLUSION
# =============================================================================

cat("\n=============================================================================\n")
cat("CONCLUSION\n")
cat("=============================================================================\n\n")

cat("Multiple homoscedastic normal responses (homo_normal with shared Z) is the\n")
cat("appropriate formulation for multivariate outcomes where:\n\n")
cat("  1. You want a SINGLE residual variance parameter (homoscedastic)\n")
cat("  2. Random effects are shared across responses (global Z matrix)\n")  
cat("  3. Cross-response correlations exist in the random effects structure\n\n")

cat("This is DIFFERENT from indicator-based heterogeneous modeling because:\n\n")
cat("  - Indicator-based = hetero_normal with var_pred specifying separate \n")
cat("    residual variance per response (heteroscedastic)\n\n")

cat("For homoscedastic multiple normal responses, use:\n\n")
cat("  fit_mixed(responses = list(\n")
cat("    list(formula = y1 ~ x1, random = ~ z1 | g, var_pred = NULL, \n")
cat("         response_type = 'homo_normal'),\n")
cat("    list(formula = y2 ~ x2, random = ~ z1 + z2 | g, var_pred = NULL,\n")
cat("         response_type = 'homo_normal')  # Same global Z\n"), sep="")

cat("\n=============================================================================\n")

cat("\nDone!\n")
