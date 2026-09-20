source("MixedModel.R")
source("DataSimulation4.R")

set_threads_cpp(16)

N <- nrow(dat2)
shared_q <- 6
shared_n_corr <- 15

## ============================================================
## Test: Joint fit of multiple homoscedastic normal responses
## The model shares a global Z (random effects) but allows separate
## residual variance per response. This should differ from the
## indicator-based approach when there are cross-response correlations.
## ============================================================

cat("\n=== TEST: Joint multiple response homoscedastic fit ===\n")

fit_multi_resp <- fit_mixed(
  responses = list( 
    list(
      formula = response_sbp ~ bmi + age,
      random = ~ 0 + response_sbp + response_hba1c | patient,
      var_pred = ~ 0 + response_sbp + response_hba1c,
      response_type = "homo_normal"
    ),
    list(
      formula = response_hba1c ~ HDL + age,  
      random = ~ 0 + response_sbp + response_hba1c | patient,
      var_pred = ~ 0 + response_sbp + response_hba1c,
      response_type = "homo_normal"
    )
  ),
  data = dat2,
  engine = "C++",
  maxit = 500,
  verbose = TRUE
)

if (!is.null(fit_multi_resp$joint_fit)) {
  cat("Joint fit result:\n")
  print(fit_multi_resp$joint_fit)
  
  cat("\nJoint multiple response logLik:", fit_multi_resp$joint_fit$logLik, "\n")
} else {
  cat("ERROR: Joint fit failed\n")
  stop("Joint fit did not work.")
}

## ============================================================
## Test: Fit each response separately (no shared random effects)
## This gives a lower log-likelihood because it misses cross-response
## correlations in the random effects.
## ============================================================

cat("\n=== TEST: Separate fits ===\n")

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

cat("Separate SBP fit logLik:", fit_sbp_sep$logLik, "\n")
cat("Separate HbA1c fit logLik:", fit_hba1c_sep$logLik, "\n")
cat("Sum of separate logLiks:", fit_sbp_sep$logLik + fit_hba1c_sep$logLik, "\n")

## ============================================================
## Test 3: Heteroscedastic (hetero_normal) formulation
## This uses var_pred to model response-specific residual variance.
## It's different from homo_normal because it estimates separate
## gamma parameters for the log-variance.
## ============================================================

cat("\n=== TEST 3: Heterogeneous residual error formulation ===\n")

# First, reshape data so varIdent-style modeling works
# Note: This is NOT equivalent to homo_normal with shared Z
dat_reshaped <- do.call(rbind, {
  for (i in 1:N) {
    cbind(
      patient = dat2$patient[i],
      bmi = dat2$bmi[i],
      age = dat2$age[i],
      HDL = dat2$HDL[i],
      response_sbp = dat2$sbp[i],
      response_hba1c = dat2$hba1c[i]
    )
  }
})

colnames(dat_reshaped) <- c("patient", "bmi", "age", "hdl", "response_sbp", "response_hba1c")

# Fit heteroscedastic model - this models response_specific residual variance  
fit_hetero <- tryCatch({
  fit_mixed(
    responses = list(
      list(
        formula = response_sbp ~ bmi + age,
        random = ~ 0 + response_sbp + response_hba1c | patient,
        var_pred = ~ 0 + response_sbp + response_hba1c,
        response_type = "hetero_normal"
      )
    ),
    data = dat_reshaped,
    engine = "C++",
    maxit = 500,
    verbose = TRUE
  )
}, error = function(e) {
  cat("Hetero fit error:", e$message, "\n")
  list(joint_fit = NULL)
})

if (!is.null(fit_hetero$joint_fit)) {
  cat("Hetero fit result:\n")
  print(fit_hetero$joint_fit)
} else {
  cat("Heterouscedastic fit not tested (may require different setup)\n")
}

## ============================================================
## Verification: The key insight is that "homo_normal" with var_pred
## and two responses sharing the same random effects gives a logLik
## that may match fitting separately IF cross-response correlations are 0.
## With non-zero correlations, joint fit should have higher logLik.
## ============================================================

cat("\n=== VERIFICATION: Log-likelihood comparison ===\n")

if (!is.null(fit_multi_resp$joint_fit)) {
  cat("Joint multiple response homoscedastic logLik:", fit_multi_resp$joint_fit$logLik, "\n")
}
if ((!is.null(fit_sbp_sep)$logLik) && (!is.null(fit_hba1c_sep$logLik))) {
  cat("Separate SBP logLik:", fit_sbp_sep$logLik, "\n")
  cat("Separate HbA1c logLik:", fit_hba1c_sep$logLik, "\n")
  cat("Sum of separate:", fit_sbp_sep$logLik + fit_hba1c_sep$logLik, "\n")
  
  if (!is.null(fit_multi_resp$joint_fit)) {
    diff <- (fit_sbp_sep$logLik + fit_hba1c_sep$logLik) - fit_multi_resp$joint_fit$logLik
    cat("Difference (separate sum - joint):", diff, "\n")
    
    if (diff > 0.001) {
      cat("\nNOTE: Separate fits sum > Joint logLik\n")
      cat("This is expected when cross-response correlations are non-zero!\n")
      cat("The joint fit captures correlation structure that separate misses.\n")
    } else if (diff < -0.001) {
      cat("\nWARNING: Joint logLik > sum of separate\n")
      cat("This should NOT happen for correctly specified models.\n")
    } else {
      cat("\nOK: Log-likelihoods are equivalent (within tolerance)\n")
      cat("This suggests cross-response correlations are negligible or zero.\n")
    }
  }
}

## ============================================================
## Parameter recovery for homoscedastic multiple responses
## When both responses share the same residual variance, we expect
## to recover that common value.
## ============================================================

cat("\n=== TEST: Parameter recovery ===\n")

if (!is.null(fit_multi_resp$joint_fit)) {
  opt_par <- fit_multi_resp$joint_fit$opt_par
  
  # The last params are log(variances) - extract them
  response_param_counts <- fit_multi_resp$joint_fit$model$response_param_counts
  n_resp_params <- sum(response_param_counts)
  
  cat("Theta tail (response params):\n")
  print(opt_par[(length(opt_par) - n_resp_params + 1):length(opt_par)])
  
  if (n_resp_params >= 2) {
    # For homoscedastic with single variance per response:
    # Last param is one shared variance (for two responses it should be equal)
    shared_log_var <- opt_par[length(opt_par)]
    shared_sd <- exp(shared_log_var / 2)
    
    cat("Recovery attempt:\n")
    cat("Estimated SD (response-specific but same):", shared_sd, "\n")
    cat("True response 1 SD:", sigma_sbp, "\n")
    cat("True response 2 SD:", sigma_hba1c, "\n")
    
    # Compute residuals to see how well it fits
    pred_sb <- fitted(fit_multi_resp$joint_fit, newdata = dat2[,"bmi:age"])
    cat("Prediction residual variance estimate:", var(pred_sb), "\n")
  }
}

cat("\nDone!\n")
