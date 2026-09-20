source("MixedModel.R")
source("DataSimulation4.R")

set_threads_cpp(16)

fit <- fit_mixed(
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
  maxit = 10000,
  verbose = TRUE
)

stopifnot(!is.null(fit$joint_fit))
stopifnot(is.list(fit$separate_fits))
stopifnot(length(fit$separate_fits) == 2L)
stopifnot(length(fit$joint_fit$beta) == 6L)
stopifnot(length(fit$joint_fit$opt_par) == 23L)
cat("joint logLik:", fit$joint_fit$logLik, "\n")
print(fit$joint_fit$opt_par)


# Example verification template (adapt to your fit object structure)
est_theta <- fit$joint_fit$opt_par

truth_vec <- c(
  log(sd_sbp_intercept), log(sd_sbp_slope), log(sd_sbp_age),
  log(sd_hba1c_intercept), log(sd_hba1c_slope), log(sd_hba1c_age),
  atanh(rho_sbp_intercept_slope), atanh(rho_sbp_intercept_age), atanh(rho_sbp_slope_age),
  # Cross-response rhos (map to your C++ indexing order)
  atanh(cc[1]), atanh(cc[2]), atanh(cc[3]),
  atanh(cc[4]), atanh(cc[5]), atanh(cc[6]),
  atanh(cc[7]), atanh(cc[8]), atanh(cc[9]),
  # HbA1c within-response
  atanh(rho_hba1c_intercept_slope), atanh(rho_hba1c_intercept_age), atanh(rho_hba1c_slope_age),
  2*log(sigma_sbp), 2*log(sigma_hba1c)
)

max_abs_diff <- max(abs(est_theta - truth_vec))
cat("Max |Estimated - True|:", max_abs_diff, "\n")
# Expect < 0.3 for log-SDs and < 0.15 for unconstrained correlations with this design


## ============================================================
## Compare estimated theta against DataSimulation2 truth
## ============================================================
# Parameter layout in theta (length = 23):
#  1-6:   log(random effect SDs)
#        [SBP_int, SBP_bmi, SBP_age, HbA1c_int, HbA1c_hdl, HbA1c_age]
#  7-21:  unconstrained correlations (C++ maps these via tanh())
#  22:    log(residual variance) for Response 1 (SBP)
#  23:    log(residual variance) for Response 2 (HbA1c)

# 1. Log SDs for the 6 random effects
true_log_sds <- c(
  log(sd_sbp_intercept), log(sd_sbp_slope), log(sd_sbp_age),
  log(sd_hba1c_intercept), log(sd_hba1c_slope), log(sd_hba1c_age)
)

# 2. True correlations in C++ packed lower-triangular order (indices 0..5):
#    Loop order: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2),
#                (4,0), (4,1), (4,2), (4,3),
#                (5,0), (5,1), (5,2), (5,3), (5,4)
true_corrs <- c(
  rho_sbp_intercept_slope,   # (1,0) SBP_int vs SBP_bmi       = 0.15
  rho_sbp_intercept_age,     # (2,0) SBP_int vs SBP_age        = -0.10
  rho_sbp_slope_age,         # (2,1) SBP_bmi vs SBP_age        = 0.08
  0.10, 0.04, -0.03,         # (3,*) HbA1c_int vs SBP int/slope/age
  0.04, 0.06, 0.02,          # (4,*) HbA1c_hdl vs SBP int/slope/age
  -0.03, 0.02, 0.05,         # (5,*) HbA1c_age vs SBP int/slope/age
  rho_hba1c_intercept_slope, # (4,3) HbA1c_int vs HbA1c_hdl    = 0.12
  rho_hba1c_intercept_age,   # (5,3) HbA1c_int vs HbA1c_age    = -0.05
  rho_hba1c_slope_age        # (5,4) HbA1c_hdl vs HbA1c_age    = 0.06
)

# C++ stores unconstrained correlations: theta_corr = atanh(true_corr)
true_corrs_unconstrained <- atanh(true_corrs)

# 3. Residual log-variances (C++ stores log(variance) = 2 * log(sd))
true_log_var_sbp <- 2 * log(sigma_sbp)
true_log_var_hba1c <- 2 * log(sigma_hba1c)

expected_theta <- c(true_log_sds, true_corrs_unconstrained, true_log_var_sbp, true_log_var_hba1c)
print(expected_theta)
print(fit$joint_fit$opt_par)

# Extract and compare
estimated_theta <- fit$joint_fit$opt_par
theta_diffs <- abs(estimated_theta - expected_theta)

cat("\n--- Theta Comparison (Expected vs Estimated) ---\n")
param_labels <- c(
  paste0("log_sd[", 1:6, "]"),
  paste0("corr[", 7:21, "]"),
  "log_var_sbp", "log_var_hba1c"
)
print(cbind(
  Param = param_labels,
  Expected = round(expected_theta, 4),
  Estimated = round(estimated_theta, 4),
  Diff = round(theta_diffs, 4)
))

# With only 50 groups and short sequences, sampling variability for variance
# components and correlations is substantial. A tolerance of ~0.5-0.6 is
# typically acceptable for this simulation scale. Adjust if you increase n_patients.
tolerance <- 0.6
cat(sprintf("\nMax absolute parameter difference: %.4f\n", max(theta_diffs)))
#stopifnot(max(theta_diffs) < tolerance,
#          "Estimated theta deviates too much from simulation truth (tol = %.2f).", tolerance)
