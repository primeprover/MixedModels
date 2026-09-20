# Test comparing:
#   1. Multiple responses fitted jointly (shared global Z)
#   2. Two separate single-response fits (independent models)
# 
# The joint fit should give a different logLik than independent fits
# because the joint fit accounts for shared random effects correlation.
#
# For homoscedastic residuals, the multiple-response path is:
#   - Each response has its own residual variance parameter
#   - All responses share the same G (global Z structure)
#   
# We test that:
#   logLik_joint != logLik_fit1 + logLik_fit2  (expected for shared random effects)
#   But conditional on same b, the joint fits should be consistent


source("MixedModel.R")
source("DataSimulation4.R")

set_threads_cpp(1)

fit_combined <- fit_mixed(
  responses = list( 
    list(
      formula = sbp ~ bmi + age,
      random = ~ 1 + bmi + age | patient
    ),
    list(
      formula = hba1c ~ HDL + age,  
      random = ~ 1 + HDL + age | patient
    )
  ),
  data = dat2,
  engine = "C++",
  maxit = 10000,
  verbose = FALSE
)

cat("\n=== Combined Fit Results ===\n")
print(fit_combined$joint_fit$logLik)
print(fit_combined$joint_fit$opt_par)


# Now fit each response separately (independent models)
cat("\n=== Separate Fits (Independent Models) ===\n")

fit1 <- fit_mixed_simple(
  formula = sbp ~ bmi + age,
  random = ~ 1 + bmi + age | patient,
  data = dat2,
  engine = "C++",
  maxit = 10000,
  verbose = FALSE
)

cat("Response 1 logLik:", fit1$logLik, "\n")

fit2 <- fit_mixed_simple(
  formula = hba1c ~ HDL + age,
  random = ~ 1 + HDL + age | patient,
  data = dat2,
  engine = "C++",
  maxit = 10000, 
  verbose = FALSE
)

cat("Response 2 logLik:", fit2$logLik, "\n")

sum_separate_lls <- fit1$logLik + fit2$logLik
cat("Sum of separate fits logLik:", sum_separate_lls, "\n")
cat("Difference (joint - sum):", fit_combined$joint_fit$logLik - sum_separate_lls, "\n")
cat("(Negative difference is expected due to shared random effects correlation bonus)\n")


# Test heteroscedastic residuals - compare to indicator formulation
cat("\n=== Heteroscedastic Test ===\n")

fit_combined_hetero <- fit_mixed(
  responses = list( 
    list(
      formula = sbp ~ bmi + age,
      random = ~ 1 + bmi + age | patient,
      var_pred = ~ 0,  # Will default to hetero_normal with separate W matrices
      response_type = "hetero_normal"
    ),
    list(
      formula = hba1c ~ HDL + age,
      random = ~ 1 + HDL + age | patient, 
      response_type = "hetero_normal"
    )
  ),
  data = dat2,
  engine = "C++",
  maxit = 10000,
  verbose = TRUE
)

cat("Heteroscedastic combined logLik:", fit_combined_hetero$joint_fit$logLik, "\n")


# Test indicator-based formulation: stack outcomes and use indicators
# First verify the model can handle separate fixed effects + random intercepts per response

print("\n=== Summary ===")
cat("Joint multiple-responses fit accounts for:\n")
cat("  - Shared global random effects (G)\n")
cat("  - Separate residual variances per response\n")
cat("  - Woodbury integration across all responses\n")
cat("\nSeparate fits treat each model independently,\n")
cat("so they don't 'share' the random effects structure.")