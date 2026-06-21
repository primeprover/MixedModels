## ============================================================
## Test script: mixed model simulation + lme comparison
## ============================================================

library(nlme)
library(lme4)

source("MixedModel.R")

set.seed(123)

## ----------------------------
## Simulation parameters
## ----------------------------
n_patients <- 10000
min_obs <- 3
max_obs <- 100

# Fixed effects
c0 <- 120
c1 <- 0.8   # bmi
c2 <- 0.5   # age

# Random effects covariance (bmi, age)
rho_b <- 0.2
sd_bmi <- 0.5
sd_age <- 0.4
Sigma_b <- matrix(
  c(sd_bmi^2, rho_b * sd_bmi * sd_age,
    rho_b * sd_bmi * sd_age, sd_age^2),
  nrow = 2
)

# Residual SD
sigma_eps <- 5

## ----------------------------
## Simulate per-patient structure
## ----------------------------
n_obs_per_patient <- sample(
  min_obs:max_obs,
  n_patients,
  replace = TRUE
)

patient_id <- rep(seq_len(n_patients), times = n_obs_per_patient)
N <- length(patient_id)

## ----------------------------
## Covariates
## ----------------------------
bmi <- rnorm(N, mean = 27, sd = 4)
age <- rnorm(N, mean = 55, sd = 10)

## ----------------------------
## Random effects
## ----------------------------
b_re <- MASS::mvrnorm(n_patients, mu = c(0, 0), Sigma = Sigma_b)

bmi_re <- b_re[, 1]
age_re <- b_re[, 2]

## Expand random effects to observation level
bmi_re_obs <- bmi_re[patient_id]
age_re_obs <- age_re[patient_id]

## ----------------------------
## Linear predictor + response
## ----------------------------
eta <- c0 +
  (c1 + bmi_re_obs) * bmi +
  (c2 + age_re_obs) * age

sbp <- eta + rnorm(N, sd = sigma_eps)

## ----------------------------
## Assemble data frame
## ----------------------------
dat <- data.frame(
  sbp = sbp,
  bmi = bmi,
  age = age,
  patient = factor(patient_id)
)

rm(patient_id, bmi_re_obs, age_re_obs)  # free memory

## ============================================================
## Reference fit using nlme::lme
## ============================================================

cat("Fitting reference lme model...\n")

fit_lme <- lme(
  fixed = sbp ~ bmi + age,
  random = ~ 1 + bmi + age | patient,
  data = dat,
  method = "REML",
  control = lmeControl(
    opt = "optim",
    msMaxIter = 100
  )
)

print(summary(fit_lme))

fit_new <- fit_mixed_simple(
  formula = sbp ~ bmi + age,
  random  = ~ 1 + bmi + age | patient,
  data    = dat
)

compare_random_effects <- function(G_new, VarCorr_lme, fit_lme) {
  sd_new <- sqrt(diag(G_new))
  cor_new <- cov2cor(G_new)

  sd_lme <- as.numeric(sqrt(diag(getVarCov(fit_lme))))

  vcov_re <- nlme::getVarCov(fit_lme, type = "random.effects")
  cor_lme <- cov2cor(vcov_re)

  sd_compare <- cbind(lme = sd_lme, new = sd_new)
  sd_compare <- cbind(sd_compare, diff = sd_compare[, "new"] - sd_compare[, "lme"])

  list(sd_compare = sd_compare, cor_compare = list(lme = cor_lme, new = cor_new))
}

# Extract fixed effects comparison
cat("\n--- Fixed effects comparison ---\n")
print(cbind(
  lme = fixed.effects(fit_lme),
  new = fit_new$beta
))

# Variance and random effects comparison
varcomp <- compare_random_effects(fit_new$G, VarCorr(fit_lme), fit_lme)
cat("\n--- Random effects SD comparison ---\n")
print(varcomp$sd_compare)

cat("\n--- Random effects correlation comparison ---\n")
cat("lme correlations:\n")
print(varcomp$cor_compare$lme)
cat("\nnew correlations:\n")
print(varcomp$cor_compare$new)

# Residual SD comparison
cat("\n--- Residual SD comparison ---\n")
cat("lme residual SD:", sigma(fit_lme), "\n")
cat("new residual SD:", fit_new$sigma, "\n")

cat("\n--- REML logLik comparison (no penalty) ---\n")
cat("lme:", as.numeric(logLik(fit_lme)), "\n")
cat("new:", fit_new$logLik, "\n")
cat("difference:", as.numeric(logLik(fit_lme)) - fit_new$logLik, "\n")

cat("\nPenalty contribution (new model):\n")
print(fit_new$penalty)


## -----------------------------
## Basic self-test for fit_mixed_simple
## -----------------------------
basic_test_fit_mixed_simple <- function(n_groups = 50, min_obs = 3, max_obs = 8, seed = 101) {
  set.seed(seed)
  n_obs_per_group <- sample(min_obs:max_obs, n_groups, replace = TRUE)
  gid <- rep(seq_len(n_groups), times = n_obs_per_group)
  Nloc <- length(gid)
  bmi_loc <- rnorm(Nloc, 27, 4)
  age_loc <- rnorm(Nloc, 55, 10)

  # random effects
  b_re_loc <- MASS::mvrnorm(n_groups, mu = c(0, 0), Sigma = Sigma_b)
  bmi_re_loc <- b_re_loc[, 1][gid]
  age_re_loc <- b_re_loc[, 2][gid]

  eta_loc <- c0 + (c1 + bmi_re_loc) * bmi_loc + (c2 + age_re_loc) * age_loc
  yloc <- eta_loc + rnorm(Nloc, sd = sigma_eps)

  dat_loc <- data.frame(sbp = yloc, bmi = bmi_loc, age = age_loc, patient = factor(gid))

  res <- tryCatch(
    fit_mixed_simple(sbp ~ bmi + age, ~ 1 + bmi + age | patient, dat_loc, maxit = 50, verbose = FALSE),
    error = function(e) e
  )

  if (inherits(res, "error")) {
    cat("basic_test: error during fit:\n")
    print(res)
    return(FALSE)
  }

  ok <- is.finite(res$logLik) && all(is.finite(res$beta)) && !is.na(res$optim_convergence)
  cat("basic_test: optim_convergence:", res$optim_convergence, "\n")
  cat("basic_test: success:", ok, "\n")
  invisible(list(ok = ok, result = res))
}

## ======================================================================
## TEST C++ ENGINE AGAINST R ENGINE
## ======================================================================
cat("\n\n=== DUAL ENGINE COMPARISON TEST ===\n")
cat("Testing R engine vs C++ engine on same data and parameters...\n\n")

# Test with R engine
cat("--- Testing R engine ---\n")
cat("Disabled R engine test for now due to performance issues on large datasets.\n")
# time_r <- system.time({
#   fit_r <- tryCatch(
#     fit_mixed_simple(
#       formula = sbp ~ bmi + age,
#       random  = ~ 0 + bmi + age | patient,
#       data    = dat,
#       engine  = "R"
#     ),
#     error = function(e) {
#       cat("Error in R engine:", as.character(e), "\n")
#       NULL
#     }
#   )
# })

# if (!is.null(fit_r)) {
#   cat("R engine: logLik =", fit_r$logLik, "\n")
#   cat("R engine: penalty =", fit_r$penalty, "\n")
#   cat("R engine: convergence =", fit_r$optim_convergence, "\n")
#   cat("R engine: elapsed time =", round(time_r["elapsed"], 3), "sec\n")
# } else {
#   cat("R engine FAILED\n")
# }

# Test with C++ engine
cat("\n--- Testing C++ engine ---\n")
time_cpp <- system.time({
  fit_cpp <- tryCatch(
    fit_mixed_simple(
      formula = sbp ~ bmi + age,
      random  = ~ 1 + bmi + age | patient,
      data    = dat,
      engine  = "C++"
    ),
    error = function(e) {
      cat("Error in C++ engine:", as.character(e), "\n")
      NULL
    }
  )
})

# Time with lme
cat("\n--- Timing LME ---\n")
time_lme <- system.time({
  fit_lme <- lme(
      sbp ~ bmi + age,
      random  = ~ 1 + bmi + age | patient,
      data    = dat
    )
})

# Time with lmer
cat("\n--- Timing LMER ---\n")
time_lmer <- system.time({
  fit_lmer <- lmer(
      sbp ~ bmi + age + (1 + bmi + age | patient),
      data    = dat
    )
})

if (!is.null(fit_cpp)) {
  cat("C++ engine: logLik =", fit_cpp$logLik, "\n")
  cat("C++ engine: penalty =", fit_cpp$penalty, "\n")
  cat("C++ engine: convergence =", fit_cpp$optim_convergence, "\n")
  cat("C++ engine: elapsed time =", round(time_cpp["elapsed"], 3), "sec\n")
} else {
  cat("C++ engine FAILED\n")
}

cat("\n--- TIMING COMPARISON ---\n")
cat("C++ engine:", round(time_cpp["elapsed"], 3), "sec\n")
cat("LME:", round(time_lme["elapsed"], 3), "sec\n")
cat("LMER:", round(time_lmer["elapsed"], 3), "sec\n")

# # Compare results if both succeeded
# if (!is.null(fit_cpp)) {
#   # cat("\n--- ENGINE COMPARISON ---\n")
#   # cat("logLik difference (R - C++):", fit_r$logLik - fit_cpp$logLik, "\n")
#   # cat("Beta difference (max absolute):", max(abs(fit_r$beta - fit_cpp$beta)), "\n")
#   # cat("G matrix difference (max absolute):", max(abs(fit_r$G - fit_cpp$G)), "\n")
#   # cat("Sigma difference:", abs(fit_r$sigma - fit_cpp$sigma), "\n")

#   cat("\n--- TIMING COMPARISON ---\n")
#   # cat("R engine:  ", round(time_r["elapsed"], 3), "sec\n")
#   cat("C++ engine:", round(time_cpp["elapsed"], 3), "sec\n")
#   # speedup <- time_r["elapsed"] / time_cpp["elapsed"]
#   # cat("Speedup (R / C++):", round(speedup, 2), "x\n")
#   cat("LME:", round(time_lme["elapsed"], 3), "sec\n")
#   cat("LMER:", round(time_lmer["elapsed"], 3), "sec\n")

#   # Use realistic numerical tolerance (1e-4 for logLik, 1e-8 for parameters)
#   loglik_tol <- 1e-4
#   param_tol <- 1e-6

#   # if (abs(fit_r$logLik - fit_cpp$logLik) < loglik_tol &&
#   #       max(abs(fit_r$beta - fit_cpp$beta)) < param_tol &&
#   #       max(abs(fit_r$G - fit_cpp$G)) < param_tol) {
#   #   cat("\n*** DUAL ENGINE VALIDATION: PASSED ***\n")
#   #   cat("Numerical differences within tolerance (logLik=", loglik_tol, ", params=", param_tol, ")\n")
#   # } else {
#   #   cat("\n*** DUAL ENGINE VALIDATION: DIFFERENCES TOO LARGE ***\n")
#   # }
# }
print(fit_cpp$timings[4:9])

th <- c(1, 2, 4, 8, 16)
times <- numeric(length(th))
for (t_idx in seq_along(th)) {
  t <- th[t_idx]
  set_threads_cpp(t)
  time_cpp <- system.time({
    fit_cpp <- tryCatch(
      fit_mixed_simple(
        formula = sbp ~ bmi + age,
        random  = ~ 1 + bmi + age | patient,
        data    = dat,
        engine  = "C++"
      ),
      error = function(e) {
        cat("Error in C++ engine:", as.character(e), "\n")
        NULL
      }
    )
  })
  cat("Threads:", t, "Elapsed time:", round(time_cpp["elapsed"], 3), "sec\n")
  times[t_idx] <- time_cpp["elapsed"]
}
print(cbind(th, times))


