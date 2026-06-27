## ============================================================
## Test script: mixed model simulation + lme comparison
## ============================================================

rm(list=ls())
library(nlme)
library(lme4)

source("MixedModel.R")
source("DataSimulation.R")

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

fit_cpp <- fit_mixed_simple(
  formula = sbp ~ bmi + age,
  random  = ~ 1 + bmi + age | patient,
  data    = dat
)

compare_random_effects <- function(G_new, VarCorr_lme) {
  sd_new <- sqrt(diag(G_new))
  cor_new <- cov2cor(G_new)

  sd_lme <- sqrt(as.numeric(diag(VarCorr_lme)))
  cor_lme <- cov2cor(VarCorr_lme)

  sd_compare <- cbind(lme = sd_lme, new = sd_new)
  sd_compare <- cbind(sd_compare, diff = sd_compare[, "new"] - sd_compare[, "lme"])

  list(sd_compare = sd_compare, cor_compare = list(lme = cor_lme, new = cor_new))
}

# Extract fixed effects comparison
cat("\n--- Fixed effects comparison ---\n")
print(cbind(
  lme = fixed.effects(fit_lme),
  cpp = fit_cpp$beta
))

# Variance and random effects comparison
varcomp <- compare_random_effects(fit_cpp$G, getVarCov(fit_lme))
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
cat("new residual SD:", fit_cpp$sigma, "\n")

cat("\n--- REML logLik comparison (no penalty) ---\n")
cat("lme:", as.numeric(logLik(fit_lme)), "\n")
cat("new:", fit_cpp$logLik, "\n")
cat("difference:", as.numeric(logLik(fit_lme)) - fit_cpp$logLik, "\n")

cat("\nPenalty contribution (new model):\n")
print(fit_cpp$penalty)


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


