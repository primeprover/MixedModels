## ============================================================
## Test script: mixed model simulation + lme comparison
## ============================================================

library(nlme)
library(lme4)

## ============================================================
## Load compiled C++ engine (if available)
## ============================================================
# Compile the C++ REML likelihood implementation
tryCatch(
  Sys.setenv(PKG_CXXFLAGS = "-fopenmp -Wno-ignored-attributes -Wno-unused"),
  Sys.setenv(PKG_LIBS = "-fopenmp"),
  Rcpp::sourceCpp("reml_loglik.cpp", rebuild = TRUE, showOutput = TRUE),
  error = function(e) {
    cat("Warning: Could not compile C++ engine. Using R engine only.\n")
    cat("  Details:", as.character(e), "\n")
  }
)

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

prepare_mixed_data <- function(formula, random, data) {
  if (!inherits(random, "formula")) {
    stop("random must be a formula, e.g. ~ bmi + age | patient")
  }
  random_rhs <- random[[2]]
  if (!is.call(random_rhs) || random_rhs[[1]] != as.name("|")) {
    stop("random formula must be of the form '~ <effects> | <group>'")
  }

  random_effects_expr <- random_rhs[[2]]
  group_expr <- random_rhs[[3]]
  random_formula <- as.formula(call("~", random_effects_expr))

  group_factor <- tryCatch(
    eval(group_expr, envir = data, enclos = parent.frame()),
    error = function(e) NULL
  )
  if (is.null(group_factor)) {
    group_name <- paste(deparse(group_expr), collapse = "")
    stop(paste("Grouping variable", group_name, "not found or could not be evaluated in data"))
  }
  group_factor <- as.factor(group_factor)

  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- model.matrix(formula, mf)
  Z <- model.matrix(random_formula, data)

  split_idx <- split(seq_along(y), group_factor)
  p <- ncol(X)
  q <- ncol(Z)
  n_groups <- length(split_idx)
  n_corr <- q * (q - 1) / 2
  sigma_y <- sd(y)
  lo_sigma <- log(sigma_y) - 6
  hi_sigma <- log(sigma_y) + 4

  group_list <- lapply(split_idx, function(idx) {
    Xi <- X[idx, , drop = FALSE]
    Zi <- Z[idx, , drop = FALSE]
    yi <- y[idx]
    list(
      X = Xi,
      Z = Zi,
      y = yi,
      n = nrow(Xi),
      ZtZ = crossprod(Zi),
      ZtY = crossprod(Zi, yi),
      ZtX = crossprod(Zi, Xi)
    )
  })

  list(
    formula = formula,
    random = random,
    data = data,
    X = X,
    Z = Z,
    y = y,
    p = p,
    q = q,
    split_idx = split_idx,
    group_list = group_list,
    n_groups = n_groups,
    n_corr = n_corr,
    sigma_y = sigma_y,
    lo_sigma = lo_sigma,
    hi_sigma = hi_sigma
  )
}

build_cholesky <- function(theta, q) {
  log_sds <- theta[1:q]
  sds <- exp(log_sds)
  n_corr <- q * (q - 1) / 2
  corrs_unconstrained <- theta[(q + 1):(q + n_corr)]
  corrs <- tanh(corrs_unconstrained)

  L_corr <- diag(1, q)
  k <- 1
  for (i in 2:q) {
    row_prod <- 1
    for (j in 1:(i - 1)) {
      L_corr[i, j] <- corrs[k] * row_prod
      row_prod <- row_prod * sqrt(1 - corrs[k]^2)
      k <- k + 1
    }
    L_corr[i, i] <- row_prod
  }

  diag(sds) %*% L_corr
}

init_theta_general <- function(prepared) {
  q <- prepared$q
  y_sd <- prepared$sigma_y
  n_corr <- prepared$n_corr

  log_sds <- rep(log(max(y_sd * 0.02, 1e-6)), q)
  corrs_init <- rep(0, n_corr)
  log_sigma_resid <- log(max(y_sd * 0.2, 1e-6))
  c(log_sds, corrs_init, log_sigma_resid)
}

reml_loglik_r <- function(theta, prepared, return_penalty = FALSE) {
  bad_result <- function() {
    if (return_penalty) {
      list(
        logLik_unpenalized = -1e20,
        logLik_penalized = -1e20,
        penalty = NA_real_
      )
    } else {
      -1e20
    }
  }

  p <- prepared$p
  q <- prepared$q
  n_corr <- prepared$n_corr
  group_list <- prepared$group_list
  lo_sigma <- prepared$lo_sigma
  hi_sigma <- prepared$hi_sigma
  y <- prepared$y

  L <- build_cholesky(theta, q)
  G <- L %*% t(L)
  G <- G + diag(1e-8, q)
  sigma <- exp(theta[length(theta)])
  sigma2 <- sigma^2

  corrs_unconstrained <- theta[(q + 1):(q + n_corr)]
  corrs <- tanh(corrs_unconstrained)
  if (any(abs(corrs) > 0.9999999)) return(bad_result())
  if (theta[length(theta)] < log(.Machine$double.eps)) return(bad_result())

  log_sd <- theta[1:q]
  penalty <- (
    sum(pmax(0, log_sd - hi_sigma)^2) +
      sum(pmax(0, lo_sigma - log_sd)^2)
  ) * 1e-6

  G_inv <- tryCatch(chol2inv(chol(G)), error = function(e) NULL)
  if (is.null(G_inv)) return(bad_result())
  logdetG <- 2 * sum(log(diag(L)))

  XtVinvX <- matrix(0, p, p)
  XtVinvy <- numeric(p)
  logdetV <- 0
  quad <- 0

  for (group in group_list) {
    Xi <- group$X
    Zi <- group$Z
    yi <- group$y
    n_i <- group$n
    ZiZi <- group$ZtZ
    K <- G_inv + ZiZi / sigma2

    cholK <- tryCatch(chol(K), error = function(e) NULL)
    if (is.null(cholK)) return(bad_result())

    logdetV <- logdetV + n_i * log(sigma2) + 2 * sum(log(diag(cholK)))

    Ziy <- group$ZtY
    K_inv_Ziy <- backsolve(cholK, forwardsolve(t(cholK), Ziy))
    Vi_inv_y <- (yi / sigma2) - (Zi %*% K_inv_Ziy) / (sigma2^2)

    Zix <- group$ZtX
    K_inv_Zix <- backsolve(cholK, forwardsolve(t(cholK), Zix))
    Vi_inv_X <- (Xi / sigma2) - (Zi %*% K_inv_Zix) / (sigma2^2)

    XtVinvX <- XtVinvX + crossprod(Xi, Vi_inv_X)
    XtVinvy <- XtVinvy + crossprod(Xi, Vi_inv_y)
    quad <- quad + crossprod(yi, Vi_inv_y)
  }

  logdetV <- logdetV + length(group_list) * logdetG

  chol_XtVinvX <- tryCatch(chol(XtVinvX), error = function(e) NULL)
  if (is.null(chol_XtVinvX)) return(bad_result())
  logdet_XtVinvX <- 2 * sum(log(diag(chol_XtVinvX)))

  beta_hat <- tryCatch(solve(XtVinvX, XtVinvy), error = function(e) NULL)
  if (is.null(beta_hat)) return(bad_result())
  quad_res <- as.numeric(quad - crossprod(beta_hat, XtVinvy))
  if (quad_res < 0) quad_res <- 0

  n <- length(y)
  ll <- -0.5 * (
    logdetV +
      logdet_XtVinvX +
      quad_res +
      (n - p) * log(2 * pi)
  )

  if (return_penalty) {
    list(
      logLik_unpenalized = ll,
      logLik_penalized = ll - penalty,
      penalty = penalty
    )
  } else {
    ll - penalty
  }
}

reml_loglik_cpp <- function(theta, model, return_penalty = FALSE) {
  # Wrapper to call compiled C++ implementation
  # Delegates to reml_loglik_cpp_impl (defined in reml_loglik.cpp)
  # Converts return format to match reml_loglik_r API

  tryCatch(
    {
      result <- reml_loglik_cpp_impl(theta, model, return_penalty = TRUE, return_timing = TRUE)

      if (return_penalty) {
        return(list(
          logLik_unpenalized = result$logLik_unpenalized,
          logLik_penalized = result$logLik_penalized,
          penalty = result$penalty,
          timings = result$timings
        ))
      } else {
        return(result$logLik_penalized)
      }
    },
    error = function(e) {
      stop("C++ engine is not available. Details: ", as.character(e))
    }
  )
}

fit_mixed_simple <- function(formula, random, data,
                             maxit = 100, verbose = TRUE,
                             engine = "C++") {
  engine <- match.arg(engine)
  
  fake_time <- system.time({}, gcFirst = FALSE)  # Dummy timing to avoid first-time overhead
  print(fake_time)
  prepare_time <- system.time(prepared <- prepare_mixed_data(formula, random, data), gcFirst = FALSE)
    
  # Route to appropriate engine
  if (engine == "C++") {
    extract_time <- system.time(model <- init_model_cpp(prepared), gcFirst = FALSE)
    
    # Delegate likelihood computation to compiled C++ code
    reml_fun <- function(theta, return_penalty = FALSE) {
      reml_loglik_cpp(theta, model, return_penalty)
    }
  } else {
    # Use pure R implementation
    reml_fun <- function(theta, return_penalty = FALSE) {
      reml_loglik_r(theta, prepared, return_penalty)
    }
  }
  initial_values_time <- system.time( {
    theta_init <- init_theta_general(prepared)
    theta_len <- prepared$q + prepared$n_corr + 1
    y_sd <- prepared$sigma_y

    lower <- rep(-Inf, theta_len)
    upper <- rep(Inf, theta_len)
    lower[1:prepared$q] <- log(pmax(y_sd * 1e-6, 1e-12)) - 8
    upper[1:prepared$q] <- log(pmax(y_sd, 1e-6)) + 8

    if (prepared$n_corr > 0) {
      corr_idx <- (prepared$q + 1):(prepared$q + prepared$n_corr)
      lower[corr_idx] <- -4.5
      upper[corr_idx] <- 4.5
    }

    lower[theta_len] <- log(pmax(y_sd * 1e-6, 1e-12)) - 8
    upper[theta_len] <- log(pmax(y_sd, 1e-6)) + 8
  }, gcFirst = FALSE)

  optim_time <- system.time( {
    opt <- optim(
      theta_init,
      fn = function(th) -reml_fun(th),
      method = "L-BFGS-B",
      lower = lower,
      upper = upper,
      control = list(maxit = maxit, trace = ifelse(isTRUE(verbose), 1, 0))
    )
  })

  if (opt$convergence != 0) warning("Optimization did not converge")

  # calc_bhat_time <- system.time({
  #   th <- opt$par
  #   L <- build_cholesky(th, prepared$q)
  #   G_hat <- L %*% t(L)
  #   sigma_hat <- exp(th[length(th)])

  #   XtVinvX <- matrix(0, prepared$p, prepared$p)
  #   XtVinvy <- numeric(prepared$p)
  #   G_jittered <- G_hat + diag(1e-8, nrow(G_hat))
  #   G_inv <- solve(G_jittered)
  #   sigma2 <- sigma_hat^2

  #   for (group in prepared$group_list) {
  #     Xi <- group$X
  #     Zi <- group$Z
  #     yi <- group$y
  #     ZiZi <- group$ZtZ
  #     K <- G_inv + ZiZi / sigma2
  #     cholK <- chol(K)
  #     Ziy <- group$ZtY
  #     K_inv_Ziy <- backsolve(cholK, forwardsolve(t(cholK), Ziy))
  #     Vi_inv_y <- (yi / sigma2) - (Zi %*% K_inv_Ziy) / (sigma2^2)
  #     Zix <- group$ZtX
  #     K_inv_Zix <- backsolve(cholK, forwardsolve(t(cholK), Zix))
  #     Vi_inv_X <- (Xi / sigma2) - (Zi %*% K_inv_Zix) / (sigma2^2)
  #     XtVinvX <- XtVinvX + crossprod(Xi, Vi_inv_X)
  #     XtVinvy <- XtVinvy + crossprod(Xi, Vi_inv_y)
  #   }

  #   beta_hat <- solve(XtVinvX, XtVinvy)
  # })

  calc_bhat_time <- system.time({
    th <- opt$par
    L <- build_cholesky(th, prepared$q)
    G_hat <- L %*% t(L)
    sigma_hat <- exp(th[length(th)])
    beta_hat <- compute_beta_hat_cpp(th, model)
  })

  ll_full <- reml_fun(opt$par, return_penalty = TRUE)
  if (!is.null(ll_full$timings)) {
    ll_full$timings$extract <- as.numeric(extract_time["elapsed"])
    ll_full$timings$prepare <- as.numeric(prepare_time["elapsed"])
    ll_full$timings$calc_bhat <- as.numeric(calc_bhat_time["elapsed"])
    ll_full$timings$initial_values <- as.numeric(initial_values_time["elapsed"])
    ll_full$timings$optim <- as.numeric(optim_time["elapsed"])
    ll_full$timings$total <- ll_full$timings$extract + ll_full$timings$prepare + ll_full$timings$calc_bhat + ll_full$timings$initial_values + ll_full$timings$optim
  }
  list(
    beta = beta_hat,
    G = G_hat,
    sigma = sigma_hat,
    logLik = ll_full$logLik_unpenalized,
    penalty = ll_full$penalty,
    logLik_raw = ll_full$logLik_penalized,
    timings = if (!is.null(ll_full$timings)) ll_full$timings else NULL,
    opt_par = opt$par,
    optim_value = opt$value,
    optim_convergence = opt$convergence,
    optim_message = if (!is.null(opt$message)) opt$message else NA_character_
  )
}

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


