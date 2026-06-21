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
