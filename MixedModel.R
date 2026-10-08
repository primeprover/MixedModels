library(optimx)
library(nloptr)

## ============================================================
## Load compiled C++ engine (if available)
## ============================================================
# Compile the C++ REML likelihood implementation
tryCatch(
  Sys.setenv(PKG_CXXFLAGS = "-fopenmp -Wno-ignored-attributes -Wno-unused"),
  Sys.setenv(PKG_LIBS = "-fopenmp"),
  Rcpp::sourceCpp("reml_loglik.cpp", rebuild = TRUE, showOutput = TRUE, windowsDebugDLL = TRUE),
  error = function(e) {
    cat("Warning: Could not compile C++ engine. Using R engine only.\n")
    cat("  Details:", as.character(e), "\n")
  }
)


prepare_mixed_data <- function(formula, random, var_pred = NULL, data,
                               residual_variance_penalty_weight = 0,
                               response_type = NULL,
                               response_spec = NULL) {
  if (!is.numeric(residual_variance_penalty_weight) ||
      length(residual_variance_penalty_weight) != 1 ||
      !is.finite(residual_variance_penalty_weight) ||
      residual_variance_penalty_weight < 0) {
    stop("residual_variance_penalty_weight must be one non-negative finite number")
  }
  if (!inherits(random, "formula")) {
    stop("random must be a formula, e.g. ~ bmi + age | patient")
  }
  random_rhs <- random[[2]]
  if (!is.call(random_rhs) || random_rhs[[1]] != as.name("|")) {
    stop("random formula must be of the form '~ <effects> | <group>'")
  }
  if (is.null(response_type)) {
    response_type <- if (!is.null(var_pred)) "hetero_normal" else "homo_normal"
  } else {
    response_type <- match.arg(response_type, c("homo_normal", "hetero_normal"))
  }

  heteroscedastic <- identical(response_type, "hetero_normal")
  W <- NULL
  if (is.null(response_spec)) {
    response_spec <- list()
  }
  response_spec <- as.list(response_spec)
  if (heteroscedastic) {
    W <- model.matrix(var_pred, data)
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
  if (!is.null(response_spec$random_map)) {
    random_map <- response_spec$random_map
    if (!is.numeric(random_map) || !all(is.finite(random_map)) || length(random_map) != q) {
      stop("response_spec$random_map must be a numeric vector with one entry per random-effect column")
    }
    if (any(random_map != as.integer(random_map)) || any(random_map < 1L)) {
      stop("response_spec$random_map must contain positive integer indices")
    }
  } else {
    random_map <- seq_len(q)
  }
  if (!is.null(response_spec$observed_groups)) {
    observed_groups <- response_spec$observed_groups
    if (!is.logical(observed_groups) || length(observed_groups) != n_groups) {
      stop("response_spec$observed_groups must be a logical vector with one entry per group")
    }
  } else {
    observed_groups <- rep(TRUE, n_groups)
  }
  n_corr <- q * (q - 1) / 2
  sigma_y <- sd(y)
  lo_sigma <- log(sigma_y) - 6
  hi_sigma <- log(sigma_y) + 4

  if (heteroscedastic) {
    group_list <- lapply(split_idx, function(idx) {
      Xi <- X[idx, , drop = FALSE]
      Zi <- Z[idx, , drop = FALSE]
      yi <- y[idx]
      wi <- W[idx, , drop = FALSE]
      list(
        X = Xi,
        Z = Zi,
        y = yi,
        n = nrow(Xi),
        ZtZ = crossprod(Zi),
        ZtY = crossprod(Zi, yi),
        ZtX = crossprod(Zi, Xi),
        W = wi,
        random_effect_indices = as.integer(random_map)
      )
    })
  } else {
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
        ZtX = crossprod(Zi, Xi),
        random_effect_indices = as.integer(random_map)
      )
    })
  }
  response_spec$type <- response_type
  response_spec$random_map <- as.integer(random_map)
  response_spec$observed_groups <- observed_groups
  if (heteroscedastic) {
    response_spec$var_pred <- var_pred
  }

  list(
    formula = formula,
    random = random,
    var_pred = var_pred,
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
    hi_sigma = hi_sigma,
    heteroscedastic = heteroscedastic,
    response_spec = response_spec,
    residual_variance_penalty_weight = residual_variance_penalty_weight
  )
}

build_cholesky <- function(theta, q) {
  log_sds <- theta[1:q]
  sds <- exp(log_sds)
  n_corr <- q * (q - 1) / 2
  ## needs fixing if n_corr == 0 (q=1)
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
  corrs_init <- rep(0.1, n_corr)
  if (!prepared$heteroscedastic) {
    log_sigma_resid <- 2 * log(max(y_sd * 0.2, 1e-6))
    c(log_sds, corrs_init, log_sigma_resid)
  } else {
    W <- model.matrix(prepared$var_pred, prepared$data)
    target <- rep(2 * log(max(y_sd * 0.2, 1e-6)), nrow(W))
    gamma_init <- qr.solve(W, target)
    #gamma_init <- rep(0, ncol(prepared$group_list[[1]]$W))
    #gamma_init <- 2*log(max(y_sd * 0.2, 1e-6))
    c(log_sds, corrs_init, gamma_init)
  }
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
      result <- reml_loglik_cpp_impl(model, theta, return_penalty = TRUE, return_timing = TRUE)

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
      stop("C++ engine is not available. Details _ : ", as.character(e))
    }
  )
}

normalize_response_spec <- function(spec, default_formula = NULL, default_random = NULL,
                                    default_var_pred = NULL, default_response_type = NULL,
                                    idx = 1L) {
  if (is.null(spec)) {
    stop("Each response specification must be a non-null list.")
  }
  if (!is.list(spec)) {
    stop("Each response specification must be a list.")
  }

  spec <- as.list(spec)

  if (is.null(spec$formula) && !is.null(default_formula)) {
    spec$formula <- default_formula
  }
  if (is.null(spec$random) && !is.null(default_random)) {
    spec$random <- default_random
  }
  if (is.null(spec$var_pred)) {
    spec$var_pred <- default_var_pred
  }
  if (is.null(spec$response_type)) {
    spec$response_type <- default_response_type
  }

  if (is.null(spec$formula) || !inherits(spec$formula, "formula")) {
    stop(sprintf("response %d must contain a valid formula.", idx))
  }
  if (is.null(spec$random) || !inherits(spec$random, "formula")) {
    stop(sprintf("response %d must contain a valid random-effects formula.", idx))
  }
  spec
}

prepare_response_specs <- function(responses = NULL, formula = NULL, random = NULL,
                                   var_pred = NULL, response_type = NULL) {
  if (is.null(responses)) {
    responses <- list(list(
      formula = formula,
      random = random,
      var_pred = var_pred,
      response_type = response_type
    ))
  } else if (!is.list(responses)) {
    stop("responses must be NULL or a list of response specifications.")
  }

  if (length(responses) == 0L) {
    stop("At least one response specification is required.")
  }

  if (is.list(responses[[1]]) && !is.null(names(responses[[1]])) &&
      all(c("formula", "random") %in% names(responses[[1]]))) {
    response_list <- responses
  } else {
    response_list <- list(responses)
  }

  lapply(seq_along(response_list), function(i) {
    normalize_response_spec(
      response_list[[i]],
      default_formula = formula,
      default_random = random,
      default_var_pred = var_pred,
      default_response_type = response_type,
      idx = i
    )
  })
}

fit_mixed <- function(responses = NULL, data = NULL,
                      maxit = 100, verbose = TRUE,
                      engine = "C++",
                      residual_variance_penalty_weight = 0, theta_init = NULL) {

  if (is.null(responses)) {
    stop("responses must be supplied as a list of response specifications.")
  }

  response_specs <- prepare_response_specs(responses = responses)

  if (length(response_specs) == 1L) {
    spec <- response_specs[[1]]
    return(fit_mixed_simple(
      formula = spec$formula, random = spec$random, var_pred = spec$var_pred,
      data = data, maxit = maxit, verbose = verbose, engine = engine,
      response_type = spec$response_type,
      residual_variance_penalty_weight = residual_variance_penalty_weight,
      response_spec = list(random_map = spec$random_map, observed_groups = spec$observed_groups)
    ))
  }

  separate_fits <- vector("list", length(response_specs))
  response_names <- character(length(response_specs))
  for (i in seq_along(response_specs)) {
    spec <- response_specs[[i]]
    response_names[i] <- if (!is.null(spec$name)) spec$name else sprintf("response%d", i)
    separate_fits[[i]] <- fit_mixed_simple(
      formula = spec$formula, random = spec$random, var_pred = spec$var_pred,
      data = data, maxit = maxit, verbose = verbose, engine = engine,
      response_type = spec$response_type,
      residual_variance_penalty_weight = residual_variance_penalty_weight,
      response_spec = list(random_map = spec$random_map, observed_groups = spec$observed_groups)
    )
  }
  names(separate_fits) <- response_names

  if (engine != "C++") {
    return(list(joint_fit = NULL, separate_fits = separate_fits))
  }

  # Extract grouping structure for joint fit
  random_rhs <- response_specs[[1]]$random[[2]]
  group_expr <- random_rhs[[3]]
  group_values <- tryCatch(
    eval(group_expr, envir = data, enclos = parent.frame()),
    error = function(e) NULL
  )
  if (is.null(group_values)) {
    stop("Could not evaluate the grouping variable for the joint fit.")
  }

  response_preparations <- lapply(seq_along(response_specs), function(i) {
    spec <- response_specs[[i]]
    prepared_i <- prepare_mixed_data(
      formula = spec$formula, random = spec$random, var_pred = spec$var_pred,
      data = data, residual_variance_penalty_weight = residual_variance_penalty_weight,
      response_type = spec$response_type,
      response_spec = list(random_map = spec$random_map, observed_groups = spec$observed_groups)
    )
    prepared_i$response_spec <- list(
      type = prepared_i$response_spec$type,
      random_map = prepared_i$response_spec$random_map,
      observed_groups = prepared_i$response_spec$observed_groups
    )
    prepared_i
  })

  # ---- MODIFICATION: Allow variable q per response ----
  response_qs <- vapply(response_preparations, function(prepared) prepared$q, integer(1))
  shared_q <- sum(response_qs)
  shared_n_corr <- shared_q * (shared_q - 1) / 2

  global_random_maps <- vector("list", length(response_preparations))
  offset <- 0L
  for (i in seq_along(response_preparations)) {
    q_i <- response_qs[i]
    global_random_maps[[i]] <- seq.int(offset + 1L, offset + q_i)
    offset <- offset + q_i
    response_preparations[[i]]$response_spec$random_map <- as.integer(global_random_maps[[i]])
    for (j in seq_along(response_preparations[[i]]$group_list)) {
      response_preparations[[i]]$group_list[[j]]$random_effect_indices <- as.integer(global_random_maps[[i]])
    }
  }

  shared_group_list <- vector("list", length(response_preparations[[1]]$group_list))
  for (g in seq_along(shared_group_list)) {
    Z_full <- matrix(0, nrow = response_preparations[[1]]$group_list[[g]]$n, ncol = shared_q)
    for (i in seq_along(response_preparations)) {
      Zi <- as.matrix(response_preparations[[i]]$group_list[[g]]$Z)
      Z_full[, global_random_maps[[i]]] <- Zi
    }
    shared_group_list[[g]] <- list(
      Z = Z_full,
      n = response_preparations[[1]]$group_list[[g]]$n,
      ZtZ = crossprod(Z_full)
    )
  }

  response_param_counts <- vapply(response_preparations, function(prepared) {
    if (identical(prepared$response_spec$type, "hetero_normal")) {
      ncol(model.matrix(prepared$var_pred, prepared$data))
    } else {
      1L
    }
  }, integer(1))

  prepared <- list(
    data = data, q = shared_q, n_corr = shared_n_corr,
    lo_sigma = response_preparations[[1]]$lo_sigma,
    hi_sigma = response_preparations[[1]]$hi_sigma,
    shared_group_list = shared_group_list, responses = response_preparations,
    residual_variance_penalty_weight = residual_variance_penalty_weight,
    response_param_counts = response_param_counts
  )

  model <- init_model_cpp(prepared)
  reml_fun <- function(theta, return_penalty = FALSE) {
    base_ll = reml_loglik_cpp(theta, model, FALSE)
    penalty = 0 
    base_ll - penalty
  }
  # *sum(theta[7:21]^2)
  reml_fun_final <- function(theta, return_penalty = FALSE) {
    reml_loglik_cpp(theta, model, return_penalty)
  }

  # ---- MODIFICATION: Informed Initialization from Separate Fits ----
  if (is.null(theta_init)) {
    init_log_sds <- numeric(shared_q)
    offset <- 0L

    # Initialize block-diagonal unconstrained correlation matrix
    init_corr_mat <- matrix(0.1, nrow = shared_q, ncol = shared_q)

    for (i in seq_along(response_preparations)) {
      q_i <- response_qs[i]
      pars <- separate_fits[[i]]$opt_par

      # 1. Log SDs
      init_log_sds[(offset + 1L):(offset + q_i)] <- pars[1L:q_i]

      # 2. Unconstrained correlations (copy within-response block)
      n_corr_i <- q_i * (q_i - 1) / 2
      if (n_corr_i > 0) {
        local_corrs <- pars[(q_i + 1L):(q_i + n_corr_i)]
        k <- 1L
        for (a in 2L:q_i) {
          for (b in 1L:(a - 1L)) {
            init_corr_mat[offset + a, offset + b] <- local_corrs[k]
            k <- k + 1L
          }
        }
      }
      offset <- offset + q_i
    }

    # Flatten correlations using the exact same packing order as build_cholesky
    init_corrs <- numeric(shared_n_corr)
    k <- 1L
    for (a in 2L:shared_q) {
      for (b in 1L:(a - 1L)) {
        init_corrs[k] <- init_corr_mat[a, b]
        k <- k + 1L
      }
    }

    # 3. Response variance parameters (concatenate in specification order)
    init_resp_pars <- c()
    for (i in seq_along(response_preparations)) {
      n_p <- response_param_counts[i]
      if (n_p > 0) {
        init_resp_pars <- c(init_resp_pars, separate_fits[[i]]$opt_par[
          (length(separate_fits[[i]]$opt_par) - n_p + 1L):length(separate_fits[[i]]$opt_par)
        ])
      }
    }

    theta_init <- c(init_log_sds, init_corrs, init_resp_pars)
  }

  y_sd <- response_preparations[[1]]$sigma_y
  theta_len <- shared_q + shared_n_corr + sum(response_param_counts)

  lower <- rep(-Inf, theta_len)
  upper <- rep(Inf, theta_len)
  lower[seq_len(shared_q)] <- log(pmax(y_sd * 1e-6, 1e-12)) - 8
  upper[seq_len(shared_q)] <- log(pmax(y_sd, 1e-6)) + 8

  if (shared_n_corr > 0) {
    corr_idx <- (shared_q + 1):(shared_q + shared_n_corr)
    lower[corr_idx] <- -4.5
    upper[corr_idx] <- 4.5
  }

  response_idx <- seq.int(shared_q + shared_n_corr + 1, theta_len)
  lower[response_idx] <- log(pmax(y_sd * 1e-6, 1e-12)) - 8
  upper[response_idx] <- log(pmax(y_sd, 1e-6)) + 8

  # ---- MODIFICATION: Stable Gradient via numDeriv ----
  if (!requireNamespace("numDeriv", quietly = TRUE)) {
    stop("The 'numDeriv' package is required for gradient computation.")
  }
  reml_grad <- function(th) numDeriv::grad(func = function(x) -reml_fun(x)/1, x = th, method.args = list(eps = 1e-6, d = 1e-6))

  opt <- optim(
    theta_init,
    fn = function(x) -reml_fun(x)/1,
    gr = reml_grad,
    method = "L-BFGS-B", # Switched to L-BFGS-B for better gradient compatibility with bounds
    lower = lower,
    upper = upper,
    control = list(maxit = maxit, trace = 6, factr = 1e6, maxfun = 1e6, maxeval = 1e6, verbose = TRUE)
  )

  if (opt$convergence != 0) warning("Optimization did not converge")
  
  th <- opt$par
  L <- build_cholesky(th, shared_q)
  G_hat <- L %*% t(L)
  beta_hat <- compute_beta_hat_cpp(th, model)
  ll_full <- reml_fun_final(th, return_penalty = TRUE)

  joint_fit <- list(
    beta = beta_hat, G = G_hat,
    sigma = exp(th[seq.int(shared_q + shared_n_corr + 1, theta_len)]),
    logLik = ll_full$logLik_unpenalized,
    penalty = ll_full$penalty,
    logLik_raw = ll_full$logLik_penalized,
    opt_par = th,
    optim_value = opt$value,
    optim_convergence = opt$convergence,
    optim_message = if (!is.null(opt$message)) opt$message else NA_character_,
    model = model, reml_fun = reml_fun,
    lower = lower, upper = upper
  )

  list(joint_fit = joint_fit, separate_fits = separate_fits)
}

fit_mixed_simple <- function(formula, random, var_pred = NULL, data,
                             maxit = 100, verbose = TRUE,
                             engine = "C++",
                             response_type = NULL,
                             residual_variance_penalty_weight = 0,
                             response_spec = NULL) {
  engine <- match.arg(engine)
  
  fake_time <- system.time({}, gcFirst = FALSE)  # Dummy timing to avoid first-time overhead
  print(fake_time)
  prepare_time <- system.time(
    prepared <- prepare_mixed_data(
      formula, random, var_pred, data, residual_variance_penalty_weight,
      response_type = response_type,
      response_spec = response_spec
    ), gcFirst = FALSE)
  print("Data prepared")
  print(names(prepared))
  # Route to appropriate engine
  if (engine == "C++") {
    extract_time <- system.time(model <- init_model_cpp(prepared), gcFirst = FALSE)
    print("Model initialized in C++ engine")
    # Delegate likelihood computation to compiled C++ code
    reml_fun <- function(theta, return_penalty = FALSE) {
      reml_loglik_cpp(theta, model, return_penalty)
    }
  } else {
    # Use pure R implementation
    if (!is.null(var_pred)) {
      stop("Heteroscedastic variance modeling is only supported in the C++ engine.")
    }
    reml_fun <- function(theta, return_penalty = FALSE) {
      reml_loglik_r(theta, prepared, return_penalty)
    }
  }
  initial_values_time <- system.time( {
    theta_init <- init_theta_general(prepared)
    print("Initial theta values:")
    print(theta_init)
    # The response owns the final parameter block.  For the currently exposed
    # normal responses this is one log-variance parameter or all W coefficients.
    theta_len <- length(theta_init)
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

    response_idx <- seq.int(prepared$q + prepared$n_corr + 1, theta_len)
    lower[response_idx] <- log(pmax(y_sd * 1e-6, 1e-12)) - 8
    upper[response_idx] <- log(pmax(y_sd, 1e-6)) + 8
  }, gcFirst = FALSE)

  # optim_time <- system.time( {
  #   opt <- optim(
  #     theta_init,
  #     fn = function(th) -reml_fun(th),
  #     method = "L-BFGS-B",
  #     lower = lower,
  #     upper = upper,
  #     control = list(maxit = maxit, trace = 2, factr = 1e3)
  #   )
  # })

  optim_time <- system.time( {
    opt <- optim(
      theta_init,
      fn = function(th) -reml_fun(th),
      method = "L-BFGS-B",
      lower = lower,
      upper = upper,
      control = list(maxit = maxit, trace = 2, factr = 1e3)
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
    optim_message = if (!is.null(opt$message)) opt$message else NA_character_,
    reml_fun = reml_fun
  )
}
